import torch
from typing import Dict, List, Optional


from fairseq.strategies.strategy_utils import generate_step_with_prob

from plot import (
    plot_cross_attn_full_layer_iteration_heatmap,
    plot_cross_attn_layer_iteration_heatmap,
    plot_layerwise_token_mask_heatmap,
    plot_self_attn_full_layer_iteration_heatmap,
    plot_self_attn_layer_iteration_heatmap,
    plot_token_mask_probs,
)
from utils import (
    _mask_tokens_for_iteration,
    _move_encoder_out_to_device,
    _normalize_iteration_indices,
    _predicted_length,
    _record_iteration,
    _resolve_sentence_pair,
    _resolve_text_alias,
    _stringify_tokens,
    clone_encoder_out,
    decode_from_encoder_output,
    get_encoder_output,
    load_mask_predict_context,
)


def _run_decoder_forward(
    model,
    tgt_tokens: torch.Tensor,
    encoder_out: Dict[str, torch.Tensor],
    layer_index: int,
    source_hidden_state: Optional[torch.Tensor] = None,
):
    captured: Dict[str, torch.Tensor] = {}
    decoder_layers = model.decoder.layers

    if layer_index < 0 or layer_index >= len(decoder_layers):
        raise IndexError(f"layer_index {layer_index} out of range for {len(decoder_layers)} decoder layers")

    layer = decoder_layers[layer_index]

    def hook(_module, _inputs, output):
        hidden, attn = output
        captured["before_patch"] = hidden.detach().cpu().clone()
        patched_hidden = hidden
        if source_hidden_state is not None:
            source_state = source_hidden_state.to(hidden.device, dtype=hidden.dtype)
            if source_state.shape != hidden.shape:
                raise ValueError(
                    "full-layer patching requires source and target decoder activations to have the same shape, "
                    f"got {tuple(source_state.shape)} and {tuple(hidden.shape)}"
                )
            patched_hidden = source_state.clone()
        captured["after_patch"] = patched_hidden.detach().cpu().clone()
        return patched_hidden, attn

    handle = layer.register_forward_hook(hook)
    try:
        decoder_out = model.decoder(tgt_tokens, encoder_out)
    finally:
        handle.remove()

    return decoder_out, captured


def _prepare_paired_patch_context(
    source_sentence: str,
    target_sentence: str,
    context: Dict[str, object],
) -> Dict[str, object]:
    task = context["task"]
    model = context["model"]
    device = context["device"]
    args = context["args"]
    tgt_dict = task.target_dictionary

    source_encoder = get_encoder_output(source_sentence, context)
    target_encoder = get_encoder_output(target_sentence, context)
    source_encoder_out = _move_encoder_out_to_device(clone_encoder_out(source_encoder["encoder_out"]), device)
    target_encoder_out = _move_encoder_out_to_device(clone_encoder_out(target_encoder["encoder_out"]), device)

    return {
        "task": task,
        "model": model,
        "device": device,
        "args": args,
        "tgt_dict": tgt_dict,
        "source_encoder": source_encoder,
        "target_encoder": target_encoder,
        "source_encoder_out": source_encoder_out,
        "target_encoder_out": target_encoder_out,
        "source_length": _predicted_length(source_encoder_out),
        "target_length": _predicted_length(target_encoder_out),
    }


def _normalize_layer_indices(decoder_layers, layer_indices: Optional[List[int]]) -> List[int]:
    if layer_indices is None:
        return list(range(len(decoder_layers)))
    return [int(layer_index) for layer_index in layer_indices]


def _prepare_target_reference(
    target_sentence: str,
    context: Dict[str, object],
    decoding_iterations: int,
):
    tgt_dict = context["task"].target_dictionary
    target_encoder = get_encoder_output(target_sentence, context)
    target_reference = decode_from_encoder_output(
        target_encoder["encoder_out"],
        context=context,
        decoding_iterations=decoding_iterations,
    )
    return target_encoder, target_reference, tgt_dict


def _apply_masked_token_updates(
    tgt_tokens: torch.Tensor,
    token_probs: torch.Tensor,
    mask_ind: torch.Tensor,
    new_tgt_tokens: torch.Tensor,
    new_token_probs: torch.Tensor,
):
    updated_token_probs = token_probs.clone()
    tgt_tokens[0, mask_ind] = new_tgt_tokens[0, mask_ind]
    updated_token_probs[0, mask_ind] = new_token_probs[0, mask_ind]
    return tgt_tokens, updated_token_probs


def _build_patch_trace_step(
    *,
    iteration: int,
    source_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    source_capture: Dict[str, torch.Tensor],
    target_capture: Dict[str, torch.Tensor],
    tgt_dict,
    remove_bpe: Optional[str],
    patch_mode: str,
    module_name: Optional[str] = None,
    token_position: Optional[int] = None,
    iteration_was_patched: Optional[bool] = None,
) -> Dict[str, object]:
    trace_step = {
        "iteration": int(iteration),
        "source_text": _stringify_tokens(source_tokens[0].detach().cpu(), tgt_dict, remove_bpe),
        "patched_text": _stringify_tokens(target_tokens[0].detach().cpu(), tgt_dict, remove_bpe),
        "patch_mode": patch_mode,
        "activation_shape": list(target_capture["after_patch"].shape),
        "source_activation_norm": round(float(source_capture["after_patch"].norm().item()), 6),
        "target_activation_norm_before_patch": round(float(target_capture["before_patch"].norm().item()), 6),
        "target_activation_norm_after_patch": round(float(target_capture["after_patch"].norm().item()), 6),
    }
    if module_name is not None:
        trace_step["module_name"] = module_name
    if token_position is not None:
        trace_step["token_position"] = token_position
    if iteration_was_patched is not None:
        trace_step["iteration_was_patched"] = iteration_was_patched
    return trace_step


def _run_attention_patching_variant(
    attention_type: str,
    *,
    source_sentence: Optional[str] = None,
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    layer_index: Optional[int] = None,
    token_position: Optional[int] = None,
    decoding_iterations: int = 5,
    patch_iterations: Optional[List[int]] = None,
    src_sentence: Optional[str] = None,
    tgt_sentence: Optional[str] = None,
    require_token_position: bool = False,
) -> Dict[str, object]:
    if require_token_position and token_position is None:
        raise ValueError("context, layer_index, and token_position are required")
    return decoder_attention_patching(
        attention_type=attention_type,
        source_sentence=source_sentence,
        target_sentence=target_sentence,
        context=context,
        layer_index=layer_index,
        token_position=token_position,
        decoding_iterations=decoding_iterations,
        patch_iterations=patch_iterations,
        src_sentence=src_sentence,
        tgt_sentence=tgt_sentence,
    )


def _run_attention_iteration_sweep_variant(
    attention_type: str,
    *,
    source_sentence: Optional[str] = None,
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    tracked_token_position: int = 0,
    patch_token_position: Optional[int] = None,
    decoding_iterations: int = 5,
    layer_indices: Optional[List[int]] = None,
    patch_iterations: Optional[List[int]] = None,
    src_sentence: Optional[str] = None,
    tgt_sentence: Optional[str] = None,
    default_patch_token_position_to_tracked: bool = False,
) -> Dict[str, object]:
    if default_patch_token_position_to_tracked and patch_token_position is None:
        patch_token_position = tracked_token_position
    return decoder_attention_layer_iteration_sweep(
        attention_type=attention_type,
        source_sentence=source_sentence,
        target_sentence=target_sentence,
        context=context,
        tracked_token_position=tracked_token_position,
        patch_token_position=patch_token_position,
        decoding_iterations=decoding_iterations,
        layer_indices=layer_indices,
        patch_iterations=patch_iterations,
        src_sentence=src_sentence,
        tgt_sentence=tgt_sentence,
    )


def decoder_patching(
    source_sentence: Optional[str] = None,
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    layer_index: Optional[int] = None,
    token_position: Optional[int] = None,
    decoding_iterations: int = 5,
    src_sentence: Optional[str] = None,
    tgt_sentence: Optional[str] = None,
) -> Dict[str, object]:
    if context is None or layer_index is None:
        raise ValueError("context and layer_index are required")
    source_sentence, target_sentence = _resolve_sentence_pair(
        source_sentence=source_sentence,
        target_sentence=target_sentence,
        src_sentence=src_sentence,
        tgt_sentence=tgt_sentence,
    )
    patch_context = _prepare_paired_patch_context(source_sentence, target_sentence, context)
    model = patch_context["model"]
    device = patch_context["device"]
    args = patch_context["args"]
    tgt_dict = patch_context["tgt_dict"]
    source_encoder = patch_context["source_encoder"]
    target_encoder = patch_context["target_encoder"]
    source_encoder_out = patch_context["source_encoder_out"]
    target_encoder_out = patch_context["target_encoder_out"]
    source_length = patch_context["source_length"]
    target_length = patch_context["target_length"]
    if source_length != target_length:
        raise ValueError(
            "full-layer patching requires source and target predicted lengths to match, "
            f"got {source_length} and {target_length}"
        )

    iterations = target_length if decoding_iterations is None else decoding_iterations
    source_tgt_tokens = torch.full((1, source_length), tgt_dict.mask(), dtype=torch.long, device=device)
    target_tgt_tokens = torch.full((1, target_length), tgt_dict.mask(), dtype=torch.long, device=device)

    iteration_trace: List[Dict[str, object]] = []
    patch_trace: List[Dict[str, object]] = []

    with torch.no_grad():
        source_decoder_out, source_capture = _run_decoder_forward(
            model,
            source_tgt_tokens,
            source_encoder_out,
            layer_index,
        )
        source_tgt_tokens, source_token_probs, _ = generate_step_with_prob(source_decoder_out)

        patched_decoder_out, target_capture = _run_decoder_forward(
            model,
            target_tgt_tokens,
            target_encoder_out,
            layer_index,
            source_hidden_state=source_capture["after_patch"],
        )
        target_tgt_tokens, target_token_probs, _ = generate_step_with_prob(patched_decoder_out)

        _record_iteration(iteration_trace, 0, target_tgt_tokens, target_token_probs, tgt_dict, args.remove_bpe)
        patch_trace.append(
            _build_patch_trace_step(
                iteration=0,
                source_tokens=source_tgt_tokens,
                target_tokens=target_tgt_tokens,
                source_capture=source_capture,
                target_capture=target_capture,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
                patch_mode="full_layer",
            )
        )

        for counter in range(1, iterations):
            source_masked_tokens, source_mask_ind, _, _ = _mask_tokens_for_iteration(
                source_tgt_tokens,
                source_token_probs,
                tgt_dict,
                counter,
                iterations,
            )
            source_decoder_out, source_capture = _run_decoder_forward(
                model,
                source_masked_tokens,
                source_encoder_out,
                layer_index,
            )
            source_new_tgt_tokens, source_new_token_probs, _ = generate_step_with_prob(source_decoder_out)
            source_tgt_tokens, source_token_probs = _apply_masked_token_updates(
                source_masked_tokens,
                source_token_probs,
                source_mask_ind,
                source_new_tgt_tokens,
                source_new_token_probs,
            )

            target_masked_tokens, target_mask_ind, selected_mask_token_ids, selected_mask_scores = _mask_tokens_for_iteration(
                target_tgt_tokens,
                target_token_probs,
                tgt_dict,
                counter,
                iterations,
            )
            patched_decoder_out, target_capture = _run_decoder_forward(
                model,
                target_masked_tokens,
                target_encoder_out,
                layer_index,
                source_hidden_state=source_capture["after_patch"],
            )
            target_new_tgt_tokens, target_new_token_probs, _ = generate_step_with_prob(patched_decoder_out)
            target_tgt_tokens, target_token_probs = _apply_masked_token_updates(
                target_masked_tokens,
                target_token_probs,
                target_mask_ind,
                target_new_tgt_tokens,
                target_new_token_probs,
            )

            _record_iteration(
                iteration_trace,
                counter,
                target_tgt_tokens,
                target_token_probs,
                tgt_dict,
                args.remove_bpe,
                selected_mask_token_ids=selected_mask_token_ids,
                selected_mask_scores=selected_mask_scores,
            )
            patch_trace.append(
                _build_patch_trace_step(
                    iteration=counter,
                    source_tokens=source_tgt_tokens,
                    target_tokens=target_tgt_tokens,
                    source_capture=source_capture,
                    target_capture=target_capture,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                    patch_mode="full_layer",
                )
            )

    final_token_ids = target_tgt_tokens[0].detach().cpu()
    return {
        "source_sentence": source_sentence,
        "target_sentence": target_sentence,
        "layer_index": layer_index,
        "token_position": token_position,
        "patch_mode": "full_layer",
        "source_encoder": source_encoder,
        "target_encoder": target_encoder,
        "decoded_text": _stringify_tokens(final_token_ids, tgt_dict, args.remove_bpe),
        "token_ids": final_token_ids.tolist(),
        "iteration_trace": iteration_trace,
        "patch_trace": patch_trace,
        "predicted_length": target_length,
    }


def decoder_layer_sweep(
    source_sentence: Optional[str] = None,
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    tracked_token_position: int = 0,
    decoding_iterations: int = 5,
    layer_indices: Optional[List[int]] = None,
    src_sentence: Optional[str] = None,
    tgt_sentence: Optional[str] = None,
) -> Dict[str, object]:
    if context is None:
        raise ValueError("context is required")
    source_sentence, target_sentence = _resolve_sentence_pair(
        source_sentence=source_sentence,
        target_sentence=target_sentence,
        src_sentence=src_sentence,
        tgt_sentence=tgt_sentence,
    )
    model = context["model"]
    decoder_layers = model.decoder.layers
    layer_indices = _normalize_layer_indices(decoder_layers, layer_indices)
    _, target_reference, tgt_dict = _prepare_target_reference(
        target_sentence=target_sentence,
        context=context,
        decoding_iterations=decoding_iterations,
    )
    if tracked_token_position < 0 or tracked_token_position >= len(target_reference["token_ids"]):
        raise IndexError(
            f"tracked_token_position {tracked_token_position} out of range for decoded length {len(target_reference['token_ids'])}"
        )

    tracked_token_label = tgt_dict[target_reference["token_ids"][tracked_token_position]]
    layer_results: List[Dict[str, object]] = []
    heatmap: List[List[float]] = []

    for layer_index in layer_indices:
        layer_result = decoder_patching(
            source_sentence=source_sentence,
            target_sentence=target_sentence,
            context=context,
            layer_index=layer_index,
            decoding_iterations=decoding_iterations,
        )
        tracked_token_mask_probs = [
            step["token_mask_probs"][tracked_token_position]
            for step in layer_result["iteration_trace"]
        ]
        tracked_token_texts_by_iteration = [
            tgt_dict[step["token_ids"][tracked_token_position]]
            for step in layer_result["iteration_trace"]
        ]
        layer_result["tracked_token_position"] = tracked_token_position
        layer_result["tracked_token_label"] = tracked_token_label
        layer_result["tracked_token_mask_probs"] = tracked_token_mask_probs
        layer_result["tracked_token_texts_by_iteration"] = tracked_token_texts_by_iteration
        layer_results.append(layer_result)
        heatmap.append(tracked_token_mask_probs)

    return {
        "source_sentence": source_sentence,
        "target_sentence": target_sentence,
        "patch_mode": "full_layer",
        "tracked_token_position": tracked_token_position,
        "tracked_token_label": tracked_token_label,
        "reference_decoded_text": target_reference["decoded_text"],
        "reference_token_ids": target_reference["token_ids"],
        "layer_indices": layer_indices,
        "iterations": [step["iteration"] for step in layer_results[0]["iteration_trace"]] if layer_results else [],
        "heatmap": heatmap,
        "layer_results": layer_results,
    }


def _get_attention_patch_spec(attention_type: str) -> Dict[str, str]:
    attention_specs = {
        "self": {
            "label": "self-attn",
            "module_attr": "self_attn",
            "module_name_suffix": "self_attn",
            "token_patch_mode": "self_attn_source_to_target",
            "full_layer_patch_mode": "self_attn_full_layer_source_to_target",
        },
        "cross": {
            "label": "cross-attn",
            "module_attr": "encoder_attn",
            "module_name_suffix": "encoder_attn",
            "token_patch_mode": "cross_attn_source_to_target",
            "full_layer_patch_mode": "cross_attn_full_layer_source_to_target",
        },
    }
    try:
        return attention_specs[attention_type]
    except KeyError as exc:
        raise ValueError(f"attention_type must be one of {sorted(attention_specs)}, got {attention_type!r}") from exc


def _get_attention_patch_mode(attention_type: str, token_position: Optional[int]) -> str:
    spec = _get_attention_patch_spec(attention_type)
    if token_position is None:
        return spec["full_layer_patch_mode"]
    return spec["token_patch_mode"]


def _get_attention_module_name(attention_type: str, layer_index: int) -> str:
    spec = _get_attention_patch_spec(attention_type)
    return f"decoder.layers.{layer_index}.{spec['module_name_suffix']}"


def _run_decoder_attention_patch_forward(
    model,
    tgt_tokens: torch.Tensor,
    encoder_out: Dict[str, torch.Tensor],
    layer_index: int,
    attention_type: str,
    token_position: Optional[int] = None,
    source_attention_state: Optional[torch.Tensor] = None,
):
    spec = _get_attention_patch_spec(attention_type)
    captured: Dict[str, torch.Tensor] = {}
    decoder_layers = model.decoder.layers

    if layer_index < 0 or layer_index >= len(decoder_layers):
        raise IndexError(f"layer_index {layer_index} out of range for {len(decoder_layers)} decoder layers")
    if token_position is not None and (token_position < 0 or token_position >= tgt_tokens.size(1)):
        raise IndexError(f"token_position {token_position} out of range for decoder length {tgt_tokens.size(1)}")

    layer = decoder_layers[layer_index]
    module = getattr(layer, spec["module_attr"])
    if module is None:
        raise ValueError(f"decoder layer {layer_index} does not have {spec['module_attr']}")

    def hook(_module, _inputs, output):
        hidden, attn = output
        if token_position is None:
            before_patch = hidden
        else:
            before_patch = hidden[token_position, 0, :]
        captured["before_patch"] = before_patch.detach().cpu().clone()

        patched_hidden = hidden
        if source_attention_state is not None:
            source_state = source_attention_state.to(hidden.device, dtype=hidden.dtype)
            if source_state.shape != before_patch.shape:
                if token_position is None:
                    raise ValueError(
                        f"{spec['label']} full-layer patching requires source and target activations to have the same shape, "
                        f"got {tuple(source_state.shape)} and {tuple(before_patch.shape)}"
                    )
                raise ValueError(
                    f"{spec['label']} token patching requires source and target token activations to have the same shape, "
                    f"got {tuple(source_state.shape)} and {tuple(before_patch.shape)}"
                )

            if token_position is None:
                patched_hidden = source_state.clone()
            else:
                patched_hidden = hidden.clone()
                patched_hidden[token_position, 0, :] = source_state

        if token_position is None:
            after_patch = patched_hidden
        else:
            after_patch = patched_hidden[token_position, 0, :]
        captured["after_patch"] = after_patch.detach().cpu().clone()
        return patched_hidden, attn

    handle = module.register_forward_hook(hook)
    try:
        decoder_out = model.decoder(tgt_tokens, encoder_out)
    finally:
        handle.remove()

    return decoder_out, captured


def decoder_attention_patching(
    attention_type: str,
    source_sentence: Optional[str] = None,
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    layer_index: Optional[int] = None,
    token_position: Optional[int] = None,
    decoding_iterations: int = 5,
    patch_iterations: Optional[List[int]] = None,
    src_sentence: Optional[str] = None,
    tgt_sentence: Optional[str] = None,
) -> Dict[str, object]:
    if context is None or layer_index is None:
        raise ValueError("context and layer_index are required")

    spec = _get_attention_patch_spec(attention_type)
    source_sentence, target_sentence = _resolve_sentence_pair(
        source_sentence=source_sentence,
        target_sentence=target_sentence,
        src_sentence=src_sentence,
        tgt_sentence=tgt_sentence,
    )
    patch_context = _prepare_paired_patch_context(source_sentence, target_sentence, context)
    model = patch_context["model"]
    device = patch_context["device"]
    args = patch_context["args"]
    tgt_dict = patch_context["tgt_dict"]
    source_encoder = patch_context["source_encoder"]
    target_encoder = patch_context["target_encoder"]
    source_encoder_out = patch_context["source_encoder_out"]
    target_encoder_out = patch_context["target_encoder_out"]
    source_length = patch_context["source_length"]
    target_length = patch_context["target_length"]
    if token_position is None:
        if source_length != target_length:
            raise ValueError(
                f"{spec['label']} full-layer patching requires source and target predicted lengths to match, "
                f"got {source_length} and {target_length}"
            )
    elif token_position < 0 or token_position >= source_length or token_position >= target_length:
        raise IndexError(
            f"token_position {token_position} out of range for source/target decoded lengths {source_length} and {target_length}"
        )

    iterations = target_length if decoding_iterations is None else decoding_iterations
    patch_iteration_list = _normalize_iteration_indices(
        patch_iterations,
        iterations,
        name="patch_iterations",
    )
    patch_iteration_set = set(patch_iteration_list)
    source_tgt_tokens = torch.full((1, source_length), tgt_dict.mask(), dtype=torch.long, device=device)
    target_tgt_tokens = torch.full((1, target_length), tgt_dict.mask(), dtype=torch.long, device=device)

    patch_mode = _get_attention_patch_mode(attention_type, token_position)
    module_name = _get_attention_module_name(attention_type, layer_index)
    iteration_trace: List[Dict[str, object]] = []
    patch_trace: List[Dict[str, object]] = []

    with torch.no_grad():
        source_decoder_out, source_capture = _run_decoder_attention_patch_forward(
            model,
            source_tgt_tokens,
            source_encoder_out,
            layer_index,
            attention_type=attention_type,
            token_position=token_position,
        )
        source_tgt_tokens, source_token_probs, _ = generate_step_with_prob(source_decoder_out)

        patched_decoder_out, target_capture = _run_decoder_attention_patch_forward(
            model,
            target_tgt_tokens,
            target_encoder_out,
            layer_index,
            attention_type=attention_type,
            token_position=token_position,
            source_attention_state=source_capture["after_patch"] if 0 in patch_iteration_set else None,
        )
        target_tgt_tokens, target_token_probs, _ = generate_step_with_prob(patched_decoder_out)

        _record_iteration(iteration_trace, 0, target_tgt_tokens, target_token_probs, tgt_dict, args.remove_bpe)
        patch_trace.append(
            _build_patch_trace_step(
                iteration=0,
                source_tokens=source_tgt_tokens,
                target_tokens=target_tgt_tokens,
                source_capture=source_capture,
                target_capture=target_capture,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
                patch_mode=patch_mode,
                module_name=module_name,
                token_position=token_position,
                iteration_was_patched=0 in patch_iteration_set,
            )
        )

        for counter in range(1, iterations):
            source_masked_tokens, source_mask_ind, _, _ = _mask_tokens_for_iteration(
                source_tgt_tokens,
                source_token_probs,
                tgt_dict,
                counter,
                iterations,
            )
            source_decoder_out, source_capture = _run_decoder_attention_patch_forward(
                model,
                source_masked_tokens,
                source_encoder_out,
                layer_index,
                attention_type=attention_type,
                token_position=token_position,
            )
            source_new_tgt_tokens, source_new_token_probs, _ = generate_step_with_prob(source_decoder_out)
            source_tgt_tokens, source_token_probs = _apply_masked_token_updates(
                source_masked_tokens,
                source_token_probs,
                source_mask_ind,
                source_new_tgt_tokens,
                source_new_token_probs,
            )

            target_masked_tokens, target_mask_ind, selected_mask_token_ids, selected_mask_scores = _mask_tokens_for_iteration(
                target_tgt_tokens,
                target_token_probs,
                tgt_dict,
                counter,
                iterations,
            )
            patched_decoder_out, target_capture = _run_decoder_attention_patch_forward(
                model,
                target_masked_tokens,
                target_encoder_out,
                layer_index,
                attention_type=attention_type,
                token_position=token_position,
                source_attention_state=source_capture["after_patch"] if counter in patch_iteration_set else None,
            )
            target_new_tgt_tokens, target_new_token_probs, _ = generate_step_with_prob(patched_decoder_out)
            target_tgt_tokens, target_token_probs = _apply_masked_token_updates(
                target_masked_tokens,
                target_token_probs,
                target_mask_ind,
                target_new_tgt_tokens,
                target_new_token_probs,
            )

            _record_iteration(
                iteration_trace,
                counter,
                target_tgt_tokens,
                target_token_probs,
                tgt_dict,
                args.remove_bpe,
                selected_mask_token_ids=selected_mask_token_ids,
                selected_mask_scores=selected_mask_scores,
            )
            patch_trace.append(
                _build_patch_trace_step(
                    iteration=counter,
                    source_tokens=source_tgt_tokens,
                    target_tokens=target_tgt_tokens,
                    source_capture=source_capture,
                    target_capture=target_capture,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                    patch_mode=patch_mode,
                    module_name=module_name,
                    token_position=token_position,
                    iteration_was_patched=counter in patch_iteration_set,
                )
            )

    final_token_ids = target_tgt_tokens[0].detach().cpu()
    result = {
        "source_sentence": source_sentence,
        "target_sentence": target_sentence,
        "layer_index": layer_index,
        "patch_mode": patch_mode,
        "module_name": module_name,
        "source_encoder": source_encoder,
        "target_encoder": target_encoder,
        "patch_iterations": patch_iteration_list,
        "decoded_text": _stringify_tokens(final_token_ids, tgt_dict, args.remove_bpe),
        "token_ids": final_token_ids.tolist(),
        "iteration_trace": iteration_trace,
        "patch_trace": patch_trace,
        "predicted_length": target_length,
    }
    if token_position is not None:
        result["token_position"] = token_position
    return result


def decoder_attention_layer_iteration_sweep(
    attention_type: str,
    source_sentence: Optional[str] = None,
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    tracked_token_position: int = 0,
    patch_token_position: Optional[int] = None,
    decoding_iterations: int = 5,
    layer_indices: Optional[List[int]] = None,
    patch_iterations: Optional[List[int]] = None,
    src_sentence: Optional[str] = None,
    tgt_sentence: Optional[str] = None,
) -> Dict[str, object]:
    if context is None:
        raise ValueError("context is required")

    source_sentence, target_sentence = _resolve_sentence_pair(
        source_sentence=source_sentence,
        target_sentence=target_sentence,
        src_sentence=src_sentence,
        tgt_sentence=tgt_sentence,
    )

    model = context["model"]
    decoder_layers = model.decoder.layers
    layer_indices = _normalize_layer_indices(decoder_layers, layer_indices)
    _, target_reference, tgt_dict = _prepare_target_reference(
        target_sentence=target_sentence,
        context=context,
        decoding_iterations=decoding_iterations,
    )

    if tracked_token_position < 0 or tracked_token_position >= len(target_reference["token_ids"]):
        raise IndexError(
            f"tracked_token_position {tracked_token_position} out of range for decoded length {len(target_reference['token_ids'])}"
        )

    if patch_token_position is not None and (
        patch_token_position < 0 or patch_token_position >= len(target_reference["token_ids"])
    ):
        raise IndexError(
            f"patch_token_position {patch_token_position} out of range for decoded length {len(target_reference['token_ids'])}"
        )

    patch_iteration_list = _normalize_iteration_indices(
        patch_iterations,
        decoding_iterations,
        name="patch_iterations",
    )

    tracked_token_label = tgt_dict[target_reference["token_ids"][tracked_token_position]]
    patch_mode = _get_attention_patch_mode(attention_type, patch_token_position)
    layer_results: List[Dict[str, object]] = []
    heatmap: List[List[float]] = []

    for layer_index in layer_indices:
        patch_iteration_results: List[Dict[str, object]] = []
        tracked_token_mask_probs: List[float] = []
        tracked_token_texts: List[str] = []
        decoded_texts: List[str] = []

        for patch_iteration in patch_iteration_list:
            patch_result = decoder_attention_patching(
                attention_type=attention_type,
                source_sentence=source_sentence,
                target_sentence=target_sentence,
                context=context,
                layer_index=layer_index,
                token_position=patch_token_position,
                decoding_iterations=decoding_iterations,
                patch_iterations=[patch_iteration],
            )
            tracked_probability = patch_result["iteration_trace"][patch_iteration]["token_mask_probs"][tracked_token_position]
            tracked_text = tgt_dict[patch_result["iteration_trace"][patch_iteration]["token_ids"][tracked_token_position]]
            tracked_token_mask_probs.append(tracked_probability)
            tracked_token_texts.append(tracked_text)
            decoded_texts.append(patch_result["decoded_text"])
            patch_iteration_results.append(
                {
                    "patch_iteration": patch_iteration,
                    "tracked_token_mask_prob": tracked_probability,
                    "tracked_token_text": tracked_text,
                    "decoded_text": patch_result["decoded_text"],
                    "patch_trace_step": patch_result["patch_trace"][patch_iteration],
                    "iteration_trace_step": patch_result["iteration_trace"][patch_iteration],
                }
            )

        layer_result = {
            "layer_index": layer_index,
            "tracked_token_position": tracked_token_position,
            "tracked_token_label": tracked_token_label,
            "tracked_token_mask_probs_by_patch_iteration": tracked_token_mask_probs,
            "tracked_token_texts_by_patch_iteration": tracked_token_texts,
            "decoded_texts_by_patch_iteration": decoded_texts,
            "patch_iteration_results": patch_iteration_results,
        }
        if patch_token_position is not None:
            layer_result["patch_token_position"] = patch_token_position
            layer_result["patch_token_label"] = tgt_dict[target_reference["token_ids"][patch_token_position]]
        layer_results.append(layer_result)
        heatmap.append(tracked_token_mask_probs)

    result = {
        "source_sentence": source_sentence,
        "target_sentence": target_sentence,
        "patch_mode": patch_mode,
        "tracked_token_position": tracked_token_position,
        "tracked_token_label": tracked_token_label,
        "reference_decoded_text": target_reference["decoded_text"],
        "reference_token_ids": target_reference["token_ids"],
        "layer_indices": layer_indices,
        "patch_iterations": patch_iteration_list,
        "heatmap": heatmap,
        "layer_results": layer_results,
    }
    if patch_token_position is not None:
        result["patch_token_position"] = patch_token_position
        result["patch_token_label"] = tgt_dict[target_reference["token_ids"][patch_token_position]]
    return result


def decoder_self_attn_token_patching(
    source_sentence: Optional[str] = None,
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    layer_index: Optional[int] = None,
    token_position: Optional[int] = None,
    decoding_iterations: int = 5,
    patch_iterations: Optional[List[int]] = None,
    src_sentence: Optional[str] = None,
    tgt_sentence: Optional[str] = None,
) -> Dict[str, object]:
    return _run_attention_patching_variant(
        "self",
        source_sentence=source_sentence,
        target_sentence=target_sentence,
        context=context,
        layer_index=layer_index,
        token_position=token_position,
        decoding_iterations=decoding_iterations,
        patch_iterations=patch_iterations,
        src_sentence=src_sentence,
        tgt_sentence=tgt_sentence,
        require_token_position=True,
    )


def decoder_self_attn_full_layer_patching(
    source_sentence: Optional[str] = None,
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    layer_index: Optional[int] = None,
    decoding_iterations: int = 5,
    patch_iterations: Optional[List[int]] = None,
    src_sentence: Optional[str] = None,
    tgt_sentence: Optional[str] = None,
) -> Dict[str, object]:
    return _run_attention_patching_variant(
        "self",
        source_sentence=source_sentence,
        target_sentence=target_sentence,
        context=context,
        layer_index=layer_index,
        token_position=None,
        decoding_iterations=decoding_iterations,
        patch_iterations=patch_iterations,
        src_sentence=src_sentence,
        tgt_sentence=tgt_sentence,
    )


def decoder_cross_attn_token_patching(
    source_sentence: Optional[str] = None,
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    layer_index: Optional[int] = None,
    token_position: Optional[int] = None,
    decoding_iterations: int = 5,
    patch_iterations: Optional[List[int]] = None,
    src_sentence: Optional[str] = None,
    tgt_sentence: Optional[str] = None,
) -> Dict[str, object]:
    return _run_attention_patching_variant(
        "cross",
        source_sentence=source_sentence,
        target_sentence=target_sentence,
        context=context,
        layer_index=layer_index,
        token_position=token_position,
        decoding_iterations=decoding_iterations,
        patch_iterations=patch_iterations,
        src_sentence=src_sentence,
        tgt_sentence=tgt_sentence,
        require_token_position=True,
    )


def decoder_cross_attn_full_layer_patching(
    source_sentence: Optional[str] = None,
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    layer_index: Optional[int] = None,
    decoding_iterations: int = 5,
    patch_iterations: Optional[List[int]] = None,
    src_sentence: Optional[str] = None,
    tgt_sentence: Optional[str] = None,
) -> Dict[str, object]:
    return _run_attention_patching_variant(
        "cross",
        source_sentence=source_sentence,
        target_sentence=target_sentence,
        context=context,
        layer_index=layer_index,
        token_position=None,
        decoding_iterations=decoding_iterations,
        patch_iterations=patch_iterations,
        src_sentence=src_sentence,
        tgt_sentence=tgt_sentence,
    )


def decoder_self_attn_layer_iteration_sweep(
    source_sentence: Optional[str] = None,
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    tracked_token_position: int = 0,
    patch_token_position: Optional[int] = None,
    decoding_iterations: int = 5,
    layer_indices: Optional[List[int]] = None,
    patch_iterations: Optional[List[int]] = None,
    src_sentence: Optional[str] = None,
    tgt_sentence: Optional[str] = None,
) -> Dict[str, object]:
    return _run_attention_iteration_sweep_variant(
        "self",
        source_sentence=source_sentence,
        target_sentence=target_sentence,
        context=context,
        tracked_token_position=tracked_token_position,
        patch_token_position=patch_token_position,
        decoding_iterations=decoding_iterations,
        layer_indices=layer_indices,
        patch_iterations=patch_iterations,
        src_sentence=src_sentence,
        tgt_sentence=tgt_sentence,
        default_patch_token_position_to_tracked=True,
    )


def decoder_self_attn_full_layer_iteration_sweep(
    source_sentence: Optional[str] = None,
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    tracked_token_position: int = 0,
    decoding_iterations: int = 5,
    layer_indices: Optional[List[int]] = None,
    patch_iterations: Optional[List[int]] = None,
    src_sentence: Optional[str] = None,
    tgt_sentence: Optional[str] = None,
) -> Dict[str, object]:
    return _run_attention_iteration_sweep_variant(
        "self",
        source_sentence=source_sentence,
        target_sentence=target_sentence,
        context=context,
        tracked_token_position=tracked_token_position,
        patch_token_position=None,
        decoding_iterations=decoding_iterations,
        layer_indices=layer_indices,
        patch_iterations=patch_iterations,
        src_sentence=src_sentence,
        tgt_sentence=tgt_sentence,
    )


def decoder_cross_attn_layer_iteration_sweep(
    source_sentence: Optional[str] = None,
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    tracked_token_position: int = 0,
    patch_token_position: Optional[int] = None,
    decoding_iterations: int = 5,
    layer_indices: Optional[List[int]] = None,
    patch_iterations: Optional[List[int]] = None,
    src_sentence: Optional[str] = None,
    tgt_sentence: Optional[str] = None,
) -> Dict[str, object]:
    return _run_attention_iteration_sweep_variant(
        "cross",
        source_sentence=source_sentence,
        target_sentence=target_sentence,
        context=context,
        tracked_token_position=tracked_token_position,
        patch_token_position=patch_token_position,
        decoding_iterations=decoding_iterations,
        layer_indices=layer_indices,
        patch_iterations=patch_iterations,
        src_sentence=src_sentence,
        tgt_sentence=tgt_sentence,
        default_patch_token_position_to_tracked=True,
    )


def decoder_cross_attn_full_layer_iteration_sweep(
    source_sentence: Optional[str] = None,
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    tracked_token_position: int = 0,
    decoding_iterations: int = 5,
    layer_indices: Optional[List[int]] = None,
    patch_iterations: Optional[List[int]] = None,
    src_sentence: Optional[str] = None,
    tgt_sentence: Optional[str] = None,
) -> Dict[str, object]:
    return _run_attention_iteration_sweep_variant(
        "cross",
        source_sentence=source_sentence,
        target_sentence=target_sentence,
        context=context,
        tracked_token_position=tracked_token_position,
        patch_token_position=None,
        decoding_iterations=decoding_iterations,
        layer_indices=layer_indices,
        patch_iterations=patch_iterations,
        src_sentence=src_sentence,
        tgt_sentence=tgt_sentence,
    )
