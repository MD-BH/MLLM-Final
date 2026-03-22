import torch
from typing import Dict, List, Optional


from fairseq.strategies.strategy_utils import generate_step_with_prob

from plot import (
    plot_cross_attn_full_layer_iteration_heatmap,
    plot_cross_attn_head_zero_ablation_heatmap,
    plot_cross_attn_layer_iteration_heatmap,
    plot_layerwise_token_mask_heatmap,
    plot_self_attn_full_layer_iteration_heatmap,
    plot_self_attn_head_zero_ablation_heatmap,
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
    task = context["task"]
    model = context["model"]
    device = context["device"]
    args = context["args"]
    tgt_dict = task.target_dictionary

    source_encoder = get_encoder_output(source_sentence, context)
    target_encoder = get_encoder_output(target_sentence, context)
    source_encoder_out = _move_encoder_out_to_device(clone_encoder_out(source_encoder["encoder_out"]), device)
    target_encoder_out = _move_encoder_out_to_device(clone_encoder_out(target_encoder["encoder_out"]), device)

    source_length = _predicted_length(source_encoder_out)
    target_length = _predicted_length(target_encoder_out)
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
            {
                "iteration": 0,
                "source_text": _stringify_tokens(source_tgt_tokens[0].detach().cpu(), tgt_dict, args.remove_bpe),
                "patched_text": _stringify_tokens(target_tgt_tokens[0].detach().cpu(), tgt_dict, args.remove_bpe),
                "patch_mode": "full_layer",
                "activation_shape": list(target_capture["after_patch"].shape),
                "source_activation_norm": round(float(source_capture["after_patch"].norm().item()), 6),
                "target_activation_norm_before_patch": round(float(target_capture["before_patch"].norm().item()), 6),
                "target_activation_norm_after_patch": round(float(target_capture["after_patch"].norm().item()), 6),
            }
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
            source_tgt_tokens = source_masked_tokens
            source_token_probs = source_token_probs.clone()
            source_tgt_tokens[0, source_mask_ind] = source_new_tgt_tokens[0, source_mask_ind]
            source_token_probs[0, source_mask_ind] = source_new_token_probs[0, source_mask_ind]

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
            target_tgt_tokens = target_masked_tokens
            target_token_probs = target_token_probs.clone()
            target_tgt_tokens[0, target_mask_ind] = target_new_tgt_tokens[0, target_mask_ind]
            target_token_probs[0, target_mask_ind] = target_new_token_probs[0, target_mask_ind]

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
                {
                    "iteration": int(counter),
                    "source_text": _stringify_tokens(source_tgt_tokens[0].detach().cpu(), tgt_dict, args.remove_bpe),
                    "patched_text": _stringify_tokens(target_tgt_tokens[0].detach().cpu(), tgt_dict, args.remove_bpe),
                    "patch_mode": "full_layer",
                    "activation_shape": list(target_capture["after_patch"].shape),
                    "source_activation_norm": round(float(source_capture["after_patch"].norm().item()), 6),
                    "target_activation_norm_before_patch": round(float(target_capture["before_patch"].norm().item()), 6),
                    "target_activation_norm_after_patch": round(float(target_capture["after_patch"].norm().item()), 6),
                }
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
    tgt_dict = context["task"].target_dictionary
    decoder_layers = model.decoder.layers

    if layer_indices is None:
        layer_indices = list(range(len(decoder_layers)))
    else:
        layer_indices = [int(layer_index) for layer_index in layer_indices]

    target_encoder = get_encoder_output(target_sentence, context)
    target_reference = decode_from_encoder_output(
        target_encoder["encoder_out"],
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

    task = context["task"]
    model = context["model"]
    device = context["device"]
    args = context["args"]
    tgt_dict = task.target_dictionary

    source_encoder = get_encoder_output(source_sentence, context)
    target_encoder = get_encoder_output(target_sentence, context)
    source_encoder_out = _move_encoder_out_to_device(clone_encoder_out(source_encoder["encoder_out"]), device)
    target_encoder_out = _move_encoder_out_to_device(clone_encoder_out(target_encoder["encoder_out"]), device)

    source_length = _predicted_length(source_encoder_out)
    target_length = _predicted_length(target_encoder_out)
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
        patch_trace_step = {
            "iteration": 0,
            "source_text": _stringify_tokens(source_tgt_tokens[0].detach().cpu(), tgt_dict, args.remove_bpe),
            "patched_text": _stringify_tokens(target_tgt_tokens[0].detach().cpu(), tgt_dict, args.remove_bpe),
            "patch_mode": patch_mode,
            "module_name": module_name,
            "activation_shape": list(target_capture["after_patch"].shape),
            "iteration_was_patched": 0 in patch_iteration_set,
            "source_activation_norm": round(float(source_capture["after_patch"].norm().item()), 6),
            "target_activation_norm_before_patch": round(float(target_capture["before_patch"].norm().item()), 6),
            "target_activation_norm_after_patch": round(float(target_capture["after_patch"].norm().item()), 6),
        }
        if token_position is not None:
            patch_trace_step["token_position"] = token_position
        patch_trace.append(patch_trace_step)

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
            source_tgt_tokens = source_masked_tokens
            source_token_probs = source_token_probs.clone()
            source_tgt_tokens[0, source_mask_ind] = source_new_tgt_tokens[0, source_mask_ind]
            source_token_probs[0, source_mask_ind] = source_new_token_probs[0, source_mask_ind]

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
            target_tgt_tokens = target_masked_tokens
            target_token_probs = target_token_probs.clone()
            target_tgt_tokens[0, target_mask_ind] = target_new_tgt_tokens[0, target_mask_ind]
            target_token_probs[0, target_mask_ind] = target_new_token_probs[0, target_mask_ind]

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

            patch_trace_step = {
                "iteration": int(counter),
                "source_text": _stringify_tokens(source_tgt_tokens[0].detach().cpu(), tgt_dict, args.remove_bpe),
                "patched_text": _stringify_tokens(target_tgt_tokens[0].detach().cpu(), tgt_dict, args.remove_bpe),
                "patch_mode": patch_mode,
                "module_name": module_name,
                "activation_shape": list(target_capture["after_patch"].shape),
                "iteration_was_patched": counter in patch_iteration_set,
                "source_activation_norm": round(float(source_capture["after_patch"].norm().item()), 6),
                "target_activation_norm_before_patch": round(float(target_capture["before_patch"].norm().item()), 6),
                "target_activation_norm_after_patch": round(float(target_capture["after_patch"].norm().item()), 6),
            }
            if token_position is not None:
                patch_trace_step["token_position"] = token_position
            patch_trace.append(patch_trace_step)

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
    tgt_dict = context["task"].target_dictionary
    decoder_layers = model.decoder.layers

    if layer_indices is None:
        layer_indices = list(range(len(decoder_layers)))
    else:
        layer_indices = [int(layer_index) for layer_index in layer_indices]

    target_encoder = get_encoder_output(target_sentence, context)
    target_reference = decode_from_encoder_output(
        target_encoder["encoder_out"],
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
    if token_position is None:
        raise ValueError("context, layer_index, and token_position are required")
    return decoder_attention_patching(
        attention_type="self",
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
    return decoder_attention_patching(
        attention_type="self",
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
    if token_position is None:
        raise ValueError("context, layer_index, and token_position are required")
    return decoder_attention_patching(
        attention_type="cross",
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
    return decoder_attention_patching(
        attention_type="cross",
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
    if patch_token_position is None:
        patch_token_position = tracked_token_position
    return decoder_attention_layer_iteration_sweep(
        attention_type="self",
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
    return decoder_attention_layer_iteration_sweep(
        attention_type="self",
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
    if patch_token_position is None:
        patch_token_position = tracked_token_position
    return decoder_attention_layer_iteration_sweep(
        attention_type="cross",
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
    return decoder_attention_layer_iteration_sweep(
        attention_type="cross",
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


def _mean_token_mask_prob(iteration_step: Dict[str, object]) -> float:
    token_mask_probs = iteration_step["token_mask_probs"]
    return round(float(sum(token_mask_probs) / len(token_mask_probs)), 6)


def _run_decoder_self_attn_head_zero_forward(
    model,
    tgt_tokens: torch.Tensor,
    encoder_out: Dict[str, torch.Tensor],
    layer_index: int,
    head_index: int,
):
    captured: Dict[str, torch.Tensor] = {}
    decoder_layers = model.decoder.layers

    if layer_index < 0 or layer_index >= len(decoder_layers):
        raise IndexError(f"layer_index {layer_index} out of range for {len(decoder_layers)} decoder layers")

    layer = decoder_layers[layer_index]
    self_attn = layer.self_attn
    num_heads = self_attn.num_heads
    head_dim = self_attn.head_dim

    if head_index < 0 or head_index >= num_heads:
        raise IndexError(f"head_index {head_index} out of range for {num_heads} self-attn heads")

    head_start = head_index * head_dim
    head_end = head_start + head_dim

    def hook(_module, inputs):
        (attn_input,) = inputs
        captured["before_zero"] = attn_input[..., head_start:head_end].detach().cpu().clone()
        patched_input = attn_input.clone()
        patched_input[..., head_start:head_end] = 0
        captured["after_zero"] = patched_input[..., head_start:head_end].detach().cpu().clone()
        return (patched_input,)

    handle = self_attn.out_proj.register_forward_pre_hook(hook)
    original_enable_torch_version = self_attn.enable_torch_version
    self_attn.enable_torch_version = False
    try:
        decoder_out = model.decoder(tgt_tokens, encoder_out)
    finally:
        self_attn.enable_torch_version = original_enable_torch_version
        handle.remove()

    return decoder_out, captured


def decoder_self_attn_head_zero_ablation(
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    layer_index: Optional[int] = None,
    head_index: Optional[int] = None,
    decoding_iterations: int = 5,
    tgt_sentence: Optional[str] = None,
) -> Dict[str, object]:
    if context is None or layer_index is None or head_index is None:
        raise ValueError("context, layer_index, and head_index are required")

    target_sentence = _resolve_text_alias(
        "target_sentence",
        target_sentence,
        tgt_sentence,
    )

    task = context["task"]
    model = context["model"]
    device = context["device"]
    args = context["args"]
    tgt_dict = task.target_dictionary

    target_encoder = get_encoder_output(target_sentence, context)
    target_encoder_out = _move_encoder_out_to_device(clone_encoder_out(target_encoder["encoder_out"]), device)
    target_length = _predicted_length(target_encoder_out)

    iterations = target_length if decoding_iterations is None else decoding_iterations
    target_tgt_tokens = torch.full((1, target_length), tgt_dict.mask(), dtype=torch.long, device=device)

    reference_decode = decode_from_encoder_output(
        target_encoder["encoder_out"],
        context=context,
        decoding_iterations=decoding_iterations,
    )

    iteration_trace: List[Dict[str, object]] = []
    patch_trace: List[Dict[str, object]] = []

    with torch.no_grad():
        decoder_out, capture = _run_decoder_self_attn_head_zero_forward(
            model,
            target_tgt_tokens,
            target_encoder_out,
            layer_index,
            head_index,
        )
        target_tgt_tokens, target_token_probs, _ = generate_step_with_prob(decoder_out)
        _record_iteration(iteration_trace, 0, target_tgt_tokens, target_token_probs, tgt_dict, args.remove_bpe)
        patch_trace.append(
            {
                "iteration": 0,
                "patch_mode": "self_attn_head_zero_ablation",
                "module_name": f"decoder.layers.{layer_index}.self_attn",
                "head_index": head_index,
                "head_output_shape": list(capture["after_zero"].shape),
                "head_output_norm_before_zero": round(float(capture["before_zero"].norm().item()), 6),
                "head_output_norm_after_zero": round(float(capture["after_zero"].norm().item()), 6),
                "average_token_mask_prob": _mean_token_mask_prob(iteration_trace[-1]),
            }
        )

        for counter in range(1, iterations):
            target_masked_tokens, target_mask_ind, selected_mask_token_ids, selected_mask_scores = _mask_tokens_for_iteration(
                target_tgt_tokens,
                target_token_probs,
                tgt_dict,
                counter,
                iterations,
            )
            decoder_out, capture = _run_decoder_self_attn_head_zero_forward(
                model,
                target_masked_tokens,
                target_encoder_out,
                layer_index,
                head_index,
            )
            target_new_tgt_tokens, target_new_token_probs, _ = generate_step_with_prob(decoder_out)
            target_tgt_tokens = target_masked_tokens
            target_token_probs = target_token_probs.clone()
            target_tgt_tokens[0, target_mask_ind] = target_new_tgt_tokens[0, target_mask_ind]
            target_token_probs[0, target_mask_ind] = target_new_token_probs[0, target_mask_ind]

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
                {
                    "iteration": int(counter),
                    "patch_mode": "self_attn_head_zero_ablation",
                    "module_name": f"decoder.layers.{layer_index}.self_attn",
                    "head_index": head_index,
                    "head_output_shape": list(capture["after_zero"].shape),
                    "head_output_norm_before_zero": round(float(capture["before_zero"].norm().item()), 6),
                    "head_output_norm_after_zero": round(float(capture["after_zero"].norm().item()), 6),
                    "average_token_mask_prob": _mean_token_mask_prob(iteration_trace[-1]),
                }
            )

    average_token_mask_probs_by_iteration = [
        _mean_token_mask_prob(iteration_step)
        for iteration_step in iteration_trace
    ]
    overall_average_mask_prob = round(
        float(sum(average_token_mask_probs_by_iteration) / len(average_token_mask_probs_by_iteration)),
        6,
    )
    final_token_ids = target_tgt_tokens[0].detach().cpu()

    return {
        "target_sentence": target_sentence,
        "layer_index": layer_index,
        "head_index": head_index,
        "patch_mode": "self_attn_head_zero_ablation",
        "module_name": f"decoder.layers.{layer_index}.self_attn",
        "target_encoder": target_encoder,
        "reference_decoded_text": reference_decode["decoded_text"],
        "reference_token_ids": reference_decode["token_ids"],
        "decoded_text": _stringify_tokens(final_token_ids, tgt_dict, args.remove_bpe),
        "token_ids": final_token_ids.tolist(),
        "iteration_trace": iteration_trace,
        "patch_trace": patch_trace,
        "average_token_mask_probs_by_iteration": average_token_mask_probs_by_iteration,
        "overall_average_mask_prob": overall_average_mask_prob,
        "predicted_length": target_length,
    }


def decoder_self_attn_head_zero_ablation_sweep(
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    layer_index: Optional[int] = None,
    decoding_iterations: int = 5,
    head_indices: Optional[List[int]] = None,
    tgt_sentence: Optional[str] = None,
) -> Dict[str, object]:
    if context is None or layer_index is None:
        raise ValueError("context and layer_index are required")

    target_sentence = _resolve_text_alias(
        "target_sentence",
        target_sentence,
        tgt_sentence,
    )

    model = context["model"]
    decoder_layers = model.decoder.layers
    if layer_index < 0 or layer_index >= len(decoder_layers):
        raise IndexError(f"layer_index {layer_index} out of range for {len(decoder_layers)} decoder layers")

    num_heads = decoder_layers[layer_index].self_attn.num_heads
    if head_indices is None:
        head_indices = list(range(num_heads))
    else:
        head_indices = [int(head_index) for head_index in head_indices]
        for head_index in head_indices:
            if head_index < 0 or head_index >= num_heads:
                raise IndexError(f"head_index {head_index} out of range for {num_heads} self-attn heads")

    head_results: List[Dict[str, object]] = []
    heatmap: List[List[float]] = []

    for head_index in head_indices:
        head_result = decoder_self_attn_head_zero_ablation(
            target_sentence=target_sentence,
            context=context,
            layer_index=layer_index,
            head_index=head_index,
            decoding_iterations=decoding_iterations,
        )
        head_results.append(head_result)
        heatmap.append(head_result["average_token_mask_probs_by_iteration"])

    return {
        "target_sentence": target_sentence,
        "layer_index": layer_index,
        "patch_mode": "self_attn_head_zero_ablation",
        "head_indices": head_indices,
        "iterations": [step["iteration"] for step in head_results[0]["iteration_trace"]] if head_results else [],
        "reference_decoded_text": head_results[0]["reference_decoded_text"] if head_results else "",
        "reference_token_ids": head_results[0]["reference_token_ids"] if head_results else [],
        "heatmap": heatmap,
        "head_results": head_results,
    }


def _run_decoder_cross_attn_head_zero_forward(
    model,
    tgt_tokens: torch.Tensor,
    encoder_out: Dict[str, torch.Tensor],
    layer_index: int,
    head_index: int,
):
    captured: Dict[str, torch.Tensor] = {}
    decoder_layers = model.decoder.layers

    if layer_index < 0 or layer_index >= len(decoder_layers):
        raise IndexError(f"layer_index {layer_index} out of range for {len(decoder_layers)} decoder layers")

    layer = decoder_layers[layer_index]
    encoder_attn = layer.encoder_attn
    if encoder_attn is None:
        raise ValueError(f"decoder layer {layer_index} does not have cross-attention")

    num_heads = encoder_attn.num_heads
    head_dim = encoder_attn.head_dim

    if head_index < 0 or head_index >= num_heads:
        raise IndexError(f"head_index {head_index} out of range for {num_heads} cross-attn heads")

    head_start = head_index * head_dim
    head_end = head_start + head_dim

    def hook(_module, inputs):
        (attn_input,) = inputs
        captured["before_zero"] = attn_input[..., head_start:head_end].detach().cpu().clone()
        patched_input = attn_input.clone()
        patched_input[..., head_start:head_end] = 0
        captured["after_zero"] = patched_input[..., head_start:head_end].detach().cpu().clone()
        return (patched_input,)

    handle = encoder_attn.out_proj.register_forward_pre_hook(hook)
    original_enable_torch_version = encoder_attn.enable_torch_version
    encoder_attn.enable_torch_version = False
    try:
        decoder_out = model.decoder(tgt_tokens, encoder_out)
    finally:
        encoder_attn.enable_torch_version = original_enable_torch_version
        handle.remove()

    return decoder_out, captured


def decoder_cross_attn_head_zero_ablation(
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    layer_index: Optional[int] = None,
    head_index: Optional[int] = None,
    decoding_iterations: int = 5,
    tgt_sentence: Optional[str] = None,
) -> Dict[str, object]:
    if context is None or layer_index is None or head_index is None:
        raise ValueError("context, layer_index, and head_index are required")

    target_sentence = _resolve_text_alias(
        "target_sentence",
        target_sentence,
        tgt_sentence,
    )

    task = context["task"]
    model = context["model"]
    device = context["device"]
    args = context["args"]
    tgt_dict = task.target_dictionary

    target_encoder = get_encoder_output(target_sentence, context)
    target_encoder_out = _move_encoder_out_to_device(clone_encoder_out(target_encoder["encoder_out"]), device)
    target_length = _predicted_length(target_encoder_out)

    iterations = target_length if decoding_iterations is None else decoding_iterations
    target_tgt_tokens = torch.full((1, target_length), tgt_dict.mask(), dtype=torch.long, device=device)

    reference_decode = decode_from_encoder_output(
        target_encoder["encoder_out"],
        context=context,
        decoding_iterations=decoding_iterations,
    )

    iteration_trace: List[Dict[str, object]] = []
    patch_trace: List[Dict[str, object]] = []

    with torch.no_grad():
        decoder_out, capture = _run_decoder_cross_attn_head_zero_forward(
            model,
            target_tgt_tokens,
            target_encoder_out,
            layer_index,
            head_index,
        )
        target_tgt_tokens, target_token_probs, _ = generate_step_with_prob(decoder_out)
        _record_iteration(iteration_trace, 0, target_tgt_tokens, target_token_probs, tgt_dict, args.remove_bpe)
        patch_trace.append(
            {
                "iteration": 0,
                "patch_mode": "cross_attn_head_zero_ablation",
                "module_name": f"decoder.layers.{layer_index}.encoder_attn",
                "head_index": head_index,
                "head_output_shape": list(capture["after_zero"].shape),
                "head_output_norm_before_zero": round(float(capture["before_zero"].norm().item()), 6),
                "head_output_norm_after_zero": round(float(capture["after_zero"].norm().item()), 6),
                "average_token_mask_prob": _mean_token_mask_prob(iteration_trace[-1]),
            }
        )

        for counter in range(1, iterations):
            target_masked_tokens, target_mask_ind, selected_mask_token_ids, selected_mask_scores = _mask_tokens_for_iteration(
                target_tgt_tokens,
                target_token_probs,
                tgt_dict,
                counter,
                iterations,
            )
            decoder_out, capture = _run_decoder_cross_attn_head_zero_forward(
                model,
                target_masked_tokens,
                target_encoder_out,
                layer_index,
                head_index,
            )
            target_new_tgt_tokens, target_new_token_probs, _ = generate_step_with_prob(decoder_out)
            target_tgt_tokens = target_masked_tokens
            target_token_probs = target_token_probs.clone()
            target_tgt_tokens[0, target_mask_ind] = target_new_tgt_tokens[0, target_mask_ind]
            target_token_probs[0, target_mask_ind] = target_new_token_probs[0, target_mask_ind]

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
                {
                    "iteration": int(counter),
                    "patch_mode": "cross_attn_head_zero_ablation",
                    "module_name": f"decoder.layers.{layer_index}.encoder_attn",
                    "head_index": head_index,
                    "head_output_shape": list(capture["after_zero"].shape),
                    "head_output_norm_before_zero": round(float(capture["before_zero"].norm().item()), 6),
                    "head_output_norm_after_zero": round(float(capture["after_zero"].norm().item()), 6),
                    "average_token_mask_prob": _mean_token_mask_prob(iteration_trace[-1]),
                }
            )

    average_token_mask_probs_by_iteration = [
        _mean_token_mask_prob(iteration_step)
        for iteration_step in iteration_trace
    ]
    overall_average_mask_prob = round(
        float(sum(average_token_mask_probs_by_iteration) / len(average_token_mask_probs_by_iteration)),
        6,
    )
    final_token_ids = target_tgt_tokens[0].detach().cpu()

    return {
        "target_sentence": target_sentence,
        "layer_index": layer_index,
        "head_index": head_index,
        "patch_mode": "cross_attn_head_zero_ablation",
        "module_name": f"decoder.layers.{layer_index}.encoder_attn",
        "target_encoder": target_encoder,
        "reference_decoded_text": reference_decode["decoded_text"],
        "reference_token_ids": reference_decode["token_ids"],
        "decoded_text": _stringify_tokens(final_token_ids, tgt_dict, args.remove_bpe),
        "token_ids": final_token_ids.tolist(),
        "iteration_trace": iteration_trace,
        "patch_trace": patch_trace,
        "average_token_mask_probs_by_iteration": average_token_mask_probs_by_iteration,
        "overall_average_mask_prob": overall_average_mask_prob,
        "predicted_length": target_length,
    }


def decoder_cross_attn_head_zero_ablation_sweep(
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    layer_index: Optional[int] = None,
    decoding_iterations: int = 5,
    head_indices: Optional[List[int]] = None,
    tgt_sentence: Optional[str] = None,
) -> Dict[str, object]:
    if context is None or layer_index is None:
        raise ValueError("context and layer_index are required")

    target_sentence = _resolve_text_alias(
        "target_sentence",
        target_sentence,
        tgt_sentence,
    )

    model = context["model"]
    decoder_layers = model.decoder.layers
    if layer_index < 0 or layer_index >= len(decoder_layers):
        raise IndexError(f"layer_index {layer_index} out of range for {len(decoder_layers)} decoder layers")

    encoder_attn = decoder_layers[layer_index].encoder_attn
    if encoder_attn is None:
        raise ValueError(f"decoder layer {layer_index} does not have cross-attention")

    num_heads = encoder_attn.num_heads
    if head_indices is None:
        head_indices = list(range(num_heads))
    else:
        head_indices = [int(head_index) for head_index in head_indices]
        for head_index in head_indices:
            if head_index < 0 or head_index >= num_heads:
                raise IndexError(f"head_index {head_index} out of range for {num_heads} cross-attn heads")

    head_results: List[Dict[str, object]] = []
    heatmap: List[List[float]] = []

    for head_index in head_indices:
        head_result = decoder_cross_attn_head_zero_ablation(
            target_sentence=target_sentence,
            context=context,
            layer_index=layer_index,
            head_index=head_index,
            decoding_iterations=decoding_iterations,
        )
        head_results.append(head_result)
        heatmap.append(head_result["average_token_mask_probs_by_iteration"])

    return {
        "target_sentence": target_sentence,
        "layer_index": layer_index,
        "patch_mode": "cross_attn_head_zero_ablation",
        "head_indices": head_indices,
        "iterations": [step["iteration"] for step in head_results[0]["iteration_trace"]] if head_results else [],
        "reference_decoded_text": head_results[0]["reference_decoded_text"] if head_results else "",
        "reference_token_ids": head_results[0]["reference_token_ids"] if head_results else [],
        "heatmap": heatmap,
        "head_results": head_results,
    }
