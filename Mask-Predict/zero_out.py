import torch
from typing import Dict, List, Optional

from fairseq.strategies.strategy_utils import generate_step_with_prob

from decoder_patching import (
    _apply_masked_token_updates,
    _get_attention_module_name,
    _get_attention_patch_spec,
)
from plot import (
    plot_cross_attn_zero_out_heatmap,
    plot_self_attn_zero_out_heatmap,
)
from utils import (
    _mask_tokens_for_iteration,
    _move_encoder_out_to_device,
    _predicted_length,
    _record_iteration,
    _resolve_text_alias,
    _stringify_tokens,
    clone_encoder_out,
    decode_from_encoder_output,
    get_encoder_output,
)

__all__ = [
    "decoder_self_attn_zero_out",
    "decoder_self_attn_zero_out_sweep",
    "decoder_cross_attn_zero_out",
    "decoder_cross_attn_zero_out_sweep",
    "plot_self_attn_zero_out_heatmap",
    "plot_cross_attn_zero_out_heatmap",
]


def _mean_token_mask_prob(iteration_step: Dict[str, object]) -> float:
    token_mask_probs = iteration_step["token_mask_probs"]
    return round(float(sum(token_mask_probs) / len(token_mask_probs)), 6)


def _get_attention_zero_out_patch_mode(attention_type: str) -> str:
    return f"{attention_type}_attn_zero_out"


def _run_decoder_attention_zero_out_forward(
    model,
    tgt_tokens: torch.Tensor,
    encoder_out: Dict[str, torch.Tensor],
    layer_index: int,
    head_index: int,
    attention_type: str,
):
    spec = _get_attention_patch_spec(attention_type)
    captured: Dict[str, torch.Tensor] = {}
    decoder_layers = model.decoder.layers

    if layer_index < 0 or layer_index >= len(decoder_layers):
        raise IndexError(f"layer_index {layer_index} out of range for {len(decoder_layers)} decoder layers")

    layer = decoder_layers[layer_index]
    attention_module = getattr(layer, spec["module_attr"])
    if attention_module is None:
        if attention_type == "cross":
            raise ValueError(f"decoder layer {layer_index} does not have cross-attention")
        raise ValueError(f"decoder layer {layer_index} does not have {spec['module_attr']}")

    num_heads = attention_module.num_heads
    head_dim = attention_module.head_dim

    if head_index < 0 or head_index >= num_heads:
        raise IndexError(f"head_index {head_index} out of range for {num_heads} {spec['label']} heads")

    head_start = head_index * head_dim
    head_end = head_start + head_dim

    def hook(_module, inputs):
        (attn_input,) = inputs
        captured["before_zero"] = attn_input[..., head_start:head_end].detach().cpu().clone()
        patched_input = attn_input.clone()
        patched_input[..., head_start:head_end] = 0
        captured["after_zero"] = patched_input[..., head_start:head_end].detach().cpu().clone()
        return (patched_input,)

    handle = attention_module.out_proj.register_forward_pre_hook(hook)
    original_enable_torch_version = attention_module.enable_torch_version
    attention_module.enable_torch_version = False
    try:
        decoder_out = model.decoder(tgt_tokens, encoder_out)
    finally:
        attention_module.enable_torch_version = original_enable_torch_version
        handle.remove()

    return decoder_out, captured


def _build_zero_out_trace_step(
    *,
    attention_type: str,
    layer_index: int,
    head_index: int,
    capture: Dict[str, torch.Tensor],
    iteration_step: Dict[str, object],
    iteration: int,
) -> Dict[str, object]:
    return {
        "iteration": int(iteration),
        "patch_mode": _get_attention_zero_out_patch_mode(attention_type),
        "module_name": _get_attention_module_name(attention_type, layer_index),
        "head_index": head_index,
        "head_output_shape": list(capture["after_zero"].shape),
        "head_output_norm_before_zero": round(float(capture["before_zero"].norm().item()), 6),
        "head_output_norm_after_zero": round(float(capture["after_zero"].norm().item()), 6),
        "average_token_mask_prob": _mean_token_mask_prob(iteration_step),
    }


def _decoder_attention_zero_out(
    attention_type: str,
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

    patch_mode = _get_attention_zero_out_patch_mode(attention_type)
    model = context["model"]
    device = context["device"]
    args = context["args"]
    tgt_dict = context["task"].target_dictionary
    module_name = _get_attention_module_name(attention_type, layer_index)

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
        decoder_out, capture = _run_decoder_attention_zero_out_forward(
            model,
            target_tgt_tokens,
            target_encoder_out,
            layer_index,
            head_index,
            attention_type=attention_type,
        )
        target_tgt_tokens, target_token_probs, _ = generate_step_with_prob(decoder_out)
        _record_iteration(iteration_trace, 0, target_tgt_tokens, target_token_probs, tgt_dict, args.remove_bpe)
        patch_trace.append(
            _build_zero_out_trace_step(
                attention_type=attention_type,
                layer_index=layer_index,
                head_index=head_index,
                capture=capture,
                iteration_step=iteration_trace[-1],
                iteration=0,
            )
        )

        for counter in range(1, iterations):
            target_masked_tokens, target_mask_ind, selected_mask_token_ids, selected_mask_scores = _mask_tokens_for_iteration(
                target_tgt_tokens,
                target_token_probs,
                tgt_dict,
                counter,
                iterations,
            )
            decoder_out, capture = _run_decoder_attention_zero_out_forward(
                model,
                target_masked_tokens,
                target_encoder_out,
                layer_index,
                head_index,
                attention_type=attention_type,
            )
            target_new_tgt_tokens, target_new_token_probs, _ = generate_step_with_prob(decoder_out)
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
                _build_zero_out_trace_step(
                    attention_type=attention_type,
                    layer_index=layer_index,
                    head_index=head_index,
                    capture=capture,
                    iteration_step=iteration_trace[-1],
                    iteration=counter,
                )
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
        "patch_mode": patch_mode,
        "module_name": module_name,
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


def _decoder_attention_zero_out_sweep(
    attention_type: str,
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

    spec = _get_attention_patch_spec(attention_type)
    patch_mode = _get_attention_zero_out_patch_mode(attention_type)
    model = context["model"]
    decoder_layers = model.decoder.layers
    if layer_index < 0 or layer_index >= len(decoder_layers):
        raise IndexError(f"layer_index {layer_index} out of range for {len(decoder_layers)} decoder layers")

    attention_module = getattr(decoder_layers[layer_index], spec["module_attr"])
    if attention_module is None:
        if attention_type == "cross":
            raise ValueError(f"decoder layer {layer_index} does not have cross-attention")
        raise ValueError(f"decoder layer {layer_index} does not have {spec['module_attr']}")

    num_heads = attention_module.num_heads
    if head_indices is None:
        head_indices = list(range(num_heads))
    else:
        head_indices = [int(head_index) for head_index in head_indices]
        for head_index in head_indices:
            if head_index < 0 or head_index >= num_heads:
                raise IndexError(f"head_index {head_index} out of range for {num_heads} {spec['label']} heads")

    head_results: List[Dict[str, object]] = []
    heatmap: List[List[float]] = []

    for head_index in head_indices:
        head_result = _decoder_attention_zero_out(
            attention_type=attention_type,
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
        "patch_mode": patch_mode,
        "head_indices": head_indices,
        "iterations": [step["iteration"] for step in head_results[0]["iteration_trace"]] if head_results else [],
        "reference_decoded_text": head_results[0]["reference_decoded_text"] if head_results else "",
        "reference_token_ids": head_results[0]["reference_token_ids"] if head_results else [],
        "heatmap": heatmap,
        "head_results": head_results,
    }


def decoder_self_attn_zero_out(
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    layer_index: Optional[int] = None,
    head_index: Optional[int] = None,
    decoding_iterations: int = 5,
    tgt_sentence: Optional[str] = None,
) -> Dict[str, object]:
    return _decoder_attention_zero_out(
        "self",
        target_sentence=target_sentence,
        context=context,
        layer_index=layer_index,
        head_index=head_index,
        decoding_iterations=decoding_iterations,
        tgt_sentence=tgt_sentence,
    )


def decoder_self_attn_zero_out_sweep(
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    layer_index: Optional[int] = None,
    decoding_iterations: int = 5,
    head_indices: Optional[List[int]] = None,
    tgt_sentence: Optional[str] = None,
) -> Dict[str, object]:
    return _decoder_attention_zero_out_sweep(
        "self",
        target_sentence=target_sentence,
        context=context,
        layer_index=layer_index,
        decoding_iterations=decoding_iterations,
        head_indices=head_indices,
        tgt_sentence=tgt_sentence,
    )


def decoder_cross_attn_zero_out(
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    layer_index: Optional[int] = None,
    head_index: Optional[int] = None,
    decoding_iterations: int = 5,
    tgt_sentence: Optional[str] = None,
) -> Dict[str, object]:
    return _decoder_attention_zero_out(
        "cross",
        target_sentence=target_sentence,
        context=context,
        layer_index=layer_index,
        head_index=head_index,
        decoding_iterations=decoding_iterations,
        tgt_sentence=tgt_sentence,
    )


def decoder_cross_attn_zero_out_sweep(
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    layer_index: Optional[int] = None,
    decoding_iterations: int = 5,
    head_indices: Optional[List[int]] = None,
    tgt_sentence: Optional[str] = None,
) -> Dict[str, object]:
    return _decoder_attention_zero_out_sweep(
        "cross",
        target_sentence=target_sentence,
        context=context,
        layer_index=layer_index,
        decoding_iterations=decoding_iterations,
        head_indices=head_indices,
        tgt_sentence=tgt_sentence,
    )
