import io
import torch
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional


from fairseq import checkpoint_utils, options, tasks
from fairseq.strategies.strategy_utils import generate_step_with_prob

from utils import REPO_ROOT

mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["savefig.dpi"] = 300




DEFAULT_MODEL_DIR = REPO_ROOT / "checkpoints" / "maskPredict_en_de"


@contextmanager
def _suppress_fairseq_output():
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FairseqModel is deprecated, please use FairseqEncoderDecoderModel or BaseFairseqModel instead",
            category=UserWarning,
        )
        yield


def _build_generation_args(
    data_bin_dir: Path,
    model_dir: Path,
    source_lang: str,
    target_lang: str,
    decoding_iterations: int,
    length_beam: int,
    max_sentences: int,
    use_cpu: bool,
):
    checkpoint_path = Path(model_dir) / "checkpoint_best.pt"
    cli_args = [
        str(data_bin_dir),
        "--path",
        str(checkpoint_path),
        "--task",
        "translation_self",
        "--source-lang",
        source_lang,
        "--target-lang",
        target_lang,
        "--remove-bpe",
        "--max-sentences",
        str(max_sentences),
        "--decoding-iterations",
        str(decoding_iterations),
        "--decoding-strategy",
        "mask_predict",
        "--length-beam",
        str(length_beam),
        "--gen-subset",
        "test",
    ]
    if use_cpu:
        cli_args.append("--cpu")

    parser = options.get_generation_parser()
    return options.parse_args_and_arch(parser, input_args=cli_args)


def _clone_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return value.detach().cpu().clone()


def clone_encoder_out(encoder_out: Dict[str, torch.Tensor]) -> Dict[str, Optional[torch.Tensor]]:
    return {
        "encoder_out": _clone_tensor(encoder_out["encoder_out"]),
        "encoder_padding_mask": _clone_tensor(encoder_out["encoder_padding_mask"]),
        "predicted_lengths": _clone_tensor(encoder_out["predicted_lengths"]),
    }


def _move_encoder_out_to_device(
    encoder_out: Dict[str, Optional[torch.Tensor]],
    device: torch.device,
) -> Dict[str, Optional[torch.Tensor]]:
    moved = {}
    for key, value in encoder_out.items():
        moved[key] = None if value is None else value.to(device)
    return moved


def _stringify_tokens(token_ids: torch.Tensor, dictionary, remove_bpe: Optional[str]) -> str:
    return dictionary.string(token_ids, remove_bpe, escape_unk=True)


def _predicted_length(encoder_out: Dict[str, torch.Tensor]) -> int:
    predicted_length = int(encoder_out["predicted_lengths"].argmax(dim=-1).item())
    return max(2, predicted_length)


def _record_iteration(
    iteration_trace: List[Dict[str, object]],
    iteration: int,
    tokens: torch.Tensor,
    probs: torch.Tensor,
    dictionary,
    remove_bpe: Optional[str],
    selected_mask_token_ids: Optional[List[int]] = None,
    selected_mask_scores: Optional[List[float]] = None,
):
    token_ids = tokens[0].detach().cpu()
    token_confidences = probs[0].detach().cpu()
    token_mask_probs = (1.0 - token_confidences).clamp(min=0.0, max=1.0)
    selected_mask_tokens = [] if selected_mask_token_ids is None else [dictionary[int(token_id)] for token_id in selected_mask_token_ids]
    iteration_trace.append(
        {
            "iteration": int(iteration),
            "text": _stringify_tokens(token_ids, dictionary, remove_bpe),
            "token_ids": token_ids.tolist(),
            "token_mask_probs": [round(float(x), 6) for x in token_mask_probs.tolist()],
            "selected_mask_tokens": selected_mask_tokens,
            "selected_mask_probs": [] if selected_mask_scores is None else [round(float(x), 6) for x in selected_mask_scores],
        }
    )


def _mask_tokens_for_iteration(
    tgt_tokens: torch.Tensor,
    token_probs: torch.Tensor,
    tgt_dict,
    counter: int,
    iterations: int,
):
    seq_len = tgt_tokens.size(1)
    num_mask = max(1, int(seq_len * (1.0 - counter / iterations)))
    mask_ind = token_probs[0].topk(num_mask, largest=False).indices
    selected_mask_token_ids = tgt_tokens[0, mask_ind].detach().cpu().tolist()
    selected_mask_scores = (1.0 - token_probs[0, mask_ind]).detach().cpu().tolist()

    masked_tokens = tgt_tokens.clone()
    masked_tokens[0, mask_ind] = tgt_dict.mask()
    return masked_tokens, mask_ind, selected_mask_token_ids, selected_mask_scores


def _resolve_text_alias(display_name: str, *values: Optional[str], required: bool = True):
    provided_values = [value for value in values if value is not None]
    if not provided_values:
        if required:
            raise ValueError(f"{display_name} is required")
        return None

    resolved_value = provided_values[0]
    for value in provided_values[1:]:
        if value != resolved_value:
            raise ValueError(f"{display_name} provided by multiple aliases but values do not match")
    return resolved_value


def _resolve_sentence_pair(
    source_sentence: Optional[str] = None,
    target_sentence: Optional[str] = None,
    src_sentence: Optional[str] = None,
    tgt_sentence: Optional[str] = None,
):
    resolved_source_sentence = _resolve_text_alias(
        "source_sentence",
        source_sentence,
        src_sentence,
    )
    resolved_target_sentence = _resolve_text_alias(
        "target_sentence",
        target_sentence,
        tgt_sentence,
    )
    return resolved_source_sentence, resolved_target_sentence


def _normalize_iteration_indices(
    iteration_indices: Optional[List[int]],
    total_iterations: int,
    name: str,
) -> List[int]:
    if iteration_indices is None:
        return list(range(total_iterations))

    normalized = [int(iteration_index) for iteration_index in iteration_indices]
    for iteration_index in normalized:
        if iteration_index < 0 or iteration_index >= total_iterations:
            raise IndexError(
                f"{name} index {iteration_index} out of range for total iterations {total_iterations}"
            )
    return normalized


def load_mask_predict_context(
    data_bin_dir: Path,
    model_dir: Path = DEFAULT_MODEL_DIR,
    source_lang: str = "en",
    target_lang: str = "de",
    decoding_iterations: int = 5,
    length_beam: int = 1,
    max_sentences: int = 20,
    use_cpu: bool = True,
) -> Dict[str, object]:
    data_bin_dir = Path(data_bin_dir)
    model_dir = Path(model_dir)
    args = _build_generation_args(
        data_bin_dir=data_bin_dir,
        model_dir=model_dir,
        source_lang=source_lang,
        target_lang=target_lang,
        decoding_iterations=decoding_iterations,
        length_beam=length_beam,
        max_sentences=max_sentences,
        use_cpu=use_cpu,
    )

    with _suppress_fairseq_output():
        task = tasks.setup_task(args)
        models, _ = checkpoint_utils.load_model_ensemble(
            args.path.split(":"),
            arg_overrides=eval(args.model_overrides),
            task=task,
        )

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    model = models[0].to(device)
    model.eval()

    return {
        "args": args,
        "task": task,
        "model": model,
        "device": device,
        "data_bin_dir": data_bin_dir,
        "model_dir": model_dir,
        "source_lang": source_lang,
        "target_lang": target_lang,
    }


def get_encoder_output(sentence: str, context: Dict[str, object]) -> Dict[str, object]:
    task = context["task"]
    model = context["model"]
    device = context["device"]

    src_tokens = task.source_dictionary.encode_line(sentence, add_if_not_exist=False).long().unsqueeze(0).to(device)
    src_lengths = torch.LongTensor([src_tokens.size(1)]).to(device)

    with torch.no_grad():
        encoder_out = model.encoder(src_tokens=src_tokens, src_lengths=src_lengths)

    return {
        "sentence": sentence,
        "src_tokens": src_tokens.detach().cpu(),
        "src_lengths": src_lengths.detach().cpu(),
        "encoder_out": clone_encoder_out(encoder_out),
    }


def decode_from_encoder_output(
    encoder_output: Dict[str, torch.Tensor],
    context: Dict[str, object],
    decoding_iterations: int = 5,
) -> Dict[str, object]:
    task = context["task"]
    model = context["model"]
    device = context["device"]
    args = context["args"]
    tgt_dict = task.target_dictionary

    encoder_out = _move_encoder_out_to_device(clone_encoder_out(encoder_output), device)
    predicted_length = _predicted_length(encoder_out)
    tgt_tokens = torch.full((1, predicted_length), tgt_dict.mask(), dtype=torch.long, device=device)
    iterations = predicted_length if decoding_iterations is None else decoding_iterations
    iteration_trace: List[Dict[str, object]] = []

    with torch.no_grad():
        decoder_out = model.decoder(tgt_tokens, encoder_out)
        tgt_tokens, token_probs, _ = generate_step_with_prob(decoder_out)
        _record_iteration(iteration_trace, 0, tgt_tokens, token_probs, tgt_dict, args.remove_bpe)

        for counter in range(1, iterations):
            masked_tokens, mask_ind, selected_mask_token_ids, selected_mask_scores = _mask_tokens_for_iteration(
                tgt_tokens,
                token_probs,
                tgt_dict,
                counter,
                iterations,
            )

            decoder_out = model.decoder(masked_tokens, encoder_out)
            new_tgt_tokens, new_token_probs, _ = generate_step_with_prob(decoder_out)

            tgt_tokens = masked_tokens
            token_probs = token_probs.clone()
            tgt_tokens[0, mask_ind] = new_tgt_tokens[0, mask_ind]
            token_probs[0, mask_ind] = new_token_probs[0, mask_ind]

            _record_iteration(
                iteration_trace,
                counter,
                tgt_tokens,
                token_probs,
                tgt_dict,
                args.remove_bpe,
                selected_mask_token_ids=selected_mask_token_ids,
                selected_mask_scores=selected_mask_scores,
            )

    final_token_ids = tgt_tokens[0].detach().cpu()
    return {
        "decoded_text": _stringify_tokens(final_token_ids, tgt_dict, args.remove_bpe),
        "token_ids": final_token_ids.tolist(),
        "iteration_trace": iteration_trace,
        "predicted_length": predicted_length,
    }


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


def run_decoder_patching_experiment(
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


def run_decoder_layer_sweep_experiment(
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
        layer_result = run_decoder_patching_experiment(
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


def _run_decoder_self_attn_patch_forward(
    model,
    tgt_tokens: torch.Tensor,
    encoder_out: Dict[str, torch.Tensor],
    layer_index: int,
    token_position: int,
    source_attn_token_state: Optional[torch.Tensor] = None,
):
    captured: Dict[str, torch.Tensor] = {}
    decoder_layers = model.decoder.layers

    if layer_index < 0 or layer_index >= len(decoder_layers):
        raise IndexError(f"layer_index {layer_index} out of range for {len(decoder_layers)} decoder layers")
    if token_position < 0 or token_position >= tgt_tokens.size(1):
        raise IndexError(f"token_position {token_position} out of range for decoder length {tgt_tokens.size(1)}")

    layer = decoder_layers[layer_index]

    def hook(_module, _inputs, output):
        hidden, attn = output
        captured["before_patch"] = hidden[token_position, 0, :].detach().cpu().clone()
        patched_hidden = hidden
        if source_attn_token_state is not None:
            source_state = source_attn_token_state.to(hidden.device, dtype=hidden.dtype)
            if source_state.shape != hidden[token_position, 0, :].shape:
                raise ValueError(
                    "self-attn token patching requires source and target token activations to have the same shape, "
                    f"got {tuple(source_state.shape)} and {tuple(hidden[token_position, 0, :].shape)}"
                )
            patched_hidden = hidden.clone()
            patched_hidden[token_position, 0, :] = source_state
        captured["after_patch"] = patched_hidden[token_position, 0, :].detach().cpu().clone()
        return patched_hidden, attn

    handle = layer.self_attn.register_forward_hook(hook)
    try:
        decoder_out = model.decoder(tgt_tokens, encoder_out)
    finally:
        handle.remove()

    return decoder_out, captured


def _run_decoder_cross_attn_patch_forward(
    model,
    tgt_tokens: torch.Tensor,
    encoder_out: Dict[str, torch.Tensor],
    layer_index: int,
    token_position: int,
    source_attn_token_state: Optional[torch.Tensor] = None,
):
    captured: Dict[str, torch.Tensor] = {}
    decoder_layers = model.decoder.layers

    if layer_index < 0 or layer_index >= len(decoder_layers):
        raise IndexError(f"layer_index {layer_index} out of range for {len(decoder_layers)} decoder layers")
    if token_position < 0 or token_position >= tgt_tokens.size(1):
        raise IndexError(f"token_position {token_position} out of range for decoder length {tgt_tokens.size(1)}")

    layer = decoder_layers[layer_index]
    if layer.encoder_attn is None:
        raise ValueError(f"decoder layer {layer_index} does not have encoder_attn")

    def hook(_module, _inputs, output):
        hidden, attn = output
        captured["before_patch"] = hidden[token_position, 0, :].detach().cpu().clone()
        patched_hidden = hidden
        if source_attn_token_state is not None:
            source_state = source_attn_token_state.to(hidden.device, dtype=hidden.dtype)
            if source_state.shape != hidden[token_position, 0, :].shape:
                raise ValueError(
                    "cross-attn token patching requires source and target token activations to have the same shape, "
                    f"got {tuple(source_state.shape)} and {tuple(hidden[token_position, 0, :].shape)}"
                )
            patched_hidden = hidden.clone()
            patched_hidden[token_position, 0, :] = source_state
        captured["after_patch"] = patched_hidden[token_position, 0, :].detach().cpu().clone()
        return patched_hidden, attn

    handle = layer.encoder_attn.register_forward_hook(hook)
    try:
        decoder_out = model.decoder(tgt_tokens, encoder_out)
    finally:
        handle.remove()

    return decoder_out, captured


def _run_decoder_self_attn_full_layer_patch_forward(
    model,
    tgt_tokens: torch.Tensor,
    encoder_out: Dict[str, torch.Tensor],
    layer_index: int,
    source_attn_hidden_state: Optional[torch.Tensor] = None,
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
        if source_attn_hidden_state is not None:
            source_hidden = source_attn_hidden_state.to(hidden.device, dtype=hidden.dtype)
            if source_hidden.shape != hidden.shape:
                raise ValueError(
                    "self-attn full-layer patching requires source and target activations to have the same shape, "
                    f"got {tuple(source_hidden.shape)} and {tuple(hidden.shape)}"
                )
            patched_hidden = source_hidden.clone()
        captured["after_patch"] = patched_hidden.detach().cpu().clone()
        return patched_hidden, attn

    handle = layer.self_attn.register_forward_hook(hook)
    try:
        decoder_out = model.decoder(tgt_tokens, encoder_out)
    finally:
        handle.remove()

    return decoder_out, captured


def _run_decoder_cross_attn_full_layer_patch_forward(
    model,
    tgt_tokens: torch.Tensor,
    encoder_out: Dict[str, torch.Tensor],
    layer_index: int,
    source_attn_hidden_state: Optional[torch.Tensor] = None,
):
    captured: Dict[str, torch.Tensor] = {}
    decoder_layers = model.decoder.layers

    if layer_index < 0 or layer_index >= len(decoder_layers):
        raise IndexError(f"layer_index {layer_index} out of range for {len(decoder_layers)} decoder layers")

    layer = decoder_layers[layer_index]
    if layer.encoder_attn is None:
        raise ValueError(f"decoder layer {layer_index} does not have encoder_attn")

    def hook(_module, _inputs, output):
        hidden, attn = output
        captured["before_patch"] = hidden.detach().cpu().clone()
        patched_hidden = hidden
        if source_attn_hidden_state is not None:
            source_hidden = source_attn_hidden_state.to(hidden.device, dtype=hidden.dtype)
            if source_hidden.shape != hidden.shape:
                raise ValueError(
                    "cross-attn full-layer patching requires source and target activations to have the same shape, "
                    f"got {tuple(source_hidden.shape)} and {tuple(hidden.shape)}"
                )
            patched_hidden = source_hidden.clone()
        captured["after_patch"] = patched_hidden.detach().cpu().clone()
        return patched_hidden, attn

    handle = layer.encoder_attn.register_forward_hook(hook)
    try:
        decoder_out = model.decoder(tgt_tokens, encoder_out)
    finally:
        handle.remove()

    return decoder_out, captured


def run_decoder_self_attn_token_patching_experiment(
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
    if context is None or layer_index is None or token_position is None:
        raise ValueError("context, layer_index, and token_position are required")
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
    if token_position < 0 or token_position >= source_length or token_position >= target_length:
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

    iteration_trace: List[Dict[str, object]] = []
    patch_trace: List[Dict[str, object]] = []

    with torch.no_grad():
        source_decoder_out, source_capture = _run_decoder_self_attn_patch_forward(
            model,
            source_tgt_tokens,
            source_encoder_out,
            layer_index,
            token_position,
        )
        source_tgt_tokens, source_token_probs, _ = generate_step_with_prob(source_decoder_out)

        patched_decoder_out, target_capture = _run_decoder_self_attn_patch_forward(
            model,
            target_tgt_tokens,
            target_encoder_out,
            layer_index,
            token_position,
            source_attn_token_state=source_capture["after_patch"] if 0 in patch_iteration_set else None,
        )
        target_tgt_tokens, target_token_probs, _ = generate_step_with_prob(patched_decoder_out)

        _record_iteration(iteration_trace, 0, target_tgt_tokens, target_token_probs, tgt_dict, args.remove_bpe)
        patch_trace.append(
            {
                "iteration": 0,
                "source_text": _stringify_tokens(source_tgt_tokens[0].detach().cpu(), tgt_dict, args.remove_bpe),
                "patched_text": _stringify_tokens(target_tgt_tokens[0].detach().cpu(), tgt_dict, args.remove_bpe),
                "patch_mode": "self_attn_source_to_target",
                "module_name": f"decoder.layers.{layer_index}.self_attn",
                "token_position": token_position,
                "activation_shape": list(target_capture["after_patch"].shape),
                "iteration_was_patched": 0 in patch_iteration_set,
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
            source_decoder_out, source_capture = _run_decoder_self_attn_patch_forward(
                model,
                source_masked_tokens,
                source_encoder_out,
                layer_index,
                token_position,
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
            patched_decoder_out, target_capture = _run_decoder_self_attn_patch_forward(
                model,
                target_masked_tokens,
                target_encoder_out,
                layer_index,
                token_position,
                source_attn_token_state=source_capture["after_patch"] if counter in patch_iteration_set else None,
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
                    "patch_mode": "self_attn_source_to_target",
                    "module_name": f"decoder.layers.{layer_index}.self_attn",
                    "token_position": token_position,
                    "activation_shape": list(target_capture["after_patch"].shape),
                    "iteration_was_patched": counter in patch_iteration_set,
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
        "patch_mode": "self_attn_source_to_target",
        "module_name": f"decoder.layers.{layer_index}.self_attn",
        "source_encoder": source_encoder,
        "target_encoder": target_encoder,
        "patch_iterations": patch_iteration_list,
        "decoded_text": _stringify_tokens(final_token_ids, tgt_dict, args.remove_bpe),
        "token_ids": final_token_ids.tolist(),
        "iteration_trace": iteration_trace,
        "patch_trace": patch_trace,
        "predicted_length": target_length,
    }


def run_decoder_self_attn_full_layer_patching_experiment(
    source_sentence: Optional[str] = None,
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    layer_index: Optional[int] = None,
    decoding_iterations: int = 5,
    patch_iterations: Optional[List[int]] = None,
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
            "self-attn full-layer patching requires source and target predicted lengths to match, "
            f"got {source_length} and {target_length}"
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

    iteration_trace: List[Dict[str, object]] = []
    patch_trace: List[Dict[str, object]] = []

    with torch.no_grad():
        source_decoder_out, source_capture = _run_decoder_self_attn_full_layer_patch_forward(
            model,
            source_tgt_tokens,
            source_encoder_out,
            layer_index,
        )
        source_tgt_tokens, source_token_probs, _ = generate_step_with_prob(source_decoder_out)

        patched_decoder_out, target_capture = _run_decoder_self_attn_full_layer_patch_forward(
            model,
            target_tgt_tokens,
            target_encoder_out,
            layer_index,
            source_attn_hidden_state=source_capture["after_patch"] if 0 in patch_iteration_set else None,
        )
        target_tgt_tokens, target_token_probs, _ = generate_step_with_prob(patched_decoder_out)

        _record_iteration(iteration_trace, 0, target_tgt_tokens, target_token_probs, tgt_dict, args.remove_bpe)
        patch_trace.append(
            {
                "iteration": 0,
                "source_text": _stringify_tokens(source_tgt_tokens[0].detach().cpu(), tgt_dict, args.remove_bpe),
                "patched_text": _stringify_tokens(target_tgt_tokens[0].detach().cpu(), tgt_dict, args.remove_bpe),
                "patch_mode": "self_attn_full_layer_source_to_target",
                "module_name": f"decoder.layers.{layer_index}.self_attn",
                "activation_shape": list(target_capture["after_patch"].shape),
                "iteration_was_patched": 0 in patch_iteration_set,
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
            source_decoder_out, source_capture = _run_decoder_self_attn_full_layer_patch_forward(
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
            patched_decoder_out, target_capture = _run_decoder_self_attn_full_layer_patch_forward(
                model,
                target_masked_tokens,
                target_encoder_out,
                layer_index,
                source_attn_hidden_state=source_capture["after_patch"] if counter in patch_iteration_set else None,
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
                    "patch_mode": "self_attn_full_layer_source_to_target",
                    "module_name": f"decoder.layers.{layer_index}.self_attn",
                    "activation_shape": list(target_capture["after_patch"].shape),
                    "iteration_was_patched": counter in patch_iteration_set,
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
        "patch_mode": "self_attn_full_layer_source_to_target",
        "module_name": f"decoder.layers.{layer_index}.self_attn",
        "source_encoder": source_encoder,
        "target_encoder": target_encoder,
        "patch_iterations": patch_iteration_list,
        "decoded_text": _stringify_tokens(final_token_ids, tgt_dict, args.remove_bpe),
        "token_ids": final_token_ids.tolist(),
        "iteration_trace": iteration_trace,
        "patch_trace": patch_trace,
        "predicted_length": target_length,
    }


def run_decoder_cross_attn_token_patching_experiment(
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
    if context is None or layer_index is None or token_position is None:
        raise ValueError("context, layer_index, and token_position are required")
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
    if token_position < 0 or token_position >= source_length or token_position >= target_length:
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

    iteration_trace: List[Dict[str, object]] = []
    patch_trace: List[Dict[str, object]] = []

    with torch.no_grad():
        source_decoder_out, source_capture = _run_decoder_cross_attn_patch_forward(
            model,
            source_tgt_tokens,
            source_encoder_out,
            layer_index,
            token_position,
        )
        source_tgt_tokens, source_token_probs, _ = generate_step_with_prob(source_decoder_out)

        patched_decoder_out, target_capture = _run_decoder_cross_attn_patch_forward(
            model,
            target_tgt_tokens,
            target_encoder_out,
            layer_index,
            token_position,
            source_attn_token_state=source_capture["after_patch"] if 0 in patch_iteration_set else None,
        )
        target_tgt_tokens, target_token_probs, _ = generate_step_with_prob(patched_decoder_out)

        _record_iteration(iteration_trace, 0, target_tgt_tokens, target_token_probs, tgt_dict, args.remove_bpe)
        patch_trace.append(
            {
                "iteration": 0,
                "source_text": _stringify_tokens(source_tgt_tokens[0].detach().cpu(), tgt_dict, args.remove_bpe),
                "patched_text": _stringify_tokens(target_tgt_tokens[0].detach().cpu(), tgt_dict, args.remove_bpe),
                "patch_mode": "cross_attn_source_to_target",
                "module_name": f"decoder.layers.{layer_index}.encoder_attn",
                "token_position": token_position,
                "activation_shape": list(target_capture["after_patch"].shape),
                "iteration_was_patched": 0 in patch_iteration_set,
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
            source_decoder_out, source_capture = _run_decoder_cross_attn_patch_forward(
                model,
                source_masked_tokens,
                source_encoder_out,
                layer_index,
                token_position,
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
            patched_decoder_out, target_capture = _run_decoder_cross_attn_patch_forward(
                model,
                target_masked_tokens,
                target_encoder_out,
                layer_index,
                token_position,
                source_attn_token_state=source_capture["after_patch"] if counter in patch_iteration_set else None,
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
                    "patch_mode": "cross_attn_source_to_target",
                    "module_name": f"decoder.layers.{layer_index}.encoder_attn",
                    "token_position": token_position,
                    "activation_shape": list(target_capture["after_patch"].shape),
                    "iteration_was_patched": counter in patch_iteration_set,
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
        "patch_mode": "cross_attn_source_to_target",
        "module_name": f"decoder.layers.{layer_index}.encoder_attn",
        "source_encoder": source_encoder,
        "target_encoder": target_encoder,
        "patch_iterations": patch_iteration_list,
        "decoded_text": _stringify_tokens(final_token_ids, tgt_dict, args.remove_bpe),
        "token_ids": final_token_ids.tolist(),
        "iteration_trace": iteration_trace,
        "patch_trace": patch_trace,
        "predicted_length": target_length,
    }


def run_decoder_cross_attn_full_layer_patching_experiment(
    source_sentence: Optional[str] = None,
    target_sentence: Optional[str] = None,
    context: Optional[Dict[str, object]] = None,
    layer_index: Optional[int] = None,
    decoding_iterations: int = 5,
    patch_iterations: Optional[List[int]] = None,
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
            "cross-attn full-layer patching requires source and target predicted lengths to match, "
            f"got {source_length} and {target_length}"
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

    iteration_trace: List[Dict[str, object]] = []
    patch_trace: List[Dict[str, object]] = []

    with torch.no_grad():
        source_decoder_out, source_capture = _run_decoder_cross_attn_full_layer_patch_forward(
            model,
            source_tgt_tokens,
            source_encoder_out,
            layer_index,
        )
        source_tgt_tokens, source_token_probs, _ = generate_step_with_prob(source_decoder_out)

        patched_decoder_out, target_capture = _run_decoder_cross_attn_full_layer_patch_forward(
            model,
            target_tgt_tokens,
            target_encoder_out,
            layer_index,
            source_attn_hidden_state=source_capture["after_patch"] if 0 in patch_iteration_set else None,
        )
        target_tgt_tokens, target_token_probs, _ = generate_step_with_prob(patched_decoder_out)

        _record_iteration(iteration_trace, 0, target_tgt_tokens, target_token_probs, tgt_dict, args.remove_bpe)
        patch_trace.append(
            {
                "iteration": 0,
                "source_text": _stringify_tokens(source_tgt_tokens[0].detach().cpu(), tgt_dict, args.remove_bpe),
                "patched_text": _stringify_tokens(target_tgt_tokens[0].detach().cpu(), tgt_dict, args.remove_bpe),
                "patch_mode": "cross_attn_full_layer_source_to_target",
                "module_name": f"decoder.layers.{layer_index}.encoder_attn",
                "activation_shape": list(target_capture["after_patch"].shape),
                "iteration_was_patched": 0 in patch_iteration_set,
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
            source_decoder_out, source_capture = _run_decoder_cross_attn_full_layer_patch_forward(
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
            patched_decoder_out, target_capture = _run_decoder_cross_attn_full_layer_patch_forward(
                model,
                target_masked_tokens,
                target_encoder_out,
                layer_index,
                source_attn_hidden_state=source_capture["after_patch"] if counter in patch_iteration_set else None,
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
                    "patch_mode": "cross_attn_full_layer_source_to_target",
                    "module_name": f"decoder.layers.{layer_index}.encoder_attn",
                    "activation_shape": list(target_capture["after_patch"].shape),
                    "iteration_was_patched": counter in patch_iteration_set,
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
        "patch_mode": "cross_attn_full_layer_source_to_target",
        "module_name": f"decoder.layers.{layer_index}.encoder_attn",
        "source_encoder": source_encoder,
        "target_encoder": target_encoder,
        "patch_iterations": patch_iteration_list,
        "decoded_text": _stringify_tokens(final_token_ids, tgt_dict, args.remove_bpe),
        "token_ids": final_token_ids.tolist(),
        "iteration_trace": iteration_trace,
        "patch_trace": patch_trace,
        "predicted_length": target_length,
    }


def run_decoder_self_attn_layer_iteration_sweep_experiment(
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

    if patch_token_position is None:
        patch_token_position = tracked_token_position
    if patch_token_position < 0 or patch_token_position >= len(target_reference["token_ids"]):
        raise IndexError(
            f"patch_token_position {patch_token_position} out of range for decoded length {len(target_reference['token_ids'])}"
        )

    patch_iteration_list = _normalize_iteration_indices(
        patch_iterations,
        decoding_iterations,
        name="patch_iterations",
    )

    tracked_token_label = tgt_dict[target_reference["token_ids"][tracked_token_position]]
    patch_token_label = tgt_dict[target_reference["token_ids"][patch_token_position]]

    layer_results: List[Dict[str, object]] = []
    heatmap: List[List[float]] = []

    for layer_index in layer_indices:
        patch_iteration_results: List[Dict[str, object]] = []
        tracked_token_mask_probs: List[float] = []
        tracked_token_texts: List[str] = []
        decoded_texts: List[str] = []

        for patch_iteration in patch_iteration_list:
            patch_result = run_decoder_self_attn_token_patching_experiment(
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

        layer_results.append(
            {
                "layer_index": layer_index,
                "tracked_token_position": tracked_token_position,
                "tracked_token_label": tracked_token_label,
                "patch_token_position": patch_token_position,
                "patch_token_label": patch_token_label,
                "tracked_token_mask_probs_by_patch_iteration": tracked_token_mask_probs,
                "tracked_token_texts_by_patch_iteration": tracked_token_texts,
                "decoded_texts_by_patch_iteration": decoded_texts,
                "patch_iteration_results": patch_iteration_results,
            }
        )
        heatmap.append(tracked_token_mask_probs)

    return {
        "source_sentence": source_sentence,
        "target_sentence": target_sentence,
        "patch_mode": "self_attn_source_to_target",
        "tracked_token_position": tracked_token_position,
        "tracked_token_label": tracked_token_label,
        "patch_token_position": patch_token_position,
        "patch_token_label": patch_token_label,
        "reference_decoded_text": target_reference["decoded_text"],
        "reference_token_ids": target_reference["token_ids"],
        "layer_indices": layer_indices,
        "patch_iterations": patch_iteration_list,
        "heatmap": heatmap,
        "layer_results": layer_results,
    }


def run_decoder_self_attn_full_layer_iteration_sweep_experiment(
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

    patch_iteration_list = _normalize_iteration_indices(
        patch_iterations,
        decoding_iterations,
        name="patch_iterations",
    )

    tracked_token_label = tgt_dict[target_reference["token_ids"][tracked_token_position]]
    layer_results: List[Dict[str, object]] = []
    heatmap: List[List[float]] = []

    for layer_index in layer_indices:
        patch_iteration_results: List[Dict[str, object]] = []
        tracked_token_mask_probs: List[float] = []
        tracked_token_texts: List[str] = []
        decoded_texts: List[str] = []

        for patch_iteration in patch_iteration_list:
            patch_result = run_decoder_self_attn_full_layer_patching_experiment(
                source_sentence=source_sentence,
                target_sentence=target_sentence,
                context=context,
                layer_index=layer_index,
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

        layer_results.append(
            {
                "layer_index": layer_index,
                "tracked_token_position": tracked_token_position,
                "tracked_token_label": tracked_token_label,
                "tracked_token_mask_probs_by_patch_iteration": tracked_token_mask_probs,
                "tracked_token_texts_by_patch_iteration": tracked_token_texts,
                "decoded_texts_by_patch_iteration": decoded_texts,
                "patch_iteration_results": patch_iteration_results,
            }
        )
        heatmap.append(tracked_token_mask_probs)

    return {
        "source_sentence": source_sentence,
        "target_sentence": target_sentence,
        "patch_mode": "self_attn_full_layer_source_to_target",
        "tracked_token_position": tracked_token_position,
        "tracked_token_label": tracked_token_label,
        "reference_decoded_text": target_reference["decoded_text"],
        "reference_token_ids": target_reference["token_ids"],
        "layer_indices": layer_indices,
        "patch_iterations": patch_iteration_list,
        "heatmap": heatmap,
        "layer_results": layer_results,
    }


def run_decoder_cross_attn_layer_iteration_sweep_experiment(
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

    if patch_token_position is None:
        patch_token_position = tracked_token_position
    if patch_token_position < 0 or patch_token_position >= len(target_reference["token_ids"]):
        raise IndexError(
            f"patch_token_position {patch_token_position} out of range for decoded length {len(target_reference['token_ids'])}"
        )

    patch_iteration_list = _normalize_iteration_indices(
        patch_iterations,
        decoding_iterations,
        name="patch_iterations",
    )

    tracked_token_label = tgt_dict[target_reference["token_ids"][tracked_token_position]]
    patch_token_label = tgt_dict[target_reference["token_ids"][patch_token_position]]

    layer_results: List[Dict[str, object]] = []
    heatmap: List[List[float]] = []

    for layer_index in layer_indices:
        patch_iteration_results: List[Dict[str, object]] = []
        tracked_token_mask_probs: List[float] = []
        tracked_token_texts: List[str] = []
        decoded_texts: List[str] = []

        for patch_iteration in patch_iteration_list:
            patch_result = run_decoder_cross_attn_token_patching_experiment(
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

        layer_results.append(
            {
                "layer_index": layer_index,
                "tracked_token_position": tracked_token_position,
                "tracked_token_label": tracked_token_label,
                "patch_token_position": patch_token_position,
                "patch_token_label": patch_token_label,
                "tracked_token_mask_probs_by_patch_iteration": tracked_token_mask_probs,
                "tracked_token_texts_by_patch_iteration": tracked_token_texts,
                "decoded_texts_by_patch_iteration": decoded_texts,
                "patch_iteration_results": patch_iteration_results,
            }
        )
        heatmap.append(tracked_token_mask_probs)

    return {
        "source_sentence": source_sentence,
        "target_sentence": target_sentence,
        "patch_mode": "cross_attn_source_to_target",
        "tracked_token_position": tracked_token_position,
        "tracked_token_label": tracked_token_label,
        "patch_token_position": patch_token_position,
        "patch_token_label": patch_token_label,
        "reference_decoded_text": target_reference["decoded_text"],
        "reference_token_ids": target_reference["token_ids"],
        "layer_indices": layer_indices,
        "patch_iterations": patch_iteration_list,
        "heatmap": heatmap,
        "layer_results": layer_results,
    }


def run_decoder_cross_attn_full_layer_iteration_sweep_experiment(
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

    patch_iteration_list = _normalize_iteration_indices(
        patch_iterations,
        decoding_iterations,
        name="patch_iterations",
    )

    tracked_token_label = tgt_dict[target_reference["token_ids"][tracked_token_position]]
    layer_results: List[Dict[str, object]] = []
    heatmap: List[List[float]] = []

    for layer_index in layer_indices:
        patch_iteration_results: List[Dict[str, object]] = []
        tracked_token_mask_probs: List[float] = []
        tracked_token_texts: List[str] = []
        decoded_texts: List[str] = []

        for patch_iteration in patch_iteration_list:
            patch_result = run_decoder_cross_attn_full_layer_patching_experiment(
                source_sentence=source_sentence,
                target_sentence=target_sentence,
                context=context,
                layer_index=layer_index,
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

        layer_results.append(
            {
                "layer_index": layer_index,
                "tracked_token_position": tracked_token_position,
                "tracked_token_label": tracked_token_label,
                "tracked_token_mask_probs_by_patch_iteration": tracked_token_mask_probs,
                "tracked_token_texts_by_patch_iteration": tracked_token_texts,
                "decoded_texts_by_patch_iteration": decoded_texts,
                "patch_iteration_results": patch_iteration_results,
            }
        )
        heatmap.append(tracked_token_mask_probs)

    return {
        "source_sentence": source_sentence,
        "target_sentence": target_sentence,
        "patch_mode": "cross_attn_full_layer_source_to_target",
        "tracked_token_position": tracked_token_position,
        "tracked_token_label": tracked_token_label,
        "reference_decoded_text": target_reference["decoded_text"],
        "reference_token_ids": target_reference["token_ids"],
        "layer_indices": layer_indices,
        "patch_iterations": patch_iteration_list,
        "heatmap": heatmap,
        "layer_results": layer_results,
    }


def plot_token_mask_probs(
    decode_result: Dict[str, object],
    dictionary_path: Path = DEFAULT_MODEL_DIR / "dict.de.txt",
    figsize=(10, 5),
):
    import matplotlib.pyplot as plt

    from fairseq.data import Dictionary

    iteration_trace = decode_result["iteration_trace"]
    if not iteration_trace:
        raise ValueError("decode_result['iteration_trace'] is empty")

    dictionary = Dictionary.load(str(dictionary_path))
    iterations = [step["iteration"] for step in iteration_trace]
    final_token_ids = iteration_trace[-1]["token_ids"]
    token_labels = [f"pos {idx}: {dictionary[token_id]}" for idx, token_id in enumerate(final_token_ids)]

    plt.figure(figsize=figsize)
    for idx, label in enumerate(token_labels):
        probs = [step["token_mask_probs"][idx] for step in iteration_trace]
        plt.plot(iterations, probs, marker="o", linewidth=2, label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Mask Probability")
    plt.title("Token Mask Probabilities Across Decoding Iterations")
    plt.xticks(iterations)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.show()


def plot_layerwise_token_mask_heatmap(
    layer_sweep_result: Dict[str, object],
    figsize=(9, 4.5),
    cmap: str = "magma",
):

    heatmap = layer_sweep_result["heatmap"]
    if not heatmap:
        raise ValueError("layer_sweep_result['heatmap'] is empty")

    layer_indices = layer_sweep_result["layer_indices"]
    iterations = layer_sweep_result["iterations"]
    tracked_token_position = layer_sweep_result["tracked_token_position"]
    tracked_token_label = layer_sweep_result["tracked_token_label"]

    plt.figure(figsize=figsize)
    image = plt.imshow(heatmap, aspect="auto", cmap=cmap, origin="lower", vmin=np.percentile(heatmap, 1), vmax=np.percentile(heatmap, 99))
    plt.colorbar(image, label="Mask Probability")
    plt.xticks(range(len(iterations)), iterations)
    plt.yticks(range(len(layer_indices)), layer_indices)
    plt.xlabel("Iteration")
    plt.ylabel("Decoder Layer")
    plt.title(
        f"Mask Probability Heatmap for token pos {tracked_token_position}: {tracked_token_label}"
    )
    plt.tight_layout()
    plt.show()


def plot_self_attn_layer_iteration_heatmap(
    sweep_result: Dict[str, object],
    figsize=(9, 4.5),
    cmap: str = "magma",
):
    heatmap = sweep_result["heatmap"]
    if not heatmap:
        raise ValueError("sweep_result['heatmap'] is empty")

    layer_indices = sweep_result["layer_indices"]
    patch_iterations = sweep_result["patch_iterations"]
    tracked_token_position = sweep_result["tracked_token_position"]
    tracked_token_label = sweep_result["tracked_token_label"]
    patch_token_position = sweep_result["patch_token_position"]
    patch_token_label = sweep_result["patch_token_label"]

    plt.figure(figsize=figsize)
    image = plt.imshow(heatmap, aspect="auto", cmap=cmap, origin="lower", vmin=0, vmax=0.2)
    plt.colorbar(image, label="Mask Probability")
    plt.xticks(range(len(patch_iterations)), patch_iterations)
    plt.yticks(range(len(layer_indices)), layer_indices)
    plt.xlabel("Patched Decoding Iteration")
    plt.ylabel("Decoder Layer")
    plt.title(
        "Self-Attn Patch Heatmap for "
        f"tracked token pos {tracked_token_position}: {tracked_token_label} "
        f"(patch token pos {patch_token_position}: {patch_token_label})"
    )
    plt.tight_layout()
    plt.show()


def plot_cross_attn_layer_iteration_heatmap(
    sweep_result: Dict[str, object],
    figsize=(9, 4.5),
    cmap: str = "magma",
):
    heatmap = sweep_result["heatmap"]
    if not heatmap:
        raise ValueError("sweep_result['heatmap'] is empty")

    layer_indices = sweep_result["layer_indices"]
    patch_iterations = sweep_result["patch_iterations"]
    tracked_token_position = sweep_result["tracked_token_position"]
    tracked_token_label = sweep_result["tracked_token_label"]
    patch_token_position = sweep_result["patch_token_position"]
    patch_token_label = sweep_result["patch_token_label"]

    plt.figure(figsize=figsize)
    image = plt.imshow(heatmap, aspect="auto", cmap=cmap, origin="lower", vmin=0, vmax=0.2)
    plt.colorbar(image, label="Mask Probability")
    plt.xticks(range(len(patch_iterations)), patch_iterations)
    plt.yticks(range(len(layer_indices)), layer_indices)
    plt.xlabel("Patched Decoding Iteration")
    plt.ylabel("Decoder Layer")
    plt.title(
        "Cross-Attn Patch Heatmap for "
        f"tracked token pos {tracked_token_position}: {tracked_token_label} "
        f"(patch token pos {patch_token_position}: {patch_token_label})"
    )
    plt.tight_layout()
    plt.show()


def plot_self_attn_full_layer_iteration_heatmap(
    sweep_result: Dict[str, object],
    figsize=(9, 4.5),
    cmap: str = "magma",
):
    heatmap = sweep_result["heatmap"]
    if not heatmap:
        raise ValueError("sweep_result['heatmap'] is empty")

    layer_indices = sweep_result["layer_indices"]
    patch_iterations = sweep_result["patch_iterations"]
    tracked_token_position = sweep_result["tracked_token_position"]
    tracked_token_label = sweep_result["tracked_token_label"]

    plt.figure(figsize=figsize)
    image = plt.imshow(heatmap, aspect="auto", cmap=cmap, origin="lower", vmin=0, vmax=0.2)
    plt.colorbar(image, label="Remask Probability")
    plt.xticks(range(len(patch_iterations)), patch_iterations)
    plt.yticks(range(len(layer_indices)), layer_indices)
    plt.xlabel("Patched Decoding Iteration")
    plt.ylabel("Decoder Layer")
    plt.title(
        "Self-Attn Full-Layer Patch Heatmap for "
        f"tracked token pos {tracked_token_position}: {tracked_token_label}"
    )
    plt.tight_layout()
    plt.show()


def plot_cross_attn_full_layer_iteration_heatmap(
    sweep_result: Dict[str, object],
    figsize=(9, 4.5),
    cmap: str = "magma",
):
    heatmap = sweep_result["heatmap"]
    if not heatmap:
        raise ValueError("sweep_result['heatmap'] is empty")

    layer_indices = sweep_result["layer_indices"]
    patch_iterations = sweep_result["patch_iterations"]
    tracked_token_position = sweep_result["tracked_token_position"]
    tracked_token_label = sweep_result["tracked_token_label"]

    plt.figure(figsize=figsize)
    image = plt.imshow(heatmap, aspect="auto", cmap=cmap, origin="lower", vmin=0, vmax=0.2)
    plt.colorbar(image, label="Remask Probability")
    plt.xticks(range(len(patch_iterations)), patch_iterations)
    plt.yticks(range(len(layer_indices)), layer_indices)
    plt.xlabel("Patched Decoding Iteration")
    plt.ylabel("Decoder Layer")
    plt.title(
        "Cross-Attn Full-Layer Patch Heatmap for "
        f"tracked token pos {tracked_token_position}: {tracked_token_label}"
    )
    plt.tight_layout()
    plt.show()
