import io
import warnings

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional

import torch

from fairseq import checkpoint_utils, options, tasks
from fairseq.strategies.strategy_utils import generate_step_with_prob

from utils import REPO_ROOT


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


def load_mask_predict_context(
    data_bin_dir: Path,
    model_dir: Path = DEFAULT_MODEL_DIR,
    source_lang: str = "en",
    target_lang: str = "de",
    decoding_iterations: int = 5,
    length_beam: int = 5,
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
    token_position: int,
    donor_hidden_state: Optional[torch.Tensor] = None,
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
        if donor_hidden_state is not None:
            patched_hidden = hidden.clone()
            patched_hidden[token_position, 0, :] = donor_hidden_state.to(hidden.device, dtype=hidden.dtype)
        captured["after_patch"] = patched_hidden[token_position, 0, :].detach().cpu().clone()
        return patched_hidden, attn

    handle = layer.register_forward_hook(hook)
    try:
        decoder_out = model.decoder(tgt_tokens, encoder_out)
    finally:
        handle.remove()

    return decoder_out, captured


def run_decoder_patching_experiment(
    donor_sentence: str,
    base_sentence: str,
    context: Dict[str, object],
    layer_index: int,
    token_position: int,
    decoding_iterations: int = 5,
) -> Dict[str, object]:
    task = context["task"]
    model = context["model"]
    device = context["device"]
    args = context["args"]
    tgt_dict = task.target_dictionary

    donor_encoder = get_encoder_output(donor_sentence, context)
    base_encoder = get_encoder_output(base_sentence, context)
    donor_encoder_out = _move_encoder_out_to_device(clone_encoder_out(donor_encoder["encoder_out"]), device)
    base_encoder_out = _move_encoder_out_to_device(clone_encoder_out(base_encoder["encoder_out"]), device)

    donor_length = _predicted_length(donor_encoder_out)
    base_length = _predicted_length(base_encoder_out)
    if token_position >= donor_length or token_position >= base_length:
        raise IndexError(
            f"token_position {token_position} out of range for donor/base decoder lengths {donor_length} and {base_length}"
        )

    iterations = base_length if decoding_iterations is None else decoding_iterations
    donor_tgt_tokens = torch.full((1, donor_length), tgt_dict.mask(), dtype=torch.long, device=device)
    base_tgt_tokens = torch.full((1, base_length), tgt_dict.mask(), dtype=torch.long, device=device)

    iteration_trace: List[Dict[str, object]] = []
    patch_trace: List[Dict[str, object]] = []

    with torch.no_grad():
        donor_decoder_out, donor_capture = _run_decoder_forward(
            model,
            donor_tgt_tokens,
            donor_encoder_out,
            layer_index,
            token_position,
        )
        donor_tgt_tokens, donor_token_probs, _ = generate_step_with_prob(donor_decoder_out)

        patched_decoder_out, base_capture = _run_decoder_forward(
            model,
            base_tgt_tokens,
            base_encoder_out,
            layer_index,
            token_position,
            donor_hidden_state=donor_capture["after_patch"],
        )
        base_tgt_tokens, base_token_probs, _ = generate_step_with_prob(patched_decoder_out)

        _record_iteration(iteration_trace, 0, base_tgt_tokens, base_token_probs, tgt_dict, args.remove_bpe)
        patch_trace.append(
            {
                "iteration": 0,
                "donor_text": _stringify_tokens(donor_tgt_tokens[0].detach().cpu(), tgt_dict, args.remove_bpe),
                "patched_text": _stringify_tokens(base_tgt_tokens[0].detach().cpu(), tgt_dict, args.remove_bpe),
                "donor_activation_norm": round(float(donor_capture["after_patch"].norm().item()), 6),
                "base_activation_norm_before_patch": round(float(base_capture["before_patch"].norm().item()), 6),
                "base_activation_norm_after_patch": round(float(base_capture["after_patch"].norm().item()), 6),
            }
        )

        for counter in range(1, iterations):
            donor_masked_tokens, donor_mask_ind, _, _ = _mask_tokens_for_iteration(
                donor_tgt_tokens,
                donor_token_probs,
                tgt_dict,
                counter,
                iterations,
            )
            donor_decoder_out, donor_capture = _run_decoder_forward(
                model,
                donor_masked_tokens,
                donor_encoder_out,
                layer_index,
                token_position,
            )
            donor_new_tgt_tokens, donor_new_token_probs, _ = generate_step_with_prob(donor_decoder_out)
            donor_tgt_tokens = donor_masked_tokens
            donor_token_probs = donor_token_probs.clone()
            donor_tgt_tokens[0, donor_mask_ind] = donor_new_tgt_tokens[0, donor_mask_ind]
            donor_token_probs[0, donor_mask_ind] = donor_new_token_probs[0, donor_mask_ind]

            base_masked_tokens, base_mask_ind, selected_mask_token_ids, selected_mask_scores = _mask_tokens_for_iteration(
                base_tgt_tokens,
                base_token_probs,
                tgt_dict,
                counter,
                iterations,
            )
            patched_decoder_out, base_capture = _run_decoder_forward(
                model,
                base_masked_tokens,
                base_encoder_out,
                layer_index,
                token_position,
                donor_hidden_state=donor_capture["after_patch"],
            )
            base_new_tgt_tokens, base_new_token_probs, _ = generate_step_with_prob(patched_decoder_out)
            base_tgt_tokens = base_masked_tokens
            base_token_probs = base_token_probs.clone()
            base_tgt_tokens[0, base_mask_ind] = base_new_tgt_tokens[0, base_mask_ind]
            base_token_probs[0, base_mask_ind] = base_new_token_probs[0, base_mask_ind]

            _record_iteration(
                iteration_trace,
                counter,
                base_tgt_tokens,
                base_token_probs,
                tgt_dict,
                args.remove_bpe,
                selected_mask_token_ids=selected_mask_token_ids,
                selected_mask_scores=selected_mask_scores,
            )
            patch_trace.append(
                {
                    "iteration": int(counter),
                    "donor_text": _stringify_tokens(donor_tgt_tokens[0].detach().cpu(), tgt_dict, args.remove_bpe),
                    "patched_text": _stringify_tokens(base_tgt_tokens[0].detach().cpu(), tgt_dict, args.remove_bpe),
                    "donor_activation_norm": round(float(donor_capture["after_patch"].norm().item()), 6),
                    "base_activation_norm_before_patch": round(float(base_capture["before_patch"].norm().item()), 6),
                    "base_activation_norm_after_patch": round(float(base_capture["after_patch"].norm().item()), 6),
                }
            )

    final_token_ids = base_tgt_tokens[0].detach().cpu()
    return {
        "donor_sentence": donor_sentence,
        "base_sentence": base_sentence,
        "layer_index": layer_index,
        "token_position": token_position,
        "donor_encoder": donor_encoder,
        "base_encoder": base_encoder,
        "decoded_text": _stringify_tokens(final_token_ids, tgt_dict, args.remove_bpe),
        "token_ids": final_token_ids.tolist(),
        "iteration_trace": iteration_trace,
        "patch_trace": patch_trace,
        "predicted_length": base_length,
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
