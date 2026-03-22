import json
import io
import re
import shutil
import subprocess
import sys
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parent
INPUT_DIR = REPO_ROOT / "input"
OUTPUT_DIR = REPO_ROOT / "output"
DEFAULT_MODEL_DIR = REPO_ROOT / "checkpoints" / "maskPredict_en_de"
DEFAULT_RUN_NAME = "en_de_demo"
PLACEHOLDER_TARGET = "Platzhalter ."


@contextmanager
def _suppress_fairseq_output():
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FairseqModel is deprecated, please use FairseqEncoderDecoderModel or BaseFairseqModel instead",
            category=UserWarning,
        )
        yield


def _build_context_generation_args(
    data_bin_dir: Path,
    model_dir: Path,
    source_lang: str,
    target_lang: str,
    decoding_iterations: int,
    length_beam: int,
    max_sentences: int,
    use_cpu: bool,
):
    from fairseq import options

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
    from fairseq import checkpoint_utils, tasks

    data_bin_dir = Path(data_bin_dir)
    model_dir = Path(model_dir)
    args = _build_context_generation_args(
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
    from fairseq.strategies.strategy_utils import generate_step_with_prob

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


def preprocess_mask_predict_data(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    source_lang: str = "en",
    target_lang: str = "de",
    run_name: str = DEFAULT_RUN_NAME,
    workers: int = 1,
    clean: bool = True,
) -> Dict[str, object]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    model_dir = Path(model_dir)

    _validate_model_files(model_dir, source_lang, target_lang)

    run_dir = output_dir / run_name
    raw_dir = run_dir / "raw"
    data_bin_dir = run_dir / "data-bin"
    logs_dir = run_dir / "logs"

    if clean and run_dir.exists():
        shutil.rmtree(run_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    data_bin_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    split_prefixes = _stage_parallel_corpus(
        input_dir=input_dir,
        raw_dir=raw_dir,
        source_lang=source_lang,
        target_lang=target_lang,
    )

    if "test" not in split_prefixes:
        raise FileNotFoundError(
            f"Missing {input_dir / f'test.{source_lang}'}; at least the test split is required."
        )

    command = [
        sys.executable,
        "preprocess.py",
        "--source-lang",
        source_lang,
        "--target-lang",
        target_lang,
        "--destdir",
        str(data_bin_dir),
        "--workers",
        str(workers),
        "--srcdict",
        str(model_dir / f"dict.{source_lang}.txt"),
        "--tgtdict",
        str(model_dir / f"dict.{target_lang}.txt"),
    ]

    for split in ("train", "valid", "test"):
        if split in split_prefixes:
            command.extend([f"--{split}pref", str(split_prefixes[split])])

    completed = _run_command(command)
    preprocess_log = logs_dir / "preprocess.log"
    _write_log(preprocess_log, completed)

    result = {
        "run_dir": run_dir,
        "raw_dir": raw_dir,
        "data_bin_dir": data_bin_dir,
        "logs_dir": logs_dir,
        "preprocess_log": preprocess_log,
        "splits": sorted(split_prefixes.keys()),
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }

    _write_json(run_dir / "preprocess_summary.json", result, extra_serializable={"command": command})
    return result


def run_mask_predict_inference(
    data_bin_dir: Path,
    output_dir: Path = OUTPUT_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    source_lang: str = "en",
    target_lang: str = "de",
    subset: str = "test",
    run_name: str = DEFAULT_RUN_NAME,
    decoding_iterations: int = 10,
    length_beam: int = 5,
    max_sentences: int = 20,
    use_cpu: bool = False,
) -> Dict[str, object]:
    data_bin_dir = Path(data_bin_dir)
    output_dir = Path(output_dir)
    model_dir = Path(model_dir)

    checkpoint_path = model_dir / "checkpoint_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    run_dir = output_dir / run_name
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "generate_cmlm.py",
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
        subset,
    ]
    if use_cpu:
        command.append("--cpu")

    completed = _run_command(command)
    inference_log = logs_dir / f"generate_{subset}.log"
    _write_log(inference_log, completed)

    parsed = _parse_generate_output(completed.stdout)
    translations_path = run_dir / f"{subset}.{target_lang}.hyp"
    translations_path.write_text(
        "\n".join(record["hypothesis"] for record in parsed["records"]) + "\n",
        encoding="utf-8",
    )

    result = {
        "run_dir": run_dir,
        "data_bin_dir": data_bin_dir,
        "logs_dir": logs_dir,
        "inference_log": inference_log,
        "translations_path": translations_path,
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "records": parsed["records"],
        "bleu": parsed["bleu"],
    }

    _write_json(run_dir / f"{subset}_results.json", result, extra_serializable={"command": command})
    return result


def trace_mask_predict_iterations(
    data_bin_dir: Path,
    output_dir: Path = OUTPUT_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    source_lang: str = "en",
    target_lang: str = "de",
    subset: str = "test",
    run_name: str = DEFAULT_RUN_NAME,
    decoding_iterations: int = 10,
    length_beam: int = 5,
    max_sentences: int = 20,
    use_cpu: bool = False,
) -> Dict[str, object]:
    from fairseq import progress_bar, tasks
    from fairseq import checkpoint_utils
    from fairseq import utils as fairseq_utils
    from fairseq.strategies.strategy_utils import (
        assign_multi_value_long,
        assign_single_value_byte,
        assign_single_value_long,
        duplicate_encoder_out,
        generate_step_with_prob,
    )
    from fairseq import pybleu

    data_bin_dir = Path(data_bin_dir)
    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    run_dir = output_dir / run_name
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    args, command = _build_generation_args(
        data_bin_dir=data_bin_dir,
        model_dir=model_dir,
        source_lang=source_lang,
        target_lang=target_lang,
        subset=subset,
        decoding_iterations=decoding_iterations,
        length_beam=length_beam,
        max_sentences=max_sentences,
        use_cpu=use_cpu,
    )
    use_cuda = torch.cuda.is_available() and not args.cpu
    torch.manual_seed(args.seed)

    with _suppress_output(), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FairseqModel is deprecated, please use FairseqEncoderDecoderModel or BaseFairseqModel instead",
            category=UserWarning,
        )
        task = tasks.setup_task(args)
        task.load_dataset(args.gen_subset)
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    with _suppress_output(), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FairseqModel is deprecated, please use FairseqEncoderDecoderModel or BaseFairseqModel instead",
            category=UserWarning,
        )
        models, _ = checkpoint_utils.load_model_ensemble(
            args.path.split(":"),
            arg_overrides=eval(args.model_overrides),
            task=task,
        )
    if use_cuda:
        models = [model.cuda() for model in models]

    with _suppress_output():
        for model in models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
                need_attn=args.print_alignment,
            )
            if args.fp16:
                model.half()

    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=fairseq_utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    ).next_epoch_itr(shuffle=False)

    records = []
    scorer = pybleu.PyBleuScorer()
    with progress_bar.build_progress_bar(args, itr) as progress:
        for sample in progress:
            batch_records = _trace_sample_iterations(
                sample=sample,
                models=models,
                src_dict=src_dict,
                tgt_dict=tgt_dict,
                args=args,
                use_cuda=use_cuda,
                fairseq_utils_module=fairseq_utils,
                assign_multi_value_long=assign_multi_value_long,
                assign_single_value_byte=assign_single_value_byte,
                assign_single_value_long=assign_single_value_long,
                duplicate_encoder_out=duplicate_encoder_out,
                generate_step_with_prob=generate_step_with_prob,
            )
            records.extend(batch_records)

    trace_path = run_dir / f"{subset}_iteration_trace.json"
    result = {
        "run_dir": run_dir,
        "data_bin_dir": data_bin_dir,
        "logs_dir": logs_dir,
        "records": records,
        "bleu": scorer.score([r["target"] for r in records], [r["hypothesis"] for r in records]) if records else None,
        "command": command,
        "trace_path": trace_path,
    }

    _write_json(trace_path, result, extra_serializable={"command": command})
    return result


def _stage_parallel_corpus(
    input_dir: Path,
    raw_dir: Path,
    source_lang: str,
    target_lang: str,
) -> Dict[str, Path]:
    split_prefixes: Dict[str, Path] = {}

    for split in ("train", "valid", "test"):
        source_file = input_dir / f"{split}.{source_lang}"
        if not source_file.exists():
            continue

        target_file = input_dir / f"{split}.{target_lang}"
        staged_source_file = raw_dir / source_file.name
        staged_target_file = raw_dir / target_file.name

        shutil.copyfile(source_file, staged_source_file)

        if target_file.exists():
            shutil.copyfile(target_file, staged_target_file)
        else:
            source_lines = _read_lines(source_file)
            placeholder_lines = [PLACEHOLDER_TARGET for _ in source_lines]
            if not placeholder_lines:
                placeholder_lines = [PLACEHOLDER_TARGET]
            staged_target_file.write_text("\n".join(placeholder_lines) + "\n", encoding="utf-8")

        split_prefixes[split] = raw_dir / split

    return split_prefixes


class _suppress_output:
    def __enter__(self):
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.stdout_cm = _redirect_stream("stdout", self.stdout)
        self.stderr_cm = _redirect_stream("stderr", self.stderr)
        self.stdout_cm.__enter__()
        self.stderr_cm.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stderr_cm.__exit__(exc_type, exc, tb)
        self.stdout_cm.__exit__(exc_type, exc, tb)
        return False


class _redirect_stream:
    def __init__(self, stream_name, target):
        self.stream_name = stream_name
        self.target = target
        self.original = None

    def __enter__(self):
        self.original = getattr(sys, self.stream_name)
        setattr(sys, self.stream_name, self.target)
        return self.target

    def __exit__(self, exc_type, exc, tb):
        setattr(sys, self.stream_name, self.original)
        return False


def _build_generation_args(
    data_bin_dir: Path,
    model_dir: Path,
    source_lang: str,
    target_lang: str,
    subset: str,
    decoding_iterations: int,
    length_beam: int,
    max_sentences: int,
    use_cpu: bool,
):
    from fairseq import options

    checkpoint_path = model_dir / "checkpoint_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

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
        subset,
    ]
    if use_cpu:
        cli_args.append("--cpu")

    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser, input_args=cli_args)
    return args, [sys.executable, "generate_cmlm.py", *cli_args]


def _validate_model_files(model_dir: Path, source_lang: str, target_lang: str) -> None:
    required_files = [
        model_dir / "checkpoint_best.pt",
        model_dir / f"dict.{source_lang}.txt",
        model_dir / f"dict.{target_lang}.txt",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing model files:\n" + "\n".join(missing))


def _trace_sample_iterations(
    sample,
    models,
    src_dict,
    tgt_dict,
    args,
    use_cuda,
    fairseq_utils_module,
    assign_multi_value_long,
    assign_single_value_byte,
    assign_single_value_long,
    duplicate_encoder_out,
    generate_step_with_prob,
) -> List[Dict[str, object]]:
    traced_records = []
    moved_sample = fairseq_utils_module.move_to_cuda(sample) if use_cuda else sample
    if "net_input" not in moved_sample:
        return traced_records

    net_input = moved_sample["net_input"]
    encoder_input = {key: value for key, value in net_input.items() if key != "prev_output_tokens"}

    with torch.no_grad():
        hypotheses, iteration_traces = _generate_with_iteration_trace(
            encoder_input=encoder_input,
            models=models,
            tgt_dict=tgt_dict,
            length_beam_size=args.length_beam,
            gold_target_len=None,
            decoding_iterations=args.decoding_iterations,
            remove_bpe=args.remove_bpe,
            assign_multi_value_long=assign_multi_value_long,
            assign_single_value_byte=assign_single_value_byte,
            assign_single_value_long=assign_single_value_long,
            duplicate_encoder_out=duplicate_encoder_out,
            generate_step_with_prob=generate_step_with_prob,
        )

    for batch_idx in range(hypotheses.size(0)):
        sample_id = int(moved_sample["id"][batch_idx].item())
        src_tokens = fairseq_utils_module.strip_pad(net_input["src_tokens"][batch_idx].data, tgt_dict.pad())
        target_tokens = fairseq_utils_module.strip_pad(moved_sample["target"][batch_idx].data, tgt_dict.pad())
        hypothesis_tokens = fairseq_utils_module.strip_pad(hypotheses[batch_idx], tgt_dict.pad())

        record = {
            "id": sample_id,
            "source": src_dict.string(src_tokens.int().cpu(), args.remove_bpe),
            "target": tgt_dict.string(target_tokens.int().cpu(), args.remove_bpe, escape_unk=True),
            "hypothesis": tgt_dict.string(hypothesis_tokens.int().cpu(), args.remove_bpe, escape_unk=True),
            "iterations": iteration_traces[batch_idx],
        }
        traced_records.append(record)

    return traced_records


def _generate_with_iteration_trace(
    encoder_input,
    models,
    tgt_dict,
    length_beam_size,
    gold_target_len,
    decoding_iterations,
    remove_bpe,
    assign_multi_value_long,
    assign_single_value_byte,
    assign_single_value_long,
    duplicate_encoder_out,
    generate_step_with_prob,
):
    assert len(models) == 1
    model = models[0]

    src_tokens = encoder_input["src_tokens"]
    src_tokens = src_tokens.new(src_tokens.tolist())
    batch_size = src_tokens.size(0)

    encoder_out = model.encoder(**encoder_input)
    beam = _predict_length_beam(gold_target_len, encoder_out["predicted_lengths"], length_beam_size)

    max_len = beam.max().item()
    length_mask = torch.triu(src_tokens.new(max_len, max_len).fill_(1).long(), 1)
    length_mask = torch.stack([length_mask[beam[batch] - 1] for batch in range(batch_size)], dim=0)
    tgt_tokens = src_tokens.new(batch_size, length_beam_size, max_len).fill_(tgt_dict.mask())
    tgt_tokens = (1 - length_mask) * tgt_tokens + length_mask * tgt_dict.pad()
    tgt_tokens = tgt_tokens.view(batch_size * length_beam_size, max_len)

    duplicate_encoder_out(encoder_out, batch_size, length_beam_size)
    hypotheses, lprobs, traces = _trace_mask_predict_generate(
        model=model,
        encoder_out=encoder_out,
        tgt_tokens=tgt_tokens,
        tgt_dict=tgt_dict,
        decoding_iterations=decoding_iterations,
        assign_multi_value_long=assign_multi_value_long,
        assign_single_value_byte=assign_single_value_byte,
        assign_single_value_long=assign_single_value_long,
        generate_step_with_prob=generate_step_with_prob,
    )

    hypotheses = hypotheses.view(batch_size, length_beam_size, max_len)
    lprobs = lprobs.view(batch_size, length_beam_size)
    tgt_lengths = (1 - length_mask).sum(-1)
    avg_log_prob = lprobs / tgt_lengths.float()
    best_lengths = avg_log_prob.max(-1)[1]
    selected_hypotheses = torch.stack([hypotheses[b, beam_idx, :] for b, beam_idx in enumerate(best_lengths)], dim=0)

    selected_traces: List[List[Dict[str, object]]] = []
    for batch_idx, beam_idx in enumerate(best_lengths.tolist()):
        sentence_trace = []
        for step in traces:
            step_tokens = step["tokens"].view(batch_size, length_beam_size, max_len)[batch_idx, beam_idx]
            step_tokens = _strip_tensor_pad(step_tokens, tgt_dict.pad())
            masked_tokens = step["masked_tokens"].view(batch_size, length_beam_size)[batch_idx, beam_idx].item()
            sentence_trace.append(
                {
                    "iteration": step["iteration"],
                    "masked_tokens": int(masked_tokens),
                    "text": tgt_dict.string(step_tokens.int().cpu(), remove_bpe, escape_unk=True),
                }
            )
        selected_traces.append(sentence_trace)

    return selected_hypotheses, selected_traces


def _trace_mask_predict_generate(
    model,
    encoder_out,
    tgt_tokens,
    tgt_dict,
    decoding_iterations,
    assign_multi_value_long,
    assign_single_value_byte,
    assign_single_value_long,
    generate_step_with_prob,
):
    flat_batch_size, seq_len = tgt_tokens.size()
    pad_mask = tgt_tokens.eq(tgt_dict.pad())
    seq_lens = seq_len - pad_mask.sum(dim=1)
    iterations = seq_len if decoding_iterations is None else decoding_iterations

    trace_steps = []

    decoder_out = model.decoder(tgt_tokens, encoder_out)
    tgt_tokens, token_probs, _ = generate_step_with_prob(decoder_out)
    assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
    assign_single_value_byte(token_probs, pad_mask, 1.0)
    trace_steps.append(
        {
            "iteration": 0,
            "masked_tokens": torch.zeros(flat_batch_size, dtype=torch.long),
            "tokens": tgt_tokens.clone().cpu(),
        }
    )

    for counter in range(1, iterations):
        num_mask = (seq_lens.float() * (1.0 - (counter / iterations))).long()
        assign_single_value_byte(token_probs, pad_mask, 1.0)

        mask_ind = _select_worst(token_probs, num_mask)
        assign_single_value_long(tgt_tokens, mask_ind, tgt_dict.mask())
        assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())

        decoder_out = model.decoder(tgt_tokens, encoder_out)
        new_tgt_tokens, new_token_probs, _ = generate_step_with_prob(decoder_out)

        assign_multi_value_long(token_probs, mask_ind, new_token_probs)
        assign_single_value_byte(token_probs, pad_mask, 1.0)

        assign_multi_value_long(tgt_tokens, mask_ind, new_tgt_tokens)
        assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())

        trace_steps.append(
            {
                "iteration": counter,
                "masked_tokens": num_mask.view(-1).int().cpu(),
                "tokens": tgt_tokens.clone().cpu(),
            }
        )

    lprobs = token_probs.log().sum(-1)
    return tgt_tokens, lprobs, trace_steps


def _predict_length_beam(gold_target_len, predicted_lengths, length_beam_size):
    if gold_target_len is not None:
        beam_starts = gold_target_len - (length_beam_size - 1) // 2
        beam_ends = gold_target_len + length_beam_size // 2 + 1
        beam = torch.stack(
            [
                torch.arange(beam_starts[batch], beam_ends[batch], device=beam_starts.device)
                for batch in range(gold_target_len.size(0))
            ],
            dim=0,
        )
    else:
        beam = predicted_lengths.topk(length_beam_size, dim=1)[1]
    beam[beam < 2] = 2
    return beam


def _select_worst(token_probs, num_mask):
    batch_size, seq_len = token_probs.size()
    masks = [
        token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1]
        for batch in range(batch_size)
    ]
    masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
    return torch.stack(masks, dim=0)


def _strip_tensor_pad(tensor, pad_idx):
    return tensor[tensor.ne(pad_idx)]


def _run_command(command: List[str]) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}:\n"
            f"{' '.join(command)}\n\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
        )
    return completed


def _write_log(log_path: Path, completed: subprocess.CompletedProcess) -> None:
    log_text = []
    log_text.append("COMMAND:")
    log_text.append(" ".join(completed.args))
    log_text.append("")
    log_text.append("STDOUT:")
    log_text.append(completed.stdout.rstrip())
    log_text.append("")
    log_text.append("STDERR:")
    log_text.append(completed.stderr.rstrip())
    log_path.write_text("\n".join(log_text).rstrip() + "\n", encoding="utf-8")


def _parse_generate_output(stdout: str) -> Dict[str, object]:
    sources: Dict[int, str] = {}
    targets: Dict[int, str] = {}
    hypotheses: Dict[int, str] = {}
    bleu: Optional[float] = None

    for raw_line in stdout.splitlines():
        if raw_line.startswith("S-"):
            sample_id, text = _parse_prefixed_line(raw_line)
            sources[sample_id] = text
        elif raw_line.startswith("T-"):
            sample_id, text = _parse_prefixed_line(raw_line)
            targets[sample_id] = text
        elif raw_line.startswith("H-"):
            sample_id, text = _parse_prefixed_line(raw_line)
            hypotheses[sample_id] = text
        elif "BLEU4 =" in raw_line:
            match = re.search(r"BLEU4 = ([0-9.]+)", raw_line)
            if match:
                bleu = float(match.group(1))

    records = []
    for sample_id in sorted(hypotheses):
        records.append(
            {
                "id": sample_id,
                "source": sources.get(sample_id, ""),
                "target": targets.get(sample_id, ""),
                "hypothesis": hypotheses[sample_id],
            }
        )

    return {"records": records, "bleu": bleu}


def _parse_prefixed_line(line: str) -> Tuple[int, str]:
    prefix, text = line.split("\t", 1)
    return int(prefix.split("-", 1)[1]), text


def _read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _write_json(path: Path, payload: Dict[str, object], extra_serializable: Optional[Dict[str, object]] = None) -> None:
    serializable = {}
    for key, value in payload.items():
        if isinstance(value, Path):
            serializable[key] = str(value)
        elif isinstance(value, list):
            serializable[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            serializable[key] = value

    if extra_serializable:
        serializable.update(extra_serializable)

    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
