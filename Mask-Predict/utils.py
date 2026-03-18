import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent
INPUT_DIR = REPO_ROOT / "input"
OUTPUT_DIR = REPO_ROOT / "output"
DEFAULT_MODEL_DIR = REPO_ROOT / "checkpoints" / "maskPredict_en_de"
DEFAULT_RUN_NAME = "en_de_demo"
PLACEHOLDER_TARGET = "Platzhalter ."


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


def _validate_model_files(model_dir: Path, source_lang: str, target_lang: str) -> None:
    required_files = [
        model_dir / "checkpoint_best.pt",
        model_dir / f"dict.{source_lang}.txt",
        model_dir / f"dict.{target_lang}.txt",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing model files:\n" + "\n".join(missing))


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
