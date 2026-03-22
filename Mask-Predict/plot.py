import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict

from utils import REPO_ROOT

mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["savefig.dpi"] = 300

DEFAULT_MODEL_DIR = REPO_ROOT / "checkpoints" / "maskPredict_en_de"


def plot_token_mask_probs(
    decode_result: Dict[str, object],
    dictionary_path: Path = DEFAULT_MODEL_DIR / "dict.de.txt",
    figsize=(10, 5),
):
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


def plot_self_attn_head_zero_ablation_heatmap(
    sweep_result: Dict[str, object],
    figsize=(9, 4.5),
    cmap: str = "magma",
):
    heatmap = sweep_result["heatmap"]
    if not heatmap:
        raise ValueError("sweep_result['heatmap'] is empty")

    head_indices = sweep_result["head_indices"]
    iterations = sweep_result["iterations"]
    layer_index = sweep_result["layer_index"]

    plt.figure(figsize=figsize)
    image = plt.imshow(heatmap, aspect="auto", cmap=cmap, origin="lower", vmin=np.percentile(heatmap, 1), vmax=np.percentile(heatmap, 99))
    plt.colorbar(image, label="Average Token Mask Probability")
    plt.xticks(range(len(iterations)), iterations)
    plt.yticks(range(len(head_indices)), head_indices)
    plt.xlabel("Decoding Iteration")
    plt.ylabel("Self-Attn Head")
    plt.title(
        f"Self-Attn Head Zero Ablation Heatmap for decoder layer {layer_index}"
    )
    plt.tight_layout()
    plt.show()


def plot_cross_attn_head_zero_ablation_heatmap(
    sweep_result: Dict[str, object],
    figsize=(9, 4.5),
    cmap: str = "magma",
):
    heatmap = sweep_result["heatmap"]
    if not heatmap:
        raise ValueError("sweep_result['heatmap'] is empty")

    head_indices = sweep_result["head_indices"]
    iterations = sweep_result["iterations"]
    layer_index = sweep_result["layer_index"]

    plt.figure(figsize=figsize)
    image = plt.imshow(heatmap, aspect="auto", cmap=cmap, origin="lower", vmin=np.percentile(heatmap, 1), vmax=np.percentile(heatmap, 99))
    plt.colorbar(image, label="Average Token Mask Probability")
    plt.xticks(range(len(iterations)), iterations)
    plt.yticks(range(len(head_indices)), head_indices)
    plt.xlabel("Decoding Iteration")
    plt.ylabel("Cross-Attn Head")
    plt.title(
        f"Cross-Attn Head Zero Ablation Heatmap for decoder layer {layer_index}"
    )
    plt.tight_layout()
    plt.show()
