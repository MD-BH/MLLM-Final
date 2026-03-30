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


def _render_attention_zero_out_heatmap(
    ax,
    heatmap,
    *,
    head_indices,
    iterations,
    title: str,
    ylabel: str,
    cmap: str = "magma",
    vmin=None,
    vmax=None,
):
    if not heatmap:
        raise ValueError("heatmap is empty")

    image = ax.imshow(
        heatmap,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        vmin=np.percentile(heatmap, 1) if vmin is None else vmin,
        vmax=np.percentile(heatmap, 99) if vmax is None else vmax,
    )
    ax.set_xticks(range(len(iterations)), iterations)
    ax.set_yticks(range(len(head_indices)), head_indices)
    ax.set_xlabel("Decoding Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return image


def _plot_attention_zero_out_heatmap(
    heatmap,
    *,
    head_indices,
    iterations,
    title: str,
    ylabel: str,
    figsize=(9, 4.5),
    cmap: str = "magma",
):
    fig, ax = plt.subplots(figsize=figsize)
    image = _render_attention_zero_out_heatmap(
        ax,
        heatmap,
        head_indices=head_indices,
        iterations=iterations,
        title=title,
        ylabel=ylabel,
        cmap=cmap,
    )
    fig.colorbar(image, ax=ax, label="Average Token Mask Probability")
    plt.tight_layout()
    plt.show()


def _plot_attention_zero_out_layer_sweep_grid(
    layer_sweep_result: Dict[str, object],
    *,
    attention_label: str,
    ylabel: str,
    figsize=(16, 9),
    cmap: str = "magma",
):
    average_layer_indices = layer_sweep_result["average_layer_indices"]
    if not average_layer_indices:
        raise ValueError("layer_sweep_result['average_layer_indices'] is empty")

    layer_result_map = {
        layer_result["layer_index"]: layer_result
        for layer_result in layer_sweep_result["layer_results"]
    }
    selected_layer_results = [
        layer_result_map[layer_index]
        for layer_index in average_layer_indices
    ]

    if len(selected_layer_results) > 5:
        raise ValueError("2x3 subplot layout supports at most 5 layer heatmaps plus 1 average heatmap")

    heatmaps = [layer_result["heatmap"] for layer_result in selected_layer_results]
    heatmaps.append(layer_sweep_result["average_heatmap"])
    flattened_values = np.concatenate([np.asarray(heatmap, dtype=float).ravel() for heatmap in heatmaps])
    vmin = np.percentile(flattened_values, 1)
    vmax = np.percentile(flattened_values, 99)
    if vmin == vmax:
        vmax = vmin + 1e-6

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.subplots_adjust(left=0.07, right=0.9, bottom=0.08, top=0.92, wspace=0.32, hspace=0.42)
    axes = axes.flatten()
    image = None

    for ax, layer_result in zip(axes, selected_layer_results):
        image = _render_attention_zero_out_heatmap(
            ax,
            layer_result["heatmap"],
            head_indices=layer_result["head_indices"],
            iterations=layer_result["iterations"],
            title=f"{attention_label} layer {layer_result['layer_index']}",
            ylabel=ylabel,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    average_ax = axes[len(selected_layer_results)]
    image = _render_attention_zero_out_heatmap(
        average_ax,
        layer_sweep_result["average_heatmap"],
        head_indices=layer_sweep_result["head_indices"],
        iterations=layer_sweep_result["iterations"],
        title=f"{attention_label} average {average_layer_indices}",
        ylabel=ylabel,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    for ax in axes[len(selected_layer_results) + 1:]:
        ax.axis("off")

    colorbar_ax = fig.add_axes([0.92, 0.14, 0.015, 0.68])
    fig.colorbar(image, cax=colorbar_ax, label="Average Token Mask Probability")
    plt.show()


def plot_self_attn_zero_out_heatmap(
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

    _plot_attention_zero_out_heatmap(
        heatmap,
        head_indices=head_indices,
        iterations=iterations,
        title=f"Self-Attn Zero-Out Heatmap for decoder layer {layer_index}",
        ylabel="Self-Attn Head",
        figsize=figsize,
        cmap=cmap,
    )


def plot_cross_attn_zero_out_heatmap(
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

    _plot_attention_zero_out_heatmap(
        heatmap,
        head_indices=head_indices,
        iterations=iterations,
        title=f"Cross-Attn Zero-Out Heatmap for decoder layer {layer_index}",
        ylabel="Cross-Attn Head",
        figsize=figsize,
        cmap=cmap,
    )


def plot_self_attn_zero_out_layer_sweep_heatmaps(
    layer_sweep_result: Dict[str, object],
    figsize=(16, 9),
    cmap: str = "magma",
):
    _plot_attention_zero_out_layer_sweep_grid(
        layer_sweep_result,
        attention_label="Self-Attn Zero-Out",
        ylabel="Self-Attn Head",
        figsize=figsize,
        cmap=cmap,
    )


def plot_cross_attn_zero_out_layer_sweep_heatmaps(
    layer_sweep_result: Dict[str, object],
    figsize=(16, 9),
    cmap: str = "magma",
):
    _plot_attention_zero_out_layer_sweep_grid(
        layer_sweep_result,
        attention_label="Cross-Attn Zero-Out",
        ylabel="Cross-Attn Head",
        figsize=figsize,
        cmap=cmap,
    )


def plot_self_attn_zero_out_average_heatmap(
    layer_sweep_result: Dict[str, object],
    figsize=(9, 4.5),
    cmap: str = "magma",
):
    heatmap = layer_sweep_result["average_heatmap"]
    if not heatmap:
        raise ValueError("layer_sweep_result['average_heatmap'] is empty")

    _plot_attention_zero_out_heatmap(
        heatmap,
        head_indices=layer_sweep_result["head_indices"],
        iterations=layer_sweep_result["iterations"],
        title=(
            "Self-Attn Zero-Out Heatmap Averaged Across "
            f"{len(layer_sweep_result['average_layer_indices'])} Decoder Layers "
            f"{layer_sweep_result['average_layer_indices']}"
        ),
        ylabel="Self-Attn Head",
        figsize=figsize,
        cmap=cmap,
    )


def plot_cross_attn_zero_out_average_heatmap(
    layer_sweep_result: Dict[str, object],
    figsize=(9, 4.5),
    cmap: str = "magma",
):
    heatmap = layer_sweep_result["average_heatmap"]
    if not heatmap:
        raise ValueError("layer_sweep_result['average_heatmap'] is empty")

    _plot_attention_zero_out_heatmap(
        heatmap,
        head_indices=layer_sweep_result["head_indices"],
        iterations=layer_sweep_result["iterations"],
        title=(
            "Cross-Attn Zero-Out Heatmap Averaged Across "
            f"{len(layer_sweep_result['average_layer_indices'])} Decoder Layers "
            f"{layer_sweep_result['average_layer_indices']}"
        ),
        ylabel="Cross-Attn Head",
        figsize=figsize,
        cmap=cmap,
    )
