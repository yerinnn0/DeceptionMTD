"""Plot nominal and deceptive state occupancies as four simple heatmaps.

Place this file in the directory containing the result PKL files, adjust the
four file names below if needed, and run:

    python plot_occupancy_1x4.py
"""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np


PKL_FILES = {
    "Without Deception": "result_gridworld_5x5_1_lp.pkl",
    r"Diversionary ($\beta=0.07$)": "final_rev1_gridworld_5x5_1_div_opt_0.07.pkl",
    r"Targeted ($\beta=0.15$)": "final_rev1_gridworld_5x5_1_tar_opt_0.15.pkl",
    r"Equivocal ($\beta=0.05$)": "final_rev1_gridworld_5x5_1_equ_opt_0.05.pkl",
}

# -1 selects the last beta result stored in each consolidated optimization PKL.
RESULT_INDEX = -1
OUTPUT_FILE = "gridworld_occupancy_1x4.png"
GRID_SHAPE = (5, 5)


def to_numpy(value):
    """Convert either a NumPy array or a Torch tensor to a NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def load_result(path):
    """Load either a consolidated PKL or a single-beta intermediate PKL."""
    with path.open("rb") as file:
        data = pickle.load(file)

    if isinstance(data, dict) and "results" in data:
        results = data["results"]
        if not results:
            raise ValueError(f"No optimization results found in {path}")
        return results[RESULT_INDEX]
    return data


def state_occupancy(result, deceptive):
    occupancy_block = result["occupancy_measure"]
    key = "deceptive_occupancy_measure" if deceptive else "occupancy_measure"
    occupancy = to_numpy(occupancy_block[key])

    if occupancy.ndim == 1:
        n_states = GRID_SHAPE[0] * GRID_SHAPE[1]
        if occupancy.size % n_states != 0:
            raise ValueError(f"Unexpected occupancy shape: {occupancy.shape}")
        occupancy = occupancy.reshape(n_states, -1)

    # Convert state-action occupancy x(s,a) to state occupancy sum_a x(s,a).
    return occupancy.sum(axis=1).reshape(GRID_SHAPE)


def main():
    base_dir = Path(__file__).resolve().parent
    occupancies = []

    for index, (title, file_name) in enumerate(PKL_FILES.items()):
        path = base_dir / file_name
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Update PKL_FILES at the top of this file."
            )
        result = load_result(path)
        occupancies.append(state_occupancy(result, deceptive=index > 0))

    vmax = max(float(values.max()) for values in occupancies)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), constrained_layout=True)

    image = None
    for axis, (title, _), values in zip(axes, PKL_FILES.items(), occupancies):
        image = axis.imshow(values, cmap="viridis", vmin=0.0, vmax=vmax)
        axis.set_title(title)
        axis.set_xticks(range(GRID_SHAPE[1]))
        axis.set_yticks(range(GRID_SHAPE[0]))
        
        axis.text(0, 0, "S", ha="center", va="center", color="black", weight="bold")
        axis.text(
            GRID_SHAPE[1] - 1,
            GRID_SHAPE[0] - 1,
            "T",
            ha="center",
            va="center",
            color="white",
            weight="bold",
        )

    fig.colorbar(image, ax=axes, label="state occupancy", shrink=0.86)
    output_path = base_dir / OUTPUT_FILE
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
