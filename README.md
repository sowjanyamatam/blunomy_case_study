# blunomy-lidar-catenary

A Python package for processing LiDAR point clouds of cable wire lines. It automatically identifies individual wires from 3D point clouds and fits catenary curve models to each one.

---

## What it does

1. Loads LiDAR data from a Parquet file
2. Separates points into individual wire clusters using DBSCAN
3. Fits a **catenary curve** to each wire:

```
y(x) = y0 + c * (cosh((x - x0) / c) - 1)
```

where `c` is the curvature parameter, `x0` is the horizontal position of the lowest point, and `y0` is the minimum elevation.

4. Outputs fitted parameters as a JSON model file, with optional visualization plots

---

## Project Architecture — ETL Pipeline

**Extract → Transform → Load/Output** (`main.py`)

**Extract**
- Loads LiDAR data from Parquet files
- Implemented in `loader.py`

**Transform**

- **(A) Clustering** (`cluster.py`): Applies PCA for dimensionality reduction, then uses DBSCAN to group points belonging to individual wires. Each cluster represents one wire segment.
- **(B) 3D → 2D Projection** (`cluster.py`): PCA finds the best-fit plane for each cluster and projects points into 2D space for curve fitting.
- **(C) Curve Fitting** (`pca_curve_fitter.py`): Fits a catenary curve to each projected cluster.

**Load / Output**
- Clustered dataset saved as CSV — config-driven
- Fitted catenary parameters saved as JSON in `models/` — config-driven

---

## Project Structure

```
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── data/
│   ├── clustered_files/
│   └── files_input/         
├── local_analysis/
│   └── analysis.ipynb
├── models/
├── tests/
│   └── test_lidar_catenary.py
└── src/
    └── lidar_catenary/
        ├── __init__.py
        ├── main.py
        ├── loader.py
        ├── cluster.py
        ├── pca_curve_fitter.py
        ├── config_loader.py
        └── config/
            └── config.yml    # tunable parameters
```

---

## Installation

**Requirements:** Python 3.10+

### From GitHub

```bash
python3 -m pip install --force-reinstall git+https://github.com/sowjanyamatam/blunomy_case_study.git
```

### For local development

```bash
git clone https://github.com/sowjanyamatam/blunomy_case_study.git
cd blunomy_case_study
pip install -e .
```

---

## Usage

### From the CLI

```bash
python3 src/lidar_catenary/main.py --dataset "data/files_input/lidar_cable_points_easy.parquet"
```

Optional flags:

```bash
--output-dir path/to/output   # default: ./lidar_output
--config path/to/config.yml   # override default config values
```

### As a Python package

```python
from lidar_catenary import Orchestrator

config_module.config = None
get_config("your-config-file-path.yml")

result = Orchestrator("your-datset-file-path.parquet").run_workflow()

# access the full model
model = result["catenary_model"]

# iterate over fitted wires
for wire in model["wires"]:
    print(wire["wire_id"], wire["x0"], wire["y0"], wire["c"])

# check summary
summary = model["summary"]
print(f"{summary['wires_fitted']} wires fitted")
```

---

## Configuration

All parameters live in `src/lidar_catenary/config/config.yml`. No code changes needed to tune the algorithm.

| Key | Default | Description |
|-----|---------|-------------|
| `clustering.epsilon_value` | `0.5` | DBSCAN `eps` — max distance between points in the same cluster. Decrease for noisier data. |
| `clustering.min_samples` | `5` | DBSCAN minimum points to form a cluster. Increase to filter out small noise clusters. |
| `min_points_for_clustering` | `10` | Minimum total points required to attempt clustering. Files below this threshold are rejected. |
| `output.save_images` | `false` | Save cluster and catenary curve plots as PNG. |
| `output.save_clustered_csv` | `false` | Save the labelled point cloud as CSV after clustering. |
| `output.save_model_json` | `false` | Save fitted catenary parameters as JSON. This is the primary output. |
| `logging.level` | `INFO` | Logging verbosity. Set to `DEBUG` for detailed per-step output. |

### Overriding config at runtime

You can pass a partial config YAML file to override specific values without editing the default config:

```bash
python3 src/lidar_catenary/main.py --dataset "data/files_input/lidar_cable_points_easy.parquet" --config my_override.yml
```

Your override file only needs to contain the keys you want to change:

```yaml
# my_override.yml
clustering:
  epsilon_value: 0.2
  min_samples: 10
```

All other values fall back to the defaults in `config.yml`. This is useful for testing different clustering parameters without touching the base config.

---

## Output format

The primary output is returned directly to the caller and optionally saved as a JSON file to `models/<dataset_name>/` (controlled by `save_model_json` in config):

```json
{
    "File_name": "lidar_cable_points_easy.parquet",
    "Row_count": 1502,
    "Timestamp": "20260512211736",
    "clustering_parameters": {
        "epsilon_value": 0.5,
        "min_samples": 5
    },
    "summary": {
        "number_of_wires": 3,
        "wires_fitted": 3,
        "wires_failed": []
    },
    "wires": [
        {"wire_id": "wire_0", "x0": 0.857, "y0": 10.001, "c": 199.70},
        {"wire_id": "wire_1", "x0": -0.356, "y0": 10.001, "c": 202.45},
        {"wire_id": "wire_2", "x0": 0.172, "y0": 9.997, "c": 201.16}
    ]
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover: catenary equation correctness, curve fitting output structure, noise label handling, data validation rules, and clustering output shape.
