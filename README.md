# blunomy-lidar-catenary

A python package for processing Lidar point cloud of cable wire lines. It automatically identifies individual wires from 3D point clouds and fits catenary curve models to each one

---

# 3D Catenary Model Generation from LiDAR Data

This project processes LiDAR data stored in Parquet files to identify individual wire structures using clustering techniques and then reconstructs their 3D catenary models.  

1. Separates the points into individual wire clusters
2. Fits a **catenary curve** to each wire by using

```
y(x) = y0 + c * (cosh((x - x0) / c) - 1 )
```
where `c` is curvature parameter, `x0` is the horizontal position of the lowest point, and `y0` is the minimum elevation.
3. outputs the fitted parameters as a JSON model file, with optional visualization plots

---

## Project Architecture - ETL Pipeline

Extract -> Transform -> Load/Output (main.py)

- EXTRACT : 
    - Loads LiDAR data from Parquet files
    - Implemented in "loader.py"

- TRANSFORM : 

    (A) Clustering:
        - Applies PCA ("Principal Component Analysis") for dimensionality reduction
        - Uses "DBSCAN" algorithm to group points belonging to individual wires
        - Each cluster represents a potential wire segment
        - Stores visualization of clusters in "local_analysis/images_src/cluster_list"

    (B) 3D -> 2D projection:
        - PCA is used to find the best-fit plane for each cluster
        - Points are projected into 2D space for curve-fitting

    (C) Curve Fitting (Catenary Model):
        - Fits a catenary curve to each projected cluster
        - stores visualization plots in "local_analysis/images_src/catenary_curve"
    
    --- Implemented in cluster.py and pca_curve_fitter.py

- LOAD/Output : 
    - Stores clustred dataset as csv (in data/clustered_files) - `Config Driven`
    - Saves fitted catenary parameters as JSON file in the "models/" - `Config Driven`


## Project Structure
```
.
├── .gitignore
├── README.md
├── data
│   ├── clustered_files
│   └── files_input # place your parquet files here
├── local_analysis
│   ├── analysis.ipynb
├── models
├── pyproject.toml
├── requirements.txt
└── src
    └── lidar_catenary
        ├── __init__.py
        ├── cluster.py
        ├── config
        │   └── config.yml  # tunable parameters
        ├── config_loader.py
        ├── loader.py
        ├── main.py
        └── pca_curve_fitter.py
```


## Installation

**Requirements:** Python 3.10+

### From Github (recommended)
```bash
pip3 install git+https://github.com/sowjanyamatam/blunomy_case_study.git@dev
```
> Note: this repository currently has the packaged code and `pyproject.toml` on the `dev` branch. Install from `dev` until the default `main` branch is updated.

### For local development
```bash
git clone https://github.com/sowjanyamatam/blunomy_case_study.git
cd blunomy_case_study

pip3 install -e .
```

---

## Usage
### From CLI

```bash
# place parquet file in data/files_input/ first
python3 src/lidar_catenary/main.py --dataset "lidar_cable_points_easy.parquet"
```

### As a python package
```python
from lidar_catenary import Orchestrator
result = Orchestrator("lidar_cable_points_easy.parquet").run_workflow()

# access the full model
model = result["catenary_model"]

# iterate over fitted wires
for wire in model["wires"]:
    print(wire["wire_id"], wire["x0"], wire["y0"], wire["c"])

# check summary
summary = model["summary"]
print(f"{summary['wires_fitted'] wires fitted}")
```

## Configuration
 
All parameters live in `lidar_catenary/config/config.yml`. No code changes needed to tune the algorithm.
 
| Key | Default | Description |
|-----|---------|-------------|
| `clustering.epsilon_value` | `0.3` | DBSCAN `eps` — maximum distance between points in the same cluster. Decrease for noisier datasets. |
| `clustering.min_samples` | `5` | DBSCAN minimum points to form a cluster. Increase to filter out small noise clusters. |
| `min_points_for_clustering` | `100` | Minimum total points required to attempt clustering. Files below this are rejected. |
| `output.save_images` | `false` | Save cluster and catenary curve plots as PNG. Disable at scale. |
| `output.save_clustered_csv` | `false` | Save the labelled point cloud as CSV after clustering. |
| `output.save_model_json` | `true` | Save fitted catenary parameters as JSON. This is the primary output. |
| `logging.level` | `INFO` | Logging verbosity. Set to `DEBUG` for detailed per-step output. |
 
**Tuning tip:** if DBSCAN is merging wires that should be separate, reduce `epsilon_value`. If it is splitting one wire into multiple clusters, increase it.
 
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
    "wires": {
        "wire_0": {
            "x0": 0.8578404739670262,
            "y0": 10.0016581594244,
            "c": 199.7019147587255
        },
        "wire_1": {
            "x0": -0.35633069688015384,
            "y0": 10.001562031454702,
            "c": 202.4510366045389
        },
        "wire_2": {
            "x0": 0.17258114575546232,
            "y0": 9.997828059778818,
            "c": 201.16211203440506
        }
    }
}
```