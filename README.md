# blunomy_case_study

# 3D Catenary Model Generation from LiDAR Data

This project processes LiDAR data stored in Parquet files to identify individual wire structures using clustering techniques and then reconstructs their 3D catenary models.  
It outputs :
 - clustered datasets in csv format
 - curve-fitted catenary parameters in json format
 - visualization plots for analysis (cluster visulaization plots and catenary curve fit plots for each wire)


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
    - Stores clustred dataset as csv (in data/clustered_files)
    - Saves fitted catenary parameters as JSON file in the "models/"


## setup

- pip3 install -r requirements.txt
- python3 src/main.py  --dataset "dataset-to-be-processed.parquet"
    

## Python files
- /src/main.py
- /src/cluster.py
- /src/loader.py
- /src/pca_curve_fitter.py
- /src/config_loader.py 
    
