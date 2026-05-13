import pandas as pd
import matplotlib.pyplot as plt
import os
from lidar_catenary.config_loader import get_config
import numpy as np
import logging
LOGGER = logging.getLogger(__name__)
#This calls the configuration method to access the data defined in the config file.
CONFIG = get_config()

class DataLoader:
    """
    This is used to read data from the provided Parquet file
    """
    required_columns = ["x", "y", "z"]
    def __init__(self, dataset_path):
        """
        File path where the file is located
        """
        self.dataset_path = dataset_path

    def read_data(self):
        """
        Returns the output of the parquet file as a DataFrame
        """
        try:
            self.lidar_data_df = pd.read_parquet(self.dataset_path)
        except Exception as e:
            raise ValueError(f"Failed to read Parquet File '{self.file_path}': {e}") from e
        LOGGER.debug("\nColumn details - \n%s",self.lidar_data_df.columns)
        LOGGER.debug("\nRows and coumns count - \n%s", self.lidar_data_df.shape)
        LOGGER.info("Data is read from parquet files and loaded for the next stage of clustering")
        return self.lidar_data_df
    
    def validate(self, dataset_df):

        # Check if dataset has exactly the required columns
        if list(dataset_df.columns) != self.required_columns:
            raise ValueError(f"Invalid columns. Expected {self.required_columns} got {list(dataset_df.columns)}")

        #Check for missing values in the dataset columns
        null_values_count = dataset_df[["x", "y", "z"]].isnull().sum()
        if null_values_count.any():
            raise ValueError(f"Null values found in columns: {null_values_count.to_dict()}")
        
        # Ensure dataset has enough poiints for clustering
        if len(dataset_df) < CONFIG['min_points_for_clustering']:
            raise ValueError(f"Too few points ({len(dataset_df)}) to cluster correctly. Minimum points required = {CONFIG['min_points_for_clustering']} ")
        
        #Check if there are any infinite points in the dataset provided
        infinite_points = np.isinf(dataset_df[["x", "y", "z"]].values)
        if infinite_points.any():
            raise ValueError("Infinite values found in the dataset")
        

    