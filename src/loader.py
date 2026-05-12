import pandas as pd
import matplotlib.pyplot as plt
import os
from config_loader import get_config
import numpy as np
import logging
LOGGER = logging.getLogger(__name__)
CONFIG = get_config()

class DataLoader:
    """
    This is used to read data from the provided Parquet file
    """
    required_columns = {"x", "y", "z"}
    def __init__(self, dataset_name):
        """
        This calls the configuration method to access the data defined in the config file.
        Name of the dataset file to be processed
        File path where the file is located
        """
        self.dataset_name = dataset_name
        self.file_path = f"{CONFIG['base_file_path']}/{self.dataset_name}" 

    def read_data(self):
        """
        Returns the output of the parquet file as a DataFrame
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset not found: {self.file_path}")
        try:
            self.lidar_data_df = pd.read_parquet(self.file_path)
            missing_columns = self.required_columns - set(self.lidar_data_df.columns)
            if missing_columns:
                raise ValueError(f"Input File is missing required columns: {missing_columns}")
        except Exception as e:
            raise ValueError(f"Failed to read Parquet File '{self.file_path}': {e}") from e
        LOGGER.debug("\nDataset:\n%s", self.lidar_data_df)
        LOGGER.debug("\nFirst few rows in Dataset - \n%s",self.lidar_data_df.head())
        LOGGER.debug("\nColumn details - \n%s",self.lidar_data_df.columns)
        LOGGER.debug("\nRows and coumns count - \n%s", self.lidar_data_df.shape)
        LOGGER.info("Data is read from parquet files and loaded for the next stage of clustering")
        return self.lidar_data_df
    
    def validate(self, dataset_df):
        null_values_count = dataset_df[["x", "y", "z"]].isnull().sum()
        if null_values_count.any():
            raise ValueError(f"Null values found in columns: {null_values_count.to_dict()}")
        if len(dataset_df) < 10:
            raise ValueError(f"Too few points ({len(dataset_df)}) to cluster it correctly")
        infinite_points = np.isinf(dataset_df[["x", "y", "z"]].values)
        if infinite_points.any():
            raise ValueError("Infinite values found in the dataset")

    