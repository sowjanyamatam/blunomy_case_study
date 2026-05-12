import pandas as pd
import matplotlib.pyplot as plt
import os
from config_loader import get_config
import logging
LOGGER = logging.getLogger(__name__)
CONFIG = get_config()

class Data_Loader:
    """
    This is used to read data from the provided Parquet file
    """
    
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
        self.lidar_data_df = pd.read_parquet(self.file_path)
        LOGGER.debug("\nDataset:\n", self.lidar_data_df)
        LOGGER.debug("\nFirst few rows in Dataset - \n",self.lidar_data_df.head())
        LOGGER.debug("\nColumn details - \n",self.lidar_data_df.columns)
        LOGGER.debug("\nRows and coumns count - \n", self.lidar_data_df.shape)
        return self.lidar_data_df

    