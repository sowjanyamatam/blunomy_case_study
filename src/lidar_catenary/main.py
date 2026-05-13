import argparse
from lidar_catenary.loader import DataLoader
from lidar_catenary.cluster import DataCluster
from lidar_catenary.pca_curve_fitter import PCACurveFitter
from lidar_catenary.config_loader import get_config
import json
import os
import logging
LOGGER = logging.getLogger(__name__)
CONFIG = get_config()

class Orchestrator:
    """
    Orchestrates the full LiDAR data prcoessing pipeline: Data loading, clustering and curve fitting
    """

    def __init__(self, dataset_path, output_dir = None):
        """
        Name of the dataset file to be processed
        """
        self.dataset_name = os.path.basename(dataset_path)
        self.dataset_path = dataset_path
        self.output_dir = output_dir or os.path.join(os.getcwd(), "lidar_output")


    def run_workflow(self):
        """
        Executes the complete pipeline in sequence and logs the output plot locations
        """
        LOGGER.info("DataLoader start..")
        data_load_object = DataLoader(self.dataset_path)
        dataset_df = data_load_object.read_data()
        data_load_object.validate(dataset_df)
        LOGGER.info("DataLoader end...\n")

        LOGGER.info("DataCluster start..")
        data_cluster_object = DataCluster(dataset_df, self.dataset_name, self.output_dir)
        labeled_dataset_df, number_of_clusters = data_cluster_object.clustering()
        LOGGER.info("DataCluster end...\n")

        LOGGER.info("PCACurveFitter start..")
        pca_curve_fitter_object = PCACurveFitter(labeled_dataset_df, self.dataset_name, number_of_clusters, self.output_dir)
        catenary_results = pca_curve_fitter_object.pca_curve_fitting()
        LOGGER.info("PCACurveFitter end...\n")

        LOGGER.debug("** Please check the 'local_analysis/images_src' folder to view the generated cluster seperation plots for the dataset and catenary curve fitting visualizations **\n")
        
        return {
            "catenary_model": catenary_results,
            "cluster_list_plot_path" : os.path.join(self.output_dir,'cluster_list'),
            "clustered_data_csv_file_path" : os.path.join(self.output_dir,'clustered_files'),
            "catenary_curve_plot_path" : os.path.join(self.output_dir,'catenary_curve'),
            "catenary_model_json_path" : os.path.join(self.output_dir,'models')
        }



if __name__ == "__main__":
    # takes the file name to be processed dynamically using 'argparse' and passes it to the workflow
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required = True, help = "Specify the file path that you want to process")
    parser.add_argument("--output-dir", required=False, default=None, help = "Directory to save all outputs (plots, csvs, models)")
    parser.add_argument("--config", required = False, default = None, help = "Path to override the existing config value")
    args = parser.parse_args()
    #file_path = f"{CONFIG['base_file_path']}/{args.dataset}" 

    # This replaces base config values with the ones which we provided while running
    get_config(user_config_path=args.config)
    file_path = args.dataset

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    if not file_path.endswith(".parquet"):
        raise ValueError("Given file is not a parquet file")
    orchestrator_object = Orchestrator(file_path, output_dir=args.output_dir)
    result = orchestrator_object.run_workflow()
    print("Result details = ", json.dumps(result, indent=4))
