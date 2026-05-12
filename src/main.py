import argparse
from loader import DataLoader
from cluster import DataCluster
from pca_curve_fitter import PCACurveFitter
from config_loader import get_config
import json
import logging
LOGGER = logging.getLogger(__name__)
CONFIG = get_config()

class Orchestrator:
    """
    Orchestrates the full LiDAR data prcoessing pipeline: Data loading, clustering and curve fitting
    """

    def __init__(self, dataset_name):
        """
        Name of the dataset file to be processed
        """
        self.dataset_name = dataset_name

    def run_workflow(self):
        """
        Executes the complete pipeline in sequence and logs the output plot locations
        """

        data_load_object = DataLoader(self.dataset_name)
        dataset_df = data_load_object.read_data()
        data_load_object.validate(dataset_df)

        data_cluster_object = DataCluster(dataset_df, self.dataset_name)
        labeled_dataset_df, number_of_clusters = data_cluster_object.clustering()

        pca_curve_fitter_object = PCACurveFitter(labeled_dataset_df, self.dataset_name)
        pca_curve_fitter_object.pca_curve_fitting()

        LOGGER.debug("** Please check the 'local_analysis/images_src' folder to view the generated cluster seperation plots for the dataset and catenary curve fitting visualizations **\n")
        
        return {
            "number_of_clusters" : int(number_of_clusters),
            "cluster_list_plot_path" : 'local_analysis/images_src/cluster_list/',
            "clustered_data_csv_file_path" : 'data/clustered_files',
            "catenary_curve_plot_path" : 'local_analysis/images_src/catenary_curve/',
            "catenary_model_json_path" : 'models/'
        }



if __name__ == "__main__":
    # takes the file name to be processed dynamically using 'argparse' and passes it to the workflow
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required = True, help = "Specify the file name that you want to process. Example = 'python3 src/main.py --dataset lidar_cable_points_easy.parquet'")
    args = parser.parse_args()
    orchestrator_object = Orchestrator(args.dataset)
    result = orchestrator_object.run_workflow()
    print("Result details = ", json.dumps(result, indent=4))
