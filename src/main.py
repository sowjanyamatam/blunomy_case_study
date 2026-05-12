import argparse
from loader import Data_Loader
from cluster import Data_Cluster
from pca_curve_fitter import PCA_Curve_Fitter
from config_loader import get_config
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

        data_load_object = Data_Loader(self.dataset_name)
        dataset_df = data_load_object.read_data()

        data_cluster_object = Data_Cluster(dataset_df, self.dataset_name)
        number_of_clusters = data_cluster_object.clustering()

        pca_curve_fitter_object = PCA_Curve_Fitter(dataset_df, self.dataset_name)
        pca_curve_fitter_object.pca_curve_fitting()

        LOGGER.info("\n** Please check the 'local_analysis/images_src' folder to view the generated cluster seperation plots for the dataset and catenary curve fitting visualizations **\n")
        LOGGER.info("Number of wires in this dataset = %s", number_of_clusters)



if __name__ == "__main__":
    """
    takes the file to be processed dynamically using 'argparse' and passes it to the workflow
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required = True, help = "Specify the file name that you want to process. Example = 'python3 src/main.py --dataset lidar_cable_points_easy.parquet'")
    args = parser.parse_args()
    orchestrator_object = Orchestrator(args.dataset)
    orchestrator_object.run_workflow()
