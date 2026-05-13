from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os
from lidar_catenary.config_loader import get_config
from datetime import datetime
import logging
LOGGER = logging.getLogger(__name__)
# CONFIG = get_config()

class DataCluster:
    """
    This class applies PCA and then uses DBScan for clustering
    """
    def __init__(self, dataset_df, dataset_name, output_dir):
        """
        Intilizes clustering setup (dataset name, clustering parameters, configuration details)
        """
        self.CONFIG = get_config()
        self.epsilon_value = self.CONFIG["clustering"]["epsilon_value"]
        self.min_samples = self.CONFIG["clustering"]["min_samples"]
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.base_dataset_name = os.path.splitext(self.dataset_name)[0]

        #Paths for Cluster list plot
        self.cluster_list_plot_folder = os.path.join(self.output_dir,'cluster_list',self.base_dataset_name)
        self.dataset_df = dataset_df

        #paths for .csv clustered data file
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.output_cluster_csv_file_path = os.path.join(self.output_dir,'clustered_files',self.base_dataset_name)
        self.output_cluster_file_name = f"{self.timestamp}_cluster_data_file.csv"
        
    def clustering(self):
        """
        Applies PCA-based dimensionality reduction followed by DBScan clustering.
        Saves visualization of clusters as image and clustered dataset in .csv form
        """
        #Stores dataset rows and columns data and findsnumber of PCA components
        self.n_samples = self.dataset_df.shape[0]
        self.n_features = self.dataset_df.shape[1]
        n_components_value = min(self.n_samples, self.n_features)

        # Fit PCA on points and tranforms it
        coordinates = self.dataset_df[["x","y","z"]].values
        pca = PCA(n_components=n_components_value).fit(coordinates)
        projected_coordinates = pca.transform(coordinates)

        #highest variance axis removal
        remove_axis = np.argmax((pca.explained_variance_ratio_))
        actual_reduced_coordinates = np.delete(projected_coordinates, remove_axis, axis = 1)

        #DBScan run
        LOGGER.info("Running DBScan with eps=%.3f, min_smaples=%d on %d points",self.epsilon_value, self.min_samples, len(self.dataset_df))
        labels_for_clusters = DBSCAN(eps = self.epsilon_value, min_samples = self.min_samples).fit_predict(actual_reduced_coordinates)
        noise_count = np.sum(labels_for_clusters == -1)
        number_of_clusters = len(set(labels_for_clusters)) - noise_count
        LOGGER.info("Clustering complete - %d clusters and %d noise count", number_of_clusters, noise_count)

        # Save cluster visualization if enabled in config file
        if self.CONFIG["output"]["save_images"]:
            os.makedirs(self.cluster_list_plot_folder, exist_ok=True)
            LOGGER.info("Saving clustering images")
            fig, ax = plt.subplots()
            for cluster_id in set(labels_for_clusters):
                if cluster_id == -1:
                    continue
                cluster_points = actual_reduced_coordinates[labels_for_clusters == cluster_id]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=2, label = f"Cluster {cluster_id}")
            fig.savefig(f"{self.cluster_list_plot_folder}/cluster_list_visualization.png")
            LOGGER.info(f"saved cluster image to {self.cluster_list_plot_folder}/cluster_list_visualization.png")
            plt.close(fig)

        self.labeled_dataset_df = self.dataset_df.copy() 
        self.labeled_dataset_df["labels"] = labels_for_clusters
        
        #Save .csv clustred data files if enabled in cnofg file
        if self.CONFIG["output"]["save_clustered_csv"]:
            os.makedirs(self.output_cluster_csv_file_path, exist_ok=True)
            self.labeled_dataset_df.to_csv(os.path.join(self.output_cluster_csv_file_path,self.output_cluster_file_name))
            LOGGER.debug("The clustered data file is saved in the 'data/clustered_files' folder.\n")

        return self.labeled_dataset_df, number_of_clusters