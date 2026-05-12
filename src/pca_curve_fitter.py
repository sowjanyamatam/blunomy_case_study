import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from config_loader import get_config
import json
import logging
LOGGER = logging.getLogger(__name__)
CONFIG = get_config()


class PCA_Curve_Fitter:
    """
    This class fits catenary curves to clustered 3D points data using PCA and curve fitting
    """
    def __init__(self, dataset_df, dataset_name):
        """
        Initilizes dataset name, configuration details and folder paths
        """
        self.dataset_df = dataset_df
        self.dataset_name = dataset_name
        self.base_dataset_name = os.path.splitext(self.dataset_name)[0]
        self.catenary_curve_folder = f"{CONFIG['graphs_output_folder']['catenary_curve']}/{self.base_dataset_name}"
        os.makedirs(self.catenary_curve_folder, exist_ok=True)
        self.catenary_json_folder = CONFIG['models']

    @staticmethod
    def curve_equation(x, x0, y0, c):
        """
        defines the catenary curve equation
        """
        return y0 + c * (np.cosh((x - x0) / c) - 1)

    def pca_curve_fitting(self):
        """
        Perfoms PCA-based transfomration and fits catanery curve for each cluster.
        Outputs the catenary models as json file with values - x0, y0, c
        Saves the catenary curve for each wire as an image 
        """
        self.n_samples = self.dataset_df.shape[0]
        self.n_features = self.dataset_df.shape[1] - 1
        self.number_of_clusters = self.dataset_df['labels'].nunique()
        wire_data = {}
        catenary_points_dict = {}
        for cluster_id in range(self.number_of_clusters):
            cluster_df = self.dataset_df[self.dataset_df['labels'] == cluster_id]
            cluster_array = cluster_df[["x","y","z"]].values
            wire_name = f"wire_{cluster_id}"
            pca_wire = PCA(n_components=min(self.n_samples, self.n_features)).fit(cluster_array)
            pca_wire_projected = pca_wire.transform(cluster_array)
            x_axis_values = pca_wire_projected[:,0]
            z_height_values = cluster_array[:,2]
            wire_data[wire_name] = {'x_axis_values':x_axis_values, 'z_height_values':z_height_values}
            params, _ = curve_fit(PCA_Curve_Fitter.curve_equation, x_axis_values, z_height_values, p0=None)
            
            x0 = params[0]
            y0 = params[1]
            c = params[2]

            catenary_points_dict[wire_name] = {"x0":x0, "y0":y0, "c":c}
        self.json_file_path = os.path.join(self.catenary_json_folder, f"{self.base_dataset_name}_catenary_parameters.json")
        with open(self.json_file_path, "w") as json_file:
            json.dump(catenary_points_dict, json_file, indent=4)
        LOGGER.info("\nThe catenary parameter file is saved in the 'models' folder.\n")

        for wire_name in catenary_points_dict:
            params = catenary_points_dict[wire_name]
            x_axis_values = wire_data[wire_name]['x_axis_values']
            z_height_values = wire_data[wire_name]['z_height_values']

            sort_idx = np.argsort(x_axis_values)
            x_line_value = x_axis_values[sort_idx]
            z_data_sorted_value = z_height_values[sort_idx]
            z_line_value = PCA_Curve_Fitter.curve_equation(x_line_value, params["x0"], params["y0"], params["c"])

            # x_line = np.linspace(x_local.min(),x_local.max(), 500)
            # z_line = curve_equation(x_line, params["x0"],params["y0"],params["c"])

            plt.figure()
            plt.scatter(x_axis_values, z_height_values, s=5, label = "Lidar Points")
            plt.plot(x_line_value, z_line_value, color = "red", label = "catenary curve fit")
            plt.title(wire_name)
            plt.xlabel("pc1_angle")
            plt.ylabel("z-height")
            plt.legend()
            plt.savefig(f"{self.catenary_curve_folder}/{wire_name}_catenary.png")
