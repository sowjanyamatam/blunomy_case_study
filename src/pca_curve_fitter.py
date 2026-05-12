import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import copy
from config_loader import get_config
import json
from datetime import datetime
import logging
LOGGER = logging.getLogger(__name__)
CONFIG = get_config()


class PCACurveFitter:
    """
    This class fits catenary curves to clustered 3D points data using PCA and curve fitting
    """
    def __init__(self, labeled_dataset_df, dataset_name, clusters_count):
        """
        Initilizes dataset name, configuration details and folder paths
        """
        self.labeled_dataset_df = labeled_dataset_df
        self.dataset_name = dataset_name
        self.base_dataset_name = os.path.splitext(self.dataset_name)[0]
        self.catenary_curve_folder = f"{CONFIG['graphs_output_folder']['catenary_curve']}/{self.base_dataset_name}"
        os.makedirs(self.catenary_curve_folder, exist_ok=True)
        self.catenary_json_folder = os.path.join(CONFIG['models'],self.base_dataset_name)
        os.makedirs(self.catenary_json_folder, exist_ok=True)
        self.clusters_count = clusters_count
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

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
        self.n_samples = self.labeled_dataset_df.shape[0]
        self.n_features = self.labeled_dataset_df.shape[1] - 1
        self.number_of_clusters = self.labeled_dataset_df['labels'].nunique()
        wire_data = {}
        catenary_points_dict = {
            "File_name" : {},
            "Row_count" : {},
            "Timestamp" : {},
            "clustering_parameters": {},
            "summary": {},
            "wires": {}
        }
        failed_wires = []
        self.epsilon_value = CONFIG["clustering"]["epsilon_value"]
        self.min_samples = CONFIG["clustering"]["min_samples"]
        catenary_points_dict["clustering_parameters"] = {
                "epsilon_value" : float(self.epsilon_value),
                "min_samples": int(self.min_samples)
        }
        catenary_points_dict["File_name"] = self.dataset_name
        catenary_points_dict["Row_count"] = self.labeled_dataset_df.shape[0]
        catenary_points_dict["Timestamp"] = self.timestamp

        for cluster_id in range(self.number_of_clusters):
            if cluster_id == -1:
                continue
            cluster_df = self.labeled_dataset_df[self.labeled_dataset_df['labels'] == cluster_id]
            cluster_array = cluster_df[["x","y","z"]].values
            wire_name = f"wire_{cluster_id}"
            pca_wire = PCA(n_components=min(self.n_samples, self.n_features)).fit(cluster_array)
            pca_wire_projected = pca_wire.transform(cluster_array)
            x_axis_values = pca_wire_projected[:,0]
            z_height_values = cluster_array[:,2]
            wire_data[wire_name] = {'x_axis_values':x_axis_values, 'z_height_values':z_height_values}
            try:
                params, _ = curve_fit(PCACurveFitter.curve_equation, x_axis_values, z_height_values, p0=None)
            except RuntimeError as e:
                LOGGER.warning("Curve Fit failed for %s: %s", wire_name, e)
                failed_wires.append(wire_name)
                continue
            
            x0 = float(params[0])
            y0 = float(params[1])
            c = float(params[2])

            catenary_points_dict["wires"][wire_name] = {"x0":x0, "y0":y0, "c":c}

        
        for wire_name in catenary_points_dict["wires"]:
            params = catenary_points_dict["wires"][wire_name]
            x_axis_values = wire_data[wire_name]['x_axis_values']
            z_height_values = wire_data[wire_name]['z_height_values']

            sort_idx = np.argsort(x_axis_values)
            x_line_value = x_axis_values[sort_idx]
            z_data_sorted_value = z_height_values[sort_idx]
            z_line_value = PCACurveFitter.curve_equation(x_line_value, params["x0"], params["y0"], params["c"])

            plt.figure()
            plt.scatter(x_axis_values, z_height_values, s=5, label = "Lidar Points")
            plt.plot(x_line_value, z_line_value, color = "red", label = "catenary curve fit")
            plt.title(wire_name)
            plt.xlabel("pc1_angle")
            plt.ylabel("z-height")
            plt.legend()
            plt.savefig(f"{self.catenary_curve_folder}/{wire_name}_catenary.png")
            plt.close()

        wires_fitted = len(catenary_points_dict["wires"])
        catenary_points_dict["summary"] = {
            "number_of_wires" : int(self.clusters_count),
            "wires_fitted" : int(wires_fitted),
            "wires_failed" : failed_wires,
        }

        self.json_file_path = os.path.join(self.catenary_json_folder, f"{self.timestamp}_catenary_parameters.json")
        with open(self.json_file_path, "w") as json_file:
            json.dump(catenary_points_dict, json_file, indent=4)
        
        LOGGER.info("The catenary model is saved to %s folder. %d wires fitted succesfully", self.json_file_path, wires_fitted)
        if failed_wires:
            LOGGER.warning("Wires that failed curve fitting: %s", failed_wires)

