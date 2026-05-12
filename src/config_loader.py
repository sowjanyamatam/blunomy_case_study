import yaml
import logging
from pathlib import Path

config = None
BASE_DIR = Path(__file__).resolve().parent.parent
config_path = BASE_DIR / "config" / "config.yml"

def get_config():
    """
    This serves as a central repository for loading configuration files and setting up 
    logging configurations and it can be re-used across other modules
    """
    global config
    if config is None:
        with open(config_path, "r") as yml_file:
            config = yaml.safe_load(yml_file)

        # DEBUG < INFO < WARNING < ERROR < CRITICAL
        logging.basicConfig(
            level = config["logging"]["level"],
            format = config["logging"]["format"]
        )
    
    return config

