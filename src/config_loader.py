import yaml
import logging

def get_config():
    """
    This serves as a central repository for loading configuration files and setting up 
    logging configurations and it can be re-used across other modules
    """

    with open("config/config.yml", "r") as yml_file:
        config = yaml.safe_load(yml_file)

    # DEBUG < INFO < WARNING < ERROR < CRITICAL
    logging.basicConfig(
        level = config["logging"]["level"],
        format = config["logging"]["format"]
    )
    
    return config

