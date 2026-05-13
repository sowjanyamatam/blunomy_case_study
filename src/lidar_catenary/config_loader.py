import yaml
import logging
from pathlib import Path

config = None
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "config.yml"
LOGGER = logging.getLogger(__name__)


def get_config(user_config_path= None):
    """
    This serves as a central repository for loading configuration files and setting up 
    logging configurations and it can be re-used across other modules
    """
    global config
    if config is None:
        with open(DEFAULT_CONFIG_PATH, "r") as yml_file:
            config = yaml.safe_load(yml_file)

        #overide any user provided config
        if user_config_path:
            user_path = Path(user_config_path)
            if not user_path.exists():
                raise FileNotFoundError(f"User provided config file not found at: {user_config_path}")
            with open(user_path, "r") as f:
                user_config = yaml.safe_load(f)
            _deep_merge(config, user_config)

        # DEBUG < INFO < WARNING < ERROR < CRITICAL
        logging.basicConfig(
            level = config["logging"]["level"],
            format = config["logging"]["format"]
        )
    LOGGER.debug("FINAL config value are")
    LOGGER.debug(config)
    return config

def _deep_merge(base: dict, override: dict):
    """
    Recursively merges overrides values into base inplace values
    """
    for key, value in override.items():

        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_merge(base[key],value)
        else:
            base[key] = value