# from pathlib import Path

# config_path = Path(__file__).resolve().parent.parent.parent / 'config'

# print(f"Config path: {config_path} loaded, file-exists: {config_path.is_dir()}")


from pathlib import Path
from omegaconf import OmegaConf

# Define the path to the config directory
config_dir = Path(__file__).parent

# Load the YAML configuration file
def load_config(filename):
    return OmegaConf.load(config_dir / filename)

# Example of accessing a specific configuration file
dir_config = load_config('dir-config.yaml')
main_config = load_config('main.yaml')