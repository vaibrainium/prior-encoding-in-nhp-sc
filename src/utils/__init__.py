from pathlib import Path

config_path = Path(__file__).resolve().parent.parent.parent / 'config'

print(f"Config path: {config_path} loaded, file-exists: {config_path.is_dir()}")