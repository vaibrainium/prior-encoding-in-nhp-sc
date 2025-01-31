from pathlib import Path

import dill
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import style

# from src.utils.psychometric_function import PsychometricFunction

def save_model(model: dict, name: str, dir: str = BASE_DIR / "src" / "models/"):    
    # print(dir.resolve().exists())
    # file_dir = Path(dir) if dir else 
    file = Path(dir) / f"{name}.pkl" 
    with open(file, "wb") as f:
        dill.dump(model, f)
        
def load_model(name: str, dir: str = BASE_DIR / "src" / "models/"):
    file = Path(dir) / f"{name}.pkl" 
    with open(file, "rb") as f:
        model = dill.load(f)
    return model