from ann import GLOBALS
from typing import Any
import pickle

def load_encoders_scalers() -> dict[str,Any]:
    """ 
    Load scalers and encoders from pickle file.
    
    Files were s aved at the time of feature engineering and 
    dataset division

    Args:
        None(): None

    Returns:
        Dict: having encoders and scalers.

    Example:{"label_geo_encoder": label_geo_encoder,
                "label_gender_encoder": label_gender_encoder,
                "model_scalers": model_scalers
                }
     """
    with open(GLOBALS["label_geo_encoder_path"], "rb") as file:
        label_geo_encoder = pickle.load(file=file)
    
    with open(GLOBALS["label_gender_encoder_path"], "rb") as file:
        label_gender_encoder = pickle.load(file=file)

    with open(GLOBALS["model_scalers_path"], "rb") as file:
        model_scalers = pickle.load(file=file)

    return {"label_geo_encoder": label_geo_encoder,
            "label_gender_encoder": label_gender_encoder,
            "model_scalers": model_scalers
            }
