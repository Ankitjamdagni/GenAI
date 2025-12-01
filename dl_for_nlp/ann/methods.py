from typing import Any
import pickle
import pandas as pd
from pathlib import Path
from keras.models import load_model # type: ignore
from datetime import datetime

def load_encoders_scalers(path:dict) -> dict[str,Any]:
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
    start = datetime.now()
    with open(path["label_geo_encoder_path"], "rb") as file:
        label_geo_encoder = pickle.load(file=file)
    
    with open(path["label_gender_encoder_path"], "rb") as file:
        label_gender_encoder = pickle.load(file=file)

    with open(path["model_scalers_path"], "rb") as file:
        model_scalers = pickle.load(file=file)

    end = datetime.now()

    print(f"Time to load encoders: {(end-start).microseconds}")

    return {"label_geo_encoder": label_geo_encoder,
            "label_gender_encoder": label_gender_encoder,
            "model_scalers": model_scalers
            }

def transforms_input_data(sample_data:Any, encoders_and_scalers:Any) :
    start = datetime.now()
    label_geo_encoder = encoders_and_scalers['label_geo_encoder']
    label_gender_encoder = encoders_and_scalers['label_gender_encoder']
    model_scalers = encoders_and_scalers['model_scalers']

    input_df = pd.DataFrame([sample_data])
    encoded_geo_value = label_geo_encoder.transform([input_df["Geography"]]).toarray()
    encoded_geo_headers = label_geo_encoder.get_feature_names_out(["Geography"])
    geo_encoded_df = pd.DataFrame(encoded_geo_value, columns=encoded_geo_headers)
    input_df = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)
    input_df = input_df.drop('Geography', axis=1)

    input_df["Gender"] = label_gender_encoder.transform(input_df["Gender"])

    # scaling now

    input_scalers = model_scalers.transform(input_df)
    end = datetime.now()

    print(f"Time to transform input: {(end-start).microseconds}")
    return input_scalers

def load_churning_model(model_path:Path):
    start = datetime.now()
    chruning_model = load_model(model_path)
    end = datetime.now()

    print(f"Time to load model: {(end-start).microseconds}")

    return chruning_model

def predict(input_scalers, prediction_model):
    model_output = prediction_model.predict(input_scalers)
    return model_output