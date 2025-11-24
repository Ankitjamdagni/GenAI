from pathlib import Path
from keras.models import load_model
import pandas as pd


# Globals
GLOBALS = {"label_geo_encoder_path" : Path(Path.cwd(), "dl_for_nlp/Data/label_encoders/label_encoder_geo_data.pkl"),
           "label_gender_encoder_path" : Path(Path.cwd(), "dl_for_nlp/Data/label_encoders/label_encoder_gender.pkl"),
           "model_scalers_path" : Path(Path.cwd(), "dl_for_nlp/Data/Churn_Modelling_scaler.pkl")
           }
    


def main() -> None:
    encoders_and_scalers = load_encoders_scalers()
    label_geo_encoder = encoders_and_scalers['label_geo_encoder']
    label_gender_encoder = encoders_and_scalers['label_gender_encoder']
    model_scalers = encoders_and_scalers['model_scalers']

    sample_data = {
        'CreditScore': '600',
        'Geography': 'France',
        'Gender': 'Male',
        'Age': '40',
        'Tenure': '3',
        'Balance': '60000',
        'NumOfProducts': '2',
        'HasCrCard': '1',
        'IsActiveMember': '1',
        'EstimatedSalary': '50000'
    }

    output = label_gender_encoder.transform([sample_data['Gender']])
    print(output)


    # load_checkpoints()

if __name__ == "__main__":
    from methods import load_encoders_scalers
    main()
