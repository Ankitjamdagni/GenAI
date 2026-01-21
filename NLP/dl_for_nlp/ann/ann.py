from pathlib import Path
import methods
import pandas as pd


# Globals
pickle_file_paths = {"label_geo_encoder_path" : Path(Path.cwd(), "dl_for_nlp/Data/label_encoders/label_encoder_geo_data.pkl"),
           "label_gender_encoder_path" : Path(Path.cwd(), "dl_for_nlp/Data/label_encoders/label_encoder_gender.pkl"),
           "model_scalers_path" : Path(Path.cwd(), "dl_for_nlp/Data/Churn_Modelling_scaler.pkl"),
           "model_path" : Path(Path.cwd(), "dl_for_nlp/Data/models/churn_ann_model.keras")
           }
    


def main() -> None:
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
    
    # load encoders
    encoders_and_scalers = methods.load_encoders_scalers(path=pickle_file_paths)

    # transform_input
    input_scalers = methods.transforms_input_data(sample_data=sample_data,
                                          encoders_and_scalers= encoders_and_scalers)

    # load_model
    model = methods.load_churning_model(pickle_file_paths["model_path"])

    # evaluate model
    # 1) load pickled data
    data_frame = pd.read_pickle("dl_for_nlp/Data/Churn_Modelling.pkl")

    
    test_loss, test_mae = model.evaluate() # type: ignore

    # predict
    output = methods.predict(input_scalers=input_scalers,
                     prediction_model=model)

    print(f"chance of leaving the bank for this customer is {output[0][0]*100}% ")


if __name__ == "__main__":
    main()
