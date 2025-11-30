import pickle
import argparse
import json
import glob
import os
from prefect import flow
from preprocessing import read_rawdata, process_vars

@flow
def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

@flow
def prepare_inference_data(raw_dir, period, type_work):
    file_path = glob.glob(f"{raw_dir}/p{period}_extrac.csv")[0]

    df = read_rawdata(period, type_work, raw_dir)
    df = process_vars(df)
    return df, file_path

@flow
def predict(model, df, model_path):
    feature_names_path = os.path.join(os.path.dirname(model_path), "feature_names.json")

    with open(feature_names_path, "r") as f:
        model_features = json.load(f)

    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    df_model = df[model_features]

    preds_proba = model.predict_proba(df_model)[:, 1]
    preds_class = (preds_proba >= 0.5).astype(int)
    return preds_proba, preds_class

@flow
def save_predictions(df, preds_proba, preds_class, output_dir, input_file):
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.basename(input_file)
    name, ext = os.path.splitext(base) 
    output_path = os.path.join(output_dir, f"{name}_pred{ext}")

    df_out = df.copy()
    df_out["pred_proba"] = preds_proba
    df_out["pred_class"] = preds_class

    df_out.to_csv(output_path, index=False)

@flow
def inference(raw_dir, model_path, output_dir, period, type_work):
    df, input_file = prepare_inference_data(raw_dir, period, type_work)
    model = load_model(model_path)
    preds_proba, preds_class = predict(model, df, model_path)
    save_predictions(df, preds_proba, preds_class, output_dir, input_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Directorio de datos crudos para inferencia")
    parser.add_argument("output", type=str, help="Directorio donde se guardan las predicciones")
    parser.add_argument("--model-path", type=str, help="Ruta al modelo .pkl")
    parser.add_argument("--period", type=str, default="", help="Periodo")

    args = parser.parse_args()
    inference(
        raw_dir=args.input,
        model_path=args.model_path,
        output_dir=args.output,
        period=args.period,
        type_work="inference"
    )