from src.data_ingestion import load_data
from src.data_preprocessing import preprocess_data
from src.model_train import train_model
from src.model_evaluate import evaluate_model

def run_pipeline():

    print("Loading data...")
    df = load_data()

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training model...")
    model_path = train_model(X_train, y_train)

    print("Evaluating model...")
    results = evaluate_model(model_path, X_test, y_test)

    print("\n Model Performance")
    print(f"Accuracy  : {results['accuracy']:.4f}")
    print(f"ROC-AUC   : {results['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])
    print("\nClassification Report:")
    print(results["classification_report"])

if __name__ == "__main__":
    run_pipeline()
