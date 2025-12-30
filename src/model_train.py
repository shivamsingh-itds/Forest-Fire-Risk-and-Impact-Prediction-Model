from sklearn.ensemble import AdaBoostClassifier
import joblib
from pathlib import Path

def train_model(X_train, y_train):

    model = AdaBoostClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Save model
    base_dir = Path(__file__).resolve().parent.parent
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "fire_classification_model.pkl"
    joblib.dump(model, model_path)

    return model_path
