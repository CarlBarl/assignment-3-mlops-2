from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib, json
from pathlib import Path
import numpy as np


def main():
    # reproducibility
    np.random.seed(42)

    # 1. load dataset
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]

    # 2. split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. define pipeline: scaler + linear regression
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])

    # 4. train
    pipe.fit(X_train, y_train)

    # 5. evaluate
    preds = pipe.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    print(f"RMSE: {rmse:.2f}")

    # 6. save model + metrics
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    joblib.dump(pipe, out_dir / "model.pkl")
    json.dump({"rmse": rmse}, open(out_dir / "metrics.json", "w"), indent=2)
    print("Model and metrics saved to 'artifacts/'")


if __name__ == "__main__":
    main()
