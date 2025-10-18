from pathlib import Path
import sys

import pytest
from fastapi import HTTPException

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from diabetes_service import api


class StubModel:
    def __init__(self, value: float):
        self.value = value
        self.calls = []

    def predict(self, X):
        self.calls.append(X)
        return [self.value]


class ErrorModel:
    def predict(self, X):
        raise ValueError("model failure")


def _build_features():
    return api.DiabetesFeatures(
        age=0.02,
        sex=-0.044,
        bmi=0.06,
        bp=-0.03,
        s1=-0.02,
        s2=0.03,
        s3=-0.02,
        s4=0.02,
        s5=0.02,
        s6=-0.001,
    )


def test_predict_returns_prediction(monkeypatch):
    stub = StubModel(42.5)
    monkeypatch.setattr(api, "_model", stub)

    result = api.predict(_build_features())

    assert result == {"prediction": 42.5}
    assert stub.calls == [[
        [0.02, -0.044, 0.06, -0.03, -0.02, 0.03, -0.02, 0.02, 0.02, -0.001]
    ]]


def test_predict_converts_model_error_to_http(monkeypatch):
    monkeypatch.setattr(api, "_model", ErrorModel())

    with pytest.raises(HTTPException) as exc:
        api.predict(_build_features())

    assert exc.value.status_code == 400
    assert exc.value.detail["error"] == "model failure"
