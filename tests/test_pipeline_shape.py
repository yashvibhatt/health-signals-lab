import joblib
import pandas as pd

from lab.forecast.__main__ import predict_next
from lab.forecast.backtest import default_specs, run_backtest, score
from lab.forecast.features import TARGET, make_features
from lab.forecast.models import fit_tree, training_frame
from tests.test_leakage import CFG, seasonal_series


def test_feature_frame_shape():
    y = seasonal_series()
    extra = pd.DataFrame({"s1": seasonal_series(seed=1)})
    f = make_features(y, 2, extra)
    assert len(f) == len(y)
    assert f.shape[1] == 13 + 3 + 1          # 13 national features + 3 per state + target
    assert f[TARGET].iloc[-2:].isna().all()  # last h targets unobserved


def test_backtest_and_metrics_shape():
    y = seasonal_series()
    origins = list(y.index[200:206])
    specs = default_specs(with_extra=False)
    preds = run_backtest(y, origins, [1, 2, 3, 4], specs, CFG)
    assert len(preds) == len(origins) * 4 * len(specs)
    m = score(preds)
    assert set(m.columns) >= {"model", "horizon", "MAE", "RMSE", "MAPE", "n"}
    assert len(m) == 4 * len(specs)


def test_train_save_load_predict_roundtrip(tmp_path):
    y = seasonal_series()
    wide = y.to_frame("nat")
    bundle = {"kind": "lightgbm", "states": [], "trained_through": y.index[-1], "models": {}, "columns": {}}
    for h in [1, 2, 3, 4]:
        _, train = training_frame(y, h, None, None)
        bundle["models"][h] = fit_tree("lightgbm", CFG["lightgbm"], train)
        bundle["columns"][h] = [c for c in train.columns if c != TARGET]
    joblib.dump(bundle, tmp_path / "m.joblib")
    fc = predict_next(joblib.load(tmp_path / "m.joblib"), wide, "nat")
    assert list(fc["horizon"]) == [1, 2, 3, 4]
    assert pd.to_datetime(fc["target_week"]).is_monotonic_increasing
    assert (fc["forecast_wili"] > 0).all()
