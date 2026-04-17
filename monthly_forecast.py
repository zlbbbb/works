from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass
class ForecastModel:
    seasonal_factors: np.ndarray
    ridge_coef: np.ndarray
    alpha: float


def parse_month_label(value: str) -> pd.Timestamp:
    text = str(value).strip().replace("月", "")
    return pd.to_datetime(text, format="%Y/%m")


def load_monthly_total_from_excel(
    excel_path: str,
    sheet_name: str | int | None = 0,
    date_col: int = 0,
    value_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if df.shape[1] < 2:
        raise ValueError("Excel表至少需要1列日期和1列数值")

    dates = df.iloc[:, date_col].map(parse_month_label)
    if value_cols is None:
        selected_cols = [c for i, c in enumerate(df.columns) if i != date_col]
    else:
        selected_cols = list(value_cols)
        missing_cols = [c for c in selected_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Excel中不存在指定列: {missing_cols}，可选列: {df.columns.tolist()}")

    values = df[selected_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    out = pd.DataFrame({"date": dates, "value": values}).dropna().sort_values("date")
    out = out.reset_index(drop=True)
    return out


def gm11_fit_and_forecast(series: Sequence[float], horizon: int) -> tuple[np.ndarray, np.ndarray]:
    x0 = np.asarray(series, dtype=float)
    n = len(x0)
    if n == 0:
        raise ValueError("训练序列不能为空")

    if n < 4:
        t = np.arange(n)
        if n == 1:
            fit = np.array([x0[0]], dtype=float)
            pred = np.repeat(x0[0], horizon).astype(float)
        else:
            coef = np.polyfit(t, x0, 1)
            fit = np.polyval(coef, t)
            pred = np.polyval(coef, np.arange(n, n + horizon))
        return np.maximum(fit, 1e-6), np.maximum(pred, 1e-6)

    x1 = np.cumsum(x0)
    z1 = 0.5 * (x1[1:] + x1[:-1])
    b_mat = np.column_stack((-z1, np.ones(len(z1))))
    y_vec = x0[1:]
    a, b = np.linalg.lstsq(b_mat, y_vec, rcond=None)[0]

    if np.isclose(a, 0.0):
        t = np.arange(n)
        coef = np.polyfit(t, x0, 1)
        fit = np.polyval(coef, t)
        pred = np.polyval(coef, np.arange(n, n + horizon))
        return np.maximum(fit, 1e-6), np.maximum(pred, 1e-6)

    def x1_hat(k: int) -> float:
        return (x0[0] - b / a) * np.exp(-a * (k - 1)) + b / a

    def x0_hat(k: int) -> float:
        return x1_hat(k) - x1_hat(k - 1)

    fit = np.array([x0[0]] + [x0_hat(k) for k in range(2, n + 1)], dtype=float)
    pred = np.array([x0_hat(k) for k in range(n + 1, n + horizon + 1)], dtype=float)
    return np.maximum(fit, 1e-6), np.maximum(pred, 1e-6)


def _build_monthly_seasonality(
    train_dates: Sequence[pd.Timestamp],
    train_values: np.ndarray,
    trend_fit: np.ndarray,
) -> np.ndarray:
    months = np.array([d.month for d in train_dates])
    ratio = train_values / np.maximum(trend_fit, 1e-6)

    seasonal = np.ones(12, dtype=float)
    for month in range(1, 13):
        month_mask = months == month
        if np.any(month_mask):
            seasonal[month - 1] = float(np.mean(ratio[month_mask]))

    seasonal /= np.mean(seasonal)
    return seasonal


def _ridge_closed_form(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> np.ndarray:
    xtx = x_train.T @ x_train
    reg = alpha * np.eye(xtx.shape[0])
    coef = np.linalg.solve(xtx + reg, x_train.T @ y_train)
    return coef


def fit_gm11_dro_model(
    train_dates: Sequence[pd.Timestamp],
    train_values: Sequence[float],
    alpha: float = 1.0,
) -> ForecastModel:
    y_train = np.asarray(train_values, dtype=float)
    trend_fit, _ = gm11_fit_and_forecast(y_train, horizon=0)
    seasonal = _build_monthly_seasonality(train_dates, y_train, trend_fit)

    month_index = np.array([d.month for d in train_dates]) - 1
    season_train = seasonal[month_index]
    base_train = trend_fit * season_train
    residual_train = y_train - base_train

    x_train = np.column_stack(
        [
            np.ones(len(y_train)),
            trend_fit,
            season_train,
            base_train,
        ]
    )
    coef = _ridge_closed_form(x_train, residual_train, alpha=alpha)
    return ForecastModel(seasonal_factors=seasonal, ridge_coef=coef, alpha=alpha)


def predict_gm11_dro(
    model: ForecastModel,
    train_values: Sequence[float],
    pred_dates: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    _, trend_pred = gm11_fit_and_forecast(train_values, horizon=len(pred_dates))
    months = np.array([d.month for d in pred_dates]) - 1
    season_pred = model.seasonal_factors[months]

    x_pred = np.column_stack(
        [
            np.ones(len(pred_dates)),
            trend_pred,
            season_pred,
            trend_pred * season_pred,
        ]
    )
    residual_pred = x_pred @ model.ridge_coef
    y_pred = trend_pred * season_pred + residual_pred
    irregular = np.where(trend_pred * season_pred != 0, y_pred / (trend_pred * season_pred), 1.0)

    return pd.DataFrame(
        {
            "date": pd.to_datetime(pred_dates),
            "predicted": y_pred,
            "trend_T": trend_pred,
            "season_S": season_pred,
            "irregular_I": irregular,
        }
    )


def evaluate_prediction(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    y_true = np.asarray(list(actual), dtype=float)
    y_pred = np.asarray(list(predicted), dtype=float)
    err = y_pred - y_true
    safe_true = np.where(np.abs(y_true) < 1e-12, np.nan, y_true)
    mape = float(np.nanmean(np.abs(err / safe_true)) * 100)
    return {
        "MAPE": mape,
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
    }


def rolling_one_step_backtest(
    dates: Sequence[pd.Timestamp],
    values: Sequence[float],
    min_train_size: int = 24,
    alpha: float = 1.0,
) -> pd.DataFrame:
    d_arr = pd.to_datetime(dates)
    y_arr = np.asarray(values, dtype=float)
    if len(y_arr) <= min_train_size:
        raise ValueError("数据长度不足，无法执行回测")

    rows = []
    for i in range(min_train_size, len(y_arr)):
        train_dates = d_arr[:i]
        train_values = y_arr[:i]
        model = fit_gm11_dro_model(train_dates, train_values, alpha=alpha)
        pred_df = predict_gm11_dro(model, train_values, [d_arr[i]])
        pred = float(pred_df["predicted"].iloc[0])
        trend = float(pred_df["trend_T"].iloc[0])
        season = float(pred_df["season_S"].iloc[0])
        irregular = float(pred_df["irregular_I"].iloc[0])

        rows.append(
            {
                "date": d_arr[i],
                "actual": y_arr[i],
                "predicted": pred,
                "trend_T": trend,
                "season_S": season,
                "irregular_I": irregular,
            }
        )

    out = pd.DataFrame(rows)
    out["error"] = out["predicted"] - out["actual"]
    safe_actual = out["actual"].where(np.abs(out["actual"]) >= 1e-12, np.nan)
    out["APE%"] = np.abs(out["error"] / safe_actual * 100)
    return out


def forecast_pipeline(
    excel_path: str,
    train_end: str,
    pred_start: str,
    pred_end: str,
    alpha: float = 1.0,
    sheet_name: str | int | None = 0,
    value_cols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame, dict[str, float], ForecastModel]:
    data = load_monthly_total_from_excel(excel_path, sheet_name=sheet_name, value_cols=value_cols)
    train_end_ts = pd.Timestamp(train_end)
    pred_start_ts = pd.Timestamp(pred_start)
    pred_end_ts = pd.Timestamp(pred_end)

    train_mask = data["date"] <= train_end_ts
    pred_mask = (data["date"] >= pred_start_ts) & (data["date"] <= pred_end_ts)

    train_df = data.loc[train_mask].copy()
    pred_df = data.loc[pred_mask].copy()

    model = fit_gm11_dro_model(train_df["date"], train_df["value"], alpha=alpha)
    pred_out = predict_gm11_dro(model, train_df["value"], pred_df["date"])
    pred_out = pred_out.merge(pred_df.rename(columns={"value": "actual"}), on="date", how="left")
    pred_out["error"] = pred_out["predicted"] - pred_out["actual"]
    safe_actual = pred_out["actual"].where(np.abs(pred_out["actual"]) >= 1e-12, np.nan)
    pred_out["APE%"] = np.abs(pred_out["error"] / safe_actual * 100)

    forecast_metrics = evaluate_prediction(pred_out["actual"], pred_out["predicted"])
    backtest_df = rolling_one_step_backtest(data["date"], data["value"], alpha=alpha)
    backtest_metrics = evaluate_prediction(backtest_df["actual"], backtest_df["predicted"])
    return pred_out, forecast_metrics, backtest_df, backtest_metrics, model


def main() -> None:
    parser = argparse.ArgumentParser(description="GM(1,1)+DRO(月度售电量)可复用预测脚本")
    parser.add_argument("--excel", required=True, help="Excel文件路径（支持相对或绝对路径）")
    parser.add_argument("--sheet", default=0, help="工作表名称或序号")
    parser.add_argument("--train-end", default="2023-12-01")
    parser.add_argument("--pred-start", default="2024-01-01")
    parser.add_argument("--pred-end", default="2024-07-01")
    parser.add_argument("--alpha", type=float, default=1.0, help="DRO等价Ridge正则系数")
    parser.add_argument(
        "--categories",
        default="总计,大工业,居民生活,农业生产,工商业",
        help="要预测的类别，逗号分隔。支持“总计/全部/all/total”表示合计列求和",
    )
    args = parser.parse_args()

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    raw_categories = [s.strip() for s in str(args.categories).split(",") if s.strip()]
    categories: list[str] = []
    seen = set()
    for cat in raw_categories:
        if cat not in seen:
            categories.append(cat)
            seen.add(cat)
    if not categories:
        categories = ["总计"]

    total_aliases = {"总计", "全部", "all", "total"}
    for category in categories:
        cat_norm = category.strip().lower()
        is_total = category in {"总计", "全部"} or cat_norm in total_aliases
        value_cols = None if is_total else [category]
        section_name = "总用电量(合计)" if is_total else category

        pred_out, forecast_metrics, backtest_df, backtest_metrics, model = forecast_pipeline(
            excel_path=args.excel,
            train_end=args.train_end,
            pred_start=args.pred_start,
            pred_end=args.pred_end,
            alpha=args.alpha,
            sheet_name=sheet,
            value_cols=value_cols,
        )

        printable = pred_out.copy()
        printable["date"] = printable["date"].dt.strftime("%Y-%m")
        print(f"\n===== {section_name} =====")
        print("2024预测区间结果:")
        print(printable.to_string(index=False, formatters={c: "{:.2f}".format for c in printable.columns if c != "date"}))

        backtest_printable = backtest_df.copy()
        backtest_printable["date"] = backtest_printable["date"].dt.strftime("%Y-%m")
        print("\n历史数据滚动回测结果:")
        print(
            backtest_printable.to_string(
                index=False, formatters={c: "{:.2f}".format for c in backtest_printable.columns if c != "date"}
            )
        )

        print("\n预测区间指标:", {k: round(v, 2) for k, v in forecast_metrics.items()})
        print("历史滚动回测指标:", {k: round(v, 2) for k, v in backtest_metrics.items()})
        print("季节系数(1-12月):", np.round(model.seasonal_factors, 4).tolist())


if __name__ == "__main__":
    main()
