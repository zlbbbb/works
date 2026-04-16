# works
forecast

## 可复用月度预测脚本

脚本路径：`/home/runner/work/works/works/monthly_forecast.py`

### 运行示例

```bash
python /home/runner/work/works/works/monthly_forecast.py \
  --excel /home/runner/work/works/works/2020-2024月度表.xlsx
```

### 作为函数复用

可直接在其他代码中导入：

- `load_monthly_total_from_excel`
- `fit_gm11_dro_model`
- `predict_gm11_dro`
- `rolling_one_step_backtest`
- `forecast_pipeline`
