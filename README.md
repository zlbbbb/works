# works
forecast

## 可复用月度预测脚本

脚本路径：`monthly_forecast.py`

### 运行示例

```bash
python monthly_forecast.py \
  --excel 2020-2024月度表.xlsx
```

按分类预测（总计 + 大工业/居民生活/农业生产/工商业）：

```bash
python monthly_forecast.py \
  --excel 2020-2024月度表.xlsx \
  --categories 总计,大工业,居民生活,农业生产,工商业
```

### 作为函数复用

可直接在其他代码中导入：

- `load_monthly_total_from_excel`
- `fit_gm11_dro_model`
- `predict_gm11_dro`
- `rolling_one_step_backtest`
- `forecast_pipeline`
