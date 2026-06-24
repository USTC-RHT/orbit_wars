# 数据统计 JSON

这个目录存放手工/离线生成的统计 JSON，避免散落在仓库根目录。

当前约定：

- `leaderboard_top100_bc_data_stats.json`：Kaggle leaderboard top100 与 BC 数据统计的匹配结果。
- `winner_fire_rate_stats_player0_view.json`：player0-view feature-row 数据里的 winner fire-rate 统计。
- `selected5_compare_*.json`：selected5 数据集对比/校验统计。

不放这里的文件：

- 训练输出目录里的 `metrics.jsonl`、`winner_fire_rate_filter_summary.json`。
- 预处理输出目录里的 `manifest.json`、`progress.json`、`conversion_progress.json`。
- ONNX 提交包里的 `manifest.json`、`meta.json`。

这些运行时产物应该继续跟随各自的输出目录。
