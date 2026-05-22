# 评估与排行榜

> **所属页面**：[Home](Home.md) > 评估与排行榜

---

## 1. 评估体系概览

本项目使用三层评估体系：

```
Level 1: 本地快速验证
  └─ Quick Match：单场 2P/4P 对战，立即查看 replay

Level 2: 本地系统评估
  └─ Round-Robin 锦标赛：所有 Agent 两两对战，更新 TrueSkill 排行榜
  └─ Gauntlet 挑战赛：新 Agent 对阵全体，快速获得相对排名

Level 3: 官方竞赛评估
  └─ Kaggle 提交：上传 main.py，与全球参赛者对战
```

---

## 2. TrueSkill 评分系统

### 2.1 算法简介

TrueSkill 是微软研究院开发的贝叶斯评分算法，相比 Elo 的优势：

- 支持多于 2 人的对局（4P 比赛）
- 通过 sigma（不确定度）量化评分可信度
- 以**保守分**（Conservative Score）= `mu - 3 * sigma` 排序，防止运气型虚高

### 2.2 本地参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `TS_MU_0` | 600.0 | 初始平均分（Kaggle 默认 1000，本地调低以便区分） |
| `TS_SIGMA_0` | 200.0 | 初始不确定度 |
| `TS_BETA` | 100.0 | 技能差异半径（控制偶然性影响） |
| `TS_TAU` | 2.0 | 动态因子（允许排名随时间漂移，适应 Agent 更新） |
| `TS_DRAW_PROB` | 0.05 | 平局概率 |

### 2.3 评分持久化

评分数据保存在：

```
orbit-wars-lab/runs/trueskill.json
```

格式示例：

```json
{
  "2p": {
    "external/tamrazov-starwars": {"mu": 820.5, "sigma": 45.2},
    "external/kashiwaba-rl":      {"mu": 750.1, "sigma": 58.3},
    ...
  },
  "4p": { ... }
}
```

每场比赛结束后实时更新，重启后端不会丢失评分历史。

---

## 3. 赛事格式

### 3.1 Quick Match（快速对战）

- 通过 Web UI `#/` 页面发起
- 选择 2 个或 4 个 Agent，选择 2P/4P 格式
- 单场比赛，SSE 实时进度，结束后自动加载 replay

### 3.2 Round-Robin（循环赛）

- 通过 Web UI `#/tournaments` 页面创建
- 参数：
  - 参赛 Agent 列表
  - 每对之间的比赛场数 K
  - 2P 或 4P 格式
- 所有场次并行执行（`ProcessPoolExecutor`），结束后更新 TrueSkill

**适用场景**：全面评估所有 Agent 的相对强度，建立排行榜。

### 3.3 Gauntlet（挑战赛）

- 新 Agent 对阵仓库中所有其他 Agent
- 比圆循环赛更快，聚焦于新 Agent 的定位
- 赛后更新 TrueSkill，排名变化反映真实实力

**适用场景**：快速验证新 Agent 或新策略的有效性。

---

## 4. Replay 管理

### 4.1 本地 Replay

每场比赛的 replay 保存为 JSON 文件：

```
runs/{run-id}/match_{n}.json
```

通过 Web UI `#/replays` 页面浏览和播放。

### 4.2 Kaggle Replay 导入

若已配置 Kaggle API Token，可在 `#/submissions` 页面查看官方 LB 提交记录，并直接在本地播放 Kaggle 上的历史对局。

**配置 Token**：

```bash
# 在 Web UI 的 Settings 页面粘贴 Token
# 或通过 API
curl -X POST http://localhost:8000/api/kaggle-settings \
  -H "Content-Type: application/json" \
  -d '{"token": "your_kaggle_api_token"}'
```

---

## 5. Kaggle 竞赛提交

### 5.1 提交文件准备

Kaggle 竞赛提交的文件为单个 `main.py`，必须满足：

1. 包含 `def agent(obs):` 函数
2. 不依赖本地文件系统路径（使用相对路径或打包资源）
3. 对于 RL Agent，需将模型权重内嵌或以相对路径引用

**规则型 Agent 提交示例**：

```python
# main.py（提交给 Kaggle）
import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

def agent(obs):
    # ... 完整策略实现 ...
    return moves
```

**RL Agent 提交注意事项**：

- 模型权重需与 `main.py` 一起打包提交（Kaggle 支持提交目录压缩包）
- 路径使用 `__file__` 相对定位：

```python
import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pt")
```

### 5.2 提交流程

```bash
# 方式 1：通过 Kaggle CLI
kaggle competitions submit \
  -c orbit-wars \
  -f main.py \
  -m "Strategy description"

# 方式 2：通过 Web UI 上传（Kaggle 竞赛页面）
```

### 5.3 分数读取

提交后约数小时内 LB 分数更新，可通过本地 Web UI 的 `#/submissions` 页面同步查看（需 Kaggle API Token）。

---

## 6. 评估指标参考

### 6.1 vs nearest-sniper 胜率

`nearest-sniper` 是最关键的基准对手，胜率是衡量 Agent 能力的核心指标：

| 胜率区间 | 水平评估 |
|----------|----------|
| < 50% | 低于基准，策略存在缺陷 |
| 50–80% | 初步有效，需优化 |
| 80–95% | 良好，可考虑提交竞赛 |
| 95–100% | 优秀，`kashiwaba-rl` 达到 100% |

### 6.2 本地 TrueSkill 保守分参考

| 保守分区间 | 相应 LB 分数参考 |
|------------|-----------------|
| > 600 | 新 Agent，样本少 |
| 400–600 | 初步可用 |
| 200–400 | 有竞争力 |
| < 200 | 顶级水平（本地参数下） |

> 注：本地保守分与 Kaggle LB 分数不可直接对比，仅用于相对排名参考。

---

## 7. 调试技巧

### 7.1 查看 replay 详情

```python
import json

with open("orbit-wars-lab/runs/latest/match_0.json") as f:
    replay = json.load(f)

# 查看终局状态
final_step = replay["steps"][-1]
for i, player_state in enumerate(final_step):
    print(f"Player {i}: reward={player_state['reward']}")
```

### 7.2 打印关键回合的观测

```python
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

env = make("orbit_wars", debug=True)
env.run([your_agent, "random"])

# 查看第 1 步的观测
step1_obs = env.steps[1][0].observation
planets = [Planet(*p) for p in step1_obs.planets]
for p in planets:
    owner = f"P{p.owner}" if p.owner >= 0 else "Neutral"
    print(f"  {owner} @ ({p.x:.1f},{p.y:.1f}) prod={p.production} ships={p.ships}")
```

### 7.3 RL Agent 推理调试

```bash
python RL/eval.py \
  --config RL/default_cfg.yaml \
  --checkpoint RL/artifacts/run_name/ckpt_last.pt \
  --games 20 \
  --opponent nearest \
  --device cpu
```

---

## 相关页面

- [竞赛简介](Competition-Overview.md)
- [系统架构](System-Architecture.md)
- [Agent 全览](Agent-Zoo.md)
- [开发指南](Development-Guide.md)
