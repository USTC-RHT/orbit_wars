# Agent 全览（Agent Zoo）

> **所属页面**：[Home](Home.md) > Agent 全览

---

## 1. 目录结构

```
agents/
├── baselines/           Kaggle 官方基准 Agent（3 个）
│   ├── random/
│   ├── starter/
│   └── nearest-sniper/
├── external/            公开竞赛笔记本中的高分 Agent（8 个）
│   ├── pilkwang-structured/
│   ├── tamrazov-starwars/
│   ├── sigmaborov-starter/
│   ├── sigmaborov-reinforce/
│   ├── yuriygreben-architect/
│   ├── ykhnkf-distance-prioritized/
│   ├── pascal-orbitwork-v14/
│   └── kashiwaba-rl/
└── mine/                自研 Agent（可放置多个）
```

每个 Agent 目录包含：

| 文件 | 说明 |
|------|------|
| `main.py` | 核心逻辑，必须实现 `def agent(obs):` |
| `agent.yaml`（可选） | 元数据：名称、作者、标签、LB 分数等 |

---

## 2. 横向对比总览

| Agent | 类型 | 代码量 | LB 分数（声称） | 特色关键词 |
|-------|------|--------|----------------|-----------|
| random | 规则 | 2 行 | — | 纯随机，最低基准 |
| starter | 规则 | 2 行 | — | 静止星球狙击 |
| nearest-sniper | 规则 | 60 行 | — | 最近目标贪心 |
| sigmaborov-starter | 规则 | 452 行 | — | 物理感知、轨道预测 |
| sigmaborov-reinforce | 规则 | 1895 行 | 958.1 | 世界模型、滚动前向模拟 |
| pilkwang-structured | 规则 | 3107 行 | ~1000 | 8 战术评分、到达时间账本 |
| yuriygreben-architect | 规则 | 3002 行 | — | 5 层架构、多阶段规划 |
| ykhnkf-distance-prioritized | 规则 | 3162 行 | ~1100 | 距离优先、完整世界模型 |
| pascal-orbitwork-v14 | 规则 | 3160 行 | — | pilkwang 分叉版，14 版迭代 |
| tamrazov-starwars | 规则 | 3306 行 | **1224** | 全面战争模式、最强规则型 |
| kashiwaba-rl | **神经网络** | 91+模块 | — | PPO，唯一 RL，vs nearest 100% 胜率 |

---

## 3. 基准 Agent 详解（Baselines）

### 3.1 `random` — 随机 Agent

| 属性 | 值 |
|------|----|
| 来源 | Kaggle 官方 |
| 代码量 | 2 行（re-export） |
| 策略 | 完全随机方向和数量发射舰队 |

**用途**：最低基准线，验证任何有逻辑的 Agent 都能打赢它。

---

### 3.2 `starter` — 静态狙击 Agent

| 属性 | 值 |
|------|----|
| 来源 | Kaggle 官方 |
| 代码量 | 2 行（re-export） |
| 策略 | 瞄准静止星球，发射恰好足量的舰队占领 |

**用途**：比 `random` 稍强，仍为参考基准。

---

### 3.3 `nearest-sniper` — 最近星球狙击 Agent ⭐

| 属性 | 值 |
|------|----|
| 来源 | Kaggle 官方 |
| 代码量 | 60 行 |
| 策略 | 对每颗己方星球，找距离最近的非己星球并尝试占领 |

**核心逻辑**：

```python
nearest = min(targets, key=lambda t: dist(mine, t))
ships_needed = nearest.ships + 1

if mine.ships >= ships_needed:
    angle = atan2(nearest.y - mine.y, nearest.x - mine.x)
    moves.append([mine.id, angle, ships_needed])
```

**局限性**：
- 不考虑星球运动，不预测未来位置
- 不规避太阳，不计算飞行时间
- 不包含任何防守逻辑

**战略地位**：是衡量其他 Agent 强度的关键标尺（`kashiwaba-rl` 对它 20/20 胜率）。

---

## 4. 外部高分 Agent 详解（External）

### 4.1 `sigmaborov-starter` — 物理感知基础 Agent

| 属性 | 值 |
|------|----|
| 来源 | sigmaborov（社区公开笔记本） |
| 代码量 | 452 行 |
| 许可证 | Apache 2.0 |
| 标签 | 规则驱动、彗星感知、太阳规避、轨道预测 |

**核心能力**（相比官方基准的进化）：

```python
# 预测轨道星球的未来位置
def predict_planet_position(planet, initial_by_id, angular_velocity, turns):
    ...

# 判断路径是否撞太阳，如果会则生成绕行航点
def safe_angle_and_distance(sx, sy, tx, ty):
    for sign in (1.0, -1.0):
        wx = CENTER_X + sign * nx * (SUN_R * mult)  # 绕行航点
        ...
```

---

### 4.2 `sigmaborov-reinforce` — 强化版规则 Agent

| 属性 | 值 |
|------|----|
| 来源 | sigmaborov |
| 代码量 | 1895 行 |
| LB 分数 | **958.1** |
| 许可证 | Apache 2.0 |
| 标签 | 世界模型、滚动前向模拟、reinforce 模块 |

在 `sigmaborov-starter` 基础上扩展：

- **WorldModel**：维护到达时间账本，预测 N 步后每颗星球的所有者
- **滚动地平线仿真**：每回合向前模拟多步，评估当前决策的未来收益
- **reinforce 模块**：己方星球受威胁时，自动计算最优增援路线
- **plan 模块**：全局规划资源分配，避免多颗星球重复攻击同一目标

---

### 4.3 `pilkwang-structured` — 多战术评分 Agent 🏆

| 属性 | 值 |
|------|----|
| 来源 | Pilkwang Kim |
| 代码量 | 3107 行 |
| 版本迭代 | 45 个版本 |
| LB 分数 | ~1000 |
| 许可证 | Apache 2.0 |
| 标签 | 规则驱动、世界模型、到达时间预测 |

**三层架构**：

| 层 | 功能 |
|----|------|
| Physics（物理层） | 计算轨道、速度、太阳碰撞规避 |
| WorldModel（世界模型） | 预测星球未来位置，维护到达时间所有权账本 |
| Strategy（策略层） | 综合 8 种战术：reinforce / rescue / recapture / capture / snipe / swarm / crash / salvage |

**核心特色**：
- 到达时间所有权预测，避免重复发送或被截胡
- 多战术评分系统，每回合对所有可执行动作打分选最优
- 专门的防守得分权重
- 严格的计算时间预算控制

---

### 4.4 `tamrazov-starwars` — 强攻型 Agent 🏆 LB MAX 1224

| 属性 | 值 |
|------|----|
| 来源 | Roman Tamrazov |
| 代码量 | 3306 行 |
| LB 最高分 | **1224** |
| 许可证 | Apache 2.0 |
| 标签 | 规则驱动、前向模拟、全面战争、最强规则型 |

**核心策略组合**：

| 战术 | 说明 |
|------|------|
| 太阳规避（Sun-dodge） | 计算绕太阳的绕行航点，避免舰队误入太阳 |
| 多步拦截（Multi-step intercept） | 预测运动星球的未来位置，提前瞄准 |
| 集火（Gang-up） | 发现对手进攻后，多颗星球同步围攻对手薄弱星球 |
| 碰撞利用（Crash-exploit） | 利用多方同时抵达时的碰撞规则占便宜 |
| 全面战争（Total-war endgame） | 剩余 55 回合切换为"放弃防守、全力进攻"模式 |

**调参特色**：`HOSTILE_TARGET_VALUE_MULT = 2.05`（激进攻击倍率），全面战争触发时机为剩余 55 回合（比竞争对手更激进）。

---

### 4.5 `yuriygreben-architect` — 五层架构 Agent

| 属性 | 值 |
|------|----|
| 来源 | Yuriy Greben |
| 代码量 | 3002 行 |
| 许可证 | Apache 2.0 |
| 标签 | 规则驱动、多阶段规划、五层架构 |

```
Config（配置层）
  └─ Physics（物理层）  — 轨道力学、太阳碰撞检测
       └─ WorldModel（世界模型）  — 到达账本、多步所有权预测
            └─ Strategy（策略层）  — 早期/开局/中期/晚期/全面战争 5 阶段
                 └─ Execution（执行层）  — 将策略转化为具体移动指令
```

**多阶段时间轴**：

| 阶段 | 触发条件 | 重点 |
|------|----------|------|
| Early（早期） | 前 40 回合 | 抢占中性星球 |
| Opening（开局） | 40–80 回合 | 扩张产能 |
| Late（中后期） | 剩余 60 回合 | 开始进攻 |
| Total War（全面战争） | 剩余 55 回合 | 全力进攻最弱对手 |

---

### 4.6 `ykhnkf-distance-prioritized` — 距离优先 Agent

| 属性 | 值 |
|------|----|
| 来源 | ykhnkf |
| 代码量 | 3162 行 |
| LB 最高分 | ~1100 |
| 许可证 | Apache 2.0 |

**核心差异**：目标选择时以距离为首要优先级——距离越近权重越高，减少舰队在途中的无效消耗。包含完整的物理模拟和世界模型，与 `pilkwang-structured` 同为完整实现，侧重点不同。

---

### 4.7 `pascal-orbitwork-v14` — 渐进迭代 Agent

| 属性 | 值 |
|------|----|
| 来源 | pascalledesma |
| 代码量 | 3160 行 |
| 版本 | 第 14 版 |
| 许可证 | Apache 2.0 |

从 `pilkwang-structured` 分叉，经过 14 个版本的调参和功能扩充，代表该流派的进一步演化。

---

### 4.8 `kashiwaba-rl` — PPO 强化学习 Agent 🤖

| 属性 | 值 |
|------|----|
| 来源 | YumeNeko |
| 入口代码量 | 91 行 + `src/` 多模块 |
| 网络参数量 | 153k |
| 训练算法 | PPO（近端策略优化） |
| 训练轮次 | 2000 次更新 |
| vs nearest-sniper | **20/20 胜率（100%）** |
| 标签 | 神经网络、强化学习 |

**网络架构**（`PlanetPolicy`，MLP）：

```
输入：
  self_features    — 当前星球自身特征（位置、船数、产能等）
  candidate_features — 候选目标星球特征
  global_features  — 全局游戏状态

三路 MLP 编码器（各 2 层 + ReLU）：
  self_encoder    → hidden_size
  global_encoder  → hidden_size
  candidate_encoder → hidden_size

拼接后：
  target_head  → 每个候选目标的打分（选择攻击目标）
  value_head   → 当前状态价值（PPO 训练用）
```

**决策方式**：每回合对每颗己方星球，对 8 个候选目标打分，确定性地选择最高分目标。

**注意事项**：依赖 PyTorch，每局开始时需要加载模型权重（~100ms），计算量高于规则型 Agent。

---

## 5. 添加自研 Agent

在 `agents/mine/` 目录下创建子目录并实现 `main.py`：

```python
# agents/mine/my_agent/main.py

def agent(obs):
    """
    obs: 游戏观测对象
    return: list of [from_planet_id, angle_radians, num_ships]
    """
    moves = []
    # 实现你的策略逻辑...
    return moves
```

可选的 `agent.yaml` 元数据文件：

```yaml
name: "My Agent"
author: "Your Name"
description: "Agent description"
tags: ["rule-based", "greedy"]
lb_score: null
```

添加后重启后端，Agent 会自动出现在 Web UI 的 Agent 列表中。

---

## 相关页面

- [RL Agent 设计](RL-Agent.md)
- [评估与排行榜](Evaluation.md)
- [开发指南](Development-Guide.md)
