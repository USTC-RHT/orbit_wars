# sigmaborov-reinforce Agent（排行榜约 LB 958）

## 策略概述

**sigmaborov 系列的进阶版**（代号 v6），在 starter 版的物理模型基础上，加入了完整的任务系统，包括多源群攻、增援任务、崩溃利用和主动防御。

---

## 与 starter 版的关键新增功能（v6 特性）

| 新增 | 说明 |
|------|------|
| `reinforce` 任务 | 主动向己方星球输送援军，防止被偷袭 |
| `3-source swarm` | 协调 3 个星球同时到达目标发起群攻 |
| `crash_exploit` | 利用敌方两队互相碰撞后的防守空窗期（仅 4 人局） |
| 多敌主动防御 | 多个敌方威胁时，预先部署防御力量 |

---

## 核心物理模型（同 starter）

- 旋转星球位置预测（`predict_planet_position`）
- 彗星路径追踪（`predict_comet_position`）
- 舰队速度模型（对数曲线）
- 太阳路径规避（线段与圆相交检测）
- 迭代到达时间估算（4 次迭代）

---

## 世界状态（World 对象）

`World` 在每回合构建：

- 向前预测 110 回合的舰队到达事件
- `ships_needed_to_capture(target, arrival_turn)` — 计算在指定时刻占领目标所需舰船
- `reinforcement_needed_for(planet, arrival_turn)` — 计算增援己方星球所需兵力
- `_compute_defense_buffers()` — 为每个己方星球设置防御储备（保留 N 艘不出征）
- `source_attack_left(src, spent)` — 剩余可用进攻兵力

---

## 任务类型

### `capture`（单源占领）
从一个星球出发，占领中性或敌方星球：
```
score = target_value(turns, mission) / (turns * turn_weight + ships_needed)
```

### `snipe`（狙击截击）
预测敌方舰队的目标，在对方到达的同时到达，混战互消：
- 同步时刻 = 敌方舰队预计到达时刻
- 己方舰队需要在同一时间窗口内到达

### `swarm`（2-源 / 3-源群攻）
协调 2~3 个星球同步到达同一目标：
- 寻找 ETA 相差 ≤ 2 回合的多颗源星球
- 组合兵力超过目标防守 + 裕量
- 3-源群攻需目标有 ≥ 20 艘舰船

### `reinforce`（增援己方）
- 扫描每个己方星球未来 40 回合内是否会被攻破
- 若需要，从附近己方星球派遣援军（最多 22 回合路程）
- 评分考虑增援能保住的生产价值

### `crash_exploit`（4 人局崩溃利用）
- 检测两支属于不同玩家的舰队是否将在同一时刻到达某星球
- 两队互消后星球防守归零
- 趁机在碰撞后 1 回合到达，轻松占领

---

## 决策执行流程

```
1. 构建 World（物理模型 + 防御储备 + 可用兵力）
2. 计算宏观局势（我方 vs 敌方生产力比，判断 ahead/behind/finishing）
3. 生成所有任务候选（capture + snipe + swarm + reinforce + crash_exploit）
4. 按评分排序，贪心分配资源
5. 翻译为 moves
```

---

## 与 pilkwang-structured 的区别

sigmaborov-reinforce 是该任务系统的**早期版本**，相比 pilkwang 缺少：
- 独立的主动 `Defense` 任务（pilkwang 有专门防御评分）
- `recapture`（收复失地）任务
- `rear_source`（后方远程增援）
- 完整的软时间截止机制（pilkwang 有 0.82s 截止控制）
- `rescue`（濒死撤离）任务
