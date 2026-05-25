# tamrazov-starwars Agent（排行榜约 LB 1224 — 最高分）

## 策略概述

该系列**评分最高的 agent**，在 pilkwang-structured 框架基础上进行了激进性调整：
更高的对敌进攻价值倍率、针对弱势敌方的团战逻辑（gang-up），以及更早触发的全面战争模式。

---

## 框架来源

代码结构与 pilkwang-structured **几乎完全相同**（WorldState、Mission、MissionType 等类设计一致），属于同一套代码演化出的不同参数分支。请先阅读 [pilkwang-structured/README.md](../pilkwang-structured/README.md) 了解基础框架。

本文档仅说明 tamrazov 的**独有特性和参数差异**。

---

## 独有特性 1：Gang-up 群殴机制

**背景**：战场中如果一支敌方舰队与另一支舰队交战后，目标星球防守减弱。此时抢先到达可获大量星球。

**实现**：
- 检测所有星球上预计在未来 N 回合发生的"战斗事件"（两支非同一玩家的舰队同时到达）
- 战斗结束后星球防守被削弱
- 以 `GANG_UP_POST_BATTLE_DELAY = 2` 为延迟，安排己方舰队在战斗结束后 2 回合到达
- 评分额外加 `GANG_UP_VALUE_BONUS`

---

## 独有特性 2：专注最弱敌方

在多人局（4 人局）中，优先集火最弱的敌方玩家：

```python
# 4 人局
WEAK_ENEMY_4P_VALUE_MULT = 1.5    # 针对最弱敌方的目标价值 × 1.5

# 2 人局  
WEAK_ENEMY_2P_VALUE_MULT = 1.25   # 针对较弱一方的目标价值 × 1.25
```

判断"最弱"：按敌方总舰船数或总生产力排名，选取最少的玩家作为主要攻击目标。

---

## 参数对比（vs pilkwang-structured）

| 参数 | pilkwang | tamrazov | 效果 |
|------|---------|----------|------|
| `HOSTILE_TARGET_VALUE_MULT` | 1.85 | **2.05** | 更倾向于打敌方而非中性星 |
| `TOTAL_WAR_REMAINING_TURNS` | 60 | **55** | 更早进入全力输出模式 |
| `EXPOSED_PLANET_VALUE_MULT` | 1.5 | **2.0** | 暴露在外的星球价值更高 |
| `MULTI_ENEMY_PROACTIVE_RATIO` | 0.25 | **0.35** | 4 人局主动防御预留比例更高 |
| `GANG_UP_POST_BATTLE_DELAY` | 不存在 | **2** | 战场碰撞后 2 回合跟进 |

---

## 游戏阶段触发时机

```
opening  (step < 80)  ──── 标准开局扩张
early    (step < 40)  ──── 抢最近中性星
全面战争              ──── 剩余 ≤ 55 回合（更早！）
very_late             ──── 剩余 ≤ 25 回合（全力清场）
```

---

## 决策执行流程

（同 pilkwang-structured，请参见该 README）

唯一区别是在**任务评分阶段**加入以下额外乘数：
1. 目标是当前最弱敌方的星球 → × `WEAK_ENEMY_xP_VALUE_MULT`
2. 该目标刚经历战斗即将出现防守空窗 → + `GANG_UP_VALUE_BONUS`
3. 目标是"暴露"星球（周围无友军支援）→ × 2.0

---

## 总结

tamrazov 相比 pilkwang 的改进思路是：
- **更早更猛地打击敌方**（敌方目标价值更高、全面战争更早触发）
- **集中火力**（专打最弱敌方，避免兵力分散）
- **利用战场机遇**（gang-up 机制，趁敌虚弱补刀）
