# JAX Transformer Policy Architecture

> B = num_envs · P = max_planets = 45 · K = max_fleet_tokens · H = hidden_size = 128

---

## 架构图

```text
EnvState
  |
  v
encode_batch
  |
  +-- planet_features -------------------> self MLP --------+
  |                                                       |
  +-- global_features -----------------> global MLP ------+--> concat -> context
  |                                                       |              |
  +-- map_features --------------------> CNN (optional) --+              |
  |                                                                      |
  +-- candidate_features --------------------------------------+          |
  |                                                            |          |
  |   split into two views                                     |          |
  |                                                            |          |
  |   shared planet features                                   |          |
  |     owner / ships / production / position / comet ...      |          |
  |       |                                                    |          |
  |       v                                                    |          |
  |   planet token MLP                                         |          |
  |       |                                                    |          |
  |       +---- [CLS + planet tokens + fleet tokens]           |          |
  |                              |                             |          |
  +-- fleet_features ------------+                             |          |
                                  v                            |          |
                             Transformer                       |          |
                           runs once/state                     |          |
                                  |                            |          |
                    +-------------+-------------+              |          |
                    |                           |              |          |
                    v                           v              |          |
                 cls_out                  planet_out           |          |
                    |                           |              |          |
                    |                           +--> gather source_ctx --+
                    |                           |
                    |                           +--> base target keys ----+
                    |                                                   |
                    v                                                   |
              global value                                               |
              broadcast to sources                                       |
                                                                        |
  relative target features                                               |
  source-to-target dx/dy/dist, proto, eta, relative ships                 |
       |                                                                 |
       v                                                                 |
  relative target MLP ---------------------------------------------------+
       |                                                                 |
       v                                                                 |
  final target keys = base target keys + relative target keys             |
                                                                        |
context ----------------------------------------------------------------+
  |                                                                     |
  +--> Fire head   -> fire_logits                                       |
  +--> Pointer     -> target_logits  using final target keys            |
  +--> Amount head -> amount_logits  using selected target key          |
  +--> Value head  -> value from cls_out                                |
```

## 简单伪代码

```python
def forward(encoded, selected_targets):
    # 1. 拆特征
    self_f = encoded.planet_features
    global_f = encoded.global_features
    cand_f = encoded.candidate_features
    fleet_f = encoded.fleet_features
    map_f = encoded.map_features

    # 2. 每个 state 只跑一次 Transformer
    planet_shared = remove_source_relative_fields(cand_f[:, 0])
    planet_tokens = MLP(planet_shared)
    fleet_tokens = MLP(fleet_f[:, 0])
    tokens = concat([cls_token, planet_tokens, fleet_tokens])

    x = Transformer(tokens)
    cls_out = x[:, 0]
    planet_out = x[:, 1 : 1 + num_planet_slots]

    # 3. 构建每个 source 的上下文
    self_h = MLP(self_f)
    global_h = MLP(global_f)
    source_ctx = gather_source_embedding(planet_out, cand_f)
    cls_per_source = broadcast(cls_out, each_source=True)
    map_h = CNN(map_f)  # optional

    context = concat([self_h, global_h, source_ctx, cls_per_source, map_h])

    # 4. 构建每个 source-target 的 key
    relative_f = keep_only_source_relative_fields(cand_f)
    relative_h = MLP(relative_f)
    base_keys = Dense(planet_out)
    target_keys = base_keys[:, None] + relative_h

    # 5. 三个 actor head + 一个 critic head
    fire_logits = FireHead(context)
    target_logits = dot(Dense(context), target_keys)

    selected_key = gather(target_keys, selected_targets)
    amount_logits = AmountHead(concat([context, selected_key]))

    value = broadcast(ValueHead(cls_out), each_source=True)
    return fire_logits, target_logits, amount_logits, value
```

---

## 关键设计说明

### Transformer 只跑一次
`candidate_features[:,0]` 取第 0 行（所有 source 共享同一份 state-level planet 信息），
清零 source-relative dims 后进 Transformer。fleet 同理取 `[:,0]`。
这样 **B 个 state 各跑 1 次**，而不是 B×P 次。

### Relative Keys 保证 per-source 精度
Transformer 输出的 `planet_out` 是全局信息，不含"这颗源星球到目标有多远"。
因此 pointer head 的 keys 设计为：
```
keys = Dense(planet_out)  ← 全局关系
     + MLP(relative_dims) ← 每个 source 的距离/方向/proto/eta
```
两路叠加后，target logits 既有全局上下文又有精确距离感知。

### Value 来自 CLS
`cls_out` 通过 self-attention 聚合了所有星球+舰队信息，天然适合做全局状态价值估计。
把同一个 `value_global [B]` broadcast 成 `[B, P]` 只是为了和 PPO 张量形状 `[T,B,P]` 对齐。

### 路径分发
```
planet_features.ndim == 3  →  _forward_transformer_grouped  ← 训练主路径
planet_features.ndim == 2  →  _forward_transformer           ← 旧 per-source 路径（慢）
use_transformer = False  →  _forward_legacy                ← 旧 MLP + pool + cross-attn
```

---

## Feature 维度速查

| 字段 | 形状 | 说明 |
|---|---|---|
| `planet_features` | `[B, P, 18]` | 源星球自身状态 + inbound 信息 |
| `candidate_features` | `[B, P, P+1, 26]` | 所有候选目标，slot 0 为 no-op |
| `global_features` | `[B, P, 11]` | 全局态势，每 source 重复 |
| `fleet_features` | `[B, P, K, 18]` | 舰队 token，per-state 共享（P 行相同） |
| `fleet_mask` | `[B, P, K]` | 舰队存活 mask |
| `map_features` | `[B, P, 6, 8, 8]` | 8×8 空间小地图，6 通道 |
| `candidate_mask` | `[B, P, P+1]` | 合法 target slot |
| `amount_mask` | `[B, P, P+1, 8]` | 合法派兵档位 |
| `source_mask` | `[B, P]` | 我方存活星球（参与 value loss） |
| `policy_mask` | `[B, P]` | 实际执行/真实 no-op（参与 policy loss） |

## 各特征含义

### `planet_features`：源星球自身特征

每个 source 星球一行，描述“我是谁、我现在强不强、我正承受什么压力”。

| idx | 含义 |
|---:|---|
| 0 | 常量 1，表示这是一个 planet/source |
| 1 | source x 坐标 |
| 2 | source y 坐标 |
| 3 | source 半径 |
| 4 | source 当前船数 |
| 5 | source 生产力 |
| 6 | source 是否绕太阳旋转 |
| 7 | source 到太阳距离 |
| 8 | 我方星球数量 |
| 9 | 敌方星球数量 |
| 10 | 我方星球总船数 |
| 11 | 敌方舰队船数 - 我方舰队船数 |
| 12 | 正飞向该 source 的我方船数 |
| 13 | 正飞向该 source 的敌方船数 |
| 14 | 我方 inbound 最短到达时间 |
| 15 | 我方 inbound 平均到达时间 |
| 16 | 敌方 inbound 最短到达时间 |
| 17 | 敌方 inbound 平均到达时间 |

### `candidate_features`：候选目标特征

每个 source 对所有 target 各有一组特征。slot 0 是 no-op，slot 1..P 是星球。

| idx | 含义 |
|---:|---|
| 0 | 是否 no-op |
| 1 | target 是否存活 |
| 2 | target 是否当前 source 自己 |
| 3 | target 是否中立 |
| 4 | target 是否友方非 source |
| 5 | target 是否敌方 |
| 6 | target x 坐标 |
| 7 | target y 坐标 |
| 8 | target 相对 source 的 dx |
| 9 | target 相对 source 的 dy |
| 10 | source 到 target 的距离 |
| 11 | target 半径 |
| 12 | target 当前船数 |
| 13 | target 生产力 |
| 14 | target 是否绕太阳旋转 |
| 15 | target 是否彗星 |
| 16 | target 船数相对 source 船数的比例 |
| 17 | source 当前船数 |
| 18 | 正飞向 target 的我方船数 |
| 19 | 正飞向 target 的敌方船数 |
| 20 | 我方 inbound 到 target 的最短时间 |
| 21 | 我方 inbound 到 target 的平均时间 |
| 22 | 敌方 inbound 到 target 的最短时间 |
| 23 | 敌方 inbound 到 target 的平均时间 |
| 24 | 从该 source 看 target 的 proto 评分 |
| 25 | 从该 source 到 target 的最快到达时间 |

其中 `2, 8, 9, 10, 16, 17, 24, 25` 是 source-relative 特征。它们不进 Transformer，而是通过 relative target MLP 加到 pointer keys 上。

### `global_features`：全局局势特征

每个 source 都重复同一份全局态势。

| idx | 含义 |
|---:|---|
| 0 | 当前回合进度 |
| 1 | 星球角速度 |
| 2 | 我方星球数量 |
| 3 | 敌方星球数量 |
| 4 | 中立星球数量 |
| 5 | 我方星球总船数 |
| 6 | 敌方星球总船数 |
| 7 | 我方舰队总船数 |
| 8 | 敌方舰队总船数 |
| 9 | 我方总生产力 |
| 10 | 敌方总生产力 |

### `fleet_features`：舰队 token 特征

每个 state 选 top-K 支舰队作为 Transformer token。当前按大船队优先排序。

| idx | 含义 |
|---:|---|
| 0 | 舰队是否我方 |
| 1 | 舰队 x 坐标 |
| 2 | 舰队 y 坐标 |
| 3 | 舰队角度 cos |
| 4 | 舰队角度 sin |
| 5 | 舰队船数 |
| 6 | 舰队速度 |
| 7 | 已发射时长（由当前坐标、出发星球坐标、出发半径和舰队速度估算） |
| 8 | 出发星球 x 坐标 |
| 9 | 出发星球 y 坐标 |
| 10 | 出发星球是否我方 |
| 11 | 出发星球当前船数 |
| 12 | 舰队距离出发星球的距离 |
| 13 | 距离最近敌方星球的距离 |
| 14 | 距离最近中立星球的距离 |
| 15 | 距离最近我方星球的距离 |

### `map_features`：CNN 小地图

8×8 网格，6 个通道。

| channel | 含义 |
|---:|---|
| 0 | 当前 source 星球 |
| 1 | 友方星球 |
| 2 | 敌方星球 |
| 3 | 中立星球 |
| 4 | 友方舰队 |
| 5 | 敌方舰队 |

### Mask 特征

| 字段 | 含义 |
|---|---|
| `candidate_mask` | 哪些 target slot 合法。slot 0 no-op 永远合法。 |
| `amount_mask` | 对每个 target，哪些发船档位合法。 |
| `fleet_mask` | top-K fleet token 哪些是真实舰队。 |
| `source_mask` | 哪些 source 是我方存活星球。 |
| `policy_mask` | 哪些 source 的采样动作可以参与 policy loss。物理层过滤掉的无效发射会被排除。 |

---

## 当前配置

```yaml
model:
  hidden_size: 128
  use_cnn: true
  use_transformer: true
  num_transformer_layers: 1
  num_heads: 2
  max_fleet_tokens: 16
```

速度满足预期后可升级到：

```yaml
  num_transformer_layers: 2
  num_heads: 4
  max_fleet_tokens: 32
```
