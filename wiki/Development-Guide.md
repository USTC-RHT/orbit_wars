# 开发指南

> **所属页面**：[Home](Home.md) > 开发指南

---

## 1. 前置条件

| 组件 | 版本要求 |
|------|----------|
| Python | 3.12+ |
| pnpm | 任意（`npm i -g pnpm` 安装） |
| Docker | 20.10+（Docker 方案可选） |
| CUDA | 可选，RL 训练时 GPU 加速用 |

---

## 2. 快速启动

### 方案 A：Docker（推荐）

```bash
git clone https://github.com/automatylicza/orbit-wars-lab.git
cd orbit-wars-lab
docker compose up
```

打开 http://localhost:6001，完成。

**首次构建**约 3–5 分钟（拉取 PyTorch CPU 镜像）。后续 `up` 秒级启动。

**端口冲突**：

```bash
PORT=7001 docker compose up
```

**macOS / 非标准 UID**（确保容器写入文件属于当前用户）：

```bash
cp .env.example .env
echo "UID=$(id -u)" > .env
echo "GID=$(id -g)" >> .env
```

---

### 方案 B：本地原生开发（迭代更快）

```bash
git clone https://github.com/automatylicza/orbit-wars-lab.git
cd orbit-wars-lab
bash scripts/dev.sh
```

脚本自动完成：
1. 创建 Python `.venv`
2. 安装 Python 依赖
3. 安装前端依赖（`pnpm install`）
4. 启动后端（`:8000`，`--reload`）+ 前端（`:6001`，HMR）

打开 http://localhost:6001。

---

## 3. 环境配置（RL 训练）

本项目使用独立的 Conda 环境进行 RL 训练：

```bash
conda activate orbit-wars-lab
# 或使用绝对路径
/home/dministrator/miniconda/envs/orbit-wars-lab/bin/python
```

主要依赖：

```
kaggle-environments>=1.28.0
torch
numpy
pyyaml
```

安装：

```bash
pip install -r orbit-wars-lab/requirements.txt
```

---

## 4. 项目结构速查

```
kaggle/
│
├── orbit-wars-lab/               本地赛事平台
│   ├── orbit_wars_app/           FastAPI 后端
│   │   ├── main.py               FastAPI 应用入口
│   │   ├── api.py                API 路由定义
│   │   ├── tournament.py         赛事调度（ProcessPoolExecutor）
│   │   ├── match.py              单场比赛执行
│   │   ├── trueskill_store.py    TrueSkill 评分存储
│   │   ├── discovery.py          Agent 目录扫描
│   │   ├── replay_store.py       Replay 读写
│   │   ├── kaggle_auth.py        Kaggle API Token
│   │   └── kaggle_submissions.py Kaggle LB 拉取
│   │
│   ├── viewer/                   Vite TypeScript 前端
│   │   ├── src/
│   │   │   ├── main.ts           SPA 入口
│   │   │   ├── router.ts         Hash 路由
│   │   │   ├── api.ts            类型化 API 调用层
│   │   │   └── views/            各页面组件
│   │   └── vite.config.ts        Vite 配置（含 /api 代理）
│   │
│   ├── agents/                   Agent 仓库
│   │   ├── baselines/            官方基准 Agent
│   │   ├── external/             社区高分 Agent
│   │   └── mine/                 自研 Agent（放这里）
│   │
│   └── runs/                     历史赛事数据
│       ├── trueskill.json        TrueSkill 持久化评分
│       └── {run-id}/             单次赛事目录
│
└── rl-agent/                     自研 PPO RL Agent
    ├── src/                      核心模块
    ├── train.py                  训练入口
    ├── eval.py                   评估入口
    └── outputs/                  训练输出（checkpoint、metrics）
```

---

## 5. 开发流程：编写自定义 Agent

### 5.1 创建 Agent 目录

```bash
mkdir -p orbit-wars-lab/agents/mine/my_strategy
```

### 5.2 实现 `main.py`

```python
# orbit-wars-lab/agents/mine/my_strategy/main.py
import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

def agent(obs):
    """
    obs: 游戏观测对象（支持 dict 或 AttrDict）
    return: list of [from_planet_id, angle_radians, num_ships]
    """
    moves = []
    
    # 解析观测
    player = obs.get("player") if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets") if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]
    
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    
    if not targets or not my_planets:
        return moves
    
    for mine in my_planets:
        if mine.ships < 10:
            continue  # 船数不足时跳过
        
        # 选择最近目标（示例：贪心策略）
        nearest = min(targets, key=lambda t: math.hypot(t.x - mine.x, t.y - mine.y))
        ships_to_send = nearest.ships + 1
        
        if mine.ships >= ships_to_send:
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([mine.id, angle, ships_to_send])
    
    return moves
```

### 5.3 添加元数据（可选）

```yaml
# orbit-wars-lab/agents/mine/my_strategy/agent.yaml
name: "My Strategy"
author: "Your Name"
description: "Greedy nearest-planet strategy with ship threshold"
tags: ["rule-based", "greedy"]
lb_score: null
```

### 5.4 验证

重启后端后，在 Web UI 的 Agent 列表（`#/agents`）中确认 Agent 出现。  
运行一场 Quick Match 验证其行为。

---

## 6. RL Agent 训练流程

### 6.1 基础训练

```bash
cd /home/dministrator/kaggle
python RL/train.py --config RL/default_cfg.yaml
```

### 6.2 自定义配置

复制并修改配置文件：

```bash
cp RL/default_cfg.yaml RL/my_cfg.yaml
# 编辑 my_cfg.yaml
python RL/train.py --config RL/my_cfg.yaml --run-name my_experiment
```

**关键配置项**：

```yaml
# my_cfg.yaml
ppo:
  total_updates: 2000       # 总更新轮次
  num_envs: 8               # 并行环境数
  rollout_steps: 128        # 每次 rollout 步数
env:
  ship_bucket_count: 8      # 派兵档位数
opponent: nearest           # 训练对手：random / nearest / self
device: auto                # 自动选择 CPU / CUDA
reward:
  terminal_weight: 1.0      # 终局胜负权重（必须开启）
  launch_penalty: 0.002     # 发射惩罚（抑制无效发射）
  production_weight: 0.05   # 产能 shaping
  planet_weight: 0.02       # 星球数 shaping
model:
  hidden_size: 128
  use_cnn: false            # 是否启用 CNN 地图分支
```

### 6.3 部署到 orbit-wars-lab

训练完成后，将 RL Agent 链接到 Agent 仓库：

```bash
# 创建软链接或复制
ln -s /home/dministrator/kaggle/RL \
      /home/dministrator/kaggle/orbit-wars-lab/agents/mine/my-rl-agent
```

---

## 7. 测试与验证

### 7.1 运行 Python 单元测试

```bash
cd orbit-wars-lab
python -m pytest tests/ -v
```

### 7.2 快速对战验证（CLI 方式）

```bash
# 使用 kaggle-environments 直接运行比赛
python -c "
from kaggle_environments import make
env = make('orbit_wars', debug=True)
env.run(['random', 'starter'])
final = env.steps[-1]
for i, s in enumerate(final):
    print(f'Player {i}: reward={s.reward}, status={s.status}')
"
```

### 7.3 查看 API 健康状态

```bash
curl http://localhost:8000/api/health
```

---

## 8. 常见问题

### Q: 后端报 `ImportError: kaggle_environments`

```bash
pip install --upgrade "kaggle-environments>=1.28.0"
```

### Q: 前端无法访问后端 API（CORS 错误）

检查 `viewer/vite.config.ts` 中的代理配置：

```typescript
server: {
  proxy: {
    "/api": "http://localhost:8000",
  }
}
```

### Q: Docker 容器中文件权限问题

创建 `.env` 文件并设置 UID/GID：

```bash
echo "UID=$(id -u)" > .env
echo "GID=$(id -g)" >> .env
```

### Q: RL 训练时 CUDA out of memory

减少 `num_envs` 或 `rollout_steps`，或切换到 CPU：

```bash
python rl-agent/train.py --config rl-agent/default_cfg.yaml --device cpu
```

### Q: Agent 在 Web UI 中不出现

1. 确认目录结构为 `agents/{baselines|external|mine}/{name}/main.py`
2. 确认 `main.py` 中存在 `def agent(obs):` 函数
3. 重启后端

---

## 相关页面

- [系统架构](System-Architecture.md)
- [Agent 全览](Agent-Zoo.md)
- [RL Agent 设计](RL-Agent.md)
- [评估与排行榜](Evaluation.md)
