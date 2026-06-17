# Orbit Wars — 竞赛工程 Wiki

> **竞赛主页**：[Kaggle · Orbit Wars](https://www.kaggle.com/competitions/orbit-wars)  
> **本 Wiki 适用范围**：从比赛角度完整介绍本仓库的设计、实现与使用方法。  
> **文档日期**：2026-05-09  
> **文档语言**：中文（简体）

---

## 目录结构

| 页面 | 说明 |
|------|------|
| [竞赛简介](Competition-Overview.md) | 比赛背景、规则摘要、评分标准 |
| [游戏规则详解](Game-Rules.md) | 棋盘、星球、舰队、彗星、战斗、胜负判定 |
| [系统架构](System-Architecture.md) | 本地赛事平台前后端架构全图 |
| [Agent 全览](Agent-Zoo.md) | 11 个 Agent 横向对比与策略解析 |
| [RL Agent 设计](RL-Agent.md) | PPO 强化学习 Agent 的特征、网络与训练 |
| [Behavioral Cloning 训练要点](BC-Training-Notes.md) | 基于强 replay 做监督学习预训练的原则、标签、loss 与避坑 |
| [Kaggle BC 讨论帖中文翻译](BC-Discussion-Translation.md) | 用户提供的 BC / imitation learning 讨论帖原文翻译 |
| [模仿学习落地计划](Imitation-Learning-Plan.md) | 从 replay 数据集到 BC checkpoint，再接 PPO/self-play 的实施路线 |
| [开发指南](Development-Guide.md) | 环境搭建、快速启动、自定义 Agent |
| [评估与排行榜](Evaluation.md) | TrueSkill 评分、本地锦标赛、提交流程 |

---

## 快速导航

```
kaggle/
├── Introduction.md          ← 比赛游戏规则原文
├── Starter_Kit.md           ← 官方入门示例笔记本
├── agents-overview.md       ← Agent 详细分析
├── architecture.md          ← 系统架构详解
├── OW-proto.md              ← Proto-V15 竞赛 Agent 源码与说明
├── orbit_wars.py            ← 独立工具脚本
├── orbit-wars-lab/          ← 本地赛事平台（后端 + 前端）
│   ├── orbit_wars_app/      ← FastAPI 后端
│   ├── viewer/              ← Vite TypeScript 前端
│   ├── agents/              ← Agent 仓库（baselines / external / mine）
│   └── runs/                ← 历史赛事数据
└── rl-agent/                ← 自研 PPO RL Agent
    ├── src/                 ← 模型、特征、训练核心模块
    ├── train.py             ← 训练入口
    ├── eval.py              ← 评估入口
    └── main.py              ← Kaggle 推理入口
```
