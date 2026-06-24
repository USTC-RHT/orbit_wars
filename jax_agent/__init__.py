"""Orbit Wars JAX agent 包。

源码按工作流物理分层：

- ``common``：固定形状环境、特征编码器、模型、归一化。
- ``preprocessing``：raw replay zips 和 feature-row 数据构建。
- ``bc``：行为克隆训练、checkpoint runtime 和 BC 评估。
- ``rl``：PPO/self-play 训练工具。
- ``onnx``：ONNX 导出和 NumPy 推理辅助。
- ``evaluation``：打包/外部对手适配器。
"""
