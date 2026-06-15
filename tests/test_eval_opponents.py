import pytest

pytest.importorskip("jax")

from jax_agent.eval_opponents.opponents import ExternalScriptJaxEvalOpponent, default_eval_opponents


def test_ckpt_eval_prefers_packaged_onnx_agent(tmp_path):
    agent_dir = tmp_path / "agents" / "mine" / "fixed-pool-tgtmask-0526-ckpt48900-onnx-producer-hybrid"
    agent_dir.mkdir(parents=True)
    (agent_dir / "main.py").write_text("def agent(obs):\n    return []\n", encoding="utf-8")

    opponents = default_eval_opponents(tmp_path, enabled_names=["ckpt_48900"])

    assert len(opponents) == 1
    assert isinstance(opponents[0], ExternalScriptJaxEvalOpponent)
    assert opponents[0].metric_prefix == "ckpt_48900"
    assert opponents[0].agent_dir == agent_dir.resolve()


def test_ckpt_eval_requires_packaged_onnx_agent_under_repo(tmp_path):
    with pytest.raises(FileNotFoundError, match="Packaged ONNX opponent not found under rl-agent"):
        default_eval_opponents(tmp_path, enabled_names=["ckpt_48900"])
