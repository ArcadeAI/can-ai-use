"""Pi: the level-1 reply delay it declares, and how the runner and the cells honour it."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
import urllib.request
from pathlib import Path

import pytest

from mcp_mirror import cells

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_harness_cells import ROOT, _root, _session, _write_run, measure_harness  # noqa: E402


def _pi_module() -> types.ModuleType:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("pi_cli", ROOT / "scripts" / "harnesses" / "pi_cli.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pi_declares_a_level_1_reply_delay() -> None:
    module = _pi_module()
    assert module.LEVEL1_REPLY_DELAY_S == 10
    assert measure_harness.level_1_reply_delay(module) == 10.0


def test_level_1_reply_delay_defaults_to_zero_and_rejects_nonsense() -> None:
    assert measure_harness.level_1_reply_delay(types.SimpleNamespace()) == 0.0
    assert measure_harness.level_1_reply_delay(types.SimpleNamespace(LEVEL1_REPLY_DELAY_S=None)) == 0.0
    with pytest.raises(measure_harness.HarnessError):
        measure_harness.level_1_reply_delay(types.SimpleNamespace(LEVEL1_REPLY_DELAY_S="5"))
    with pytest.raises(measure_harness.HarnessError):
        measure_harness.level_1_reply_delay(types.SimpleNamespace(LEVEL1_REPLY_DELAY_S=-1))


def test_stub_holds_its_reply_for_the_declared_delay() -> None:
    """The stub answers a model request only after the delay; without one it answers at once."""
    body = json.dumps({"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}).encode()

    def ask(stub: measure_harness.ModelStub) -> float:
        request = urllib.request.Request(f"{stub.url}/v1/messages", data=body, method="POST",
                                         headers={"content-type": "application/json"})
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
        return time.monotonic() - started

    with measure_harness.ModelStub("anthropic", reply_delay_s=0.6) as stub:
        assert ask(stub) >= 0.6
        assert stub.brain.reply_delay_s == 0.6
    with measure_harness.ModelStub("anthropic") as stub:
        assert ask(stub) < 0.5
        assert stub.brain.reply_delay_s == 0.0


def test_level_1_cells_cite_the_reply_delay_and_level_2_cells_do_not(tmp_path: Path) -> None:
    root = _root(tmp_path)
    level_1 = [_session(root, "stdio", 1)]
    run = _write_run(tmp_path, level_1)
    manifest = json.loads(run.path.read_text())
    manifest["level_1"]["reply_delay_s"] = 5.0
    manifest["level_1"]["sessions"][0]["reply_delay_s"] = 5.0
    run.path.write_text(json.dumps(manifest))
    run = cells.HarnessRun(run.path)
    held = "the stub held its reply 5 s (LEVEL1_REPLY_DELAY_S in scripts/harnesses/demo.py)"
    startup = cells.HARNESS_RULES["h-lists-tools-at-startup"](run)
    assert any(held in line for line in startup.evidence)
    level_2 = cells.HARNESS_RULES["h-calls-read-only-tool"](run)
    assert not any(held in line for line in level_2.evidence)

    plain = _write_run(tmp_path, [_session(_root(tmp_path / "plain"), "stdio", 1)])
    assert not any("held its reply" in line for line in cells.HARNESS_RULES["h-lists-tools-at-startup"](plain).evidence)
