"""Exercise the runtime/file boundary; real Git semantics have separate tests."""
import json
import shlex
from unittest.mock import MagicMock

import pandas as pd
import pytest

from evaluation.benchmarks.swe_bench.run_infer import (
    _has_existing_result,
    _prepare_portable_patch,
    complete_runtime,
)
from evaluation.utils.shared import EvalException
from openhands.events.action import CmdRunAction, FileReadAction
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    FileReadObservation,
)
from openhands.runtime.base import Runtime

PREFIX = b"diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+"
INSTANCE = pd.Series({"repo": "example/repo", "version": "1", "base_commit": "base"})


@pytest.fixture
def patch_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "evaluation.benchmarks.swe_bench.run_infer.DATASET_TYPE", "nv-internal-1"
    )
    runtime = MagicMock(spec=Runtime)
    runtime._swe_patch_context = str(tmp_path / "portable")
    overrides = {}

    def run_action(action):
        if isinstance(action, CmdRunAction):
            assert action.command != "cat patch.diff", "Raw patch reached terminal"
            assert "base64" not in action.command
            if "normalize" in shlex.split(action.command):
                if "normalize" in overrides:
                    return overrides["normalize"]
                # Real normalization is qualified by helper and pipeline tests.
                return CmdOutputObservation(content="", command=action.command, exit_code=0)
            return CmdOutputObservation(content="", command=action.command, exit_code=0)
        assert isinstance(action, FileReadAction)
        if action.path in overrides:
            return overrides[action.path]
        if action.path == getattr(runtime, '_swe_patch_context', '') + "/patch.diff":
            return FileReadObservation(content="portable UTF-8 patch\n", path=action.path)
        assert action.path == "patch.diff"
        try:
            content = (tmp_path / action.path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ErrorObservation("File could not be decoded as utf-8: patch.diff.")
        return FileReadObservation(content=content, path=action.path)

    runtime.run_action.side_effect = run_action
    return runtime, tmp_path / "patch.diff", overrides


@pytest.mark.parametrize("line", ["ASCII", "café", "\x1bPsynthetic"])
def test_utf8_patch_unchanged(patch_runtime, line):
    runtime, patch_file, _ = patch_runtime
    patch = PREFIX + line.encode("utf-8") + b"\n"
    patch_file.write_bytes(patch)
    assert complete_runtime(runtime, INSTANCE) == {
        "git_patch": patch.decode("utf-8").rstrip("\n")
    }
    assert not any(
        isinstance(c.args[0], CmdRunAction) and "normalize" in c.args[0].command
        for c in runtime.run_action.call_args_list
    )


@pytest.mark.parametrize("payload", [b"\xff\x1bPsynthetic\n", b"\xff\r\n", b"\xff\n"])
def test_invalid_utf8_uses_portable_file_and_legacy_schema(patch_runtime, payload):
    runtime, patch_file, _ = patch_runtime
    patch_file.write_bytes(PREFIX + payload)
    assert complete_runtime(runtime, INSTANCE) == {"git_patch": "portable UTF-8 patch\n"}
    calls = [c.args[0] for c in runtime.run_action.call_args_list]
    normalization = next(c for c in calls if isinstance(c, CmdRunAction) and "normalize" in c.command)
    assert normalization.hard_timeout == 90
    assert shlex.split(normalization.command)[-2:] == ["--output", runtime._swe_patch_context + "/patch.diff"]
    assert isinstance(calls[-1], FileReadAction)


def test_missing_pre_agent_context_is_reported(patch_runtime):
    runtime, patch_file, _ = patch_runtime
    patch_file.write_bytes(PREFIX + b"\xff\n")
    del runtime._swe_patch_context
    with pytest.raises(EvalException, match="Missing pre-agent"):
        complete_runtime(runtime, INSTANCE)


def test_normalization_failure_is_not_an_empty_patch(patch_runtime):
    runtime, patch_file, overrides = patch_runtime
    patch_file.write_bytes(PREFIX + b"\xff\n")
    overrides["normalize"] = CmdOutputObservation(content="invalid patch", command="normalize", exit_code=1)
    with pytest.raises(EvalException, match="Failed to make UTF-8 Git patch"):
        complete_runtime(runtime, INSTANCE)


def test_portable_file_read_error_is_reported(patch_runtime):
    runtime, patch_file, overrides = patch_runtime
    patch_file.write_bytes(PREFIX + b"\xff\n")
    overrides[runtime._swe_patch_context + "/patch.diff"] = ErrorObservation("File not found")
    with pytest.raises(EvalException, match="Failed to read portable git patch"):
        complete_runtime(runtime, INSTANCE)


def test_other_read_error_does_not_start_recovery(patch_runtime):
    runtime, _, overrides = patch_runtime
    overrides["patch.diff"] = ErrorObservation("File not found")
    with pytest.raises(EvalException, match="Failed to read git patch"):
        complete_runtime(runtime, INSTANCE)
    assert isinstance(runtime.run_action.call_args.args[0], FileReadAction)


@pytest.mark.parametrize("dataset,policy", [("R2E-Gym", "fix"), ("SWE-bench", "nowarn")])
def test_preparation_copies_helper_and_captures_policy(patch_runtime, monkeypatch, dataset, policy):
    runtime, _, _ = patch_runtime
    monkeypatch.setattr("evaluation.benchmarks.swe_bench.run_infer.DATASET_TYPE", dataset)
    _prepare_portable_patch(runtime, INSTANCE)
    runtime.copy_to.assert_called_once()
    assert runtime.copy_to.call_args.args[0].endswith("/portable_patch.py")
    command = shlex.split(runtime.run_action.call_args.args[0].command)
    assert command[-2:] == ["--whitespace", policy]
    assert command[command.index("--base") + 1] == "base"
    assert runtime._swe_patch_context.startswith("/tmp/openhands-patch-")


def test_saved_portable_patch_is_recognized(tmp_path):
    completions = tmp_path / "llm_completions" / "example"
    completions.mkdir(parents=True)
    (completions / "completion.json").write_text("{}")
    result = {"instance_id": "example", "test_result": {"git_patch": "GIT binary patch\nliteral 1\n"}}
    (tmp_path / "output.jsonl").write_text(json.dumps(result) + "\n")
    assert _has_existing_result(str(tmp_path), "example") == (True, result)
