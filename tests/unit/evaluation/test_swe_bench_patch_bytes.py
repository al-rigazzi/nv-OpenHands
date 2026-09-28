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


@pytest.mark.parametrize("line", [
    "ASCII", "café", "\x1bPsynthetic", "CR\rmiddle", "VT\vmiddle",
    "FF\fmiddle", "NEL\x85middle", "LS\u2028middle", "PS\u2029middle",
])
def test_utf8_patch_unchanged(patch_runtime, line):
    runtime, patch_file, _ = patch_runtime
    patch = PREFIX + line.encode("utf-8") + b"\n"
    patch_file.write_bytes(patch)
    assert complete_runtime(runtime, INSTANCE) == {
        "git_patch": "\n".join(patch.decode("utf-8").splitlines())
    }
    assert not any(
        isinstance(c.args[0], CmdRunAction) and "normalize" in c.args[0].command
        for c in runtime.run_action.call_args_list
    )



def test_utf8_binary_notice_keeps_original_filter(patch_runtime):
    runtime, patch_file, _ = patch_runtime
    text = PREFIX.decode() + "changed\n"
    patch_file.write_text(
        text + "diff --git a/image b/image\nBinary files a/image and b/image differ\n",
        encoding="utf-8",
    )
    assert complete_runtime(runtime, INSTANCE) == {"git_patch": text.rstrip("\n")}
    assert not any(
        isinstance(c.args[0], CmdRunAction) and "normalize" in c.args[0].command
        for c in runtime.run_action.call_args_list
    )


@pytest.mark.parametrize("payload", [b"\xff\x1bPsynthetic\n", b"\xff\r\n", b"\xff\n"])
def test_invalid_utf8_uses_portable_file_and_legacy_schema(patch_runtime, payload):
    runtime, patch_file, _ = patch_runtime
    patch_file.write_bytes(PREFIX + payload)
    assert complete_runtime(runtime, INSTANCE) == {"git_patch": "portable UTF-8 patch"}
    calls = [c.args[0] for c in runtime.run_action.call_args_list]
    normalization = next(c for c in calls if isinstance(c, CmdRunAction) and "normalize" in c.command)
    assert normalization.hard_timeout == 600
    command = shlex.split(normalization.command)
    directory = runtime._swe_patch_context
    assert command == [
        "python", directory + "/portable_patch.py", "normalize",
        directory + "/state", "patch.diff", directory + "/patch.diff", "600",
    ]
    assert isinstance(calls[-1], FileReadAction)
    assert calls[-1].hard_timeout == 600


@pytest.mark.parametrize("failed_attempts", range(5))
@pytest.mark.parametrize("portable", [False, True])
def test_extraction_preserves_original_retry_timeouts(
    patch_runtime, monkeypatch, failed_attempts, portable
):
    runtime, patch_file, _ = patch_runtime
    patch_file.write_bytes(PREFIX + (b"\xff" if portable else b"text") + b"\n")
    original = runtime.run_action.side_effect
    attempts = 0

    def run_action(action):
        nonlocal attempts
        if isinstance(action, CmdRunAction) and action.command.startswith("git diff "):
            attempts += 1
            if attempts <= failed_attempts:
                return CmdOutputObservation(content="retry", command=action.command, exit_code=1)
        return original(action)

    runtime.run_action.side_effect = run_action
    monkeypatch.setattr(
        "evaluation.benchmarks.swe_bench.run_infer.sleep_if_should_continue",
        lambda seconds: None, raising=False,
    )
    complete_runtime(runtime, INSTANCE)
    calls = [call.args[0] for call in runtime.run_action.call_args_list]
    diffs = [call for call in calls if isinstance(call, CmdRunAction) and call.command.startswith("git diff ")]
    assert [call.hard_timeout for call in diffs] == [600, 600, 600, 600, 700][:failed_attempts + 1]
    # The original code increments n_retries before reading/falling back.
    expected = [600, 600, 600, 700, 800][failed_attempts]
    reads = [call for call in calls if isinstance(call, FileReadAction)]
    assert [call.hard_timeout for call in reads] == [expected] * (2 if portable else 1)
    if portable:
        normalization = next(call for call in calls if isinstance(call, CmdRunAction) and "normalize" in shlex.split(call.command))
        command = shlex.split(normalization.command)
        assert normalization.hard_timeout == expected
        directory = runtime._swe_patch_context
        assert command == [
            "python", directory + "/portable_patch.py", "normalize",
            directory + "/state", "patch.diff", directory + "/patch.diff",
            str(expected),
        ]


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
    preparation = runtime.run_action.call_args.args[0]
    command = shlex.split(preparation.command)
    directory = runtime._swe_patch_context
    workspace = "/testbed" if dataset == "R2E-Gym" else "/workspace/example__repo__1"
    assert command == [
        "python", directory + "/portable_patch.py", "prepare", workspace,
        "base", directory + "/state", policy, "600",
    ]
    assert preparation.hard_timeout == 600
    assert runtime._swe_patch_context.startswith("/tmp/openhands-patch-")


def test_saved_portable_patch_is_recognized(tmp_path):
    completions = tmp_path / "llm_completions" / "example"
    completions.mkdir(parents=True)
    (completions / "completion.json").write_text("{}")
    result = {"instance_id": "example", "test_result": {"git_patch": "GIT binary patch\nliteral 1\n"}}
    (tmp_path / "output.jsonl").write_text(json.dumps(result) + "\n")
    assert _has_existing_result(str(tmp_path), "example") == (True, result)


@pytest.mark.parametrize("separator", ["\r", "\v", "\f", "\x85", "\u2028", "\u2029"])
@pytest.mark.parametrize("binary_last", [False, True])
def test_recovered_patch_keeps_legacy_utf8_block_output(
    patch_runtime, separator, binary_last
):
    """Legacy newline/control handling can be lossy; recovery must not change it."""
    runtime, patch_file, overrides = patch_runtime
    patch_file.write_bytes(PREFIX + b"\xff\n")
    # Literal payload validity is exercised by the real-Git helper/pipeline tests.
    binary = (
        "diff --git a/b b/b\nindex 1..2 100644\nGIT binary patch\n"
        "literal 1\nA00000\n\nliteral 1\nA00000\n\n"
    )
    text = PREFIX.decode() + "before" + separator + "after\n"
    restored = text + binary if binary_last else binary + text
    overrides[runtime._swe_patch_context + "/patch.diff"] = FileReadObservation(
        content=restored, path="portable"
    )
    result = complete_runtime(runtime, INSTANCE)["git_patch"]
    expected_text = PREFIX.decode() + "before\nafter"
    if binary_last:
        assert result == expected_text + "\n" + binary
    else:
        assert result == binary + expected_text


def test_recovered_empty_patch_keeps_legacy_empty_string(patch_runtime):
    runtime, patch_file, overrides = patch_runtime
    patch_file.write_bytes(PREFIX + b"\xff\n")
    overrides[runtime._swe_patch_context + "/patch.diff"] = FileReadObservation(
        content="", path="portable"
    )
    assert complete_runtime(runtime, INSTANCE) == {"git_patch": ""}



def test_recovered_utf8_block_keeps_legacy_partial_header_selection(patch_runtime):
    runtime, patch_file, overrides = patch_runtime
    patch_file.write_bytes(PREFIX + b"\xff\n")
    restored = PREFIX.decode() + "prefix\rdiff --git counterfeit\rBinary files marker\n"
    overrides[runtime._swe_patch_context + "/patch.diff"] = FileReadObservation(
        content=restored, path="portable"
    )
    assert complete_runtime(runtime, INSTANCE) == {"git_patch": PREFIX.decode() + "prefix"}
