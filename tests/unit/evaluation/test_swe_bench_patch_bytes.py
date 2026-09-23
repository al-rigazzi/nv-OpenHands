import base64
import json
import subprocess
from unittest.mock import MagicMock

import pandas as pd
import pytest

from evaluation.benchmarks.swe_bench.binary_patch_utils import remove_binary_diffs
from evaluation.benchmarks.swe_bench.run_infer import (
    _has_existing_result,
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
BINARY = b"diff --git a/b b/b\nBinary files a/b and b/b differ\n"
INSTANCE = pd.Series({"repo": "example/repo", "version": "1", "base_commit": "base"})


@pytest.fixture
def patch_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "evaluation.benchmarks.swe_bench.run_infer.DATASET_TYPE", "nv-internal-1"
    )
    runtime = MagicMock(spec=Runtime)
    overrides = {}

    def run_action(action):
        if isinstance(action, CmdRunAction):
            assert action.command != "cat patch.diff", "Raw patch reached terminal"
            if action.command == "base64 < patch.diff > patch.diff.base64":
                result = subprocess.run(
                    action.command,
                    shell=True,
                    cwd=tmp_path,
                    capture_output=True,
                    check=True,
                    timeout=5,
                )
                assert result.stdout == b""
            return CmdOutputObservation(content="", command=action.command, exit_code=0)
        assert isinstance(action, FileReadAction)
        assert action.path in ("patch.diff", "patch.diff.base64")
        if action.path in overrides:
            return overrides[action.path]
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


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff\x1bPsynthetic\n",
        b"\xff\x1b[31msynthetic\n",
        b"\xff\x1b]synthetic\n",
        b"\xff\x85\x00\x1bP\r\n",
        b"\xff\n\\ No newline at end of file\n",
    ],
)
def test_invalid_utf8_round_trips_bytes(patch_runtime, payload):
    runtime, patch_file, _ = patch_runtime
    patch = PREFIX + payload
    patch_file.write_bytes(patch)
    result = complete_runtime(runtime, INSTANCE)
    assert result["git_patch"] is None
    assert base64.b64decode(result["git_patch_b64"], validate=True) == patch


def test_excluded_final_binary_block_keeps_text_newline(patch_runtime):
    runtime, patch_file, _ = patch_runtime
    text_patch = PREFIX + b"\xff\r\n"
    patch_file.write_bytes(text_patch + BINARY)
    result = complete_runtime(runtime, INSTANCE)
    assert base64.b64decode(result["git_patch_b64"]) == text_patch


@pytest.mark.parametrize("patch", [b"", BINARY, BINARY + BINARY])
def test_empty_or_binary_only_patch_is_empty(patch):
    assert remove_binary_diffs(patch) == b""


@pytest.mark.parametrize("content", ["%%% invalid %%%", "A", "é"])
def test_malformed_encoded_read_is_rejected(patch_runtime, content):
    runtime, patch_file, overrides = patch_runtime
    patch_file.write_bytes(PREFIX + b"\xff\n")
    overrides["patch.diff.base64"] = FileReadObservation(
        content=content, path="patch.diff.base64"
    )
    with pytest.raises(ValueError):
        complete_runtime(runtime, INSTANCE)


def test_encoded_file_read_error_is_reported(patch_runtime):
    runtime, patch_file, overrides = patch_runtime
    patch_file.write_bytes(PREFIX + b"\xff\n")
    overrides["patch.diff.base64"] = ErrorObservation("File not found")
    with pytest.raises(EvalException, match="Failed to read encoded git patch"):
        complete_runtime(runtime, INSTANCE)


def test_other_read_error_does_not_start_recovery(patch_runtime):
    runtime, _, overrides = patch_runtime
    overrides["patch.diff"] = ErrorObservation("File not found")
    with pytest.raises(AssertionError):
        complete_runtime(runtime, INSTANCE)
    assert isinstance(runtime.run_action.call_args.args[0], FileReadAction)


def test_saved_byte_patch_is_recognized(tmp_path):
    completions = tmp_path / "llm_completions" / "example"
    completions.mkdir(parents=True)
    (completions / "completion.json").write_text("{}")
    result = {
        "instance_id": "example",
        "test_result": {
            "git_patch": None,
            "git_patch_b64": base64.b64encode(PREFIX + b"\xff\n").decode(),
        },
    }
    (tmp_path / "output.jsonl").write_text(json.dumps(result) + "\n")
    assert _has_existing_result(str(tmp_path), "example") == (True, result)
