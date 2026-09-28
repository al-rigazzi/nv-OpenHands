"""Real-Git checks for the standalone converter; no runtime dependencies."""

import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from evaluation.benchmarks.swe_bench.portable_patch import (
    _filter_blocks,
    normalize,
    prepare,
)


def git(repo, *args, data=None):
    result = subprocess.run(
        ['git', '-C', str(repo), *args], input=data, capture_output=True, timeout=10
    )
    assert result.returncode == 0, result.stderr.decode('ascii', 'backslashreplace')
    return result.stdout


@pytest.fixture
def case(tmp_path, monkeypatch):
    for key, value in {
        'GIT_CONFIG_NOSYSTEM': '1',
        'GIT_CONFIG_GLOBAL': os.devnull,
        'GIT_ATTR_NOSYSTEM': '1',
    }.items():
        monkeypatch.setenv(key, value)
    repo = tmp_path / 'agent'
    repo.mkdir()
    git(repo, 'init', '-q')
    git(repo, 'config', 'user.name', 'Synthetic')
    git(repo, 'config', 'user.email', 'test@example.invalid')
    attrs = tmp_path / 'global-attrs'
    attrs.write_bytes(b'')
    git(repo, 'config', 'core.attributesFile', str(attrs))
    (repo / 'file.txt').write_bytes(b'base\n')
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'base')
    return repo, tmp_path / 'context', git(repo, 'rev-parse', 'HEAD').decode().strip()


def convert(case, edits, policy='fix', diff_args=()):
    repo, context, base = case
    reference, target = repo.parent / 'reference', repo.parent / 'target'
    shutil.copytree(repo, reference)
    shutil.copytree(repo, target)
    prepare(repo, base, context, whitespace=policy)
    edits(repo)
    git(repo, 'add', '-A')
    raw = git(repo, 'diff', '--no-color', '--cached', *diff_args, base)
    selected = b''.join(_filter_blocks(raw))
    patch, output = repo.parent / 'raw.patch', repo.parent / 'portable.patch'
    patch.write_bytes(raw)

    def source_state():
        return (
            git(repo, 'rev-parse', 'HEAD'),
            (repo / '.git/index').read_bytes(),
            (repo / '.git/config').read_bytes(),
        )

    before = source_state()
    normalize(context, patch, output)
    assert before == source_state()
    if selected.strip():
        git(reference, 'apply', '--whitespace=' + policy, '-', data=selected)
        git(
            target, 'apply', '--whitespace=' + policy, '-',
            data=b''.join(_filter_blocks(output.read_bytes())),
        )

    def files(directory):
        return {
            str(p.relative_to(directory)): (
                p.read_bytes() if not p.is_symlink() else os.fsencode(os.readlink(p)),
                p.stat(follow_symlinks=False).st_mode & 0o777,
            )
            for p in directory.rglob('*')
            if '.git' not in p.parts and (p.is_file() or p.is_symlink())
        }

    assert files(target) == files(reference)
    output.read_text(encoding='utf-8')
    return raw, output.read_bytes(), target


@pytest.mark.parametrize(
    'content',
    [
        b'changed \xff\n',
        b'changed \xff\x1bPunterminated\n',
        b'changed \xff\x1b]0;title\x07\n',
        b'changed \xff\x1b[31mred\n',
        b'changed \xff\x85\x0b\x0c\rdata\r\n',
        b'changed \xff no final newline',
        b'changed \xff trailing  \t\n',
        b'changed \xff\n' * 8000,
    ],
)
def test_preserves_application_bytes(case, content):
    _, portable, _ = convert(
        case, lambda repo: (repo / 'file.txt').write_bytes(content)
    )
    assert b'GIT binary patch\n' in portable


@pytest.mark.parametrize('size', [17, 18, 19, 40, 41, 42, 43, 4096])
def test_incompressible_content_applies_and_reverses(case, size):
    # Small cases cross Git's 26/52-byte framing boundaries and base85 padding;
    # the large case spans many frames. Git itself is the format oracle.
    content = bytes(random.Random(924).choices(range(1, 256), k=size - 1)) + b'\xff'
    original = content[::-1]
    repo, context, _ = case
    (repo / 'file.txt').write_bytes(original)
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'incompressible base')
    case = repo, context, git(repo, 'rev-parse', 'HEAD').decode().strip()
    _, portable, target = convert(
        case, lambda r: (r / 'file.txt').write_bytes(content), policy='nowarn'
    )
    assert b'GIT binary patch\n' in portable
    assert (target / 'file.txt').read_bytes() == content
    git(target, 'apply', '--reverse', '-', data=portable)
    assert (target / 'file.txt').read_bytes() == original


@pytest.mark.parametrize('to_empty', [False, True])
def test_empty_blob_transition_applies_and_reverses(case, to_empty):
    original, content = (b'old \xff\n', b'') if to_empty else (b'', b'new \xff\n')
    repo, context, _ = case
    (repo / 'file.txt').write_bytes(original)
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'empty transition base')
    case = repo, context, git(repo, 'rev-parse', 'HEAD').decode().strip()
    _, portable, target = convert(
        case, lambda r: (r / 'file.txt').write_bytes(content)
    )
    assert (target / 'file.txt').is_file()
    assert (target / 'file.txt').read_bytes() == content
    git(target, 'apply', '--reverse', '-', data=portable)
    assert (target / 'file.txt').read_bytes() == original


def test_keeps_valid_blocks_and_defers_original_binary_notice_filtering(case):
    repo, context, _ = case
    (repo / 'z.bin').write_bytes(b'old\0binary')
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'binary base')
    case = repo, context, git(repo, 'rev-parse', 'HEAD').decode().strip()

    def edits(repo):
        (repo / 'file.txt').write_bytes(b'new \xff\n')
        (repo / 'valid.txt').write_bytes(b'valid trailing  \t\n')
        (repo / 'z.bin').write_bytes(b'new\0binary')

    raw, portable, _ = convert(case, edits)
    valid = next(
        b for b in _filter_blocks(raw) if b.startswith(b'diff --git a/valid.txt ')
    )
    assert valid in portable and b'diff --git a/z.bin ' in portable
    from evaluation.benchmarks.swe_bench.binary_patch_utils import remove_binary_diffs
    assert 'diff --git a/z.bin ' not in remove_binary_diffs(portable.decode('utf-8'))


@pytest.mark.parametrize('operation', ['new', 'delete', 'mode', 'rename', 'symlink'])
def test_operations(case, operation):
    repo, context, _ = case
    (repo / 'file.txt').write_bytes(b'old \xff\n' * 30)
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'nonutf base')
    case = repo, context, git(repo, 'rev-parse', 'HEAD').decode().strip()

    def edits(repo):
        if operation == 'new':
            (repo / 'new.txt').write_bytes(b'new \xff\n')
        elif operation == 'delete':
            (repo / 'file.txt').unlink()
        elif operation == 'mode':
            (repo / 'file.txt').write_bytes(b'new \xff\n')
            (repo / 'file.txt').chmod(0o755)
        elif operation == 'rename':
            (repo / 'file.txt').rename(repo / 'renamed.txt')
            with (repo / 'renamed.txt').open('ab') as f:
                f.write(b'added \xff\n')
        else:
            os.symlink(b'invalid-\xff', os.fsencode(repo / 'link'))

    raw, portable, _ = convert(case, edits)
    if operation == 'rename':
        assert b'rename from file.txt\nrename to renamed.txt' in raw
        assert b'rename from file.txt\nrename to renamed.txt' in portable


def test_attribute_and_config_snapshot_is_frozen(case):
    repo, context, base = case
    (repo / '.gitattributes').write_bytes(b'file.txt -whitespace\n')
    prepare(repo, base, context)
    (repo / '.gitattributes').write_bytes(b'file.txt whitespace\n')
    git(repo, 'config', 'core.whitespace', 'blank-at-eol')
    (repo / '.git/info/attributes').write_bytes(b'file.txt whitespace\n')
    (repo / 'file.txt').write_bytes(b'new \xff trailing  \t\n')
    git(repo, 'add', 'file.txt')
    raw, output = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    raw.write_bytes(git(repo, 'diff', '--cached', base))
    normalize(context, raw, output)
    git(repo, 'reset', '--hard', base)
    git(repo, 'apply', '--whitespace=fix', str(output))
    assert (repo / 'file.txt').read_bytes() == b'new \xff trailing  \t\n'


def test_nowarn_does_not_fix_whitespace(case):
    _, _, target = convert(
        case, lambda r: (r / 'file.txt').write_bytes(b'new \xff  \t\n'), policy='nowarn'
    )
    assert (target / 'file.txt').read_bytes() == b'new \xff  \t\n'


def test_normalized_identity_is_nonempty_applicable_patch(case):
    repo, context, _ = case
    (repo / 'file.txt').write_bytes(b'old \xff\n')
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'nonutf base')
    case = repo, context, git(repo, 'rev-parse', 'HEAD').decode().strip()
    _, portable, target = convert(
        case, lambda r: (r / 'file.txt').write_bytes(b'old \xff  \t\n')
    )
    assert portable.strip() and (target / 'file.txt').read_bytes() == b'old \xff\n'
    git(target, 'apply', '--reverse', '-', data=portable)
    assert (target / 'file.txt').read_bytes() == b'old \xff\n'


def test_excluded_rename_does_not_become_source_deletion(case):
    repo, context, _ = case
    (repo / 'file.txt').write_bytes(b'old \xff\n' * 30)
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'nonutf base')
    base = git(repo, 'rev-parse', 'HEAD').decode().strip()
    prepare(repo, base, context)
    (repo / 'file.txt').rename(repo / 'renamed.txt')
    with (repo / 'renamed.txt').open('ab') as f:
        f.write(b'added \xff\n')
    git(repo, 'add', '-A')
    raw, out = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    raw.write_bytes(git(repo, 'diff', '--cached', base))
    normalize(context, raw, out)
    git(repo, 'reset', '--hard', base)
    (repo / 'renamed.txt').write_bytes(b'preinstalled scaffold\n')
    git(repo, 'apply', '--whitespace=fix', '--exclude=renamed.txt', str(out))
    assert (repo / 'file.txt').exists()
    assert (repo / 'renamed.txt').read_bytes() == b'preinstalled scaffold\n'


def test_filter_preserves_final_lf_before_dropped_block():
    kept = b'diff --git a/a b/a\n+bad \xff\x85\r\n'
    dropped = b'diff --git a/z b/z\nBinary files a/z and b/z differ\n'
    assert b''.join(_filter_blocks(kept + dropped)) == kept


def test_filter_matches_legacy_header_and_body_selection():
    kept = b'diff --git a/Binary files b/Binary files\n+bad \xff\rdata\n'
    dropped = b'diff --git a/z b/z\n+text containing Binary files is also omitted\n'
    final = b'diff --git a/last b/last\n+bad \xff\rdiff --git not a new block'
    raw = b'Binary files in preamble\n' + kept + dropped + final
    assert _filter_blocks(raw) == [kept, final]


def test_empty_and_malformed(case):
    repo, context, base = case
    prepare(repo, base, context)
    raw, out = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    raw.write_bytes(b'')
    normalize(context, raw, out)
    assert out.read_bytes() == b''
    raw.write_bytes(b'invalid \xff patch\n')
    with pytest.raises(RuntimeError):
        normalize(context, raw, out)
    assert out.read_bytes() == b''


def test_initial_tracked_drift_preserves_raw_git_application(case):
    repo, context, base = case
    (repo / 'file.txt').write_bytes(b'preexisting tracked drift \xff  \t\n')
    prepare(repo, base, context)
    git(repo, 'add', '-A')
    raw = git(repo, 'diff', '--cached', base)
    patch, output = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    patch.write_bytes(raw)
    normalize(context, patch, output)
    git(repo, 'reset', '--hard', base)
    git(repo, 'apply', '--whitespace=fix', '-', data=raw)
    expected = (repo / 'file.txt').read_bytes()
    git(repo, 'reset', '--hard', base)
    git(repo, 'apply', '--whitespace=fix', str(output))
    assert (repo / 'file.txt').read_bytes() == expected


def test_rejects_zero_deadline(case):
    repo, context, base = case
    with pytest.raises(TimeoutError, match='time budget'):
        prepare(repo, base, context, timeout=0)


@pytest.mark.parametrize("operation", ["prepare", "normalize"])
def test_default_budget_allows_git_work_after_ninety_seconds(case, monkeypatch, operation):
    repo, context, base = case
    patch, output = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    if operation == 'normalize':
        prepare(repo, base, context)
        (repo / 'file.txt').write_bytes(b'changed \xff\n')
        git(repo, 'add', '-A')
        patch.write_bytes(git(repo, 'diff', '--cached', base))
    clock = [0]
    monkeypatch.setattr(
        'evaluation.benchmarks.swe_bench.portable_patch.time',
        SimpleNamespace(monotonic=lambda: clock[0]),
    )
    original = subprocess.run
    budgets = []

    def delayed_git(*args, **kwargs):
        budgets.append(kwargs['timeout'])
        clock[0] = 120  # Simulate slow work without a real two-minute sleep.
        return original(*args, **kwargs)

    monkeypatch.setattr(subprocess, 'run', delayed_git)
    if operation == 'prepare':
        prepare(repo, base, context)
        assert (context / 'state.json').is_file()
    else:
        normalize(context, patch, output)
        assert 'GIT binary patch' in output.read_text(encoding='utf-8')
    assert budgets[0] == 600 and 480 in budgets


def test_old_git_attribute_path_fallback(case, monkeypatch):
    repo, context, base = case
    monkeypatch.delenv('GIT_ATTR_NOSYSTEM')
    attrs = repo.parent / 'global-attrs'
    attrs.write_bytes(b'file.txt -whitespace\n')
    original = subprocess.run
    calls = []

    def old_git(command, **kwargs):
        calls.append(command)
        if 'var' in command and command[-1].startswith('GIT_ATTR_'):
            return subprocess.CompletedProcess(command, 1, b'', b'')
        if '--exec-path' in command:
            return subprocess.CompletedProcess(command, 0, b'/usr/lib/git-core\n', b'')
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, 'run', old_git)
    prepare(repo, base, context)
    assert any('--exec-path' in command for command in calls)
    attrs.write_bytes(b'file.txt whitespace\n')
    (repo / 'file.txt').write_bytes(b'changed \xff  \t\n')
    git(repo, 'add', '-A')
    raw, output = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    raw.write_bytes(git(repo, 'diff', '--cached', base))
    normalize(context, raw, output)
    git(repo, 'reset', '--hard', base)
    git(repo, 'apply', '--whitespace=fix', str(output))
    assert (repo / 'file.txt').read_bytes() == b'changed \xff  \t\n'


@pytest.mark.parametrize('source', ['global', 'info', 'config', 'ignored'])
def test_initial_policy_sources_survive_agent_changes(case, source):
    repo, context, base = case
    if source == 'config':
        git(repo, 'config', 'core.whitespace', '-blank-at-eol')
        policy_path = None
    elif source == 'global':
        policy_path = repo.parent / 'global-attrs'
    elif source == 'info':
        policy_path = repo / '.git/info/attributes'
    else:
        (repo / '.git/info/exclude').write_text('.gitattributes\n')
        policy_path = repo / '.gitattributes'
    if policy_path:
        policy_path.write_bytes(b'file.txt -whitespace\n')
    prepare(repo, base, context)
    if policy_path:
        policy_path.write_bytes(b'file.txt whitespace\n')
    git(repo, 'config', 'core.whitespace', 'blank-at-eol')
    (repo / 'file.txt').write_bytes(b'new \xff  \t\n')
    git(repo, 'add', 'file.txt')
    raw, out = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    raw.write_bytes(git(repo, 'diff', '--cached', base))
    normalize(context, raw, out)
    git(repo, 'reset', '--hard', base)
    git(repo, 'apply', '--whitespace=fix', str(out))
    assert (repo / 'file.txt').read_bytes() == b'new \xff  \t\n'


@pytest.mark.parametrize('operation', ['rename', 'copy'])
@pytest.mark.parametrize('name,quoted', [
    ('old name.txt', True),
    ('old\tname.txt', True),
    ('old\nname.txt', True),
    ('old caf\u00e9.txt', False),
    ('old caf\u00e9\t"\\\n.txt', True),
])
def test_path_operations_preserve_identity_and_rename_reversal(
    case, operation, name, quoted,
):
    repo, context, _ = case
    destination = 'new ' + name
    original = b'old \xff\n' * 30
    (repo / name).write_bytes(original)
    git(repo, 'config', 'core.quotePath', str(quoted).lower())
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'quoted source base')
    case = repo, context, git(repo, 'rev-parse', 'HEAD').decode().strip()

    def edits(repo):
        if operation == 'rename':
            (repo / name).rename(repo / destination)
        else:
            shutil.copyfile(repo / name, repo / destination)
        with (repo / destination).open('ab') as handle:
            handle.write(b'added \xff\n')

    raw, portable, target = convert(case, edits, diff_args=('--find-copies-harder',))
    headers = [line for line in raw.split(b'\n') if line.startswith(operation.encode())]
    assert len(headers) == 2 and all(line in portable for line in headers)
    if operation == 'rename':
        git(target, 'apply', '--reverse', '-', data=portable)
        assert not (target / destination).exists()
    assert (target / name).read_bytes() == original


def test_prepare_does_not_mutate_source(case):
    repo, context, base = case
    (repo / '.git/info/attributes').write_bytes(b'file.txt -whitespace\n')
    tracked = [
        repo / '.git/index',
        repo / '.git/config',
        repo / '.git/info/attributes',
        repo / 'file.txt',
    ]
    before = {p: p.read_bytes() for p in tracked}
    head = git(repo, 'rev-parse', 'HEAD')
    prepare(repo, base, context)
    assert {p: p.read_bytes() for p in tracked} == before
    assert git(repo, 'rev-parse', 'HEAD') == head


def test_cli_failure_diagnostics_cannot_emit_terminal_controls(case):
    repo, context, _ = case
    (repo / 'z.txt').write_bytes(b'old second\n')
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'two-file base')
    base = git(repo, 'rev-parse', 'HEAD').decode().strip()
    prepare(repo, base, context)
    (repo / 'file.txt').write_bytes(b'changed \xff\x1bPunterminated\r  \t\n')
    (repo / 'z.txt').write_bytes(b'new second\n')
    git(repo, 'add', '-A')
    raw = git(repo, 'diff', '--cached', base).replace(
        b'-old second\n', b'-not the old second\n'
    )
    patch, output = repo.parent / 'bad.patch', repo.parent / 'output.patch'
    patch.write_bytes(raw)
    result = subprocess.run(
        [
            sys.executable,
            prepare.__code__.co_filename,
            'normalize',
            str(context),
            str(patch),
            str(output),
            '600',
        ],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 1 and not output.exists()
    assert result.stdout == b''
    assert result.stderr.endswith(b'\n')
    assert all(32 <= byte < 127 for byte in result.stderr[:-1])
    assert all(
        marker in result.stderr for marker in (b'xff', b'x1b', b'Punterminated')
    )
    assert b'z.txt: patch does not apply' in result.stderr


@pytest.mark.parametrize(
    'content',
    [
        b'plain UTF-8\n',
        'caf\u00e9\n'.encode(),
        b'vertical\x0btab\n',
        b'form\x0cfeed\n',
        'next\u0085line\n'.encode(),
        'line\u2028separator\n'.encode(),
        'paragraph\u2029separator\n'.encode(),
        b'carriage\rreturn\r\n',
        b'without final newline',
        b'with final newline\n',
        b'prefix\rdiff --git counterfeit\rBinary files marker\n',
    ],
)
@pytest.mark.parametrize('valid_name', ['a-valid.txt', 'z-valid.txt'])
def test_mixed_fallback_preserves_every_valid_block_byte(case, content, valid_name):
    # The helper's responsibility ends at byte-safe representation. The runtime
    # must still run the original text filter, including its splitlines policy.
    from evaluation.benchmarks.swe_bench.binary_patch_utils import remove_binary_diffs

    repo, context, base = case
    prepare(repo, base, context)
    (repo / 'file.txt').write_bytes(b'undecodable \xff\n')
    (repo / valid_name).write_bytes(content)
    git(repo, 'add', '-A')
    raw = git(repo, 'diff', '--no-color', '--cached', base)
    valid = next(
        block for block in re.split(rb'(?m)(?=^diff --git )', raw)
        if block.startswith(f'diff --git a/{valid_name} '.encode())
    )
    patch, output = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    patch.write_bytes(raw)
    normalize(context, patch, output)
    converted = output.read_bytes()
    assert valid in converted
    assert converted.count(b'GIT binary patch\n') == 1
    # Valid-file output is exactly the old caller's result, even where the old
    # filter intentionally changes CRLF, control characters, or the last LF.
    final = remove_binary_diffs(converted.decode('utf-8'))
    expected = remove_binary_diffs(valid.decode('utf-8'))
    start = final.index(f'diff --git a/{valid_name} ')
    end = final.find('\ndiff --git ', start)
    actual = final[start:] if end < 0 else final[start:end]
    assert actual == expected


@pytest.mark.parametrize('direction', ['file_to_directory', 'directory_to_file'])
def test_mixed_file_directory_transition_uses_complete_index_change(case, direction):
    repo, context, _ = case
    path = repo / 'foo'
    if direction == 'file_to_directory':
        path.write_bytes(b'original valid text\n')
    else:
        path.mkdir()
        (path / 'bar').write_bytes(b'original valid text\n')
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'path transition base')
    case = repo, context, git(repo, 'rev-parse', 'HEAD').decode().strip()

    def edits(repo):
        if direction == 'file_to_directory':
            path.unlink()
            path.mkdir()
            (path / 'bar').write_bytes(b'new undecodable \xff\n')
        else:
            (path / 'bar').unlink()
            path.rmdir()
            path.write_bytes(b'new undecodable \xff\n')

    raw, portable, _ = convert(case, edits)
    deletion = next(
        block for block in _filter_blocks(raw) if b'deleted file mode ' in block
    )
    assert deletion in portable
    assert portable.count(b'GIT binary patch\n') == 1


def test_all_utf8_blocks_do_not_run_application_policy(case, monkeypatch):
    repo, context, base = case
    prepare(repo, base, context)
    (repo / 'file.txt').write_bytes(b'valid trailing  \t\n')
    git(repo, 'add', '-A')
    raw = git(repo, 'diff', '--cached', base)
    patch, output = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    patch.write_bytes(raw)
    original = subprocess.run

    def reject_application(command, **kwargs):
        assert not ('apply' in command and '--cached' in command)
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, 'run', reject_application)
    normalize(context, patch, output)
    assert output.read_bytes() == raw


def test_fix_encodes_the_evaluators_result_not_unfixed_file_bytes(case):
    _, portable, target = convert(
        case, lambda repo: (repo / 'file.txt').write_bytes(b'changed \xff  \t\n')
    )
    assert b'GIT binary patch\n' in portable
    assert (target / 'file.txt').read_bytes() == b'changed \xff\n'


@pytest.mark.parametrize('operation', ['prepare', 'normalize'])
def test_git_timeout_is_reported_without_output_or_source_mutation(
    case, monkeypatch, operation
):
    repo, context, base = case
    patch, output = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    if operation == 'normalize':
        prepare(repo, base, context)
        (repo / 'file.txt').write_bytes(b'changed \xff\n')
        git(repo, 'add', '-A')
        patch.write_bytes(git(repo, 'diff', '--cached', base))
    observed = [repo / '.git/index', repo / '.git/config', repo / 'file.txt']
    before = {path: path.read_bytes() for path in observed}

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])

    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        if operation == 'prepare':
            prepare(repo, base, context, timeout=7)
        else:
            normalize(context, patch, output, timeout=7)
    assert not output.exists()
    assert {path: path.read_bytes() for path in observed} == before


def test_sha256_repository_uses_full_blob_ids_and_reversible_literals(case):
    original, context, _ = case
    repo = original.parent / 'sha256'
    repo.mkdir()
    git(repo, 'init', '-q', '--object-format=sha256')
    git(repo, 'config', 'user.name', 'Synthetic')
    git(repo, 'config', 'user.email', 'test@example.invalid')
    (repo / 'file.txt').write_bytes(b'base\n')
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'SHA256 base')
    base = git(repo, 'rev-parse', 'HEAD').decode().strip()
    _, portable, target = convert(
        (repo, context, base),
        lambda path: (path / 'file.txt').write_bytes(b'new \xff\n'),
    )
    index = next(line for line in portable.splitlines() if line.startswith(b'index '))
    old, new = index.split()[1].split(b'..')
    assert len(old) == len(new) == 64
    git(target, 'apply', '--reverse', '-', data=portable)
    assert (target / 'file.txt').read_bytes() == b'base\n'


@pytest.mark.parametrize('xdg', ['', None])
def test_old_git_missing_xdg_uses_home_attributes(case, monkeypatch, xdg):
    repo, context, base = case
    git(repo, 'config', '--unset', 'core.attributesFile')
    fallback_home = repo.parent / 'fallback-home'
    attrs = fallback_home / '.config/git/attributes'
    attrs.parent.mkdir(parents=True)
    attrs.write_bytes(b'file.txt -whitespace\n')
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: fallback_home))
    if xdg is None:
        monkeypatch.delenv('XDG_CONFIG_HOME', raising=False)
    else:
        monkeypatch.setenv('XDG_CONFIG_HOME', xdg)
    original = subprocess.run

    def old_git(command, **kwargs):
        if 'var' in command and command[-1] == 'GIT_ATTR_GLOBAL':
            return subprocess.CompletedProcess(command, 1, b'', b'')
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, 'run', old_git)
    prepare(repo, base, context)
    attrs.write_bytes(b'file.txt whitespace\n')
    (repo / 'file.txt').write_bytes(b'changed \xff  \t\n')
    git(repo, 'add', '-A')
    raw, output = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    raw.write_bytes(git(repo, 'diff', '--cached', base))
    normalize(context, raw, output)
    git(repo, 'reset', '--hard', base)
    git(repo, 'apply', '--whitespace=fix', str(output))
    assert (repo / 'file.txt').read_bytes() == b'changed \xff  \t\n'


def test_prepare_deadline_covers_policy_file_copy(case, monkeypatch):
    repo, context, base = case
    attrs = repo / '.gitattributes'
    attrs.write_bytes(b'file.txt -whitespace\n')
    observed = [repo / '.git/index', repo / '.git/config', attrs]
    before = {path: path.read_bytes() for path in observed}
    clock = [0]
    monkeypatch.setattr(
        'evaluation.benchmarks.swe_bench.portable_patch.time',
        SimpleNamespace(monotonic=lambda: clock[0]),
    )
    original = Path.read_bytes

    def delayed_copy(path):
        result = original(path)
        if path == attrs:
            clock[0] = 8
        return result

    monkeypatch.setattr(Path, 'read_bytes', delayed_copy)
    with pytest.raises(TimeoutError, match='time budget'):
        prepare(repo, base, context, timeout=7)
    assert (context / 'worktree/.gitattributes').is_file()
    assert not (context / 'state.json').exists()
    assert {path: path.read_bytes() for path in observed} == before


def test_external_object_store_preserves_source_and_converts(case, monkeypatch):
    repo, context, base = case
    target = repo.parent / 'target'
    shutil.copytree(repo, target)
    objects = repo.parent / 'external-objects'
    shutil.move(repo / '.git/objects', objects)
    monkeypatch.setenv('GIT_OBJECT_DIRECTORY', str(objects))

    def source_state():
        tracked = [repo / '.git/index', repo / '.git/config', repo / 'file.txt']
        return (
            git(repo, 'rev-parse', 'HEAD'),
            {path: path.read_bytes() for path in tracked},
            {
                path.relative_to(objects): path.read_bytes()
                for path in objects.rglob('*') if path.is_file()
            },
        )

    before = source_state()
    prepare(repo, base, context)
    assert source_state() == before
    (repo / 'file.txt').write_bytes(b'changed \xff  \t\n')
    git(repo, 'add', '-A')
    raw, output = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    raw.write_bytes(git(repo, 'diff', '--cached', base))
    before = source_state()
    normalize(context, raw, output)
    assert source_state() == before
    assert not (repo / '.git/objects').exists()
    with monkeypatch.context() as target_environment:
        target_environment.delenv('GIT_OBJECT_DIRECTORY')
        git(target, 'apply', '--whitespace=fix', str(output))
        assert (target / 'file.txt').read_bytes() == b'changed \xff\n'
        git(target, 'apply', '--reverse', str(output))
        assert (target / 'file.txt').read_bytes() == b'base\n'
