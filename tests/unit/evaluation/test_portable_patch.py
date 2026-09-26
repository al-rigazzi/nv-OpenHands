"""Real-Git checks for the standalone converter; no runtime dependencies."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from evaluation.benchmarks.swe_bench.portable_patch import (
    Git,
    PatchConversionError,
    _attribute_path,
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
        git(target, 'apply', '--whitespace=' + policy, '-', data=output.read_bytes())

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


def test_keeps_valid_blocks_and_omits_original_binary(case):
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
    assert valid in portable and b'diff --git a/z.bin ' not in portable


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
    with pytest.raises(PatchConversionError):
        normalize(context, raw, out)
    assert out.read_bytes() == b''


def test_rejects_initial_tracked_drift_and_zero_deadline(case):
    repo, context, base = case
    (repo / 'file.txt').write_bytes(b'preexisting tracked drift\n')
    with pytest.raises(PatchConversionError):
        prepare(repo, base, context)
    with pytest.raises(PatchConversionError, match='time budget'):
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
    original = Git.run

    def delayed_git(self, *args, **kwargs):
        clock[0] = 120  # Simulate slow work without a real two-minute sleep.
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Git, 'run', delayed_git)
    if operation == 'prepare':
        prepare(repo, base, context)
        assert (context / 'state.json').is_file()
    else:
        normalize(context, patch, output)
        assert 'GIT binary patch' in output.read_text(encoding='utf-8')


def test_old_git_attribute_path_fallback(case, monkeypatch):
    repo, _, _ = case
    runner = Git(repo, 5)
    original = runner.run

    def old_git(*args, **kwargs):
        if args[0] == 'var':
            return b''
        if args[0] == '--exec-path':
            return b'/usr/lib/git-core\n'
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, 'run', old_git)
    runner.env.pop('GIT_ATTR_NOSYSTEM')
    assert _attribute_path(runner, 'GIT_ATTR_SYSTEM') == Path('/etc/gitattributes')
    assert _attribute_path(runner, 'GIT_ATTR_GLOBAL') == repo.parent / 'global-attrs'


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


def test_publication_is_atomic_and_private(case, monkeypatch):
    repo, context, base = case
    prepare(repo, base, context)
    (repo / 'file.txt').write_bytes(b'changed \xff\n')
    git(repo, 'add', '-A')
    patch, output = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    patch.write_bytes(git(repo, 'diff', '--cached', base))
    output.write_bytes(b'previous patch')
    before = set(repo.parent.iterdir())
    replace = os.replace

    def failed_replace(source, destination):
        assert Path(source).stat().st_mode & 0o777 == 0o600
        assert output.read_bytes() == b'previous patch'
        raise OSError('publication failed')

    monkeypatch.setattr(os, 'replace', failed_replace)
    with pytest.raises(OSError, match='publication failed'):
        normalize(context, patch, output)
    assert output.read_bytes() == b'previous patch'
    assert set(repo.parent.iterdir()) == before
    monkeypatch.setattr(os, 'replace', replace)
    normalize(context, patch, output)
    assert output.stat().st_mode & 0o777 == 0o600
    assert set(repo.parent.iterdir()) == before
    git(repo, 'reset', '--hard', base)
    git(repo, 'apply', str(output))
    assert (repo / 'file.txt').read_bytes() == b'changed \xff\n'


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
            '--context',
            str(context),
            '--patch',
            str(patch),
            '--output',
            str(output),
        ],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 1 and not output.exists()
    assert result.stdout == b''
    assert result.stderr.endswith(b'\n')
    assert all(32 <= byte < 127 for byte in result.stderr[:-1])
    assert b'<0xff><0x1b>Punterminated<0x0d>' in result.stderr
    assert b'z.txt: patch does not apply' in result.stderr
