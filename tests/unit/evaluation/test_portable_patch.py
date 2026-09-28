"""Real-Git oracles for direct blob conversion, including its whitespace delta."""

import hashlib
import os
import random
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from evaluation.benchmarks.swe_bench import portable_patch
from evaluation.benchmarks.swe_bench.portable_patch import convert as convert_patch


def git(repo, *args, data=None):
    result = subprocess.run(
        ['git', '-C', str(repo), *args], input=data, capture_output=True, timeout=10
    )
    assert result.returncode == 0, result.stderr.decode('ascii', 'backslashreplace')
    return result.stdout


def files(directory):
    return {
        str(p.relative_to(directory)): (
            os.fsencode(os.readlink(p)) if p.is_symlink() else p.read_bytes(),
            p.stat(follow_symlinks=False).st_mode & 0o777,
        )
        for p in directory.rglob('*') if p.is_symlink() or p.is_file()
    }


def worktree_files(directory):
    return {p: v for p, v in files(directory).items() if p.split('/')[0] != '.git'}


def _filter_blocks(raw):
    # Original binary-notice selection only, without the caller's text normalization.
    return [block for block in re.split(rb'(?m)(?=^diff --git )', raw)
            if not any(b'Binary files' in line for line in block.split(b'\n')
                       if not line.startswith(b'diff --git '))]


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
    git(repo, 'config', 'core.attributesFile', os.devnull)
    (repo / 'file.txt').write_bytes(b'base\n')
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'base')
    return repo, git(repo, 'rev-parse', 'HEAD').decode().strip()


def recover(case, edits, policy='nowarn', diff_args=()):
    # Direct conversion's oracle is raw patch application with NO whitespace fix.
    # A separate test below demonstrates the deliberate difference from R2E's fix.
    repo, base = case
    reference, target = repo.parent / 'reference', repo.parent / 'target'
    shutil.copytree(repo, reference)
    shutil.copytree(repo, target)
    edits(repo)
    git(repo, 'add', '-A')
    raw = git(repo, 'diff', '--no-color', '--cached', *diff_args, base)
    patch, output = repo.parent / 'raw.patch', repo.parent / 'portable.patch'
    patch.write_bytes(raw)
    before = files(repo)
    convert_patch(repo, patch, output)
    assert files(repo) == before
    portable = output.read_bytes()
    portable.decode('utf-8')
    selected = b''.join(_filter_blocks(raw))
    if selected.strip():
        git(reference, 'apply', '--whitespace=' + policy, '-', data=selected)
        git(target, 'apply', '--whitespace=' + policy, '-',
            data=b''.join(_filter_blocks(portable)))
    assert worktree_files(target) == worktree_files(reference)
    return raw, portable, target


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
    _, portable, _ = recover(
        case, lambda repo: (repo / 'file.txt').write_bytes(content)
    )
    assert b'GIT binary patch\n' in portable


@pytest.mark.parametrize('size', [17, 18, 19, 40, 41, 42, 43, 4096])
def test_incompressible_content_applies_and_reverses(case, size):
    # Small cases cross Git's 26/52-byte framing boundaries and base85 padding;
    # the large case spans many frames. Git itself is the format oracle.
    content = bytes(random.Random(924).choices(range(1, 256), k=size - 1)) + b'\xff'
    original = content[::-1]
    repo, _ = case
    (repo / 'file.txt').write_bytes(original)
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'incompressible base')
    case = repo, git(repo, 'rev-parse', 'HEAD').decode().strip()
    _, portable, target = recover(
        case, lambda r: (r / 'file.txt').write_bytes(content), policy='nowarn'
    )
    assert b'GIT binary patch\n' in portable
    assert (target / 'file.txt').read_bytes() == content
    git(target, 'apply', '--reverse', '-', data=portable)
    assert (target / 'file.txt').read_bytes() == original


@pytest.mark.parametrize('to_empty', [False, True])
def test_empty_blob_transition_applies_and_reverses(case, to_empty):
    original, content = (b'old \xff\n', b'') if to_empty else (b'', b'new \xff\n')
    repo, _ = case
    (repo / 'file.txt').write_bytes(original)
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'empty transition base')
    case = repo, git(repo, 'rev-parse', 'HEAD').decode().strip()
    _, portable, target = recover(
        case, lambda r: (r / 'file.txt').write_bytes(content)
    )
    assert (target / 'file.txt').is_file()
    assert (target / 'file.txt').read_bytes() == content
    git(target, 'apply', '--reverse', '-', data=portable)
    assert (target / 'file.txt').read_bytes() == original


def test_keeps_valid_blocks_and_defers_original_binary_notice_filtering(case):
    repo, _ = case
    (repo / 'z.bin').write_bytes(b'old\0binary')
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'binary base')
    case = repo, git(repo, 'rev-parse', 'HEAD').decode().strip()

    def edits(repo):
        (repo / 'file.txt').write_bytes(b'new \xff\n')
        (repo / 'valid.txt').write_bytes(b'valid trailing  \t\n')
        (repo / 'z.bin').write_bytes(b'new\0binary')

    raw, portable, _ = recover(case, edits)
    valid = next(
        b for b in _filter_blocks(raw) if b.startswith(b'diff --git a/valid.txt ')
    )
    assert valid in portable and b'diff --git a/z.bin ' in portable
    from evaluation.benchmarks.swe_bench.binary_patch_utils import remove_binary_diffs
    assert 'diff --git a/z.bin ' not in remove_binary_diffs(portable.decode('utf-8'))


@pytest.mark.parametrize('operation', ['new', 'delete', 'mode', 'rename', 'symlink'])
def test_operations(case, operation):
    repo, _ = case
    (repo / 'file.txt').write_bytes(b'old \xff\n' * 30)
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'nonutf base')
    case = repo, git(repo, 'rev-parse', 'HEAD').decode().strip()

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

    raw, portable, _ = recover(case, edits)
    if operation == 'rename':
        assert b'rename from file.txt\nrename to renamed.txt' in raw
        assert b'rename from file.txt\nrename to renamed.txt' in portable


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
    repo, _ = case
    destination = 'new ' + name
    original = b'old \xff\n' * 30
    (repo / name).write_bytes(original)
    git(repo, 'config', 'core.quotePath', str(quoted).lower())
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'quoted source base')
    case = repo, git(repo, 'rev-parse', 'HEAD').decode().strip()

    def edits(repo):
        if operation == 'rename':
            (repo / name).rename(repo / destination)
        else:
            shutil.copyfile(repo / name, repo / destination)
        with (repo / destination).open('ab') as handle:
            handle.write(b'added \xff\n')

    raw, portable, target = recover(case, edits, diff_args=('--find-copies-harder',))
    headers = [line for line in raw.split(b'\n') if line.startswith(operation.encode())]
    assert len(headers) == 2 and all(line in portable for line in headers)
    if operation == 'rename':
        git(target, 'apply', '--reverse', '-', data=portable)
        assert not (target / destination).exists()
    assert (target / name).read_bytes() == original


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

    repo, base = case
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
    convert_patch(repo, patch, output)
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
    repo, _ = case
    path = repo / 'foo'
    if direction == 'file_to_directory':
        path.write_bytes(b'original valid text\n')
    else:
        path.mkdir()
        (path / 'bar').write_bytes(b'original valid text\n')
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'path transition base')
    case = repo, git(repo, 'rev-parse', 'HEAD').decode().strip()

    def edits(repo):
        if direction == 'file_to_directory':
            path.unlink()
            path.mkdir()
            (path / 'bar').write_bytes(b'new undecodable \xff\n')
        else:
            (path / 'bar').unlink()
            path.rmdir()
            path.write_bytes(b'new undecodable \xff\n')

    raw, portable, _ = recover(case, edits)
    deletion = next(
        block for block in _filter_blocks(raw) if b'deleted file mode ' in block
    )
    assert deletion in portable
    assert portable.count(b'GIT binary patch\n') == 1


@pytest.mark.parametrize('source', ['tracked', 'info', 'config'])
def test_attributes_do_not_normalize_encoded_blobs(case, source):
    # The old converter froze these policies and encoded the --whitespace=fix
    # result. The direct helper intentionally ignores them and encodes raw blobs.
    repo, _ = case
    if source == 'tracked':
        (repo / '.gitattributes').write_text('file.txt whitespace=trailing-space\n')
    elif source == 'info':
        (repo / '.git/info/attributes').write_text('file.txt whitespace=trailing-space\n')
    else:
        git(repo, 'config', 'core.whitespace', 'trailing-space')
    git(repo, 'add', '-A')
    git(repo, 'commit', '--allow-empty', '-qm', 'original policy')
    base = git(repo, 'rev-parse', 'HEAD').decode().strip()
    reference, target = repo.parent / 'reference', repo.parent / 'target'
    shutil.copytree(repo, reference)
    shutil.copytree(repo, target)
    content = b'changed \xff  \t\n'
    (repo / 'file.txt').write_bytes(content)
    (repo / 'valid.txt').write_bytes(b'normal text  \t\n')
    git(repo, 'add', '-A')
    raw = git(repo, 'diff', '--cached', base)
    # Policy changes after extraction no longer require any captured state.
    (repo / '.git/info/attributes').write_text('file.txt -whitespace\n')
    git(repo, 'config', 'core.whitespace', '-trailing-space')
    patch, output = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    patch.write_bytes(raw)
    before = files(repo)
    convert_patch(repo, patch, output)
    assert files(repo) == before
    git(reference, 'apply', '--whitespace=fix', '-', data=raw)
    git(target, 'apply', '--whitespace=fix', str(output))
    assert (reference / 'file.txt').read_bytes() == b'changed \xff\n'
    assert (target / 'file.txt').read_bytes() == content
    assert (reference / 'valid.txt').read_bytes() == b'normal text\n'
    assert (target / 'valid.txt').read_bytes() == b'normal text\n'


def test_recorded_blob_ids_survive_head_index_and_worktree_changes(case):
    repo, base = case
    target = repo.parent / 'target'
    shutil.copytree(repo, target)
    (repo / 'file.txt').write_bytes(b'recorded \xff\n')
    git(repo, 'add', '-A')
    patch, output = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    patch.write_bytes(git(repo, 'diff', '--cached', base))
    git(repo, 'commit', '-qm', 'later HEAD')
    (repo / 'file.txt').write_bytes(b'later index\n')
    git(repo, 'add', '-A')
    (repo / 'file.txt').write_bytes(b'later worktree\n')
    before = files(repo)
    convert_patch(repo, patch, output)
    assert files(repo) == before
    git(target, 'apply', str(output))
    assert (target / 'file.txt').read_bytes() == b'recorded \xff\n'


def test_excluded_rename_keeps_source_and_existing_destination(case):
    repo, _ = case
    (repo / 'file.txt').write_bytes(b'old \xff\n' * 30)
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'rename base')
    base = git(repo, 'rev-parse', 'HEAD').decode().strip()

    def edits(repo):
        (repo / 'file.txt').rename(repo / 'renamed.txt')
        with (repo / 'renamed.txt').open('ab') as handle:
            handle.write(b'added \xff\n')

    _, portable, target = recover((repo, base), edits)
    git(target, 'reset', '--hard', base)
    (target / 'renamed.txt').write_bytes(b'preinstalled scaffold\n')
    git(target, 'apply', '--exclude=renamed.txt', '-', data=portable)
    assert (target / 'file.txt').exists()
    assert (target / 'renamed.txt').read_bytes() == b'preinstalled scaffold\n'


@pytest.mark.parametrize('invalid_notice', [False, True])
def test_binary_notice_selection_remains_legacy(case, invalid_notice):
    from evaluation.benchmarks.swe_bench.binary_patch_utils import remove_binary_diffs

    repo, _ = case
    name = b'bad-\xff' if invalid_notice else b'file.bin'
    raw = (b'diff --git a/' + name + b' b/' + name + b'\n'
           b'Binary files a/' + name + b' and b/' + name + b' differ\n')
    patch, output = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    patch.write_bytes(raw)
    convert_patch(repo, patch, output)
    assert output.read_bytes() == (b'' if invalid_notice else raw)
    assert remove_binary_diffs(output.read_text()) == ''


def test_utf8_only_is_byte_identical_and_needs_no_git(case, monkeypatch):
    repo, base = case
    (repo / 'file.txt').write_bytes(b'valid trailing  \t\r\n')
    git(repo, 'add', '-A')
    patch, output = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    raw = git(repo, 'diff', '--cached', base)
    patch.write_bytes(raw)

    def reject_git(*args, **kwargs):
        pytest.fail('Decodable blocks must not invoke Git')

    monkeypatch.setattr(subprocess, 'run', reject_git)
    convert_patch(repo, patch, output)
    assert output.read_bytes() == raw


@pytest.mark.parametrize('object_format', ['sha1', 'sha256'])
def test_full_blob_ids_and_reversible_literals(case, object_format):
    initial, _ = case
    repo = initial.parent / object_format
    repo.mkdir()
    git(repo, 'init', '-q', '--object-format=' + object_format)
    git(repo, 'config', 'user.name', 'Synthetic')
    git(repo, 'config', 'user.email', 'test@example.invalid')
    (repo / 'file.txt').write_bytes(b'base\n')
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'base')
    base = git(repo, 'rev-parse', 'HEAD').decode().strip()
    _, portable, target = recover(
        (repo, base), lambda p: (p / 'file.txt').write_bytes(b'new \xff\n')
    )
    index = re.search(rb'(?m)^index ([0-9a-f]+)\.\.([0-9a-f]+)', portable)
    assert [len(oid) for oid in index.groups()] == [40 if object_format == 'sha1' else 64] * 2
    git(target, 'apply', '--reverse', '-', data=portable)
    assert (target / 'file.txt').read_bytes() == b'base\n'


@pytest.mark.parametrize('zero_side', ['old', 'new'])
def test_all_zero_abbreviation_is_an_existing_blob(case, zero_side):
    repo, _ = case
    # Find a real blob with a 0000 prefix, without populating Git with trial blobs.
    for number in range(1000000):
        content = f'prefix candidate {number}'.encode() + b' \xff\n'
        digest = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        if digest.startswith('0000'):
            break
    else:
        pytest.fail('No deterministic zero-prefix fixture found')
    old, new = (content, b'other \xff\n') if zero_side == 'old' else (b'other \xff\n', content)
    (repo / 'file.txt').write_bytes(old)
    git(repo, 'config', 'core.abbrev', '4')
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'short zero base')
    base = git(repo, 'rev-parse', 'HEAD').decode().strip()
    raw, portable, target = recover(
        (repo, base), lambda p: (p / 'file.txt').write_bytes(new)
    )
    index = re.search(rb'(?m)^index ([0-9a-f]+)\.\.([0-9a-f]+)', raw)
    assert index.groups()[zero_side == 'new'] == b'0000'
    assert digest.encode() in portable
    assert (target / 'file.txt').read_bytes() == new
    git(target, 'apply', '--reverse', '-', data=portable)
    assert (target / 'file.txt').read_bytes() == old


@pytest.mark.parametrize('damage', ['index', 'header', 'hunk', 'missing', 'commit', 'absent'])
def test_malformed_or_unresolved_inputs_fail_without_output(case, damage):
    repo, base = case
    (repo / 'file.txt').write_bytes(b'changed \xff\n')
    git(repo, 'add', '-A')
    raw = git(repo, 'diff', '--cached', base)
    if damage == 'index':
        raw = re.sub(rb'(?m)^index .*\n', b'', raw)
    elif damage == 'header':
        raw = raw.replace(b'\n--- ', b'\nbroken ')
    elif damage == 'hunk':
        raw = raw.replace(b'@@ -1 +1 @@', b'@@ -1,999 +1,999 @@')
    elif damage in ('missing', 'commit'):
        oid = b'f' * 40 if damage == 'missing' else base.encode()
        raw = re.sub(rb'(?m)^index .*', b'index ' + oid + b'..' + oid + b' 100644', raw)
    else:
        raw = raw.replace(b'\nindex ', b'\nnew file mode 100644\nindex ')
    patch, output = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    patch.write_bytes(raw)
    before = files(repo)
    with pytest.raises((ValueError, RuntimeError)):
        convert_patch(repo, patch, output)
    assert not output.exists() and files(repo) == before


def test_cli_diagnostics_escape_control_bytes(case):
    repo, _ = case
    patch, output = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    patch.write_bytes(b'diff --git a/file.txt b/file.txt\nindex abc..abc\n--- a/file.txt\n'
                      b'+++ b/file.txt\n@@ invalid \xff\x1bPunterminated\n')
    result = subprocess.run(
        [sys.executable, convert_patch.__code__.co_filename,
         str(repo), str(patch), str(output), '600'], capture_output=True, timeout=10,
    )
    assert result.returncode == 1 and not output.exists()
    assert result.stdout == b'' and result.stderr.endswith(b'\n')
    assert all(32 <= byte < 127 for byte in result.stderr[:-1])


def test_empty_patch(case):
    repo, _ = case
    patch, output = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    patch.write_bytes(b'')
    convert_patch(repo, patch, output)
    assert output.read_bytes() == b''


def test_default_budget_is_shared_and_allows_more_than_ninety_seconds(case, monkeypatch):
    repo, base = case
    (repo / 'file.txt').write_bytes(b'changed \xff\n')
    git(repo, 'add', '-A')
    patch, output = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    patch.write_bytes(git(repo, 'diff', '--cached', base))
    clock, budgets, run = [0], [], subprocess.run
    monkeypatch.setattr(portable_patch, 'time', SimpleNamespace(monotonic=lambda: clock[0]))

    def simulate_slow_first_call(command, **kwargs):
        budgets.append(kwargs['timeout'])
        clock[0] = 120
        return run(command, **kwargs)

    monkeypatch.setattr(subprocess, 'run', simulate_slow_first_call)
    convert_patch(repo, patch, output)
    assert budgets[0] == 600 and all(budget == 480 for budget in budgets[1:])
    assert output.exists()


@pytest.mark.parametrize('timeout', [0, -1])
def test_expired_deadline_leaves_no_output(case, timeout):
    repo, _ = case
    patch, output = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    patch.write_bytes(b'')
    with pytest.raises(TimeoutError):
        convert_patch(repo, patch, output, timeout=timeout)
    assert not output.exists()


@pytest.mark.parametrize('expiry', ['git', 'elapsed'])
def test_deadline_failure_preserves_source(case, monkeypatch, expiry):
    repo, base = case
    (repo / 'file.txt').write_bytes(b'changed \xff\n')
    git(repo, 'add', '-A')
    patch, output = repo.parent / 'raw.patch', repo.parent / 'out.patch'
    patch.write_bytes(git(repo, 'diff', '--cached', base))
    before, clock, run = files(repo), [0], subprocess.run
    monkeypatch.setattr(portable_patch, 'time', SimpleNamespace(monotonic=lambda: clock[0]))

    def expire(command, **kwargs):
        if expiry == 'git':
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        result = run(command, **kwargs)
        if 'cat-file' in command:
            clock[0] = 8
        return result

    monkeypatch.setattr(subprocess, 'run', expire)
    with pytest.raises((subprocess.TimeoutExpired, TimeoutError)):
        convert_patch(repo, patch, output, timeout=7)
    assert not output.exists() and files(repo) == before
