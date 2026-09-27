"""Represent non-UTF-8 text hunks as standard Git binary patches.

The private context snapshots the evaluator's initial attributes and whitespace
policy. Binary hunks require that evaluator's base file contents. This is patch
conversion for a known apply policy, not a transport encoding for arbitrary
consumers. The agent repository is never modified by this module.
"""

import base64
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zlib


class PatchConversionError(Exception):
    pass


class Git:
    def __init__(self, cwd, timeout, private=False):
        self.cwd = Path(cwd)
        self.deadline = time.monotonic() + timeout
        self.env = dict(os.environ)
        self.env['GIT_OPTIONAL_LOCKS'] = '0'
        if private:
            self.env = {k: v for k, v in self.env.items() if not k.startswith('GIT_')}
            self.env.update(
                GIT_CONFIG_NOSYSTEM='1',
                GIT_CONFIG_GLOBAL=os.devnull,
                GIT_ATTR_NOSYSTEM='1',
                GIT_OPTIONAL_LOCKS='0',
                GIT_TERMINAL_PROMPT='0',
                LC_ALL='C',
            )

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise PatchConversionError('Patch conversion exceeded its time budget')
        return remaining

    def run(self, *args, data=None, allowed=(0,)):
        try:
            result = subprocess.run(
                ['git', *map(str, args)],
                cwd=self.cwd,
                env=self.env,
                input=data,
                capture_output=True,
                timeout=self.remaining(),
            )
        except subprocess.TimeoutExpired as exc:
            raise PatchConversionError(
                'Git exceeded the patch conversion time budget'
            ) from exc
        if result.returncode not in allowed:
            # Git diagnostics can contain undecodable source bytes; never emit them raw.
            detail = repr(result.stderr[:2048])
            raise PatchConversionError(f'Git failed ({result.returncode}): {detail}')
        return result.stdout


def _read_optional(path):
    return path.read_bytes() if path is not None and path.is_file() else b''


def _attribute_path(git, name):
    if name == 'GIT_ATTR_SYSTEM' and git.env.get('GIT_ATTR_NOSYSTEM', '').lower() in (
        '1',
        'true',
        'yes',
        'on',
    ):
        return None
    result = git.run('var', name, allowed=(0, 1, 128, 129))
    if result:
        if b'\n' in result.rstrip(b'\n'):
            raise PatchConversionError(f'Unsupported multiple paths for {name}')
        path = Path(os.fsdecode(result.rstrip(b'\n')))
        return path if path.is_absolute() else git.cwd / path
    # git var added these paths in newer Git. Support the standard Linux and
    # Conda layouts used by task images without guessing custom installations.
    if name == 'GIT_ATTR_GLOBAL':
        configured = git.run(
            'config', '--path', '--get', 'core.attributesFile', allowed=(0, 1)
        )
        if configured:
            path = Path(os.fsdecode(configured.rstrip(b'\n')))
            return path if path.is_absolute() else git.cwd / path
        return (
            Path(git.env.get('XDG_CONFIG_HOME') or str(Path.home() / '.config'))
            / 'git/attributes'
        )
    executable = Path(os.fsdecode(git.run('--exec-path').rstrip(b'\n')))
    if str(executable) in ('/usr/lib/git-core', '/usr/libexec/git-core'):
        return Path('/etc/gitattributes')
    if executable.name == 'git-core' and executable.parent.name in ('lib', 'libexec'):
        prefix = executable.parent.parent
        if str(prefix) == '/usr/local' or prefix.name in (
            'miniforge3',
            'miniconda3',
            'anaconda3',
            'conda',
        ):
            return prefix / 'etc/gitattributes'
    raise PatchConversionError(
        'Cannot locate system attributes with this Git installation'
    )


def prepare(repo, base, context, whitespace='fix', timeout=600):
    """Snapshot the base and initial evaluator attribute policy."""
    repo, context = Path(repo).resolve(), Path(context).resolve()
    if context == repo or repo in context.parents:
        raise PatchConversionError('Patch context must be outside the repository')
    if whitespace not in ('fix', 'nowarn'):
        raise PatchConversionError('Unsupported whitespace policy')
    source = Git(repo, timeout)
    base = (
        source.run('rev-parse', '--verify', '--end-of-options', f'{base}^{{commit}}')
        .decode('ascii')
        .strip()
    )
    objects = Path(
        os.fsdecode(source.run('rev-parse', '--git-path', 'objects').rstrip(b'\n'))
    )
    if not objects.is_absolute():
        objects = repo / objects
    object_format = (
        source.run('rev-parse', '--show-object-format').decode('ascii').strip()
    )
    whitespace_config = source.run('config', '--get', 'core.whitespace', allowed=(0, 1))
    system = _attribute_path(source, 'GIT_ATTR_SYSTEM')
    global_attrs = _attribute_path(source, 'GIT_ATTR_GLOBAL')
    info = Path(
        os.fsdecode(
            source.run('rev-parse', '--git-path', 'info/attributes').rstrip(b'\n')
        )
    )
    if not info.is_absolute():
        info = repo / info
    context.mkdir(mode=0o700, parents=False, exist_ok=False)
    worktree = context / 'worktree'
    worktree.mkdir()
    private = Git(worktree, source.remaining(), private=True)
    private.run('init', '-q', f'--object-format={object_format}')
    # All writes go to this private repository; only original objects are borrowed.
    (worktree / '.git/objects/info/alternates').write_bytes(
        os.fsencode(objects.resolve()) + b'\n'
    )
    combined_attrs = context / 'global-attributes'
    combined_attrs.write_bytes(
        _read_optional(system) + b'\n' + _read_optional(global_attrs)
    )
    private.run('config', 'core.attributesFile', combined_attrs)
    if whitespace_config:
        private.run(
            'config', 'core.whitespace', whitespace_config.rstrip(b'\n').decode('ascii')
        )
    (worktree / '.git/info/attributes').write_bytes(_read_optional(info))
    private.run('read-tree', base)
    # Worktree attributes take precedence over the index, including untracked
    # attributes installed by the task image. Snapshot before the agent runs.
    paths = source.run(
        'ls-files', '-z', '--cached', '--others', '--', ':(glob)**/.gitattributes'
    )
    for name in sorted(set(paths.split(b'\0')) - {b''}):
        private.remaining()
        path = repo / os.fsdecode(name)
        if not path.is_file() or path.is_symlink():
            continue  # Missing paths use the base index; Git ignores symlinks.
        target = worktree / os.fsdecode(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    (context / 'state.json').write_text(
        json.dumps({'base': base, 'whitespace': whitespace}), encoding='ascii'
    )


def _filter_blocks(raw):
    """Keep raw LF boundaries and the existing Binary files block selection."""
    return [
        block
        for block in re.split(rb'(?m)(?=^diff --git )', raw)
        if block and not any(
            b'Binary files' in line
            for line in block.split(b'\n')
            if not line.startswith(b'diff --git ')
        )
    ]


def _patch_path(git, block, reverse=False):
    # Let Git decode quoted rename/copy paths, including octal and control bytes.
    flags = ('--reverse',) if reverse else ()
    stats = git.run('apply', '--numstat', '-z', *flags, '-', data=block)
    return stats.split(b'\t', 2)[2][:-1]


def _blob(git, tree, path):
    if path is None:
        return None, b''
    oid = git.run('rev-parse', '--verify', tree + ':' + os.fsdecode(path)).strip()
    return oid, git.run('cat-file', 'blob', oid.decode('ascii'))


def _binary_body(old, new):
    """Git literal hunks: zlib data in length-prefixed, padded base85 lines."""
    lines = [b'GIT binary patch']
    # Forward and reverse literals also preserve whitespace-normalized no-ops.
    for blob in (new, old):
        lines.append(f'literal {len(blob)}'.encode('ascii'))
        compressed = zlib.compress(blob)
        for offset in range(0, len(compressed), 52):
            chunk = compressed[offset:offset + 52]
            size = len(chunk)
            prefix = bytes([size + (64 if size <= 26 else 70)])
            lines.append(prefix + base64.b85encode(chunk, pad=True))
        lines.append(b'')
    return b'\n'.join(lines) + b'\n'


def normalize(context, patch, output, timeout=600):
    """Write a UTF-8 patch while retaining operation identity and valid blocks."""
    context = Path(context).resolve()
    state = json.loads((context / 'state.json').read_text(encoding='ascii'))
    worktree = context / 'worktree'
    git = Git(worktree, timeout, private=True)
    blocks = _filter_blocks(Path(patch).read_bytes())
    raw = b''.join(blocks)
    if not raw.strip():
        converted = b''
    else:
        git.run('read-tree', state['base'])
        git.run(
            'apply', '--cached', '--whitespace=' + state['whitespace'], '-', data=raw
        )
        tree = git.run('write-tree').decode('ascii').strip()
        converted_blocks = []
        for block in blocks:
            git.remaining()
            try:
                block.decode('utf-8')
                converted_blocks.append(block)
                continue
            except UnicodeDecodeError:
                pass
            # Input is the already-applied Git diff; retain its operation headers.
            headers = block[:block.index(b'\n--- ')].split(b'\n')
            path = _patch_path(git, block)
            old_path = new_path = path
            if any(line.startswith((b'rename from ', b'copy from ')) for line in headers):
                old_path = _patch_path(git, block, reverse=True)
            if any(line.startswith(b'new file mode ') for line in headers):
                old_path = None
            if any(line.startswith(b'deleted file mode ') for line in headers):
                new_path = None
            old_oid, old_blob = _blob(git, state['base'], old_path)
            new_oid, new_blob = _blob(git, tree, new_path)
            width = len(state['base'])
            old_oid = old_oid or b'0' * width
            new_oid = new_oid or b'0' * width
            index = b'index ' + old_oid + b'..' + new_oid
            header = re.sub(
                rb'(?m)^index [0-9a-f]+\.\.[0-9a-f]+', index, b'\n'.join(headers)
            )
            converted_blocks.append(
                header + b'\n' + _binary_body(old_blob, new_blob)
            )
        converted = b''.join(converted_blocks)
    converted.decode('utf-8')
    git.remaining()
    Path(output).write_bytes(converted)


def main():
    try:
        operation, *args = sys.argv[1:]
        timeout = float(args.pop())
        {'prepare': prepare, 'normalize': normalize}[operation](*args, timeout=timeout)
    except Exception as exc:
        print(
            'Patch conversion failed: ' + ascii(str(exc)),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
