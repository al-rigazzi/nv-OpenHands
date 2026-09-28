"""Make undecodable diff blocks safe for the existing UTF-8 patch interface."""

import base64
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zlib


def _git(repo, deadline, private=False):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    if private:
        env = {k: v for k, v in env.items() if not k.startswith('GIT_')}
        env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
                   GIT_ATTR_NOSYSTEM='1', GIT_TERMINAL_PROMPT='0', LC_ALL='C')

    def run(*args, data=None, allowed=(0,)):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Patch conversion exceeded its time budget')
        result = subprocess.run(['git', *map(str, args)], cwd=repo, env=env,
                                input=data, capture_output=True, timeout=remaining)
        if result.returncode not in allowed:
            raise RuntimeError(f'Git failed ({result.returncode}): {result.stderr[:2048]!r}')
        return result.stdout

    return run


def _attribute_path(run, name):
    if name == 'GIT_ATTR_SYSTEM' and os.getenv('GIT_ATTR_NOSYSTEM', '').lower() in (
        '1', 'true', 'yes', 'on'
    ):
        return None
    path = run('var', name, allowed=(0, 1, 128, 129)).rstrip(b'\n')
    if not path and name == 'GIT_ATTR_GLOBAL':
        path = run('config', '--path', '--get', 'core.attributesFile', allowed=(0, 1)).rstrip(b'\n')
        if not path:
            return Path(os.getenv('XDG_CONFIG_HOME') or Path.home() / '.config') / 'git/attributes'
    if path:
        if b'\n' in path:
            raise RuntimeError(f'Unsupported multiple paths for {name}')
        return Path(os.fsdecode(path))
    # Older Git lacks git-var attribute paths. Retain the supported task-image layouts.
    executable = Path(os.fsdecode(run('--exec-path').rstrip(b'\n')))
    if str(executable) in ('/usr/lib/git-core', '/usr/libexec/git-core'):
        return Path('/etc/gitattributes')
    prefix = executable.parent.parent
    if executable.name == 'git-core' and executable.parent.name in ('lib', 'libexec') and (
        str(prefix) == '/usr/local' or prefix.name in ('miniforge3', 'miniconda3', 'anaconda3', 'conda')
    ):
        return prefix / 'etc/gitattributes'
    raise RuntimeError('Cannot locate system attributes with this Git installation')


def prepare(repo, base, context, whitespace='fix', timeout=600):
    """Freeze the evaluator's initial policy; never write to the agent repository."""
    repo, context = Path(repo).resolve(), Path(context).resolve()
    if context == repo or repo in context.parents or whitespace not in ('fix', 'nowarn'):
        raise ValueError('Invalid private patch context or whitespace policy')
    deadline = time.monotonic() + timeout
    source = _git(repo, deadline)
    base = source('rev-parse', '--verify', '--end-of-options', f'{base}^{{commit}}').decode().strip()
    context.mkdir(mode=0o700)
    worktree = context / 'worktree'
    worktree.mkdir()
    git = _git(worktree, deadline, private=True)
    git('init', '-q', '--object-format=' + source('rev-parse', '--show-object-format').decode().strip())
    # Borrow the effective source object store, including GIT_OBJECT_DIRECTORY.
    objects = repo / os.fsdecode(source('rev-parse', '--git-path', 'objects').rstrip(b'\n'))
    (worktree / '.git/objects/info/alternates').write_bytes(os.fsencode(objects.resolve()) + b'\n')
    attrs = context / 'global-attributes'

    def read(path):
        path = repo / path if path is not None else None
        return path.read_bytes() if path is not None and path.is_file() else b''

    attrs.write_bytes(b'\n'.join(read(_attribute_path(source, name)) for name in (
        'GIT_ATTR_SYSTEM', 'GIT_ATTR_GLOBAL'
    )))
    git('config', 'core.attributesFile', attrs)
    config = source('config', '--get', 'core.whitespace', allowed=(0, 1)).rstrip(b'\n')
    if config:
        git('config', 'core.whitespace', config.decode('ascii'))
    info = Path(os.fsdecode(source('rev-parse', '--git-path', 'info/attributes').rstrip(b'\n')))
    (worktree / '.git/info/attributes').write_bytes(read(info))
    git('read-tree', base)
    paths = source('ls-files', '-z', '--cached', '--others', '--', ':(glob)**/.gitattributes')
    for name in set(paths.split(b'\0')) - {b''}:
        path = repo / os.fsdecode(name)
        if path.is_file() and not path.is_symlink():
            target = worktree / os.fsdecode(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    if time.monotonic() >= deadline:
        raise TimeoutError('Patch preparation exceeded its time budget')
    (context / 'state.json').write_text(json.dumps({'base': base, 'whitespace': whitespace}))


def _filter_blocks(raw):
    return [block for block in re.split(rb'(?m)(?=^diff --git )', raw) if block and not any(
        b'Binary files' in line for line in block.split(b'\n') if not line.startswith(b'diff --git ')
    )]


def _binary_body(old, new):
    """Git's forward/reverse literal records, including normalized no-op changes."""
    lines = [b'GIT binary patch']
    for blob in (new, old):
        lines.append(f'literal {len(blob)}'.encode())
        compressed = zlib.compress(blob)
        for offset in range(0, len(compressed), 52):
            chunk = compressed[offset:offset + 52]
            prefix = bytes([len(chunk) + (64 if len(chunk) <= 26 else 70)])
            lines.append(prefix + base64.b85encode(chunk, pad=True))
        lines.append(b'')
    return b'\n'.join(lines) + b'\n'


def normalize(context, patch, output, timeout=600):
    """Convert only undecodable blocks; the caller retains the legacy text filter."""
    deadline = time.monotonic() + timeout
    context = Path(context)
    raw = Path(patch).read_bytes()
    blocks = [block for block in re.split(rb'(?m)(?=^diff --git )', raw) if block]
    selected = _filter_blocks(raw)
    bad = []
    for i, block in enumerate(blocks):
        try:
            block.decode('utf-8')
        except UnicodeDecodeError:
            if block in selected:
                bad.append(i)
            else:
                blocks[i] = b''
    if bad:
        state = json.loads((context / 'state.json').read_text())
        git = _git(context / 'worktree', deadline, private=True)
        git('read-tree', state['base'])
        # Apply together: a valid deletion can enable an undecodable file/directory addition.
        git('apply', '--cached', '--whitespace=' + state['whitespace'], '-', data=b''.join(selected))
        tree = git('write-tree').decode().strip()
        for i in bad:
            block = blocks[i]
            headers = block[:block.index(b'\n--- ')].split(b'\n')
            path = git('apply', '--numstat', '-z', '-', data=block).split(b'\t', 2)[2][:-1]
            old_path = path
            if any(line.startswith((b'rename from ', b'copy from ')) for line in headers):
                old_path = git('apply', '--numstat', '-z', '--reverse', '-', data=block).split(b'\t', 2)[2][:-1]
            blobs, ids = [], []
            for revision, name, absent in (
                (state['base'], old_path, b'new file mode '), (tree, path, b'deleted file mode ')
            ):
                if any(line.startswith(absent) for line in headers):
                    oid, blob = b'0' * len(state['base']), b''
                else:
                    oid = git('rev-parse', '--verify', revision + ':' + os.fsdecode(name)).strip()
                    blob = git('cat-file', 'blob', oid.decode('ascii'))
                ids.append(oid)
                blobs.append(blob)
            header = re.sub(rb'(?m)^index [0-9a-f]+\.\.[0-9a-f]+',
                            b'index ' + ids[0] + b'..' + ids[1], b'\n'.join(headers))
            blocks[i] = header + b'\n' + _binary_body(*blobs)
    if time.monotonic() >= deadline:
        raise TimeoutError('Patch conversion exceeded its time budget')
    converted = b''.join(blocks)
    converted.decode('utf-8')
    Path(output).write_bytes(converted)


if __name__ == '__main__':
    try:
        operation, *args = sys.argv[1:]
        timeout = float(args.pop())
        {'prepare': prepare, 'normalize': normalize}[operation](*args, timeout=timeout)
    except Exception as exc:
        print('Patch conversion failed: ' + ascii(str(exc)), file=sys.stderr)
        sys.exit(1)
