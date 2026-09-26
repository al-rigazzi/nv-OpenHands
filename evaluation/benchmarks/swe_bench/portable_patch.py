"""Represent non-UTF-8 text hunks as standard Git binary patches.

The private context snapshots the evaluator's initial attributes and whitespace
policy. Binary hunks require that evaluator's base file contents. This is patch
conversion for a known apply policy, not a transport encoding for arbitrary
consumers. The agent repository is never modified by this module.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


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
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.remaining(),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PatchConversionError(
                'Git exceeded the patch conversion time budget'
            ) from exc
        if result.returncode not in allowed:
            # Git diagnostics can contain undecodable source bytes; never emit them raw.
            detail = result.stderr[:2048].decode('ascii', errors='backslashreplace')
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


def prepare(repo, base, context, whitespace='fix', timeout=60):
    """Snapshot a clean base and the initial evaluator attribute policy."""
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
    source.run('diff', '--quiet', '--no-ext-diff', '--no-textconv', base, '--')
    objects = Path(
        os.fsdecode(source.run('rev-parse', '--git-path', 'objects').rstrip(b'\n'))
    )
    if not objects.is_absolute():
        objects = repo / objects
    object_format = (
        source.run('rev-parse', '--show-object-format').decode('ascii').strip()
    )
    if object_format not in ('sha1', 'sha256'):
        raise PatchConversionError('Unsupported Git object format')
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
        json.dumps(
            {
                'version': 1,
                'base': base,
                'whitespace': whitespace,
                'object_format': object_format,
            },
            ensure_ascii=True,
        ),
        encoding='ascii',
    )


def _filter_blocks(raw):
    """Match the existing byte-mode Binary files block selection, including LF."""
    lines = raw.split(b'\n')
    lines[:-1] = [line + b'\n' for line in lines[:-1]]
    blocks, block, binary = [], [], False
    for line in lines:
        if line.startswith(b'diff --git '):
            if block and not binary:
                blocks.append(b''.join(block))
            block, binary = [line], False
        else:
            if b'Binary files' in line:
                binary = True
            block.append(line)
    if block and not binary:
        blocks.append(b''.join(block))
    return blocks


def _unquote(path):
    if not path.startswith(b'"'):
        return path
    if not path.endswith(b'"'):
        raise PatchConversionError('Malformed quoted Git path')
    result = bytearray()
    escapes = {
        ord(k): v for k, v in zip('abfnrtv\\"', (7, 8, 12, 10, 13, 9, 11, 92, 34))
    }
    i = 1
    while i < len(path) - 1:
        value = path[i]
        i += 1
        if value != 92:
            result.append(value)
            continue
        if i >= len(path) - 1:
            raise PatchConversionError('Malformed Git path escape')
        match = re.match(rb'[0-7]{1,3}', path[i:-1])
        if match:
            value = int(match[0], 8)
            if value > 255:
                raise PatchConversionError('Invalid Git path byte')
            result.append(value)
            i += len(match[0])
        elif path[i] in escapes:
            result.append(escapes[path[i]])
            i += 1
        else:
            raise PatchConversionError('Unsupported Git path escape')
    return bytes(result)


def _blob(git, tree, path):
    if path is None:
        return None, b''
    listing = git.run('ls-tree', '-z', tree, '--', ':(literal)' + os.fsdecode(path))
    entries = listing.rstrip(b'\0').split(b'\0') if listing else []
    if len(entries) != 1:
        raise PatchConversionError('Patch path missing or ambiguous in expected tree')
    metadata, actual_path = entries[0].split(b'\t', 1)
    mode, kind, oid = metadata.split()
    if (
        actual_path != path
        or kind != b'blob'
        or mode not in (b'100644', b'100755', b'120000')
    ):
        raise PatchConversionError('Unsupported patch file type')
    return oid, git.run('cat-file', 'blob', oid.decode('ascii'))


def _binary_body(git, directory, old, new):
    directory.mkdir(exist_ok=True)
    old_path, new_path = directory / 'old', directory / 'new'
    # The binary representation repo has no attribute relationship to task files.
    if not (directory / '.git').exists():
        git.run('init', '-q', directory)
        (directory / '.git/info/attributes').write_text('* -diff\n', encoding='ascii')
    original_cwd = git.cwd
    try:
        git.cwd = directory
        old_path.write_bytes(b'' if old == new else old)
        new_path.write_bytes(new)
        patch = git.run(
            'diff',
            '--no-index',
            '--binary',
            '--no-ext-diff',
            '--no-textconv',
            '--',
            'old',
            'new',
            allowed=(0, 1),
        )
    finally:
        git.cwd = original_cwd
    marker = b'GIT binary patch\n'
    if marker not in patch:
        raise PatchConversionError('Git did not produce a binary representation')
    body = patch.split(marker, 1)[1]
    if old == new:
        # Whitespace-only changes may normalize to the original nonempty blob.
        # Keep a nonempty, reversible no-op patch so existing patch_exists logic
        # does not turn a successfully applicable proposal into an empty result.
        literal = body.split(b'\n\n', 1)[0]
        body = literal + b'\n\n' + literal + b'\n\n'
    return marker + body


def normalize(context, patch, output, timeout=60):
    """Write a UTF-8 patch while retaining operation identity and valid blocks."""
    context = Path(context).resolve()
    state = json.loads((context / 'state.json').read_text(encoding='ascii'))
    if state.get('version') != 1 or state.get('whitespace') not in ('fix', 'nowarn'):
        raise PatchConversionError('Unsupported patch context')
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
            if not block.startswith(b'diff --git '):
                raise PatchConversionError('Non-UTF-8 bytes outside a Git file block')
            lines = block.split(b'\n')
            try:
                content_start = next(
                    i for i, line in enumerate(lines) if line.startswith(b'--- ')
                )
            except StopIteration as exc:
                raise PatchConversionError(
                    'Unsupported undecodable patch without text hunks'
                ) from exc
            headers = lines[:content_start]
            b'\n'.join(headers).decode(
                'utf-8'
            )  # Paths and metadata must remain valid text.
            stats = git.run('apply', '--numstat', '-z', '-', data=block)
            fields = stats.split(b'\t', 2)
            if (
                len(fields) != 3
                or fields[2].count(b'\0') != 1
                or not fields[2].endswith(b'\0')
            ):
                raise PatchConversionError('Unsupported multi-path patch block')
            path = fields[2][:-1]
            old_path = new_path = path
            for line in headers:
                if line.startswith((b'rename from ', b'copy from ')):
                    old_path = _unquote(line.split(b' from ', 1)[1])
                if line.startswith(b'new file mode '):
                    old_path = None
                if line.startswith(b'deleted file mode '):
                    new_path = None
            old_oid, old_blob = _blob(git, state['base'], old_path)
            new_oid, new_blob = _blob(git, tree, new_path)
            width = 40 if state['object_format'] == 'sha1' else 64
            old_oid = old_oid or b'0' * width
            new_oid = new_oid or b'0' * width
            index = b'index ' + old_oid + b'..' + new_oid
            replaced = False
            for i, line in enumerate(headers):
                if line.startswith(b'index '):
                    pieces = line.split()
                    headers[i] = index + (b' ' + pieces[2] if len(pieces) == 3 else b'')
                    replaced = True
            if not replaced:
                raise PatchConversionError('Patch lacks index metadata')
            converted_blocks.append(
                b'\n'.join(headers)
                + b'\n'
                + _binary_body(git, context / 'binary', old_blob, new_blob)
            )
        converted = b''.join(converted_blocks)
    converted.decode('utf-8')
    git.remaining()
    output = Path(output)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=output.name + '.', delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(converted)
            handle.close()
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    initial = commands.add_parser('prepare')
    initial.add_argument('--repo', required=True)
    initial.add_argument('--base', required=True)
    initial.add_argument('--context', required=True)
    initial.add_argument('--whitespace', choices=('fix', 'nowarn'), default='fix')
    final = commands.add_parser('normalize')
    final.add_argument('--context', required=True)
    final.add_argument('--patch', required=True)
    final.add_argument('--output', required=True)
    for command in (initial, final):
        command.add_argument('--timeout', type=float, default=60)
    args = vars(parser.parse_args())
    operation = args.pop('operation')
    try:
        (prepare if operation == 'prepare' else normalize)(**args)
    except Exception as exc:
        print(
            ('Patch conversion failed: ' + str(exc))
            .encode('ascii', 'backslashreplace')
            .decode('ascii'),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
