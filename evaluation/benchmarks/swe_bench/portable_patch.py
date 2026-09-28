"""Encode undecodable Git text diffs as binary literals of their recorded blobs."""

import base64
import os
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path


def _binary_body(old, new):
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


def convert(repo, patch, output, timeout=600):
    """Keep text blocks unchanged; undecodable files retain exact staged bytes."""
    deadline = time.monotonic() + timeout

    def git(*args, data=None):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Patch conversion exceeded its time budget')
        result = subprocess.run(['git', *args], cwd=repo, input=data,
                                env=dict(os.environ, GIT_OPTIONAL_LOCKS='0'),
                                capture_output=True, timeout=remaining)
        if result.returncode:
            raise RuntimeError(f'Git failed ({result.returncode}): {result.stderr[:2048]!r}')
        return result.stdout

    blocks = re.split(rb'(?m)(?=^diff --git )', Path(patch).read_bytes())
    for i, block in enumerate(blocks):
        try:
            block.decode('utf-8')
            continue
        except UnicodeDecodeError:
            pass
        if any(b'Binary files' in line for line in block.split(b'\n')
               if not line.startswith(b'diff --git ')):
            blocks[i] = b''  # Preserve the caller's existing binary-notice selection.
            continue
        header = block[:block.index(b'\n--- ')]
        index = re.search(rb'(?m)^index ([0-9a-f]+)\.\.([0-9a-f]+)(?: [0-7]{6})?$', header)
        if not index:
            raise ValueError('Undecodable diff has no blob index')
        git('apply', '--numstat', '-', data=block)  # Validate syntax without applying.
        ids, blobs = [], []
        for side, abbreviated in enumerate(index.groups()):
            absent = b'\n' + (b'new' if side == 0 else b'deleted') + b' file mode ' in header
            if absent:
                if abbreviated.strip(b'0'):
                    raise ValueError('Absent file has a nonzero blob index')
                ids.append(None)
                blobs.append(b'')
            else:
                oid = git('rev-parse', '--verify', abbreviated.decode() + '^{blob}').strip()
                ids.append(oid)
                blobs.append(git('cat-file', 'blob', oid.decode()))
        if not any(ids):
            raise ValueError('Undecodable diff has no existing blob')
        width = len(next(oid for oid in ids if oid))
        ids = [oid or b'0' * width for oid in ids]
        header = re.sub(rb'(?m)^index [0-9a-f]+\.\.[0-9a-f]+',
                        b'index ' + ids[0] + b'..' + ids[1], header)
        # Binary literals bypass text whitespace fixing: encode the recorded blobs verbatim.
        blocks[i] = header + b'\n' + _binary_body(*blobs)
    if time.monotonic() >= deadline:
        raise TimeoutError('Patch conversion exceeded its time budget')
    converted = b''.join(blocks)
    converted.decode('utf-8')
    Path(output).write_bytes(converted)


if __name__ == '__main__':
    try:
        repo, patch, output, timeout = sys.argv[1:]
        convert(repo, patch, output, timeout=float(timeout))
    except Exception as exc:
        print('Patch conversion failed: ' + ascii(str(exc)), file=sys.stderr)
        sys.exit(1)
