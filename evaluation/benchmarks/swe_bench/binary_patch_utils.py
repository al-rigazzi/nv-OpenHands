"""Utilities for handling binary files and patch generation in SWE-bench evaluation."""


def remove_binary_diffs(patch_text):
    """Remove binary file diffs from a git patch.

    Args:
        patch_text (str | bytes): The git patch text or undecoded patch bytes

    Returns:
        str | bytes: The patch with binary diffs removed, preserving its type
    """
    raw = isinstance(patch_text, bytes)
    lines = patch_text.split(b'\n') if raw else patch_text.splitlines()
    if raw:
        lines[:-1] = [line + b'\n' for line in lines[:-1]]
    cleaned_lines = []
    block = []
    is_binary_block = False

    for line in lines:
        if line.startswith(b'diff --git ' if raw else 'diff --git '):
            if block and not is_binary_block:
                cleaned_lines.extend(block)
            block = [line]
            is_binary_block = False
        elif (b'Binary files' if raw else 'Binary files') in line:
            is_binary_block = True
            block.append(line)
        else:
            block.append(line)

    if block and not is_binary_block:
        cleaned_lines.extend(block)
    return (b'' if raw else '\n').join(cleaned_lines)


def remove_binary_files_from_git():
    """Generate a bash command to remove binary files from git staging.

    Returns:
        str: A bash command that removes binary files from git staging
    """
    return """
    for file in $(git status --porcelain | grep -E "^(M| M|\\?\\?|A| A)" | cut -c4-); do
        if [ -f "$file" ] && (file "$file" | grep -q "executable" || git check-attr binary "$file" | grep -q "binary: set"); then
            git rm -f "$file" 2>/dev/null || rm -f "$file"
            echo "Removed: $file"
        fi
    done
    """.strip()
