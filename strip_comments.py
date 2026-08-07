#!/usr/bin/env python3
"""
Strip comments, standalone docstrings, and debug print()/logging calls
from a markdown file containing fenced code blocks (as produced by
repo-to-markdown dumps), while preserving code structure.

Usage:
    python3 strip_comments.py codebase.md codebase-no-comments.md
"""

import io
import re
import sys
import tokenize


def custom_untokenize(tokens):
    """Reconstruct source from a filtered token list without using
    tokenize.untokenize()'s full-mode reconstruction, which fills
    row gaps (e.g. from a deleted multi-line docstring) with
    backslash-newline continuations instead of plain newlines. This
    version fills gaps with plain '\\n' instead.
    """
    out = []
    prev_row, prev_col = 1, 0
    for toktype, tokval, start, end, line in tokens:
        if toktype == tokenize.ENDMARKER:
            continue
        row, col = start
        if row > prev_row:
            out.append("\n" * (row - prev_row))
            prev_col = 0
        if col > prev_col:
            out.append(" " * (col - prev_col))
        out.append(tokval)
        # Derive the new position from the characters we actually
        # emitted, rather than trusting the token's declared `end`
        # (e.g. NEWLINE's `end` stays on the same row even though
        # its text is '\n', which would otherwise cause a doubled
        # blank line here).
        if "\n" in tokval:
            n_newlines = tokval.count("\n")
            prev_row = row + n_newlines
            prev_col = len(tokval) - tokval.rfind("\n") - 1
        else:
            prev_row = row
            prev_col = col + len(tokval)
    return "".join(out)


def fix_empty_blocks(source: str) -> str:
    """After stripping, a block whose *only* statement was a bare
    string literal (removed as a 'docstring') is left with no body,
    which is a SyntaxError. Detect any line ending in ':' whose next
    non-blank line is not more indented, and insert a 'pass' line."""
    lines = source.split("\n")
    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        out.append(line)
        stripped = line.rstrip()
        if stripped.endswith(":") and stripped.strip() != "":
            indent = len(line) - len(line.lstrip(" "))
            j = i + 1
            while j < n and lines[j].strip() == "":
                j += 1
            next_indent = (len(lines[j]) - len(lines[j].lstrip(" "))) if j < n else -1
            if j >= n or next_indent <= indent:
                out.append(" " * (indent + 4) + "pass")
        i += 1
    return "\n".join(out)


def strip_python(source: str) -> str:
    """Remove comments and standalone (expression-statement) string
    literals -- i.e. docstrings -- from Python source using the
    tokenize module, which is string/f-string safe. Also drops blank
    lines left behind and collapses runs of >2 blank lines to 1.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenizeError:
        return source  # fall back untouched if it doesn't parse

    out_tokens = []
    prev_toktype = tokenize.NEWLINE
    for tok in tokens:
        toktype, tokval, start, end, line = tok
        if toktype == tokenize.COMMENT:
            continue
        if toktype == tokenize.STRING and prev_toktype in (
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.INDENT,
            tokenize.DEDENT,
        ):
            # Looks like a standalone string statement (docstring).
            # Only drop it if the *next* meaningful token is a
            # NEWLINE (i.e. it truly is its own statement).
            continue
        out_tokens.append(tok)
        if toktype not in (tokenize.NL, tokenize.COMMENT):
            prev_toktype = toktype

    try:
        result = custom_untokenize(out_tokens)
    except Exception:
        return source

    # Remove simple one-line print(...)/logger.debug(...) calls.
    result = re.sub(
        r"^[ \t]*(print|logger\.(debug|info))\(.*\)\s*$",
        "",
        result,
        flags=re.MULTILINE,
    )

    # Remove ALL blank lines (visual spacing isn't needed for LLM
    # consumption, and every blank line is wasted tokens).
    result = "\n".join(l.rstrip() for l in result.split("\n"))
    result = re.sub(r"\n[ \t]*\n+", "\n", result)
    result = fix_empty_blocks(result)
    return result.strip("\n") + "\n"


def strip_yaml(source: str) -> str:
    """Remove full-line and inline # comments from YAML, keeping
    structure/values intact. Skips '#' inside quotes crudely by only
    stripping when a '#' is preceded by whitespace or is line-start."""
    out_lines = []
    for line in source.split("\n"):
        # crude but effective for typical config yaml: split on
        # ' #' that isn't inside quotes
        if '"' in line or "'" in line:
            out_lines.append(line.rstrip())
            continue
        m = re.search(r"(^|\s)#", line)
        if m:
            line = line[: m.start()].rstrip()
        out_lines.append(line.rstrip())
    result = "\n".join(out_lines)
    result = re.sub(r"\n[ \t]*\n+", "\n", result)
    return result.strip("\n") + "\n"


FENCE_RE = re.compile(r"^```([a-zA-Z0-9_+-]*)\s*$", re.MULTILINE)


def process_markdown(md_text: str) -> str:
    lines = md_text.split("\n")
    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = re.match(r"^```([a-zA-Z0-9_+-]*)\s*$", line)
        if m:
            lang = m.group(1).lower()
            out.append(line)
            i += 1
            block_lines = []
            while i < n and lines[i].strip() != "```":
                block_lines.append(lines[i])
                i += 1
            block_src = "\n".join(block_lines)
            if lang in ("py", "python"):
                block_src = strip_python(block_src).rstrip("\n")
            elif lang in ("yaml", "yml"):
                block_src = strip_yaml(block_src).rstrip("\n")
            out.append(block_src)
            if i < n:
                out.append(lines[i])  # closing ```
                i += 1
        else:
            out.append(line)
            i += 1
    return "\n".join(out)


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 strip_comments.py <input.md> <output.md>")
        sys.exit(1)
    src_path, dst_path = sys.argv[1], sys.argv[2]
    with open(src_path, "r", encoding="utf-8") as f:
        text = f.read()
    result = process_markdown(text)
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"Wrote {dst_path}")


if __name__ == "__main__":
    main()
