"""One-shot: retarget the test suite's monkeypatches after the alcor split.

``monkeypatch.setattr(alcor, "name", value)`` worked because alcor was a single
module. With alcor a package, each submodule binds its own copy, so the patch
has to reach all of them -- that is what the ``patch_alcor`` fixture does.

This rewrites the call sites mechanically, re-indenting continuation lines by
the width the prefix shrank, and adds the fixture to the signature of every
test that ends up using it.
"""

import re
from pathlib import Path

CALLS = [
    ('monkeypatch.setattr(alcor_mod, ', 'patch_alcor('),
    ('monkeypatch.setattr(alcor, ', 'patch_alcor('),
    ('monkeypatch.setattr("skycam_utils.alcor.', 'patch_alcor("'),
]

total = 0
for path in sorted(Path("skycam_utils/tests").glob("test_alcor*.py")):
    lines = path.read_text().splitlines(keepends=True)
    out, i, changed = [], 0, 0
    while i < len(lines):
        line = lines[i]
        for old, new in CALLS:
            if old in line:
                break
        else:
            out.append(line)
            i += 1
            continue

        indent = len(line) - len(line.lstrip(" "))
        # continuation lines were aligned under `monkeypatch.setattr(`; realign
        # them under the shorter `patch_alcor(` rather than blindly dedenting
        old_align = indent + old.index("(") + 1
        new_align = indent + new.index("(") + 1

        rewritten = line.replace(old, new)
        # the string form carries the module path inside the literal, so the
        # quote that closed it has to become the argument separator
        if old.endswith("skycam_utils.alcor."):
            rewritten = re.sub(r'(patch_alcor\("[A-Za-z_][A-Za-z0-9_]*)",\s*',
                               r'\1", ', rewritten)
        out.append(rewritten)
        depth = rewritten.count("(") - rewritten.count(")")
        i += 1
        while depth > 0 and i < len(lines):
            cont = lines[i]
            lead = len(cont) - len(cont.lstrip(" "))
            out.append(" " * new_align + cont.lstrip(" ")
                       if lead == old_align else cont)
            depth += cont.count("(") - cont.count(")")
            i += 1
        changed += 1

    text = "".join(out)

    # give every test that now calls the fixture access to it
    def add_fixture(m):
        name, args = m.group(1), m.group(2)
        body_start = m.end()
        # find this function's body to see whether it uses patch_alcor
        nxt = text.find("\ndef ", body_start)
        body = text[body_start:nxt if nxt != -1 else len(text)]
        if "patch_alcor(" not in body or re.search(r'\bpatch_alcor\b', args):
            return m.group(0)
        newargs = (args + ", patch_alcor") if args.strip() else "patch_alcor"
        return f"def {name}({newargs}):"

    text = re.sub(r"def (test_[A-Za-z0-9_]*)\(([^)]*)\):", add_fixture, text)

    if changed:
        path.write_text(text)
        print(f"{path.name}: {changed} call sites")
        total += changed
print(f"total: {total}")
