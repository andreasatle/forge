#!/usr/bin/env python3
"""Generate compact markdown indexes for src/baps and tests using AST only."""

import ast
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src" / "baps"
TESTS = ROOT / "tests"
TEST_INDEX = ROOT / "CODEBASE_TEST.md"
IMPORTANT_FIELD_CLASSES = {
    "GameSpec",
    "State",
    "RuntimeContext",
    "RunConfig",
    "RoleConfig",
    "PlayGameContext",
    "VerificationResult",
    "SummarizationContext",
    "ToolDefinition",
    "ToolCall",
    "ToolCallRecord",
}


def _fmt_args(args: ast.arguments) -> str:
    parts: list[str] = []
    all_positional = args.posonlyargs + args.args
    n_defaults = len(args.defaults)
    default_offset = len(all_positional) - n_defaults

    for i, arg in enumerate(args.posonlyargs):
        s = arg.arg
        if arg.annotation:
            s += f": {ast.unparse(arg.annotation)}"
        if i >= default_offset:
            s += f" = {ast.unparse(args.defaults[i - default_offset])}"
        parts.append(s)
    if args.posonlyargs:
        parts.append("/")

    for i, arg in enumerate(args.args):
        idx = len(args.posonlyargs) + i
        s = arg.arg
        if arg.annotation:
            s += f": {ast.unparse(arg.annotation)}"
        if idx >= default_offset:
            s += f" = {ast.unparse(args.defaults[idx - default_offset])}"
        parts.append(s)

    if args.vararg:
        s = f"*{args.vararg.arg}"
        if args.vararg.annotation:
            s += f": {ast.unparse(args.vararg.annotation)}"
        parts.append(s)
    elif args.kwonlyargs:
        parts.append("*")

    for i, arg in enumerate(args.kwonlyargs):
        s = arg.arg
        if arg.annotation:
            s += f": {ast.unparse(arg.annotation)}"
        kw_default = args.kw_defaults[i]
        if kw_default is not None:
            s += f" = {ast.unparse(kw_default)}"
        parts.append(s)

    if args.kwarg:
        s = f"**{args.kwarg.arg}"
        if args.kwarg.annotation:
            s += f": {ast.unparse(args.kwarg.annotation)}"
        parts.append(s)

    return ", ".join(parts)


def _fmt_return(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    return f" -> {ast.unparse(node.returns)}" if node.returns else ""


def _first_doc_line(node: ast.AST) -> str | None:
    if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef, ast.ClassDef, ast.Module)):
        return None
    doc = ast.get_docstring(node)
    if not doc:
        return None
    return doc.strip().splitlines()[0]


def _is_trivial_init_doc(doc: str) -> bool:
    """Return True if an __init__ docstring is too generic to index."""
    return doc.strip().splitlines()[0].lower().startswith("initialize")


def _skip_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if this class method should be excluded from the index."""
    if not node.name.startswith("_"):
        return False
    if node.name == "__init__":
        doc = ast.get_docstring(node)
        if doc and not _is_trivial_init_doc(doc):
            return False  # meaningful __init__ docstring — include
        return True
    return True  # other private/dunder methods — skip


def _base_name(base: ast.expr) -> str:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        parts: list[str] = []
        cur: ast.AST = base
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return ast.unparse(base)


def _is_dataclass(cls: ast.ClassDef) -> bool:
    for dec in cls.decorator_list:
        if _base_name(dec).endswith("dataclass"):
            return True
    return False


def _is_protocol_class(cls: ast.ClassDef) -> bool:
    return any(_base_name(base).endswith("Protocol") for base in cls.bases)


def _class_fields(cls: ast.ClassDef) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    for item in cls.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            fields.append((item.target.id, ast.unparse(item.annotation)))
    return fields


def _should_show_fields(cls: ast.ClassDef, fields: list[tuple[str, str]]) -> bool:
    if not fields:
        return False
    if cls.name in IMPORTANT_FIELD_CLASSES:
        return True
    if _is_protocol_class(cls):
        return True
    base_leafs = {_base_name(base).split(".")[-1] for base in cls.bases}
    if "BaseModel" in base_leafs:
        return True
    if _is_dataclass(cls):
        return True
    return False


def _index_source_file(path: Path) -> list[str]:
    source = path.read_text()
    line_count = len(source.splitlines())
    rel = path.relative_to(ROOT)

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"### {rel} ({line_count} lines)", f"- (parse error: {exc.msg})"]

    imports: set[str] = set()
    classes: list[ast.ClassDef] = []
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                classes.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                functions.append(node)

    module_doc = _first_doc_line(tree)
    header = f"### {rel} ({line_count} lines)"
    header = f"{header} — {module_doc}" if module_doc else f"{header} — MISSING"
    out = [header]

    def append_inline_doc(entry: str, node: ast.AST) -> str:
        doc = _first_doc_line(node)
        if doc is None:
            return f"{entry} — MISSING"
        return f"{entry} — {doc}"

    if classes:
        out.append("- Classes:")
        for cls in classes:
            bases = ", ".join(ast.unparse(b) for b in cls.bases) if cls.bases else ""
            header = f"{cls.name}({bases})" if bases else cls.name
            entry = append_inline_doc(f"  - {header}", cls)
            out.append(entry)
            fields = _class_fields(cls)
            if _should_show_fields(cls, fields):
                out.append("    - Fields:")
                for field_name, field_type in fields:
                    out.append(f"      - {field_name}: {field_type}")
            for item in cls.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if _skip_method(item):
                    continue
                sig = f"{item.name}({_fmt_args(item.args)}){_fmt_return(item)}"
                mentry = append_inline_doc(f"    - {sig}", item)
                out.append(mentry)

    if functions:
        out.append("- Functions:")
        for fn in functions:
            sig = f"{fn.name}({_fmt_args(fn.args)}){_fmt_return(fn)}"
            entry = append_inline_doc(f"  - {sig}", fn)
            out.append(entry)

    if imports:
        out.append(f"- Imports: {', '.join(sorted(imports))}")

    return out


def _is_fixture_fn(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in node.decorator_list:
        name = _base_name(dec)
        if name.endswith("fixture") or name.endswith("pytest.fixture"):
            return True
        if isinstance(dec, ast.Call):
            called = _base_name(dec.func)
            if called.endswith("fixture") or called.endswith("pytest.fixture"):
                return True
    return False


def _index_test_module(path: Path) -> list[str]:
    source = path.read_text()
    line_count = len(source.splitlines())
    rel = path.relative_to(ROOT)
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"### {rel} ({line_count} lines)", f"- (parse error: {exc.msg})"]

    imports: set[str] = set()
    classes: list[ast.ClassDef] = []
    test_functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    fixtures: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.ClassDef):
            classes.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_fixture_fn(node):
                fixtures.append(node)
            elif node.name.startswith("test_"):
                test_functions.append(node)

    module_doc = _first_doc_line(tree)
    header = f"### {rel} ({line_count} lines)"
    header = f"{header} — {module_doc}" if module_doc else f"{header} — MISSING"
    out = [header]

    def append_inline_doc(entry: str, node: ast.AST) -> str:
        doc = _first_doc_line(node)
        if doc is None:
            return f"{entry} — MISSING"
        return f"{entry} — {doc}"

    if classes:
        out.append("- Classes:")
        for cls in classes:
            bases = ", ".join(ast.unparse(b) for b in cls.bases) if cls.bases else ""
            header = f"{cls.name}({bases})" if bases else cls.name
            out.append(append_inline_doc(f"  - {header}", cls))
            fields = _class_fields(cls)
            if _should_show_fields(cls, fields):
                out.append("    - Fields:")
                for field_name, field_type in fields:
                    out.append(f"      - {field_name}: {field_type}")
            for item in cls.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if _is_fixture_fn(item) or item.name.startswith("test_"):
                    sig = f"{item.name}({_fmt_args(item.args)}){_fmt_return(item)}"
                    out.append(append_inline_doc(f"    - {sig}", item))

    if test_functions:
        out.append("- Test Functions:")
        for fn in test_functions:
            sig = f"{fn.name}({_fmt_args(fn.args)}){_fmt_return(fn)}"
            out.append(append_inline_doc(f"  - {sig}", fn))

    if fixtures:
        out.append("- Fixtures:")
        for fx in fixtures:
            sig = f"{fx.name}({_fmt_args(fx.args)}){_fmt_return(fx)}"
            out.append(append_inline_doc(f"  - {sig}", fx))

    if imports:
        out.append(f"- Imports: {', '.join(sorted(imports))}")

    return out


def _is_substantive_source_file(path: Path) -> bool:
    if path.name != "__init__.py":
        return True
    source = path.read_text().strip()
    if not source:
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return True
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.Expr)):
            return True
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id != "__all__":
                    return True
    return False


def _package_dir(path: Path) -> str:
    rel = path.relative_to(SRC)
    parts = rel.parts
    if len(parts) == 1:
        return "root"
    return parts[0]


def _collect_protocols(source_files: list[Path]) -> list[tuple[str, str, str | None]]:
    """Return (name, rel_path, docstring) for all public Protocol classes."""
    entries: list[tuple[str, str, str | None]] = []
    for path in source_files:
        if not _is_substantive_source_file(path):
            continue
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_") and _is_protocol_class(node):
                entries.append((node.name, str(path.relative_to(ROOT)), _first_doc_line(node)))
    return sorted(entries)


def _build_dir_index_lines(pkg_name: str, files: list[Path]) -> list[str]:
    lines: list[str] = [f"# Codebase API Index — src/baps/{pkg_name}", ""]
    for path in sorted(files):
        if not _is_substantive_source_file(path):
            continue
        lines.extend(_index_source_file(path))
        lines.append("")
    return lines


def _build_protocols_index_lines(
    protocols: list[tuple[str, str, str | None]],
) -> list[str]:
    lines: list[str] = ["# Protocol Index — src/baps", ""]
    for name, rel, doc in protocols:
        entry = f"- **{name}** — `{rel}`"
        if doc:
            entry += f" — {doc}"
        lines.append(entry)
    return lines


def _build_test_index_lines(test_files: list[Path]) -> list[str]:
    lines: list[str] = ["# Codebase Test Index — tests", ""]
    for path in test_files:
        lines.extend(_index_test_module(path))
        lines.append("")
    return lines


def _build_master_index_lines(
    api_entries: list[tuple[str, str]],
    proto_entry: tuple[str, str],
    test_entry: tuple[str, str],
) -> list[str]:
    lines = [
        "# Codebase Index",
        "Generated by scripts/index_codebase.py — run to refresh.",
        "",
        "## API Indexes (per package)",
    ]
    for filename, description in api_entries:
        lines.append(f"- {filename} — {description}")
    lines += [
        "",
        "## Protocol Index",
        f"- {proto_entry[0]} — {proto_entry[1]}",
        "",
        "## Test Index",
        f"- {test_entry[0]} — {test_entry[1]}",
    ]
    return lines


def main() -> None:
    stale_patterns = [
        "CODEBASE_API_*.md",
        "CODEBASE_INDEX.md",
        "CODEBASE_TEST.md",
        "CODEBASE_INDEX_*.md",
    ]
    for pattern in stale_patterns:
        for stale in sorted(ROOT.glob(pattern)):
            stale.unlink()
            print(f"Deleted {stale.relative_to(ROOT)}")

    source_files = sorted(SRC.rglob("*.py"))
    test_files = sorted(TESTS.rglob("*.py")) if TESTS.exists() else []

    by_dir: dict[str, list[Path]] = defaultdict(list)
    for path in source_files:
        by_dir[_package_dir(path)].append(path)

    api_entries: list[tuple[str, str]] = []
    for pkg_name in sorted(by_dir):
        if pkg_name == "root":
            continue
        lines = _build_dir_index_lines(pkg_name, by_dir[pkg_name])
        out_path = ROOT / f"CODEBASE_API_{pkg_name}.md"
        out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"Wrote {out_path.relative_to(ROOT)}")
        api_entries.append((out_path.name, f"public API surface for src/baps/{pkg_name}/"))

    protocols = _collect_protocols(source_files)
    proto_lines = _build_protocols_index_lines(protocols)
    proto_path = ROOT / "CODEBASE_API_protocols.md"
    proto_path.write_text("\n".join(proto_lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {proto_path.relative_to(ROOT)}")
    proto_entry = (proto_path.name, "all Protocol classes across the codebase")

    test_lines = _build_test_index_lines(test_files)
    TEST_INDEX.write_text("\n".join(test_lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {TEST_INDEX.relative_to(ROOT)}")
    test_entry = (TEST_INDEX.name, "test functions and fixtures across tests/")

    master_lines = _build_master_index_lines(api_entries, proto_entry, test_entry)
    master_path = ROOT / "CODEBASE_INDEX.md"
    master_path.write_text("\n".join(master_lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {master_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
