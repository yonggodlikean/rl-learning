"""Repository-driven learning catalog compiler (contract v1).

Build-time generation only: the curriculum follows the chapter registry in
``checkpoint/chapters.py``; experiments follow the *actual* repository content
under the configured domain roots. This module never imports or executes
experiment files, never exports answer material, and never talks to GitHub.

Public API:
    build_catalog(repo_root=None, registry=None) -> dict
    write_catalog(output=None, repo_root=None, registry=None) -> Path
    build_hub(output, repo_root=None, registry=None) -> dict
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoint"
if str(CHECKPOINT_DIR) not in sys.path:
    sys.path.insert(0, str(CHECKPOINT_DIR))


SCHEMA_VERSION = 1
SOURCES = {"chapters": "checkpoint/chapters.py", "cases": "hub/catalog_config.json"}
UPDATE = {
    "mode": "build-time",
    "command": "python3 hub/build_catalog.py",
    "publish_command": "python3 checkpoint/publish_pages.py",
}

DEFAULT_REPOSITORY_URL = "https://github.com/yonggodlikean/rl-learning"
DEFAULT_BRANCH = "main"
DEFAULT_CHECKPOINT_BASE_URL = "https://yonggodlikean.github.io/rl-learning/checkpoint/"
DEFAULT_DOMAINS = [
    {"id": "classic_rl", "title": "经典强化学习", "path": "classic_rl"},
    {"id": "recommender", "title": "推荐系统", "path": "recommender"},
    {"id": "rlhf", "title": "RLHF 对齐", "path": "rlhf"},
    {"id": "quant_rl", "title": "量化强化学习", "path": "quant_rl"},
    {"id": "post_training", "title": "后训练", "path": "post_training"},
    {"id": "papers", "title": "论文精读", "path": "papers"},
    {"id": "reports", "title": "研究报告", "path": "reports"},
]

CONFIG_NAME = "hub/catalog_config.json"
SOURCE_EXTS = (".py", ".ipynb", ".js", ".ts", ".sh")
#: Directory names (lower-cased) that must never be walked or configured as a
#: case root: the source site itself, build caches, infrastructure, and any
#: data/model/secret containers.
FORBIDDEN_DIR_NAMES = {
    "checkpoint", "hub", "docs", "dist", "build", "vendor",
    "__pycache__", "node_modules", "venv", "env", "cache", "exports",
    "infrastructure", "data", "datasets", "model", "models", "checkpoints",
    "secret", "secrets",
}
MAX_CASE_DEPTH = 6
MAX_METADATA_BYTES = 64 * 1024

CATALOG_NAME = "catalog.json"
UI_ASSETS = ("index.html", "styles.css", "app.js")
# No supplementary JS is currently coordinated. Only names listed here (if any)
# would ever be copied besides UI_ASSETS and the freshly generated catalog.
SUPPLEMENTARY_ASSETS = ()


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _quote_path(path):
    return "/".join(quote(part, safe="") for part in path.split("/"))


def _blob_url(repo_url, branch, path):
    return "%s/blob/%s/%s" % (repo_url, quote(branch, safe=""), _quote_path(path))


def _tree_url(repo_url, branch, path):
    return "%s/tree/%s/%s" % (repo_url, quote(branch, safe=""), _quote_path(path))


def _checkpoint_url(base, chapter_id):
    return "%s?chapter=%s" % (base, quote(chapter_id, safe=""))


def _question_url(base, chapter_id, question_id):
    return "%s?chapter=%s&question=%s" % (
        base, quote(chapter_id, safe=""), quote(question_id, safe=""),
    )


def _warn(warnings, repo_root, path, message):
    rel = path
    if isinstance(path, Path):
        try:
            rel = path.resolve().relative_to(repo_root).as_posix()
        except ValueError:
            rel = path.as_posix()
    warnings.append({"path": rel, "message": message})


def _is_forbidden_dir(name):
    return name.startswith(".") or name.lower() in FORBIDDEN_DIR_NAMES


def _symlink_component(base, rel):
    """Return the first symlinked path component under *base*, else None."""
    current = Path(base)
    for part in Path(rel).parts:
        if part in ("", "."):
            continue
        current = current / part
        if current.is_symlink():
            return current
    return None


def _read_text_capped(path):
    """Read a small UTF-8 metadata file, capped and decode-checked.

    Returns ``(text, error)``; ``error`` is a message when the file is too
    large or not valid UTF-8, otherwise None.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, "无法读取文件: %s" % exc
    if len(raw) > MAX_METADATA_BYTES:
        return None, "文件超过 %d 字节上限，已跳过" % MAX_METADATA_BYTES
    try:
        return raw.decode("utf-8"), None
    except UnicodeDecodeError as exc:
        return None, "文件不是合法 UTF-8: %s" % exc


def _parse_https(value):
    """Return a normalised safe HTTPS URL or None (no userinfo/query/fragment)."""
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.strip()
    try:
        parts = urlsplit(v)
    except ValueError:
        return None
    if parts.scheme != "https" or not parts.hostname:
        return None
    if parts.username or parts.password:
        return None
    if parts.query or parts.fragment:
        return None
    return v


def _default_registry():
    import chapters  # noqa: WPS433 - resolved against the checkpoint directory
    return chapters


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def _safe_repo_url(value, warnings, repo_root):
    parsed = _parse_https(value)
    if parsed is not None:
        return parsed.rstrip("/")
    _warn(warnings, repo_root, "hub/catalog_config.json",
          "repository_url 必须是干净的 https:// 地址，已回退默认值")
    return DEFAULT_REPOSITORY_URL


def _safe_branch(value, warnings, repo_root):
    if isinstance(value, str) and value.strip():
        return value.strip()
    _warn(warnings, repo_root, "hub/catalog_config.json",
          "branch 必须是非空字符串，已回退默认值")
    return DEFAULT_BRANCH


def _safe_checkpoint_base(value, warnings, repo_root):
    parsed = _parse_https(value)
    if parsed is not None:
        return parsed if parsed.endswith("/") else parsed + "/"
    _warn(warnings, repo_root, "hub/catalog_config.json",
          "checkpoint_base_url 必须是干净的 https:// 地址，已回退默认值")
    return DEFAULT_CHECKPOINT_BASE_URL


def _normalise_relative(raw):
    """Return POSIX relative parts or None when unsafe/empty/'.'."""
    if not isinstance(raw, str):
        return None
    candidate = raw.strip().replace("\\", "/").strip("/")
    parts = [part for part in candidate.split("/") if part not in ("", ".")]
    if not parts or parts == ["."]:
        return None
    if ".." in parts:
        return None
    return parts


def _resolve_domain(repo_root, entry, warnings, seen_ids, seen_paths):
    if not isinstance(entry, dict):
        _warn(warnings, repo_root, "hub/catalog_config.json", "domain 条目必须是对象，已忽略")
        return None
    domain_id = entry.get("id")
    path = entry.get("path")
    title = entry.get("title")
    if not isinstance(domain_id, str) or not domain_id.strip():
        _warn(warnings, repo_root, "hub/catalog_config.json", "domain 缺少合法 id，已忽略")
        return None
    domain_id = domain_id.strip()
    if isinstance(path, str) and os.path.isabs(path.strip().replace("\\", "/")):
        _warn(warnings, repo_root, "hub/catalog_config.json",
              "拒绝绝对 domain 路径: %r" % path)
        return None
    parts = _normalise_relative(path)
    if parts is None:
        _warn(warnings, repo_root, "hub/catalog_config.json",
              "拒绝空/'.'或越出仓库的 domain 路径: %r" % path)
        return None
    if any(_is_forbidden_dir(part) for part in parts):
        _warn(warnings, repo_root, "hub/catalog_config.json",
              "拒绝隐藏/基础设施/data/model/secret 类 domain 路径: %r" % path)
        return None
    linked = _symlink_component(repo_root, Path(*parts))
    if linked is not None:
        _warn(warnings, repo_root, Path("hub/catalog_config.json"),
              "拒绝含符号链接的 domain 路径: %r" % path)
        return None
    raw = "/".join(parts)
    root = (repo_root / raw).resolve()
    if root != repo_root and repo_root not in root.parents:
        _warn(warnings, repo_root, Path("hub/catalog_config.json"),
              "domain 路径越出仓库: %r" % path)
        return None
    if not root.is_dir():
        return None
    if domain_id in seen_ids:
        _warn(warnings, repo_root, Path("hub/catalog_config.json"),
              "重复的 domain id，已忽略: %r" % domain_id)
        return None
    if root in seen_paths:
        _warn(warnings, repo_root, Path("hub/catalog_config.json"),
              "重复的 domain 路径，已忽略: %r" % raw)
        return None
    seen_ids.add(domain_id)
    seen_paths.add(root)
    if not isinstance(title, str) or not title.strip():
        title = domain_id
    return {"id": domain_id, "title": title.strip(), "path": raw}


def _load_config(repo_root, warnings):
    path = repo_root / CONFIG_NAME
    data = {}
    if path.is_symlink():
        _warn(warnings, repo_root, path, "拒绝符号链接配置文件，使用默认配置")
    elif not path.is_file():
        _warn(warnings, repo_root, path, "缺少 catalog_config.json，使用默认配置")
    else:
        text, error = _read_text_capped(path)
        if error is not None:
            _warn(warnings, repo_root, path, "catalog_config.json 无法读取: %s" % error)
            data = {}
        else:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                _warn(warnings, repo_root, path, "catalog_config.json 不是合法 JSON: %s" % exc)
                data = {}
            if data and not isinstance(data, dict):
                _warn(warnings, repo_root, path, "catalog_config.json 顶层必须是对象，使用默认配置")
                data = {}

    config = {
        "repository_url": _safe_repo_url(data.get("repository_url"), warnings, repo_root),
        "branch": _safe_branch(data.get("branch"), warnings, repo_root),
        "checkpoint_base_url": _safe_checkpoint_base(
            data.get("checkpoint_base_url"), warnings, repo_root),
    }

    if "domains" not in data:
        raw_domains = list(DEFAULT_DOMAINS)
    elif isinstance(data.get("domains"), list):
        raw_domains = data["domains"]
    else:
        _warn(warnings, repo_root, path, "domains 必须是数组，使用默认域")
        raw_domains = list(DEFAULT_DOMAINS)

    domains = []
    seen_ids, seen_paths = set(), set()
    for entry in raw_domains:
        resolved_domain = _resolve_domain(repo_root, entry, warnings, seen_ids, seen_paths)
        if resolved_domain is not None:
            domains.append(resolved_domain)
    config["domains"] = domains
    return config


# --------------------------------------------------------------------------- #
# chapters
# --------------------------------------------------------------------------- #


def _bank_source_path(bank, repo_root, warnings, chapter_id):
    raw = getattr(bank, "__file__", None)
    if not raw:
        return None
    try:
        resolved = Path(raw).resolve()
        rel = resolved.relative_to(repo_root).as_posix()
    except (OSError, ValueError):
        _warn(warnings, repo_root, Path("checkpoint/chapters.py"),
              "章节 %r 的来源模块不在仓库内，无法生成 source_url" % chapter_id)
        return None
    return rel


def _build_chapters(repo_root, registry, config, warnings):
    chapters = []
    for order, descriptor in enumerate(registry.list_chapters(), start=1):
        chapter_id = descriptor["id"]
        chapter = registry.get_chapter(chapter_id)
        bank = chapter.bank
        questions = []
        code_count = 0
        for question in bank.list_questions():
            question_type = question.get("type")
            if question_type == "code":
                code_count += 1
            questions.append({
                "id": question["id"],
                "title": question.get("title", ""),
                "type": question_type,
                "category": question.get("category", ""),
                "max_score": question.get("max_score", 0),
                "entry_point": question.get("entry_point"),
                "url": _question_url(config["checkpoint_base_url"], chapter_id, question["id"]),
            })
        source_path = _bank_source_path(bank, repo_root, warnings, chapter_id)
        source_url = None
        if source_path is not None:
            source_url = _blob_url(config["repository_url"], config["branch"], source_path)
        chapters.append({
            "id": chapter_id,
            "title": descriptor.get("title", getattr(chapter, "title", "")),
            "reference": descriptor.get("reference", getattr(chapter, "reference", "")),
            "order": order,
            "count": len(questions),
            "total_score": sum(q["max_score"] for q in questions),
            "code_count": code_count,
            "source_path": source_path,
            "source_url": source_url,
            "checkpoint_url": _checkpoint_url(config["checkpoint_base_url"], chapter_id),
            "questions": questions,
        })
    return chapters


# --------------------------------------------------------------------------- #
# cases
# --------------------------------------------------------------------------- #


def _read_manifest(path):
    text, error = _read_text_capped(path)
    if error is not None:
        return None, "lab.json 无法读取: %s" % error
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, "lab.json 不是合法 JSON: %s" % exc
    if not isinstance(data, dict):
        return None, "lab.json 顶层必须是对象"
    return data, None


def _valid_entrypoint(case_dir, value):
    if value is None:
        return None, None
    if not isinstance(value, str) or not value.strip():
        return None, "entrypoint 必须是相对路径字符串"
    raw = value.strip().replace("\\", "/")
    if raw.startswith("/") or os.path.isabs(raw):
        return None, "拒绝绝对 entrypoint: %r" % value
    if len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha():
        return None, "拒绝绝对 entrypoint: %r" % value
    parts = Path(raw).parts
    if ".." in parts:
        return None, "拒绝越出案例目录的 entrypoint: %r" % value
    linked = _symlink_component(case_dir, raw)
    if linked is not None:
        return None, "entrypoint 路径含符号链接: %r" % value
    target = case_dir / raw
    resolved = target.resolve()
    base = case_dir.resolve()
    if resolved != base and base not in resolved.parents:
        return None, "entrypoint 越出案例目录: %r" % value
    if not target.is_file():
        return None, "entrypoint 不是普通文件: %r" % value
    return raw, None


def _filter_relations(values, known_ids, repo_root, warnings, manifest_path, field):
    if values is None:
        return []
    if not isinstance(values, list):
        _warn(warnings, repo_root, manifest_path,
              "lab.json 的 %s 必须是字符串数组，已忽略" % field)
        return []
    kept = []
    for value in values:
        if not isinstance(value, str):
            _warn(warnings, repo_root, manifest_path,
                  "lab.json 的 %s 含非字符串项，已忽略" % field)
            continue
        if value in known_ids:
            kept.append(value)
        else:
            _warn(warnings, repo_root, manifest_path,
                  "未知章节关系 %s=%r（暂不建立导航，未来构建可识别）" % (field, value))
    return kept


def _read_case(repo_root, case_dir, filenames, known_ids, config, warnings):
    lab = case_dir / "lab.json"
    readme = case_dir / "README.md"
    lab_present = "lab.json" in filenames and not lab.is_symlink()
    readme_present = "README.md" in filenames and not readme.is_symlink()
    sources = sorted(
        name for name in filenames
        if name.lower().endswith(SOURCE_EXTS)
        and not name.startswith(".")
        and not (case_dir / name).is_symlink()
    )
    readme_source_ok = readme_present and bool(sources)
    if not lab_present and not readme_source_ok:
        return None

    manifest = None
    if lab_present:
        manifest, error = _read_manifest(lab)
        if error is not None:
            _warn(warnings, repo_root, lab, error)
    if manifest is None and not readme_source_ok:
        return None

    case_id = case_dir.relative_to(repo_root).as_posix()
    readme_path = readme.relative_to(repo_root).as_posix() if readme_present else None

    if manifest is not None:
        discovery = "manifest"
        title = manifest.get("title")
        if not isinstance(title, str) or not title.strip():
            _warn(warnings, repo_root, lab, "lab.json 缺少 title，回退为目录名")
            title = case_dir.name
        description = manifest.get("description")
        description = description if isinstance(description, str) else ""
        chapter_ids = _filter_relations(
            manifest.get("chapter_ids"), known_ids, repo_root, warnings, lab, "chapter_ids")
        prerequisites = _filter_relations(
            manifest.get("prerequisites"), known_ids, repo_root, warnings, lab, "prerequisites")
        entrypoint, error = _valid_entrypoint(case_dir, manifest.get("entrypoint"))
        if error is not None:
            _warn(warnings, repo_root, lab, error)
        command = manifest.get("command")
        command = command if isinstance(command, str) and command.strip() else None
    else:
        discovery = "readme"
        title = case_dir.name
        description = ""
        chapter_ids = []
        prerequisites = []
        entrypoint = None
        command = None

    status = "source" if (entrypoint is not None or sources) else "documented"
    return {
        "id": case_id,
        "title": title,
        "description": description,
        "domain_id": None,
        "path": case_id,
        "readme_path": readme_path,
        "source_url": _tree_url(config["repository_url"], config["branch"], case_id),
        "readme_url": _blob_url(config["repository_url"], config["branch"], readme_path)
        if readme_path else None,
        "chapter_ids": chapter_ids,
        "prerequisites": prerequisites,
        "entrypoint": entrypoint,
        "command": command,
        "status": status,
        "discovery": discovery,
        "runtime_verified": False,
    }


def _discover_cases(repo_root, domain_root, domain_id, known_ids, config, warnings):
    found = []
    for dirpath, dirnames, filenames in os.walk(domain_root, topdown=True):
        base = Path(dirpath)
        relative = base.relative_to(domain_root)
        depth = len(relative.parts)
        keep = []
        for name in sorted(dirnames):
            if _is_forbidden_dir(name):
                continue
            child = base / name
            if child.is_symlink():
                _warn(warnings, repo_root, child, "跳过符号链接目录")
                continue
            keep.append(name)
        dirnames[:] = keep
        if depth >= MAX_CASE_DEPTH:
            dirnames[:] = []
        case = _read_case(repo_root, base, filenames, known_ids, config, warnings)
        if case is not None:
            case["domain_id"] = domain_id
            found.append(case)
    return found


def _build_cases(repo_root, config, known_ids, warnings):
    domains = []
    cases = []
    seen = set()
    for domain in config["domains"]:
        domain_root = repo_root / domain["path"]
        domain_cases = _discover_cases(
            repo_root, domain_root, domain["id"], known_ids, config, warnings)
        unique = []
        for case in domain_cases:
            if case["id"] in seen:
                _warn(warnings, repo_root, Path(case["id"]),
                      "重复的案例路径（重叠 domain 根），已忽略")
                continue
            seen.add(case["id"])
            unique.append(case)
        cases.extend(unique)
        domains.append({
            "id": domain["id"],
            "title": domain["title"],
            "path": domain["path"],
            "source_url": _tree_url(config["repository_url"], config["branch"], domain["path"]),
            "case_count": len(unique),
        })
    return domains, cases


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #


def build_catalog(repo_root=None, registry=None):
    """Compile the public catalog dict from the registry and repository content."""
    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    reg = registry if registry is not None else _default_registry()
    warnings = []
    config = _load_config(root, warnings)
    chapters = _build_chapters(root, reg, config, warnings)
    known_ids = {chapter["id"] for chapter in chapters}
    domains, cases = _build_cases(root, config, known_ids, warnings)

    question_count = sum(chapter["count"] for chapter in chapters)
    code_count = sum(chapter["code_count"] for chapter in chapters)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "repository": {"url": config["repository_url"], "branch": config["branch"]},
        "sources": dict(SOURCES),
        "update": dict(UPDATE),
        "chapters": chapters,
        "domains": domains,
        "cases": cases,
        "stats": {
            "chapter_count": len(chapters),
            "question_count": question_count,
            "code_count": code_count,
            "case_count": len(cases),
        },
        "warnings": warnings,
    }


def _dump(data):
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def write_catalog(output=None, repo_root=None, registry=None):
    """Write the catalog JSON. Defaults to ``hub/catalog.json`` in the repo."""
    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    data = build_catalog(repo_root=root, registry=registry)
    if output is not None:
        raw = Path(output).expanduser()
        if raw.is_symlink():
            raise ValueError("拒绝写入符号链接目标: %s" % output)
        dest = raw.resolve()
    else:
        dest = root / "hub" / CATALOG_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(_dump(data), encoding="utf-8")
    return dest


def _protected_roots(repo_root, config):
    roots = [repo_root / "hub", repo_root / "checkpoint"]
    for domain in config["domains"]:
        roots.append(repo_root / domain["path"])
    protected = []
    for root in roots:
        try:
            protected.append(root.resolve())
        except OSError:
            continue
    return protected


def _resolve_site_output(output, repo_root, config):
    if output is None or not str(output).strip():
        raise ValueError("必须指定静态输出目录")
    raw = Path(output).expanduser()
    # Lexical (non-resolving) absolute path so system links like /var -> /private/var
    # are not mistaken for user-controlled symlinks.
    raw_abs = Path(os.path.abspath(raw))
    if raw_abs.is_symlink():
        raise ValueError("输出路径本身是符号链接: %s" % raw_abs)

    def _inside(path, base):
        return path == base or base in path.parents

    lex_repo = Path(os.path.abspath(repo_root))
    for candidate in [raw_abs] + list(raw_abs.parents):
        if candidate.is_symlink() and (_inside(candidate, lex_repo)
                                       or _inside(lex_repo, candidate)):
            raise ValueError("输出路径含符号链接组件: %s" % candidate)
    out = raw_abs.resolve()
    if out in repo_root.parents or out == repo_root:
        raise ValueError("输出目录不可覆盖仓库根目录或其父目录: %s" % out)
    for protected in _protected_roots(repo_root, config):
        if out == protected or protected in out.parents:
            raise ValueError("输出目录不可写入源码/内容目录: %s" % out)
    return out


def build_hub(output, repo_root=None, registry=None):
    """Copy the explicit UI assets plus a fresh catalog into a static directory."""
    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    warnings = []
    config = _load_config(root, warnings)
    out = _resolve_site_output(output, root, config)
    hub_dir = root / "hub"

    assets = list(UI_ASSETS) + list(SUPPLEMENTARY_ASSETS)
    for name in assets:
        source = hub_dir / name
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(source)
    for name in assets:
        destination = out / name
        if destination.is_symlink():
            raise ValueError("拒绝覆盖符号链接资源: %s" % destination)

    out.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in assets:
        shutil.copy2(hub_dir / name, out / name)
        copied.append(str(out / name))

    catalog = build_catalog(repo_root=root, registry=registry)
    catalog_path = out / CATALOG_NAME
    if catalog_path.is_symlink():
        raise ValueError("拒绝覆盖符号链接 catalog: %s" % catalog_path)
    catalog_path.write_text(_dump(catalog), encoding="utf-8")
    return {
        "output": str(out),
        "catalog": catalog,
        "warnings": warnings,
        "assets": copied,
        "catalog_path": str(catalog_path),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="catalog JSON output path")
    parser.add_argument("--site-output", help="static site output directory")
    parser.add_argument("--repo-root", help="repository root (default: this checkout)")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root) if args.repo_root else None
    if args.site_output:
        result = build_hub(args.site_output, repo_root=repo_root)
        stats = result["catalog"]["stats"]
        print("Built hub site in %s (%d chapters, %d cases)" % (
            result["output"], stats["chapter_count"], stats["case_count"]))
    if args.output or not args.site_output:
        dest = write_catalog(args.output, repo_root=repo_root)
        print("Wrote catalog to", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
