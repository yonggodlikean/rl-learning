"""Contract tests for the repository-driven hub catalog compiler (v1)."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
HUB_DIR = REPO_ROOT / "hub"
if str(HUB_DIR) not in sys.path:
    sys.path.insert(0, str(HUB_DIR))

import build_catalog  # noqa: E402
import publish_pages  # noqa: E402


REAL_IDS = [
    "easyrl-2.1-2.2.2", "easyrl-2.3",
    "monte-carlo", "stochastic-approximation",
]


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


class FakeBank:
    def __init__(self, questions):
        self._questions = [dict(q) for q in questions]

    def list_questions(self):
        return [dict(q) for q in self._questions]


class FakeChapter:
    def __init__(self, questions, title, reference):
        self.bank = FakeBank(questions)
        self.title = title
        self.reference = reference


class FakeRegistry:
    """Minimal registry shaped like checkpoint/chapters.py."""

    def __init__(self, chapters):
        self._order = []
        self._map = {}
        for cid, title, reference, questions in chapters:
            self._order.append({"id": cid, "title": title, "reference": reference})
            self._map[cid] = FakeChapter(questions, title, reference)

    def list_chapters(self):
        return [dict(entry) for entry in self._order]

    def get_chapter(self, cid):
        return self._map[cid]


def question(qid, qtype="choice", score=5, **extra):
    data = {"id": qid, "type": qtype, "category": "concept",
            "title": "题 %s" % qid, "max_score": score}
    data.update(extra)
    return data


def make_repo(root, domains=None, config=None):
    root = Path(root)
    (root / "checkpoint").mkdir(parents=True, exist_ok=True)
    (root / "hub").mkdir(exist_ok=True)
    for name in ("index.html", "styles.css", "app.js"):
        (root / "hub" / name).write_text("<html>asset</html>" if name.endswith("html")
                                         else "asset", encoding="utf-8")
    if config is None:
        config = {
            "repository_url": "https://github.com/yonggodlikean/rl-learning",
            "branch": "main",
            "checkpoint_base_url": "https://yonggodlikean.github.io/rl-learning/checkpoint/",
            "domains": domains if domains is not None else [
                {"id": "classic_rl", "title": "经典强化学习", "path": "classic_rl"},
            ],
        }
    (root / "hub" / "catalog_config.json").write_text(
        json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return root


def add_domain(root, path):
    (Path(root) / path).mkdir(parents=True, exist_ok=True)


def write_readme_case(base, name, source_name="train.py"):
    case = Path(base) / name
    case.mkdir(parents=True, exist_ok=True)
    (case / "README.md").write_text("# 案例\n", encoding="utf-8")
    (case / source_name).write_text("print('never executed')\n", encoding="utf-8")
    return case


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #


class RealRegistryTest(unittest.TestCase):
    def test_real_catalog_shape_and_order(self):
        data = build_catalog.build_catalog()
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["sources"]["chapters"], "checkpoint/chapters.py")
        self.assertEqual(data["update"]["command"], "python3 hub/build_catalog.py")
        self.assertEqual([c["id"] for c in data["chapters"]], REAL_IDS)
        chapter_counts = [c["count"] for c in data["chapters"]]
        code_counts = [c["code_count"] for c in data["chapters"]]
        self.assertEqual(data["stats"]["chapter_count"], len(REAL_IDS))
        self.assertEqual(data["stats"]["question_count"], sum(chapter_counts))
        self.assertEqual(data["stats"]["code_count"], sum(code_counts))
        self.assertEqual([d["id"] for d in data["domains"]],
                         ["classic_rl", "recommender", "rlhf", "quant_rl",
                          "post_training", "papers", "reports"])
        # Domain case counts reflect actual repo content and may grow.
        self.assertEqual(
            sum(d["case_count"] for d in data["domains"]),
            data["stats"]["case_count"])
        self.assertEqual(data["warnings"], [])

    def test_metadata_allowlist_only(self):
        data = build_catalog.build_catalog()
        blob = json.dumps(data, ensure_ascii=False)
        for banned in ('"public_payload"', '"grade"', '"reveal"', '"answer"',
                       '"solution"', '"tests"', '"prompt"'):
            self.assertNotIn(banned, blob)
        expected_counts = {"easyrl-2.1-2.2.2": 12, "easyrl-2.3": 12,
                           "monte-carlo": 18, "stochastic-approximation": 20}
        for chapter in data["chapters"]:
            self.assertEqual(chapter["count"], expected_counts[chapter["id"]])
            self.assertEqual(chapter["total_score"], 100)
            for q in chapter["questions"]:
                self.assertEqual(set(q), {"id", "title", "type", "category",
                                          "max_score", "entry_point", "url"})

    def test_question_scoped_urls(self):
        data = build_catalog.build_catalog()
        chap = data["chapters"][0]
        self.assertIn("?chapter=easyrl-2.1-2.2.2", chap["checkpoint_url"])
        self.assertTrue(chap["questions"][9]["url"].endswith("&question=q10"))
        self.assertEqual(chap["source_path"], "checkpoint/bank.py")
        self.assertTrue(chap["source_url"].endswith("/blob/main/checkpoint/bank.py"))

    def test_write_catalog_generates_hub_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            dest = build_catalog.write_catalog(repo_root=root, registry=FakeRegistry([]))
            self.assertEqual(dest, (root / "hub" / "catalog.json").resolve())
            self.assertTrue(dest.is_file())


class InjectedRegistryTest(unittest.TestCase):
    def test_new_chapter_same_q01_appears_generically(self):
        registry = FakeRegistry([
            ("alpha-1", "第一章", "ref-a", [question("q01")]),
            ("beta-2", "第二章", "ref-b", [question("q01", "code", 9, entry_point="solve")]),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            data = build_catalog.build_catalog(repo_root=root, registry=registry)
        self.assertEqual([c["id"] for c in data["chapters"]], ["alpha-1", "beta-2"])
        self.assertEqual([c["order"] for c in data["chapters"]], [1, 2])
        self.assertEqual(data["chapters"][0]["questions"][0]["id"], "q01")
        self.assertEqual(data["chapters"][1]["questions"][0]["id"], "q01")
        self.assertEqual(data["chapters"][1]["code_count"], 1)


class CaseDiscoveryTest(unittest.TestCase):
    def _build(self, root, known=REAL_IDS):
        registry = FakeRegistry([(cid, cid, "r", [question("q01")]) for cid in known])
        return build_catalog.build_catalog(repo_root=root, registry=registry)

    def test_readme_case_and_manifest_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            add_domain(root, "classic_rl")
            write_readme_case(root / "classic_rl", "scenarios/example")
            lab_case = root / "classic_rl" / "demos" / "gridworld"
            lab_case.mkdir(parents=True)
            (lab_case / "solve.py").write_text("x=1\n", encoding="utf-8")
            (lab_case / "lab.json").write_text(json.dumps({
                "title": "GridWorld",
                "description": "d",
                "chapter_ids": ["easyrl-2.3"],
                "prerequisites": ["easyrl-2.1-2.2.2"],
                "entrypoint": "solve.py",
                "command": "python solve.py",
                "unknown_key": "ignored",
            }), encoding="utf-8")
            data = self._build(root)
        ids = sorted(c["id"] for c in data["cases"])
        self.assertIn("classic_rl/scenarios/example", ids)
        self.assertIn("classic_rl/demos/gridworld", ids)
        readme_case = next(c for c in data["cases"] if c["discovery"] == "readme")
        manifest_case = next(c for c in data["cases"] if c["discovery"] == "manifest")
        self.assertEqual(readme_case["chapter_ids"], [])
        self.assertEqual(readme_case["status"], "source")
        self.assertEqual(readme_case["readme_path"], "classic_rl/scenarios/example/README.md")
        self.assertEqual(manifest_case["chapter_ids"], ["easyrl-2.3"])
        self.assertEqual(manifest_case["prerequisites"], ["easyrl-2.1-2.2.2"])
        self.assertEqual(manifest_case["entrypoint"], "solve.py")
        self.assertFalse(manifest_case["runtime_verified"])
        self.assertEqual(manifest_case["domain_id"], "classic_rl")
        self.assertEqual(data["stats"]["case_count"], 2)
        self.assertEqual(data["domains"][0]["case_count"], 2)

    def test_empty_root_has_zero_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            add_domain(root, "classic_rl")
            (root / "classic_rl" / ".gitkeep").write_text("", encoding="utf-8")
            data = self._build(root)
        self.assertEqual(data["cases"], [])
        self.assertEqual(data["warnings"], [])

    def test_bare_readme_not_a_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            add_domain(root, "classic_rl")
            bare = root / "classic_rl" / "notes"
            bare.mkdir(parents=True)
            (bare / "README.md").write_text("# no source\n", encoding="utf-8")
            container = root / "classic_rl" / "group"
            container.mkdir(parents=True)
            (container / "child").mkdir()
            data = self._build(root)
        self.assertEqual(data["cases"], [])

    def test_manifest_without_source_is_documented(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            add_domain(root, "classic_rl")
            case = root / "classic_rl" / "docs_only"
            case.mkdir(parents=True)
            (case / "lab.json").write_text(json.dumps({"title": "Docs"}), encoding="utf-8")
            data = self._build(root)
        self.assertEqual(len(data["cases"]), 1)
        self.assertEqual(data["cases"][0]["status"], "documented")
        self.assertEqual(data["cases"][0]["entrypoint"], None)

    def test_unknown_chapter_relations_warn_and_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            add_domain(root, "classic_rl")
            case = root / "classic_rl" / "future"
            case.mkdir(parents=True)
            (case / "run.py").write_text("x=1\n", encoding="utf-8")
            (case / "lab.json").write_text(json.dumps({
                "title": "Future",
                "chapter_ids": ["未注册章节"],
                "prerequisites": ["also-unknown", "easyrl-2.3"],
            }), encoding="utf-8")
            data = self._build(root)
        self.assertEqual(data["cases"][0]["chapter_ids"], [])
        self.assertEqual(data["cases"][0]["prerequisites"], ["easyrl-2.3"])
        messages = " ".join(w["message"] for w in data["warnings"])
        self.assertIn("未知章节关系", messages)

    def test_malformed_manifest_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            add_domain(root, "classic_rl")
            case = root / "classic_rl" / "broken"
            case.mkdir(parents=True)
            (case / "lab.json").write_text("{not json", encoding="utf-8")
            (case / "run.py").write_text("x=1\n", encoding="utf-8")
            (case / "README.md").write_text("# 说明\n", encoding="utf-8")
            data = self._build(root)
        self.assertTrue(any("lab.json" in w["message"] for w in data["warnings"]))
        # README+source fallback still yields a case, not a fake manifest success.
        self.assertEqual(len(data["cases"]), 1)
        self.assertEqual(data["cases"][0]["discovery"], "readme")

    def test_non_utf8_manifest_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            add_domain(root, "classic_rl")
            case = root / "classic_rl" / "binary"
            case.mkdir(parents=True)
            (case / "lab.json").write_bytes(b"\xff\xfe\x00bad")
            (case / "run.py").write_text("x=1\n", encoding="utf-8")
            data = self._build(root)
        self.assertTrue(any("UTF-8" in w["message"] for w in data["warnings"]))
        self.assertEqual(data["cases"], [])


class SafetyTest(unittest.TestCase):
    def _build(self, root):
        registry = FakeRegistry([(cid, cid, "r", [question("q01")]) for cid in REAL_IDS])
        return build_catalog.build_catalog(repo_root=root, registry=registry)

    def test_forbidden_and_hidden_dirs_pruned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            add_domain(root, "classic_rl")
            for forbidden in ("data", "datasets", "model", "__pycache__", ".hidden"):
                base = root / "classic_rl" / forbidden
                base.mkdir(parents=True, exist_ok=True)
                (base / "README.md").write_text("# x\n", encoding="utf-8")
                (base / "leak.py").write_text("secret=1\n", encoding="utf-8")
            nested = root / "classic_rl" / "ok" / "secrets"
            nested.mkdir(parents=True)
            (nested / "leak.py").write_text("k=1\n", encoding="utf-8")
            data = self._build(root)
        self.assertEqual(data["cases"], [])

    def test_symlinked_case_dir_skipped_with_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            add_domain(root, "classic_rl")
            real = Path(tmp) / "outside"
            write_readme_case(real, "linked")
            os.symlink(real / "linked", root / "classic_rl" / "linked")
            data = self._build(root)
        self.assertEqual(data["cases"], [])
        self.assertTrue(any("符号链接" in w["message"] for w in data["warnings"]))

    def test_unsafe_roots_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp, domains=[
                {"id": "esc", "title": "e", "path": "../outside"},
                {"id": "abs", "title": "a", "path": "/etc"},
                {"id": "src", "title": "s", "path": "hub"},
                {"id": "dot", "title": "d", "path": "."},
                {"id": "empty", "title": "x", "path": ""},
                {"id": "ok", "title": "ok", "path": "classic_rl"},
            ])
            add_domain(root, "classic_rl")
            data = self._build(root)
        self.assertEqual([d["id"] for d in data["domains"]], ["ok"])
        self.assertGreaterEqual(len([w for w in data["warnings"]]), 4)

    def test_symlinked_domain_root_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp, domains=[
                {"id": "link", "title": "l", "path": "linked_root"},
                {"id": "ok", "title": "ok", "path": "classic_rl"},
            ])
            add_domain(root, "classic_rl")
            outside = Path(tmp) / "outside_root"
            outside.mkdir()
            os.symlink(outside, root / "linked_root")
            data = self._build(root)
        self.assertEqual([d["id"] for d in data["domains"]], ["ok"])

    def test_duplicate_domains_deduped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp, domains=[
                {"id": "classic_rl", "title": "a", "path": "classic_rl"},
                {"id": "classic_rl", "title": "b", "path": "classic_rl"},
                {"id": "alias", "title": "alias", "path": "classic_rl"},
            ])
            add_domain(root, "classic_rl")
            write_readme_case(root / "classic_rl", "one")
            data = self._build(root)
        self.assertEqual([d["id"] for d in data["domains"]], ["classic_rl"])
        self.assertEqual(data["stats"]["case_count"], 1)

    def test_unsafe_entrypoints_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            add_domain(root, "classic_rl")
            outside = Path(tmp) / "outside.py"
            outside.write_text("x=1\n", encoding="utf-8")
            case = root / "classic_rl" / "escapes"
            case.mkdir(parents=True)
            (case / "run.py").write_text("x=1\n", encoding="utf-8")
            for value in ("../outside.py", "/etc/passwd", "..\\outside.py"):
                (case / "lab.json").write_text(
                    json.dumps({"title": "E", "entrypoint": value}), encoding="utf-8")
                data = self._build(root)
                self.assertIsNone(data["cases"][0]["entrypoint"], value)
                messages = " ".join(w["message"] for w in data["warnings"])
                self.assertIn("entrypoint", messages, value)

    def test_symlinked_entrypoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            add_domain(root, "classic_rl")
            case = root / "classic_rl" / "linked"
            case.mkdir(parents=True)
            target = Path(tmp) / "real.py"
            target.write_text("x=1\n", encoding="utf-8")
            os.symlink(target, case / "link.py")
            (case / "lab.json").write_text(
                json.dumps({"title": "L", "entrypoint": "link.py"}), encoding="utf-8")
            data = self._build(root)
        self.assertIsNone(data["cases"][0]["entrypoint"])
        self.assertTrue(any("符号链接" in w["message"] for w in data["warnings"]))

    def test_unsafe_config_urls_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp, config={
                "repository_url": "javascript:alert(1)",
                "branch": "main",
                "checkpoint_base_url": "https://evil.test/x?a=1#frag",
                "domains": [{"id": "ok", "title": "ok", "path": "classic_rl"}],
            })
            add_domain(root, "classic_rl")
            data = self._build(root)
        self.assertEqual(data["repository"]["url"], build_catalog.DEFAULT_REPOSITORY_URL)
        self.assertEqual(data["domains"][0]["source_url"].count("?"), 0)
        self.assertTrue(data["domains"][0]["source_url"].startswith(
            "https://github.com/yonggodlikean/rl-learning/tree/main/"))


class BuildHubTest(unittest.TestCase):
    def test_build_hub_copies_only_ui_assets_and_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            add_domain(root, "classic_rl")
            write_readme_case(root / "classic_rl", "one")
            harness = Path(tmp) / "harness"
            (root / "checkpoint" / "bank.py").write_text("secret=1\n", encoding="utf-8")
            result = build_catalog.build_hub(harness, repo_root=root,
                                             registry=FakeRegistry([("a", "A", "r", [question("q01")])]))
            out = Path(result["output"])
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ["app.js", "catalog.json", "index.html", "styles.css"])
            self.assertTrue((out / "catalog.json").is_file())
            self.assertFalse(any(p.suffix == ".py" for p in out.rglob("*")))
            self.assertFalse((out / "one").exists())

    def test_build_hub_rejects_source_and_ancestor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            add_domain(root, "classic_rl")
            registry = FakeRegistry([("a", "A", "r", [question("q01")])])
            for bad in (root / "hub", root / "checkpoint", root / "classic_rl", root):
                with self.assertRaises(ValueError):
                    build_catalog.build_hub(bad, repo_root=root, registry=registry)

    def test_build_hub_rejects_symlink_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            add_domain(root, "classic_rl")
            real = Path(tmp) / "real_out"
            real.mkdir()
            link = Path(tmp) / "link_out"
            os.symlink(real, link)
            with self.assertRaises(ValueError):
                build_catalog.build_hub(link, repo_root=root,
                                        registry=FakeRegistry([("a", "A", "r", [question("q01")])]))


class PublisherTest(unittest.TestCase):
    def test_temp_build_pages_and_root_redirect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            add_domain(root, "classic_rl")
            write_readme_case(root / "classic_rl", "one")
            docs = Path(tmp) / "out_docs"
            publish_pages.build_pages(docs=docs, repo_root=root)
            html = (docs / "index.html").read_text(encoding="utf-8")
            self.assertIn("./hub/", html)
            self.assertTrue((docs / "hub" / "catalog.json").is_file())
            self.assertTrue((docs / "hub" / "index.html").is_file())
            self.assertTrue((docs / "checkpoint" / "index.html").is_file())
            self.assertTrue((docs / ".rl-learning-generated").is_file())
            catalog = json.loads((docs / "hub" / "catalog.json").read_text(encoding="utf-8"))
        self.assertEqual(catalog["stats"]["chapter_count"], len(REAL_IDS))

    def test_build_pages_refuses_foreign_docs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            add_domain(root, "classic_rl")
            docs = Path(tmp) / "docs"
            docs.mkdir()
            (docs / "stranger.txt").write_text("keep me\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                publish_pages.build_pages(docs=docs, repo_root=root)
            self.assertTrue((docs / "stranger.txt").is_file())

    def test_build_pages_rejects_symlink_docs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            add_domain(root, "classic_rl")
            real = Path(tmp) / "real"
            real.mkdir()
            link = Path(tmp) / "docs_link"
            os.symlink(real, link)
            with self.assertRaises(RuntimeError):
                publish_pages.build_pages(docs=link, repo_root=root)

    def test_build_pages_rejects_symlink_to_marked_generated_docs(self):
        # A symlink to an *already generated* (marked) docs folder must still be
        # rejected; resolving first would wrongly accept it.
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            add_domain(root, "classic_rl")
            generated = Path(tmp) / "generated_docs"
            publish_pages.build_pages(docs=generated, repo_root=root)
            self.assertTrue((generated / ".rl-learning-generated").is_file())
            before = (generated / "hub" / "catalog.json").read_text(encoding="utf-8")
            link = Path(tmp) / "docs_link"
            os.symlink(generated, link)
            with self.assertRaises(RuntimeError):
                publish_pages.build_pages(docs=link, repo_root=root)
            # target must be untouched
            self.assertEqual(
                (generated / "hub" / "catalog.json").read_text(encoding="utf-8"),
                before)

    def test_build_pages_rejects_symlinked_subtarget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_repo(tmp)
            add_domain(root, "classic_rl")
            docs = Path(tmp) / "docs"
            publish_pages.build_pages(docs=docs, repo_root=root)
            elsewhere = Path(tmp) / "elsewhere"
            elsewhere.mkdir()
            import shutil
            shutil.rmtree(docs / "hub")
            os.symlink(elsewhere, docs / "hub")
            with self.assertRaises(RuntimeError):
                publish_pages.build_pages(docs=docs, repo_root=root)


if __name__ == "__main__":
    unittest.main()
