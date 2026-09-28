"""Export the chapter registry as a portable, static self-study website.

Only the allow-listed browser assets are copied. The generated site-data.js
contains answers and reference solutions intentionally: this is a self-study
site, not a protected exam. Never put secret grading material in its banks.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import chapters


ROOT = Path(__file__).resolve().parent
STATIC_FILES = ("index.html", "app.js", "worker.js", "styles.css")
VENDOR_DIRS = ("pyodide", "katex", "editor")


def export_data():
    """Return the registry in the format consumed by the static frontend."""
    exported = []
    for descriptor in chapters.list_chapters():
        chapter = chapters.get_chapter(descriptor["id"])
        question_bank = chapter.bank
        answers = {}
        reveals = {}
        for question in question_bank.list_questions():
            question_id = question["id"]
            if question["type"] == "choice":
                answers[question_id] = {
                    "type": "choice",
                    "answer": question["_answer"],
                }
            elif question["type"] == "numeric":
                expected = question["_expected"]
                answers[question_id] = {
                    "type": "numeric",
                    "expected": list(expected) if isinstance(expected, (tuple, list)) else expected,
                    "answer_type": question["answer_type"],
                    "tolerance": question["tolerance"],
                }
            elif question["type"] in ("open", "code"):
                reveals[question_id] = question_bank.reveal(question_id)
        exported.append({
            **descriptor,
            "payload": question_bank.public_payload(),
            "answers": answers,
            "reveals": reveals,
        })
    return {
        "default_chapter_id": chapters.DEFAULT_CHAPTER_ID,
        "chapters": exported,
    }


def build(output):
    output = Path(output).expanduser().resolve()
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError("输出目录不可覆盖项目源码目录或其父目录")
    output.mkdir(parents=True, exist_ok=True)

    for name in STATIC_FILES:
        source = ROOT / name
        if not source.is_file():
            raise FileNotFoundError(source)
        if name == "index.html":
            html = source.read_text(encoding="utf-8")
            # Older local builds referenced the domain root. GitHub project
            # Pages lives under /<repo>/checkpoint/, so keep URLs relative.
            html = re.sub(r'((?:href|src)=["\'])/(vendor/|styles\.css|app\.js)',
                          r'\1./\2', html)
            if 'site-data.js' not in html:
                marker = '<script src="./app.js" defer></script>'
                if html.count(marker) != 1:
                    raise ValueError("index.html 缺少唯一的 app.js 脚本标签")
                html = html.replace(marker,
                                    '<script src="./site-data.js" defer></script>\n  ' + marker)
            (output / name).write_text(html, encoding="utf-8")
        else:
            shutil.copy2(source, output / name)

    for name in VENDOR_DIRS:
        source = ROOT / "vendor" / name
        if source.is_dir():
            shutil.copytree(source, output / "vendor" / name, dirs_exist_ok=True)

    data = export_data()
    # An external script avoids a fragile inline-script escaping dependency.
    (output / "site-data.js").write_text(
        "window.RL_SITE_DATA = " + json.dumps(data, ensure_ascii=False, allow_nan=False) + ";\n",
        encoding="utf-8",
    )
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "dist"),
                        help="static website output directory (default: rl-checkpoint/dist)")
    args = parser.parse_args()
    data = build(args.output)
    print("Built %d chapter(s) in %s" % (len(data["chapters"]), args.output))


if __name__ == "__main__":
    main()
