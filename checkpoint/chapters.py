"""Chapter registry for the RL checkpoint.

A *chapter* pairs a chapter id/title with a *bank module* that exposes the
same small public API as :mod:`bank`:

    public_payload()          -> dict served by GET /api/questions
    grade(question_id, ans)   -> dict (raises the bank's grade errors)
    reveal(question_id)       -> dict (raises the bank's reveal errors)
    list_questions()          -> list of internal question dicts
    REFERENCE                 -> human-readable source reference (str)

The existing EasyRL bank is registered here as the default chapter. A future
source-grounded chapter module can be added without touching the server by
calling :func:`register_bank` (see AUTHORING.md).

Both the HTTP server and the static site builder resolve chapters through this
module, so the registry stays the single source of truth for "which chapters
exist". Chapter ids are opaque, stable strings chosen by the chapter author.
"""

import bank as _default_bank

#: Chapter id served when a request omits ``chapter``. Existing behaviour
#: (no ``chapter`` query field / JSON field) maps to this chapter.
DEFAULT_CHAPTER_ID = "easyrl-2.1-2.2.2"


class ChapterError(Exception):
    """Base class for chapter-registry errors."""


class UnknownChapter(ChapterError):
    """Raised when a chapter id is requested that is not registered."""


class Chapter:
    """One registered chapter: an id, a title and a same-API bank module."""

    __slots__ = ("id", "title", "bank", "reference")

    def __init__(self, chapter_id, title, bank_module, reference=None):
        if not isinstance(chapter_id, str) or not chapter_id:
            raise ChapterError("chapter id must be a non-empty string")
        if not isinstance(title, str) or not title:
            raise ChapterError("chapter title must be a non-empty string")
        self.id = chapter_id
        self.title = title
        self.bank = bank_module
        self.reference = reference or getattr(bank_module, "REFERENCE", "")

    def summary(self):
        """Return the small public descriptor used by GET /api/chapters."""
        questions = self.bank.list_questions()
        return {
            "id": self.id,
            "title": self.title,
            "reference": self.reference,
            "count": len(questions),
            "total_score": sum(q["max_score"] for q in questions),
        }

    def __repr__(self):  # pragma: no cover - debug aid
        return "Chapter(id=%r, title=%r)" % (self.id, self.title)


_REGISTRY = {}


def register(chapter, replace=False):
    """Register a :class:`Chapter`. Duplicate ids raise unless ``replace``."""
    if not isinstance(chapter, Chapter):
        raise ChapterError("register() expects a Chapter instance")
    if chapter.id in _REGISTRY and not replace:
        raise ChapterError("chapter already registered: %r" % (chapter.id,))
    _REGISTRY[chapter.id] = chapter
    return chapter


def register_bank(chapter_id, title, bank_module, reference=None, replace=False):
    """Convenience wrapper: build and register a :class:`Chapter`."""
    return register(Chapter(chapter_id, title, bank_module, reference), replace=replace)


def list_chapters():
    """Return descriptors for every registered chapter, in registration order."""
    return [chapter.summary() for chapter in _REGISTRY.values()]


def get_chapter(chapter_id=None):
    """Return a :class:`Chapter` by id. ``None`` selects the default chapter."""
    key = DEFAULT_CHAPTER_ID if chapter_id is None else chapter_id
    try:
        return _REGISTRY[key]
    except KeyError:
        raise UnknownChapter("unknown chapter id: %r" % (chapter_id,))


def resolve_bank(chapter_id=None):
    """Return the bank module for a chapter id (default when ``None``)."""
    return get_chapter(chapter_id).bank


# Register the existing frozen EasyRL bank as the default chapter.
register_bank(
    DEFAULT_CHAPTER_ID,
    "马尔可夫过程与贝尔曼方程 · EasyRL §2.1–§2.2.2",
    _default_bank,
)
