"""
Internationalization for FaissDocumentIndex's log/print messages, via `gettext`
(Python's standard module).

The messages in the source code (faiss_index.py, config.py, utils_ocr.py) are written
in English — that's the msgid gettext uses, and also the text shown when there's no
translation for the detected language (gettext's default behavior: with no compiled
catalog for the language, `_()` returns the original msgid unchanged).
locale/pt/LC_MESSAGES/ carries the Portuguese translation.

The language is picked from the machine's locale (environment variables LANGUAGE,
LC_ALL, LC_MESSAGES, LANG, in that order — that's how `gettext.translation()` decides
when `languages` isn't passed explicitly). FAISS_INDEX_LANG lets you force a language
(e.g.: "pt"), independent of the machine's locale.

Compiled catalogs (.mo) live at locale/<language>/LC_MESSAGES/faiss_index.mo. The
corresponding .po (editable source) sits alongside it, for reference/editing. See
locale/faiss_index.pot to add a new language.
"""
import gettext
import os
from pathlib import Path

_DOMAIN = "faiss_index"
_LOCALE_DIR = Path(__file__).resolve().parent / "locale"

_forced_lang = os.environ.get("FAISS_INDEX_LANG")
_languages = [_forced_lang] if _forced_lang else None

_translation = gettext.translation(
    _DOMAIN,
    localedir=str(_LOCALE_DIR),
    languages=_languages,
    fallback=True,  # no catalog for the detected language -> returns the msgid (English) as-is
)

_ = _translation.gettext
