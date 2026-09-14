"""
Mixin for FaissDocumentIndex: the LLM-calibrated section schema (section name -> text
patterns that mark its start), used by the "sections" strategy when no
`structure_provider` is configured — see `extract_sections`/`register_document_type`.
Composed into the class in `core.py`; not meant to be imported directly by users.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List

from . import constants
from .i18n import _

logger = logging.getLogger(__name__)


class SectionSchemaMixin:

    MAX_SECTION_SAMPLE_DOCS = constants.MAX_SECTION_SAMPLE_DOCS
    MAX_SECTION_SAMPLE_CHARS = constants.MAX_SECTION_SAMPLE_CHARS

    def extract_sections(self, text: str, document_type: str) -> Dict[str, str]:
        """
        Extracts sections from a text based on the calibrated schema for the document type.
        This method doesn't assume any fixed structure: it uses the section schema (section
        name -> text patterns that mark its start) calibrated via LLM and stored in
        `self.section_schemas[document_type]` (see `register_document_type`).

        Parameters:
            text (str): The full document text to process.
            document_type (str): Document type whose section schema will be used.

        Returns:
            Dict[str, str]: A dictionary where the keys are "completo" (full text),
                             "cabecalho" (text before the first recognized section) and
                             the schema's section names, and the values are the
                             corresponding texts.

        Raises:
            ValueError: If no section schema is calibrated for `document_type`.
        """
        schema = self.section_schemas.get(document_type)
        if not schema:
            raise ValueError(
                f"No section schema calibrated for document type '{document_type}'. "
                "Call register_document_type(document_type, sample_texts=...) before using the 'sections' strategy."
            )

        sections = {"completo": text, "cabecalho": ""}
        sections.update({section_name: "" for section_name in schema})

        lines = text.split('\n')
        current_section = "cabecalho"

        for line in lines:
            lower = line.lower().strip()

            for section_name, patterns in schema.items():
                if any(pattern in lower for pattern in patterns):
                    current_section = section_name
                    break

            sections[current_section] += line + "\n"

        return {k: v.strip() for k, v in sections.items() if v.strip()}

    def register_document_type(
        self,
        document_type: str,
        sample_texts: List[str],
        force_recalibrate: bool = False
    ) -> Dict[str, List[str]]:
        """
        Calibrates, via LLM, the section schema of a document type from sample texts,
        and stores the result in cache (`self.section_schemas[document_type]`).
        `build_indices` calls this method automatically when there's no cached or
        on-disk schema for the type; it can also be called manually to recalibrate.

        Parameters:
            document_type (str): Document type to calibrate.
            sample_texts (List[str]): Sample document texts of that type.
            force_recalibrate (bool): If True, recalibrates even if a schema is already cached.

        Returns:
            Dict[str, List[str]]: The calibrated schema (section name -> text patterns).
            Empty dict if the LLM couldn't identify any section.
        """
        if document_type in self.section_schemas and not force_recalibrate:
            logger.info(_("Section schema for '%(document_type)s' is already cached. Skipping recalibration.") % {"document_type": document_type})
            return self.section_schemas[document_type]

        if not sample_texts:
            raise ValueError("At least one sample text must be provided to calibrate the section schema.")

        logger.info(
            _("Calibrating section schema for '%(document_type)s' via LLM (%(model)s) with %(num_samples)s sample document(s)...")
            % {"document_type": document_type, "model": self.section_extraction_model, "num_samples": len(sample_texts)}
        )
        schema = self._infer_section_schema_via_llm(document_type, sample_texts)

        if not schema:
            logger.warning(
                _("The LLM did not identify any section for '%(document_type)s'. The 'sections' strategy will be skipped for this document type.")
                % {"document_type": document_type}
            )

        self.section_schemas[document_type] = schema
        return schema

    def _infer_section_schema_via_llm(self, document_type: str, sample_texts: List[str]) -> Dict[str, List[str]]:
        """
        Calls an OpenAI chat model to infer the recurring structural sections across
        the sample texts and the patterns that mark the start of each one.
        """
        samples = sample_texts[: self.MAX_SECTION_SAMPLE_DOCS]
        samples_block = "\n\n---\n\n".join(text[: self.MAX_SECTION_SAMPLE_CHARS] for text in samples)

        prompt = (
            f"You will analyze {len(samples)} sample document(s) of type '{document_type}'. "
            "Identify the structural sections that repeat across them (for example: header, "
            "facts, grounds, requests, conclusion — but adapt the names to the actual "
            "document type, don't assume it's a legal document). For each section, list "
            "2 to 6 short words or phrases, in the document's language, that usually "
            "appear on the line that marks the start of that section.\n\n"
            f"Sample documents:\n{samples_block}"
        )

        try:
            parsed = self.chat_provider.complete_structured(prompt, constants.SECTION_SCHEMA_JSON_SCHEMA)
        except Exception as e:
            logger.error(_("Failed to calibrate section schema via LLM for '%(document_type)s': %(error)s") % {"document_type": document_type, "error": e}, exc_info=True)
            return {}

        schema: Dict[str, List[str]] = {}
        for section in parsed.get("sections", []):
            name = section.get("name", "").strip().lower().replace(" ", "_")
            patterns = [p.strip().lower() for p in section.get("patterns", []) if p.strip()]
            if name and patterns:
                schema[name] = patterns

        return schema

    def _section_schema_path(self, document_type: str, directory: Path) -> Path:
        return Path(directory) / f"{document_type}_section_schema.json"

    def _save_section_schema(self, document_type: str, output_dir: Path) -> None:
        """Persists to disk the calibrated section schema for `document_type`, if any."""
        schema = self.section_schemas.get(document_type)
        if not schema:
            return
        path = self._section_schema_path(document_type, output_dir)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False, indent=4)
        logger.info(_("Section schema for '%(document_type)s' saved to '%(path)s'.") % {"document_type": document_type, "path": path})

    def _load_section_schema(self, document_type: str, document_dir: Path) -> None:
        """Loads from disk, if it exists, the calibrated section schema for `document_type`."""
        path = self._section_schema_path(document_type, document_dir)
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.section_schemas[document_type] = json.load(f)
            logger.info(_("Section schema for '%(document_type)s' loaded from '%(path)s'.") % {"document_type": document_type, "path": path})
        except Exception as e:
            logger.warning(_("Failed to load section schema from '%(path)s': %(error)s") % {"path": path, "error": e})
