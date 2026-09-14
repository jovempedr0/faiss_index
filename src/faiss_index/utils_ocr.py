import os
import re
import logging
import pdfplumber
import tempfile
import pytesseract
import warnings
from io import BytesIO
from pathlib import Path

from typing import Any

from PIL import ImageEnhance
from pdf2image import convert_from_path
from subprocess import run, CalledProcessError
from concurrent.futures import ThreadPoolExecutor

from . import config
from .i18n import _
from .providers import VisionProvider, OpenAICompatibleChatProvider

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Some OCR-purpose vision models (e.g. Baidu's Unlimited-OCR) emit per-region detections
# as "<|det|>type [x1, y1, x2, y2]<|/det|>text", mixed with free-form preamble text.
_VLM_DETECTION_TAG_RE = re.compile(r'<\|det\|>.*?<\|/det\|>([^<]*)')

_vlm_provider: VisionProvider = None


def set_vlm_provider(provider: VisionProvider) -> None:
    """
    Overrides the VisionProvider used for VLM-based OCR (see `config.OCR_VLM_MODEL`).
    Only needed to plug in a backend that isn't OpenAI-compatible — by default, a
    `providers.OpenAICompatibleChatProvider` is built lazily from `config.OCR_VLM_MODEL`
    (and OPENAI_API_KEY/OPENAI_BASE_URL from the environment).
    """
    global _vlm_provider
    _vlm_provider = provider


def _get_vlm_provider() -> VisionProvider:
    global _vlm_provider
    if _vlm_provider is None:
        _vlm_provider = OpenAICompatibleChatProvider(model=config.OCR_VLM_MODEL, vision_model=config.OCR_VLM_MODEL)
    return _vlm_provider


def convert_any_to_pdf(file_path, output_dir=None):
    """
    Converts any file supported by LibreOffice to PDF.

    Parameters:
        file_path (str): The path to the file to convert.
        output_dir (str, optional): The directory where the PDF will be saved. If None,
                                     the PDF is saved in the same directory as the original file.
    Returns:
        str: The path to the new PDF file.
    """
    if not output_dir:
        output_dir = os.path.dirname(file_path)

    try:
        # Use LibreOffice in "headless" mode (no GUI) for the conversion
        # The '--convert-to pdf' argument converts the file to PDF
        # The '--outdir' argument specifies the output directory
        # The command tries to convert any file to PDF
        command = [
            'libreoffice',
            '--headless',
            '--convert-to', 'pdf',
            file_path,
            '--outdir', output_dir
        ]

        run(command, check=True)

        # Builds the new PDF file's name
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        pdf_path = os.path.join(output_dir, f"{base_name}.pdf")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found after conversion: {pdf_path}")

        return pdf_path

    except FileNotFoundError:
        # Error if LibreOffice isn't found
        raise RuntimeError("LibreOffice not found. Make sure it's installed and accessible on the system PATH.")
    except CalledProcessError as e:
        logger.error(_("Error converting file %(file_path)s using LibreOffice: %(error)s") % {"file_path": file_path, "error": e})
        raise e
    except Exception as e:
        logger.error(_("Unexpected error during conversion of %(file_path)s: %(error)s") % {"file_path": file_path, "error": e})
        raise ValueError(f"Unsupported file format or conversion error: {file_path}")


def extract_text_from_file_ocr_fallback(file_path: str, ocr_dpi: int = 300, max_workers: int = None) -> str:
    """
    Extracts text from a PDF or DOCX file, running OCR as a fallback.

    Parameters:
        file_path (str): Path to the file to process. Must end with .pdf or .docx.
        ocr_dpi (int): DPI resolution for PDF-to-image conversion (default: 200)
        max_workers (int): Max number of workers for parallel processing (None = automatic)

    Returns:
        str: The text extracted from the file, concatenated into a single string with
             line breaks between pages/paragraphs. Returns an empty string on error.
    """
    temp_pdf_path = None

    try:
        if file_path.lower().endswith(".pdf"):
            return process_pdf_file(file_path, ocr_dpi, max_workers)
        else:
            logger.info(_("Converting file to PDF..."))
            temp_pdf_path = convert_any_to_pdf(file_path, os.path.dirname(file_path))

            return process_pdf_file(temp_pdf_path, ocr_dpi, max_workers)

    except Exception as e:
        logger.error(_("Error processing file %(file_path)s: %(error)s") % {"file_path": file_path, "error": e})
        return ""
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
                logger.info(_("Temporary file removed: %(temp_pdf_path)s") % {"temp_pdf_path": temp_pdf_path})
            except OSError as e:
                logger.error(_("Error trying to delete temporary file %(temp_pdf_path)s: %(error)s") % {"temp_pdf_path": temp_pdf_path, "error": e})


def process_pdf_file(file_path: str, ocr_dpi: int, max_workers: int) -> str:
    """
    Processes a PDF file, using OCR as a fallback for problematic pages.

    Parameters:
        file_path (str): Path to the PDF file to process.
        ocr_dpi (int): DPI resolution to use for OCR text extraction.
        max_workers (int): Max number of threads for parallel page processing.

    Returns:
        str: Text extracted from the PDF, including text obtained via OCR for problematic pages.
    """
    all_text, problematic_pages = extract_text_from_pdf(file_path)

    if problematic_pages:
        all_text = process_problematic_pages(
            file_path, all_text, problematic_pages, ocr_dpi, max_workers
        )

    return '\n'.join(all_text)


def extract_text_from_pdf(file_path: str) -> tuple[list[str], list[int]]:
    """
    Extracts the text from each page of a PDF file, identifying problematic pages.

    Parameters:
        file_path (str): Path to the PDF file to process.

    Returns:
        tuple[list[str], list[int]]:
            - A list with the text extracted from each valid page of the PDF.
            - A list with the indices of pages considered problematic (no text, or invalid text patterns).

    Notes:
        A page is considered problematic if it has no extractable text or shows invalid
        text patterns (e.g.: "(cid:"). If an error occurs while extracting a page, its
        index is also added to the list of problematic pages.
    """
    all_text = []
    problematic_pages = []
    MIN_TEXT_LENGTH = 50  # Minimum character count for a page to be considered "valid"

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                text = page.extract_text()
                if text is None or len(text.strip()) < MIN_TEXT_LENGTH or "(cid:" in text:
                    problematic_pages.append(i)
                    continue

                all_text.append(text)
            except Exception as e:
                logger.warning(_("Error extracting text from page %(page)s: %(error)s") % {"page": i, "error": e})
                problematic_pages.append(i)

    return all_text, problematic_pages


def process_problematic_pages(
    file_path: str,
    all_text: list[str],
    problematic_pages: list[int],
    ocr_dpi: int,
    max_workers: int
) -> list[str]:
    """
    Processes problematic pages of a PDF using OCR as a fallback.

    Parameters:
        file_path (str): Path to the input PDF file.
        all_text (list[str]): List with the text extracted from all pages of the PDF.
        problematic_pages (list[int]): List of indices of pages that had text-extraction problems.
        ocr_dpi (int): DPI resolution used to convert the pages into images for OCR.
        max_workers (int): Max number of threads/processes to parallelize OCR.

    Returns:
        list[str]: List of page texts, with problematic pages processed via OCR.
    """
    logger.info(_("Using OCR as fallback for %(count)s problematic pages.") % {"count": len(problematic_pages)})

    with tempfile.TemporaryDirectory() as temp_dir:
        images = convert_pdf_to_images(
            file_path, problematic_pages, temp_dir, ocr_dpi
        )
        ocr_results = run_ocr_on_images(images, max_workers)
        logger.info(_("OCR completed for problematic pages."))
        return merge_ocr_results(all_text, problematic_pages, ocr_results)


def convert_pdf_to_images(
    file_path: str,
    problematic_pages: list[int],
    temp_dir: str,
    ocr_dpi: int
) -> dict[int, Any]:
    """
    Converts specific pages of a PDF file into images.

    Parameters:
        file_path (str): Path to the input PDF file.
        problematic_pages (list[int]): List of indices of the problematic pages to convert (0-based).
        temp_dir (str): Temporary directory where the images will be saved.
        ocr_dpi (int): DPI resolution used for the conversion to images.

    Returns:
        dict[int, Any]: A dictionary where the keys are the converted pages' indices and
            the values are the corresponding image objects.
    """
    if not problematic_pages:
        return {}

    images = convert_from_path(
        file_path,
        dpi=ocr_dpi,
        output_folder=temp_dir,
        first_page=min(problematic_pages) + 1,
        last_page=max(problematic_pages) + 1,
        thread_count=os.cpu_count() or 1
    )

    return dict(zip(problematic_pages, images))


def preprocess_image(img):
    """
    Pre-processes an image to optimize it for OCR.

    This function converts the image to grayscale and increases contrast, making it
    easier for OCR tools to extract text.

    Parameters:
        img (PIL.Image.Image): Input image to process.

    Returns:
        PIL.Image.Image: Processed image, in grayscale and with increased contrast.
    """
    img = img.convert('L')  # Convert to grayscale
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)  # Increase contrast
    return img


def _extract_text_from_vlm_response(content: str) -> str:
    """
    Keeps just the recognized text portions from a VLM OCR response. When the model
    tags each region as "<|det|>type [bbox]<|/det|>text" (see `_VLM_DETECTION_TAG_RE`),
    extracts the text after each tag, in order, discarding the bounding boxes and any
    surrounding preamble/noise. Falls back to the raw (stripped) response when no such
    tags are present, so other vision models can be used too.
    """
    matches = _VLM_DETECTION_TAG_RE.findall(content)
    if matches:
        return "\n".join(m.strip() for m in matches if m.strip())
    return content.strip()


def _ocr_via_vlm(img) -> str:
    """
    Runs OCR on a single image via a vision-capable chat model (`config.OCR_VLM_MODEL`,
    through the VisionProvider set with `set_vlm_provider` or, by default, an
    OpenAI-compatible one) — lets image OCR run against a local server (LM Studio,
    oMLX, vLLM, etc.) serving an OCR-purpose VLM, instead of pytesseract.
    """
    buffer = BytesIO()
    img.save(buffer, format="PNG")

    provider = _get_vlm_provider()
    raw_text = provider.describe_image(buffer.getvalue(), "Extract all text from this image, verbatim.")
    return _extract_text_from_vlm_response(raw_text)


def process_image(img, lang='por'):
    """
    Processes a single image using OCR (Optical Character Recognition). Runs via a
    vision-capable chat model when `config.OCR_VLM_MODEL` is set (see `_ocr_via_vlm`),
    otherwise via pytesseract (the default).

    Parameters:
        img: Image to process (format compatible with pytesseract/PIL).
        lang (str, optional): Language code for pytesseract to use. Defaults to 'por'
            (Portuguese). Not used when OCR runs via `config.OCR_VLM_MODEL`.

    Returns:
        str: Text extracted from the image. Returns an empty string on error.

    Exceptions:
        On error during processing, a warning is logged and an empty string is returned.
    """
    try:
        if config.OCR_VLM_MODEL:
            return _ocr_via_vlm(img)
        img = preprocess_image(img)
        text = pytesseract.image_to_string(
            img,
            lang=lang,
            config='--oem 1 --psm 6'
        )
        return text.strip()
    except Exception as e:
        logger.warning(_("Error in image OCR: %(error)s") % {"error": e})
        return ""


def run_ocr_on_images(images: dict[int, Any], max_workers: int) -> dict[int, str]:
    """
    Runs OCR on a dictionary of images using multiple threads.

    Parameters:
        images (dict[int, Any]): Dictionary where the keys are integer identifiers and
            the values are the images to process.
        max_workers (int): Max number of threads to use for parallel processing.

    Returns:
        dict[int, str]: Dictionary where the keys match the images' identifiers and the
            values are the texts extracted via OCR.
    """
    with ThreadPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
        return {
            i: executor.submit(process_image, img).result()
            for i, img in images.items()
        }


def merge_ocr_results(all_text: list[str], problematic_pages: list[int], ocr_results: dict[int, str]) -> list[str]:
    """
    Merges OCR results into a list of texts, in original page order.

    Parameters:
        all_text (list[str]): Texts of the valid (non-problematic) pages, compacted
            in page order (extract_text_from_pdf skips problematic pages rather than
            leaving a placeholder, so this list is shorter than the page count).
        problematic_pages (list[int]): Indices of pages considered problematic, to be
            filled in from `ocr_results` (dropped if OCR produced no text for them).
        ocr_results (dict[int, str]): Dictionary mapping page indices to texts extracted via OCR.

    Returns:
        list[str]: Texts in original page order, valid pages interleaved with any
            recovered OCR text for the problematic ones.
    """
    problematic_set = set(problematic_pages)
    valid_pages = iter(all_text)
    merged: list[str] = []

    for i in range(len(all_text) + len(problematic_pages)):
        if i in problematic_set:
            text = ocr_results.get(i, "")
            if text:
                merged.append(text)
        else:
            merged.append(next(valid_pages))

    return merged
