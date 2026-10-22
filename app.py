"""
JanSahayak AI - app.py (Phase 2: real text extraction)

What this version does:
  - Reads selectable text from PDFs with pypdf.
  - Reads scanned PDFs and photos with Tesseract OCR (via pytesseract).
  - Shows the extracted text. The text is NOT verified.

What it does NOT do yet: AI explanation, chat, database, login.

Privacy:
  - Files and extracted text stay in memory for your browser session only.
  - Nothing is written to disk and nothing is printed to the logs.
"""

import hashlib
import io
import re
import time

import pypdfium2 as pdfium
import pytesseract
import streamlit as st
from PIL import Image, ImageOps
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Page settings (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="JanSahayak AI",
    page_icon="🏛️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Protect the free server from huge images ("decompression bombs").
Image.MAX_IMAGE_PIXELS = 40_000_000

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LANGUAGES = ["English", "বাংলা (Bengali)", "हिन्दी (Hindi)"]

TAGLINES = {
    "English": "Understand government documents in simple language",
    "বাংলা (Bengali)": "সরকারি নথি সহজ ভাষায় বুঝুন",
    "हिन्दी (Hindi)": "सरकारी दस्तावेज़ सरल भाषा में समझें",
}

IMAGE_TYPES = ["jpg", "jpeg", "png"]
ALL_TYPES = ["pdf"] + IMAGE_TYPES

# Limits that protect the free cloud resources
MAX_FILE_MB = 8                # biggest file accepted
MAX_PAGES = 10                 # PDF pages read per document
MAX_TOTAL_SECONDS = 90         # total time allowed per extraction
OCR_TIMEOUT_PER_PAGE = 40      # seconds allowed for one OCR page
MAX_IMAGE_SIDE = 2800          # pictures larger than this are shrunk
PDF_RENDER_SCALE = 2.0         # how sharply scanned PDF pages are drawn
MIN_TEXT_CHARS = 25            # fewer characters than this = "no real text"

# Language of the TEXT INSIDE the document (used by the OCR engine)
OCR_LANGUAGE_OPTIONS = {
    "English": "eng",
    "Bengali + English": "ben+eng",
    "Hindi + English": "hin+eng",
    "Bengali + Hindi + English": "ben+hin+eng",
}
DEFAULT_OCR_FOR_UI = {
    "English": "English",
    "বাংলা (Bengali)": "Bengali + English",
    "हिन्दी (Hindi)": "Hindi + English",
}
OCR_LANGUAGE_NAMES = {"eng": "English", "ben": "Bengali", "hin": "Hindi"}

# Manual rotation choices for photos (None = let the app decide)
ROTATION_OPTIONS = {
    "Automatic": None,
    "Rotate 90° right": 90,
    "Rotate 180°": 180,
    "Rotate 90° left": 270,
}

# ---------------------------------------------------------------------------
# Styling (responsive, works on small phone screens)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-left: 1rem; padding-right: 1rem; }
    .hero {
        background: linear-gradient(120deg, #14305a 0%, #1f4f8f 100%);
        border-left: 8px solid #ff9933;
        padding: 1.4rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .hero h1 { color: #ffffff; margin: 0; font-size: 2rem; line-height: 1.2; }
    .hero p  { color: #e8eef7; margin: 0.5rem 0 0 0; font-size: 1.05rem; }
    .badge {
        display: inline-block;
        background: #fff3e0;
        color: #8a4b00;
        border: 1px solid #ffb74d;
        padding: 0.15rem 0.7rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    @media (max-width: 600px) {
        .hero h1 { font-size: 1.6rem; }
        .hero p  { font-size: 0.95rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------------
def show_hero(tagline):
    """Show the title banner at the top of the app."""
    st.markdown(
        f"""
        <div class="hero">
            <h1>🏛️ JanSahayak AI</h1>
            <p>{tagline}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_privacy_warning():
    """Show the privacy notice."""
    st.warning(
        "**Privacy notice:** This is a public test app. Do NOT upload real "
        "personal documents. Use only **sample or redacted documents** (hide "
        "your name, Aadhaar number, PAN, phone number, address, and bank "
        "details). Files and extracted text are kept in memory for your "
        "session only and are not saved by this app."
    )


def format_size(size_in_bytes):
    """Turn bytes into friendly text such as '2.4 MB'."""
    if size_in_bytes < 1024:
        return f"{size_in_bytes} bytes"
    if size_in_bytes < 1024 * 1024:
        return f"{size_in_bytes / 1024:.1f} KB"
    return f"{size_in_bytes / (1024 * 1024):.1f} MB"


def clean_text(text):
    """Tidy extracted text: remove odd characters and extra blank lines."""
    text = text.replace("\x00", "").replace("\x0c", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def new_result():
    """An empty extraction result."""
    return {
        "text": "",
        "error": None,
        "method": "None",
        "pages_total": 1,
        "pages_read": 0,
        "text_pages": 0,
        "ocr_pages": 0,
        "notes": [],
        "seconds": 0.0,
    }


# ---------------------------------------------------------------------------
# OCR engine check
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_ocr_status():
    """Check once whether Tesseract is installed and which languages it has."""
    try:
        version = str(pytesseract.get_tesseract_version())
        languages = list(pytesseract.get_languages(config=""))
        return {"ok": True, "version": version, "languages": languages}
    except Exception:
        return {"ok": False, "version": "", "languages": []}


def usable_ocr_language(wanted, installed):
    """Keep only the OCR languages that are installed on the server."""
    parts = wanted.split("+")
    available = [p for p in parts if p in installed]
    missing = [p for p in parts if p not in installed]
    return "+".join(available), missing


# ---------------------------------------------------------------------------
# Image preparation and OCR
# ---------------------------------------------------------------------------
def prepare_image(image, rotation_choice, is_photo):
    """Make an image easier for OCR: fix rotation, shrink, grey, contrast."""
    # Phones store rotation inside the photo; this applies it.
    image = ImageOps.exif_transpose(image)

    # Remove transparency (PNG) by placing the image on white paper.
    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, rgba).convert("RGB")
    else:
        image = image.convert("RGB")

    # Shrink very large images to save memory and time.
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

    gray = ImageOps.autocontrast(ImageOps.grayscale(image))

    if is_photo:
        if rotation_choice is None:
            # Automatic: ask Tesseract which way the page is turned.
            try:
                osd = pytesseract.image_to_osd(
                    gray, output_type=pytesseract.Output.DICT, timeout=20
                )
                angle = int(osd.get("rotate", 0))
                confidence = float(osd.get("orientation_conf", 0))
                if angle in (90, 180, 270) and confidence >= 5.0:
                    gray = gray.rotate(-angle, expand=True, fillcolor=255)
            except Exception:
                pass  # If detection fails, keep the image as it is.
        else:
            gray = gray.rotate(-rotation_choice, expand=True, fillcolor=255)

    return gray


def extract_from_image(file_bytes, ocr_lang, rotation_choice):
    """Read text from a JPG/JPEG/PNG image using OCR."""
    result = new_result()
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()
    except Exception:
        result["error"] = (
            "This image could not be opened. The file may be damaged or too "
            "large. Please try another JPG, JPEG, or PNG file."
        )
        return result

    try:
        gray = prepare_image(image, rotation_choice, True)
        text = pytesseract.image_to_string(
            gray, lang=ocr_lang, timeout=OCR_TIMEOUT_PER_PAGE
        )
    except RuntimeError:
        result["error"] = (
            "Reading this image took too long, so it was stopped. Try a "
            "smaller or clearer picture."
        )
        return result
    except Exception:
        result["error"] = "The OCR engine could not read this image."
        return result

    result["pages_read"] = 1
    result["text"] = clean_text(text)
    if result["text"]:
        result["ocr_pages"] = 1
        result["method"] = "OCR (Tesseract)"
    return result


# ---------------------------------------------------------------------------
# PDF reading
# ---------------------------------------------------------------------------
def extract_from_pdf(file_bytes, ocr_lang, ocr_available, force_ocr, start_time):
    """Read a PDF page by page: pypdf for real text, OCR for scanned pages."""
    result = new_result()

    # Step 1: open the PDF
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            try:
                unlocked = reader.decrypt("")
            except Exception:
                unlocked = 0
            if not unlocked:
                result["error"] = (
                    "This PDF is password protected, so it cannot be read."
                )
                return result
        total_pages = len(reader.pages)
    except Exception:
        result["error"] = "This PDF could not be opened. The file may be damaged."
        return result

    if total_pages == 0:
        result["error"] = "This PDF has no pages."
        return result

    result["pages_total"] = total_pages
    pages_to_read = min(total_pages, MAX_PAGES)
    if total_pages > MAX_PAGES:
        result["notes"].append(
            f"This PDF has {total_pages} pages. Only the first {MAX_PAGES} "
            "pages were read to save resources."
        )

    # Step 2: go through the pages one by one
    sections = []
    pdf_doc = None
    try:
        for index in range(pages_to_read):
            if time.monotonic() - start_time > MAX_TOTAL_SECONDS:
                result["notes"].append(
                    f"Stopped after page {index} because the time limit of "
                    f"{MAX_TOTAL_SECONDS} seconds was reached."
                )
                break

            page_number = index + 1
            result["pages_read"] += 1

            # 2a: try the selectable text first (unless OCR is forced)
            page_text = ""
            if not (force_ocr and ocr_available):
                try:
                    page_text = clean_text(reader.pages[index].extract_text() or "")
                except Exception:
                    page_text = ""

            if len(page_text) >= MIN_TEXT_CHARS or (page_text and not ocr_available):
                result["text_pages"] += 1
                sections.append((page_number, page_text))
                continue

            # 2b: no real text on this page, so use OCR if we can
            if not ocr_available:
                result["notes"].append(
                    f"Page {page_number} has no selectable text, and OCR is "
                    "not available on this server."
                )
                continue

            try:
                if pdf_doc is None:
                    pdf_doc = pdfium.PdfDocument(file_bytes)
                page = pdf_doc[index]
                picture = page.render(scale=PDF_RENDER_SCALE).to_pil()
                page.close()

                gray = prepare_image(picture, None, False)
                remaining = MAX_TOTAL_SECONDS - (time.monotonic() - start_time)
                timeout = int(max(5, min(OCR_TIMEOUT_PER_PAGE, remaining)))
                ocr_text = clean_text(
                    pytesseract.image_to_string(gray, lang=ocr_lang, timeout=timeout)
                )
                if ocr_text:
                    result["ocr_pages"] += 1
                    sections.append((page_number, ocr_text))
                else:
                    result["notes"].append(f"No text was found on page {page_number}.")
            except RuntimeError:
                result["notes"].append(
                    f"Page {page_number} took too long to read and was skipped."
                )
            except Exception:
                result["notes"].append(f"Page {page_number} could not be read.")
    finally:
        if pdf_doc is not None:
            try:
                pdf_doc.close()
            except Exception:
                pass

    # Step 3: put the pages together
    if len(sections) == 1 and result["pages_read"] == 1:
        result["text"] = sections[0][1]
    else:
        result["text"] = "\n\n".join(
            f"----- Page {number} -----\n{text}" for number, text in sections
        )

    if result["text_pages"] and result["ocr_pages"]:
        result["method"] = "Selectable PDF text (pypdf) + OCR (Tesseract)"
    elif result["ocr_pages"]:
        result["method"] = "OCR (Tesseract)"
    elif result["text_pages"]:
        result["method"] = "Selectable PDF text (pypdf)"
    return result


def run_extraction(file_bytes, extension, ocr_label, rotation_label, force_ocr):
    """Choose the right method for the file and return the result."""
    status = get_ocr_status()
    wanted = OCR_LANGUAGE_OPTIONS[ocr_label]
    ocr_lang, missing = usable_ocr_language(wanted, status["languages"])
    ocr_available = status["ok"] and bool(ocr_lang)
    start_time = time.monotonic()

    if extension == "pdf":
        result = extract_from_pdf(
            file_bytes, ocr_lang, ocr_available, force_ocr, start_time
        )
    elif not ocr_available:
        result = new_result()
        result["error"] = (
            "The OCR engine is not available on this server, so pictures "
            "cannot be read right now."
        )
    else:
        result = extract_from_image(
            file_bytes, ocr_lang, ROTATION_OPTIONS[rotation_label]
        )

    if ocr_available and missing:
        names = ", ".join(OCR_LANGUAGE_NAMES.get(code, code) for code in missing)
        result["notes"].append(f"OCR language data is missing on the server: {names}.")

    result["seconds"] = round(time.monotonic() - start_time, 1)
    return result


def make_key(file_bytes, settings):
    """A fingerprint of the file and settings. Used to know if results are current."""
    digest = hashlib.sha256(file_bytes)
    digest.update(repr(settings).encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Preview helpers
# ---------------------------------------------------------------------------
def preview_image(file_bytes, filename):
    """Check that the image is valid, then show it."""
    try:
        Image.open(io.BytesIO(file_bytes)).verify()
        image = Image.open(io.BytesIO(file_bytes))
        st.image(image, caption=filename)
    except Exception:
        st.error(
            "This image could not be opened. The file may be damaged. "
            "Please try another JPG, JPEG, or PNG file."
        )


def preview_pdf(uploaded_file):
    """Show a PDF preview if this Streamlit setup supports it."""
    try:
        st.pdf(uploaded_file, height=500)
    except Exception:
        st.info(
            "📄 PDF preview is not available in this setup. Your PDF was "
            "received successfully."
        )


# ---------------------------------------------------------------------------
# Showing the extraction result
# ---------------------------------------------------------------------------
def show_extraction_result(result, key):
    st.markdown("#### Extracted text")

    if result["error"]:
        st.error(result["error"])
        return

    for note in result["notes"]:
        st.warning(note)

    if not result["text"]:
        st.warning(
            "No text could be extracted from this document. Tips: use a "
            "clearer, well-lit photo; keep the page flat and straight; make "
            "sure the document language is selected correctly; or, for a "
            "PDF, try the 'Read with OCR' option."
        )
        return

    st.info(
        "⚠️ This text has **not been verified**. OCR can make mistakes, "
        "especially with Bengali and Hindi, handwriting, stamps, and blurry "
        "photos. Always compare it with the original document."
    )
    st.caption(
        f"Method: {result['method']} · Pages read: {result['pages_read']} of "
        f"{result['pages_total']} · Characters: {len(result['text'])} · "
        f"Time: {result['seconds']} s"
    )

    st.text_area(
        "Extracted text (not verified)",
        value=result["text"],
        height=400,
        key=f"extracted_text_{key[:16]}",
    )
    st.download_button(
        "Download text as .txt",
        data=result["text"].encode("utf-8"),
        file_name="extracted_text.txt",
        mime="text/plain",
        key="download_text_button",
    )
    if st.button("Clear extracted text", key="clear_text_button"):
        st.session_state.pop("extraction", None)
        st.rerun()


# ---------------------------------------------------------------------------
# Pages (tabs)
# ---------------------------------------------------------------------------
def home_tab(language):
    st.subheader("Welcome!")
    st.write(
        "Government forms and notices are often written in difficult language. "
        "**JanSahayak AI** is being built to help every citizen understand "
        "them in simple **Bengali, Hindi, and English**."
    )

    st.markdown("#### What can it do now?")
    st.markdown(
        """
        - 📄 **Upload** a PDF or a photo of a document
        - 🔎 **Read the text** from the document (PDF text or OCR)
        - ⚠️ The text is **not verified**. Always check it against the original.
        """
    )

    st.markdown("#### What is coming next?")
    st.markdown(
        """
        - 💡 Simple explanations in Bengali, Hindi, and English
        - 🤖 An assistant that answers questions about your document
        """
    )

    st.info(
        f"**Current status:** Phase 2 (text extraction). Selected language: "
        f"**{language}**. AI explanation is **Coming Soon**."
    )
    show_privacy_warning()
    st.caption("Tap **Upload Document** above to try it.")


def upload_tab(language):
    st.subheader("Upload Document")
    show_privacy_warning()
    st.write(f"Explanation language: **{language}**")

    uploaded_file = st.file_uploader(
        "Choose a PDF, JPG, JPEG, or PNG file",
        type=ALL_TYPES,
        help=f"Maximum {MAX_FILE_MB} MB. The file stays in memory only.",
        key="document_uploader",
    )

    if uploaded_file is None:
        # The file was removed, so forget any extracted text too.
        st.session_state.pop("extraction", None)
        st.info("No file uploaded yet.")
        st.caption(
            f"Limits: {MAX_FILE_MB} MB per file, first {MAX_PAGES} PDF pages, "
            f"about {MAX_TOTAL_SECONDS} seconds of reading time."
        )
        return

    if uploaded_file.size > MAX_FILE_MB * 1024 * 1024:
        st.error(
            f"This file is {format_size(uploaded_file.size)}. "
            f"Please upload a file smaller than {MAX_FILE_MB} MB."
        )
        return

    file_bytes = uploaded_file.getvalue()
    if len(file_bytes) == 0:
        st.error("This file is empty. Please choose a different file.")
        return

    extension = uploaded_file.name.lower().rsplit(".", 1)[-1]

    st.success("✅ Document uploaded successfully!")
    st.write(f"**Filename:** {uploaded_file.name}")
    st.write(f"**Size:** {format_size(uploaded_file.size)}")

    st.markdown("#### Preview")
    if extension in IMAGE_TYPES:
        preview_image(file_bytes, uploaded_file.name)
    elif extension == "pdf":
        preview_pdf(uploaded_file)
    else:
        st.warning("Preview is not available for this file type.")

    # ----- Text extraction -----------------------------------------------
    st.markdown("#### Read the text")
    status = get_ocr_status()
    if not status["ok"]:
        st.warning(
            "The OCR engine is not available on this server. PDFs with "
            "selectable text can still be read, but scanned PDFs and photos "
            "cannot."
        )

    ocr_options = list(OCR_LANGUAGE_OPTIONS.keys())
    ocr_label = st.selectbox(
        "Language of the text in the document",
        ocr_options,
        index=ocr_options.index(DEFAULT_OCR_FOR_UI[language]),
        help="Choose the language(s) written in the document. More languages make reading slower.",
    )

    rotation_label = "Automatic"
    force_ocr = False
    if extension in IMAGE_TYPES:
        rotation_label = st.selectbox(
            "Image rotation",
            list(ROTATION_OPTIONS.keys()),
            help="Use this if the photo is sideways or upside down and Automatic did not fix it.",
        )
    else:
        force_ocr = st.checkbox(
            "Read with OCR instead of PDF text",
            help="Turn this on if the PDF text looks garbled or has strange symbols.",
        )

    settings = (extension, ocr_label, rotation_label, force_ocr)
    current_key = make_key(file_bytes, settings)

    st.caption(
        f"Limits: first {MAX_PAGES} pages, about {MAX_TOTAL_SECONDS} seconds. "
        "Reading can take up to a minute for scanned documents."
    )

    if st.button("Extract text", type="primary", key="extract_button"):
        with st.spinner("Reading the document. Please wait..."):
            try:
                result = run_extraction(
                    file_bytes, extension, ocr_label, rotation_label, force_ocr
                )
            except Exception:
                result = new_result()
                result["error"] = (
                    "Something went wrong while reading this document. "
                    "Please try another file."
                )
        # Kept in memory for this browser session only.
        st.session_state["extraction"] = {"key": current_key, "result": result}

    saved = st.session_state.get("extraction")
    if saved and saved["key"] == current_key:
        show_extraction_result(saved["result"], current_key)
    elif saved:
        st.info("The file or settings changed. Tap **Extract text** to read again.")

    # ----- Coming Soon: AI explanation -----------------------------------
    st.markdown("#### AI document explanation")
    st.markdown('<span class="badge">Coming Soon</span>', unsafe_allow_html=True)
    st.button(
        "Explain this document (Coming Soon)",
        disabled=True,
        key="explain_button_disabled",
    )
    st.caption(
        "No AI explanation is available yet. The extracted text above is "
        "only text taken from your file. It has not been checked or explained."
    )


def assistant_tab(language):
    st.subheader("AI Assistant")
    st.markdown('<span class="badge">Coming Soon</span>', unsafe_allow_html=True)
    st.write(
        "In a future phase, you will be able to ask questions about your "
        f"document here in **{language}**."
    )
    st.info(
        "The AI Assistant has not been built yet, so the question box below "
        "is switched off."
    )
    st.text_input(
        "Ask a question about your document",
        placeholder="Coming Soon",
        disabled=True,
        key="assistant_question_disabled",
    )
    st.button("Ask (Coming Soon)", disabled=True, key="assistant_ask_disabled")


def about_tab():
    st.subheader("About JanSahayak AI")
    st.write(
        "**JanSahayak** means *a helper of the people*. This project aims to "
        "help Indian citizens understand complicated government documents and "
        "forms in simple Bengali, Hindi, and English."
    )
    st.markdown("#### Available in Phase 2")
    st.markdown(
        """
        - Upload PDF, JPG, JPEG, and PNG files
        - Text extraction from PDFs (pypdf) and from scans and photos (Tesseract OCR)
        - Preview, language selector, and privacy notice
        """
    )
    st.markdown("#### Important")
    st.markdown(
        """
        - Extracted text is **not verified** and may contain mistakes.
        - Files and text are kept in memory for your session only.
        """
    )
    st.markdown("#### Not built yet")
    st.markdown(
        """
        - AI explanations
        - AI Assistant chat
        - Database and login
        """
    )
    show_privacy_warning()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    language = st.selectbox(
        "Language / ভাষা / भाषा",
        LANGUAGES,
        key="language_selector",
    )

    show_hero(TAGLINES[language])

    tab_home, tab_upload, tab_assistant, tab_about = st.tabs(
        ["🏠 Home", "📄 Upload Document", "🤖 AI Assistant", "ℹ️ About"]
    )

    with tab_home:
        home_tab(language)
    with tab_upload:
        upload_tab(language)
    with tab_assistant:
        assistant_tab(language)
    with tab_about:
        about_tab()


# Friendly error screen. Only the error TYPE is shown, never document content.
try:
    main()
except Exception as error:
    st.error("Something went wrong. Please refresh the page and try again.")
    st.caption(f"Technical detail: {type(error).__name__}")

# END OF FILE (if your GitHub copy does not end with this line, it was cut off)
