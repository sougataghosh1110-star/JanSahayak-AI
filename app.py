"""
JanSahayak AI - app.py (Phase 1)
Mobile-friendly Streamlit interface for a future government document assistant.

Phase 1 has NO OCR, NO document analysis, NO AI API, NO database, NO login.
Uploaded files stay in memory only and are never saved to disk.
"""

import io

import streamlit as st
from PIL import Image  # Pillow is installed automatically with Streamlit

# ---------------------------------------------------------------------------
# Page settings (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="JanSahayak AI",
    page_icon="🏛️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

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
MAX_FILE_MB = 10  # Keep files small so the app stays fast on phones

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
# Helper functions
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
        "**Privacy notice:** This app is still being tested. Do NOT upload "
        "real personal documents. Use only **sample or redacted documents** "
        "(hide your name, Aadhaar number, PAN, phone number, address, and "
        "bank details). Uploaded files are not saved."
    )


def format_size(size_in_bytes):
    """Turn bytes into friendly text such as '2.4 MB'."""
    if size_in_bytes < 1024:
        return f"{size_in_bytes} bytes"
    if size_in_bytes < 1024 * 1024:
        return f"{size_in_bytes / 1024:.1f} KB"
    return f"{size_in_bytes / (1024 * 1024):.1f} MB"


def preview_image(file_bytes, filename):
    """Check that the image is valid, then show it. Shows an error if not."""
    try:
        # verify() checks the file is a real image; it must be reopened after.
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
            "received successfully. The preview will be improved in a later phase."
        )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def home_tab(language):
    st.subheader("Welcome!")
    st.write(
        "Government forms and notices are often written in difficult language. "
        "**JanSahayak AI** is being built to help every citizen understand "
        "them in simple **Bengali, Hindi, and English**."
    )

    st.markdown("#### What will it do?")
    st.markdown(
        """
        - 📄 **Upload** a government document as a PDF or photo
        - 🌐 **Choose** Bengali, Hindi, or English
        - 💡 **Understand** what the document says and what to do next
        """
    )

    st.info(
        f"**Current status:** Phase 1 (interface only). Selected language: "
        f"**{language}**. No document analysis is available yet."
    )
    show_privacy_warning()
    st.caption("Tap **Upload Document** above to try the uploader.")


def upload_tab(language):
    st.subheader("Upload Document")
    show_privacy_warning()
    st.write(f"Explanation language: **{language}**")

    uploaded_file = st.file_uploader(
        "Choose a PDF, JPG, JPEG, or PNG file",
        type=ALL_TYPES,
        help="Your file stays in memory only while this page is open.",
        key="document_uploader",
    )

    if uploaded_file is None:
        st.info("No file uploaded yet.")
        return

    # Error handling: file size check
    if uploaded_file.size > MAX_FILE_MB * 1024 * 1024:
        st.error(
            f"This file is {format_size(uploaded_file.size)}. "
            f"Please upload a file smaller than {MAX_FILE_MB} MB."
        )
        return

    # Error handling: empty file check
    file_bytes = uploaded_file.getvalue()
    if len(file_bytes) == 0:
        st.error("This file is empty. Please choose a different file.")
        return

    st.success("✅ Document uploaded successfully!")
    st.write(f"**Filename:** {uploaded_file.name}")
    st.write(f"**Size:** {format_size(uploaded_file.size)}")

    st.markdown("#### Preview")
    extension = uploaded_file.name.lower().rsplit(".", 1)[-1]
    if extension in IMAGE_TYPES:
        preview_image(file_bytes, uploaded_file.name)
    elif extension == "pdf":
        preview_pdf(uploaded_file)
    else:
        st.warning("Preview is not available for this file type.")

    st.markdown("#### AI document explanation")
    st.markdown('<span class="badge">Coming Soon</span>', unsafe_allow_html=True)
    st.button(
        "Explain this document (Coming Soon)",
        disabled=True,
        key="explain_button_disabled",
    )
    st.caption(
        "No document analysis is available yet. Nothing is read, explained, or saved."
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
    st.markdown("#### Available in Phase 1")
    st.markdown(
        """
        - Document upload (PDF, JPG, JPEG, PNG)
        - Preview and file details
        - Language selector
        - Privacy notice
        """
    )
    st.markdown("#### Not built yet")
    st.markdown(
        """
        - AI explanations
        - Reading text from documents (OCR)
        - Database and login
        """
    )
    show_privacy_warning()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    # Language selector at the top: easy to reach on a phone
    language = st.selectbox(
        "Language / ভাষা / भाषा",
        LANGUAGES,
        key="language_selector",
    )

    show_hero(TAGLINES[language])

    # Tabs work well on phones: tap a tab to switch pages
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


# Basic error handling: show a friendly message instead of a crash screen
try:
    main()
except Exception as error:
    st.error("Something went wrong. Please refresh the page and try again.")
    st.caption(f"Technical detail: {type(error).__name__}")
