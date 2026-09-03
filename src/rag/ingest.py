from pathlib import Path
from pypdf import PdfReader
from docx import Document as Docx
import sys

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DOCUMENTS_DIR


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def load_document(path: Path) -> str:
    """Load document content based on file type."""
    # Lowercased to match how SUPPORTED_EXTENSIONS is checked, so a file named
    # NOTES.MD is not accepted by the caller and then rejected here.
    suffix = path.suffix.lower()

    if suffix == ".txt" or suffix == ".md":
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        doc = Docx(path)
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Unsupported format: {path}")


def ingest_documents():
    """Ingest all documents from the documents directory."""
    documents = []

    base_dir = Path(DOCUMENTS_DIR)
    if not base_dir.exists():
        print(f"Warning: Documents directory {DOCUMENTS_DIR} does not exist")
        return documents

    # rglob already walks subdirectories, so anything that is not a supported
    # file is simply skipped.
    for path in base_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            print(f"Loading: {path}")
            try:
                documents.append({
                    "path": str(path),
                    "text": load_document(path)
                })
            except Exception as e:
                print(f"Error loading {path}: {e}")

    return documents


if __name__ == "__main__":
    docs = ingest_documents()
    print(f"Found {len(docs)} documents")
