from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

loader = DirectoryLoader(
    "data/papers",
    glob="*.pdf",
    loader_cls=PyMuPDFLoader,
)


def load_documents():
    """Load documents from the configured directory. Call this when you need docs; not run on import."""
    return loader.load()


if __name__ == "__main__":
    documents = load_documents()
    print(len(documents))
