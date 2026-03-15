from langchain_text_splitters import RecursiveCharacterTextSplitter
from DocLoader import load_documents
import re


splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        
    chunk_overlap=150
)


def clean_research_text(text):
    """Remove noisy parts of research papers"""

    # Remove reference section
    text = re.split(r'\bReferences\b', text, flags=re.IGNORECASE)[0]

    # Remove citation numbers like [12]
    text = re.sub(r'\[\d+\]', '', text)

    # Remove figure/table mentions
    text = re.sub(r'Figure \d+', '', text)
    text = re.sub(r'Table \d+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def get_chunks(documents=None):
    """
    Return cleaned + chunked documents.
    Loads documents only when function is called.
    """

    if documents is None:
        documents = load_documents()

  
    for doc in documents:
        doc.page_content = clean_research_text(doc.page_content)

    
    chunks = splitter.split_documents(documents)

    
    chunks = [c for c in chunks if len(c.page_content) > 200]

    return chunks


if __name__ == "__main__":
    chunks = get_chunks()
    print("Total chunks:", len(chunks))
    print("\nExample chunk:\n")
    print(chunks[0].page_content[:500])