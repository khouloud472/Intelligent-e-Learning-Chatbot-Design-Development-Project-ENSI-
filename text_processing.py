from langchain.text_splitter import RecursiveCharacterTextSplitter
import matplotlib.pyplot as plt
import seaborn as sns

def split_texts(docs: list, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    """Découpe les documents en chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "(?<=\. )", " "],
        add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    valid_chunks = [chunk for chunk in chunks if len(chunk.page_content.split()) > 10]
    print(f"✅ {len(valid_chunks)} fragments valides créés")
    return valid_chunks

def visualize_chunk_lengths(texts: list):
    """Visualise la longueur des chunks"""
    chunk_lengths = [len(text.page_content.split()) for text in texts]
    plt.figure(figsize=(10, 6))
    plt.hist(chunk_lengths, bins=30, color='teal', edgecolor='black')
    plt.title("Distribution de la longueur des chunks")
    plt.xlabel("Nombre de mots par chunk")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.show()

def visualize_chunks_per_document(docs: list, texts: list):
    """Visualise la répartition par document"""
    doc_chunk_counts = {}
    for text in texts:
        source = text.metadata.get('source', 'Inconnu')
        doc_chunk_counts[source] = doc_chunk_counts.get(source, 0) + 1
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x=list(doc_chunk_counts.values()), color='orange')
    plt.title("Nombre de chunks par document")
    plt.xlabel("Nombre de chunks")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.show()

def visualize_chunks(chunks: list, original_docs: list):
    """Lance toutes les visualisations des chunks"""
    visualize_chunk_lengths(chunks)
    visualize_chunks_per_document(original_docs, chunks)