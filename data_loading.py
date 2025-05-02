import matplotlib.pyplot as plt
from collections import Counter
from langchain_community.document_loaders import PyPDFDirectoryLoader

def load_documents(pdf_dir: str = "pdfs") -> list:
    """Charge les documents PDF depuis un répertoire"""
    print(f"\n📂 Chargement depuis : '{pdf_dir}'")
    try:
        loader = PyPDFDirectoryLoader(pdf_dir)
        documents = loader.load()
        valid_docs = [doc for doc in documents if doc.page_content.strip()]
        print(f"✅ {len(valid_docs)} documents valides")
        return valid_docs
    except Exception as e:
        print(f"❌ Erreur : {str(e)}")
        exit()

def visualize_pages(documents: list):
    """Visualise la distribution du nombre de pages"""
    page_counts = [len(doc.page_content.split('\n')) for doc in documents]
    plt.figure(figsize=(10, 6))
    plt.hist(page_counts, bins=20, color='blue', edgecolor='black')
    plt.title("Distribution du nombre de pages par document")
    plt.xlabel("Nombre de pages")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.show()

def visualize_document_lengths(documents: list):
    """Visualise la longueur des documents en mots"""
    word_counts = [len(doc.page_content.split()) for doc in documents]
    plt.figure(figsize=(10, 6))
    plt.hist(word_counts, bins=20, color='green', edgecolor='black')
    plt.title("Distribution du nombre de mots par document")
    plt.xlabel("Nombre de mots")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.show()

def visualize_sources(documents: list):
    """Visualise la répartition par source"""
    sources = [doc.metadata.get('source', 'Inconnu') for doc in documents]
    source_counts = Counter(sources)
    plt.figure(figsize=(10, 6))
    plt.pie(source_counts.values(), labels=source_counts.keys(), autopct='%1.1f%%', startangle=90)
    plt.title("Répartition des documents par source")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def visualize_data(documents: list):
    """Lance toutes les visualisations"""
    visualize_pages(documents)
    visualize_document_lengths(documents)
    visualize_sources(documents)