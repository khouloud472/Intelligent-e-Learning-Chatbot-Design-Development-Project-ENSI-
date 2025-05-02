from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import time

def init_embeddings() -> HuggingFaceEmbeddings:
    """Initialise le modèle d'embedding"""
    print("\n⚙️ Initialisation des embeddings...")
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation du modèle: {str(e)}")
        exit()

def generate_embeddings(chunks: list) -> np.ndarray:
    """Vectorise les chunks"""
    try:
        model = init_embeddings()
        texts = [chunk.page_content for chunk in chunks]
        print(f"🔢 Génération des embeddings pour {len(texts)} textes...")
        return np.array(model.embed_documents(texts))
    except Exception as e:
        print(f"❌ Erreur lors de la génération des embeddings: {str(e)}")
        exit()

def calculate_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Calcule la matrice de similarité cosinus"""
    try:
        print("📐 Calcul de la matrice de similarité...")
        return cosine_similarity(embeddings)
    except Exception as e:
        print(f"❌ Erreur dans le calcul de similarité: {str(e)}")
        return np.array([])

def analyze_embeddings(embeddings: np.ndarray) -> dict:
    """Analyse complète des embeddings"""
    print("\n📊 Analyse des embeddings:")
    metrics = {
        'num_embeddings': len(embeddings),
        'embedding_dim': embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
        'similarity_matrix': None,
        'distance_matrix': None
    }
    
    try:
        # Calcul des métriques de base
        metrics['similarity_matrix'] = calculate_similarity_matrix(embeddings)
        if metrics['similarity_matrix'].size > 0:
            metrics.update({
                'mean_similarity': np.mean(metrics['similarity_matrix']),
                'median_similarity': np.median(metrics['similarity_matrix']),
                'max_similarity': np.max(metrics['similarity_matrix']),
                'min_similarity': np.min(metrics['similarity_matrix'])
            })
        
        # Calcul des distances
        metrics['distance_matrix'] = squareform(pdist(embeddings, metric='cosine'))
        if metrics['distance_matrix'].size > 0:
            metrics.update({
                'mean_distance': np.mean(metrics['distance_matrix']),
                'median_distance': np.median(metrics['distance_matrix'])
            })
        
        # Affichage des résultats
        print(f"- Nombre d'embeddings: {metrics['num_embeddings']}")
        print(f"- Dimension des embeddings: {metrics['embedding_dim']}")
        if 'mean_similarity' in metrics:
            print(f"- Similarité moyenne: {metrics['mean_similarity']:.4f}")
            print(f"- Similarité médiane: {metrics['median_similarity']:.4f}")
            print(f"- Similarité max: {metrics['max_similarity']:.4f}")
            print(f"- Similarité min: {metrics['min_similarity']:.4f}")
        if 'mean_distance' in metrics:
            print(f"- Distance moyenne: {metrics['mean_distance']:.4f}")
            print(f"- Distance médiane: {metrics['median_distance']:.4f}")
        
        return metrics
    
    except Exception as e:
        print(f"⚠️ Erreur dans l'analyse: {str(e)}")
        return metrics

def visualize_embeddings(embeddings: np.ndarray):
    """Lance toutes les visualisations des embeddings"""
    try:
        # Analyse numérique
        metrics = analyze_embeddings(embeddings)
        
        # Visualisations de base
        plt.figure(figsize=(15, 10))
        
        # 1. Distribution des valeurs d'embedding
        plt.subplot(2, 2, 1)
        sns.histplot(embeddings.flatten(), bins=50, kde=True, color="teal")
        plt.title("Distribution des valeurs d'embedding")
        plt.xlabel("Valeur")
        plt.ylabel("Fréquence")
        
        # 2. Visualisation PCA
        plt.subplot(2, 2, 2)
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5)
        plt.title("Projection PCA (2D)")
        plt.xlabel("Composante 1")
        plt.ylabel("Composante 2")
        
        # 3. Matrice de similarité (échantillon)
        if metrics['similarity_matrix'] is not None:
            plt.subplot(2, 2, 3)
            sample_size = min(50, len(metrics['similarity_matrix']))
            sns.heatmap(metrics['similarity_matrix'][:sample_size, :sample_size], 
                        cmap="YlOrRd", vmin=0, vmax=1)
            plt.title(f"Similarité cosinus (premiers {sample_size})")
        
        # 4. Distribution des similarités
        if metrics['similarity_matrix'] is not None:
            plt.subplot(2, 2, 4)
            sns.histplot(metrics['similarity_matrix'].flatten(), bins=50, color='purple')
            plt.title("Distribution des similarités")
            plt.xlabel("Similarité cosinus")
            plt.ylabel("Fréquence")
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ Erreur dans la visualisation: {str(e)}")