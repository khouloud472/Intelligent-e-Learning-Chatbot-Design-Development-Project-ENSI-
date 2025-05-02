import time
from langchain_community.vectorstores import FAISS
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
from typing import Tuple, List

from torch import cosine_similarity

def init_faiss_index(texts: list, embeddings_model, eval_data: List[Tuple[str, List[str]]] = None) -> Tuple[FAISS, dict]:
    """Initialise et évalue un index FAISS avec des métriques avancées."""
    print("\n⚙️ Initialisation de l'index FAISS...")
    
    try:
        index_name = "faiss_index_paragraph_v2"
        metrics = {}
        
        if os.path.exists(index_name):
            print("📂 Chargement de l'index existant...")
            db = FAISS.load_local(index_name, embeddings_model, allow_dangerous_deserialization=True)
        else:
            print("🛠 Création d'un nouvel index...")
            embeddings = embeddings_model.embed_documents([t.page_content for t in texts])
            embeddings = np.array(embeddings)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            db = FAISS.from_embeddings(
                text_embeddings=list(zip([t.page_content for t in texts], embeddings)),
                embedding=embeddings_model,
                metadatas=[t.metadata for t in texts]
            )
            db.save_local(index_name)
        
        print(f"📊 Vecteurs indexés: {db.index.ntotal}")
        
        # Calcul des métriques de base
        metrics['index_size'] = db.index.ntotal
        metrics['embedding_dim'] = db.index.d
        
        # Évaluation si des données sont fournies
        if eval_data:
            metrics.update(evaluate_faiss_performance(db, eval_data))
            visualize_evaluation_metrics(metrics)
            plot_index_statistics(db)
        
        return db, metrics  # Retourne explicitement le tuple (db, metrics)
    
    except Exception as e:
        print(f"❌ Erreur FAISS: {str(e)}")
        raise e  # Propage l'exception plutôt que de quitter
    

    
    """
    Initialise et évalue un index FAISS avec des métriques avancées.
    
    Args:
        texts: Liste de documents LangChain
        embeddings_model: Modèle d'embeddings
        eval_data: Données d'évaluation [(query, expected_results)]
    
    Returns:
        Tuple: (instance FAISS, métriques d'évaluation)
    """
    print("\n⚙️ Initialisation de l'index FAISS...")
    
    try:
        index_name = "faiss_index_paragraph_v2"
        metrics = {}
        
        if os.path.exists(index_name):
            print("📂 Chargement de l'index existant...")
            db = FAISS.load_local(index_name, embeddings_model, allow_dangerous_deserialization=True)
        else:
            print("🛠 Création d'un nouvel index...")
            embeddings = embeddings_model.embed_documents([t.page_content for t in texts])
            embeddings = np.array(embeddings)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            db = FAISS.from_embeddings(
                text_embeddings=list(zip([t.page_content for t in texts], embeddings)),
                embedding=embeddings_model,
                metadatas=[t.metadata for t in texts]
            )
            db.save_local(index_name)
        
        print(f"📊 Vecteurs indexés: {db.index.ntotal}")
        
        # Calcul des métriques de base
        metrics['index_size'] = db.index.ntotal
        metrics['embedding_dim'] = db.index.d
        
        # Évaluation si des données sont fournies
        if eval_data:
            metrics.update(evaluate_faiss_performance(db, eval_data))
            visualize_evaluation_metrics(metrics)
            plot_index_statistics(db)
        
        return db, metrics
    
    except Exception as e:
        print(f"❌ Erreur FAISS: {str(e)}")
        exit()

def evaluate_faiss_performance(db: FAISS, eval_data: List[Tuple[str, List[str]]]) -> dict:
    """
    Évalue la performance de l'index avec des métriques avancées.
    
    Args:
        db: Instance FAISS
        eval_data: Liste de tuples (requête, résultats attendus)
    
    Returns:
        dict: Métriques d'évaluation
    """
    print("\n🔍 Évaluation de la performance...")
    metrics = {
        'precision': [],
        'recall': [],
        'avg_precision': [],
        'query_times': []
    }
    
    for query, expected in eval_data:
        start_time = time.time()
        results = db.similarity_search(query, k=5)
        metrics['query_times'].append(time.time() - start_time)
        
        retrieved = [r.page_content for r in results]
        relevant_retrieved = len(set(retrieved) & set(expected))
        
        precision = relevant_retrieved / len(retrieved) if retrieved else 0
        recall = relevant_retrieved / len(expected) if expected else 0
        avg_precision = average_precision_score(
            [1 if doc in expected else 0 for doc in retrieved],
            range(len(retrieved), 0, -1)
        )
        
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['avg_precision'].append(avg_precision)
    
    # Calcul des moyennes
    metrics['mean_precision'] = np.mean(metrics['precision'])
    metrics['mean_recall'] = np.mean(metrics['recall'])
    metrics['mean_avg_precision'] = np.mean(metrics['avg_precision'])
    metrics['mean_query_time'] = np.mean(metrics['query_times'])
    
    print(f"📈 Performance moyenne - Précision: {metrics['mean_precision']:.2f}, Rappel: {metrics['mean_recall']:.2f}")
    print(f"⏱ Temps moyen par requête: {metrics['mean_query_time']:.4f}s")
    
    return metrics

def visualize_evaluation_metrics(metrics: dict):
    """Visualise les métriques d'évaluation"""
    plt.figure(figsize=(15, 10))
    
    # Courbe Precision-Recall
    plt.subplot(2, 2, 1)
    precision, recall, _ = precision_recall_curve(
        [1]*len(metrics['precision']), metrics['precision'])
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Courbe Precision-Recall')
    plt.grid(True)
    
    # Distribution des temps de requête
    plt.subplot(2, 2, 2)
    sns.histplot(metrics['query_times'], bins=20, kde=True)
    plt.xlabel('Temps (s)')
    plt.ylabel('Fréquence')
    plt.title('Distribution des temps de requête')
    plt.grid(True)
    
    # Comparaison des métriques
    plt.subplot(2, 2, 3)
    sns.barplot(x=['Précision', 'Rappel', 'mAP'], 
                y=[metrics['mean_precision'], metrics['mean_recall'], metrics['mean_avg_precision']])
    plt.ylim(0, 1)
    plt.title('Comparaison des métriques moyennes')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_index_statistics(db: FAISS):
    """Visualise les statistiques de l'index"""
    # Analyse de la distribution des distances
    random_vectors = np.random.randn(1000, db.index.d).astype('float32')
    random_vectors = random_vectors / np.linalg.norm(random_vectors, axis=1, keepdims=True)
    D, _ = db.index.search(random_vectors, 5)  # Distance aux 5 plus proches voisins
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(D.flatten(), bins=50, kde=True)
    plt.xlabel('Distance cosinus')
    plt.ylabel('Fréquence')
    plt.title('Distribution des distances aux voisins')
    
    plt.subplot(1, 2, 2)
    plt.plot(np.sort(D.flatten()), np.linspace(0, 1, len(D.flatten())), 'b-')
    plt.xlabel('Distance')
    plt.ylabel('CDF')
    plt.title('Fonction de répartition des distances')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_index_quality(db: FAISS, sample_size=1000):
    """Analyse approfondie de la qualité de l'index"""
    print("\n🔎 Analyse de qualité de l'index...")
    
    # Échantillonnage de vecteurs
    if db.index.ntotal > sample_size:
        np.random.seed(42)
        sample_indices = np.random.choice(db.index.ntotal, sample_size, replace=False)
        sample_vectors = db.index.reconstruct_batch(sample_indices)
    else:
        sample_vectors = db.index.reconstruct_n(0, db.index.ntotal)
    
    # Calcul des similarités intra-index
    similarities = cosine_similarity(sample_vectors)
    np.fill_diagonal(similarities, 0)  # Exclure l'auto-similarité
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(similarities.flatten(), bins=50, kde=True)
    plt.title('Distribution des similarités intra-index')
    plt.xlabel('Similarité cosinus')
    plt.ylabel('Fréquence')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(similarities[:50, :50], cmap="YlOrRd", vmin=0, vmax=1)
    plt.title('Matrice de similarité (50 premiers)')
    
    plt.tight_layout()
    plt.show()