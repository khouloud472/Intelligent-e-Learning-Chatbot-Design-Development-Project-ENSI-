# 🔧 Importation des bibliothèques
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np

# 📥 Chargement des données de test depuis un fichier JSON
def load_test_data(file_path: str = "test.json") -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"⚠️ Fichier non trouvé : {file_path}")
    
    with path.open('r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if "details" not in data:
                # Crée une structure vide si 'details' n'existe pas
                data["details"] = []
                # Alternative: lever une exception
                # raise ValueError("❌ Le fichier ne contient pas de clé 'details'")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Erreur de décodage JSON : {e}")
        
        
# 🔍 Analyse des résultats
def analyze_test_results(results: dict) -> dict:
    analysis = {
        'by_source': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'similarity_distribution': []
    }

    for res in results['details']:
        source = res.get('source', 'Inconnu')
        analysis['by_source'][source]['total'] += 1
        if res.get('correct', False):
            analysis['by_source'][source]['correct'] += 1
        analysis['similarity_distribution'].append(res.get('similarity', 0.0))

    return analysis

# 📊 Visualisation des résultats
def visualize_analysis(analysis: dict):
    plt.figure(figsize=(12, 6))

    # 🔹 Accuracy par source
    sources = list(analysis['by_source'].keys())
    total = [v['total'] for v in analysis['by_source'].values()]
    correct = [v['correct'] for v in analysis['by_source'].values()]
    accuracies = [c / t if t > 0 else 0.0 for c, t in zip(correct, total)]

    sns.barplot(x=sources, y=accuracies, palette="crest")
    plt.title("🎯 Précision (Accuracy) par Source", fontsize=14)
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("accuracy_by_source.png")
    plt.close()

    # 🔹 Distribution des similarités
    plt.figure(figsize=(10, 5))
    sns.histplot(analysis['similarity_distribution'], bins=30, kde=True, color='skyblue')
    plt.title("📈 Distribution des Similarités", fontsize=14)
    plt.xlabel("Score de similarité")
    plt.ylabel("Nombre de paires")
    plt.tight_layout()
    plt.savefig("similarity_distribution.png")
    plt.close()

    # 🔸 Affichage global
    avg_accuracy = sum(correct) / sum(total) if sum(total) > 0 else 0.0
    print(f"\n📌 Précision moyenne globale : {avg_accuracy:.2%}")
    for src, v in analysis['by_source'].items():
        acc = v['correct'] / v['total'] if v['total'] > 0 else 0.0
        print(f" - {src}: {acc:.2%} ({v['correct']} / {v['total']})")

