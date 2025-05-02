import torch
import numpy as np

def predict_with_confidence(model, question_embedding, doc_embeddings):
    """
    Calcule les similarités entre une question et une liste d'embeddings de documents
    en utilisant un modèle de similarité, puis convertit ces scores en probabilités.

    Args:
        model: Modèle PyTorch entraîné pour prédire la similarité
        question_embedding: Vecteur embedding de la question (1D)
        doc_embeddings: Liste de vecteurs embeddings des documents candidats (2D)

    Returns:
        similarities (np.ndarray): Score de similarité [0, 1] pour chaque document
        confidences (np.ndarray): Distribution des probabilités (softmax) pour refléter la confiance
    """

    # Mode évaluation (pas de calcul de gradient)
    with torch.no_grad():
        # Concaténation question + chaque document (similaire à l'entraînement)
        inputs = [np.concatenate([question_embedding, de]) for de in doc_embeddings]

        # Conversion en tenseur PyTorch de type float
        inputs = torch.FloatTensor(np.array(inputs))  # ⚠️ Supprimé une parenthèse en trop ici

        # Prédiction des similarités via le modèle
        similarities = model(inputs).squeeze().numpy()

        # Transformation softmax pour obtenir des probabilités de confiance
        confidences = np.exp(similarities) / np.sum(np.exp(similarities))

        return similarities, confidences
