from data_loading import load_documents, visualize_data
from text_processing import split_texts, visualize_chunks
from embeddings import generate_embeddings, visualize_embeddings, init_embeddings
from faiss_index import init_faiss_index
from evaluation import load_test_data, analyze_test_results, visualize_analysis
from interactive import interactive_qa_loop
from training import train_enhanced_model

import time

def main():
    print("🚀 Démarrage du pipeline de traitement...\n")

    try:
        # Étape 1: Chargement des documents
        print("="*50)
        print("ÉTAPE 1: CHARGEMENT DES DOCUMENTS")
        print("="*50)
        start_time = time.time()
        documents = load_documents("pdfs")
        visualize_data(documents)
        print(f"⏱ Temps écoulé: {time.time() - start_time:.2f}s")

        # Étape 2: Découpage des textes
        print("\n" + "="*50)
        print("ÉTAPE 2: DÉCOUPAGE DES TEXTES")
        print("="*50)
        start_time = time.time()
        chunks = split_texts(documents)
        visualize_chunks(chunks, documents)
        print(f"⏱ Temps écoulé: {time.time() - start_time:.2f}s")

        # Étape 3: Initialisation du modèle d'embeddings et génération
        print("\n" + "="*50)
        print("ÉTAPE 3: VECTORISATION")
        print("="*50)
        start_time = time.time()
        embeddings_model = init_embeddings()
        embeddings = generate_embeddings(chunks)
        visualize_embeddings(embeddings)
        print(f"⏱ Temps écoulé: {time.time() - start_time:.2f}s")

        # Étape 4: Indexation FAISS
        print("\n" + "="*50)
        print("ÉTAPE 4: INDEXATION FAISS")
        print("="*50)
        start_time = time.time()
        db, metrics = init_faiss_index(chunks, embeddings_model)  # Déballage correct du tuple

        # Récupération d’informations sur l’index
        index_info = "Non disponible"
        try:
            if hasattr(db, '_index'):
                index_info = str(db._index.ntotal)
            elif hasattr(db, 'index'):
                if callable(db.index):
                    index = db.index()
                    index_info = str(index.ntotal)
                else:
                    index_info = str(db.index.ntotal)
        except Exception as e:
            print(f"⚠️ Impossible de récupérer les infos de l'index: {str(e)}")

        # Étape 5: Entraînement du modèle supervisé
        print("\n" + "="*50)
        print("ÉTAPE 5: ENTRAÎNEMENT DU MODÈLE")
        print("="*50)
        start_time = time.time()
        doc_embeddings = embeddings_model.embed_documents([chunk.page_content for chunk in chunks])
        model = train_enhanced_model(doc_embeddings)
        print(f"⏱ Temps écoulé: {time.time() - start_time:.2f}s")

        # Étape 6: Évaluation
        print("\n" + "="*50)
        print("ÉTAPE 6: ÉVALUATION")
        print("="*50)
        try:
            test_data = load_test_data()
            print(f"📄 {len(test_data['details'])} questions chargées pour l'évaluation.")
            analysis = analyze_test_results(test_data)
            visualize_analysis(analysis)
        except Exception as e:
            print(f"⚠️ Erreur d'évaluation: {str(e)}")

        # Statistiques finales
        print("\n📊 Statistiques générales :")
        print(f"- Documents chargés : {len(documents)}")
        print(f"- Chunks créés      : {len(chunks)}")
        print(f"- Embeddings dim.   : {embeddings.shape[1]}")
        print(f"- Vecteurs indexés  : {index_info}")
        print(f"⏱ Pipeline total terminé en {time.time() - start_time:.2f}s")

          # Étape 7: Boucle interactive utilisateur
        print("\n" + "="*50)
        print("ÉTAPE 7: INTERACTION UTILISATEUR")
        print("="*50)
        interactive_qa_loop(model, embeddings_model, db)  # db est maintenant correctement passé

        print("\n✅ Pipeline complet exécuté avec succès !")

    except Exception as e:
        print(f"\n❌ Erreur dans le pipeline : {str(e)}")
        raise e

if __name__ == "__main__":
    main()
