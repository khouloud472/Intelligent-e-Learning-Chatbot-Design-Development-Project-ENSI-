import textwrap  # Pour formater l'affichage du texte de manière lisible
import numpy as np
def interactive_qa_loop(model, embeddings_model, db):
    """
    Boucle interactive de question/réponse utilisant une base de données vectorielle.
    
    Args:
        model: Modèle de similarité entraîné
        embeddings_model: Modèle d'embedding
        db: Instance FAISS (doit être l'objet DB directement)
    """
    print("\n🔍 Mode interactif (tapez 'quit' pour quitter)")

    while True:
        question = input("\n❓ Question: ").strip()

        if question.lower() in ('quit', 'exit', 'q'):
            break
            
        try:
            # Vérification que db est bien l'objet FAISS
            if not hasattr(db, 'similarity_search'):
                raise AttributeError("L'objet DB ne possède pas la méthode similarity_search")
                
            # Recherche des documents similaires
            docs = db.similarity_search(question, k=3)
            
            # Affichage des résultats
            print("\n📌 Meilleure réponse:")
            print(textwrap.fill(docs[0].page_content[:400], width=80))
            
            print("\n🔍 Autres résultats:")
            for i, doc in enumerate(docs[1:], 1):
                print(f"\n{i}. {doc.page_content[:200]}...")
                
        except Exception as e:
            print(f"⚠️ Erreur: {str(e)}")
            print("Type de l'objet db:", type(db))
            if isinstance(db, tuple):
                print("db est un tuple. Essayez db[0] pour accéder à l'instance FAISS.")