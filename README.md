# 🏥 RAG Oncologie

**Équipe :** Douae Hajji | Ayoub Bourti | Hajar Tarafouss | Aya Bourhdifa | Saloua Tirichi

## 📋 Description
Système RAG (Retrieval-Augmented Generation) pour l'oncologie avec retriever hybride (BM25 + embeddings sémantiques).

## ✨ Fonctionnalités
- 🔍 Recherche hybride (BM25 + multilingual-e5-small)
- 🎯 Détection automatique du cancer et du type de question
- 📚 13 types de documents médicaux
- 🇫🇷 Optimisé pour le français médical
- ⚡ Hit@3 : 85-90%

## 🚀 Installation
```bash
git clone https://github.com/Ayoubbourti/RAG-Oncologie.git
cd RAG-Oncologie
pip install -r requirements.txt
📖 Utilisation
python
from rag_oncologie import HybridRetriever

retriever = HybridRetriever(chunks, embeddings, encode_query)
results = retriever.search_auto("Traitement du cancer de la prostate ?")
🎯 Types supportés
Type	Exemple
Traitement	Protocoles chimiothérapie
Effets secondaires	Gestion des toxicités
Diagnostic	Guidelines diagnostiques
Métastases	Sites métastatiques
Urgences	Complications aiguës
Résistance	Mécanismes de résistance
Suivi	Post-traitement
📊 Performance
Hit@3 : 85-90%

Temps réponse : <1 seconde

📄 Licence
Projet académique

*2026*
