"""
Interface Gradio pour RAG Oncologie
====================================
Lancez avec : python rag_oncologie_ui.py
Puis ouvrez  http://localhost:7860
"""

import json, pickle, time, uuid, warnings
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

import numpy as np
import gradio as gr

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH        = "merged_cancers_vfrancais.json"
EMBEDDINGS_PATH  = "embeddings.pkl"
CUSTOM_DATA_PATH = "custom_entries.json"
MODEL_NAME       = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES RAG
# ─────────────────────────────────────────────────────────────────────────────
PREFIX_TO_CANCER = {
    "SCLC":"Cancer du poumon à petites cellules",
    "CPNPC":"Cancer Pulmonaire Non à Petites Cellules (CPNPC)",
    "BREAST":"Cancer du sein",
    "CRC":"Cancer Colorectal (CCR)",
    "PROSTATE":"Cancer de la prostate",
    "THYROID":"Cancer de la thyroïde",
    "OVARIAN":"Cancer épithélial de l'ovaire",
    "MEL":"Mélanome",
    "HGG":"Gliomes de haut grade",
    "STOMACH":"Cancer gastrique",
    "KIDNEY":"Cancer du rein",
    "BLADDER":"Cancer de la vessie",
    "CERVICAL":"Cancer du col de l'utérus",
}

NAME_NORMALIZATION = {
    "Thyroid Cancer":"Cancer de la thyroïde",
    "Prostate Cancer":"Cancer de la prostate",
    "Kidney Cancer (Renal Cell Carcinoma)":"Cancer du rein",
    "Gastric Cancer (Gastric Adenocarcinoma)":"Cancer gastrique",
    "Gastric Cancer":"Cancer gastrique",
    "Breast Cancer":"Cancer du sein",
    "Ovarian Cancer":"Cancer épithélial de l'ovaire",
    "Colorectal Cancer":"Cancer Colorectal (CCR)",
    "Melanoma":"Mélanome",
    "Glioma":"Gliomes de haut grade",
    "Small Cell Lung Cancer":"Cancer du poumon à petites cellules",
}

CANCER_KEYWORDS = {
    "Cancer de la prostate":["prostate"],
    "Cancer du sein":["sein","breast","her2","brca"],
    "Cancer Colorectal (CCR)":["colorectal","colon","rectum","ccr"],
    "Cancer du poumon à petites cellules":["poumon petites cellules","sclc","petites cellules"],
    "Cancer Pulmonaire Non à Petites Cellules (CPNPC)":["cpnpc","nsclc","poumon non"],
    "Cancer de la thyroïde":["thyroïde","thyroid"],
    "Cancer épithélial de l'ovaire":["ovaire","ovarian"],
    "Mélanome":["mélanome","melanome","melanoma"],
    "Gliomes de haut grade":["gliome","glioblastome","gbm","hgg"],
    "Cancer gastrique":["gastrique","estomac","gastric"],
    "Cancer du rein":["rein","rénal","carcinome rénal"],
    "Cancer de la vessie":["vessie","bladder","urothélial"],
    "Cancer du col de l'utérus":["col","utérus","cervical","cervix"],
}

QTYPE_KEYWORDS = {
    "traitement":["traitement","protocole","thérapie","chimiothérapie","immunothérapie","chirurgie","ligne","prise en charge"],
    "diagnostic":["diagnostic","diagnostiquer","détection","bilan","examens","dépistage","imagerie","biopsie"],
    "effets_secondaires":["effets secondaires","toxicité","tolérance","effets indésirables","nausées","neutropénie"],
    "suivi":["suivi","surveillance","contrôle","récidive","rémission"],
    "pronostic":["pronostic","survie","mortalité","facteur pronostique"],
    "facteurs_risque":["facteur de risque","risque","prédisposition","génétique"],
    "resistance":["résistance","réfractaire","échec thérapeutique","rechute"],
    "urgence":["urgence","urgences","compression","hypercalcémie","neutropénie fébrile"],
}

QTYPE_TO_DOCTYPE = {
    "traitement":"treatment_protocol",
    "diagnostic":"diagnostic_guideline",
    "effets_secondaires":"toxicity_management",
    "suivi":"followup",
    "pronostic":"cancer_knowledge",
    "facteurs_risque":"cancer_knowledge",
    "resistance":"resistance_mechanism",
    "urgence":"oncology_emergency",
}

QTYPE_PREFERRED_SUBTYPES = {
    "traitement":["traitement_protocole"],
    "diagnostic":["guideline_diagnostic"],
    "effets_secondaires":["toxicité"],
    "suivi":["suivi"],
    "pronostic":["connaissance_prognosis"],
    "facteurs_risque":["connaissance_risk_factors"],
    "resistance":["résistance"],
    "urgence":["urgence"],
}

PROMPT_STRUCTURES = {
    "traitement":"Structure: 1) Lignes de traitement, 2) Médicaments et doses, 3) Critères de choix.",
    "diagnostic":"Structure: 1) Examens initiaux, 2) Confirmation diagnostique, 3) Staging.",
    "pronostic":"Structure: 1) Facteurs pronostiques, 2) Survie par stade, 3) Récidive.",
    "effets_secondaires":"Structure: 1) Effets fréquents, 2) Effets graves (grade 3-4), 3) Prise en charge.",
    "suivi":"Structure: 1) Calendrier de suivi, 2) Examens recommandés, 3) Signes d'alarme.",
    "facteurs_risque":"Structure: 1) Facteurs non modifiables, 2) Facteurs modifiables, 3) Dépistage.",
    "default":"Réponds de manière structurée et cliniquement pertinente.",
}

DOC_TYPE_LABELS = {
    "treatment_protocol":"🧬 Protocole",
    "cancer_knowledge":"📚 Connaissance",
    "metastasis":"🔴 Métastase",
    "palliative_care":"💙 Palliatif",
    "toxicity_management":"⚠️ Toxicité",
    "followup":"📅 Suivi",
    "resistance_mechanism":"🔬 Résistance",
    "staging_system":"📊 Stadification",
    "oncology_score":"🔢 Score",
    "oncology_emergency":"🚨 Urgence",
    "contraindication_interaction":"💊 Contre-indication",
    "diagnostic_guideline":"🔍 Guideline diag.",
    "medical_literature":"📄 Littérature",
    "clinical_reasoning":"🩺 Cas clinique",
    "biomarker_genetics":"🧪 Biomarqueur",
    "custom":"✨ Personnalisé",
}

ALL_DOC_TYPES = sorted(DOC_TYPE_LABELS.keys())
ALL_CANCERS   = sorted(PREFIX_TO_CANCER.values())

# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────
def detect_cancer(query):
    q = query.lower()
    for cancer, kws in CANCER_KEYWORDS.items():
        for kw in kws:
            if kw in q:
                return cancer
    return None

def detect_qtype(query):
    q = query.lower()
    for qtype, kws in QTYPE_KEYWORDS.items():
        for kw in kws:
            if kw in q:
                return qtype
    return None

def s(val, maxlen=400):
    if not val:
        return ""
    if isinstance(val, list):
        return "; ".join(str(v) for v in val if v)[:maxlen]
    if isinstance(val, dict):
        return " | ".join(f"{k}: {v}" for k,v in val.items() if v)[:maxlen]
    return str(val)[:maxlen]

def chunk_document(item: Dict) -> List[Dict]:
    chunks = []
    cancer   = item.get("cancer_name","unknown")
    doc_type = item.get("document_type","unknown")
    doc_id   = item.get("document_id","unknown")

    def add(text, subtype):
        text = text.strip()
        if len(text) > 60:
            chunks.append({"text":f"[{cancer}] [{subtype}] {text[:700]}",
                           "metadata":{"cancer":cancer,"type":doc_type,"subtype":subtype,"doc_id":doc_id}})

    if doc_type == "treatment_protocol":
        for p in item.get("protocols",[]):
            drugs = s([f"{d.get('drug_name','?')} {d.get('dose','')}" for d in p.get("drugs",[])])
            add(f"Protocole: {p.get('protocol_name','N/A')}. Stade: {p.get('stage','')}. "
                f"Ligne: {p.get('line_of_therapy','')}. Médicaments: {drugs}. "
                f"Résultats: {s(p.get('expected_outcomes',[]))}", "traitement_protocole")
    elif doc_type == "cancer_knowledge":
        for key,label in [("definition","Définition"),("epidemiology","Épidémiologie"),
                          ("risk_factors","Facteurs de risque"),("common_symptoms","Symptômes fréquents"),
                          ("screening_methods","Dépistage"),("diagnostic_tests","Tests diagnostiques"),
                          ("prognosis","Pronostic"),("pathophysiology","Physiopathologie")]:
            val = item.get(key)
            if val: add(f"{label}: {s(val)}", f"connaissance_{key}")
    elif doc_type == "metastasis":
        for p in item.get("metastatic_profiles",[]):
            add(f"Métastase de {p.get('primary_cancer','?')} vers {p.get('metastatic_site','?')}. "
                f"Traitement: {s(p.get('treatment_options',[]))}", "métastase")
    elif doc_type == "toxicity_management":
        for p in item.get("toxicity_profiles",[]):
            add(f"Toxicité: {p.get('toxicity_name','?')}. "
                f"Médicaments: {s(p.get('causing_drugs',[]))}. "
                f"Gestion: {s(p.get('management_by_grade',{}))}", "toxicité")
    elif doc_type == "followup":
        for p in item.get("followup_protocols",[]):
            add(f"Suivi — Cancer: {p.get('cancer_type','?')}. "
                f"Tests: {s(p.get('recommended_tests',[]))}. "
                f"Signes alarme: {s(p.get('warning_signs',[]))}", "suivi")
    elif doc_type == "resistance_mechanism":
        for p in item.get("resistance_profiles",[]):
            add(f"Résistance: {p.get('drug_name','?')}. Mécanisme: {p.get('mechanism','')}. "
                f"Alternatifs: {s(p.get('alternative_treatments',[]))}", "résistance")
    elif doc_type == "oncology_emergency":
        for p in item.get("oncology_emergencies",[]):
            add(f"Urgence: {p.get('emergency_name','?')}. {p.get('definition','')}. "
                f"Actions: {s(p.get('immediate_actions',[]))}", "urgence")
    elif doc_type == "diagnostic_guideline":
        for p in item.get("diagnostic_guidelines",[]):
            add(f"Guideline: {p.get('cancer_type','?')}. Tests: {s(p.get('recommended_tests',[]))}. "
                f"Algorithme: {s(p.get('diagnostic_algorithm',[]))}", "guideline_diagnostic")
    elif doc_type == "medical_literature":
        for p in item.get("literature_evidence",[]):
            add(f"Étude: {p.get('title','?')} ({p.get('journal','')} {p.get('year','')}). "
                f"Résultats: {s(p.get('main_findings',[]))}", "littérature")
    elif doc_type == "contraindication_interaction":
        for p in item.get("drug_safety_profiles",[]):
            add(f"Sécurité: {p.get('drug_name','?')}. "
                f"Contre-indications: {s(p.get('contraindications',[]))}. "
                f"Interactions: {s(p.get('drug_interactions',[]))}", "contre_indication")
    
    for field in ("definition","content","text","description"):
        val = item.get(field)
        if val and not chunks:
            add(str(val), "general")
            break
    return chunks

# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVER
# ─────────────────────────────────────────────────────────────────────────────
class HybridRetriever:
    DOC_TYPE_BOOST = {
        "treatment_protocol":1.4,"cancer_knowledge":1.2,"toxicity_management":1.3,
        "followup":1.2,"resistance_mechanism":1.3,"oncology_emergency":1.5,
        "diagnostic_guideline":1.2,"medical_literature":0.9,
    }
    
    def __init__(self, chunks, embeddings, encode_fn, bm25_w=0.35, sem_w=0.65):
        from rank_bm25 import BM25Okapi
        self.chunks=chunks
        self.embeddings=embeddings
        self.encode_fn=encode_fn
        self.bm25_w=bm25_w
        self.sem_w=sem_w
        self.bm25 = BM25Okapi([c["text"].lower().split() for c in chunks])

    @staticmethod
    def _mm(arr):
        mn,mx=arr.min(),arr.max()
        return (arr-mn)/(mx-mn+1e-9)

    def search(self, query, k=5, cancer_filter=None, strict=True, qtype=None):
        bm25_raw  = np.array(self.bm25.get_scores(query.lower().split()))
        q_emb     = self.encode_fn(query)
        sem_raw   = np.dot(self.embeddings, q_emb)
        bm25_norm = self._mm(bm25_raw)
        sem_norm=self._mm(sem_raw)
        hybrid    = self.bm25_w*bm25_norm + self.sem_w*sem_norm + 0.15*bm25_norm*sem_norm
        preferred    = QTYPE_PREFERRED_SUBTYPES.get(qtype,[]) if qtype else []
        target_dtype = QTYPE_TO_DOCTYPE.get(qtype)
        results, seen = [], set()
        
        if target_dtype and cancer_filter:
            for h in self._collect(hybrid,bm25_norm,sem_norm,preferred,cancer_filter,True,force_dtype=target_dtype,k=2):
                results.append(h)
                seen.add(h["_idx"])
        
        results.extend(self._collect(hybrid,bm25_norm,sem_norm,preferred,cancer_filter,strict,exclude_idx=seen,k=k))
        
        if target_dtype and not any(r["type"]==target_dtype for r in results):
            fb=self._collect(hybrid,bm25_norm,sem_norm,preferred,None,False,force_dtype=target_dtype,k=2,
                             exclude_idx={r["_idx"] for r in results})
            results=fb+results
        
        results=sorted(results,key=lambda x:x["score"],reverse=True)
        for r in results: r.pop("_idx",None)
        return results[:k]

    def _collect(self, hybrid,bm25_norm,sem_norm,preferred,cancer_filter,strict,
                 force_dtype=None,exclude_idx=None,k=10):
        exclude_idx=exclude_idx or set()
        results=[]
        for idx in np.argsort(hybrid)[::-1]:
            if idx in exclude_idx or len(results)>=k: continue
            c=self.chunks[idx]
            cancer=c["metadata"].get("cancer","unknown")
            dtype=c["metadata"].get("type","unknown")
            subtype=c["metadata"].get("subtype","")
            if force_dtype and dtype!=force_dtype: continue
            if cancer_filter:
                if strict and cancer_filter.lower()!=cancer.lower(): continue
                elif not strict and cancer_filter.lower() not in cancer.lower(): continue
            boost=1.6 if subtype in preferred else 1.0
            score=hybrid[idx]*self.DOC_TYPE_BOOST.get(dtype,1.0)*boost
            results.append({"text":c["text"],"score":float(score),"cancer":cancer,
                            "type":dtype,"subtype":subtype,
                            "bm25":float(bm25_norm[idx]),"sem":float(sem_norm[idx]),"_idx":int(idx)})
        return results

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAT GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
retriever   = None
embed_model = None
init_status = "⏳ Non initialisé — cliquez sur Initialiser le système"

def _load_custom(): 
    return json.load(open(CUSTOM_DATA_PATH,encoding="utf-8")) if Path(CUSTOM_DATA_PATH).exists() else []

def _save_custom(entries): 
    json.dump(entries, open(CUSTOM_DATA_PATH,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
def load_system(progress=gr.Progress()):
    global retriever, init_status, embed_model
    
    if not Path(DATA_PATH).exists():
        init_status=f"❌ Fichier introuvable : {DATA_PATH}"
        return init_status
    
    progress(0.05,desc="Chargement des données…")
    data=json.load(open(DATA_PATH,encoding="utf-8"))
    data.extend(_load_custom())
    
    progress(0.15,desc="Normalisation…")
    for doc in data:
        name=doc.get("cancer_name","") or ""
        if name in NAME_NORMALIZATION: 
            doc["cancer_name"]=NAME_NORMALIZATION[name]
        elif not name or name=="unknown":
            pfx=doc.get("document_id","").split("_")[0]
            c=PREFIX_TO_CANCER.get(pfx)
            if c: doc["cancer_name"]=c
    
    progress(0.30,desc="Chunking…")
    all_chunks=[ch for doc in data for ch in chunk_document(doc)]
    
    progress(0.45,desc="Modèle d'embeddings…")
    try:
        from sentence_transformers import SentenceTransformer
        embed_model=SentenceTransformer(MODEL_NAME)
    except Exception as e:
        init_status=f"❌ Modèle : {e}"
        return init_status
    
    embeddings=None
    if Path(EMBEDDINGS_PATH).exists():
        progress(0.60,desc="Cache embeddings…")
        embeddings=pickle.load(open(EMBEDDINGS_PATH,"rb"))
        if len(embeddings)!=len(all_chunks): embeddings=None
    
    if embeddings is None:
        progress(0.65,desc="Calcul embeddings (patience)…")
        embeddings=embed_model.encode([c["text"] for c in all_chunks],batch_size=64,
                                      show_progress_bar=False,normalize_embeddings=True)
        pickle.dump(embeddings,open(EMBEDDINGS_PATH,"wb"))
    
    progress(0.90,desc="Retriever…")
    retriever=HybridRetriever(all_chunks,embeddings,lambda q:embed_model.encode([q],normalize_embeddings=True)[0])
    counts=Counter(c["metadata"]["cancer"] for c in all_chunks)
    custom_n=len(_load_custom())
    init_status=(f"✅ Prêt — {len(all_chunks):,} chunks | {len(data)} docs "
                 f"({custom_n} perso) | {len(counts)} cancers")
    progress(1.0,desc="Terminé !")
    return init_status

# ─────────────────────────────────────────────────────────────────────────────
# ENRICHISSEMENT
# ─────────────────────────────────────────────────────────────────────────────
def _rebuild(new_chunks):
    global retriever, embed_model
    if retriever is None: 
        return "❌ Initialisez d'abord le système."
    if not new_chunks:    
        return "⚠️ Aucun chunk valide."
    
    new_embs  = embed_model.encode([c["text"] for c in new_chunks],batch_size=32,normalize_embeddings=True)
    all_chunks = retriever.chunks + new_chunks
    all_embs   = np.vstack([retriever.embeddings, new_embs])
    pickle.dump(all_embs, open(EMBEDDINGS_PATH,"wb"))
    retriever  = HybridRetriever(all_chunks, all_embs,
                                 lambda q: embed_model.encode([q],normalize_embeddings=True)[0])
    return f"✅ {len(new_chunks)} chunks ajoutés — total : {len(all_chunks):,}"

def get_base_stats():
    if retriever is None:
        return "<p style='color:#64748b;padding:12px;'>Système non initialisé.</p>"
    
    counts = Counter(c["metadata"]["cancer"] for c in retriever.chunks)
    types  = Counter(c["metadata"]["type"]   for c in retriever.chunks)
    custom = sum(1 for c in retriever.chunks if c["metadata"].get("subtype") in ("manuel","texte_libre","general"))
    
    rows_c = "".join(f"<tr><td style='padding:3px 10px;color:#c7d2fe;'>{cn}</td>"
                     f"<td style='padding:3px 10px;color:#94a3b8;text-align:right;'>{n}</td></tr>"
                     for cn,n in sorted(counts.items(),key=lambda x:-x[1]))
    
    rows_t = "".join(f"<tr><td style='padding:3px 10px;color:#86efac;'>{DOC_TYPE_LABELS.get(t,t)}</td>"
                     f"<td style='padding:3px 10px;color:#94a3b8;text-align:right;'>{n}</td></tr>"
                     for t,n in sorted(types.items(),key=lambda x:-x[1]))
    
    return f"""
    <div style='display:flex;gap:14px;flex-wrap:wrap;'>
      <div style='flex:1;min-width:200px;background:#1e2030;border:1px solid #2d3148;border-radius:10px;padding:14px;'>
        <div style='color:#a5b4fc;font-weight:600;margin-bottom:8px;font-size:.9rem;'>
          📊 {len(retriever.chunks):,} chunks total
          {"&nbsp;·&nbsp;<span style='color:#34d399;'>✨ "+str(custom)+" perso</span>" if custom else ""}
        </div>
        <table style='width:100%;border-collapse:collapse;font-size:.78rem;'>
          <tr><th style='color:#475569;text-align:left;padding:3px 10px;'>Cancer</th>
              <th style='color:#475569;text-align:right;padding:3px 10px;'>Chunks</th></tr>
          {rows_c}
        </table>
      </div>
      <div style='flex:1;min-width:200px;background:#1e2030;border:1px solid #2d3148;border-radius:10px;padding:14px;'>
        <div style='color:#a5b4fc;font-weight:600;margin-bottom:8px;font-size:.9rem;'>📁 Par type de document</div>
        <table style='width:100%;border-collapse:collapse;font-size:.78rem;'>
          <tr><th style='color:#475569;text-align:left;padding:3px 10px;'>Type</th>
              <th style='color:#475569;text-align:right;padding:3px 10px;'>Chunks</th></tr>
          {rows_t}
        </table>
      </div>
    </div>"""

def enrich_from_json(file_obj):
    if file_obj is None: 
        return "⚠️ Aucun fichier sélectionné.", get_base_stats()
    try:
        raw = json.load(open(file_obj.name, encoding="utf-8"))
    except Exception as e:
        return f"❌ JSON invalide : {e}", get_base_stats()
    
    docs = raw if isinstance(raw, list) else [raw]
    new_chunks = [ch for doc in docs for ch in chunk_document(doc)]
    existing = _load_custom()
    existing.extend(docs)
    _save_custom(existing)
    return _rebuild(new_chunks), get_base_stats()

def enrich_manual(cancer_name, doc_type, title, content):
    if not cancer_name.strip() or not content.strip():
        return "⚠️ Cancer et contenu obligatoires.", get_base_stats()
    
    doc_id = f"CUSTOM_{uuid.uuid4().hex[:8].upper()}"
    doc = {"cancer_name":cancer_name.strip(),"document_type":doc_type,
           "document_id":doc_id,"definition":f"{title}: {content}" if title.strip() else content}
    new_chunks = chunk_document(doc)
    
    if not new_chunks:
        new_chunks = [{"text":f"[{cancer_name}] [manuel] {title}: {content}"[:700],
                       "metadata":{"cancer":cancer_name,"type":doc_type,"subtype":"manuel","doc_id":doc_id}}]
    
    existing=_load_custom()
    existing.append(doc)
    _save_custom(existing)
    return _rebuild(new_chunks), get_base_stats()

def enrich_text_block(cancer_name, doc_type, raw_text):
    if not cancer_name.strip() or not raw_text.strip():
        return "⚠️ Cancer et texte obligatoires.", get_base_stats()
    
    paragraphs = [p.strip() for p in raw_text.split("\n\n") if len(p.strip())>60] or [raw_text.strip()]
    new_chunks, docs = [], []
    
    for para in paragraphs:
        doc_id = f"CUSTOM_{uuid.uuid4().hex[:8].upper()}"
        new_chunks.append({"text":f"[{cancer_name}] [texte_libre] {para[:700]}",
                           "metadata":{"cancer":cancer_name,"type":doc_type,"subtype":"texte_libre","doc_id":doc_id}})
        docs.append({"cancer_name":cancer_name,"document_type":doc_type,"document_id":doc_id,"definition":para})
    
    existing=_load_custom()
    existing.extend(docs)
    _save_custom(existing)
    return _rebuild(new_chunks), get_base_stats()

def delete_custom():
    _save_custom([])
    if Path(EMBEDDINGS_PATH).exists(): 
        Path(EMBEDDINGS_PATH).unlink()
    return "🗑 Entrées personnalisées supprimées. Relancez l'initialisation.", get_base_stats()

def refresh_stats(): 
    return get_base_stats()

# ─────────────────────────────────────────────────────────────────────────────
# SOURCES HTML
# ─────────────────────────────────────────────────────────────────────────────
def format_sources_html(results):
    if not results: 
        return ""
    
    cards=[]
    for i,r in enumerate(results,1):
        label = DOC_TYPE_LABELS.get(r["type"],f"📁 {r['type']}")
        bar_w = min(int(r["score"]*200),100)
        text  = r["text"][:300].replace("<","&lt;").replace(">","&gt;")
        cards.append(f"""
        <div style="background:#1e2030;border:1px solid #2d3148;border-radius:10px;padding:14px;margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="font-weight:600;color:#a5b4fc;font-size:.85rem;">{i}. {label}</span>
            <span style="font-size:.75rem;color:#6b7280;">Score {r['score']:.3f}</span>
          </div>
          <div style="background:#111827;border-radius:4px;height:4px;margin-bottom:10px;">
            <div style="background:linear-gradient(90deg,#6366f1,#8b5cf6);width:{bar_w}%;height:4px;border-radius:4px;"></div>
          </div>
          <p style="color:#94a3b8;font-size:.82rem;margin:0;line-height:1.5;">{text}…</p>
          <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap;">
            <span style="background:#312e81;color:#c7d2fe;font-size:.72rem;padding:2px 8px;border-radius:12px;">🎯 {r['cancer']}</span>
            <span style="background:#1e3a5f;color:#93c5fd;font-size:.72rem;padding:2px 8px;border-radius:12px;">BM25 {r['bm25']:.2f}</span>
            <span style="background:#1a3040;color:#67e8f9;font-size:.72rem;padding:2px 8px;border-radius:12px;">Sém. {r['sem']:.2f}</span>
          </div>
        </div>""")
    
    return "".join(cards)

# ─────────────────────────────────────────────────────────────────────────────
# RAG QUERY
# ─────────────────────────────────────────────────────────────────────────────
def query_rag(question, n_results, use_llm, history):
    global retriever
    
    if not question.strip():
        yield history,"","⚠️ Entrez une question."
        return
    
    if retriever is None:
        yield history,"","❌ Initialisez d'abord le système."
        return
    
    t0=time.time()
    cancer=detect_cancer(question)
    qtype=detect_qtype(question)
    results=retriever.search(question,k=n_results,cancer_filter=cancer,strict=bool(cancer),qtype=qtype)
    
    if not results: 
        results=retriever.search(question,k=n_results,qtype=qtype)

    if use_llm:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            context="\n\n".join(f"[Source {i+1}]\n{r['text']}" for i,r in enumerate(results))
            struct=PROMPT_STRUCTURES.get(qtype,PROMPT_STRUCTURES["default"])
            prompt=(f"<|im_start|>system\n"
                    f"Tu es un assistant médical expert en oncologie"
                    f"{', spécialisé en '+cancer if cancer else ''}.\n"
                    f"RÈGLE ABSOLUE : Tu dois TOUJOURS répondre UNIQUEMENT en français. "
                    f"N'utilise jamais l'anglais. Chaque mot de ta réponse doit être en français.\n"
                    f"Réponds UNIQUEMENT à partir du contexte fourni. {struct}\n<|im_end|>\n"
                    f"<|im_start|>user\nCONTEXTE:\n{context}\n\n"
                    f"QUESTION (réponds en français): {question}\n<|im_end|>\n"
                    f"<|im_start|>assistant\nVoici ma réponse en français :\n")
            
            tok=AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct",trust_remote_code=True)
            mdl=AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct",trust_remote_code=True,
                torch_dtype=torch.float32,low_cpu_mem_usage=True)
            mdl.eval()
            
            inp=tok(prompt,return_tensors="pt",truncation=True,max_length=3000)
            with torch.no_grad():
                out=mdl.generate(**inp,max_new_tokens=400,do_sample=False,
                                 repetition_penalty=1.1,pad_token_id=tok.eos_token_id)
            answer=tok.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True).strip()
            
            en_kw=["the ","is ","are ","this ","that ","treatment ","cancer "]
            if sum(1 for w in en_kw if w in answer.lower())>=3:
                p2=(f"<|im_start|>system\nIMPORTANT: Réponds UNIQUEMENT en français. Jamais en anglais.\n"
                    f"Tu es un expert médical en oncologie.\n<|im_end|>\n"
                    f"<|im_start|>user\nRéponds en français à cette question.\n"
                    f"Contexte: {context[:1500]}\n\nQuestion: {question}\n<|im_end|>\n"
                    f"<|im_start|>assistant\nEn français :\n")
                i2=tok(p2,return_tensors="pt",truncation=True,max_length=3000)
                with torch.no_grad():
                    o2=mdl.generate(**i2,max_new_tokens=400,do_sample=False,
                                    repetition_penalty=1.1,pad_token_id=tok.eos_token_id)
                answer=tok.decode(o2[0][i2["input_ids"].shape[1]:],skip_special_tokens=True).strip()
                
        except Exception as e:
            answer=f"⚠️ LLM non disponible ({e})\n\n"
            answer+="\n\n---\n\n".join(f"**Source {i+1}** ({r['type']})\n{r['text'][:500]}"
                                       for i,r in enumerate(results))
    else:
        answer =f"**🔍 Top {len(results)} résultats** — *{question}*\n\n"
        answer+=f"🎯 Cancer : `{cancer or 'non spécifié'}` | Type : `{qtype or 'général'}`\n\n---\n\n"
        for i,r in enumerate(results,1):
            label=DOC_TYPE_LABELS.get(r["type"],r["type"])
            answer+=f"### {i}. {label} — score {r['score']:.3f}\n**Cancer :** {r['cancer']}\n\n{r['text'][:600]}…\n\n---\n\n"

    elapsed=round(time.time()-t0,2)
    meta=f"⏱ {elapsed}s | 🎯 {cancer or '—'} | 🏷 {qtype or '—'} | 📦 {len(results)} sources"
    history=history or []
    history.append({"role":"user","content":question})
    history.append({"role":"assistant","content":answer})
    yield history, format_sources_html(results), meta

def clear_chat(): 
    return [],"",""

# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE GRADIO
# ─────────────────────────────────────────────────────────────────────────────
DARK_CSS="""
body,.gradio-container{background:#0f1117 !important;color:#e2e8f0;font-family:'Inter',sans-serif;}
.gr-button-primary{background:linear-gradient(135deg,#6366f1,#8b5cf6)!important;border:none!important;
  color:#fff!important;border-radius:8px!important;font-weight:600!important;}
.gr-button-secondary{background:#1e2030!important;border:1px solid #2d3148!important;
  color:#a5b4fc!important;border-radius:8px!important;}
.gr-textbox textarea,.gr-textbox input{background:#1e2030!important;border:1px solid #2d3148!important;
  color:#e2e8f0!important;border-radius:8px!important;}
.tab-nav button{color:#94a3b8!important;background:transparent!important;border-bottom:2px solid transparent!important;}
.tab-nav button.selected{color:#a5b4fc!important;border-bottom:2px solid #6366f1!important;}
.gr-accordion{background:#1e2030!important;border:1px solid #2d3148!important;border-radius:8px!important;}
footer{display:none!important;}
"""

EXAMPLES=[
    "Quel est le traitement du cancer de la prostate à un stade avancé ?",
    "Effets secondaires chimiothérapie cancer du poumon petites cellules",
    "Comment diagnostiquer le cancer de l'ovaire ?",
    "Mécanismes de résistance au mélanome",
    "Suivi après cancer de la thyroïde",
    "Urgences oncologiques — hypercalcémie maligne",
    "Protocole HER2 positif cancer du sein",
]

with gr.Blocks(css=DARK_CSS,title="RAG Oncologie",theme=gr.themes.Base()) as demo:

    gr.HTML("""
    <div style="text-align:center;padding:24px 0 12px;">
      <h1 style="font-size:2rem;font-weight:700;
                 background:linear-gradient(135deg,#818cf8,#c084fc);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px;">
        🏥 RAG Oncologie
      </h1>
      <p style="color:#64748b;font-size:.9rem;">
        Système de recherche augmentée · Oncologie · Français
      </p>
    </div>""")

    with gr.Tabs():

        # ── 1. CHAT ──────────────────────────────────────────────────────────
        with gr.Tab("💬 Assistant"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot=gr.Chatbot(label="Conversation",height=460,
                                      layout="bubble",placeholder="Posez une question pour commencer…")
                    with gr.Row():
                        question_box=gr.Textbox(placeholder="Posez votre question médicale…",
                                                label="",scale=5,container=False)
                        send_btn =gr.Button("Envoyer ▶",variant="primary",scale=1)
                        clear_btn=gr.Button("🗑",scale=0,min_width=48)
                    gr.Examples(EXAMPLES,inputs=question_box,label="Questions fréquentes")

                with gr.Column(scale=2):
                    with gr.Accordion("⚙️ Paramètres",open=True):
                        n_results=gr.Slider(2,10,value=5,step=1,label="Nombre de sources")
                        use_llm  =gr.Checkbox(label="Activer LLM Phi-3 (RAM requise)",value=False)
                    meta_box=gr.Textbox(label="Métadonnées",interactive=False,lines=1)
                    gr.HTML("<div style='color:#64748b;font-size:.8rem;margin:6px 0;'>Sources récupérées</div>")
                    sources_out=gr.HTML("<p style='color:#475569;font-size:.85rem;padding:10px;'>"
                                        "Les sources apparaîtront ici.</p>")

            def _send(q,n,u,h): 
                yield from query_rag(q,n,u,h)
            
            send_btn.click(_send,[question_box,n_results,use_llm,chatbot],[chatbot,sources_out,meta_box])
            question_box.submit(_send,[question_box,n_results,use_llm,chatbot],[chatbot,sources_out,meta_box])
            clear_btn.click(clear_chat,[],[chatbot,sources_out,meta_box])

        # ── 2. ENRICHISSEMENT ────────────────────────────────────────────────
        with gr.Tab("📥 Enrichir la base"):
            gr.HTML("""
            <div style='background:#1e2030;border:1px solid #2d3148;border-radius:10px;
                        padding:16px;margin-bottom:16px;'>
              <h3 style='color:#a5b4fc;margin-top:0;font-size:1rem;'>
                Ajoutez vos propres données pour enrichir la base de connaissances
              </h3>
              <p style='color:#64748b;font-size:.85rem;margin:0;'>
                Les ajouts sont sauvegardés dans <code>custom_entries.json</code> et intégrés
                immédiatement au retriever sans redémarrage.
              </p>
            </div>""")

            enrich_msg   = gr.Textbox(label="Résultat", interactive=False, lines=2)
            stats_html   = gr.HTML(get_base_stats())

            with gr.Tabs():

                # 2a. Upload JSON
                with gr.Tab("📂 Fichier JSON"):
                    gr.Markdown("""
**Format attendu** : un fichier JSON contenant une liste de documents (ou un seul document).
Chaque document doit avoir au minimum `cancer_name`, `document_type` et au moins un champ de contenu
(ex. `definition`, `protocols`, `toxicity_profiles`, etc.).

```json
[
  {
    "cancer_name": "Cancer du sein",
    "document_type": "cancer_knowledge",
    "document_id": "BREAST_custom_001",
    "definition": "Nouvelle découverte sur les récepteurs HER2..."
  }
]
```""")
                    json_file = gr.File(label="Sélectionnez un fichier JSON", file_types=[".json"])
                    json_btn  = gr.Button("⬆️ Importer et intégrer", variant="primary")
                    json_btn.click(enrich_from_json, [json_file], [enrich_msg, stats_html])

                # 2b. Formulaire manuel
                with gr.Tab("✏️ Saisie manuelle"):
                    gr.Markdown("Ajoutez une entrée directement depuis l'interface.")
                    with gr.Row():
                        m_cancer  = gr.Dropdown(choices=ALL_CANCERS, label="Cancer *", allow_custom_value=True)
                        m_doctype = gr.Dropdown(choices=ALL_DOC_TYPES, value="cancer_knowledge", label="Type de document *")
                    m_title   = gr.Textbox(label="Titre / Sujet", placeholder="ex: Nouveau protocole FOLFOX modifié")
                    m_content = gr.Textbox(label="Contenu *", lines=6,
                                          placeholder="Décrivez ici le contenu médical à ajouter…")
                    man_btn   = gr.Button("➕ Ajouter à la base", variant="primary")
                    man_btn.click(enrich_manual, [m_cancer, m_doctype, m_title, m_content], [enrich_msg, stats_html])

                # 2c. Texte libre (copier-coller d'un article)
                with gr.Tab("📋 Texte libre"):
                    gr.Markdown("""Collez un texte long (article, résumé d'étude, guideline…).
Il sera découpé automatiquement en paragraphes et indexé.""")
                    with gr.Row():
                        t_cancer  = gr.Dropdown(choices=ALL_CANCERS, label="Cancer *", allow_custom_value=True)
                        t_doctype = gr.Dropdown(choices=ALL_DOC_TYPES, value="medical_literature", label="Type *")
                    t_text  = gr.Textbox(label="Texte à indexer *", lines=10,
                                         placeholder="Collez ici votre texte…\n\nSéparez les sections par une ligne vide.")
                    txt_btn = gr.Button("📥 Indexer le texte", variant="primary")
                    txt_btn.click(enrich_text_block, [t_cancer, t_doctype, t_text], [enrich_msg, stats_html])

            # Refresh stats + supprimer
            with gr.Row():
                refresh_btn = gr.Button("🔄 Actualiser les stats", scale=2)
                delete_btn  = gr.Button("🗑 Supprimer toutes les entrées perso", variant="secondary", scale=1)
            refresh_btn.click(refresh_stats, [], stats_html)
            delete_btn.click(delete_custom, [], [enrich_msg, stats_html])

        # ── 3. INITIALISATION ────────────────────────────────────────────────
        with gr.Tab("🚀 Initialisation"):
            gr.HTML("""
            <div style='background:#1e2030;border:1px solid #2d3148;border-radius:10px;
                        padding:20px;margin-bottom:16px;'>
              <h3 style='color:#a5b4fc;margin-top:0;'>Démarrage du système</h3>
              <ol style='color:#94a3b8;line-height:2;font-size:.9rem;'>
                <li>Placez <code>merged_cancers_vfrancais.json</code> dans le même dossier.</li>
                <li>Cliquez sur <strong>Initialiser</strong>.</li>
                <li>Premier lancement : calcul des embeddings (mis en cache → rapide ensuite).</li>
                <li>Les entrées personnalisées (<code>custom_entries.json</code>) sont chargées automatiquement.</li>
              </ol>
            </div>""")
            init_btn   = gr.Button("🚀 Initialiser le système", variant="primary")
            status_box = gr.Textbox(value=init_status, label="Statut", interactive=False, lines=2)
            init_btn.click(load_system, [], status_box)

        # ── 4. À PROPOS ──────────────────────────────────────────────────────
        with gr.Tab("ℹ️ À propos"):
            gr.Markdown("""
## Architecture RAG Oncologie

| Composant | Détail |
|---|---|
| **Retrieval BM25** | Recherche lexicale (rank-bm25) |
| **Retrieval sémantique** | `paraphrase-multilingual-MiniLM-L12-v2` |
| **Fusion hybride** | BM25 × 0.35 + Sémantique × 0.65 + Cohérence |
| **LLM optionnel** | Microsoft Phi-3 Mini 4k Instruct |
| **Enrichissement** | JSON · Formulaire manuel · Texte libre |

### Cancers couverts
Prostate · Sein · Colorectal · Poumon (SCLC/CPNPC) · Thyroïde · Ovaire ·
Mélanome · Gliomes HGG · Gastrique · Rein · Vessie · Col de l'utérus

*100% open-source — fonctionne en local.*
            """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)
PYEOF