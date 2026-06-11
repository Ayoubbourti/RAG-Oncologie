"""
Interface Gradio pour RAG Oncologie
====================================
Lancez avec : python rag_oncologie_ui.py
Puis ouvrez http://localhost:7860
"""

import json
import pickle
import re
import time
import warnings
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

import numpy as np
import gradio as gr

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH       = "merged_cancers_vfrancais.json"
EMBEDDINGS_PATH = "embeddings.pkl"
MODEL_NAME      = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES RAG
# ──────────────────────────────────────────────────────────────────────────────
PREFIX_TO_CANCER = {
    "SCLC":     "Cancer du poumon à petites cellules",
    "CPNPC":    "Cancer Pulmonaire Non à Petites Cellules (CPNPC)",
    "BREAST":   "Cancer du sein",
    "CRC":      "Cancer Colorectal (CCR)",
    "PROSTATE": "Cancer de la prostate",
    "THYROID":  "Cancer de la thyroïde",
    "OVARIAN":  "Cancer épithélial de l'ovaire",
    "MEL":      "Mélanome",
    "HGG":      "Gliomes de haut grade",
    "STOMACH":  "Cancer gastrique",
    "KIDNEY":   "Cancer du rein",
    "BLADDER":  "Cancer de la vessie",
    "CERVICAL": "Cancer du col de l'utérus",
}

NAME_NORMALIZATION = {
    "Thyroid Cancer": "Cancer de la thyroïde",
    "Prostate Cancer": "Cancer de la prostate",
    "Kidney Cancer (Renal Cell Carcinoma)": "Cancer du rein",
    "Gastric Cancer (Gastric Adenocarcinoma)": "Cancer gastrique",
    "Gastric Cancer": "Cancer gastrique",
    "Breast Cancer": "Cancer du sein",
    "Ovarian Cancer": "Cancer épithélial de l'ovaire",
    "Colorectal Cancer": "Cancer Colorectal (CCR)",
    "Melanoma": "Mélanome",
    "Glioma": "Gliomes de haut grade",
    "Small Cell Lung Cancer": "Cancer du poumon à petites cellules",
}

CANCER_KEYWORDS = {
    "Cancer de la prostate":                              ["prostate"],
    "Cancer du sein":                                     ["sein", "breast", "her2", "brca"],
    "Cancer Colorectal (CCR)":                            ["colorectal", "colon", "rectum", "ccr"],
    "Cancer du poumon à petites cellules":                ["poumon petites cellules", "sclc", "petites cellules"],
    "Cancer Pulmonaire Non à Petites Cellules (CPNPC)":   ["cpnpc", "nsclc", "poumon non"],
    "Cancer de la thyroïde":                              ["thyroïde", "thyroid"],
    "Cancer épithélial de l'ovaire":                      ["ovaire", "ovarian"],
    "Mélanome":                                           ["mélanome", "melanome", "melanoma"],
    "Gliomes de haut grade":                              ["gliome", "glioblastome", "gbm", "hgg"],
    "Cancer gastrique":                                   ["gastrique", "estomac", "gastric"],
    "Cancer du rein":                                     ["rein", "rénal", "carcinome rénal"],
    "Cancer de la vessie":                                ["vessie", "bladder", "urothélial"],
    "Cancer du col de l'utérus":                          ["col", "utérus", "cervical", "cervix"],
}

QTYPE_KEYWORDS = {
    "traitement":         ["traitement", "protocole", "thérapie", "chimiothérapie", "immunothérapie", "chirurgie", "radiothérapie", "ligne", "prise en charge"],
    "diagnostic":         ["diagnostic", "diagnostiquer", "détection", "bilan", "examens", "dépistage", "imagerie", "biopsie"],
    "effets_secondaires": ["effets secondaires", "toxicité", "tolérance", "effets indésirables", "nausées", "neutropénie"],
    "suivi":              ["suivi", "surveillance", "contrôle", "récidive", "rémission"],
    "pronostic":          ["pronostic", "survie", "mortalité", "facteur pronostique"],
    "facteurs_risque":    ["facteur de risque", "risque", "prédisposition", "génétique"],
    "resistance":         ["résistance", "réfractaire", "échec thérapeutique", "rechute"],
    "urgence":            ["urgence", "urgences", "compression", "hypercalcémie", "neutropénie fébrile"],
}

QTYPE_TO_DOCTYPE = {
    "traitement":         "treatment_protocol",
    "diagnostic":         "diagnostic_guideline",
    "effets_secondaires": "toxicity_management",
    "suivi":              "followup",
    "pronostic":          "cancer_knowledge",
    "facteurs_risque":    "cancer_knowledge",
    "resistance":         "resistance_mechanism",
    "urgence":            "oncology_emergency",
}

QTYPE_PREFERRED_SUBTYPES = {
    "traitement":         ["traitement_protocole"],
    "diagnostic":         ["guideline_diagnostic"],
    "effets_secondaires": ["toxicité"],
    "suivi":              ["suivi"],
    "pronostic":          ["connaissance_prognosis"],
    "facteurs_risque":    ["connaissance_risk_factors"],
    "resistance":         ["résistance"],
    "urgence":            ["urgence"],
}

PROMPT_STRUCTURES = {
    "traitement":         "Structure: 1) Lignes de traitement, 2) Médicaments et doses, 3) Critères de choix.",
    "diagnostic":         "Structure: 1) Examens initiaux, 2) Confirmation diagnostique, 3) Staging.",
    "pronostic":          "Structure: 1) Facteurs pronostiques, 2) Survie par stade, 3) Récidive.",
    "effets_secondaires": "Structure: 1) Effets fréquents, 2) Effets graves (grade 3-4), 3) Prise en charge.",
    "suivi":              "Structure: 1) Calendrier de suivi, 2) Examens recommandés, 3) Signes d'alarme.",
    "facteurs_risque":    "Structure: 1) Facteurs non modifiables, 2) Facteurs modifiables, 3) Dépistage.",
    "default":            "Réponds de manière structurée et cliniquement pertinente.",
}

DOC_TYPE_LABELS = {
    "treatment_protocol":           "🧬 Protocole",
    "cancer_knowledge":             "📚 Connaissance",
    "metastasis":                   "🔴 Métastase",
    "palliative_care":              "💙 Palliatif",
    "toxicity_management":          "⚠️ Toxicité",
    "followup":                     "📅 Suivi",
    "resistance_mechanism":         "🔬 Résistance",
    "staging_system":               "📊 Stadification",
    "oncology_score":               "🔢 Score",
    "oncology_emergency":           "🚨 Urgence",
    "contraindication_interaction": "💊 Contre-indication",
    "diagnostic_guideline":         "🔍 Guideline diag.",
    "medical_literature":           "📄 Littérature",
    "clinical_reasoning":           "🩺 Cas clinique",
    "biomarker_genetics":           "🧪 Biomarqueur",
}


# ──────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ──────────────────────────────────────────────────────────────────────────────

def detect_cancer(query: str) -> Optional[str]:
    q = query.lower()
    for cancer, kw_list in CANCER_KEYWORDS.items():
        for kw in kw_list:
            if kw in q:
                return cancer
    return None


def detect_qtype(query: str) -> Optional[str]:
    q = query.lower()
    for qtype, kw_list in QTYPE_KEYWORDS.items():
        for kw in kw_list:
            if kw in q:
                return qtype
    return None


def s(val, maxlen=400):
    if val is None or val == "" or val == []:
        return ""
    if isinstance(val, list):
        parts = [s(v, maxlen) for v in val if v]
        return "; ".join(p for p in parts if p)[:maxlen]
    if isinstance(val, dict):
        parts = [f"{k}: {s(v, 100)}" for k, v in val.items() if v]
        return " | ".join(parts)[:maxlen]
    return str(val)[:maxlen]


def chunk_document(item: Dict) -> List[Dict]:
    chunks = []
    cancer   = item.get("cancer_name", "unknown")
    doc_type = item.get("document_type", "unknown")
    doc_id   = item.get("document_id", "unknown")

    def add(text: str, subtype: str):
        text = text.strip()
        if len(text) > 60:
            chunks.append({
                "text": f"[{cancer}] [{subtype}] {text[:700]}",
                "metadata": {"cancer": cancer, "type": doc_type, "subtype": subtype, "doc_id": doc_id},
            })

    if doc_type == "treatment_protocol":
        for p in item.get("protocols", []):
            drugs_str = s([f"{d.get('drug_name','?')} {d.get('dose','')}" for d in p.get("drugs", [])])
            add(
                f"Protocole: {p.get('protocol_name','N/A')}. Cancer: {p.get('cancer_type','')}. "
                f"Stade: {p.get('stage','')}. Ligne: {p.get('line_of_therapy','')}. "
                f"Médicaments: {drugs_str}. Résultats: {s(p.get('expected_outcomes',[]))}",
                "traitement_protocole",
            )
    elif doc_type == "cancer_knowledge":
        field_map = {
            "definition": "Définition", "epidemiology": "Épidémiologie",
            "risk_factors": "Facteurs de risque", "common_symptoms": "Symptômes fréquents",
            "screening_methods": "Dépistage", "diagnostic_tests": "Tests diagnostiques",
            "prognosis": "Pronostic", "pathophysiology": "Physiopathologie",
        }
        for key, label in field_map.items():
            val = item.get(key)
            if val:
                add(f"{label}: {s(val)}", f"connaissance_{key}")
    elif doc_type == "metastasis":
        for p in item.get("metastatic_profiles", []):
            add(
                f"Métastase de {p.get('primary_cancer','?')} vers {p.get('metastatic_site','?')}. "
                f"Traitement: {s(p.get('treatment_options',[]))}.",
                "métastase",
            )
    elif doc_type == "toxicity_management":
        for p in item.get("toxicity_profiles", []):
            add(
                f"Toxicité: {p.get('toxicity_name','?')}. "
                f"Médicaments: {s(p.get('causing_drugs',[]))}. "
                f"Gestion: {s(p.get('management_by_grade',{}))}.",
                "toxicité",
            )
    elif doc_type == "followup":
        for p in item.get("followup_protocols", []):
            add(
                f"Suivi — Cancer: {p.get('cancer_type','?')}. Fréquence: {p.get('followup_frequency','')}. "
                f"Tests: {s(p.get('recommended_tests',[]))}. Signes alarme: {s(p.get('warning_signs',[]))}.",
                "suivi",
            )
    elif doc_type == "resistance_mechanism":
        for p in item.get("resistance_profiles", []):
            add(
                f"Résistance: {p.get('drug_name','?')}. Mécanisme: {p.get('mechanism','')}. "
                f"Alternatifs: {s(p.get('alternative_treatments',[]))}.",
                "résistance",
            )
    elif doc_type == "oncology_emergency":
        for p in item.get("oncology_emergencies", []):
            add(
                f"Urgence: {p.get('emergency_name','?')}. Définition: {p.get('definition','')}. "
                f"Symptômes: {s(p.get('symptoms',[]))}. Actions: {s(p.get('immediate_actions',[]))}.",
                "urgence",
            )
    elif doc_type == "diagnostic_guideline":
        for p in item.get("diagnostic_guidelines", []):
            add(
                f"Guideline: {p.get('cancer_type','?')}. Tests: {s(p.get('recommended_tests',[]))}. "
                f"Algorithme: {s(p.get('diagnostic_algorithm',[]))}.",
                "guideline_diagnostic",
            )
    elif doc_type == "medical_literature":
        for p in item.get("literature_evidence", []):
            add(
                f"Étude: {p.get('title','?')} ({p.get('journal','')} {p.get('year','')}). "
                f"Résultats: {s(p.get('main_findings',[]))}.",
                "littérature",
            )
    elif doc_type == "contraindication_interaction":
        for p in item.get("drug_safety_profiles", []):
            add(
                f"Sécurité: {p.get('drug_name','?')}. "
                f"Contre-indications: {s(p.get('contraindications',[]))}. "
                f"Interactions: {s(p.get('drug_interactions',[]))}.",
                "contre_indication",
            )
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# RETRIEVER
# ──────────────────────────────────────────────────────────────────────────────

class HybridRetriever:
    DOC_TYPE_BOOST = {
        "treatment_protocol": 1.4, "cancer_knowledge": 1.2,
        "toxicity_management": 1.3, "followup": 1.2,
        "resistance_mechanism": 1.3, "oncology_emergency": 1.5,
        "diagnostic_guideline": 1.2, "medical_literature": 0.9,
    }

    def __init__(self, chunks, embeddings, encode_fn, bm25_w=0.35, sem_w=0.65):
        from rank_bm25 import BM25Okapi
        self.chunks     = chunks
        self.embeddings = embeddings
        self.encode_fn  = encode_fn
        self.bm25_w     = bm25_w
        self.sem_w      = sem_w
        tokenized       = [c["text"].lower().split() for c in chunks]
        self.bm25       = BM25Okapi(tokenized)

    @staticmethod
    def _mm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    def search(self, query, k=5, cancer_filter=None, strict=True, qtype=None):
        bm25_raw  = np.array(self.bm25.get_scores(query.lower().split()))
        q_emb     = self.encode_fn(query)
        sem_raw   = np.dot(self.embeddings, q_emb)
        bm25_norm = self._mm(bm25_raw)
        sem_norm  = self._mm(sem_raw)
        coherence = bm25_norm * sem_norm
        hybrid    = self.bm25_w * bm25_norm + self.sem_w * sem_norm + 0.15 * coherence

        preferred    = QTYPE_PREFERRED_SUBTYPES.get(qtype, []) if qtype else []
        target_dtype = QTYPE_TO_DOCTYPE.get(qtype)
        results, seen_idx = [], set()

        if target_dtype and cancer_filter:
            for h in self._collect(hybrid, bm25_norm, sem_norm, preferred,
                                   cancer_filter, strict=True, force_dtype=target_dtype, k=2):
                results.append(h)
                seen_idx.add(h["_idx"])

        results.extend(self._collect(hybrid, bm25_norm, sem_norm, preferred,
                                     cancer_filter, strict=strict, exclude_idx=seen_idx, k=k))

        if target_dtype and not any(r["type"] == target_dtype for r in results):
            fallback = self._collect(hybrid, bm25_norm, sem_norm, preferred,
                                     cancer_filter=None, strict=False, force_dtype=target_dtype,
                                     k=2, exclude_idx={r["_idx"] for r in results})
            results = fallback + results

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        for r in results:
            r.pop("_idx", None)
        return results[:k]

    def _collect(self, hybrid, bm25_norm, sem_norm, preferred,
                 cancer_filter, strict, force_dtype=None, exclude_idx=None, k=10):
        exclude_idx = exclude_idx or set()
        results = []
        for idx in np.argsort(hybrid)[::-1]:
            if idx in exclude_idx or len(results) >= k:
                continue
            c       = self.chunks[idx]
            cancer  = c["metadata"].get("cancer", "unknown")
            dtype   = c["metadata"].get("type", "unknown")
            subtype = c["metadata"].get("subtype", "")
            if force_dtype and dtype != force_dtype:
                continue
            if cancer_filter:
                if strict and cancer_filter.lower() != cancer.lower():
                    continue
                elif not strict and cancer_filter.lower() not in cancer.lower():
                    continue
            boost = 1.6 if subtype in preferred else 1.0
            score = hybrid[idx] * self.DOC_TYPE_BOOST.get(dtype, 1.0) * boost
            results.append({
                "text": c["text"], "score": float(score),
                "cancer": cancer, "type": dtype, "subtype": subtype,
                "bm25": float(bm25_norm[idx]), "sem": float(sem_norm[idx]),
                "_idx": int(idx),
            })
        return results

    def search_auto(self, query, k=5):
        cancer = detect_cancer(query)
        qtype  = detect_qtype(query)
        return self.search(query, k=k, cancer_filter=cancer, strict=bool(cancer), qtype=qtype)


# ──────────────────────────────────────────────────────────────────────────────
# ÉTAT GLOBAL
# ──────────────────────────────────────────────────────────────────────────────
retriever   = None
init_status = "⏳ Non initialisé — cliquez sur **Initialiser le système**"


def load_system(progress=gr.Progress()):
    global retriever, init_status

    if not Path(DATA_PATH).exists():
        init_status = f"❌ Fichier introuvable : `{DATA_PATH}`\nPlacez votre JSON dans le même dossier."
        return init_status

    progress(0.05, desc="Chargement des données…")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    progress(0.15, desc="Normalisation des noms…")
    for doc in data:
        name = doc.get("cancer_name", "") or ""
        if name in NAME_NORMALIZATION:
            doc["cancer_name"] = NAME_NORMALIZATION[name]
        elif not name or name == "unknown":
            prefix = doc.get("document_id", "").split("_")[0]
            cancer = PREFIX_TO_CANCER.get(prefix)
            if cancer:
                doc["cancer_name"] = cancer

    progress(0.30, desc="Création des chunks…")
    all_chunks = []
    for doc in data:
        all_chunks.extend(chunk_document(doc))

    progress(0.45, desc="Chargement du modèle d'embeddings…")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        init_status = f"❌ Erreur modèle : {e}"
        return init_status

    embeddings = None
    if Path(EMBEDDINGS_PATH).exists():
        progress(0.60, desc="Chargement du cache d'embeddings…")
        with open(EMBEDDINGS_PATH, "rb") as f:
            embeddings = pickle.load(f)
        if len(embeddings) != len(all_chunks):
            embeddings = None

    if embeddings is None:
        progress(0.65, desc="Calcul des embeddings (peut prendre quelques minutes)…")
        texts      = [c["text"] for c in all_chunks]
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        with open(EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(embeddings, f)

    progress(0.90, desc="Initialisation du retriever…")
    encode_fn = lambda q: model.encode([q], normalize_embeddings=True)[0]
    retriever = HybridRetriever(all_chunks, embeddings, encode_fn)

    counts  = Counter(c["metadata"]["cancer"] for c in all_chunks)
    n_types = len(set(c["metadata"]["type"] for c in all_chunks))
    init_status = (
        f"✅ Système prêt — {len(all_chunks):,} chunks | "
        f"{len(data)} documents | {len(counts)} cancers | {n_types} types"
    )
    progress(1.0, desc="Terminé !")
    return init_status


def format_sources_html(results: List[Dict]) -> str:
    if not results:
        return ""
    cards = []
    for i, r in enumerate(results, 1):
        label  = DOC_TYPE_LABELS.get(r["type"], f"📁 {r['type']}")
        bar_w  = min(int(r["score"] * 200), 100)
        text   = r["text"][:300].replace("<", "&lt;").replace(">", "&gt;")
        cards.append(f"""
        <div style="background:#1e2030;border:1px solid #2d3148;border-radius:10px;padding:14px 16px;margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <span style="font-weight:600;color:#a5b4fc;font-size:0.85rem;">{i}. {label}</span>
            <span style="font-size:0.75rem;color:#6b7280;">Score {r['score']:.3f}</span>
          </div>
          <div style="background:#111827;border-radius:4px;height:4px;margin-bottom:10px;">
            <div style="background:linear-gradient(90deg,#6366f1,#8b5cf6);width:{bar_w}%;height:4px;border-radius:4px;"></div>
          </div>
          <p style="color:#94a3b8;font-size:0.82rem;margin:0;line-height:1.5;">{text}…</p>
          <div style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;">
            <span style="background:#312e81;color:#c7d2fe;font-size:0.72rem;padding:2px 8px;border-radius:12px;">🎯 {r['cancer']}</span>
            <span style="background:#1e3a5f;color:#93c5fd;font-size:0.72rem;padding:2px 8px;border-radius:12px;">BM25 {r['bm25']:.2f}</span>
            <span style="background:#1a3040;color:#67e8f9;font-size:0.72rem;padding:2px 8px;border-radius:12px;">Sém. {r['sem']:.2f}</span>
          </div>
        </div>""")
    return "".join(cards)


def query_rag(question: str, n_results: int, use_llm: bool, history: list):
    global retriever

    if not question.strip():
        yield history, "", "⚠️ Entrez une question."
        return

    if retriever is None:
        yield history, "", "❌ Initialisez d'abord le système (onglet Initialisation)."
        return

    t0     = time.time()
    cancer = detect_cancer(question)
    qtype  = detect_qtype(question)

    results = retriever.search(question, k=n_results, cancer_filter=cancer,
                               strict=bool(cancer), qtype=qtype)
    if not results:
        results = retriever.search(question, k=n_results, qtype=qtype)

    if use_llm:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            context = "\n\n".join(f"[Source {i+1}]\n{r['text']}" for i, r in enumerate(results))
            struct  = PROMPT_STRUCTURES.get(qtype, PROMPT_STRUCTURES["default"])
            prompt  = (
                f"<|im_start|>system\n"
                f"Tu es un assistant médical expert en oncologie"
                f"{', spécialisé en ' + cancer if cancer else ''}.\n"
                f"RÈGLE ABSOLUE : Tu dois TOUJOURS répondre UNIQUEMENT en français. "
                f"N'utilise jamais l'anglais. Chaque mot de ta réponse doit être en français.\n"
                f"Réponds UNIQUEMENT à partir du contexte fourni. Si l'information manque, dis-le en français.\n"
                f"{struct}\n"
                f"<|im_end|>\n"
                f"<|im_start|>user\n"
                f"CONTEXTE:\n{context}\n\n"
                f"QUESTION (réponds en français): {question}\n"
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"Voici ma réponse en français :\n"
            )
            tok = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True,
                torch_dtype=torch.float32, low_cpu_mem_usage=True)
            mdl.eval()
            inp = tok(prompt, return_tensors="pt", truncation=True, max_length=3000)
            with torch.no_grad():
                out = mdl.generate(**inp, max_new_tokens=400, do_sample=False,
                                   repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
            answer = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            # Détection anglais → relance en forçant le français
            english_markers = ["the ", "is ", "are ", "this ", "that ", "treatment ", "cancer "]
            if sum(1 for w in english_markers if w in answer.lower()) >= 3:
                prompt2 = ("<|im_start|>system\nIMPORTANT: Tu dois répondre UNIQUEMENT en français. Jamais en anglais.\nTu es un expert médical en oncologie.\n<|im_end|>\n<|im_start|>user\nRéponds en français à cette question en utilisant ce contexte.\nContexte: " + context[:1500] + "\n\nQuestion: " + question + "\n<|im_end|>\n<|im_start|>assistant\nVoici la réponse en français:\n")
                inp2 = tok(prompt2, return_tensors="pt", truncation=True, max_length=3000)
                with torch.no_grad():
                    out2 = mdl.generate(**inp2, max_new_tokens=400, do_sample=False, repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
                answer = tok.decode(out2[0][inp2["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            answer  = f"⚠️ LLM non disponible ({e})\n\n"
            answer += "\n\n---\n\n".join(
                f"**Source {i+1}** ({r['type']})\n{r['text'][:500]}"
                for i, r in enumerate(results))
    else:
        answer  = f"**🔍 Top {len(results)} résultats** — *{question}*\n\n"
        answer += f"🎯 Cancer : `{cancer or 'non spécifié'}` | Type : `{qtype or 'général'}`\n\n---\n\n"
        for i, r in enumerate(results, 1):
            label  = DOC_TYPE_LABELS.get(r["type"], r["type"])
            answer += f"### {i}. {label} — score {r['score']:.3f}\n"
            answer += f"**Cancer :** {r['cancer']}\n\n"
            answer += f"{r['text'][:600]}…\n\n---\n\n"

    elapsed      = round(time.time() - t0, 2)
    meta         = f"⏱ {elapsed}s | 🎯 {cancer or '—'} | 🏷 {qtype or '—'} | 📦 {len(results)} sources"
    sources_html = format_sources_html(results)

    history = history or []
    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": answer})

    yield history, sources_html, meta


def clear_chat():
    return [], "", ""


# ──────────────────────────────────────────────────────────────────────────────
# INTERFACE GRADIO
# ──────────────────────────────────────────────────────────────────────────────

DARK_CSS = """
body, .gradio-container { background:#0f1117 !important; color:#e2e8f0; font-family:'Inter',sans-serif; }
.gr-button-primary { background:linear-gradient(135deg,#6366f1,#8b5cf6) !important; border:none !important;
                     color:#fff !important; border-radius:8px !important; font-weight:600 !important; }
.gr-button-secondary { background:#1e2030 !important; border:1px solid #2d3148 !important;
                        color:#a5b4fc !important; border-radius:8px !important; }
.gr-textbox textarea, .gr-textbox input { background:#1e2030 !important; border:1px solid #2d3148 !important;
                                           color:#e2e8f0 !important; border-radius:8px !important; }
.tab-nav button { color:#94a3b8 !important; background:transparent !important; border-bottom:2px solid transparent !important; }
.tab-nav button.selected { color:#a5b4fc !important; border-bottom:2px solid #6366f1 !important; }
footer { display:none !important; }
"""

EXAMPLES = [
    "Quel est le traitement du cancer de la prostate à un stade avancé ?",
    "Effets secondaires chimiothérapie cancer du poumon petites cellules",
    "Comment diagnostiquer le cancer de l'ovaire ?",
    "Mécanismes de résistance au mélanome",
    "Suivi après cancer de la thyroïde",
    "Urgences oncologiques — hypercalcémie maligne",
    "Protocole HER2 positif cancer du sein",
]

with gr.Blocks(css=DARK_CSS, title="RAG Oncologie", theme=gr.themes.Base()) as demo:

    gr.HTML("""
    <div style="text-align:center;padding:28px 0 16px;">
      <h1 style="font-size:2rem;font-weight:700;
                 background:linear-gradient(135deg,#818cf8,#c084fc);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:6px;">
        🏥 RAG Oncologie
      </h1>
      <p style="color:#64748b;font-size:0.92rem;">
        Système de recherche augmentée · Base de connaissances en oncologie · Français
      </p>
    </div>
    """)

    with gr.Tabs():

        # ── Onglet 1 : Chat ───────────────────────────────────────────────────
        with gr.Tab("💬 Assistant"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=480,
                        layout="bubble",
                        placeholder="Posez une question pour commencer…",
                    )
                    with gr.Row():
                        question_box = gr.Textbox(
                            placeholder="Posez votre question médicale en oncologie…",
                            label="",
                            scale=5,
                            container=False,
                        )
                        send_btn  = gr.Button("Envoyer ▶", variant="primary", scale=1)
                        clear_btn = gr.Button("🗑", scale=0, min_width=48)

                    gr.Examples(EXAMPLES, inputs=question_box, label="Questions fréquentes")

                with gr.Column(scale=2):
                    with gr.Accordion("⚙️ Paramètres", open=True):
                        n_results = gr.Slider(2, 10, value=5, step=1, label="Nombre de sources")
                        use_llm   = gr.Checkbox(label="Activer LLM Phi-3 (RAM requise)", value=False)

                    meta_box = gr.Textbox(label="Métadonnées", interactive=False, lines=1)

                    gr.HTML("<div style='color:#64748b;font-size:0.8rem;margin:4px 0 8px;'>Sources récupérées</div>")
                    sources_html = gr.HTML(
                        value="<p style='color:#475569;font-size:0.85rem;padding:12px;'>"
                              "Les sources apparaîtront ici après votre première question.</p>"
                    )

            def _send(q, n, ulm, hist):
                yield from query_rag(q, n, ulm, hist)

            send_btn.click(_send,    [question_box, n_results, use_llm, chatbot], [chatbot, sources_html, meta_box])
            question_box.submit(_send, [question_box, n_results, use_llm, chatbot], [chatbot, sources_html, meta_box])
            clear_btn.click(clear_chat, [], [chatbot, sources_html, meta_box])

        # ── Onglet 2 : Initialisation ─────────────────────────────────────────
        with gr.Tab("🚀 Initialisation"):
            gr.HTML("""
            <div style='background:#1e2030;border:1px solid #2d3148;border-radius:10px;
                        padding:20px;margin-bottom:16px;'>
              <h3 style='color:#a5b4fc;margin-top:0;'>Démarrage du système</h3>
              <ol style='color:#94a3b8;line-height:2;'>
                <li>Placez <code>merged_cancers_vfrancais.json</code> dans le même dossier que ce script.</li>
                <li>Cliquez sur <strong>Initialiser</strong>.</li>
                <li>Le premier lancement calcule les embeddings et les met en cache (<code>embeddings.pkl</code>).</li>
                <li>Les lancements suivants seront instantanés.</li>
              </ol>
            </div>
            """)
            init_btn   = gr.Button("🚀 Initialiser le système", variant="primary")
            status_box = gr.Textbox(value=init_status, label="Statut", interactive=False, lines=2)
            init_btn.click(load_system, [], status_box)

        # ── Onglet 3 : À propos ───────────────────────────────────────────────
        with gr.Tab("ℹ️ À propos"):
            gr.Markdown("""
## Architecture RAG Oncologie

| Composant | Détail |
|---|---|
| **Retrieval BM25** | Recherche lexicale (rank-bm25) |
| **Retrieval sémantique** | `paraphrase-multilingual-MiniLM-L12-v2` |
| **Fusion hybride** | BM25 × 0.35 + Sémantique × 0.65 + Cohérence |
| **LLM optionnel** | Microsoft Phi-3 Mini 4k Instruct |

### Cancers couverts
Prostate · Sein · Colorectal · Poumon (SCLC/CPNPC) · Thyroïde · Ovaire ·
Mélanome · Gliomes HGG · Gastrique · Rein · Vessie · Col de l'utérus

*Interface Gradio — 100% open-source, fonctionne en local.*
            """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )