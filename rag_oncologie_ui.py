"""
Interface Gradio pour RAG Oncologie — Version Améliorée
=========================================================
Backend : SmartRetriever (notebook) + Qwen2-1.5B-Instruct
Nouvelles fonctionnalités :
  • Filtres avancés (cancer, type de doc, date)
  • SmartRetriever avec boost adaptatif et détection sémantique
  • LLM : Qwen2-1.5B-Instruct (local)
  • Historique des sessions persistant (JSON)
  • Export de la conversation (TXT / MD)
  • Onglet Évaluation / Benchmark RAG
  • Affichage sources complet avec snippets extensibles
  • Statistiques de la base enrichies
  • Traduction automatique des réponses (EN → FR)

Lancez avec : python rag_oncologie_ui.py
Puis ouvrez  http://localhost:7860
"""

import json, pickle, time, uuid, warnings, re, datetime
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
EMBEDDINGS_PATH  = "embeddings_v2.pkl"
CUSTOM_DATA_PATH = "custom_entries.json"
SESSIONS_PATH    = "sessions_history.json"
MODEL_NAME       = "intfloat/multilingual-e5-small"
QWEN2_MODEL      = "Qwen/Qwen2-1.5B-Instruct"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-fr"  # Modèle de traduction EN→FR

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
PREFIX_TO_CANCER = {
    "SCLC":    "Cancer du poumon à petites cellules",
    "CPNPC":   "Cancer Pulmonaire Non à Petites Cellules (CPNPC)",
    "BREAST":  "Cancer du sein",
    "CRC":     "Cancer Colorectal (CCR)",
    "PROSTATE":"Cancer de la prostate",
    "THYROID": "Cancer de la thyroïde",
    "OVARIAN": "Cancer épithélial de l'ovaire",
    "MEL":     "Mélanome",
    "HGG":     "Gliomes de haut grade",
    "STOMACH": "Cancer gastrique",
    "KIDNEY":  "Cancer du rein",
    "BLADDER": "Cancer de la vessie",
    "CERVICAL":"Cancer du col de l'utérus",
}

NAME_NORMALIZATION = {
    "Thyroid Cancer":                      "Cancer de la thyroïde",
    "Prostate Cancer":                     "Cancer de la prostate",
    "Kidney Cancer (Renal Cell Carcinoma)":"Cancer du rein",
    "Gastric Cancer (Gastric Adenocarcinoma)":"Cancer gastrique",
    "Gastric Cancer":                      "Cancer gastrique",
    "Breast Cancer":                       "Cancer du sein",
    "Ovarian Cancer":                      "Cancer épithélial de l'ovaire",
    "Colorectal Cancer":                   "Cancer Colorectal (CCR)",
    "Melanoma":                            "Mélanome",
    "Glioma":                              "Gliomes de haut grade",
    "Small Cell Lung Cancer":              "Cancer du poumon à petites cellules",
}

CANCER_ALIASES = {
    "prostate":         "Cancer de la prostate",
    "sein":             "Cancer du sein", "breast": "Cancer du sein",
    "her2":             "Cancer du sein", "brca": "Cancer du sein",
    "colorectal":       "Cancer Colorectal (CCR)",
    "colon":            "Cancer Colorectal (CCR)", "rectum": "Cancer Colorectal (CCR)",
    "poumon petites":   "Cancer du poumon à petites cellules",
    "sclc":             "Cancer du poumon à petites cellules",
    "cpnpc":            "Cancer Pulmonaire Non à Petites Cellules (CPNPC)",
    "nsclc":            "Cancer Pulmonaire Non à Petites Cellules (CPNPC)",
    "thyroïde":         "Cancer de la thyroïde", "thyroid": "Cancer de la thyroïde",
    "ovaire":           "Cancer épithélial de l'ovaire",
    "mélanome":         "Mélanome", "melanome": "Mélanome",
    "gliome":           "Gliomes de haut grade", "glioblastome": "Gliomes de haut grade",
    "gbm":              "Gliomes de haut grade",
    "gastrique":        "Cancer gastrique", "estomac": "Cancer gastrique",
    "rein":             "Cancer du rein", "rénal": "Cancer du rein",
    "vessie":           "Cancer de la vessie", "urothélial": "Cancer de la vessie",
    "col utérus":       "Cancer du col de l'utérus", "cervical": "Cancer du col de l'utérus",
}

KNOWN_CANCERS = list(PREFIX_TO_CANCER.values())

DOC_TYPE_LABELS = {
    "treatment_protocol":        "🧬 Protocole",
    "cancer_knowledge":          "📚 Connaissance",
    "metastasis":                "🔴 Métastase",
    "palliative_care":           "💙 Palliatif",
    "toxicity_management":       "⚠️ Toxicité",
    "followup":                  "📅 Suivi",
    "resistance_mechanism":      "🔬 Résistance",
    "staging_system":            "📊 Stadification",
    "oncology_score":            "🔢 Score",
    "oncology_emergency":        "🚨 Urgence",
    "contraindication_interaction":"💊 Contre-indication",
    "diagnostic_guideline":      "🔍 Guideline diag.",
    "medical_literature":        "📄 Littérature",
    "clinical_reasoning":        "🩺 Cas clinique",
    "biomarker_genetics":        "🧪 Biomarqueur",
    "drug":                      "💉 Médicament",
    "custom":                    "✨ Personnalisé",
}

QTYPE_TO_DOCTYPE = {
    "traitement":       "treatment_protocol",
    "diagnostic":       "diagnostic_guideline",
    "effets_secondaires":"toxicity_management",
    "suivi":            "followup",
    "pronostic":        "cancer_knowledge",
    "facteurs_risque":  "cancer_knowledge",
    "resistance":       "resistance_mechanism",
    "urgence":          "oncology_emergency",
    "metastase":        "metastasis",
}

QTYPE_PREFERRED_SUBTYPES = {
    "traitement":        ["traitement_protocole"],
    "diagnostic":        ["guideline_diagnostic"],
    "effets_secondaires":["toxicité", "toxicite"],
    "suivi":             ["suivi"],
    "pronostic":         ["connaissance_prognosis", "connaissance_prognose"],
    "facteurs_risque":   ["connaissance_risk_factors"],
    "resistance":        ["résistance", "resistance"],
    "urgence":           ["urgence"],
    "metastase":         ["métastase", "metastase"],
}

PROMPT_STRUCTURES = {
    "traitement":        "Structure: 1) Lignes de traitement, 2) Médicaments et doses, 3) Critères de choix.",
    "diagnostic":        "Structure: 1) Examens initiaux, 2) Confirmation diagnostique, 3) Staging.",
    "pronostic":         "Structure: 1) Facteurs pronostiques, 2) Survie par stade, 3) Récidive.",
    "effets_secondaires":"Structure: 1) Effets fréquents, 2) Effets graves (grade 3-4), 3) Prise en charge.",
    "suivi":             "Structure: 1) Calendrier de suivi, 2) Examens recommandés, 3) Signes d'alarme.",
    "facteurs_risque":   "Structure: 1) Facteurs non modifiables, 2) Facteurs modifiables, 3) Dépistage.",
    "urgence":           "Structure: 1) Reconnaissance, 2) Actions immédiates, 3) Traitement.",
    "metastase":         "Structure: 1) Site métastatique, 2) Options thérapeutiques, 3) Pronostic.",
    "default":           "Réponds de manière structurée et cliniquement pertinente.",
}

ALL_DOC_TYPES = sorted(DOC_TYPE_LABELS.keys())
ALL_CANCERS   = sorted(PREFIX_TO_CANCER.values())

EXAMPLES = [
    "Quel est le traitement du cancer de la prostate à un stade avancé ?",
    "Effets secondaires chimiothérapie cancer du poumon petites cellules",
    "Comment diagnostiquer le cancer de l'ovaire ?",
    "Mécanismes de résistance au mélanome",
    "Suivi après cancer de la thyroïde",
    "Urgences oncologiques — hypercalcémie maligne",
    "Protocole HER2 positif cancer du sein",
    "Métastases osseuses du cancer de la prostate",
    "Facteurs de risque du cancer colorectal",
    "Stadification du glioblastome",
]

# ─────────────────────────────────────────────────────────────────────────────
# TRADUCTEUR (EN → FR)
# ─────────────────────────────────────────────────────────────────────────────
class Translator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False
    
    def load(self, progress_fn=None):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            if progress_fn: progress_fn(0.1, desc="Chargement tokenizer traduction…")
            self.tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
            if progress_fn: progress_fn(0.5, desc="Chargement modèle traduction…")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL)
            self.loaded = True
            if progress_fn: progress_fn(1.0, desc="Traducteur prêt !")
            return True, "✅ Traducteur EN→FR chargé"
        except Exception as e:
            return False, f"⚠️ Traducteur non disponible (fonctionne sans traduction) : {e}"
    
    def translate(self, text: str, max_length: int = 512) -> str:
        """Traduit un texte de l'anglais vers le français"""
        if not self.loaded or not text:
            return text
        
        try:
            # Détection rapide : si le texte est déjà majoritairement français, on ne traduit pas
            french_markers = ['le ', 'la ', 'les ', 'de ', 'et ', 'dans ', 'pour ', 'avec ']
            en_markers = ['the ', 'and ', 'of ', 'to ', 'in ', 'for ', 'with ']
            
            text_lower = text.lower()[:200]
            french_count = sum(1 for m in french_markers if m in text_lower)
            english_count = sum(1 for m in en_markers if m in text_lower)
            
            # Si le texte a plus de marqueurs français, on ne traduit pas
            if french_count > english_count:
                return text
            
            # Traduction
            inputs = self.tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated
        except Exception as e:
            print(f"Erreur traduction: {e}")
            return text

translator = Translator()

# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────
def s(val, maxlen=400):
    if not val: return ""
    if isinstance(val, list):  return "; ".join(str(v) for v in val if v)[:maxlen]
    if isinstance(val, dict):  return " | ".join(f"{k}: {v}" for k,v in val.items() if v)[:maxlen]
    return str(val)[:maxlen]

def chunk_document(item: Dict) -> List[Dict]:
    chunks   = []
    cancer   = item.get("cancer_name", "unknown")
    doc_type = item.get("document_type", "unknown")
    doc_id   = item.get("document_id", "unknown")

    def add(text, subtype):
        text = text.strip()
        if len(text) > 60:
            chunks.append({"text": f"[{cancer}] [{subtype}] {text[:700]}",
                           "metadata": {"cancer": cancer, "type": doc_type,
                                        "subtype": subtype, "doc_id": doc_id}})

    if doc_type == "treatment_protocol":
        for p in item.get("protocols", []):
            drugs = s([f"{d.get('drug_name','?')} {d.get('dose','')}" for d in p.get("drugs",[])])
            add(f"Protocole: {p.get('protocol_name','N/A')}. Stade: {p.get('stage','')}. "
                f"Ligne: {p.get('line_of_therapy','')}. Médicaments: {drugs}. "
                f"Résultats: {s(p.get('expected_outcomes',[]))}", "traitement_protocole")
    elif doc_type == "cancer_knowledge":
        for key, label in [
            ("definition","Définition"), ("epidemiology","Épidémiologie"),
            ("risk_factors","Facteurs de risque"), ("common_symptoms","Symptômes fréquents"),
            ("screening_methods","Dépistage"), ("diagnostic_tests","Tests diagnostiques"),
            ("prognosis","Pronostic"), ("pathophysiology","Physiopathologie")]:
            val = item.get(key)
            if val: add(f"{label}: {s(val)}", f"connaissance_{key}")
    elif doc_type == "metastasis":
        for p in item.get("metastatic_profiles", []):
            add(f"Métastase de {p.get('primary_cancer','?')} vers {p.get('metastatic_site','?')}. "
                f"Traitement: {s(p.get('treatment_options',[]))}", "métastase")
    elif doc_type == "toxicity_management":
        for p in item.get("toxicity_profiles", []):
            add(f"Toxicité: {p.get('toxicity_name','?')}. "
                f"Médicaments: {s(p.get('causing_drugs',[]))}. "
                f"Gestion: {s(p.get('management_by_grade',{}))}", "toxicité")
    elif doc_type == "followup":
        for p in item.get("followup_protocols", []):
            add(f"Suivi — Cancer: {p.get('cancer_type','?')}. "
                f"Tests: {s(p.get('recommended_tests',[]))}. "
                f"Signes alarme: {s(p.get('warning_signs',[]))}", "suivi")
    elif doc_type == "resistance_mechanism":
        for p in item.get("resistance_profiles", []):
            add(f"Résistance: {p.get('drug_name','?')}. Mécanisme: {p.get('mechanism','')}. "
                f"Alternatifs: {s(p.get('alternative_treatments',[]))}", "résistance")
    elif doc_type == "oncology_emergency":
        for p in item.get("oncology_emergencies", []):
            add(f"Urgence: {p.get('emergency_name','?')}. {p.get('definition','')}. "
                f"Actions: {s(p.get('immediate_actions',[]))}", "urgence")
    elif doc_type == "diagnostic_guideline":
        for p in item.get("diagnostic_guidelines", []):
            add(f"Guideline: {p.get('cancer_type','?')}. Tests: {s(p.get('recommended_tests',[]))}. "
                f"Algorithme: {s(p.get('diagnostic_algorithm',[]))}", "guideline_diagnostic")
    elif doc_type == "medical_literature":
        for p in item.get("literature_evidence", []):
            add(f"Étude: {p.get('title','?')} ({p.get('journal','')} {p.get('year','')}). "
                f"Résultats: {s(p.get('main_findings',[]))}", "littérature")
    elif doc_type == "contraindication_interaction":
        for p in item.get("drug_safety_profiles", []):
            add(f"Sécurité: {p.get('drug_name','?')}. "
                f"Contre-indications: {s(p.get('contraindications',[]))}. "
                f"Interactions: {s(p.get('drug_interactions',[]))}", "contre_indication")
    elif doc_type in ("drug", "drug_profile_standard"):
        for p in item.get("drug_profiles", []):
            add(f"Médicament: {p.get('drug_name','?')}. Classe: {p.get('drug_class','')}. "
                f"Indications: {s(p.get('approved_indications',[]))}", "medicament")
    elif doc_type == "staging_system":
        for p in item.get("staging_systems", []):
            add(f"Stadification: {p.get('staging_system','')}. "
                f"Description: {p.get('system_description','')}", "stadification")

    for field in ("definition", "content", "text", "description"):
        val = item.get(field)
        if val and not chunks:
            add(str(val), "general")
            break
    return chunks

# ─────────────────────────────────────────────────────────────────────────────
# SMART RETRIEVER (backend notebook)
# ─────────────────────────────────────────────────────────────────────────────
class SmartRetriever:
    """
    Retriever hybride intelligent (notebook RAG_Oncologie) :
    - Détection sémantique du type de question (regex avancés)
    - Détection automatique du cancer (aliases + partial)
    - Boost adaptatif par type de document selon qtype
    - Fusion BM25 + sémantique + cohérence
    - Déduplication des résultats
    """

    TYPE_BOOST_MAP = {
        "treatment_protocol":          {"traitement": 2.5, "general": 1.5},
        "drug":                        {"traitement": 1.8, "general": 1.2},
        "toxicity_management":         {"effets_secondaires": 3.0, "general": 1.3},
        "contraindication_interaction":{"effets_secondaires": 2.0},
        "metastasis":                  {"metastase": 3.0, "general": 1.4},
        "oncology_emergency":          {"urgence": 3.0, "general": 1.5},
        "diagnostic_guideline":        {"diagnostic": 2.5, "general": 1.3},
        "biomarker_genetics":          {"diagnostic": 2.0},
        "staging_system":              {"pronostic": 2.0, "general": 1.2},
        "cancer_knowledge":            {"pronostic": 1.5, "general": 1.1,
                                        "facteurs_risque": 1.8},
        "followup":                    {"suivi": 2.5, "general": 1.2},
        "resistance_mechanism":        {"resistance": 2.5, "general": 1.3},
        "medical_literature":          {"general": 0.9},
    }

    QTYPE_PATTERNS = {
        "traitement":        [r"traitement|thérapie|protocole|médicament|chimioth|immunoth|radioth|prise en charge|comment traiter"],
        "effets_secondaires":[r"effet.?secondaire|toxicit|indésirable|nausée|vomissement|fatigue|tolérance"],
        "metastase":         [r"métastase|métastatique|dissémination|localisation secondaire"],
        "urgence":           [r"urgence|compression médullaire|hypercalcémie|aplasie|syndrome de lyse"],
        "diagnostic":        [r"diagnostic|dépistage|marqueur|biomarqueur|examen|imagerie|comment diagnostiquer|quel examen"],
        "pronostic":         [r"pronostic|survie|récidive|évolution|espérance de vie|facteur pronostique"],
        "suivi":             [r"suivi|surveillance|post.traitement|après traitement|comment suivre"],
        "resistance":        [r"résistance|mécanisme de résistance|acquise|primaire|échappement"],
        "facteurs_risque":   [r"facteur.?de.?risque|risque|prédisposition|génétique|hérédita"],
    }

    def __init__(self, chunks, embeddings, encode_fn):
        from rank_bm25 import BM25Okapi
        self.chunks     = chunks
        self.embeddings = embeddings
        self.encode_fn  = encode_fn
        self.bm25       = BM25Okapi([self._tok(c["text"]) for c in chunks])

    def _tok(self, text):
        return [t for t in re.sub(r"[^\w\s]", " ", text.lower()).split() if len(t) > 1]

    def _norm(self, arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10: return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    def detect_cancer(self, query: str) -> Optional[str]:
        q = query.lower()
        for alias, cancer in CANCER_ALIASES.items():
            if alias in q: return cancer
        for cancer in KNOWN_CANCERS:
            if cancer.lower() in q: return cancer
        return None

    def detect_qtype(self, query: str) -> str:
        q = query.lower()
        scores = {}
        for qtype, patterns in self.QTYPE_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, q, re.IGNORECASE))
            if score > 0: scores[qtype] = score
        return max(scores, key=scores.get) if scores else "general"

    def search(self, query, k=5, cancer_filter=None, qtype=None,
               doc_type_filter=None, strict_cancer=True):
        # Auto-détection si non fourni
        if cancer_filter is None:
            cancer_filter = self.detect_cancer(query)
        if qtype is None:
            qtype = self.detect_qtype(query)

        # Scores bruts
        bm25_raw = np.array(self.bm25.get_scores(self._tok(query)))
        q_emb    = self.encode_fn(query)
        sem_raw  = np.dot(self.embeddings, q_emb).flatten()

        # Normalisation
        bm25_n = self._norm(bm25_raw)
        sem_n  = self._norm(sem_raw)

        # Fusion hybride avec cohérence
        coherence = bm25_n * sem_n
        hybrid    = 0.35 * bm25_n + 0.50 * sem_n + 0.15 * coherence

        # Boost par type de doc et qtype
        boosted = hybrid.copy()
        preferred_subs = QTYPE_PREFERRED_SUBTYPES.get(qtype, [])
        for idx in range(len(self.chunks)):
            dtype   = self.chunks[idx]["metadata"].get("type", "unknown")
            subtype = self.chunks[idx]["metadata"].get("subtype", "")
            boosts  = self.TYPE_BOOST_MAP.get(dtype, {})
            boost   = boosts.get(qtype, boosts.get("general", 1.0))
            if subtype in preferred_subs: boost *= 1.4
            boosted[idx] = hybrid[idx] * boost
            # Bonus cancer exact
            doc_cancer = self.chunks[idx]["metadata"].get("cancer", "")
            if cancer_filter and doc_cancer == cancer_filter:
                boosted[idx] *= 1.2

        results, seen_texts = [], set()
        for idx in np.argsort(boosted)[::-1]:
            if len(results) >= k: break
            c        = self.chunks[idx]
            dtype    = c["metadata"].get("type", "unknown")
            doc_cncr = c["metadata"].get("cancer", "unknown")

            # Filtre type de doc
            if doc_type_filter and dtype != doc_type_filter:
                continue

            # Filtre cancer
            if cancer_filter:
                match = (cancer_filter.lower() == doc_cncr.lower() if strict_cancer
                         else cancer_filter.lower() in doc_cncr.lower())
                if not match:
                    if len(results) == 0 and len(seen_texts) == 0:
                        pass  # On garde au moins 1 résultat
                    else:
                        continue

            # Déduplication
            preview = c["text"][:100]
            if preview in seen_texts: continue
            seen_texts.add(preview)

            results.append({
                "text":    c["text"],
                "score":   float(boosted[idx]),
                "cancer":  doc_cncr,
                "type":    dtype,
                "subtype": c["metadata"].get("subtype", ""),
                "bm25":    float(bm25_n[idx]),
                "sem":     float(sem_n[idx]),
            })

        # Fallback sans filtre cancer si rien trouvé
        if not results and cancer_filter:
            return self.search(query, k=k, cancer_filter=None, qtype=qtype,
                               doc_type_filter=doc_type_filter, strict_cancer=False)
        return results

# ─────────────────────────────────────────────────────────────────────────────
# QWEN2 LLM
# ─────────────────────────────────────────────────────────────────────────────
class Qwen2LLM:
    def __init__(self):
        self.model     = None
        self.tokenizer = None
        self.loaded    = False

    def load(self, progress_fn=None):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            if progress_fn: progress_fn(0.1, desc="Chargement tokenizer Qwen2…")
            self.tokenizer = AutoTokenizer.from_pretrained(
                QWEN2_MODEL, trust_remote_code=True)
            if progress_fn: progress_fn(0.5, desc="Chargement modèle Qwen2-1.5B…")
            self.model = AutoModelForCausalLM.from_pretrained(
                QWEN2_MODEL,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self.model.eval()
            self.loaded = True
            if progress_fn: progress_fn(1.0, desc="Qwen2 prêt !")
            return True, "✅ Qwen2-1.5B-Instruct chargé"
        except Exception as e:
            return False, f"❌ Erreur Qwen2 : {e}"

    def generate(self, prompt: str, max_new_tokens: int = 350) -> tuple:
        if not self.loaded:
            return "❌ Qwen2 non chargé. Initialisez-le dans l'onglet Initialisation.", 0.0
        import torch
        # Le prompt demande explicitement une réponse en français
        formatted = (
            f"<|im_start|>system\n"
            f"Tu es un assistant médical expert en oncologie. "
            f"Tu dois répondre UNIQUEMENT en français, de façon précise et structurée. "
            f"Cite toujours les sources [1], [2]… si disponibles.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = self.tokenizer(formatted, return_tensors="pt",
                                truncation=True, max_length=2000)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.time() - t0
        response = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return response, latency

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAT GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
retriever   = None
embed_model = None
qwen2_llm   = Qwen2LLM()
init_status = "⏳ Non initialisé — cliquez sur Initialiser le système"
translate_enabled = True  # Option de traduction activée par défaut

def _load_custom():
    return json.load(open(CUSTOM_DATA_PATH, encoding="utf-8")) if Path(CUSTOM_DATA_PATH).exists() else []

def _save_custom(entries):
    json.dump(entries, open(CUSTOM_DATA_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def _load_sessions():
    return json.load(open(SESSIONS_PATH, encoding="utf-8")) if Path(SESSIONS_PATH).exists() else []

def _save_sessions(sessions):
    json.dump(sessions, open(SESSIONS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
def load_system(progress=gr.Progress()):
    global retriever, init_status, embed_model

    if not Path(DATA_PATH).exists():
        init_status = f"❌ Fichier introuvable : {DATA_PATH}"
        return init_status, get_base_stats()

    progress(0.05, desc="Chargement des données…")
    data = json.load(open(DATA_PATH, encoding="utf-8"))
    data.extend(_load_custom())

    progress(0.15, desc="Normalisation…")
    for doc in data:
        name = doc.get("cancer_name", "") or ""
        if name in NAME_NORMALIZATION:
            doc["cancer_name"] = NAME_NORMALIZATION[name]
        elif not name or name == "unknown":
            pfx = doc.get("document_id", "").split("_")[0]
            c = PREFIX_TO_CANCER.get(pfx)
            if c: doc["cancer_name"] = c

    progress(0.30, desc="Chunking…")
    all_chunks = [ch for doc in data for ch in chunk_document(doc)]

    progress(0.45, desc="Modèle d'embeddings…")
    try:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        init_status = f"❌ Modèle embeddings : {e}"
        return init_status, get_base_stats()

    embeddings = None
    if Path(EMBEDDINGS_PATH).exists():
        progress(0.60, desc="Cache embeddings…")
        embeddings = pickle.load(open(EMBEDDINGS_PATH, "rb"))
        if len(embeddings) != len(all_chunks): embeddings = None

    if embeddings is None:
        progress(0.65, desc="Calcul embeddings (patience)…")
        embeddings = embed_model.encode(
            [f"passage: {c['text']}" for c in all_chunks],
            batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        pickle.dump(embeddings, open(EMBEDDINGS_PATH, "wb"))

    progress(0.90, desc="SmartRetriever…")
    retriever = SmartRetriever(
        all_chunks, embeddings,
        lambda q: embed_model.encode([f"query: {q}"], normalize_embeddings=True)[0])

    counts = Counter(c["metadata"]["cancer"] for c in all_chunks)
    custom_n = len(_load_custom())
    init_status = (f"✅ Prêt — {len(all_chunks):,} chunks | {len(data)} docs "
                   f"({custom_n} perso) | {len(counts)} cancers | SmartRetriever actif")
    progress(1.0, desc="Terminé !")
    return init_status, get_base_stats()

def load_qwen2(progress=gr.Progress()):
    ok, msg = qwen2_llm.load(progress_fn=lambda v, desc="": progress(v, desc=desc))
    return msg

def load_translator(progress=gr.Progress()):
    ok, msg = translator.load(progress_fn=lambda v, desc="": progress(v, desc=desc))
    return msg

# ─────────────────────────────────────────────────────────────────────────────
# ENRICHISSEMENT
# ─────────────────────────────────────────────────────────────────────────────
def _rebuild(new_chunks):
    global retriever, embed_model
    if retriever is None: return "❌ Initialisez d'abord le système."
    if not new_chunks:    return "⚠️ Aucun chunk valide."
    new_embs   = embed_model.encode(
        [f"passage: {c['text']}" for c in new_chunks],
        batch_size=32, normalize_embeddings=True)
    all_chunks = retriever.chunks + new_chunks
    all_embs   = np.vstack([retriever.embeddings, new_embs])
    pickle.dump(all_embs, open(EMBEDDINGS_PATH, "wb"))
    retriever  = SmartRetriever(
        all_chunks, all_embs,
        lambda q: embed_model.encode([f"query: {q}"], normalize_embeddings=True)[0])
    return f"✅ {len(new_chunks)} chunks ajoutés — total : {len(all_chunks):,}"

def enrich_from_json(file_obj):
    if file_obj is None: return "⚠️ Aucun fichier sélectionné.", get_base_stats()
    try:
        raw = json.load(open(file_obj.name, encoding="utf-8"))
    except Exception as e:
        return f"❌ JSON invalide : {e}", get_base_stats()
    docs = raw if isinstance(raw, list) else [raw]
    new_chunks = [ch for doc in docs for ch in chunk_document(doc)]
    existing = _load_custom(); existing.extend(docs); _save_custom(existing)
    return _rebuild(new_chunks), get_base_stats()

def enrich_manual(cancer_name, doc_type, title, content):
    if not cancer_name.strip() or not content.strip():
        return "⚠️ Cancer et contenu obligatoires.", get_base_stats()
    doc_id = f"CUSTOM_{uuid.uuid4().hex[:8].upper()}"
    doc = {"cancer_name": cancer_name.strip(), "document_type": doc_type,
           "document_id": doc_id,
           "definition": f"{title}: {content}" if title.strip() else content}
    new_chunks = chunk_document(doc)
    if not new_chunks:
        new_chunks = [{"text": f"[{cancer_name}] [manuel] {title}: {content}"[:700],
                       "metadata": {"cancer": cancer_name, "type": doc_type,
                                    "subtype": "manuel", "doc_id": doc_id}}]
    existing = _load_custom(); existing.append(doc); _save_custom(existing)
    return _rebuild(new_chunks), get_base_stats()

def enrich_text_block(cancer_name, doc_type, raw_text):
    if not cancer_name.strip() or not raw_text.strip():
        return "⚠️ Cancer et texte obligatoires.", get_base_stats()
    paragraphs = [p.strip() for p in raw_text.split("\n\n") if len(p.strip()) > 60] or [raw_text.strip()]
    new_chunks, docs = [], []
    for para in paragraphs:
        doc_id = f"CUSTOM_{uuid.uuid4().hex[:8].upper()}"
        new_chunks.append({"text": f"[{cancer_name}] [texte_libre] {para[:700]}",
                           "metadata": {"cancer": cancer_name, "type": doc_type,
                                        "subtype": "texte_libre", "doc_id": doc_id}})
        docs.append({"cancer_name": cancer_name, "document_type": doc_type,
                     "document_id": doc_id, "definition": para})
    existing = _load_custom(); existing.extend(docs); _save_custom(existing)
    return _rebuild(new_chunks), get_base_stats()

def delete_custom():
    _save_custom([])
    if Path(EMBEDDINGS_PATH).exists(): Path(EMBEDDINGS_PATH).unlink()
    return "🗑 Entrées supprimées. Relancez l'initialisation.", get_base_stats()

def refresh_stats(): return get_base_stats()

# ─────────────────────────────────────────────────────────────────────────────
# STATISTIQUES HTML
# ─────────────────────────────────────────────────────────────────────────────
def get_base_stats():
    if retriever is None:
        return "<p style='color:#64748b;padding:12px;'>Système non initialisé.</p>"

    counts = Counter(c["metadata"]["cancer"] for c in retriever.chunks)
    types  = Counter(c["metadata"]["type"]   for c in retriever.chunks)
    custom = sum(1 for c in retriever.chunks if c["metadata"].get("subtype") in
                 ("manuel", "texte_libre", "general"))
    qwen_status = "✅ Qwen2 prêt" if qwen2_llm.loaded else "⚫ Qwen2 non chargé"
    trans_status = "🌐 Traduction activée" if translator.loaded else "⚠️ Traduction non disponible"

    rows_c = "".join(
        f"<tr><td style='padding:3px 10px;color:#c7d2fe;'>{cn}</td>"
        f"<td style='padding:3px 10px;color:#94a3b8;text-align:right;'>{n}</td></tr>"
        for cn, n in sorted(counts.items(), key=lambda x: -x[1]))

    rows_t = "".join(
        f"<tr><td style='padding:3px 10px;color:#86efac;'>{DOC_TYPE_LABELS.get(t,t)}</td>"
        f"<td style='padding:3px 10px;color:#94a3b8;text-align:right;'>{n}</td></tr>"
        for t, n in sorted(types.items(), key=lambda x: -x[1]))

    return f"""
    <div style='margin-bottom:10px;display:flex;gap:10px;flex-wrap:wrap;'>
      <span style='background:#1e3a5f;color:#93c5fd;font-size:.8rem;padding:4px 12px;border-radius:20px;'>
        📦 {len(retriever.chunks):,} chunks
      </span>
      <span style='background:#1a3040;color:#67e8f9;font-size:.8rem;padding:4px 12px;border-radius:20px;'>
        🧠 SmartRetriever actif
      </span>
      <span style='background:#1e2030;color:#a5b4fc;font-size:.8rem;padding:4px 12px;border-radius:20px;'>
        {qwen_status}
      </span>
      <span style='background:#2d3748;color:#86efac;font-size:.8rem;padding:4px 12px;border-radius:20px;'>
        {trans_status}
      </span>
      {"<span style='background:#14532d;color:#86efac;font-size:.8rem;padding:4px 12px;border-radius:20px;'>✨ "+str(custom)+" perso</span>" if custom else ""}
    </div>
    <div style='display:flex;gap:14px;flex-wrap:wrap;'>
      <div style='flex:1;min-width:200px;background:#1e2030;border:1px solid #2d3148;border-radius:10px;padding:14px;'>
        <div style='color:#a5b4fc;font-weight:600;margin-bottom:8px;font-size:.9rem;'>🎯 Par cancer</div>
        <table style='width:100%;border-collapse:collapse;font-size:.78rem;'>
          <tr><th style='color:#475569;text-align:left;padding:3px 10px;'>Cancer</th>
              <th style='color:#475569;text-align:right;padding:3px 10px;'>Chunks</th></tr>
          {rows_c}
        </table>
      </div>
      <div style='flex:1;min-width:200px;background:#1e2030;border:1px solid #2d3148;border-radius:10px;padding:14px;'>
        <div style='color:#a5b4fc;font-weight:600;margin-bottom:8px;font-size:.9rem;'>📁 Par type</div>
        <table style='width:100%;border-collapse:collapse;font-size:.78rem;'>
          <tr><th style='color:#475569;text-align:left;padding:3px 10px;'>Type</th>
              <th style='color:#475569;text-align:right;padding:3px 10px;'>Chunks</th></tr>
          {rows_t}
        </table>
      </div>
    </div>"""

# ─────────────────────────────────────────────────────────────────────────────
# SOURCES HTML (snippets extensibles)
# ─────────────────────────────────────────────────────────────────────────────
def format_sources_html(results):
    if not results: return ""
    cards = []
    for i, r in enumerate(results, 1):
        label    = DOC_TYPE_LABELS.get(r["type"], f"📁 {r['type']}")
        bar_w    = min(int(r["score"] * 180), 100)
        preview  = r["text"][:280].replace("<", "&lt;").replace(">", "&gt;")
        full     = r["text"].replace("<", "&lt;").replace(">", "&gt;")
        
        # Traduction optionnelle des sources
        if translator.loaded:
            preview_trans = translator.translate(preview)
            full_trans = translator.translate(full)
        else:
            preview_trans = preview
            full_trans = full
            
        uid      = f"src_{i}_{int(time.time()*1000)}"
        cards.append(f"""
        <div style="background:#1e2030;border:1px solid #2d3148;border-radius:10px;
                    padding:14px;margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="font-weight:600;color:#a5b4fc;font-size:.85rem;">{i}. {label}</span>
            <span style="font-size:.75rem;color:#6b7280;">Score {r['score']:.3f}</span>
          </div>
          <div style="background:#111827;border-radius:4px;height:4px;margin-bottom:10px;">
            <div style="background:linear-gradient(90deg,#6366f1,#8b5cf6);
                        width:{bar_w}%;height:4px;border-radius:4px;"></div>
          </div>
          <p id="{uid}_short" style="color:#94a3b8;font-size:.82rem;margin:0 0 6px;line-height:1.5;">
            {preview_trans}…
          </p>
          <p id="{uid}_full" style="color:#94a3b8;font-size:.82rem;margin:0 0 6px;line-height:1.5;display:none;">
            {full_trans}
          </p>
          <button onclick="
            var s=document.getElementById('{uid}_short');
            var f=document.getElementById('{uid}_full');
            var b=this;
            if(f.style.display==='none'){{f.style.display='block';s.style.display='none';b.textContent='▲ Réduire';}}
            else{{f.style.display='none';s.style.display='block';b.textContent='▼ Voir tout';}}
          " style="background:none;border:1px solid #374151;color:#6b7280;font-size:.7rem;
                   padding:2px 8px;border-radius:6px;cursor:pointer;">▼ Voir tout</button>
          <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap;">
            <span style="background:#312e81;color:#c7d2fe;font-size:.72rem;padding:2px 8px;border-radius:12px;">
              🎯 {r['cancer']}
            </span>
            <span style="background:#1e3a5f;color:#93c5fd;font-size:.72rem;padding:2px 8px;border-radius:12px;">
              BM25 {r['bm25']:.2f}
            </span>
            <span style="background:#1a3040;color:#67e8f9;font-size:.72rem;padding:2px 8px;border-radius:12px;">
              Sém. {r['sem']:.2f}
            </span>
            <span style="background:#1c1f2e;color:#fbbf24;font-size:.72rem;padding:2px 8px;border-radius:12px;">
              {r.get('subtype','—')}
            </span>
          </div>
        </div>""")
    return "".join(cards)

# ─────────────────────────────────────────────────────────────────────────────
# SESSIONS
# ─────────────────────────────────────────────────────────────────────────────
def save_session(history, session_name):
    if not history: return "⚠️ Conversation vide."
    sessions = _load_sessions()
    name = session_name.strip() or f"Session {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}"
    sessions.append({"id": uuid.uuid4().hex[:8], "name": name,
                     "date": datetime.datetime.now().isoformat(),
                     "messages": history})
    _save_sessions(sessions)
    return f"✅ Session « {name} » sauvegardée."

def list_sessions():
    sessions = _load_sessions()
    if not sessions:
        return "<p style='color:#64748b;padding:12px;'>Aucune session sauvegardée.</p>"
    rows = "".join(
        f"<tr>"
        f"<td style='padding:6px 10px;color:#c7d2fe;'>{s['name']}</td>"
        f"<td style='padding:6px 10px;color:#64748b;font-size:.8rem;'>"
        f"{s['date'][:16].replace('T',' ')}</td>"
        f"<td style='padding:6px 10px;color:#94a3b8;text-align:right;'>"
        f"{len(s['messages'])//2} échanges</td>"
        f"</tr>"
        for s in reversed(sessions))
    return f"""
    <div style='background:#1e2030;border:1px solid #2d3148;border-radius:10px;padding:14px;'>
      <div style='color:#a5b4fc;font-weight:600;margin-bottom:10px;'>
        📚 {len(sessions)} session(s) sauvegardée(s)
      </div>
      <table style='width:100%;border-collapse:collapse;font-size:.82rem;'>
        <tr><th style='color:#475569;text-align:left;padding:6px 10px;'>Nom</th>
            <th style='color:#475569;text-align:left;padding:6px 10px;'>Date</th>
            <th style='color:#475569;text-align:right;padding:6px 10px;'>Échanges</th></tr>
        {rows}
      </table>
    </div>"""

def export_conversation(history, fmt):
    if not history: return None
    lines = []
    if fmt == "Markdown":
        lines.append("# Conversation RAG Oncologie\n")
        lines.append(f"*Exporté le {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}*\n\n---\n")
        for msg in history:
            role = "👤 **Utilisateur**" if msg["role"] == "user" else "🏥 **Assistant**"
            lines.append(f"### {role}\n{msg['content']}\n\n---\n")
        ext, mime = "md", "text/markdown"
    else:
        lines.append(f"CONVERSATION RAG ONCOLOGIE\n")
        lines.append(f"Exporté le {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
        lines.append("=" * 60 + "\n")
        for msg in history:
            role = "UTILISATEUR" if msg["role"] == "user" else "ASSISTANT"
            lines.append(f"\n[{role}]\n{msg['content']}\n\n{'-'*40}\n")
        ext, mime = "txt", "text/plain"

    path = f"/tmp/rag_export_{uuid.uuid4().hex[:6]}.{ext}"
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return path

# ─────────────────────────────────────────────────────────────────────────────
# QUERY RAG PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def query_rag(question, n_results, use_llm, llm_tokens,
              cancer_override, doctype_override, history):
    global retriever

    if not question.strip():
        yield history, "", "⚠️ Entrez une question."
        return
    if retriever is None:
        yield history, "", "❌ Initialisez d'abord le système."
        return
    if use_llm and not qwen2_llm.loaded:
        yield history, "", "❌ Chargez d'abord Qwen2 dans l'onglet Initialisation."
        return

    t0 = time.time()

    # Filtres manuels
    cancer_filter   = cancer_override  if cancer_override  not in ("", "Auto") else None
    doctype_filter  = doctype_override if doctype_override not in ("", "Tous") else None

    # Détection automatique
    detected_cancer = retriever.detect_cancer(question)
    detected_qtype  = retriever.detect_qtype(question)
    cancer_used     = cancer_filter or detected_cancer
    qtype_used      = detected_qtype

    results = retriever.search(
        question, k=n_results,
        cancer_filter=cancer_used,
        qtype=qtype_used,
        doc_type_filter=doctype_filter,
        strict_cancer=bool(cancer_used))

    if not results:
        results = retriever.search(question, k=n_results)

    if use_llm:
        context = "\n\n".join(
            f"[Source {i+1}] ({r['type']} — {r['cancer']})\n{r['text']}"
            for i, r in enumerate(results))
        struct = PROMPT_STRUCTURES.get(qtype_used, PROMPT_STRUCTURES["default"])
        prompt = (f"CONTEXTE MÉDICAL:\n{context}\n\n"
                  f"QUESTION: {question}\n\n"
                  f"{struct}\n"
                  f"Réponds précisément en français en citant les sources [1], [2]…")
        answer, llm_latency = qwen2_llm.generate(prompt, max_new_tokens=int(llm_tokens))
        if not answer.strip():
            answer = "⚠️ Réponse vide — essayez de reformuler."
    else:
        answer  = f"**🔍 Top {len(results)} résultats** — *{question}*\n\n"
        answer += (f"🎯 Cancer : `{cancer_used or 'non spécifié'}` | "
                   f"Type : `{qtype_used}` | Mode : RAG pur\n\n---\n\n")
        for i, r in enumerate(results, 1):
            label  = DOC_TYPE_LABELS.get(r["type"], r["type"])
            # Traduction du texte de la source si nécessaire
            source_text = r['text'][:700]
            if translator.loaded:
                source_text = translator.translate(source_text)
            answer += (f"### {i}. {label} — score {r['score']:.3f}\n"
                       f"**Cancer :** {r['cancer']}\n\n"
                       f"{source_text}…\n\n---\n\n")
        llm_latency = 0.0

    elapsed = round(time.time() - t0, 2)
    meta = (f"⏱ {elapsed}s"
            f"{f' (dont LLM {llm_latency:.1f}s)' if use_llm else ''} | "
            f"🎯 {cancer_used or '—'} | 🏷 {qtype_used} | 📦 {len(results)} sources"
            f"{f' | 🔍 filtre: {doctype_filter}' if doctype_filter else ''}"
            f"{' | 🌐 Traduction activée' if translator.loaded else ''}")

    history = history or []
    history.append({"role": "user",    "content": question})
    history.append({"role": "assistant","content": answer})
    yield history, format_sources_html(results), meta

def clear_chat():
    return [], "", ""

def set_translation(enable):
    global translate_enabled
    translate_enabled = enable
    return f"🌐 Traduction {'activée' if enable else 'désactivée'}"

# ─────────────────────────────────────────────────────────────────────────────
# ÉVALUATION / BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
BENCH_QUESTIONS = [
    ("Quel est le traitement de première ligne du cancer du sein HER2+ ?", "Cancer du sein", "traitement"),
    ("Effets secondaires de la doxorubicine ?", None, "effets_secondaires"),
    ("Comment diagnostiquer le cancer colorectal ?", "Cancer Colorectal (CCR)", "diagnostic"),
    ("Mécanismes de résistance à l'erlotinib ?", None, "resistance"),
    ("Signes d'hypercalcémie maligne ?", None, "urgence"),
    ("Suivi après cancer de la thyroïde opéré ?", "Cancer de la thyroïde", "suivi"),
    ("Pronostic du glioblastome stade IV ?", "Gliomes de haut grade", "pronostic"),
    ("Métastases hépatiques du cancer colorectal — traitements ?", "Cancer Colorectal (CCR)", "metastase"),
]

def run_benchmark():
    if retriever is None:
        return "<p style='color:#ef4444;'>❌ Initialisez d'abord le système.</p>"

    rows = []
    total_time = 0.0
    for q, cancer, qtype in BENCH_QUESTIONS:
        t0 = time.time()
        results = retriever.search(q, k=3, cancer_filter=cancer, qtype=qtype)
        elapsed = round(time.time() - t0, 4)
        total_time += elapsed

        top = results[0] if results else None
        cancer_ok = (top["cancer"] == cancer) if (top and cancer) else "—"
        qtype_ok  = any(r["type"] == QTYPE_TO_DOCTYPE.get(qtype, "") for r in results)

        rows.append(
            f"<tr>"
            f"<td style='padding:6px 10px;color:#e2e8f0;font-size:.8rem;max-width:280px;'>{q[:60]}…</td>"
            f"<td style='padding:6px 10px;color:#a5b4fc;font-size:.78rem;'>{qtype}</td>"
            f"<td style='padding:6px 10px;color:#86efac;font-size:.78rem;'>{len(results)}</td>"
            f"<td style='padding:6px 10px;color:#fbbf24;font-size:.78rem;'>{top['score']:.3f if top else '—'}</td>"
            f"<td style='padding:6px 10px;font-size:.78rem;'>{'✅' if cancer_ok is True else ('—' if cancer_ok == '—' else '⚠️')}</td>"
            f"<td style='padding:6px 10px;font-size:.78rem;'>{'✅' if qtype_ok else '⚠️'}</td>"
            f"<td style='padding:6px 10px;color:#67e8f9;font-size:.78rem;'>{elapsed:.3f}s</td>"
            f"</tr>")

    return f"""
    <div style='background:#1e2030;border:1px solid #2d3148;border-radius:10px;padding:16px;'>
      <div style='color:#a5b4fc;font-weight:600;margin-bottom:12px;font-size:.95rem;'>
        🏆 Résultats Benchmark — {len(BENCH_QUESTIONS)} questions
        <span style='color:#64748b;font-size:.8rem;margin-left:12px;'>
          Temps total: {total_time:.2f}s | Moy: {total_time/len(BENCH_QUESTIONS):.3f}s/q
        </span>
      </div>
      <div style='overflow-x:auto;'>
        <table style='width:100%;border-collapse:collapse;font-size:.82rem;'>
          <tr style='border-bottom:1px solid #2d3148;'>
            <th style='color:#475569;text-align:left;padding:6px 10px;'>Question</th>
            <th style='color:#475569;text-align:left;padding:6px 10px;'>Q-Type</th>
            <th style='color:#475569;text-align:left;padding:6px 10px;'>Résultats</th>
            <th style='color:#475569;text-align:left;padding:6px 10px;'>Top score</th>
            <th style='color:#475569;text-align:left;padding:6px 10px;'>Cancer ✓</th>
            <th style='color:#475569;text-align:left;padding:6px 10px;'>Type ✓</th>
            <th style='color:#475569;text-align:left;padding:6px 10px;'>Latence</th>
          </tr>
          {"".join(rows)}
        </table>
      </div>
    </div>"""

def run_custom_bench(question, cancer_sel, qtype_sel, k_val):
    if retriever is None:
        return "<p style='color:#ef4444;'>❌ Initialisez d'abord le système.</p>"
    if not question.strip():
        return "<p style='color:#fbbf24;'>⚠️ Entrez une question.</p>"

    cancer  = cancer_sel  if cancer_sel  not in ("", "Auto") else None
    qtype   = qtype_sel   if qtype_sel   not in ("", "Auto") else None
    t0 = time.time()
    results = retriever.search(question, k=int(k_val), cancer_filter=cancer, qtype=qtype)
    elapsed = round(time.time() - t0, 4)

    auto_c = retriever.detect_cancer(question)
    auto_q = retriever.detect_qtype(question)

    cards = ""
    for i, r in enumerate(results, 1):
        label = DOC_TYPE_LABELS.get(r["type"], r["type"])
        bar_w = min(int(r["score"] * 180), 100)
        source_text = r['text'][:400]
        if translator.loaded:
            source_text = translator.translate(source_text)
        cards += f"""
        <div style='background:#111827;border-radius:8px;padding:12px;margin-bottom:8px;'>
          <div style='display:flex;justify-content:space-between;'>
            <span style='color:#a5b4fc;font-weight:600;font-size:.85rem;'>{i}. {label}</span>
            <span style='color:#64748b;font-size:.75rem;'>Score {r['score']:.4f}</span>
          </div>
          <div style='background:#1e2030;height:3px;border-radius:2px;margin:6px 0;'>
            <div style='background:linear-gradient(90deg,#6366f1,#8b5cf6);
                        width:{bar_w}%;height:3px;'></div>
          </div>
          <p style='color:#94a3b8;font-size:.8rem;margin:4px 0;'>{source_text}…</p>
          <div style='display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;'>
            <span style='background:#312e81;color:#c7d2fe;font-size:.7rem;padding:1px 8px;border-radius:10px;'>🎯 {r['cancer']}</span>
            <span style='background:#1e3a5f;color:#93c5fd;font-size:.7rem;padding:1px 8px;border-radius:10px;'>BM25 {r['bm25']:.3f}</span>
            <span style='background:#1a3040;color:#67e8f9;font-size:.7rem;padding:1px 8px;border-radius:10px;'>Sém. {r['sem']:.3f}</span>
          </div>
        </div>"""

    return f"""
    <div style='background:#1e2030;border:1px solid #2d3148;border-radius:10px;padding:16px;'>
      <div style='color:#a5b4fc;font-weight:600;margin-bottom:4px;'>
        🔍 {len(results)} résultats en {elapsed}s
      </div>
      <div style='color:#64748b;font-size:.8rem;margin-bottom:12px;'>
        Cancer auto-détecté : <strong style='color:#c7d2fe;'>{auto_c or '—'}</strong> |
        Q-Type auto-détecté : <strong style='color:#86efac;'>{auto_q}</strong>
      </div>
      {cards}
    </div>"""

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
DARK_CSS = """
body, .gradio-container {
    background: #0f1117 !important;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}
.gr-button-primary {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important; color: #fff !important;
    border-radius: 8px !important; font-weight: 600 !important;
}
.gr-button-secondary {
    background: #1e2030 !important; border: 1px solid #2d3148 !important;
    color: #a5b4fc !important; border-radius: 8px !important;
}
.gr-textbox textarea, .gr-textbox input {
    background: #1e2030 !important; border: 1px solid #2d3148 !important;
    color: #e2e8f0 !important; border-radius: 8px !important;
}
.tab-nav button { color: #94a3b8 !important; background: transparent !important;
    border-bottom: 2px solid transparent !important; }
.tab-nav button.selected { color: #a5b4fc !important;
    border-bottom: 2px solid #6366f1 !important; }
.gr-accordion { background: #1e2030 !important;
    border: 1px solid #2d3148 !important; border-radius: 8px !important; }
.gr-slider input[type=range] { accent-color: #6366f1; }
footer { display: none !important; }
"""

# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE GRADIO
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=DARK_CSS, title="RAG Oncologie v2", theme=gr.themes.Base()) as demo:

    gr.HTML("""
    <div style="text-align:center;padding:24px 0 10px;">
      <h1 style="font-size:2rem;font-weight:700;
                 background:linear-gradient(135deg,#818cf8,#c084fc);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px;">
        🏥 RAG Oncologie — v2
      </h1>
      <p style="color:#64748b;font-size:.9rem;">
        SmartRetriever · Qwen2-1.5B · Filtres avancés · Benchmark · Export · Traduction EN→FR
      </p>
    </div>""")

    with gr.Tabs():

        # ── 1. ASSISTANT ──────────────────────────────────────────────────────
        with gr.Tab("💬 Assistant"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversation", height=460,
                        placeholder="Posez une question pour commencer…")
                    with gr.Row():
                        question_box = gr.Textbox(
                            placeholder="Votre question médicale…",
                            label="", scale=5, container=False)
                        send_btn  = gr.Button("Envoyer ▶", variant="primary", scale=1)
                        clear_btn = gr.Button("🗑", scale=0, min_width=48)
                    gr.Examples(EXAMPLES, inputs=question_box, label="Questions fréquentes")

                with gr.Column(scale=2):
                    with gr.Accordion("⚙️ Paramètres de recherche", open=True):
                        n_results = gr.Slider(2, 12, value=5, step=1,
                                              label="Nombre de sources")
                        cancer_override = gr.Dropdown(
                            choices=["Auto"] + ALL_CANCERS,
                            value="Auto", label="🎯 Forcer cancer")
                        doctype_override = gr.Dropdown(
                            choices=["Tous"] + ALL_DOC_TYPES,
                            value="Tous", label="📁 Forcer type de doc")
                    with gr.Accordion("🤖 LLM Qwen2", open=False):
                        use_llm   = gr.Checkbox(label="Activer Qwen2-1.5B", value=False)
                        llm_tokens = gr.Slider(100, 600, value=350, step=50,
                                               label="Max tokens générés")
                        gr.HTML("<p style='color:#64748b;font-size:.78rem;margin:4px 0;'>"
                                "Chargez Qwen2 dans l'onglet Initialisation d'abord.</p>")
                    with gr.Accordion("🌐 Traduction", open=False):
                        trans_status = gr.HTML("Traducteur non chargé")
                        trans_btn = gr.Button("📥 Charger traducteur EN→FR", variant="secondary")
                    meta_box = gr.Textbox(label="Métadonnées", interactive=False, lines=1)
                    gr.HTML("<div style='color:#64748b;font-size:.8rem;margin:6px 0;'>"
                            "Sources récupérées</div>")
                    sources_out = gr.HTML(
                        "<p style='color:#475569;font-size:.85rem;padding:10px;'>"
                        "Les sources apparaîtront ici.</p>")

            # Export
            with gr.Accordion("💾 Sauvegarder / Exporter la conversation", open=False):
                with gr.Row():
                    session_name = gr.Textbox(label="Nom de la session", scale=3,
                                              placeholder="ex: Cas prostate avancé")
                    save_btn  = gr.Button("💾 Sauvegarder", scale=1, variant="secondary")
                    save_msg  = gr.Textbox(label="", scale=2, interactive=False)
                with gr.Row():
                    export_fmt  = gr.Radio(["Markdown", "Texte"], value="Markdown",
                                           label="Format d'export")
                    export_btn  = gr.Button("📥 Exporter", variant="secondary")
                    export_file = gr.File(label="Fichier exporté", interactive=False)

            def update_trans_status():
                if translator.loaded:
                    return "<span style='color:#86efac;'>✅ Traducteur EN→FR chargé et actif</span>"
                else:
                    return "<span style='color:#fbbf24;'>⚠️ Traducteur non chargé (cliquez pour charger)</span>"
            
            trans_btn.click(load_translator, [], trans_status)
            
            def _send(q, n, u, tok, co, dto, h):
                yield from query_rag(q, n, u, tok, co, dto, h)

            send_btn.click(_send,
                [question_box, n_results, use_llm, llm_tokens,
                 cancer_override, doctype_override, chatbot],
                [chatbot, sources_out, meta_box])
            question_box.submit(_send,
                [question_box, n_results, use_llm, llm_tokens,
                 cancer_override, doctype_override, chatbot],
                [chatbot, sources_out, meta_box])
            clear_btn.click(clear_chat, [], [chatbot, sources_out, meta_box])
            save_btn.click(save_session, [chatbot, session_name], save_msg)
            export_btn.click(export_conversation, [chatbot, export_fmt], export_file)

        # ── 2. ENRICHIR ──────────────────────────────────────────────────────
        with gr.Tab("📥 Enrichir la base"):
            gr.HTML("""
            <div style='background:#1e2030;border:1px solid #2d3148;border-radius:10px;
                        padding:16px;margin-bottom:16px;'>
              <h3 style='color:#a5b4fc;margin-top:0;font-size:1rem;'>
                Ajoutez vos propres données (persistées dans custom_entries.json)
              </h3>
            </div>""")
            enrich_msg = gr.Textbox(label="Résultat", interactive=False, lines=2)
            stats_html = gr.HTML(get_base_stats())

            with gr.Tabs():
                with gr.Tab("📂 Fichier JSON"):
                    gr.Markdown("""**Format** : liste de documents avec `cancer_name`, `document_type`, contenu.""")
                    json_file = gr.File(label="Fichier JSON", file_types=[".json"])
                    gr.Button("⬆️ Importer", variant="primary").click(
                        enrich_from_json, [json_file], [enrich_msg, stats_html])

                with gr.Tab("✏️ Saisie manuelle"):
                    with gr.Row():
                        m_cancer  = gr.Dropdown(ALL_CANCERS, label="Cancer *",
                                                allow_custom_value=True)
                        m_doctype = gr.Dropdown(ALL_DOC_TYPES, value="cancer_knowledge",
                                                label="Type *")
                    m_title   = gr.Textbox(label="Titre")
                    m_content = gr.Textbox(label="Contenu *", lines=5)
                    gr.Button("➕ Ajouter", variant="primary").click(
                        enrich_manual, [m_cancer, m_doctype, m_title, m_content],
                        [enrich_msg, stats_html])

                with gr.Tab("📋 Texte libre"):
                    with gr.Row():
                        t_cancer  = gr.Dropdown(ALL_CANCERS, label="Cancer *",
                                                allow_custom_value=True)
                        t_doctype = gr.Dropdown(ALL_DOC_TYPES, value="medical_literature",
                                                label="Type *")
                    t_text = gr.Textbox(label="Texte (séparez les sections par ligne vide) *",
                                        lines=8)
                    gr.Button("📥 Indexer", variant="primary").click(
                        enrich_text_block, [t_cancer, t_doctype, t_text],
                        [enrich_msg, stats_html])

            with gr.Row():
                gr.Button("🔄 Actualiser stats", scale=2).click(
                    refresh_stats, [], stats_html)
                gr.Button("🗑 Supprimer entrées perso", variant="secondary", scale=1).click(
                    delete_custom, [], [enrich_msg, stats_html])

        # ── 3. ÉVALUATION ─────────────────────────────────────────────────────
        with gr.Tab("📊 Évaluation"):
            gr.HTML("""
            <div style='background:#1e2030;border:1px solid #2d3148;border-radius:10px;
                        padding:14px;margin-bottom:14px;'>
              <h3 style='color:#a5b4fc;margin-top:0;font-size:1rem;'>
                🏆 Benchmark SmartRetriever — Teste la qualité du retrieval
              </h3>
              <p style='color:#64748b;font-size:.85rem;margin:0;'>
                8 questions médicales couvrant tous les types de documents.
                Vérifie : nombre de résultats, scores, correspondance cancer/qtype.
              </p>
            </div>""")

            bench_btn    = gr.Button("🚀 Lancer le benchmark automatique", variant="primary")
            bench_result = gr.HTML("<p style='color:#475569;padding:10px;'>"
                                   "Cliquez pour lancer l'évaluation.</p>")
            bench_btn.click(run_benchmark, [], bench_result)

            gr.HTML("<hr style='border-color:#2d3148;margin:20px 0;'>")
            gr.HTML("<div style='color:#a5b4fc;font-weight:600;margin-bottom:10px;'>"
                    "🔬 Test personnalisé</div>")
            with gr.Row():
                eval_q  = gr.Textbox(label="Question *", scale=3,
                                     placeholder="Entrez une question médicale…")
                eval_c  = gr.Dropdown(["Auto"] + ALL_CANCERS, value="Auto",
                                      label="Cancer", scale=1)
                eval_qt = gr.Dropdown(
                    ["Auto"] + list(QTYPE_TO_DOCTYPE.keys()),
                    value="Auto", label="Q-Type", scale=1)
                eval_k  = gr.Slider(1, 10, value=5, step=1, label="k", scale=0)
            eval_btn    = gr.Button("🔍 Tester", variant="secondary")
            eval_result = gr.HTML()
            eval_btn.click(run_custom_bench,
                [eval_q, eval_c, eval_qt, eval_k], eval_result)

        # ── 4. SESSIONS ───────────────────────────────────────────────────────
        with gr.Tab("📚 Historique"):
            gr.HTML("""
            <div style='background:#1e2030;border:1px solid #2d3148;border-radius:10px;
                        padding:14px;margin-bottom:14px;'>
              <p style='color:#64748b;font-size:.85rem;margin:0;'>
                Les sessions sont persistées localement dans
                <code>sessions_history.json</code>.
              </p>
            </div>""")
            sess_html = gr.HTML(list_sessions())
            gr.Button("🔄 Actualiser", variant="secondary").click(
                list_sessions, [], sess_html)

        # ── 5. INITIALISATION ──────────────────────────────────────────────────
        with gr.Tab("🚀 Initialisation"):
            gr.HTML("""
            <div style='background:#1e2030;border:1px solid #2d3148;border-radius:10px;
                        padding:20px;margin-bottom:16px;'>
              <h3 style='color:#a5b4fc;margin-top:0;'>Démarrage du système</h3>
              <ol style='color:#94a3b8;line-height:2;font-size:.9rem;'>
                <li>Placez <code>merged_cancers_vfrancais.json</code> dans le même dossier.</li>
                <li>Cliquez sur <strong>Initialiser le RAG</strong> (embeddings mis en cache).</li>
                <li>Optionnel : chargez <strong>Qwen2-1.5B</strong> pour la génération LLM.</li>
                <li>Optionnel : chargez le <strong>traducteur EN→FR</strong> pour traduire les sources anglaises.</li>
              </ol>
            </div>""")

            with gr.Row():
                init_btn  = gr.Button("🚀 Initialiser le RAG", variant="primary", scale=2)
                qwen2_btn = gr.Button("🤖 Charger Qwen2-1.5B", variant="secondary", scale=1)
                trans_init_btn = gr.Button("🌐 Charger traducteur EN→FR", variant="secondary", scale=1)

            status_box  = gr.Textbox(value=init_status, label="Statut RAG",
                                     interactive=False, lines=2)
            qwen2_status = gr.Textbox(label="Statut Qwen2", interactive=False, lines=1)
            trans_status_box = gr.Textbox(label="Statut Traducteur", interactive=False, lines=1)
            init_stats  = gr.HTML()

            init_btn.click(load_system,  [], [status_box, init_stats])
            qwen2_btn.click(load_qwen2, [], qwen2_status)
            trans_init_btn.click(load_translator, [], trans_status_box)

        # ── 6. À PROPOS ────────────────────────────────────────────────────────
        with gr.Tab("ℹ️ À propos"):
            gr.Markdown("""
## Architecture RAG Oncologie v2

| Composant | Détail |
|---|---|
| **Chunking** | 12 types de documents, clés JSON réelles |
| **Embeddings** | `intfloat/multilingual-e5-small` (prefix query/passage) |
| **BM25** | `rank-bm25` — recherche lexicale |
| **SmartRetriever** | BM25×0.35 + Sém×0.50 + Cohérence×0.15 + boost adaptatif par qtype |
| **Détection** | Regex sémantiques (cancer + type de question) |
| **LLM** | Qwen2-1.5B-Instruct (local, CPU) |
| **Traduction** | Helsinki-NLP/opus-mt-en-fr (traduction automatique EN→FR) |
| **Enrichissement** | JSON · Formulaire · Texte libre (persisté) |
| **Évaluation** | Benchmark 8 questions + test personnalisé |
| **Export** | Markdown / Texte plain |

### Cancers couverts
Prostate · Sein · Colorectal · Poumon (SCLC/CPNPC) · Thyroïde · Ovaire ·
Mélanome · Gliomes HGG · Gastrique · Rein · Vessie · Col de l'utérus

*100 % open-source — fonctionne entièrement en local.*
            """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)