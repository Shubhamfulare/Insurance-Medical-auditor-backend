# rag_store.py
from sentence_transformers import SentenceTransformer
import faiss
import os
import json ,re

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_rules():
    with open("knowledge_base/rules.txt", "r") as f:
        return f.read().split("\n\n")
    

def load_rules_file():
    # file_path = os.path.join(os.path.dirname(__file__), "test.txt")

    with open("knowledge_base/rules.txt", "r", encoding="utf-8") as f:
        rules = json.load(f)

    return rules

rules = load_rules()
embeddings = model.encode(rules)

index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(embeddings)


def retrieve_rules(query, top_k=3):
    q_emb = model.encode([query])
    distances, indices = index.search(q_emb, top_k)
    return [rules[i] for i in indices[0]]

def extract_evidence(text: str):
    text_lower = text.lower()
    evidence = []

    # ---- Blood Pressure (HTN) ----
    if (
        "blood pressure" in text_lower
        or re.search(r"\bbp\b", text_lower)
        or re.search(r"\b\d{2,3}/\d{2,3}\s*mmhg\b", text_lower)
    ):
        evidence.append("blood_pressure")

    # ---- Fever ----
    if "fever" in text_lower or "temp" in text_lower:
        evidence.append("fever")

    # ---- Infection diagnosis ----
    if any(x in text_lower for x in ["infection", "pneumonia", "sepsis"]):
        evidence.append("infection_diagnosis")

    # ---- X-ray ----
    if any(x in text_lower for x in ["x-ray", "xray", "chest xray"]):
        evidence.append("chest_xray")
        evidence.append("x_ray")

    return evidence



def pre_validate(relevant_rules, extracted_evidence, diagnosis):
    validation_results = []
    diagnosis_lower = (diagnosis or "").lower()

    for rule in relevant_rules:

        if isinstance(rule, str):
            rule = json.loads(rule)

        rule_id = rule.get("rule_id")
        required = rule.get("required_evidence", [])

        # Apply rule only if diagnosis matches
        if rule_id == "HTN_001":
            if "htn" not in diagnosis_lower and "hypertension" not in diagnosis_lower:
                continue

        if rule_id == "ANTIBIOTIC_001":
            if not any(x in diagnosis_lower for x in ["infection", "pneumonia", "sepsis"]):
                continue

        if rule_id == "DENTAL_001":
            if not any(x in diagnosis_lower for x in ["dental", "tooth", "root canal"]):
                continue

        if rule_id == "PNEUMONIA_001":
            if "pneumonia" not in diagnosis_lower:
                continue

        missing_evidence = [
            req for req in required
            if req not in extracted_evidence
        ]

        validation_results.append({
            "rule_id": rule_id,
            "condition": rule.get("condition"),
            "missing_evidence": missing_evidence,
            "is_valid": len(missing_evidence) == 0
        })

    return validation_results