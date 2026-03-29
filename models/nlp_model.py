"""
Clinical Note NLP Model
Rule-based + pattern-matching entity extraction.
Zero external model downloads — works entirely from curated medical lexicons.
"""

import re
from typing import Dict, List

# ── Medical Lexicons ────────────────────────────────────────────────────────────

SYMPTOMS = {
    "fever", "chills", "fatigue", "weakness", "malaise", "pain", "headache",
    "nausea", "vomiting", "diarrhea", "constipation", "cough", "dyspnea",
    "shortness of breath", "chest pain", "chest tightness", "palpitations",
    "dizziness", "syncope", "confusion", "altered mental status", "seizure",
    "edema", "swelling", "rash", "itching", "pruritus", "jaundice",
    "abdominal pain", "back pain", "joint pain", "muscle pain", "myalgia",
    "arthralgia", "sore throat", "rhinorrhea", "nasal congestion", "wheezing",
    "hemoptysis", "hematuria", "melena", "hematochezia", "weight loss",
    "anorexia", "night sweats", "insomnia", "anxiety", "depression",
    "polyuria", "polydipsia", "polyphagia", "diaphoresis", "pallor",
    "cyanosis", "hypoxia", "tachycardia", "bradycardia", "hypotension",
    "hypertension", "tachypnea", "bradypnea", "oliguria", "anuria",
}

CONDITIONS = {
    "diabetes", "diabetes mellitus", "type 2 diabetes", "type 1 diabetes",
    "hypertension", "heart failure", "congestive heart failure", "chf",
    "myocardial infarction", "mi", "coronary artery disease", "cad",
    "atrial fibrillation", "afib", "stroke", "tia", "copd",
    "chronic obstructive pulmonary disease", "asthma", "pneumonia",
    "sepsis", "septic shock", "acute kidney injury", "aki",
    "chronic kidney disease", "ckd", "cirrhosis", "liver failure",
    "hepatitis", "pancreatitis", "appendicitis", "cholecystitis",
    "pulmonary embolism", "pe", "deep vein thrombosis", "dvt",
    "anemia", "thrombocytopenia", "leukocytosis", "leukopenia",
    "hyperglycemia", "hypoglycemia", "hyperkalemia", "hyponatremia",
    "hypothyroidism", "hyperthyroidism", "cancer", "malignancy",
    "metastasis", "lymphoma", "leukemia", "tumor", "fracture",
    "infection", "cellulitis", "urinary tract infection", "uti",
    "meningitis", "encephalitis", "dementia", "alzheimer",
    "parkinson", "multiple sclerosis", "epilepsy", "obesity",
    "osteoporosis", "rheumatoid arthritis", "lupus", "crohn",
    "ulcerative colitis", "gerd", "peptic ulcer",
}

MEDICATIONS = {
    # Antibiotics
    "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline",
    "cephalexin", "clindamycin", "vancomycin", "metronidazole",
    "piperacillin", "tazobactam", "meropenem", "levofloxacin",
    # Cardiovascular
    "metoprolol", "atenolol", "lisinopril", "enalapril", "amlodipine",
    "losartan", "valsartan", "furosemide", "spironolactone", "digoxin",
    "warfarin", "heparin", "aspirin", "clopidogrel", "atorvastatin",
    "simvastatin", "nitroglycerin", "amiodarone", "carvedilol",
    # Diabetes
    "metformin", "insulin", "glipizide", "glimepiride", "sitagliptin",
    "empagliflozin", "liraglutide", "dapagliflozin",
    # Pain / Analgesia
    "ibuprofen", "acetaminophen", "morphine", "hydromorphone", "oxycodone",
    "tramadol", "ketorolac", "naproxen", "gabapentin", "pregabalin",
    # Other common
    "omeprazole", "pantoprazole", "ondansetron", "prednisone",
    "methylprednisolone", "dexamethasone", "albuterol", "ipratropium",
    "tiotropium", "fluticasone", "montelukast", "lorazepam", "diazepam",
    "haloperidol", "quetiapine", "sertraline", "escitalopram", "fluoxetine",
    "levothyroxine", "hydrochlorothiazide", "potassium chloride",
}

# Lab value patterns
LAB_PATTERNS = [
    # e.g. "WBC 12.5", "glucose of 245 mg/dL"
    r'\b(WBC|RBC|Hgb|Hematocrit|Hct|Platelets?|Plt|Glucose|HbA1c|A1C|Creatinine|BUN|'
    r'eGFR|Sodium|Na|Potassium|K|Chloride|Cl|Bicarbonate|CO2|Calcium|Ca|Magnesium|Mg|'
    r'Phosphorus|ALT|AST|ALP|Bilirubin|Albumin|TSH|T4|T3|INR|PT|PTT|aPTT|'
    r'Troponin|CK|CK-MB|BNP|NT-proBNP|Lactate|Procalcitonin|CRP|ESR|'
    r'LDH|Ferritin|Iron|Transferrin|Cortisol|HCG|PSA|CEA|CA-125|'
    r'SpO2|O2 sat|Oxygen saturation)\s*[:\-=]?\s*(\d+\.?\d*)\s*'
    r'(mg/dL|g/dL|mmol/L|mEq/L|IU/L|U/L|ng/mL|pg/mL|%|cells/μL|K/μL|mmHg)?',

    # e.g. "BP 140/90", "HR 88"
    r'\b(BP|Blood pressure|HR|Heart rate|RR|Respiratory rate|Temp|Temperature|'
    r'SpO2|O2)\s*[:\-=]?\s*(\d+\.?\d*(?:/\d+\.?\d*)?)\s*(mmHg|bpm|°[CF]|%)?',
]


def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())


def _extract_entities(text: str, lexicon: set, min_len: int = 3) -> List[str]:
    """Find lexicon terms in text using word-boundary matching."""
    norm = _normalize(text)
    found = []
    for term in sorted(lexicon, key=len, reverse=True):  # longest-first
        if len(term) < min_len:
            continue
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, norm):
            found.append(term.title())
    return sorted(set(found))


def _extract_lab_values(text: str) -> List[str]:
    """Extract numeric lab results."""
    found = []
    for pattern in LAB_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            found.append(m.group(0).strip())
    return list(dict.fromkeys(found))  # deduplicate preserving order


def extract_entities(note: str) -> Dict[str, List[str]]:
    """
    Extract clinical entities from a free-text note.

    Returns:
        {
            "symptoms":    [...],
            "conditions":  [...],
            "medications": [...],
            "lab_values":  [...],
        }
    """
    if not note or not note.strip():
        return {"symptoms": [], "conditions": [], "medications": [], "lab_values": []}

    return {
        "symptoms":    _extract_entities(note, SYMPTOMS),
        "conditions":  _extract_entities(note, CONDITIONS),
        "medications": _extract_entities(note, MEDICATIONS),
        "lab_values":  _extract_lab_values(note),
    }


def get_entity_summary(entities: Dict[str, List[str]]) -> str:
    total = sum(len(v) for v in entities.values())
    return f"{total} entities extracted across {sum(1 for v in entities.values() if v)} categories"


# ── Sample notes ───────────────────────────────────────────────────────────────
SAMPLE_NOTES = {
    "Diabetic with HTN": (
        "65-year-old male with history of type 2 diabetes mellitus and hypertension "
        "presents with fatigue, polyuria, and polydipsia for 3 days. Glucose 320 mg/dL, "
        "HbA1c 11.2%. BP 158/96 mmHg, HR 88 bpm, SpO2 97%. Currently on metformin 1000 mg "
        "twice daily and lisinopril 10 mg daily. Started insulin glargine 10 units at bedtime. "
        "Patient reports nausea and dizziness."
    ),
    "Chest Pain / ACS": (
        "55-year-old female presenting with acute chest pain radiating to left arm, "
        "diaphoresis and shortness of breath. HR 110 bpm, BP 90/60 mmHg. "
        "Troponin 2.4 ng/mL elevated. EKG shows ST elevation. Diagnosis: myocardial infarction. "
        "Administered aspirin 325 mg, clopidogrel 600 mg, heparin drip started. "
        "Coronary artery disease noted on history."
    ),
    "Sepsis / Infection": (
        "72-year-old with fever 39.2°C, chills, confusion, and hypotension BP 82/50 mmHg. "
        "WBC 22.5 K/μL, Lactate 4.1 mmol/L, Creatinine 2.8 mg/dL. "
        "Diagnosis: septic shock secondary to urinary tract infection. "
        "Started vancomycin and piperacillin-tazobactam. "
        "Procalcitonin elevated at 8.9 ng/mL. "
        "Acute kidney injury noted. Fluid resuscitation ongoing."
    ),
}