## 🛡️ Clinical Safety Classification & Flag Assignment System
## 👨‍⚕️ by Ridwan Oladipo, MD — Medical AI Specialist

# =============================================================================
# Overview:
#   Drug Interaction Severity Assessment & Polypharmacy Analysis
#   Powered by GPT-5 + Clinical Knowledge Graphs
# =============================================================================

import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

project_root = Path.cwd().parent
sys.path.append(str(project_root / "src"))

load_dotenv(project_root / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from rag_pipeline import (
    load_data,
    build_or_load_faiss_index,
    check_polypharmacy as check_polypharmacy_orig,
    check_polypharmacy_light as check_polypharmacy_light_orig,
)

kb_df, lookups = load_data()
index, embeddings = build_or_load_faiss_index(kb_df, client)


def check_polypharmacy(drug_query: str):
    text = drug_query.replace("interaction", "").replace("and", ",")
    drugs = [d.strip() for d in text.split(",") if d.strip()]
    return check_polypharmacy_orig(drugs, kb_df, lookups, index, embeddings, client)


def check_polypharmacy_light(drug_query: str):
    text = drug_query.replace("interaction", "").replace("and", ",")
    drugs = [d.strip() for d in text.split(",") if d.strip()]
    return check_polypharmacy_light_orig(drugs, kb_df, lookups)


# =============================================================================
# CORE CLASSIFICATION FUNCTIONS
# =============================================================================

def classify_severity_gpt5(interaction_text):
    """
    Classify drug interaction severity using GPT-5.

    Args:
        interaction_text: Interaction description from knowledge base

    Returns:
        dict: {'severity': '🟥'|'🟨'|'🟩', 'explanation': str}
    """
    prompt = f"""You are a medical doctor and clinical pharmacist reviewing a drug-drug interaction.

Interaction evidence:
{interaction_text}

Task:
Determine the CLINICAL SEVERITY of this interaction even if words such as
"contraindicated", "major", "moderate", or "minor" do NOT appear.
Use both the provided evidence and your own pharmacologic knowledge.

Classification rules:
- 🟥 (Contraindicated): Life-threatening or severe interaction - avoid combination entirely.
- 🟨 (Caution): Clinically significant or moderate risk - requires monitoring or dose adjustment.
- 🟩 (No Interaction): No meaningful pharmacologic or clinical interaction expected.

Return ONLY valid JSON - no markdown, no commentary, no code fences.
Output must start with {{ and end with }}.

Expected format:
{{"severity": "🟥" or "🟨" or "🟩", "explanation": "brief reasoning"}}

Examples:
- "may increase bleeding risk" → 🟨
- "contraindicated with..." → 🟥
- "no interaction known" or "minimal clinical effect" → 🟩
"""
    try:
        response = client.responses.create(
            model="gpt-5",
            input=prompt,
            text={"format": {"type": "text"}}
        )

        result_text = (response.output_text or "").strip()

        if not result_text or not result_text.startswith("{") or not result_text.endswith("}"):
            raise ValueError("Invalid JSON format from GPT-5")

        return json.loads(result_text)

    except (json.JSONDecodeError, Exception) as e:
        raise


def synthesize_tier2_evidence(hits, drug1, drug2):
    """
    Synthesize clinical guidance from semantically similar interactions.

    Args:
        hits: List of similar drug interactions from FAISS
        drug1, drug2: Query drug names

    Returns:
        str: Clinical synthesis note
    """
    evidence_text = "\n\n".join([
        f"Similar interaction {i + 1} (confidence: {hit['retrieval_score']:.2f}):\n"
        f"{hit['drug1']} + {hit['drug2']}: {hit['evidence']}"
        for i, hit in enumerate(hits[:3])
    ])

    prompt = f"""You are a medical doctor and clinical pharmacist analyzing SIMILAR drug interactions.

Original query: {drug1} and {drug2}
Evidence from SIMILAR drug combinations:
{evidence_text}

Task:
Write a cautious 2-3 sentence clinical note summarizing potential risks
based on pharmacologic or class similarities.
Use both the provided evidence and your own pharmacologic knowledge to ensure
the synthesis is clinically meaningful.

Rules:
- Use phrases like "may exhibit similar interactions", "class-related concerns", 
  "pharmacologically related compounds suggest...".
- Be explicit that this is inferred, not direct evidence.
- Recommend monitoring any adverse effects mentioned.
- Return ONLY valid JSON: {{"synthesis": "clinical note"}}.
"""

    try:
        response = client.responses.create(
            model="gpt-5",
            input=prompt,
            text={"format": {"type": "text"}}
        )

        result_text = (response.output_text or "").strip()

        if not result_text or not result_text.startswith("{") or not result_text.endswith("}"):
            raise ValueError("Invalid JSON format from GPT-5")

        result = json.loads(result_text)
        return result.get("synthesis", "Unable to synthesize evidence.")

    except (json.JSONDecodeError, Exception):
        raise


# =============================================================================
# COMPREHENSIVE POLYPHARMACY ANALYZER
# =============================================================================

def analyze_polypharmacy_comprehensive(drug_query):
    """
    Perform comprehensive polypharmacy analysis with single LLM call.

    Args:
        drug_query: String like "Warfarin, Aspirin and Ibuprofen interaction"

    Returns:
        dict: Complete analysis with per-pair flags and overall assessment
    """
    rag_output = check_polypharmacy_light(drug_query)

    if 'error' in rag_output:
        return {'error': rag_output['error']}

    drugs = rag_output['drugs']
    num_pairs = rag_output['num_pairs']
    results = rag_output['results']

    formatted_pairs = []
    for i, pair_result in enumerate(results, 1):
        tier = pair_result['tier']
        query = pair_result['query']
        hits = pair_result.get('hits', [])

        if tier == 1:
            evidence = hits[0]['evidence']
            formatted_pairs.append(
                f"Pair {i}: {query}\n"
                f"   Tier: 1 (Direct KB match - highest confidence)\n"
                f"   Evidence: {evidence}\n"
                f"   Confidence: 1.0"
            )
        elif tier == 2:
            top_hits_text = "\n   ".join([
                f"- {hit['drug1']} + {hit['drug2']}: {hit['evidence']} (score: {hit['retrieval_score']:.2f})"
                for hit in hits[:3]
            ])
            formatted_pairs.append(
                f"Pair {i}: {query}\n"
                f"   Tier: 2 (Semantic/similar matches - inferred risk)\n"
                f"   Similar evidence:\n   {top_hits_text}\n"
                f"   Top confidence: {hits[0]['retrieval_score']:.2f}"
            )
        else:
            message = pair_result.get('message', 'No documented interaction')
            formatted_pairs.append(
                f"Pair {i}: {query}\n"
                f"   Tier: 3 (No evidence found)\n"
                f"   Status: {message}\n"
                f"   Confidence: 0.0"
            )

    pairs_text = "\n\n".join(formatted_pairs)

    prompt = f"""You are a medical doctor and clinical pharmacist performing COMPREHENSIVE polypharmacy analysis.

PATIENT SCENARIO:
Patient is taking {len(drugs)} medications: {', '.join(drugs)}
This generates {num_pairs} drug-drug interaction pair(s) to analyze.

INTERACTION ANALYSIS RESULTS:
{pairs_text}

TIER DEFINITIONS:
- Tier 1: Direct evidence from DrugBank knowledge base (definitive)
- Tier 2: Inferred from pharmacologically similar drugs (cautious interpretation needed)
- Tier 3: No documented interaction found (absence of evidence ≠ evidence of absence)

YOUR TASK:
1. For EACH pair, assign a severity flag:
   - 🟥 (High Risk): Life-threatening or severe interaction - avoid combination
   - 🟨 (Caution): Clinically significant risk - requires monitoring/dose adjustment
   - 🟩 (Safe): No meaningful interaction or minimal clinical significance

2. Consider CUMULATIVE effects across all pairs (e.g., multiple drugs affecting bleeding risk)

3. Provide OVERALL clinical recommendation considering:
   - Highest risk pair (determines overall flag)
   - Synergistic risks (do multiple pairs compound the same adverse effect?)
   - Clinical context (polypharmacy burden in elderly, renal/hepatic function)

4. Use both the provided evidence AND your own pharmacologic knowledge to ensure clinically meaningful analysis.

RESPONSE FORMAT (strict JSON - no markdown, no code fences):
{{
  "pair_analyses": [
    {{
      "pair_number": 1,
      "query": "Drug1 and Drug2 interaction",
      "flag": "🟥" or "🟨" or "🟩",
      "tier": 1 or 2 or 3,
      "confidence": 0.0 to 1.0,
      "reasoning": "Brief clinical explanation (1-2 sentences)",
      "monitoring": "What to monitor (if applicable)" or null
    }}
  ],
  "overall_assessment": {{
    "overall_flag": "🟥" or "🟨" or "🟩",
    "risk_level": "High Risk" or "Moderate Risk" or "Low Risk",
    "flag_counts": {{"🟥": X, "🟨": Y, "🟩": Z}},
    "cumulative_concerns": "Describe any synergistic/additive risks across pairs",
    "clinical_synthesis": "Comprehensive 3-4 sentence recommendation for this drug combination",
    "action_required": "Immediate action needed (if any)" or null
  }}
}}

CRITICAL RULES:
- Tier 1 evidence gets highest weight in classification
- Tier 2 requires cautious interpretation (default to 🟨 unless clearly benign)
- Tier 3 does NOT mean "safe" - it means "unknown" (usually → 🟩 with disclaimer)
- If 2+ pairs have bleeding/QT/serotonin/CNS depression concerns → flag cumulative risk
- Be clinically practical - not all interactions require intervention
"""

    try:
        response = client.responses.create(
            model="gpt-5",
            input=prompt,
            text={"format": {"type": "text"}}
        )

        result_text = (response.output_text or "").strip()

        if not result_text:
            raise ValueError("Empty response from GPT-5")

        if result_text.startswith("```"):
            lines = result_text.split("\n")
            json_lines = [l for l in lines if not l.startswith("```")]
            result_text = "\n".join(json_lines).strip()

        if not result_text.startswith("{") or not result_text.endswith("}"):
            raise ValueError(f"Invalid JSON format from GPT-5: {result_text[:100]}")

        analysis = json.loads(result_text)
        analysis['drugs'] = drugs
        analysis['num_pairs'] = num_pairs
        analysis['rag_results'] = results

        return analysis

    except json.JSONDecodeError as e:
        return {
            'error': f'JSON parsing error: {e}',
            'raw_response': result_text[:500] if 'result_text' in locals() else 'No response'
        }
    except Exception as e:
        return {
            'error': f'LLM analysis failed: {e}',
            'fallback': 'Use legacy pair-by-pair analysis'
        }


# =============================================================================
# REPORTING & VISUALIZATION
# =============================================================================

def format_polypharmacy_report(analysis):
    """Display formatted polypharmacy analysis report."""
    if 'error' in analysis:
        print(f"Error: {analysis['error']}")
        return

    print("=" * 70)
    print(f"POLYPHARMACY ANALYSIS: {', '.join(analysis['drugs'])}")
    print("=" * 70)

    overall = analysis['overall_assessment']
    print(f"\nOVERALL RISK: {overall['overall_flag']} {overall['risk_level']}")
    print(
        f"Flag Distribution: 🟥 {overall['flag_counts']['🟥']} | 🟨 {overall['flag_counts']['🟨']} | 🟩 {overall['flag_counts']['🟩']}")

    print(f"\nCLINICAL SYNTHESIS:")
    print(f"{overall['clinical_synthesis']}")

    if overall.get('cumulative_concerns'):
        print(f"\nCUMULATIVE CONCERNS:")
        print(f"{overall['cumulative_concerns']}")

    if overall.get('action_required'):
        print(f"\nACTION REQUIRED:")
        print(f"{overall['action_required']}")

    print(f"\n{'=' * 70}")
    print(f"PAIR-BY-PAIR ANALYSIS ({analysis['num_pairs']} pairs)")
    print(f"{'=' * 70}")

    for pair_analysis in analysis['pair_analyses']:
        print(f"\n{pair_analysis['flag']} Pair {pair_analysis['pair_number']}: {pair_analysis['query']}")
        print(f"Tier: {pair_analysis['tier']} | Confidence: {pair_analysis['confidence']}")
        print(f"Reasoning: {pair_analysis['reasoning']}")
        if pair_analysis.get('monitoring'):
            print(f"Monitor: {pair_analysis['monitoring']}")

    print("\n" + "=" * 70)


# =============================================================================
# PRECOMPUTED SAMPLE GENERATION
# =============================================================================

RESULTS_FILE = Path("../data/precomputed_samples.json")


def append_result(scenario_name, analysis):
    """Save verified result to precomputed samples database."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append({
        'scenario': scenario_name,
        'analysis': analysis
    })

    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved: {scenario_name} (Total: {len(all_results)} samples)")


# =============================================================================
# CLINICAL SCENARIO SAMPLES
# =============================================================================

scenarios = [
    ("Hypertensive patient on anticoagulation", "Warfarin, Lisinopril and Aspirin interaction"),
    ("Post-MI patient with pain", "Clopidogrel, Aspirin and Diclofenac interaction"),
    ("Type 2 Diabetic patient with hypertension", "Metformin, Lisinopril and Hydrochlorothiazide interaction"),
    ("Depression with chronic pain", "Sertraline, Tramadol and Ibuprofen interaction"),
    ("Rheumatoid arthritis with gastroprotection", "Methotrexate, Prednisone and Omeprazole interaction"),
    ("Atrial fibrillation with antiplatelet therapy", "Warfarin and Aspirin interaction"),
    ("Type 2 Diabetes with dyslipidemia", "Metformin and Atorvastatin interaction"),
    ("Breast cancer chemotherapy support", "Tamoxifen and Ondansetron interaction"),
    ("Elderly patient with polypharmacy", "Warfarin and Omeprazole interaction"),
    ("Cold & flu relief (OTC combo)", "Tylenol and Benadryl interaction"),
]

for scenario_name, drug_query in scenarios:
    print(f"\nProcessing: {scenario_name}")
    analysis = analyze_polypharmacy_comprehensive(drug_query)
    format_polypharmacy_report(analysis)
    append_result(scenario_name, analysis)

print("\nAll precomputed samples generated successfully.")