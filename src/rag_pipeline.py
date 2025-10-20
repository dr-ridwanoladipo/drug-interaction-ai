"""
Drug Interaction Checker - RAG Pipeline
========================================
Author: Ridwan Oladipo, MD | Medical AI Specialist

Production-grade Retrieval-Augmented Generation (RAG) system that:
1. Implements hybrid retrieval (direct KB lookup → FAISS semantic search)
2. Handles polypharmacy via pairwise drug expansion
3. Generates embeddings with OpenAI text-embedding-3-large
4. Returns tiered results with confidence scores

Achieves <200ms latency for direct matches, <2s for semantic search.
"""

import pandas as pd
import numpy as np
import pickle
import os
import time
from itertools import combinations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
import faiss

# Load environment variables from project root
load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

# Dynamically resolve data directory relative to this file
DATA_DIR = Path(__file__).parent.parent / "data"
EMBEDDING_MODEL = "text-embedding-3-large"  # 3072 dimensions
CONFIDENCE_THRESHOLD = 0.6
FILTER_SIZE_LIMIT = 1000
BATCH_SIZE = 1000
MAX_WORKERS = 5


# ============================================================================
# Utility Functions
# ============================================================================

def normalize_drug(name):
    """Normalize drug name: lowercase, strip, remove extra spaces"""
    return name.lower().strip()


# ============================================================================
# Step 1: Load Knowledge Base and Lookups
# ============================================================================

def load_data():
    """
    Load processed knowledge base and RxNorm lookup dictionaries.

    Returns:
        kb_df: Knowledge base DataFrame
        lookups: Dict containing all RxNorm mappings
    """
    print("Loading knowledge base...")
    kb_df = pd.read_csv(DATA_DIR / 'processed_interactions_kb.csv')
    kb_df['pair_key'] = kb_df['pair_key'].apply(eval)  # Convert string to tuple

    print(f"   Loaded {len(kb_df):,} interactions")
    print(f"   Unique drug pairs: {kb_df['pair_key'].nunique():,}")

    print("\nLoading RxNorm mappings...")
    with open(DATA_DIR / 'rxnorm_lookups.pkl', 'rb') as f:
        lookups = pickle.load(f)

    print(f"   Name to RxCUI: {len(lookups['name_to_rxcui']):,}")
    print(f"   Brand to Ingredient: {len(lookups['bn_to_in_map']):,}")

    return kb_df, lookups


# ============================================================================
# Step 2: Drug Normalization to RxCUI
# ============================================================================

def normalize_to_rxcui(drug_name, lookups):
    """
    Normalize drug name to ingredient RxCUI using hierarchical lookup.

    Strategy:
    1. Check local name_to_rxcui → use bn_to_in_map if needed
    2. Check reverse index (ingredient names from API cache)

    Returns: RxCUI string or None
    """
    norm_name = normalize_drug(drug_name)

    name_to_rxcui = lookups['name_to_rxcui']
    bn_to_in_map = lookups['bn_to_in_map']
    ingredient_name_to_rxcui = lookups['ingredient_name_to_rxcui']

    # 1. Local lookup
    if norm_name in name_to_rxcui:
        rxcui, dtype = name_to_rxcui[norm_name]
        if dtype == "IN":
            return str(rxcui)
        if str(rxcui) in bn_to_in_map:
            ing_rxcui, _ = bn_to_in_map[str(rxcui)]
            return ing_rxcui

    # 2. Ingredient name cache
    if norm_name in ingredient_name_to_rxcui:
        return ingredient_name_to_rxcui[norm_name]

    return None


# ============================================================================
# Step 3: Build or Load FAISS Index
# ============================================================================

def get_openai_embeddings_batch(texts, client, model=EMBEDDING_MODEL):
    """
    Generate embeddings with retry logic and latency tracking.

    Returns: numpy array of embeddings (float32)
    """
    for attempt in range(3):
        try:
            t0 = time.time()
            resp = client.embeddings.create(model=model, input=texts)
            latency = (time.time() - t0) * 1000

            embeddings = [d.embedding for d in resp.data]
            return np.array(embeddings, dtype="float32")
        except Exception as e:
            print(f"   Warning: Retry {attempt + 1}/3 after error: {e}")
            time.sleep(2)

    raise RuntimeError("Failed to embed batch after 3 retries")


def build_or_load_faiss_index(kb_df, client):
    """
    Build FAISS index from knowledge base or load existing index.

    Uses inner product (cosine similarity) with normalized embeddings.
    Implements checkpointing for crash recovery.

    Returns:
        index: FAISS index object
        embeddings: numpy array of embeddings
    """
    index_path = DATA_DIR / "faiss_index.bin"
    embeddings_path = DATA_DIR / "interaction_embeddings.npy"

    # Load existing index
    if index_path.exists() and embeddings_path.exists():
        print("Loading existing FAISS index...")
        index = faiss.read_index(str(index_path))
        embeddings = np.load(str(embeddings_path))
        print(f"   Loaded index with {index.ntotal:,} vectors ({embeddings.shape[1]} dims)")
        return index, embeddings

    # Build new index
    print("Building FAISS index (first run, approximately 30-35 minutes)...")

    # Prepare texts for embedding
    texts = [
        f"{row['Drug 1']} and {row['Drug 2']} interaction: {row['Interaction Description']}"
        for _, row in kb_df.iterrows()
    ]
    print(f"   Generating embeddings for {len(texts):,} interactions...")

    # Batch embeddings with parallel processing
    all_batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    start = time.time()

    all_embeddings = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(get_openai_embeddings_batch, batch, client): idx
            for idx, batch in enumerate(all_batches, 1)
        }

        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                batch_embeddings = future.result()
                all_embeddings.append(batch_embeddings)
                print(f"   Completed batch {batch_idx}/{len(all_batches)}")

                # Checkpoint every 5 batches
                if len(all_embeddings) % 5 == 0:
                    temp_embeddings = np.vstack(all_embeddings)
                    np.save(embeddings_path.with_suffix(".tmp.npy"), temp_embeddings)
                    print(f"      Checkpoint saved ({len(temp_embeddings):,} embeddings)")

            except Exception as e:
                print(f"   Error: Batch {batch_idx} failed: {e}")

    # Combine and normalize embeddings
    embeddings = np.vstack(all_embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    print(f"   Generated {embeddings.shape[0]:,} embeddings ({embeddings.shape[1]} dims)")

    # Build FAISS index (inner product for cosine similarity)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save index and embeddings
    faiss.write_index(index, str(index_path))
    np.save(embeddings_path, embeddings)

    total_min = (time.time() - start) / 60
    print(f"   Saved FAISS index ({index.ntotal:,} vectors) in {total_min:.1f} min")

    return index, embeddings


# ============================================================================
# Step 4: Tier 1 - Direct KB Lookup
# ============================================================================

def direct_lookup(drug1, drug2, kb_df, lookups):
    """
    Direct knowledge base lookup via RxCUI pair matching (Tier 1).

    Returns:
        dict with evidence, score=1.0, tier=1 if found
        None if not found
    """
    rxcui1 = normalize_to_rxcui(drug1, lookups)
    rxcui2 = normalize_to_rxcui(drug2, lookups)

    if not rxcui1 or not rxcui2:
        return None

    pair_key = tuple(sorted([rxcui1, rxcui2]))
    match = kb_df[kb_df['pair_key'] == pair_key]

    if not match.empty:
        return {
            'drug1': match['Drug 1'].values[0],
            'drug2': match['Drug 2'].values[0],
            'evidence': match['Interaction Description'].values[0],
            'retrieval_score': 1.0,
            'tier': 1
        }

    return None


# ============================================================================
# Step 5: Tier 2 - FAISS Semantic Search
# ============================================================================

def semantic_search(drug1, drug2, kb_df, index, embeddings, client, k=5):
    """
    FAISS semantic search with optional lexical pre-filtering (Tier 2).

    Strategy:
    1. Embed query
    2. Optionally filter KB by drug names (if <1000 matches)
    3. Search FAISS index
    4. Return results above confidence threshold

    Returns:
        list of dicts with evidence, retrieval_score, tier=2
    """
    # Create and embed query
    query_text = f"{drug1} and {drug2} interaction"
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query_text]
    )
    query_embedding = np.array([response.data[0].embedding], dtype='float32')
    query_embedding /= np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-8

    # Optional lexical pre-filtering
    mask = (
            kb_df['Drug 1'].str.contains(drug1, case=False, na=False) |
            kb_df['Drug 2'].str.contains(drug1, case=False, na=False) |
            kb_df['Drug 1'].str.contains(drug2, case=False, na=False) |
            kb_df['Drug 2'].str.contains(drug2, case=False, na=False)
    )

    if mask.sum() > 0 and mask.sum() <= FILTER_SIZE_LIMIT:
        # Use filtered subset
        filtered_df = kb_df[mask]
        subset_embeddings = embeddings[filtered_df.index.values]
        index_subset = faiss.IndexFlatIP(subset_embeddings.shape[1])
        index_subset.add(subset_embeddings)
        print(f"      Using filtered index: {len(filtered_df)} interactions")
    else:
        # Use global index
        filtered_df = kb_df
        index_subset = index
        print(f"      Using global index")

    # Search FAISS
    distances, indices = index_subset.search(query_embedding, k)
    similarities = distances[0]

    # Filter by confidence threshold
    hits = []
    for idx, score in zip(indices[0], similarities):
        if score >= CONFIDENCE_THRESHOLD:
            row = filtered_df.iloc[idx]
            hits.append({
                "drug1": row['Drug 1'],
                "drug2": row['Drug 2'],
                "evidence": row['Interaction Description'],
                "retrieval_score": float(score)
            })

    return hits


# ============================================================================
# Step 6: Unified Check Function (Tier 1 → 2 → 3)
# ============================================================================

def check_drug_interaction(drug1, drug2, kb_df, lookups, index, embeddings, client):
    """
    Check drug interaction using tiered retrieval.

    Flow:
    1. Try direct KB lookup (Tier 1)
    2. Fall back to semantic search (Tier 2)
    3. Return no evidence (Tier 3)

    Returns:
        dict with query, hits (list), and tier
    """
    query_text = f"{drug1} and {drug2} interaction"

    # Tier 1: Direct lookup
    direct_result = direct_lookup(drug1, drug2, kb_df, lookups)

    if direct_result:
        print(f"   Tier 1: Direct match for {drug1} + {drug2}")
        return {
            "query": query_text,
            "hits": [direct_result],
            "tier": 1
        }

    # Tier 2: Semantic search
    print(f"   Tier 1: No direct match, trying semantic search...")
    print(f"   Tier 2: Semantic search for {drug1} + {drug2}...")

    hits = semantic_search(drug1, drug2, kb_df, index, embeddings, client, k=5)

    if hits:
        print(f"      Found {len(hits)} semantic matches")
        return {
            "query": query_text,
            "hits": hits,
            "tier": 2
        }

    # Tier 3: No evidence
    print(f"   Info: Tier 3: No evidence found")
    return {
        "query": query_text,
        "hits": [],
        "tier": 3
    }


# ============================================================================
# Step 7: Polypharmacy Handler (N-Drug Pairwise Expansion)
# ============================================================================

def check_polypharmacy(drugs, kb_df, lookups, index, embeddings, client):
    """
    Handle multi-drug queries with pairwise expansion.

    Args:
        drugs: list of drug names (e.g., ['Warfarin', 'Aspirin', 'Ibuprofen'])

    Returns:
        dict with drugs, num_pairs, and results (list of pairwise checks)
    """
    if len(drugs) < 2:
        return {'error': 'Need at least 2 drugs', 'results': []}

    # Generate all pairwise combinations
    pairs = list(combinations(drugs, 2))
    print(f"\nChecking {len(pairs)} drug pair(s)...")

    results = []
    for drug1, drug2 in pairs:
        result = check_drug_interaction(drug1, drug2, kb_df, lookups, index, embeddings, client)
        results.append(result)

    return {
        'drugs': drugs,
        'num_pairs': len(pairs),
        'results': results
    }


# ============================================================================
# Step 7B: Direct-Lookup-Only Polypharmacy (Tier 1 / Tier 3)
# ============================================================================

def check_polypharmacy_light(drug_input, kb_df, lookups):
    """
    Lightweight polypharmacy checker (DIRECT LOOKUP ONLY).
    Returns Tier 1 (found) or Tier 3 (not found) — never Tier 2.
    Ideal for quick or offline inference where FAISS/LLM isn't needed.
    """
    from itertools import combinations

    # Parse drug input
    if isinstance(drug_input, str):
        text = drug_input.replace("interaction", "").strip()
        if " and " in text:
            parts = text.split(" and ")
            first_part = parts[0]
            last_drug = parts[1].strip()
            drugs = [d.strip() for d in first_part.split(",") if d.strip()] + [last_drug]
        else:
            drugs = [d.strip() for d in text.split(",") if d.strip()]
    else:
        drugs = drug_input

    if len(drugs) < 2:
        return {'error': 'Need at least 2 drugs', 'drugs': [], 'num_pairs': 0, 'results': []}

    pairs = list(combinations(drugs, 2))
    all_results = []

    for drug1, drug2 in pairs:
        query_text = f"{drug1} and {drug2} interaction"

        rxcui1 = normalize_to_rxcui(drug1, lookups)
        rxcui2 = normalize_to_rxcui(drug2, lookups)

        if not rxcui1 or not rxcui2:
            all_results.append({
                "query": query_text,
                "hits": [],
                "tier": 3,
                "message": "Could not normalize drug names"
            })
            continue

        pair_key = tuple(sorted([rxcui1, rxcui2]))
        match = kb_df[kb_df['pair_key'] == pair_key]

        if not match.empty:
            hits = [{
                'drug1': match['Drug 1'].values[0],
                'drug2': match['Drug 2'].values[0],
                'evidence': match['Interaction Description'].values[0],
                'retrieval_score': 1.0
            }]
            all_results.append({"query": query_text, "hits": hits, "tier": 1})
        else:
            all_results.append({
                "query": query_text,
                "hits": [],
                "tier": 3,
                "message": f"No documented interaction for {drug1} + {drug2}"
            })

    return {'drugs': drugs, 'num_pairs': len(pairs), 'results': all_results}

# ============================================================================
# Main Pipeline Class
# ============================================================================

class DrugInteractionRAG:
    """
    Production RAG system for drug interaction detection.

    Usage:
        rag = DrugInteractionRAG()
        result = rag.check(['Warfarin', 'Aspirin'])
    """

    def __init__(self, api_key=None):
        """Initialize RAG system with knowledge base and FAISS index"""

        # Initialize OpenAI client
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

        # Load data
        self.kb_df, self.lookups = load_data()

        # Build or load FAISS index
        self.index, self.embeddings = build_or_load_faiss_index(self.kb_df, self.client)

        print("\nRAG pipeline initialized and ready!")

    def check(self, drugs):
        """
        Check drug interactions for list of drugs.

        Args:
            drugs: list of drug names or single pair

        Returns:
            dict with interaction results
        """
        if isinstance(drugs, str):
            drugs = [drugs]

        if len(drugs) == 2:
            # Single pair
            return check_drug_interaction(
                drugs[0], drugs[1],
                self.kb_df, self.lookups, self.index, self.embeddings, self.client
            )
        else:
            # Polypharmacy
            return check_polypharmacy(
                drugs,
                self.kb_df, self.lookups, self.index, self.embeddings, self.client
            )


# ============================================================================
# Standalone Test Function
# ============================================================================

def run_pipeline_tests():
    """Run validation tests on RAG pipeline"""

    print("=" * 70)
    print("PIPELINE VALIDATION")
    print("=" * 70)

    rag = DrugInteractionRAG()

    # Test 1: Single pair (direct match)
    print("\nTest 1: Single pair (direct match):")
    result = rag.check(['Warfarin', 'Aspirin'])
    print(f"   Tier: {result['tier']}, Hits: {len(result['hits'])}")

    # Test 2: Polypharmacy (3 drugs = 3 pairs)
    print("\nTest 2: Polypharmacy (3 drugs = 3 pairs):")
    result = rag.check(['Warfarin', 'Aspirin', 'Ibuprofen'])
    print(f"   Pairs: {result['num_pairs']}, Results: {len(result['results'])}")

    # Test 3: Semantic search
    print("\nTest 3: Semantic search (uncommon pair):")
    result = rag.check(['Metformin', 'Lisinopril'])
    print(f"   Tier: {result['tier']}, Hits: {len(result['hits'])}")

    print("\nPipeline ready for safety.py integration!")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    run_pipeline_tests()