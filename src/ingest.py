"""
Drug Interaction Checker - Data Ingestion Pipeline
===================================================
Author: Ridwan Oladipo, MD | Medical AI Specialist

Production-grade ingestion pipeline that:
1. Loads RxNorm mappings and builds hierarchical drug name normalization
2. Resolves brand/generic/synonym names to ingredient RxCUIs
3. Maps DrugBank interactions to RxCUI pairs
4. Generates clean knowledge base for RAG retrieval

Achieves 90% mapping coverage on 191K DrugBank interactions.
"""

import pandas as pd
import numpy as np
import requests
import pickle
import time
import re
from collections import defaultdict, Counter
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

TYPE_PRIORITY = {'IN': 1, 'BN': 2, 'SN': 3, 'PSN': 4, 'PIN': 5, 'SCD': 6, 'SBD': 7}
DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR


# ============================================================================
# Utility Functions
# ============================================================================

def normalize_drug(name):
    """Normalize drug name: lowercase, strip, remove extra spaces"""
    return re.sub(r'\s+', ' ', name.lower().strip())


def atomic_save(obj, path):
    """Atomic save to prevent corruption on crash"""
    temp_file = path.with_suffix(".tmp")
    with open(temp_file, "wb") as f:
        pickle.dump(obj, f)
    temp_file.replace(path)


def get_ingredient_from_api(rxcui, max_retries=3):
    """Resolve BN/SY/PIN/PSN RxCUI → IN RxCUI via RxNav API"""
    for attempt in range(max_retries):
        try:
            url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/related.json?tty=IN"
            resp = requests.get(url, timeout=10)

            if resp.status_code == 429:
                time.sleep(2)
                continue

            data = resp.json()
            if 'relatedGroup' in data:
                for group in data['relatedGroup'].get('conceptGroup', []):
                    if group.get('tty') == 'IN' and 'conceptProperties' in group:
                        ing_rxcui = group['conceptProperties'][0]['rxcui']
                        ing_name = group['conceptProperties'][0]['name']
                        return ing_rxcui, ing_name
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue

    return None, None


# ============================================================================
# Step 1: Build RxNorm Lookup Dictionaries
# ============================================================================

def build_rxnorm_lookups(rxnorm_path):
    """
    Build primary lookup dictionaries from RxNorm mappings.

    Returns:
        name_to_rxcui: dict mapping normalized names to (rxcui, type)
        rxcui_to_names: dict mapping rxcui to set of all its names
    """
    print("Loading RxNorm mappings...")
    rxnorm_df = pd.read_csv(rxnorm_path)

    print(f"   Shape: {rxnorm_df.shape}")
    print(f"   Type distribution:\n{rxnorm_df['type'].value_counts()}")

    name_to_rxcui = {}
    rxcui_to_names = defaultdict(set)

    for _, row in rxnorm_df.iterrows():
        rxcui = row['rxcui']
        name_norm = row['name_norm']
        drug_type = row['type']

        rxcui_to_names[rxcui].add(name_norm)

        if name_norm in name_to_rxcui:
            existing_priority = TYPE_PRIORITY.get(name_to_rxcui[name_norm][1], 99)
            new_priority = TYPE_PRIORITY.get(drug_type, 99)
            if new_priority < existing_priority:
                name_to_rxcui[name_norm] = (rxcui, drug_type)
        else:
            name_to_rxcui[name_norm] = (rxcui, drug_type)

    print(f"Built lookup dictionaries:")
    print(f"   Unique names: {len(name_to_rxcui):,}")
    print(f"   Unique RxCUIs: {len(rxcui_to_names):,}")

    return name_to_rxcui, rxcui_to_names


# ============================================================================
# Step 2: Build Brand→Ingredient Cache
# ============================================================================

def build_brand_to_ingredient_cache(rxnorm_df, cache_path):
    """
    Build brand→ingredient mapping cache via RxNav API.

    Note: First run takes 3-4 days. Subsequent runs load from cache.

    Returns:
        bn_to_in_map: dict mapping brand RxCUI to (ingredient_rxcui, ingredient_name)
    """
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            bn_to_in_map = pickle.load(f)
        print(f"Loaded existing cache: {len(bn_to_in_map):,} entries")
        return bn_to_in_map

    print("Building brand→ingredient cache via API (this will take 3-4 days)...")

    non_in_df = rxnorm_df[rxnorm_df['type'] != "IN"]
    unique_non_in = non_in_df[['rxcui', 'name_norm']].drop_duplicates(subset='rxcui')

    bn_to_in_map = {}
    total = len(unique_non_in)
    processed = 0

    for idx, row in unique_non_in.iterrows():
        rxcui = str(row['rxcui'])

        if rxcui in bn_to_in_map:
            processed += 1
            continue

        ing_rxcui, ing_name = get_ingredient_from_api(rxcui)
        processed += 1

        if ing_rxcui:
            bn_to_in_map[rxcui] = (ing_rxcui, ing_name)

        if processed % 50 == 0:
            atomic_save(bn_to_in_map, cache_path)
            print(
                f"   Progress: {processed}/{total} processed, {len(bn_to_in_map):,} mapped ({100 * len(bn_to_in_map) / processed:.1f}% success)")

        time.sleep(0.2)

    atomic_save(bn_to_in_map, cache_path)
    print(f"Final: {len(bn_to_in_map):,}/{total:,} entries ({100 * len(bn_to_in_map) / total:.1f}% coverage)")

    return bn_to_in_map


# ============================================================================
# Step 3: Build Reverse Ingredient Index
# ============================================================================

def build_ingredient_index(bn_to_in_map, api_cache_path):
    """
    Build reverse index: ingredient_name → ingredient_rxcui

    Returns:
        ingredient_name_to_rxcui: dict mapping normalized ingredient names to RxCUIs
    """
    ingredient_name_to_rxcui = {}

    for brand_rxcui, (ing_rxcui, ing_name) in bn_to_in_map.items():
        norm_name = normalize_drug(ing_name)
        if norm_name not in ingredient_name_to_rxcui:
            ingredient_name_to_rxcui[norm_name] = ing_rxcui

    print(f"Reverse index: {len(ingredient_name_to_rxcui):,} unique ingredient names")

    # Merge previous API discoveries
    if api_cache_path.exists():
        with open(api_cache_path, "rb") as f:
            api_discoveries = pickle.load(f)
        ingredient_name_to_rxcui.update(api_discoveries)
        print(f"Merged {len(api_discoveries):,} previous API discoveries")
        print(f"Total ingredient cache: {len(ingredient_name_to_rxcui):,}")

    # Add common aliases
    ingredient_name_to_rxcui['paracetamol'] = '161'

    return ingredient_name_to_rxcui


# ============================================================================
# Step 4: Normalization Function
# ============================================================================

def normalize_to_ingredient_rxcui(drug_name, name_to_rxcui, bn_to_in_map, ingredient_name_to_rxcui):
    """
    Normalize any drug name (brand, synonym, ingredient) to ingredient RxCUI.

    Strategy:
    1. Check local name_to_rxcui → use bn_to_in_map if needed
    2. Check reverse index (ingredient names from API cache)
    3. Fallback to live RxNav API call

    Returns: (rxcui, 'IN') or (None, None)
    """
    norm_name = normalize_drug(drug_name)

    # 1. Local lookup
    if norm_name in name_to_rxcui:
        rxcui, dtype = name_to_rxcui[norm_name]
        if dtype == "IN":
            return str(rxcui), "IN"
        if str(rxcui) in bn_to_in_map:
            ing_rxcui, _ = bn_to_in_map[str(rxcui)]
            return ing_rxcui, "IN"

    # 2. Reverse index
    if norm_name in ingredient_name_to_rxcui:
        return ingredient_name_to_rxcui[norm_name], "IN"

    # 3. API fallback
    try:
        url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={drug_name}"
        resp = requests.get(url, timeout=5).json()

        if 'idGroup' in resp and 'rxnormId' in resp['idGroup']:
            rxcui = resp['idGroup']['rxnormId'][0]

            ing_url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/related.json?tty=IN"
            ing_resp = requests.get(ing_url, timeout=5).json()

            if 'relatedGroup' in ing_resp:
                for group in ing_resp['relatedGroup'].get('conceptGroup', []):
                    if group.get('tty') == 'IN' and 'conceptProperties' in group:
                        return group['conceptProperties'][0]['rxcui'], "IN"

            return str(rxcui), None
    except:
        pass

    return None, None


# ============================================================================
# Step 5: Map DrugBank Interactions
# ============================================================================

def map_drugbank_to_rxcui(interactions_path, name_to_rxcui, bn_to_in_map, ingredient_name_to_rxcui, api_cache_path):
    """
    Map DrugBank drug names to ingredient RxCUIs.

    Returns:
        interactions_df: DataFrame with added drug1_rxcui and drug2_rxcui columns
        newly_resolved: dict of new API discoveries
    """
    print("Loading DrugBank interactions...")
    interactions_df = pd.read_csv(interactions_path)
    print(f"   Shape: {interactions_df.shape}")

    newly_resolved = {}

    def map_to_ingredient_rxcui(drug_name):
        """Map DrugBank drug name to ingredient RxCUI with dynamic caching"""
        norm_name = normalize_drug(drug_name)

        # 1. Local lookup
        if norm_name in name_to_rxcui:
            rxcui, dtype = name_to_rxcui[norm_name]
            if dtype == "IN":
                return str(rxcui)
            if str(rxcui) in bn_to_in_map:
                ing_rxcui, _ = bn_to_in_map[str(rxcui)]
                return ing_rxcui

        # 2. Reverse index
        if norm_name in ingredient_name_to_rxcui:
            return ingredient_name_to_rxcui[norm_name]

        # 3. API fallback with caching
        try:
            url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={drug_name}"
            resp = requests.get(url, timeout=5).json()

            if 'idGroup' in resp and 'rxnormId' in resp['idGroup']:
                rxcui = resp['idGroup']['rxnormId'][0]

                ing_url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/related.json?tty=IN"
                ing_resp = requests.get(ing_url, timeout=5).json()

                if 'relatedGroup' in ing_resp:
                    for group in ing_resp['relatedGroup'].get('conceptGroup', []):
                        if group.get('tty') == 'IN' and 'conceptProperties' in group:
                            ing_rxcui = group['conceptProperties'][0]['rxcui']
                            ing_name = group['conceptProperties'][0]['name']

                            # Cache for future runs
                            norm_ing_name = normalize_drug(ing_name)
                            ingredient_name_to_rxcui[norm_ing_name] = ing_rxcui
                            newly_resolved[norm_ing_name] = ing_rxcui

                            if len(newly_resolved) % 50 == 0:
                                print(f"   Cached {len(newly_resolved):,} new ingredients...")

                            return ing_rxcui

                return str(rxcui)
        except:
            pass

        return None

    # Map drugs
    print("Mapping Drug 1...")
    interactions_df['drug1_rxcui'] = interactions_df['Drug 1'].apply(map_to_ingredient_rxcui)

    print("Mapping Drug 2...")
    interactions_df['drug2_rxcui'] = interactions_df['Drug 2'].apply(map_to_ingredient_rxcui)

    # Save new discoveries
    if api_cache_path.exists():
        with open(api_cache_path, "rb") as f:
            prev = pickle.load(f)
        prev.update(newly_resolved)
        newly_resolved = prev

    with open(api_cache_path, "wb") as f:
        pickle.dump(newly_resolved, f)

    print(f"Saved {len(newly_resolved):,} total API discoveries")

    return interactions_df, newly_resolved


# ============================================================================
# Step 6: Build Clean Knowledge Base
# ============================================================================

def build_knowledge_base(interactions_df):
    """
    Build clean, deduplicated knowledge base from mapped interactions.

    Returns:
        kb_df: Clean knowledge base with bidirectional pair keys
    """
    # Filter to only mapped pairs
    kb_df = interactions_df[
        interactions_df['drug1_rxcui'].notnull() &
        interactions_df['drug2_rxcui'].notnull()
        ].copy()

    # Create bidirectional pair keys
    kb_df['pair_key'] = kb_df.apply(
        lambda row: tuple(sorted([row['drug1_rxcui'], row['drug2_rxcui']])),
        axis=1
    )

    # Remove duplicates
    kb_df = kb_df.drop_duplicates(subset='pair_key')

    print(f"Clean Knowledge Base:")
    print(f"   Total interactions: {len(kb_df):,}")
    print(f"   Unique drug pairs: {kb_df['pair_key'].nunique():,}")

    return kb_df


# ============================================================================
# Step 7: Display Statistics
# ============================================================================

def display_statistics(interactions_df, kb_df, bn_to_in_map):
    """Display mapping statistics and validation"""

    # Mapping statistics
    unmapped_drug1 = interactions_df['drug1_rxcui'].isnull().sum()
    unmapped_drug2 = interactions_df['drug2_rxcui'].isnull().sum()
    total = len(interactions_df)

    print(f"\nMapping Results:")
    print(f"   Drug 1 mapped: {total - unmapped_drug1:,} / {total:,} ({100 * (1 - unmapped_drug1 / total):.1f}%)")
    print(f"   Drug 2 mapped: {total - unmapped_drug2:,} / {total:,} ({100 * (1 - unmapped_drug2 / total):.1f}%)")
    print(f"   Pair-level mapped: {len(kb_df):,} / {total:,} ({100 * len(kb_df) / total:.1f}%)")

    # Synonym unification validation
    collapse_check = Counter(bn_to_in_map.values())
    print("\nTop 10 ingredients by brand/synonym count:")
    for (ing_rxcui, ing_name), count in collapse_check.most_common(10):
        print(f"   {ing_name:40} ({ing_rxcui}) → {count} variants")

    # Acetaminophen unification
    brands_to_161 = [bn for bn, (ing, _) in bn_to_in_map.items() if ing == '161']
    print(f"\nAcetaminophen (RxCUI 161): {len(brands_to_161)} brand variants unified")


# ============================================================================
# Step 8: Save Outputs
# ============================================================================

def save_outputs(kb_df, name_to_rxcui, rxcui_to_names, bn_to_in_map, ingredient_name_to_rxcui):
    """Save processed knowledge base and lookup caches"""

    # Save knowledge base
    kb_path = DATA_DIR / 'processed_interactions_kb.csv'
    kb_df.to_csv(kb_path, index=False)
    print(f"Saved: {kb_path}")

    # Save all lookup dictionaries
    lookups_path = DATA_DIR / 'rxnorm_lookups.pkl'
    with open(lookups_path, 'wb') as f:
        pickle.dump({
            'name_to_rxcui': dict(name_to_rxcui),
            'rxcui_to_names': dict(rxcui_to_names),
            'bn_to_in_map': dict(bn_to_in_map),
            'ingredient_name_to_rxcui': dict(ingredient_name_to_rxcui)
        }, f)
    print(f"Saved: {lookups_path}")

    print("\nData ingestion complete!")
    print(f"Final statistics:")
    print(f"   Knowledge base: {len(kb_df):,} interactions")
    print(f"   Name→RxCUI mappings: {len(name_to_rxcui):,}")
    print(f"   Brand→Ingredient cache: {len(bn_to_in_map):,}")
    print(f"   Ingredient name cache: {len(ingredient_name_to_rxcui):,}")


# ============================================================================
# Main Pipeline
# ============================================================================

def run_ingestion_pipeline():
    """Execute complete data ingestion pipeline"""

    print("=" * 70)
    print("Drug Interaction Checker - Data Ingestion Pipeline")
    print("=" * 70)

    # Paths
    rxnorm_path = DATA_DIR / 'rxnorm_mappings.csv'
    interactions_path = DATA_DIR / 'drugbank_interactions.csv'
    bn_cache_path = CACHE_DIR / 'bn_to_in_map.pkl'
    api_cache_path = CACHE_DIR / 'new_api_ingredients.pkl'

    # Step 1: Build RxNorm lookups
    rxnorm_df = pd.read_csv(rxnorm_path)
    name_to_rxcui, rxcui_to_names = build_rxnorm_lookups(rxnorm_path)

    # Step 2: Build brand→ingredient cache
    bn_to_in_map = build_brand_to_ingredient_cache(rxnorm_df, bn_cache_path)

    # Step 3: Build reverse ingredient index
    ingredient_name_to_rxcui = build_ingredient_index(bn_to_in_map, api_cache_path)

    # Add paracetamol alias to primary lookup
    name_to_rxcui['paracetamol'] = ('161', 'IN')

    # Step 4: Map DrugBank interactions
    interactions_df, newly_resolved = map_drugbank_to_rxcui(
        interactions_path,
        name_to_rxcui,
        bn_to_in_map,
        ingredient_name_to_rxcui,
        api_cache_path
    )

    # Step 5: Build clean knowledge base
    kb_df = build_knowledge_base(interactions_df)

    # Step 6: Display statistics
    display_statistics(interactions_df, kb_df, bn_to_in_map)

    # Step 7: Save outputs
    save_outputs(kb_df, name_to_rxcui, rxcui_to_names, bn_to_in_map, ingredient_name_to_rxcui)

    return kb_df, name_to_rxcui, rxcui_to_names, bn_to_in_map, ingredient_name_to_rxcui


if __name__ == "__main__":
    run_ingestion_pipeline()