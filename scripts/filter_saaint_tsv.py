"""
Prepare SAAINT-DB data - FIXED to handle < values
"""
import pandas as pd
import numpy as np
import re

print("="*70)
print("PREPARING SAAINT-DB DATA (FIXED VERSION)")
print("="*70)

# File paths
main_file = "../data/raw/saaintdb_20251121_all.xlsx"
affinity_file = "../data/raw/saaintdb_affinity_all.tsv"
output_file = "../data/saaint_cleaned.csv"
pdb_list_file = "../data/pdb_ids.txt"

# Step 1: Load files
print("\n[1/10] Loading main database...")
main_df = pd.read_excel(main_file)
print(f"       Main database: {len(main_df)} rows")

print("\n[2/10] Loading affinity database...")
affinity_df = pd.read_csv(affinity_file, sep='\t')
print(f"       Affinity database: {len(affinity_df)} rows")

# Step 2: Filter for non-empty affinity values
print("\n[3/10] Filtering for valid affinity values...")
print(f"       Checking column: 'Affinity_KD(nM)'")

# Remove rows where affinity is 'N.A.' or empty
affinity_df_clean = affinity_df[
    (affinity_df['Affinity_KD(nM)'] != 'N.A.') &
    (affinity_df['Affinity_KD(nM)'].notna()) &
    (affinity_df['Affinity_KD(nM)'] != '')
].copy()

print(f"       After removing N.A./empty: {len(affinity_df_clean)}")

# Step 3: Parse affinity values (handle < values)
print("\n[4/10] Parsing affinity values (including '<' values)...")

def parse_affinity(value):
    """
    Parse affinity value, handling cases like:
    - "57.6" → 57.6
    - "<0.05" → 0.05 (use the limit as upper bound)
    - ">1000" → 1000 (use the limit as lower bound)
    """
    if pd.isna(value) or value == 'N.A.' or value == '':
        return np.nan

    # Convert to string
    value_str = str(value).strip()

    # Handle < or > prefix
    if value_str.startswith('<') or value_str.startswith('>'):
        # Extract the number
        number_str = value_str[1:].strip()
        try:
            return float(number_str)
        except:
            return np.nan

    # Regular number
    try:
        return float(value_str)
    except:
        return np.nan

affinity_df_clean['Kd_nM_parsed'] = affinity_df_clean['Affinity_KD(nM)'].apply(parse_affinity)

print(f"       Successfully parsed: {affinity_df_clean['Kd_nM_parsed'].notna().sum()}")
print(f"       Failed to parse: {affinity_df_clean['Kd_nM_parsed'].isna().sum()}")

# Show examples of what couldn't parse
if affinity_df_clean['Kd_nM_parsed'].isna().sum() > 0:
    print(f"\n       Examples of values that couldn't parse:")
    failed = affinity_df_clean[affinity_df_clean['Kd_nM_parsed'].isna()]['Affinity_KD(nM)'].head(20)
    for val in failed:
        print(f"         '{val}'")

# Remove unparseable values
affinity_df_clean = affinity_df_clean[affinity_df_clean['Kd_nM_parsed'].notna()].copy()
print(f"\n       Rows with parseable affinity: {len(affinity_df_clean)}")

# Step 4: Merge with main database
print("\n[5/10] Merging with main database...")
merged = pd.merge(
    affinity_df_clean,
    main_df[['PDB_ID', 'Resolution', 'H_chain_ID', 'L_chain_ID', 'Method', 'Ab_type', 'Deposit_date']],
    on=['PDB_ID', 'H_chain_ID', 'L_chain_ID'],
    how='inner'
)
print(f"       After merge: {len(merged)} rows")

# Check how many PDBs were lost in merge
affinity_pdbs = set(affinity_df_clean['PDB_ID'].unique())
merged_pdbs = set(merged['PDB_ID'].unique())
lost_in_merge = affinity_pdbs - merged_pdbs
if len(lost_in_merge) > 0:
    print(f"       ⚠ Warning: {len(lost_in_merge)} PDBs from affinity file not found in main database")

# Step 5: Filter for resolution <= 3.0 Angstroms
print("\n[6/10] Filtering for resolution ≤ 3.0 Å...")
merged['Resolution'] = pd.to_numeric(merged['Resolution'], errors='coerce')
filtered = merged[merged['Resolution'] <= 3.0].copy()
print(f"       After resolution filter: {len(filtered)} rows")

# Step 6: Remove any remaining N.A. in resolution
filtered = filtered.dropna(subset=['Resolution'])
print(f"       After removing N.A. resolution: {len(filtered)} rows")

# Step 7: Calculate pKd
print("\n[7/10] Calculating pKd values...")
# pKd = -log10(Kd in Molar units)
# Convert nM to M: divide by 1e9
filtered['Kd_M'] = filtered['Kd_nM_parsed'] / 1e9
filtered['pKd'] = -np.log10(filtered['Kd_M'])

# Flag which values were originally < or >
filtered['affinity_is_limit'] = filtered['Affinity_KD(nM)'].astype(str).str.contains('<|>', regex=True)

print(f"       pKd calculated for {filtered['pKd'].notna().sum()} rows")
print(f"       Values that were limits (</>): {filtered['affinity_is_limit'].sum()}")

# Step 8: Remove duplicates (keep one entry per unique PDB-antibody-antigen combo)
print("\n[8/10] Removing duplicate entries...")
initial_count = len(filtered)
filtered = filtered.drop_duplicates(subset=['PDB_ID', 'H_chain_ID', 'L_chain_ID'], keep='first')
print(f"       Removed {initial_count - len(filtered)} duplicate entries")

# Step 9: Select final columns
final_columns = [
    'PDB_ID', 'Resolution', 'Method', 'Ab_type', 'Deposit_date',
    'H_chain_ID', 'L_chain_ID', 'Ag_chain_ID(s)', 'Ag_type(s)',
    'Affinity_KD(nM)', 'Kd_nM_parsed', 'pKd', 'affinity_is_limit',
    'Affinity_method', 'Affinity_temp(K)', 'PMID', 'DOI'
]
final_df = filtered[final_columns].copy()

# Sort by PDB_ID
final_df = final_df.sort_values('PDB_ID')

# Step 10: Save
print(f"\n[9/10] Saving to {output_file}...")
final_df.to_csv(output_file, index=False)
print(f"       ✓ Saved!")

# Save PDB list
unique_pdbs = final_df['PDB_ID'].unique()
with open(pdb_list_file, 'w') as f:
    for pdb_id in unique_pdbs:
        f.write(f"{pdb_id.lower()}\n")
print(f"\n[10/10] ✓ Saved {len(unique_pdbs)} PDB IDs to {pdb_list_file}")

# Summary statistics
print("\n" + "="*70)
print("FINAL DATASET SUMMARY")
print("="*70)

print(f"\n📊 DATASET SIZE:")
print(f"   Total entries: {len(final_df)}")
print(f"   Unique PDB structures: {len(unique_pdbs)}")
print(f"   Entries with limit values (</>): {final_df['affinity_is_limit'].sum()}")

print(f"\n🔬 RESOLUTION:")
print(f"   Min:  {final_df['Resolution'].min():.2f} Å")
print(f"   Max:  {final_df['Resolution'].max():.2f} Å")
print(f"   Mean: {final_df['Resolution'].mean():.2f} Å")

print(f"\n💊 AFFINITY (Kd in nM):")
print(f"   Min:  {final_df['Kd_nM_parsed'].min():.4f} nM")
print(f"   Max:  {final_df['Kd_nM_parsed'].max():.2f} nM")
print(f"   Mean: {final_df['Kd_nM_parsed'].mean():.2f} nM")

print(f"\n📈 pKd:")
print(f"   Min:  {final_df['pKd'].min():.2f}")
print(f"   Max:  {final_df['pKd'].max():.2f}")
print(f"   Mean: {final_df['pKd'].mean():.2f}")

print(f"\n📅 DEPOSIT DATE RANGE:")
print(f"   Earliest: {final_df['Deposit_date'].min()}")
print(f"   Latest:   {final_df['Deposit_date'].max()}")

print(f"\n🔬 METHOD:")
print(final_df['Method'].value_counts())

print(f"\n🧬 ANTIBODY TYPE (Top 5):")
print(final_df['Ab_type'].value_counts().head(5))

print("\n" + "="*70)
print("✓ DATA PREPARATION COMPLETE!")
print("="*70)

print(f"\n📂 OUTPUT FILES:")
print(f"   1. {output_file}")
print(f"   2. {pdb_list_file}")
print(f"\n🎯 NEXT STEP: Download {len(unique_pdbs)} PDB files")

# Compare to manual count
print(f"\n📊 COMPARISON:")
print(f"   Your manual Google Sheets count: 1,228 unique PDBs")
print(f"   This script: {len(unique_pdbs)} unique PDBs")
print(f"   Difference: {abs(1228 - len(unique_pdbs))} PDBs")
if len(unique_pdbs) < 1228:
    print(f"   → Lost {1228 - len(unique_pdbs)} PDBs due to resolution > 3.0 Å filter")