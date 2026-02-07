#!/usr/bin/env python
"""
Synthetic Prompt Generator for UCSF-PDGM

Generates clinical prompts from CSV metadata columns.

Usage:
    python scripts/generate_prompts.py \
        --csv data/UCSF-PDGM-metadata_v5.csv \
        --output data/prompts.csv
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate clinical prompts from metadata")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to UCSF-PDGM metadata CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for prompts CSV",
    )
    return parser.parse_args()


def format_sex(sex: str) -> str:
    """Format sex value."""
    sex = str(sex).strip().upper()
    if sex == "M":
        return "male"
    elif sex == "F":
        return "female"
    return "patient"


def format_diagnosis(diagnosis: str) -> str:
    """Format pathologic diagnosis."""
    diagnosis = str(diagnosis).strip().lower()

    if "glioblastoma" in diagnosis:
        return "glioblastoma"
    elif "astrocytoma" in diagnosis:
        return "astrocytoma"
    elif "oligodendroglioma" in diagnosis:
        return "oligodendroglioma"
    elif "glioma" in diagnosis:
        return "diffuse glioma"
    else:
        return diagnosis


def format_idh(idh: str) -> str:
    """Format IDH status."""
    idh = str(idh).strip().lower()

    if "wildtype" in idh:
        return "IDH-wildtype"
    elif "mutant" in idh or "mutated" in idh:
        return "IDH-mutant"
    else:
        return ""


def format_mgmt(mgmt: str) -> str:
    """Format MGMT status."""
    mgmt = str(mgmt).strip().lower()

    if mgmt == "positive":
        return "MGMT-methylated"
    elif mgmt == "negative":
        return "MGMT-unmethylated"
    elif mgmt == "indeterminate":
        return "MGMT-indeterminate"
    else:
        return ""


def format_eor(eor: str) -> str:
    """Format extent of resection."""
    eor = str(eor).strip().upper()

    if eor == "GTR":
        return "status post gross total resection"
    elif eor == "STR":
        return "status post subtotal resection"
    elif eor.lower() == "biopsy":
        return "biopsy only"
    else:
        return ""


def generate_prompt(row: pd.Series) -> str:
    """Generate clinical prompt from metadata row.

    Template:
    "{age}-year-old {sex}, {diagnosis}, {idh}, {mgmt}, {treatment}"

    Args:
        row: Pandas Series containing subject metadata.

    Returns:
        Generated clinical prompt string.
    """
    parts = []

    # Age and sex
    age = row.get("Age at MRI", "")
    sex = format_sex(row.get("Sex", ""))
    if pd.notna(age) and age != "":
        parts.append(f"{int(age)}-year-old {sex}")
    else:
        parts.append(sex)

    # Diagnosis
    diagnosis = row.get("Final pathologic diagnosis (WHO 2021)", "")
    if pd.notna(diagnosis) and diagnosis != "":
        parts.append(format_diagnosis(diagnosis))

    # IDH status
    idh = row.get("IDH", "")
    if pd.notna(idh) and idh != "":
        idh_formatted = format_idh(idh)
        if idh_formatted:
            parts.append(idh_formatted)

    # MGMT status
    mgmt = row.get("MGMT status", "")
    if pd.notna(mgmt) and mgmt != "":
        mgmt_formatted = format_mgmt(mgmt)
        if mgmt_formatted:
            parts.append(mgmt_formatted)

    # Extent of resection
    eor = row.get("EOR", "")
    if pd.notna(eor) and eor != "":
        eor_formatted = format_eor(eor)
        if eor_formatted:
            parts.append(eor_formatted)

    # Join parts with commas
    prompt = ", ".join(parts)

    return prompt


def main():
    args = parse_args()

    # Load metadata
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise ValueError(f"Metadata CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Generate prompts
    prompts = []
    for _, row in df.iterrows():
        subject_id = row.get("ID", "")
        if pd.isna(subject_id) or subject_id == "":
            continue

        prompt = generate_prompt(row)
        prompts.append({
            "subject_id": subject_id,
            "prompt": prompt,
        })

    # Create output DataFrame
    prompts_df = pd.DataFrame(prompts)
    print(f"Generated {len(prompts_df)} prompts")

    # Show examples
    print("\nExample prompts:")
    for i, row in prompts_df.head(5).iterrows():
        print(f"  {row['subject_id']}: \"{row['prompt']}\"")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompts_df.to_csv(output_path, index=False)
    print(f"\nSaved prompts to {output_path}")


if __name__ == "__main__":
    main()
