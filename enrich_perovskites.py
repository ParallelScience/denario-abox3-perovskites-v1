"""
Enrich ABO3 perovskite dataset with elemental A/B site descriptors and
perovskite-specific structural descriptors (Goldschmidt tau, mu, etc.).
Run after download_perovskites.py has saved the base CSV.
"""

import pandas as pd
import numpy as np
from pymatgen.core import Composition, Element

def compute_descriptors(formula):
    """Compute A/B site labels and Goldschmidt descriptors."""
    try:
        c = Composition(formula)
        elems = {str(e): float(c[e]) for e in c.element_composition}
        if "O" not in elems or len(elems) != 3:
            return None
        non_o = {k: v for k, v in elems.items() if k != "O"}
        if len(non_o) != 2:
            return None

        # A-site = less electronegative, B-site = more electronegative
        en = {k: Element(k).X for k in non_o}
        A = min(en, key=en.get)
        B = max(en, key=en.get)

        def get_val(elem, attr):
            """Get float value from Element attribute, unwrapping FloatWithUnit."""
            try:
                raw = getattr(Element(elem), attr)
                return float(raw)  # FloatWithUnit has __float__
            except Exception:
                return np.nan

        r_A = get_val(A, "atomic_radius")
        r_B = get_val(B, "atomic_radius")
        r_O = get_val("O", "atomic_radius")

        # Goldschmidt tolerance factor
        if all(x > 0 for x in [r_A, r_B, r_O]):
            tau = (r_A + r_O) / (np.sqrt(2) * (r_B + r_O))
            mu = r_B / r_O
            radius_ratio = r_A / r_B
        else:
            tau = mu = radius_ratio = np.nan

        return {
            "A_site": A,
            "B_site": B,
            "A_Z": get_val(A, "Z"),
            "B_Z": get_val(B, "Z"),
            "A_radius": r_A,
            "B_radius": r_B,
            "A_en": get_val(A, "X"),
            "B_en": get_val(B, "X"),
            "A_ie1": get_val(A, "ionization_energy"),
            "B_ie1": get_val(B, "ionization_energy"),
            "A_group": get_val(A, "group"),
            "B_group": get_val(B, "group"),
            "en_diff": get_val(B, "X") - get_val(A, "X"),
            "tau": tau,
            "mu": mu,
            "radius_ratio": radius_ratio,
            "B_valence": _b_valence(B),
        }
    except Exception:
        return None


def _b_valence(elem):
    """Get maximum common oxidation state for B-site element."""
    try:
        ox = Element(elem).common_oxidation_states
        return max(ox) if ox else np.nan
    except Exception:
        return np.nan


def main():
    df = pd.read_csv("/home/node/work/projects/materials_project_v3/perovskite_data.csv")
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Descriptor columns to recompute
    desc_cols = [
        "A_site", "B_site", "A_Z", "B_Z", "A_radius", "B_radius",
        "A_en", "B_en", "A_ie1", "B_ie1", "A_group", "B_group",
        "en_diff", "tau", "mu", "radius_ratio", "B_valence"
    ]

    # Drop old broken descriptor columns (if present)
    existing_desc = [c for c in df.columns if c in desc_cols]
    if existing_desc:
        print(f"Removing {len(existing_desc)} old descriptor columns")
        df = df.drop(columns=existing_desc)

    # Compute descriptors fresh for every formula
    print("Computing elemental descriptors...")
    rows = []
    for i, formula in enumerate(df["formula"]):
        d = compute_descriptors(formula)
        rows.append(d if d is not None else {k: np.nan for k in desc_cols})
        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(df)}")

    desc_df = pd.DataFrame(rows, index=df.index)
    df = pd.concat([df, desc_df], axis=1)

    # Pugh ratio (G/K) as mechanical quality metric
    df["pugh_ratio"] = np.where(
        (df["K_VRH"].notna()) & (df["K_VRH"] > 0),
        df["G_VRH"] / df["K_VRH"],
        np.nan
    )

    # Save
    out = "/home/node/work/projects/materials_project_v3/perovskite_data.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {len(df)} rows × {len(df.columns)} columns → {out}")

    # Coverage report
    print("\nProperty coverage:")
    for col in ["band_gap", "formation_energy_per_atom", "energy_above_hull",
                "K_VRH", "G_VRH", "pugh_ratio", "tau", "mu",
                "total_magnetization", "efermi"]:
        if col in df.columns:
            n = int(df[col].notna().sum())
            print(f"  {col}: {n}/{len(df)} ({100*n/len(df):.0f}%)")

    # Tau validation
    print("\nSample tau values (known perovskites):")
    samples = ["SrTiO3", "BaTiO3", "CaSiO3", "MgSiO3", "LaMnO3", "NaNbO3"]
    for s in samples:
        match = df[df["formula"] == s]
        if len(match) > 0:
            row = match.iloc[0]
            print(f"  {s}: A={row['A_site']}, B={row['B_site']}, "
                  f"tau={row['tau']:.4f}, mu={row['mu']:.4f}, "
                  f"gap={row['band_gap']:.3f} eV, E_hull={row['energy_above_hull']:.4f}")

    print("\nCrystal systems:")
    print(df["crystal_system"].value_counts().to_string())
    print("\nMagnetic orderings:")
    print(df["magnetic_ordering"].value_counts().to_string())
    print("\nB-site elements (top 15):")
    print(df["B_site"].value_counts().head(15).to_string())
    print("\nA-site elements (top 10):")
    print(df["A_site"].value_counts().head(10).to_string())
    print(f"\nB-valence distribution:")
    print(df["B_valence"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()