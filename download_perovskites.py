"""
Download large ABO3 perovskite dataset from Materials Project via chemsys searches.
Strategy: search all A-B-O chemical systems for all A-site and B-site combinations,
then filter to ABO3 stoichiometry.
"""

import pandas as pd
import numpy as np
import time
import os
import json
from pymatgen.core import Composition, Element
from mp_api.client import MPRester

MP_API_KEY = os.getenv("MP_API_KEY")

# All available fields from /materials/summary (confirmed working)
AVAILABLE_FIELDS = [
    "material_id", "formula_pretty", "composition", "composition_reduced",
    "formula_anonymous", "chemsys", "nelements",
    "nsites", "volume", "density", "density_atomic",
    "symmetry",
    "formation_energy_per_atom", "energy_above_hull",
    "is_stable", "decomposes_to", "equilibrium_reaction_energy_per_atom",
    "band_gap", "is_gap_direct", "is_metal", "efermi",
    "is_magnetic", "ordering",
    "total_magnetization", "total_magnetization_normalized_vol",
    "total_magnetization_normalized_formula_units",
    "num_magnetic_sites", "num_unique_magnetic_sites",
    "types_of_magnetic_species",
    "bulk_modulus", "shear_modulus",
    "universal_anisotropy", "homogeneous_poisson",
    "weighted_surface_energy", "weighted_surface_energy_EV_PER_ANG2",
    "weighted_work_function", "surface_anisotropy", "shape_factor",
    "energy_per_atom", "uncorrected_energy_per_atom",
    "theoretical", "has_props", "possible_species",
]

# Perovskite A-site cations
A_ELEMENTS = [
    "Li", "Na", "K", "Rb", "Cs",
    "Mg", "Ca", "Sr", "Ba", "Ra",
    "Sc", "Y",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Ti", "Zr", "Hf",
    "V", "Nb", "Ta",
    "Bi", "Pb", "Sb",
]

# Perovskite B-site metals
B_ELEMENTS = [
    "Ti", "Zr", "Hf",
    "V", "Nb", "Ta", "Cr", "Mo", "W",
    "Mn", "Tc", "Re",
    "Fe", "Ru", "Os",
    "Co", "Rh", "Ir",
    "Ni", "Pd", "Pt",
    "Cu", "Zn",
    "Al", "Ga", "In", "Sn", "Ge", "Si", "P", "As", "Sb",
]

# All unique ABO3 chemical systems (A-B-O)
ABO3_SYSTEMS = []
for a in A_ELEMENTS:
    for b in B_ELEMENTS:
        if a != b:
            ABO3_SYSTEMS.append(f"{a}-{b}-O")

print(f"Total ABO3 chemical systems: {len(ABO3_SYSTEMS)}")


def extract_record(doc):
    """Extract flat record from a MaterialsDocument (summary endpoint)."""
    d = {}
    d["material_id"] = doc.get("material_id", None)
    d["formula"] = doc.get("formula_pretty", None)
    d["theoretical"] = doc.get("theoretical", None)
    d["chemsys"] = doc.get("chemsys", None)
    d["nelements"] = doc.get("nelements", None)

    d["volume"] = doc.get("volume", None)
    d["nsites"] = doc.get("nsites", None)
    d["density"] = doc.get("density", None)
    d["density_atomic"] = doc.get("density_atomic", None)
    d["energy_per_atom"] = doc.get("energy_per_atom", None)

    sym = doc.get("symmetry")
    if sym is not None:
        d["spacegroup_symbol"] = getattr(sym, "symbol", None)
        d["spacegroup_number"] = getattr(sym, "number", None)
        d["crystal_system"] = getattr(sym, "crystal_system", None)
    else:
        d["spacegroup_symbol"] = d["spacegroup_number"] = d["crystal_system"] = None

    d["formation_energy_per_atom"] = doc.get("formation_energy_per_atom", None)
    d["energy_above_hull"] = doc.get("energy_above_hull", None)
    d["equilibrium_reaction_energy_per_atom"] = doc.get("equilibrium_reaction_energy_per_atom", None)
    d["is_stable"] = doc.get("is_stable", None)

    d["band_gap"] = doc.get("band_gap", None)
    d["is_gap_direct"] = doc.get("is_gap_direct", None)
    d["is_metal"] = doc.get("is_metal", None)
    d["efermi"] = doc.get("efermi", None)

    d["is_magnetic"] = doc.get("is_magnetic", None)
    d["magnetic_ordering"] = doc.get("ordering", None)
    d["total_magnetization"] = doc.get("total_magnetization", None)
    d["total_magnetization_per_fu"] = doc.get("total_magnetization_normalized_formula_units", None)
    d["num_magnetic_sites"] = doc.get("num_magnetic_sites", None)

    # Elastic: API returns dicts with voigt/reuss/vrh
    km = doc.get("bulk_modulus")
    gm = doc.get("shear_modulus")
    if isinstance(km, dict):
        d["K_VRH"] = km.get("vrh")
        d["K_voigt"] = km.get("voigt")
        d["K_reuss"] = km.get("reuss")
    else:
        d["K_VRH"] = km
        d["K_voigt"] = None
        d["K_reuss"] = None
    if isinstance(gm, dict):
        d["G_VRH"] = gm.get("vrh")
        d["G_voigt"] = gm.get("voigt")
        d["G_reuss"] = gm.get("reuss")
    else:
        d["G_VRH"] = gm
        d["G_voigt"] = None
        d["G_reuss"] = None

    d["elastic_anisotropy"] = doc.get("universal_anisotropy", None)
    d["poisson_ratio"] = doc.get("homogeneous_poisson", None)

    d["surface_energy"] = doc.get("weighted_surface_energy", None)
    d["work_function"] = doc.get("weighted_work_function", None)

    return d


def is_abo3_stoichiometry(formula):
    """Verify ABO3 stoichiometry: 1 A + 1 B + 3 O atoms."""
    try:
        c = Composition(formula)
        elems = list(c.element_composition.keys())
        if len(elems) != 3:
            return False
        if Element("O") not in elems:
            return False
        amt_O = float(c[Element("O")])
        non_o_total = sum(float(c[Element(e)]) for e in elems if str(e) != "O")
        if abs(non_o_total - 2) < 0.05 and abs(amt_O - 3) < 0.05:
            return True
        return False
    except:
        return False


def get_ABO3_descriptors(formula):
    """Derive A/B site labels and compute Goldschmidt descriptors."""
    try:
        c = Composition(formula)
        elems = {str(e): float(c[e]) for e in c.element_composition}
        if "O" not in elems or len(elems) != 3:
            return {}
        non_o = {k: v for k, v in elems.items() if k != "O"}
        if len(non_o) != 2:
            return {}

        en = {k: Element(k).X for k in non_o}
        A = min(en, key=en.get)  # less electronegative = A-site
        B = max(en, key=en.get)  # more electronegative = B-site

        def ep(elem, attr):
            try:
                return getattr(Element(elem), attr)
            except:
                return None

        d = {"A_site": A, "B_site": B}
        d["A_Z"] = ep(A, "Z")
        d["B_Z"] = ep(B, "Z")
        d["A_radius"] = ep(A, "atomic_radius")
        d["B_radius"] = ep(B, "atomic_radius")
        d["A_en"] = ep(A, "X")
        d["B_en"] = ep(B, "X")
        d["A_ie1"] = ep(A, "ionization_energy")
        d["B_ie1"] = ep(B, "ionization_energy")
        d["A_period"] = None  # period not available in pymatgen
        d["B_period"] = None
        d["A_group"] = ep(A, "group")
        d["B_group"] = ep(B, "group")
        d["en_diff"] = d["B_en"] - d["A_en"] if (d["B_en"] and d["A_en"]) else None

        try:
            r_A = Element(A).atomic_radius
            r_B = Element(B).atomic_radius
            r_O = Element("O").atomic_radius
            if all([r_A, r_B, r_O]):
                # Goldschmidt tolerance factor: (r_A + r_O) / sqrt(2) / (r_B + r_O)
                # Use atomic radii in Angstrom
                d["tau"] = (r_A + r_O) / (np.sqrt(2) * (r_B + r_O))
                d["mu"] = r_B / r_O
                d["radius_ratio"] = r_A / r_B
            else:
                d["tau"] = d["mu"] = d["radius_ratio"] = None
        except:
            d["tau"] = d["mu"] = d["radius_ratio"] = None

        # B-site valence state (common oxidation)
        try:
            ox = Element(B).common_oxidation_states
            d["B_valence"] = max(ox) if ox else None
        except:
            d["B_valence"] = None

        return d
    except:
        return {}


def main():
    print("=== ABO3 Perovskite Download via Chemical System Search ===")
    all_records = []
    seen_ids = set()
    chunk_size = 30  # systems per batch

    with MPRester(MP_API_KEY) as mpr:
        for i in range(0, len(ABO3_SYSTEMS), chunk_size):
            chunk = ABO3_SYSTEMS[i:i+chunk_size]
            try:
                docs = mpr.materials.summary.search(
                    chemsys=",".join(chunk),  # comma-separated = OR in MP
                    fields=AVAILABLE_FIELDS
                )
                n_new = 0
                for doc in docs:
                    if doc.get("material_id") not in seen_ids:
                        seen_ids.add(doc.get("material_id"))
                        all_records.append(extract_record(doc))
                        n_new += 1
                elapsed = (i + chunk_size) / len(ABO3_SYSTEMS) * 100
                print(f"  Systems {i+1}–{min(i+chunk_size, len(ABO3_SYSTEMS))}/{len(ABO3_SYSTEMS)} "
                      f"({elapsed:.0f}%): +{len(docs)} docs, {n_new} new (total {len(all_records)})")
            except Exception as e:
                print(f"  Systems {i+1}–{min(i+chunk_size, len(ABO3_SYSTEMS))} error: {e}")
            time.sleep(0.5)

    df = pd.DataFrame(all_records)
    print(f"\nTotal raw records: {len(df)}")

    if len(df) == 0:
        print("ERROR: No records. Check API key.")
        return None

    df = df.drop_duplicates(subset=["material_id"], keep="first")
    print(f"After dedup: {len(df)}")

    # ABO3 stoichiometry filter
    df["is_abo3"] = df["formula"].apply(is_abo3_stoichiometry)
    df_perov = df[df["is_abo3"] == True].copy()
    print(f"Verified ABO3: {len(df_perov)}")

    # Elemental descriptors
    elem_data = [get_ABO3_descriptors(f) for f in df_perov["formula"]]
    elem_df = pd.DataFrame(elem_data).fillna(np.nan)
    df_perov = pd.concat([df_perov.reset_index(drop=True), elem_df], axis=1)

    # Pugh ratio
    df_perov["pugh_ratio"] = np.where(
        (df_perov["K_VRH"].notna()) & (df_perov["K_VRH"] > 0),
        df_perov["G_VRH"] / df_perov["K_VRH"],
        np.nan
    )

    out_path = "/home/node/work/projects/materials_project_v3/perovskite_data.csv"
    cols = [c for c in df_perov.columns if c != "is_abo3"]
    df_perov[cols].to_csv(out_path, index=False)
    print(f"\nSaved {len(df_perov)} ABO3 materials → {out_path}")

    print(f"\n=== Dataset Summary ===")
    print(f"Total ABO3 materials: {len(df_perov)}")
    print(f"Unique formulas: {df_perov['formula'].nunique()}")
    print(f"\nTop formulas:")
    print(df_perov["formula"].value_counts().head(15).to_string())
    print(f"\nProperty coverage:")
    for col in ["band_gap", "formation_energy_per_atom", "energy_above_hull",
                "K_VRH", "G_VRH", "efermi", "tau", "total_magnetization",
                "surface_energy", "pugh_ratio"]:
        if col in df_perov.columns:
            n = df_perov[col].notna().sum()
            print(f"  {col}: {n}/{len(df_perov)} ({100*n/len(df_perov):.0f}%)")
    print(f"\nCrystal systems:")
    print(df_perov["crystal_system"].value_counts().to_string())
    print(f"\nB-site elements (top 15):")
    print(df_perov["B_site"].value_counts().head(15).to_string())

    return df_perov


if __name__ == "__main__":
    main()