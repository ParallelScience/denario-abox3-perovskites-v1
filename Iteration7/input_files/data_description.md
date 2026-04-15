# ABO3 Perovskite Dataset — Materials Project

## Overview

This dataset contains **1283 stoichiometrically-verified ABO3 perovskite compounds** downloaded from the Materials Project database (https://materialsproject.org) via the `/materials/summary` API endpoint. The data was retrieved by querying all A–B–O chemical systems where A ∈ {Li, Na, K, Rb, Cs, Mg, Ca, Sr, Ba, Sc, Y, La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Ti, Zr, Hf, V, Nb, Ta, Bi, Pb, Sb} and B ∈ {Ti, Zr, Hf, V, Nb, Ta, Cr, Mo, W, Mn, Tc, Re, Fe, Ru, Os, Co, Rh, Ir, Ni, Pd, Pt, Cu, Zn, Al, Ga, In, Sn, Ge, Si, P, As, Sb}.

Each entry represents a distinct Materials Project document identified by a unique `material_id`. The dataset contains **574 unique ABO3 formulas** across 7 crystal systems, with full coverage of thermodynamic, electronic, and magnetic properties.

---

## File Inventory

| File | Description |
|------|-------------|
| `/home/node/work/projects/materials_project_v3/perovskite_data.csv` | Main dataset — 1283 rows × 75 columns |

### CSV Column Inventory

**Identifiers**
- `material_id` (str): Materials Project unique ID (e.g., "mp-12345")
- `formula` (str): Chemical formula in standard format (e.g., "SrTiO3")
- `theoretical` (bool): True if computationally predicted, False if experimental
- `chemsys` (str): Chemical system string (e.g., "Sr-Ti-O")

**Structural**
- `nelements` (int): Number of elements in the formula (=3 for ABO3)
- `nsites` (int): Total number of atomic sites in the unit cell
- `volume` (float): Unit cell volume in Å³
- `density` (float): Mass density in g/cm³
- `density_atomic` (float): Atomic density in atoms/Å³
- `energy_per_atom` (float): Total energy per atom in eV/atom

**Symmetry**
- `spacegroup_symbol` (str): Space group symbol (e.g., "Pm-3m")
- `spacegroup_number` (int): Space group number (1–230)
- `crystal_system` (str): One of {Cubic, Hexagonal, Monoclinic, Orthorhombic, Tetragonal, Trigonal, Triclinic}

**Thermodynamic**
- `formation_energy_per_atom` (float): DFT formation energy in eV/atom (relative to elemental reference states)
- `energy_above_hull` (float): Energy above the convex hull in eV/atom (>0 = metastable, 0 = stable)
- `equilibrium_reaction_energy_per_atom` (float): Reaction energy from the most stable decomposition pathway in eV/atom
- `is_stable` (bool): True if energy_above_hull = 0 (on the convex hull)

**Electronic**
- `band_gap` (float): DFT PBE band gap in eV
- `is_gap_direct` (bool): True if the band gap is direct
- `is_metal` (bool): True if the material is metallic (band_gap = 0)
- `efermi` (float): Fermi level position in eV (relative to vacuum)

**Magnetic**
- `is_magnetic` (bool): True if the calculation included spin polarization
- `magnetic_ordering` (str): Magnetic ground state — one of {NM (non-magnetic), FM (ferromagnetic), AFM (antiferromagnetic), FiM (ferrimagnetic)}
- `total_magnetization` (float): Total magnetic moment in μ_B per formula unit
- `total_magnetization_per_fu` (float): Magnetic moment per formula unit (normalized)
- `num_magnetic_sites` (int): Number of magnetic atomic sites in the structure

**Elastic (VRH averages)**
- `K_VRH` (float): Voigt–Reuss–Hill bulk modulus in GPa
- `K_voigt` (float): Voigt bulk modulus in GPa
- `K_reuss` (float): Reuss bulk modulus in GPa
- `G_VRH` (float): Voigt–Reuss–Hill shear modulus in GPa
- `G_voigt` (float): Voigt shear modulus in GPa
- `G_reuss` (float): Reuss shear modulus in GPa
- `elastic_anisotropy` (float): Universal elastic anisotropy index (0 = isotropic)
- `poisson_ratio` (float): Homogeneous Poisson ratio
- `pugh_ratio` (float): G_VRH / K_VRH ratio — measures shear rigidity relative to bulk incompressibility

**Surface**
- `surface_energy` (float): Weighted surface energy in J/m²
- `work_function` (float): Weighted work function in eV

**Elemental A-site descriptors (derived from composition)**
- `A_site` (str): A-site element symbol
- `A_Z` (int): Atomic number of A-site element
- `A_radius` (float): Atomic radius of A-site element in Å
- `A_en` (float): Pauling electronegativity of A-site element
- `A_ie1` (float): First ionization energy of A-site element in eV
- `A_group` (int): Periodic table group number of A-site element
- `A_period` (float): Period of A-site element (null for all entries — not available in pymatgen Element API)

**Elemental B-site descriptors (derived from composition)**
- `B_site` (str): B-site element symbol
- `B_Z` (int): Atomic number of B-site element
- `B_radius` (float): Atomic radius of B-site element in Å
- `B_en` (float): Pauling electronegativity of B-site element
- `B_ie1` (float): First ionization energy of B-site element in eV
- `B_group` (int): Periodic table group number of B-site element
- `B_period` (float): Period of B-site element (null for all entries)
- `B_valence` (int): Maximum common oxidation state of B-site element

**Perovskite structural descriptors (derived from radii)**
- `tau` (float): Goldschmidt tolerance factor = (r_A + r_O) / [√2 × (r_B + r_O)]
  - tau ≈ 1.0 is ideal; 0.8–1.0 typically permits stable perovskite
- `mu` (float): Octahedral factor = r_B / r_O
  - mu ≈ 0.41–0.73 for stable octahedral tilting
- `radius_ratio` (float): r_A / r_B ratio
- `en_diff` (float): B-site minus A-site electronegativity difference

---

## Data Generating Process

The data was retrieved from the Materials Project REST API (`/materials/summary` endpoint) using the `mp-api` Python client (v0.46.0). For each A–B–O chemical system in the defined element lists, all materials containing exactly those three elements were queried. Each returned document represents a DFT-optimized crystal structure with computed properties from the Materials Project 2024.1 dataset.

Stoichiometry was verified post-download: only formulas with exactly 1 A-site atom, 1 B-site atom, and 3 oxygen atoms per formula unit were retained. Duplicates (multiple DFT relaxations of the same composition) were removed by `material_id`.

---

## Known Properties and Caveats

- **Elastic data sparsity**: Only 215/1283 materials (17%) have computed elastic constants. This is the primary gap — most perovskites in the database lack mechanical property data.
- **DFT accuracy**: Band gaps from PBE functional are systematically underestimated; formation energies have typical errors of ±50 meV/atom.
- **Metastability**: 1125/1283 materials have energy_above_hull > 0, indicating metastable or unstable phases relative to the convex hull. This is not an error — it reflects the breadth of computed perovskite configurations.
- **Theoretical vs. experimental**: All entries are from DFT calculations; no experimental property measurements are included.
- **Pugh ratio nulls**: pugh_ratio = NaN wherever K_VRH is null (215 non-null entries).
- **A_period / B_period**: Period attribute is not available in the current pymatgen Element API — these columns are null for all entries.

---

## Suggested Analyses

The dataset supports a range of research directions including but not limited to:

- Prediction of elastic/mechanical properties for the 1068 materials without K_VRH or G_VRH
- Stability classification (stable vs. metastable) based on chemical and structural features
- Band gap prediction or optoelectronic property mapping across the perovskite space
- Magnetic ground state prediction from composition and structure
- Machine learning surrogate models for DFT-computed property surfaces
- Exploration of composition–structure–property relationships in ABO3 perovskites

The dataset is intentionally described without prioritizing any specific direction.