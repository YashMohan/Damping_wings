# Damping Wings — Ly-α Reionization Analysis Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-ApJ%202025-orange)](https://doi.org/REPLACE_WITH_DOI)

A Python pipeline for simulating Lyman-alpha damping wing profiles and constraining reionization-epoch parameters using Fisher Information Matrix analysis. Developed as part of research published in The Astrophysical Journal (2025).

---

## What This Does

During the Epoch of Reionization (z ~ 6–8), the intergalactic medium transitions from neutral to ionized hydrogen. Quasar spectra from this era carry imprints of this process in the form of **Lyman-alpha damping wings** — characteristic absorption features that encode information about the neutral hydrogen fraction and reionization topology.

This pipeline:

1. **Generates ionized simulation boxes** using semi-numerical simulations (via 21cmFAST), parameterised by neutral hydrogen fraction (x_HI), minimum halo mass (M_min), quasar lifetime (t_q), and quasar host mass (M_qso)
2. **Calculates damping wing profiles** along simulated sightlines through the IGM, treating dark matter halos as quasar hosts
3. **Constructs observables** — median damping wing profiles and sightline-to-sightline scatter — from ensembles of simulated quasar spectra
4. **Derives parameter constraints** using Fisher Information Matrix analysis, quantifying how well a given survey can constrain reionization parameters

![Ionized box with 50% neutral hydrogen fraction](xh_50.png)

### Key Result

A sample of **64 quasars at redshift z = 7** can constrain:
- Neutral hydrogen fraction x_HI to **2%**
- Minimum halo mass M_min to **0.53 dex**
- Quasar lifetime t_q to **0.12 dex**
- Quasar host mass M_qso to **0.32 dex**

This constraining power is **comparable to predictions for current 21 cm radio experiments**, achieved from standard optical quasar spectra alone.

---

## Papers

If you use this code, please cite:

- Y. M. Sharma et al., *"Behavior of the Ly-α Damping Wings as a Function of Reionization Topology"*, The Astrophysical Journal, 2025. [DOI: REPLACE]
- Y. M. Sharma et al., *"Constraining Ly-α Damping Wings Using Fisher Matrix"*, The Astrophysical Journal, 2025. [DOI: REPLACE]

---

## Pipeline Architecture

The pipeline is built around an OOP-based architecture for efficient batch simulation:

```
Parameters (parameters_file.py)
    ↓
Ionized Box Generation (generating_ionized_boxes.py)
    ↓
Sightline Calculation (calculating_skewers.py)
    ↓
Damping Wing Profiles (damping_wings.py)
    ↓
Fisher Matrix Analysis → Parameter Constraints
```

**Key modules:**
- `generating_ionized_boxes.py` — OOP pipeline for batch generation of 21cmFAST simulation boxes across parameter grids
- `damping_wings.py` — computes Ly-α damping wing transmission profiles along IGM sightlines
- `calculating_skewers.py` — generates sightlines through simulation boxes, handling both random and fixed halo positions
- `calculating_m_pixels.py` — pixel mass calculation utilities
- `optimized_code_running_models.py` — parallel execution manager for large parameter sweeps
- `config/parameters_file.py` — parameter grid definitions and ranges
- `config/constants.py` — physical constants, box size, sightline settings

---

## Installation

### Prerequisites

```bash
# Install 21cmFAST (semi-numerical reionization simulation)
pip install 21cmFAST
# Full installation guide: https://github.com/21cmfast/21cmFAST

# Install HMFcalc (halo mass function)
pip install HMFcalc
# Full installation guide: https://github.com/halomod/HMFcalc
```

### Install Dependencies

```bash
git clone https://github.com/YashMohan/Damping_wings.git
cd Damping_wings
pip install -r requirements.txt
```

### Configure

Edit `config/constants.py` and set `newpath` to your desired output directory:

```python
newpath = "/your/output/directory/"
```

---

## Usage

### Running the Full Pipeline

```bash
python optimized_code_running_models.py
```

### Configuring Parameters

**To change the parameter grid** (which parameters to vary and their ranges):
```python
# Edit config/parameters_file.py
# Modify the parameter list and value ranges
```

**To change corner model parameters:**
```python
# Edit Param_Ranges dictionary in optimized_code_running_models.py
```

**To change face model parameters:**
```python
# Edit variables_list in optimized_code_running_models.py
```

### Halo Field Modes

The pipeline supports two halo placement modes:
- `Halo_field_off/` — halos placed randomly within the simulation volume
- `Halo_field_on/` — halos placed at physically motivated positions

---

## Example Output

![Damping Wing Profiles](results/damping_wing_profiles_z7.png)

*Comprehensive quantile plots of damping wing profiles at z = 7, showing median profiles and sightline-to-sightline scatter (ΔSW₆₈) for varying reionization parameters.*

---

## Physical Context for Non-Specialists

The methodology in this pipeline — **constraining parameters of a physical system from noisy observational data using information-theoretic methods** — is directly analogous to problems in:

- **Industrial surrogate modelling** — fast emulators replacing expensive simulations
- **Sensor fusion** — inferring system state from multiple noisy measurements
- **Uncertainty quantification** — calibrated parameter bounds from limited data

The Fisher Information Matrix approach used here provides the theoretical lower bound on parameter uncertainty (Cramér-Rao bound), making it a principled tool for experimental design and data analysis in any domain.

---

## Requirements

See `requirements.txt`. Key dependencies:

- Python 3.8+
- 21cmFAST
- NumPy, SciPy, Matplotlib
- tqdm, tabulate

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

**Yash Mohan Sharma**
Postdoctoral Researcher, Max Planck Institute for Astronomy
yashmohansharma96@gmail.com · [LinkedIn](https://linkedin.com/in/thisisyashmohan) · [GitHub](https://github.com/YashMohan)
