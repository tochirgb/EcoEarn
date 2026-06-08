# CGCNN for Materials Property Prediction

A Crystal Graph Neural Network (CGCNN) trained from scratch to predict 
material properties — formation energy and band gap — directly from 
crystal structures, using data from the Materials Project.

Built as an independent research project by a first-year Metallurgical 
and Materials Engineering student at PEC Chandigarh.

---

## What this does

- Pulls ~10,000 real crystal structures from the Materials Project API
- Represents each crystal as a graph (atoms as nodes, bonds as edges 
  with Gaussian-expanded distance features within an 8 Å cutoff)
- Trains a ~1.2M parameter CGCNN to predict formation energy and band gap
- Uses the trained model as a fast DFT surrogate to screen 20,000 
  candidate oxides for stability and target band gaps — in seconds 
  instead of days

**Test MAE: 0.05–0.10 eV/atom on formation energy**

---

## Stack

- Python
- PyTorch + PyTorch Geometric
- pymatgen
- mp-api (Materials Project API)
- Google Colab (T4 GPU)

---

## Model Architecture

Atom embedding → 4× CGConv layers with batch normalization 
→ Global mean pooling → MLP → property prediction

Trained for 120 epochs using Adam optimizer with 
ReduceLROnPlateau scheduling and MSE loss.

---

## Why this matters

Running DFT simulations on thousands of candidate materials takes days. 
This model acts as a surrogate — predicting stability and electronic 
properties in milliseconds per structure, enabling high-throughput 
screening at a scale that traditional simulation cannot match.

---

## Limitations

- Predicts DFT-level band gaps (which systematically underestimate 
  experimental values by ~30–50%)
- Struggles with amorphous materials and large unit cells (50+ atoms)
- For higher accuracy: ALIGNN, CHGNet, or M3GNet are better choices

---

## References

- Xie & Grossman, 2018 — original CGCNN paper
- Materials Project (materialsproject.org)
