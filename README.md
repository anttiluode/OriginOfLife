# The Crystallization Hypothesis & Rajapinta-Takens Framework

![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![Status](https://img.shields.io/badge/Status-Theoretical%20%2F%20Experimental-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> *"The inside is multiplicative. The outside is additive. The Rajapinta is the boundary that turns one into the other."*

This repository contains the code, interactive simulators, and the foundational theoretical paper proposing that the origin of
life—and the fundamental nature of neural computation—is best understood as a series of 

**topological crystallization events**

It provides a unified mathematical and computational framework (the **Rajapinta-Takens Operator**) that steps behind biological boundaries
(skulls, cell membranes, Axon Initial Segments) to reconstruct the hidden, multidimensional wave-states of living systems from flat 1D signals.

**Note on Authorship:** The foundational paper in this repo, *The Crystallization Hypothesis*, was written in collaboration with Claude Opus 4 (Anthropic) 
during an extended theoretical exploration of nonlinear dynamics and EEG signal processing.

---

## 📄 The Paper

**[`origin_of_life_paper.md`](origin_of_life_paper.md)** 

Read the full theoretical framework here. It argues that life began not when chemistry became complex, but when a continuous prebiotic oscillatory 
field preserved a stable topological invariant (a winding number) across a crystallization boundary (a lipid membrane).

## 🛠️ The Toolkit (Code & Simulators)

This repository isn't just theory. It contains the literal "Phase-Space Spectrometers" built to prove the math.

### 1. Kernel Recovery Optimization
**File:** [`Kernel_Recovery_Demo.py`](Kernel_Recovery_Demo.py)

*   **What it does:** Proves that information destroyed by a physical boundary is recoverable. It generates a continuous multiplicative wave-state (the Kernel),
*    filters it through a simulated skull, crushes it via a non-linear threshold (the Rajapinta), and then uses Differential Evolution on a 3D Takens embedding to
*    successfully reverse-engineer the original frequencies of the hidden arms.

### 2. The RT Delay Bundle Explorer

**File:** [`index.html`](index.html)

Try it at: https://anttiluode.github.io/OriginOfLife/ 

*   **What it does:** A browser-based interactive 3D topology simulator. It demonstrates how a single 1D signal (like an EEG wire) contains multiple,
*   overlapping physical realities. By tuning three simultaneous Takens delays (Fast, Medium, Slow), users can watch a chaotic 1D signal visually unfold
*    into a pristine, nested 3D torus.

## 🧠 Core Concepts

If you are diving into the code, here is the vocabulary of the framework:

*   **The Kernel ($\mathcal{K}$):** The hidden, continuous, high-dimensional reality. In biology, this operates multiplicatively (e.g., oscillating
*    chemical gradients, or phase-locked dendritic arm chains).
*    
*   **The Rajapinta ($\mathcal{R}$):** The observation boundary. The skull, the electrode, the cell membrane, or the Axon Initial Segment.
*    It acts as a linear filter + a nonlinear threshold that "crystallizes" the continuous wave into a discrete output (a spike, a digital measurement,
*    a genetic token).
*    
*   **Takens Multi-Lens:** Based on Takens' Delay Embedding Theorem. A mathematical telescope that takes the flattened 1D output of the Rajapinta
*   and inflates it back into 3D topology to read the original "winding numbers" of the biological oscillator.

---

## 🚀 Installation & Usage

For the Python scripts:
```bash
pip install numpy scipy matplotlib mne sounddevice
```

Run the Kernel Recovery Demo:

```Bash
python Kernel_Recovery_Demo.py
```

For the 3D Delay Bundle Explorer:
Simply double-click rtdelaybundle.html to open it in any modern web browser (uses WebGL/Three.js).

📜 License

MIT License. See LICENSE for details.

Do what you want with the geometry. The topology belongs to the universe.
