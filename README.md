# MRPA-II

## Multi-Resolution Prosodic Analysis II

MRPA-II is a research project investigating whether prosodic structure can be
discovered directly from acoustic signals without assuming predefined linguistic
categories such as Pitch Accent, Accentual Phrase, or Downstep.

Instead of assigning linguistic labels to scales, MRPA-II treats prosodic
structure as a topology emerging from:

- Boundary
- Nucleus
- ScaleProfile
- Recursive Containment

The long-term objective is the construction of a Universal Prosodic Topology
(UPT).

## Vision

MRPA-II is part of a larger research program:

```text
GNST
 ↓
Semantic Topology

MRPA
 ↓
Prosodic Topology

LLINGTOPOL
 ↓
Linguistic Topology
 =
Semantic Topology
 ×
Prosodic Topology
```

## Core Hypothesis

Prosody is not a set of predefined labels. Instead, it emerges from a hierarchy
of variable-length prosodic nodes.

```text
ProsodicNode {
    Boundary
    Nucleus
    ScaleProfile
    Children
}
```

Language-specific concepts such as the following are interpreted later through
language adapters:

- Accent
- Downstep
- Tone
- Tone Sandhi
- Accentual Phrase

## Universal Prosodic Topology

MRPA-II assumes the following language-independent representation:

- Boundary
- Nucleus
- ScaleProfile
- Recursive Containment

No predefined labels are assigned to scales. `ScaleProfile` is treated as a
universal coordinate system.

## Current Pipeline

```text
Audio
 ↓
F0 Extraction
 ↓
F0 Interpolation
 ↓
Continuous Wavelet Transform (CWT)
 ↓
ScaleProfile
 ↓
Boundary Discovery
 ↓
Variable-Length Node Discovery
 ↓
Prosodic Topology
```

## Quick Start

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download JSUT

Download JSUT v1.1 and extract it under `data/jsut/raw/jsut_ver1.1/`.

Expected structure:

```text
data/
└── jsut/
    └── raw/
        └── jsut_ver1.1/
            ├── basic5000/
            ├── travel1000/
            ├── repeat500/
            └── ...
```

### 3. Extract F0

```bash
python analysis/f0/extract_real_f0.py \
  --input \
  data/jsut/raw/jsut_ver1.1/basic5000/wav/BASIC5000_0459.wav \
  --output \
  data/jsut/f0/BASIC5000_0459_f0.npy
```

Example output:

```text
saved: BASIC5000_0459_f0.npy
shape: (329,)
voiced frames: 208
```

### 4. Interpolate F0

```bash
python analysis/f0/interpolate_f0.py \
  --input \
  data/jsut/f0/BASIC5000_0459_f0.npy \
  --output \
  data/jsut/f0/BASIC5000_0459_f0_interp.npy
```

### 5. Compute CWT

```bash
python analysis/f0/extract_cwt.py \
  --input \
  data/jsut/f0/BASIC5000_0459_f0_interp.npy \
  --output \
  data/jsut/cwt/BASIC5000_0459_cwt.npz
```

Example output:

```text
input shape : (329,)
cwt shape   : (64, 329)
```

### 6. Extract Scale Profile

```bash
python analysis/f0/scale_energy.py \
  --input \
  data/jsut/cwt/BASIC5000_0459_cwt.npz
```

Example output:

```text
Peak: Scale=64
Freq=0.01270
Energy=84.33
```

### 7. Plot F0

```bash
python analysis/f0/plot_f0.py \
  --input \
  data/jsut/f0/BASIC5000_0459_f0.npy
```

### 8. Plot Scale Energy

```bash
python analysis/f0/plot_scale_energy.py \
  --input \
  data/jsut/cwt/BASIC5000_0459_cwt.npz
```

## Research Phases

### MRPA-I ✅ — Prosodic Spectral Analysis

Achievements:

- Artificial prosody generation
- Frequency sweep experiments
- Scale-frequency correspondence
- Real JSUT F0 extraction
- CWT scale analysis

Output: `ScaleProfile`

### MRPA-II 🚧 — Prosodic Node Discovery

Objectives:

1. Boundary Discovery
2. Nucleus Discovery
3. Variable-Length Prosodic Nodes
4. ScaleProfile Assignment

Research questions:

- What defines a prosodic boundary?
- What defines a prosodic nucleus?
- Can prosodic nodes emerge without linguistic labels?

### MRPA-III — Prosodic Topology

Objectives:

1. Node Similarity
2. Node Graph
3. Prosodic Topology

Output: Prosodic Manifold

### MRPA-IV — Universal Prosodic Topology

Objectives:

- Cross-linguistic validation in Japanese, Mandarin, and Korean
- Establish Boundary, Nucleus, and ScaleProfile as universal prosodic primitives

### MRPA-V — GNST Integration

Integrate Semantic Topology and Prosodic Topology. Boundary prediction becomes:

```text
BoundaryScore = f(SemanticCoordinate, ProsodicCoordinate)
```

### MRPA-VI — LLINGTOPOL Integration

Linguistic Topology is defined as:

```text
Linguistic Topology = Semantic Topology × Prosodic Topology
```

## Repository Structure

```text
analysis/
docs/
external/

data/
├── jsut/
│   ├── raw/
│   ├── f0/
│   ├── cwt/
│   └── results/

figures/
```

## Current Status

- ✅ JSUT integration
- ✅ F0 extraction
- ✅ F0 interpolation
- ✅ Continuous Wavelet Transform
- ✅ Scale Profile extraction
- ✅ Frequency sweep experiments
- ✅ Real utterance analysis
- 🚧 Surprise-based Boundary Discovery
- 🚧 Variable-Length Prosodic Nodes
- 🚧 Node ScaleProfile Learning
- 🚧 Prosodic Topology Construction

## Long-Term Goal

MRPA aims to discover a Universal Prosodic Topology and integrate it with
LLINGTOPOL:

```text
Semantic Topology
×
Prosodic Topology
=
Linguistic Topology
```

Linguistic structure should emerge from topology rather than predefined
symbolic categories.
