MRPA-II

Multi-Resolution Prosodic Analysis II

MRPA-II is a research project investigating whether prosodic structure can be discovered directly from acoustic signals without assuming predefined linguistic categories such as Pitch Accent, Accentual Phrase, Downstep, Tone, Tone Sandhi, or Stress.

Instead of viewing prosody as a collection of language-specific labels, MRPA-II treats prosodic structure as an emergent topology constructed from a small set of universal prosodic primitives.

The long-term goal is the discovery of a Universal Prosodic Topology (UPT).

Vision

MRPA-II is part of a broader research program aimed at modeling language through topology rather than predefined symbolic categories.

Plain Text
GNST
↓
Semantic Topology
 
MRPA
↓
Prosodic Topology
 
LLINGTOPOL
↓
Linguistic Topology
 
 
Semantic Topology
×
Prosodic Topology
↓
Linguistic Topology
 

Within this framework:

GNST explores semantic structure.
MRPA explores prosodic structure.
LLINGTOPOL aims to unify them into a common topological representation of language.
Core Hypotheses

MRPA-II is built upon four hypotheses.

H1. Prosodic structure emerges from variable-length nodes

Prosodic organization is not fundamentally composed of predefined units such as syllables, accent phrases, or intonation phrases.

Instead, it emerges from variable-length prosodic nodes discovered directly from acoustic signals.

H2. Boundary and Nucleus are more fundamental than labels

MRPA treats the following as primary observations:

Plain Text
Boundary
Nucleus

Linguistic concepts such as:

Pitch Accent
Downstep
Tone
Tone Sandhi
Stress
Accentual Phrase

are considered secondary interpretations.

H3. ScaleProfile provides a universal coordinate system

Prosodic organization exists across multiple temporal scales.

MRPA represents those scales through a continuous ScaleProfile rather than predefined linguistic categories.

H4. Topology precedes interpretation

Prosodic topology should be discovered first.

Language-specific analyses should emerge later through language adapters.

Design Principles
Boundary First

Prosodic structure begins with boundary discovery.

Boundaries organize acoustic events into larger structures.

Topology Before Labels

MRPA does not assume predefined linguistic categories during discovery.

Topology comes first.

Labels come later.

Universal Before Language-Specific

Universal primitives are discovered first.

Language-specific interpretations are introduced only when necessary.

Variable-Length Representation

Prosodic units are not assumed to have fixed durations.

They may be:

short
long
nested
hierarchical
Universal Prosodic Primitives

MRPA-II currently assumes four language-independent primitives.

Plain Text
Boundary
Nucleus
ScaleProfile
Recursive Containment

These primitives form the basis of prosodic topology.

Language-specific phenomena are treated as projections from these primitives.

Prosodic Node

The core representational unit is the ProsodicNode.

Python
ProsodicNode {
Boundary
Nucleus
ScaleProfile
 
Children
}

Nodes may recursively contain other nodes.

This allows prosodic structure to emerge across multiple temporal resolutions.

Current Implemented Pipeline

The following components are currently implemented.

Plain Text
Audio
↓
F0 Extraction
↓
F0 Interpolation
↓
Continuous Wavelet Transform
↓
ScaleProfile Extraction

Current outputs:

Plain Text
F0 Curves
CWT Representations
Scale Profiles
Planned Pipeline

The next stages of development are:

Plain Text
Audio
↓
F0 Extraction
↓
F0 Interpolation
↓
Continuous Wavelet Transform
↓
ScaleProfile
↓
Boundary Discovery
↓
Nucleus Discovery
↓
Variable-Length Prosodic Nodes
↓
Prosodic Topology
↓
Universal Prosodic Topology
Installation
System Requirements

Tested environment:

Plain Text
Ubuntu 22.04+
Python 3.11+

Recommended:

Plain Text
CPU: 8 cores or more
RAM: 16 GB+
Storage: 20 GB+

GPU is currently not required.

Clone Repository
Shell
git clone https://github.com/hajimewada/mrpa-II.git
 
cd mrpa-II
Create Virtual Environment

Linux / macOS

Shell
python -m venv .venv
 
source .venv/bin/activate

Windows PowerShell

PowerShell
python -m venv .venv
 
.venv\Scripts\Activate.ps1

Verify Python version:

Shell
python --version

Expected:

Plain Text
Python 3.11.x
Install Dependencies
Shell
pip install --upgrade pip
 
pip install -r requirements.txt

Verify installation:

Shell
python -c "import numpy, scipy, matplotlib"

No errors should be reported.

Repository Structure
Plain Text
mrpa-II/
├── analysis/
├── docs/
├── external/
├── data/
├── figures/
├── requirements.txt
└── README.md
Dataset Setup
Download JSUT

Download JSUT v1.1 from:

Plain Text
https://sites.google.com/site/shinnosuketakamichi/publication/jsut

Extract it into:

Plain Text
data/jsut/raw/jsut_ver1.1/

Expected structure:

Plain Text
data/
└── jsut/
└── raw/
└── jsut_ver1.1/
├── basic5000/
├── travel1000/
├── repeat500/
├── onomatopee300/
└── ...

Verify:

Shell
ls data/jsut/raw/jsut_ver1.1/basic5000/wav | head

Expected output:

Plain Text
BASIC5000_0001.wav
BASIC5000_0002.wav
...
Quick Start
Step 1: Extract F0
Shell
python analysis/f0/extract_real_f0.py \
--input data/jsut/raw/jsut_ver1.1/basic5000/wav/BASIC5000_0459.wav \
--output data/jsut/f0/BASIC5000_0459_f0.npy

Expected:

Plain Text
saved: BASIC5000_0459_f0.npy
shape: (329,)
voiced frames: 208
Step 2: Interpolate F0
Shell
python analysis/f0/interpolate_f0.py \
--input data/jsut/f0/BASIC5000_0459_f0.npy \
--output data/jsut/f0/BASIC5000_0459_f0_interp.npy

Purpose:

Plain Text
Generate a continuous F0 contour by filling unvoiced gaps.
Step 3: Compute CWT
Shell
python analysis/f0/extract_cwt.py \
--input data/jsut/f0/BASIC5000_0459_f0_interp.npy \
--output data/jsut/cwt/BASIC5000_0459_cwt.npz

Expected output:

Plain Text
input shape : (329,)
cwt shape : (64, 329)
 
Step 4: Extract ScaleProfile
Shell
python analysis/f0/scale_energy.py \
--input data/jsut/cwt/BASIC5000_0459_cwt.npz

Example:

Plain Text
Peak:
Scale = 64
Freq = 0.01270
Energy = 84.33
Step 5: Plot F0
Shell
python analysis/f0/plot_f0.py \
--input data/jsut/f0/BASIC5000_0459_f0.npy
 

Output:

Plain Text
figures/BASIC5000_0459_f0.png
Step 6: Plot Scale Energy
Shell
python analysis/f0/plot_scale_energy.py \
--input data/jsut/cwt/BASIC5000_0459_cwt.npz

Output:

Plain Text
figures/BASIC5000_0459_scale_energy.png
Research Roadmap
MRPA-I ✅
Prosodic Spectral Analysis

Achievements:

Artificial prosody generation
Frequency sweep experiments
Scale-frequency correspondence
Real JSUT F0 extraction
Continuous Wavelet Transform analysis
ScaleProfile extraction

Output:

Plain Text
ScaleProfile
MRPA-II 🚧
Prosodic Primitive Discovery

Objectives:

Plain Text
Boundary Discovery
↓
Nucleus Discovery
↓
ScaleProfile Assignment

Output:

Plain Text
Boundary
Nucleus
ScaleProfile

Research Questions:

What defines a prosodic boundary?
What defines a prosodic nucleus?
Can prosodic primitives emerge without linguistic labels?
MRPA-III
Variable-Length Prosodic Nodes

Objectives:

Plain Text
Boundary
+
Nucleus
+
ScaleProfile
↓
ProsodicNode

Output:

Plain Text
Variable-Length Prosodic Nodes
MRPA-IV
Prosodic Topology

Objectives:

Plain Text
Node Similarity
↓
Node Graph
↓
Prosodic Topology

Output:

Plain Text
Prosodic Topology
Prosodic Manifold
MRPA-V
Universal Prosodic Topology

Cross-linguistic validation:

Japanese
Mandarin
Korean
English

Goal:

Determine whether

Plain Text
Boundary
Nucleus
ScaleProfile

serve as universal prosodic primitives.

MRPA-VI
GNST and LLINGTOPOL Integration

Objectives:

Plain Text
Semantic Topology
×
Prosodic Topology
↓
Linguistic Topology

Applications:

Speech-language modeling
Topological language representations
Boundary prediction
Multiscale linguistic analysis
Current Status

Implemented:

Plain Text
✅ JSUT integration
✅ F0 extraction
✅ F0 interpolation
✅ Continuous Wavelet Transform
✅ ScaleProfile extraction
✅ Frequency sweep experiments
✅ Real utterance analysis

In Progress:

Plain Text
🚧 Boundary discovery
🚧 Nucleus discovery
🚧 Variable-length node construction
🚧 ScaleProfile assignment
🚧 Prosodic topology construction
Future Direction

MRPA is expected to provide the prosodic foundation for future topology-based routing models.

One possible direction is the development of:

Plain Text
DBTM
(Boundary-Driven Topological Model)

where prosodic boundaries dynamically influence information routing and structural organization.

DBTM is intentionally treated as future work and is separate from the core objectives of MRPA-II.

Long-Term Goal

The ultimate objective of MRPA is the discovery of a Universal Prosodic Topology and its integration with semantic topology.

Plain Text
Semantic Topology
×
Prosodic Topology
↓
Linguistic Topology

Under this view, linguistic structure is not predefined.

It emerges from topological organization discovered directly from speech and language signals.
