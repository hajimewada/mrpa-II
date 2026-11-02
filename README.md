MRPA-II

Multi-Resolution Prosodic Analysis II

MRPA-II is a research project investigating whether prosodic structure can be discovered directly from acoustic signals without assuming predefined linguistic categories such as Pitch Accent, Accentual Phrase, Downstep, Tone, or Stress.

Rather than treating prosody as a collection of language-specific labels, MRPA-II explores whether prosodic organization emerges from a small set of universal prosodic primitives and their topological relationships.

The long-term goal is the construction of a Universal Prosodic Topology (UPT).

Vision

MRPA-II is part of a broader research program aimed at representing linguistic structure through topology rather than predefined symbolic categories.

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

MRPA provides the prosodic component of this framework.

Core Hypotheses

MRPA-II is built on four central hypotheses.

H1. Prosodic structure emerges from variable-length nodes

Prosodic organization is not fundamentally composed of predefined units such as syllables, accent phrases, or intonation phrases.

Instead, it emerges from variable-length prosodic nodes discovered directly from acoustic structure.

H2. Boundaries and nuclei are more fundamental than labels

Traditional linguistic categories may be secondary interpretations of more primitive prosodic phenomena.

MRPA therefore treats:

Boundary
Nucleus

as primary observations.

Labels such as:

Pitch Accent
Downstep
Tone
Stress
Accentual Phrase

are introduced only as language-specific interpretations.

H3. ScaleProfile provides a shared coordinate system

Prosodic organization operates across multiple temporal scales.

MRPA represents this structure through a continuous ScaleProfile rather than a predefined hierarchy of categories.

Different languages may occupy different regions within the same scale space.

H4. Topology comes before linguistic interpretation

Prosodic topology should be discovered first.

Language-specific analyses are applied later through language adapters.

Design Principles
Boundary First

Prosodic structure begins with boundary discovery.

Boundaries determine how acoustic events are organized into larger structures.

Topology Before Labels

MRPA does not assume linguistic categories during discovery.

Topology is learned before assigning labels.

Universal Before Language-Specific

The initial representation should be language-independent whenever possible.

Language-specific concepts are introduced through adapters.

Variable-Length Representation

Prosodic units are not assumed to have fixed duration.

Nodes may be short, long, nested, or recursively organized.

Universal Prosodic Primitives

MRPA-II assumes four language-independent primitives.

Plain Text
Boundary
Nucleus
ScaleProfile
Recursive Containment

These primitives serve as the basis for constructing prosodic topology.

Language-specific phenomena such as:

Pitch Accent
Downstep
Tone
Tone Sandhi
Stress
Accentual Phrase

are treated as projections from these primitives.

Prosodic Node

The core representational unit of MRPA-II is the ProsodicNode.

Python
ProsodicNode:
{
Boundary
Nucleus
ScaleProfile
 
Children
}

A ProsodicNode may recursively contain other ProsodicNodes.

This allows prosodic structure to emerge naturally across multiple temporal resolutions.

Current Implemented Pipeline

The following components are currently implemented.

Plain Text
Audio
↓
F0 Extraction
↓
F0 Interpolation
↓
Continuous Wavelet Transform (CWT)
↓
ScaleProfile Extraction

Current repository outputs:

Plain Text
F0 Curves
CWT Representations
ScaleProfile Measurements
Planned Pipeline

The next stages of MRPA-II extend the current system toward node discovery and topology construction.

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
Repository Structure
Plain Text
analysis/
docs/
external/
 
data/
├── jsut/
│ ├── raw/
│ ├── f0/
│ ├── cwt/
│ └── results/
 
figures/
Quick Start
1. Environment
Shell
python -m venv .venv
source .venv/bin/activate
 
pip install -r requirements.txt
2. Download JSUT

Download JSUT v1.1 and extract it under:

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
└── ...
3. Extract F0
Shell
python analysis/f0/extract_real_f0.py \
--input data/jsut/raw/jsut_ver1.1/basic5000/wav/BASIC5000_0459.wav \
--output data/jsut/f0/BASIC5000_0459_f0.npy

Example output:

Plain Text
saved: BASIC5000_0459_f0.npy
shape: (329,)
voiced frames: 208
 
4. Interpolate F0
Shell
python analysis/f0/interpolate_f0.py \
--input data/jsut/f0/BASIC5000_0459_f0.npy \
--output data/jsut/f0/BASIC5000_0459_f0_interp.npy
5. Compute CWT
Shell
python analysis/f0/extract_cwt.py \
--input data/jsut/f0/BASIC5000_0459_f0_interp.npy \
--output data/jsut/cwt/BASIC5000_0459_cwt.npz

Example output:

Plain Text
input shape : (329,)
cwt shape : (64, 329)
6. Extract ScaleProfile
Shell
python analysis/f0/scale_energy.py \
--input data/jsut/cwt/BASIC5000_0459_cwt.npz

Example output:

Plain Text
Peak:
Scale = 64
Freq = 0.01270
Energy = 84.33
7. Plot F0
Shell
python analysis/f0/plot_f0.py \
--input data/jsut/f0/BASIC5000_0459_f0.npy
8. Plot Scale Energy
Shell
python analysis/f0/plot_scale_energy.py \
--input data/jsut/cwt/BASIC5000_0459_cwt.npz
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

Research Questions:

What defines a prosodic boundary?
What defines a prosodic nucleus?
Can prosodic primitives emerge without linguistic labels?

Output:

Plain Text
Boundary
Nucleus
ScaleProfile
MRPA-III
Variable-Length Prosodic Nodes

Objectives:

Plain Text
Primitive Integration
↓
Node Construction
↓
Recursive Containment

Output:

Plain Text
ProsodicNode
MRPA-IV
Prosodic Topology

Objectives:

Plain Text
Node Similarity
↓
Node Graph
↓
Topology Construction

Output:

Plain Text
Prosodic Topology
Prosodic Manifold
MRPA-V
Universal Prosodic Topology

Objectives:

Cross-linguistic validation across:

Japanese
Mandarin
Korean
English

Research Goal:

Evaluate whether

Plain Text
Boundary
Nucleus
ScaleProfile

function as universal prosodic primitives.

MRPA-VI
GNST and LLINGTOPOL Integration

Objectives:

Plain Text
Semantic Topology
×
Prosodic Topology
↓
Linguistic Topology

Potential applications include:

Boundary prediction
Information routing
Multimodal topology learning
Speech-language integration
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
🚧 Variable-length prosodic nodes
🚧 ScaleProfile assignment to nodes
🚧 Prosodic topology construction
Future Direction

MRPA is expected to provide the prosodic foundation for future topology-based models of linguistic information flow.

One possible direction is the development of Boundary-Driven Topological Models (DBTM), where prosodic boundaries influence routing and structural organization across multiple processing scales.

This direction remains future work and is intentionally separated from the core objectives of MRPA-II.

Long-Term Goal

The ultimate objective of MRPA is to discover a Universal Prosodic Topology and integrate it with semantic topology research.

Plain Text
Semantic Topology
×
Prosodic Topology
↓
Linguistic Topology

Under this view, linguistic structure is not predefined.

It emerges from the interaction of topological structures learned directly from language and speech.
