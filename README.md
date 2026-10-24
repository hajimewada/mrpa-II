# mrpa-II: Prosodic Node Discovery
Summary:
MRPA-II: Prosodic Node Discovery MRPA-II shifts the focus from prosodic spectrum analysis to prosodic structure discovery.  Rather than treating Accent, Accentual Phrase, or Downstep as predefined linguistic categories, MRPA-II attempts to discover prosodic structure directly from F0 dynamics.


MRPA-II shifts the focus from prosodic spectrum analysis to prosodic structure discovery.

Rather than treating Accent, Accentual Phrase, or Downstep as predefined linguistic categories, MRPA-II attempts to discover prosodic structure directly from F0 dynamics.

The central hypothesis is that prosodic structure can be represented as a hierarchy of variable-length nodes, where each node consists of:

ProsodicNode {
Boundary,
Nucleus,
ScaleProfile,
Children
}

Key Ideas
Traditional approaches assume fixed linguistic units and then analyze prosodic behavior within them.

MRPA-II reverses this process:

F0 -> Prosodic Surprise -> Boundary Discovery -> Variable-Length Node Induction -> ScaleProfile Assignment -> Recursive Prosodic Structure

Nodes are therefore induced from acoustic evidence rather than predefined linguistic labels.

Boundary Discovery
Instead of fixed windows, boundaries are inferred from unexpected changes in prosodic behavior.

Candidate approaches include:

Raw F0 slope change
F0 curvature change
Prosodic surprise
ScaleProfile transition
The long-term objective is to define boundaries using prosodic information surprise:

Boundary(t)
−log P(F0_t | Context)

This formulation is intended to support cross-language generalization.

Nucleus Discovery
Each node contains a nucleus representing its most prominent prosodic location.

Possible cues include:

Local F0 prominence
Energy prominence
Duration prominence
ScaleProfile concentration
The nucleus serves as the prosodic analogue of a semantic center.

ScaleProfile
Each node is assigned a multi-resolution ScaleProfile derived from CWT analysis.

Unlike earlier phases, scales are not assigned predefined meanings.

MRPA-II treats ScaleProfile as a universal prosodic coordinate representation.

Language-specific interpretations such as:

Accent
Downstep
Tone
Tone Sandhi
Accentual Phrase
are introduced later by language adapters.

Research Goal
The goal of MRPA-II is to discover Prosodic Nodes that are:

variable-length
recursively composable
language-independent
and suitable for constructing a Universal Prosodic Topology.

Expected Outcome
MRPA-II produces:

Boundary candidates
Nucleus candidates
ScaleProfiles
Prosodic Nodes
which form the basis of MRPA-III: Prosodic Topology.



Quick Start
1. Create Environment
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
2. Download JSUT
Download JSUT v1.1 and extract it under:

data/jsut/raw/jsut_ver1.1/
Expected structure:

data/
└── jsut/
    └── raw/
        └── jsut_ver1.1/
            ├── basic5000/
            ├── travel1000/
            └── ...
3. Extract F0
python analysis/f0/extract_real_f0.py \
  --input \
  data/jsut/raw/jsut_ver1.1/basic5000/wav/BASIC5000_0459.wav \
  --output \
  data/jsut/f0/BASIC5000_0459_f0.npy
Example output:

saved: BASIC5000_0459_f0.npy
shape: (329,)
voiced frames: 208
4. Interpolate F0
python analysis/f0/interpolate_f0.py \
  --input \
  data/jsut/f0/BASIC5000_0459_f0.npy \
  --output \
  data/jsut/f0/BASIC5000_0459_f0_interp.npy
5. Compute CWT
python analysis/f0/extract_cwt.py \
  --input \
  data/jsut/f0/BASIC5000_0459_f0_interp.npy \
  --output \
  data/jsut/cwt/BASIC5000_0459_cwt.npz
Expected output:

input shape : (329,)
cwt shape   : (64, 329)
6. Compute Scale Profile
python analysis/f0/scale_energy.py \
  --input \
  data/jsut/cwt/BASIC5000_0459_cwt.npz
Example output:

Peak: Scale=64
Freq=0.01270
Energy=84.33
7. Visualize F0
python analysis/f0/plot_f0.py \
  --input \
  data/jsut/f0/BASIC5000_0459_f0.npy
8. Visualize Scale Profile
python analysis/f0/plot_scale_energy.py \
  --input \
  data/jsut/cwt/BASIC5000_0459_cwt.npz
Current Experimental Pipeline
JSUT
 ↓
F0 Extraction
 ↓
F0 Interpolation
 ↓
Continuous Wavelet Transform
 ↓
Scale Profile
 ↓
Boundary Discovery
 ↓
Prosodic Node Discovery
 ↓
Prosodic Topology


Current Status
✅ Real F0 extraction from JSUT

✅ Continuous Wavelet Transform analysis

✅ Scale Profile extraction

✅ Phrase frequency sweep experiments

✅ Real utterance analysis

🚧 Surprise-based Boundary Discovery

🚧 Variable-Length Prosodic Nodes

🚧 Prosodic Topology Construction
