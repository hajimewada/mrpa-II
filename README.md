# MRPA-II

**Multi-Resolution Prosodic Analysis II**

MRPA-II is a research project investigating whether prosodic structure
can be discovered directly from acoustic signals without assuming
predefined linguistic categories such as Pitch Accent, Accentual Phrase,
Downstep, Tone, Tone Sandhi, or Stress.

Instead of viewing prosody as a collection of language-specific labels,
MRPA-II treats prosodic structure as an emergent topology constructed
from a small set of universal prosodic primitives.

The long-term goal is the discovery of a
**Universal Prosodic Topology (UPT)**.

## Vision

MRPA-II is part of a broader research program aimed at modeling language
through topology rather than predefined symbolic categories.

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

Semantic Topology × Prosodic Topology
                  ↓
        Linguistic Topology
```

Within this framework:

- GNST explores semantic structure.
- MRPA explores prosodic structure.
- LLINGTOPOL aims to unify them into a common topological representation
  of language.

## Core Hypotheses

MRPA-II is based on four central hypotheses.

### H1. Prosodic structure emerges from variable-length nodes

Prosodic organization is not assumed to be fundamentally composed of
predefined units such as syllables, accentual phrases, or intonation
phrases.

Instead, MRPA-II investigates whether prosodic organization can emerge
from variable-length nodes discovered directly from acoustic signals.

### H2. Boundary and Nucleus are more fundamental than labels

MRPA-II treats the following as candidate universal primitives:

```text
Boundary
Nucleus
```

Language-specific concepts such as the following are treated as
secondary interpretations:

- Pitch Accent
- Downstep
- Tone
- Tone Sandhi
- Stress
- Accentual Phrase

### H3. ScaleProfile provides a shared coordinate system

Prosodic organization operates across multiple temporal scales.

MRPA-II represents these scales through a continuous `ScaleProfile`
rather than a predefined hierarchy of linguistic categories.

The working hypothesis is that language-specific prosodic systems may
occupy different regions or configurations within a shared scale space.

### H4. Topology precedes linguistic interpretation

Prosodic topology should first be discovered from acoustic structure.

Language-specific interpretations should be introduced later through
language adapters rather than being built into the initial discovery
process.

## Design Principles

### Boundary First

Prosodic structure begins with the discovery of acoustic changes that
may function as boundaries.

The model should not assume in advance that every detected change
corresponds to a traditional linguistic boundary.

### Topology Before Labels

MRPA-II does not assign predefined linguistic labels during the initial
discovery stage.

Topology is constructed first. Linguistic interpretation follows.

### Universal Before Language-Specific

The initial representation should be language-independent whenever
possible.

Language-specific concepts are introduced through adapters only after
the shared representation has been evaluated.

### Variable-Length Representation

Prosodic units are not assumed to have fixed durations.

Candidate nodes may be:

- short or long;
- adjacent or nested;
- locally defined;
- recursively organized.

## Universal Prosodic Primitives

MRPA-II currently investigates four candidate language-independent
primitives:

```text
Boundary
Nucleus
ScaleProfile
Recursive Containment
```

These primitives are intended to provide the basis for constructing a
prosodic topology.

Language-specific phenomena are treated as possible projections or
interpretations of configurations within that topology.

## Prosodic Node

The proposed core representational unit is the `ProsodicNode`.

```python
@dataclass
class ProsodicNode:
    start: float
    end: float
    boundary: object
    nucleus: object
    scale_profile: object
    children: list["ProsodicNode"]
```

This is a conceptual interface rather than a finalized implementation.

A node may recursively contain other nodes, allowing prosodic structure
to be represented across multiple temporal resolutions.

## Research Pipeline

### Implemented research components

The preceding MRPA work has investigated the following processing
stages:

```text
Audio
  ↓
F0 Extraction
  ↓
F0 Interpolation
  ↓
Continuous Wavelet Transform
  ↓
ScaleProfile Analysis
```

These stages provide the experimental foundation for MRPA-II.

### Planned MRPA-II pipeline

MRPA-II extends the preceding analysis toward primitive, node, and
topology discovery:

```text
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
Cross-Linguistic Validation
```

The distinction between implemented components and planned components is
intentional. MRPA-II is currently at the design and initial
implementation stage.

## Repository Status

The repository currently contains the project definition and research
roadmap.

```text
mrpa-II/
├── README.md
└── .git/
```

The source-code structure described below is planned and will be added
incrementally as the implementation proceeds.

## Planned Repository Structure

```text
mrpa-II/
├── analysis/
│   ├── f0/
│   │   ├── extract_real_f0.py
│   │   ├── interpolate_f0.py
│   │   ├── plot_f0.py
│   │   ├── extract_cwt.py
│   │   ├── scale_energy.py
│   │   └── plot_scale_energy.py
│   ├── boundary/
│   ├── nucleus/
│   ├── nodes/
│   └── topology/
├── docs/
├── external/
├── data/
│   └── jsut/
│       ├── raw/
│       ├── f0/
│       ├── cwt/
│       └── results/
├── figures/
├── tests/
├── requirements.txt
└── README.md
```

Generated datasets and intermediate analysis files should not normally be
committed to Git.

## Installation

> [!IMPORTANT]
> The installation and execution commands in this section describe the
> planned executable repository. They will become operational as the
> corresponding source files and `requirements.txt` are added.

### 1. System requirements

Planned baseline environment:

- Linux or macOS
- Python 3.11 or later
- Git
- A Python virtual environment

A GPU is not expected to be required for the initial F0 and CWT analysis.

### 2. Clone the repository

```bash
git clone https://github.com/hajimewada/mrpa-II.git
cd mrpa-II
```

Confirm the repository state:

```bash
git status
git log -1 --oneline --decorate
```

### 3. Check the Python installation

```bash
python3 --version
```

The planned baseline is Python 3.11 or later.

If the `python` command already points to the intended Python
installation, it may be used instead of `python3`.

### 4. Create a virtual environment

#### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

After activation, verify the interpreter:

```bash
python --version
python -c "import sys; print(sys.executable)"
```

The executable path should point to the `.venv` directory.

### 5. Upgrade the Python packaging tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 6. Install project dependencies

Once `requirements.txt` has been added:

```bash
python -m pip install -r requirements.txt
```

A planned initial dependency set may include packages for:

- numerical array processing;
- scientific signal processing;
- audio loading