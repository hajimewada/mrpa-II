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

### Implemented pipeline

The current implementation provides a validated and persistent
Continuous Wavelet Transform pipeline:

```text
Continuous F0
  ↓
Input and Time-Axis Validation
  ↓
Interpolation Metadata Validation
  ↓
CWT Signal Preparation
  ↓
Physical Period and Scale Generation
  ↓
Continuous Wavelet Transform
  ↓
CWT Result Validation
  ↓
Metadata Generation and Validation
  ↓
Persistent CWT Representation
```

This pipeline transforms an interpolated continuous-F0 track and its
time axis into a compressed CWT archive and a versioned metadata file.

### Planned MRPA-II pipeline

MRPA-II extends the implemented signal-processing foundation toward
primitive, node, and topology discovery:

```text
Audio
  ↓
F0 Extraction
  ↓
F0 Interpolation
  ↓
Continuous F0 Validation
  ↓
CWT Signal Preparation
  ↓
Physical Scale Generation
  ↓
Continuous Wavelet Transform
  ↓
Persisted CWT Representation
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

The distinction between implemented and planned components is
intentional. The validated CWT foundation is implemented; primitive,
node, and topology discovery remain active research stages.

## Current Implementation Status

### Completed Milestones

#### Milestone 1: Input validation and loading

Implemented functionality includes:

- command-line argument parsing;
- input and output path validation;
- continuous-F0 and time-axis loading;
- array-shape and finite-value validation;
- uniform sampling-period inference;
- optional voicing-mask loading;
- optional interpolation-mask loading;
- interpolation-metadata loading and consistency validation;
- interpolation-ratio validation.

#### Milestone 2: CWT signal preparation

Implemented functionality includes:

- F0-domain conversion;
- configurable signal centering;
- signal scaling infrastructure;
- dynamic-range validation;
- integrated preparation of the signal used for CWT computation.

#### Milestone 3: Physical scale generation

Implemented functionality includes:

- period-range resolution;
- logarithmic and linear period-grid generation;
- period-to-frequency conversion;
- frequency-to-scale conversion;
- physical-scale generation;
- scale and period round-trip validation;
- physical-coordinate summaries.

#### Milestone 4: Validated CWT extraction

Implemented functionality includes:

- CWT signal and scale validation;
- CWT configuration validation;
- continuous wavelet transform computation;
- coefficient-shape validation;
- finite-output validation;
- scale-frequency alignment validation;
- CWT result validation;
- CWT result summarization;
- integrated extraction orchestration;
- command-line integration;
- CLI acceptance testing.

#### Milestone 5: Metadata and output persistence

Implemented functionality includes:

- typed metadata schemas;
- CWT configuration metadata;
- prepared-signal metadata;
- physical-scale metadata;
- CWT result metadata;
- metadata construction;
- metadata validation;
- compressed CWT archive generation;
- JSON metadata persistence;
- CLI persistence integration;
- end-to-end persistence acceptance testing.

## Repository Status

MRPA-II is in active implementation.

The current codebase contains a complete and tested CWT preprocessing
pipeline with validated input handling, signal preparation, physical
scale generation, transform computation, metadata construction, output
persistence, and command-line integration.

Current validation status:

```text
tests/test_extract_cwt.py : 292 passed
full test suite           : 306 passed
```

One known non-blocking warning originates from the installed `pyworld`
package's use of the deprecated `pkg_resources` API.

## Current Output Format

The CWT pipeline produces two primary artifacts:

```text
cwt.npz
cwt_metadata.json
```

If `--metadata-output` is not specified, the metadata filename is
derived from the output archive name.

For example:

```text
cwt.npz
  ↓
cwt_metadata.json
```

### CWT archive

The compressed NumPy archive contains:

```text
coefficients
frequencies
```

`coefficients` is a two-dimensional matrix with the shape:

```text
number_of_scales × number_of_frames
```

`frequencies` is a one-dimensional array aligned with the scale
dimension of the coefficient matrix.

### Metadata file

The JSON metadata document uses a versioned top-level schema:

```text
format_version
cwt_configuration
signal_summary
scale_summary
cwt_summary
```

The metadata records:

- wavelet configuration;
- sampling period and sampling frequency;
- CWT computation method and precision;
- signal-preparation configuration and summary statistics;
- physical period, frequency, and scale ranges;
- CWT dimensions and coefficient statistics.

The format version allows future schema evolution while preserving an
explicit compatibility boundary.

## Implemented Repository Structure

The implemented repository includes the following core components:

```text
mrpa-II/
├── analysis/
│   └── f0/
│       ├── interpolate_f0.py
│       └── extract_cwt.py
├── tests/
│   └── test_extract_cwt.py
└── README.md
```

Additional files and supporting components may also be present as the
repository evolves.

## Planned Repository Structure

The broader planned structure remains:

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

Generated datasets and intermediate analysis artifacts should not
normally be committed to Git.

## Installation

> [!IMPORTANT]
> The CWT extraction and persistence pipeline is implemented and tested.
> Dependency installation should use the repository's current dependency
> specification when available.

### 1. System requirements

Baseline environment:

- Linux or macOS;
- Python 3.11 or later;
- Git;
- a Python virtual environment.

The current F0 and CWT processing pipeline does not require a GPU.

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

The baseline is Python 3.11 or later.

If `python` already points to the intended Python installation, it may
be used instead of `python3`.

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

If the repository contains `requirements.txt`, install the declared
dependencies with:

```bash
python -m pip install -r requirements.txt
```

The current processing pipeline relies on packages for:

- numerical array processing;
- continuous wavelet analysis;
- F0 and audio processing;
- automated testing.

## Current Executable Example

The CWT pipeline accepts an interpolated continuous-F0 array and its
corresponding time axis.

```bash
python analysis/f0/extract_cwt.py \
    --input continuous_f0.npy \
    --time-input time.npy \
    --output cwt.npz
```

The default output set is:

```text
cwt.npz
cwt_metadata.json
```

A metadata path may be specified explicitly:

```bash
python analysis/f0/extract_cwt.py \
    --input continuous_f0.npy \
    --time-input time.npy \
    --output cwt.npz \
    --metadata-output metadata.json
```

Optional inputs can be supplied when available:

```bash
python analysis/f0/extract_cwt.py \
    --input continuous_f0.npy \
    --time-input time.npy \
    --voicing-input voicing_mask.npy \
    --interpolation-mask-input interpolation_mask.npy \
    --interpolation-metadata interpolation_metadata.json \
    --output cwt.npz \
    --metadata-output cwt_metadata.json
```

The CLI validates all declared inputs before running the transform. On
success, it computes the CWT representation, constructs and validates
the metadata, and saves both output artifacts.

## Testing

Run the CWT test module:

```bash
python -m pytest tests/test_extract_cwt.py -q
```

Run the complete test suite:

```bash
python -m pytest -q
```

Current validated baseline:

```text
tests/test_extract_cwt.py : 292 passed
full test suite           : 306 passed
```

## Research Direction

With the validated CWT and persistence foundation in place, the next
research stages can focus on:

- loading and validating persisted CWT representations;
- ScaleProfile construction;
- boundary-candidate discovery;
- nucleus-candidate discovery;
- variable-length node construction;
- recursive containment;
- cross-linguistic comparison;
- integration with broader linguistic topology research.

The central methodological constraint remains unchanged:

> Discover acoustic and prosodic topology first. Introduce
> language-specific linguistic interpretation afterward.
