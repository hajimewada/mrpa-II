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

The current implementation provides a validated and persistent pipeline
from continuous F0 to Boundary candidate discovery:

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
ScaleProfile Feature Extraction
  ↓
Scale-Level and Frame-Level Statistics
  ↓
Boundary Evidence Construction
  ↓
Boundary Strength Computation
  ↓
Local-Maximum Candidate Discovery
  ↓
Threshold and Minimum-Distance Filtering
  ↓
Persistent Boundary Candidate Representation
```

This pipeline transforms an interpolated continuous-F0 track into
validated CWT, ScaleProfile, and Boundary candidate representations.
Each processing stage produces compressed numerical arrays and
versioned JSON metadata.


### Planned MRPA-II pipeline

MRPA-II extends the implemented Boundary discovery foundation toward
Nucleus, node, and topology discovery:

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

The validated CWT, ScaleProfile, and Boundary candidate pipelines are
implemented. Nucleus discovery, variable-length node construction,
prosodic topology, and cross-linguistic validation remain planned
research stages.


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


#### Milestone 6: CWT feature extraction and ScaleProfile construction

Implemented functionality includes:

- persisted CWT loading and validation;
- real and complex CWT magnitude computation;
- scale-level statistical feature extraction;
- frame-level statistical feature extraction;
- dominant-frequency, frequency-centroid, and frequency-dispersion computation;
- zero-energy frame handling;
- ScaleProfile construction and validation;
- versioned metadata construction and validation;
- compressed ScaleProfile archive generation;
- JSON metadata persistence;
- integrated extraction orchestration;
- command-line integration;
- end-to-end CLI acceptance testing.


#### Milestone 7: Boundary candidate discovery

Implemented functionality includes:

- persisted ScaleProfile loading and validation;
- frame-level feature-difference computation;
- median and MAD normalization;
- multi-feature Boundary evidence construction;
- weighted Boundary strength computation;
- Boundary strength validation;
- local-maximum detection;
- deterministic plateau handling;
- threshold-based candidate filtering;
- minimum candidate-distance enforcement;
- Boundary candidate construction and validation;
- versioned metadata construction and validation;
- compressed Boundary archive generation;
- JSON metadata persistence;
- integrated extraction orchestration;
- command-line integration;
- end-to-end CLI acceptance testing.

#### Milestone 8: Nucleus candidate discovery

Implemented functionality includes:

- persisted ScaleProfile loading and validation;
- persisted Boundary candidate loading and validation;
- local-contrast computation with configurable frame radius;
- median and MAD normalization of local contrast;
- six-feature Nucleus evidence construction;
- configurable weighted Nucleus strength computation;
- Nucleus strength validation;
- endpoint-aware local-maximum detection;
- deterministic first-index plateau handling;
- inclusive threshold-based candidate filtering;
- minimum candidate-distance enforcement;
- configurable Boundary-adjacent candidate exclusion;
- Nucleus candidate construction and validation;
- versioned metadata construction and validation;
- compressed Nucleus archive generation;
- strict JSON metadata persistence;
- integrated extraction orchestration;
- configurable command-line integration;
- end-to-end CLI acceptance and error-path testing.

The Nucleus output archive contains:

- `nucleus_strength`;
- `candidate_frame_indices`;
- `candidate_strengths`.

Current validation status:

- 382 Nucleus-specific tests passing;
- 1,116 tests passing across the full repository;
- no failures or skipped Nucleus tests;
- `git diff --check` completing without output.

A known warning remains in PyWORLD 0.3.5 because its package initialization imports the deprecated `pkg_resources` API.

## Repository Status

MRPA-II is in active implementation.

MRPA-II is in active implementation.
The current codebase contains tested and integrated pipelines for CWT preprocessing, ScaleProfile feature extraction, Boundary candidate discovery, and Nucleus candidate discovery. These pipelines include validated input handling, deterministic numerical processing, versioned metadata construction, compressed output persistence, command-line integration, and end-to-end acceptance testing.



Current validation status:

```text
tests/test_extract_nucleus_candidates.py : 382 passed
full test suite                           : 1,116 passed
git diff --check                          : no output
```

One known non-blocking warning originates from the installed `pyworld`
0.3.5 package's use of the deprecated `pkg_resources` API.

## Current Output Formats

MRPA-II currently implements four connected processing stages:

```text
CWT extraction
    ↓
ScaleProfile extraction
    ↓
Boundary candidate discovery
    ↓
Nucleus candidate discovery
```

Each stage produces a compressed NumPy archive and a versioned JSON
metadata document.

If `--metadata-output` is not specified, the metadata filename is
derived from the corresponding output archive name.

### CWT outputs

```text
cwt.npz
cwt_metadata.json
```

The CWT archive contains:

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

The CWT metadata document contains:

```text
format_version
cwt_configuration
signal_summary
scale_summary
cwt_summary
```

It records the wavelet configuration, sampling period, sampling
frequency, transform method and precision, signal statistics, physical
frequency and scale ranges, output dimensions, and coefficient
statistics.

### ScaleProfile outputs

```text
scale_profile.npz
scale_profile_metadata.json
```

The ScaleProfile archive contains fourteen arrays:

```text
frequencies
magnitude
scale_mean
scale_std
scale_median
scale_max
scale_rms
frame_total
frame_mean
frame_max
dominant_scale_index
dominant_frequency_hz
frequency_centroid_hz
frequency_dispersion_hz
```

These arrays summarize the CWT magnitude across both scale and frame
dimensions. The associated metadata records the source CWT contract,
feature configuration, array shapes, scale statistics, and frame
statistics.

### Boundary candidate outputs

```text
boundary_candidates.npz
boundary_candidates_metadata.json
```

The Boundary archive contains:

```text
boundary_strength
candidate_frame_indices
candidate_strengths
```

Boundary candidates are derived from normalized frame-level changes in
six ScaleProfile features. Candidate discovery applies weighted evidence
integration, deterministic local-maximum detection, an inclusive
threshold, and minimum candidate-distance enforcement.

The associated metadata records the source ScaleProfile contract,
Boundary configuration, output shapes, strength statistics, and candidate
statistics.

### Nucleus candidate outputs

```text
nucleus_candidates.npz
nucleus_candidates_metadata.json
```

The Nucleus archive contains:

```text
nucleus_strength
candidate_frame_indices
candidate_strengths
```

Nucleus candidates are derived from normalized local contrast in six
ScaleProfile features. Candidate discovery applies configurable feature
weights, endpoint-aware local-maximum detection, deterministic
first-index plateau handling, an inclusive threshold, minimum
candidate-distance enforcement, and configurable exclusion around
Boundary candidates.

The associated metadata records:

```text
format_version
source_scale_profile
source_boundary
nucleus_configuration
shape_summary
nucleus_summary
```

The metadata includes source compatibility information, feature names and
weights, local-contrast configuration, candidate-selection settings,
output shapes, strength ranges, candidate statistics, and the number of
candidates removed by Boundary exclusion.

All persisted metadata uses versioned schemas and strict JSON
serialization. Candidate archives use compressed NumPy storage and are
validated against their source metadata before being written.

## Implemented Repository Structure

The implemented repository includes the following core components:

```text
mrpa-II/
├── analysis/
│   ├── boundary/
│   │   ├── __init__.py
│   │   └── extract_boundary_candidates.py
│   ├── f0/
│   │   ├── interpolate_f0.py
│   │   ├── extract_cwt.py
│   │   └── extract_scale_profile.py
│   └── nucleus/
│       ├── __init__.py
│       └── extract_nucleus_candidates.py
├── tests/
│   ├── test_extract_cwt.py
│   ├── test_extract_scale_profile.py
│   ├── test_extract_boundary_candidates.py
│   └── test_extract_nucleus_candidates.py
└── README.md
```

The implemented modules provide validated numerical processing,
versioned metadata, compressed persistence, command-line interfaces, and
end-to-end acceptance tests for the CWT, ScaleProfile, Boundary, and
Nucleus stages.


## Repository Roadmap

The repository currently implements the following connected analysis stages:

```text
continuous F0
    ↓
CWT representation
    ↓
ScaleProfile features
    ↓
Boundary candidates
    ↓
Nucleus candidates
```

The broader planned structure is:

```text
mrpa-II/
├── analysis/
│   ├── f0/
│   │   ├── extract_real_f0.py
│   │   ├── interpolate_f0.py
│   │   ├── plot_f0.py
│   │   ├── extract_cwt.py
│   │   ├── extract_scale_profile.py
│   │   ├── scale_energy.py
│   │   └── plot_scale_energy.py
│   ├── boundary/
│   │   ├── __init__.py
│   │   └── extract_boundary_candidates.py
│   ├── nucleus/
│   │   ├── __init__.py
│   │   └── extract_nucleus_candidates.py
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

The `nodes` and `topology` stages remain future work. The implemented
CWT, ScaleProfile, Boundary, and Nucleus stages establish the validated
numerical and persistence foundation for those later stages.

Generated datasets, compressed intermediate representations, metadata
artifacts, and experiment outputs should not normally be committed to
Git unless they are intentionally included as small test fixtures.

## Installation

> [!IMPORTANT]
> The CWT, ScaleProfile, Boundary candidate, and Nucleus candidate
> pipelines are implemented and tested. Dependency installation should
> use the repository's current dependency specification.

### 1. System requirements

Baseline environment:

- Linux, macOS, or Windows;
- Python 3.11 or later;
- Git;
- a Python virtual environment.

The currently implemented CWT, ScaleProfile, Boundary, and Nucleus
pipelines use CPU-based numerical processing and do not require a GPU.

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

Install the dependencies declared by the repository:

```bash
python -m pip install -r requirements.txt
```

The implemented processing pipeline relies on packages for:

- numerical array processing;
- continuous wavelet analysis;
- F0 and audio processing;
- compressed NumPy persistence;
- JSON metadata persistence;
- automated testing.

A known non-blocking warning originates from `pyworld` 0.3.5 importing
the deprecated `pkg_resources` API.

## Executable Pipeline

The implemented command-line tools form a connected processing path:

```text
continuous F0 arrays
    ↓
extract_cwt.py
    ↓
extract_scale_profile.py
    ↓
extract_boundary_candidates.py
    ↓
extract_nucleus_candidates.py
```

Each stage validates its inputs and associated metadata before performing
new computation.

### 1. CWT extraction

The CWT command accepts an interpolated continuous-F0 array and its
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
    --metadata-output cwt_metadata.json
```

Optional voicing, interpolation-mask, and interpolation-metadata inputs
may be supplied when available:

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

### 2. ScaleProfile extraction

The ScaleProfile stage loads and validates the persisted CWT arrays and
metadata, then derives scale-level and frame-level feature summaries.

```bash
python analysis/f0/extract_scale_profile.py \
    --input cwt.npz \
    --metadata-input cwt_metadata.json \
    --output scale_profile.npz
```

The default output set is:

```text
scale_profile.npz
scale_profile_metadata.json
```

### 3. Boundary candidate discovery

The Boundary stage loads the validated ScaleProfile artifacts and
discovers candidate prosodic boundaries.

```bash
python analysis/boundary/extract_boundary_candidates.py \
    --input scale_profile.npz \
    --metadata-input scale_profile_metadata.json \
    --output boundary_candidates.npz
```

The default output set is:

```text
boundary_candidates.npz
boundary_candidates_metadata.json
```

Boundary threshold and candidate spacing can be overridden:

```bash
python analysis/boundary/extract_boundary_candidates.py \
    --input scale_profile.npz \
    --metadata-input scale_profile_metadata.json \
    --output boundary_candidates.npz \
    --threshold 1.0 \
    --minimum-distance 1
```

### 4. Nucleus candidate discovery

The Nucleus stage consumes both the ScaleProfile artifacts and the
Boundary candidate artifacts.

```bash
python analysis/nucleus/extract_nucleus_candidates.py \
    --input scale_profile.npz \
    --metadata-input scale_profile_metadata.json \
    --boundary-input boundary_candidates.npz \
    --boundary-metadata-input boundary_candidates_metadata.json \
    --output nucleus_candidates.npz
```

The default output set is:

```text
nucleus_candidates.npz
nucleus_candidates_metadata.json
```

An explicit metadata output path and candidate-discovery settings may be
provided:

```bash
python analysis/nucleus/extract_nucleus_candidates.py \
    --input scale_profile.npz \
    --metadata-input scale_profile_metadata.json \
    --boundary-input boundary_candidates.npz \
    --boundary-metadata-input boundary_candidates_metadata.json \
    --output nucleus_candidates.npz \
    --metadata-output nucleus_metadata.json \
    --local-contrast-radius 2 \
    --threshold 1.0 \
    --minimum-distance 1 \
    --boundary-exclusion-distance 1
```

The six Nucleus evidence weights can also be overridden:

```bash
python analysis/nucleus/extract_nucleus_candidates.py \
    --input scale_profile.npz \
    --metadata-input scale_profile_metadata.json \
    --boundary-input boundary_candidates.npz \
    --boundary-metadata-input boundary_candidates_metadata.json \
    --output nucleus_candidates.npz \
    --frame-total-weight 1.0 \
    --frame-mean-weight 1.0 \
    --frame-max-weight 1.0 \
    --dominant-frequency-weight 1.0 \
    --frequency-centroid-weight 1.0 \
    --frequency-dispersion-weight 1.0
```

A Boundary exclusion distance of `0` disables Boundary-adjacent
candidate exclusion.

On success, each CLI validates its input artifacts, performs the
requested extraction, validates the new arrays and metadata, and writes
compressed NPZ and strict JSON outputs. Invalid paths, incompatible
metadata, malformed arrays, and invalid configuration values produce a
nonzero exit status.

## JSUT Batch Processing

The complete MRPA-II pipeline can be applied to JSUT utterances with:

```bash
scripts/process_jsut.sh
```

The script connects the implemented processing stages:

```text
JSUT WAV
    ↓
observed F0
    ↓
interpolated continuous F0
    ↓
Continuous Wavelet Transform
    ↓
ScaleProfile
    ↓
Boundary candidates
    ↓
Nucleus candidates
```

### Expected JSUT layout

By default, the script expects JSUT v1.1 under:

```text
~/projects/mrpa/data/jsut/raw/jsut_ver1.1
```

For example:

```text
~/projects/mrpa/data/jsut/raw/jsut_ver1.1/
├── basic5000/
│   └── wav/
├── countersuffix26/
├── loanword128/
├── onomatopee300/
├── precedent130/
├── repeat500/
├── travel1000/
├── utparaphrase512/
└── voiceactress100/
```

The default MRPA-II output root is:

```text
~/projects/mrpa/data/jsut/results/mrpa-II
```

### Process one utterance

Use `LIMIT=1` for an initial end-to-end check:

```bash
SUBSET=basic5000 \
LIMIT=1 \
FORCE=1 \
scripts/process_jsut.sh
```

A successful run reports:

```text
processed=1
failed=0
output_root=/home/hajimewada/projects/mrpa/data/jsut/results/mrpa-II
```

### Process ten utterances

Use the following command for a ten-utterance batch test:

```bash
SUBSET=basic5000 \
LIMIT=10 \
FORCE=1 \
scripts/process_jsut.sh
```

The validated ten-utterance run completed successfully:

```text
processed=10
failed=0
```

All ten utterances produced:

```text
complete markers       : 10
CWT archives           : 10
ScaleProfile archives  : 10
Boundary archives      : 10
Nucleus archives       : 10
complete pipeline logs : 10
```

### Resume an interrupted batch

Each successfully completed utterance receives a `.complete` marker.

Use `FORCE=0` to skip utterances that already have this marker:

```bash
SUBSET=basic5000 \
LIMIT=100 \
FORCE=0 \
scripts/process_jsut.sh
```

Use `FORCE=1` to remove the previous completion state and reprocess the selected utterances:

```bash
SUBSET=basic5000 \
LIMIT=10 \
FORCE=1 \
scripts/process_jsut.sh
```

`LIMIT=0` processes all WAV files in the selected subset.

### Environment variables

The batch script accepts the following environment variables:

```text
JSUT_DATA_ROOT
JSUT_CORPUS_ROOT
OUTPUT_ROOT
SUBSET
LIMIT
FORCE
PYTHON
```

Default values:

```text
JSUT_DATA_ROOT
  ~/projects/mrpa/data/jsut

JSUT_CORPUS_ROOT
  $JSUT_DATA_ROOT/raw/jsut_ver1.1

OUTPUT_ROOT
  $JSUT_DATA_ROOT/results/mrpa-II

SUBSET
  basic5000

LIMIT
  0

FORCE
  0

PYTHON
  <repository-root>/.venv/bin/python
```

### Per-utterance output layout

Each utterance receives an isolated output directory:

```text
results/mrpa-II/
└── basic5000/
    └── BASIC5000_0001/
        ├── .complete
        ├── f0/
        │   ├── BASIC5000_0001_f0.npy
        │   ├── BASIC5000_0001_f0_time.npy
        │   ├── BASIC5000_0001_f0_metadata.json
        │   ├── BASIC5000_0001_f0_interpolated.npy
        │   ├── BASIC5000_0001_voicing.npy
        │   ├── BASIC5000_0001_interpolation_mask.npy
        │   └── BASIC5000_0001_interpolation_metadata.json
        ├── cwt/
        │   ├── BASIC5000_0001_cwt.npz
        │   └── BASIC5000_0001_cwt_metadata.json
        ├── scale_profile/
        │   ├── BASIC5000_0001_scale_profile.npz
        │   └── BASIC5000_0001_scale_profile_metadata.json
        ├── boundary/
        │   ├── BASIC5000_0001_boundary_candidates.npz
        │   └── BASIC5000_0001_boundary_candidates_metadata.json
        ├── nucleus/
        │   ├── BASIC5000_0001_nucleus_candidates.npz
        │   └── BASIC5000_0001_nucleus_candidates_metadata.json
        └── logs/
            └── pipeline.log
```

The `.complete` marker is created only after all expected artifacts have
been generated successfully.

### Inspect a pipeline log

For a specific utterance:

```bash
cat \
  ~/projects/mrpa/data/jsut/results/mrpa-II/basic5000/BASIC5000_0001/logs/pipeline.log
```

A successful log ends with:

```text
pipeline_status=complete
```

### Verify batch artifacts

```bash
RESULT_ROOT="$HOME/projects/mrpa/data/jsut/results/mrpa-II/basic5000"

printf 'complete=%s\n' \
  "$(find "$RESULT_ROOT" -name '.complete' -type f | wc -l)"

printf 'cwt_npz=%s\n' \
  "$(find "$RESULT_ROOT" -path '*/cwt/*_cwt.npz' -type f | wc -l)"

printf 'scale_profile_npz=%s\n' \
  "$(find "$RESULT_ROOT" -path '*/scale_profile/*_scale_profile.npz' -type f | wc -l)"

printf 'boundary_npz=%s\n' \
  "$(find "$RESULT_ROOT" -path '*/boundary/*_boundary_candidates.npz' -type f | wc -l)"

printf 'nucleus_npz=%s\n' \
  "$(find "$RESULT_ROOT" -path '*/nucleus/*_nucleus_candidates.npz' -type f | wc -l)"

printf 'complete_logs=%s\n' \
  "$(grep -l \
      'pipeline_status=complete' \
      "$RESULT_ROOT"/*/logs/pipeline.log \
      | wc -l)"
```

For the validated ten-utterance batch, every count is `10`.



## Testing

Run an individual pipeline test module:

```bash
python -m pytest tests/test_extract_cwt.py -q
python -m pytest tests/test_extract_scale_profile.py -q
python -m pytest tests/test_extract_boundary_candidates.py -q
python -m pytest tests/test_extract_nucleus_candidates.py -q
```

Run the complete test suite:

```bash
python -m pytest -q
```

Check staged or unstaged whitespace errors:

```bash
git diff --check
git diff --cached --check
```

Current validated baseline:

```text
tests/test_extract_nucleus_candidates.py : 382 passed
full test suite                           : 1,116 passed
Nucleus test skips                        : 0
full-suite failures                       : 0
git diff --check                          : no output
```

One known warning remains in `pyworld` 0.3.5 because its package
initialization imports the deprecated `pkg_resources` API.

## Research Direction

The validated implementation now covers:

- CWT representation and physical frequency coordinates;
- ScaleProfile construction;
- multi-feature Boundary candidate discovery;
- multi-feature Nucleus candidate discovery;
- versioned metadata and compressed persistence;
- command-line orchestration and acceptance testing.

The next research and implementation stages can focus on:

- variable-length prosodic node construction;
- Boundary- and Nucleus-conditioned segment formation;
- recursive containment and overlapping hierarchy;
- parent-child and cross-scale relationships;
- topological consistency validation;
- cross-linguistic comparison;
- language-specific interpretation layers;
- integration with broader linguistic topology research.

The central methodological constraint remains unchanged:

> Discover acoustic and prosodic topology first. Introduce
> language-specific linguistic interpretation afterward.