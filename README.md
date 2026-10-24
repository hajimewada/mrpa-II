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
