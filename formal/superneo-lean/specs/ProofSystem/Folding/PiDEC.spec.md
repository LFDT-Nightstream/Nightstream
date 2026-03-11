# PiDEC — Π_DEC Proof-System Wrapper

## Purpose

- **What it is**: Proof-system wrapper that lifts `SuperNeo.PiDEC` into proof-system types and re-exports both the compatibility assumption surface and the paper-facing Section 7.1 entrypoints.
- **Key property**: the wrapper exposes `Π_DEC` not only from the compact assumption bundle, but also from an explicit realized CE instance, one concrete Section 7.1 provider, one generic proof-system `Section71Instance` plus compact specialization, the packaged theorem-native `ProtocolSection71Setup`, one paper-faithful `ProtocolSection71TheoremInstance`, the single-object `ProtocolSection71Context`, the explicit protocol-side `ProtocolSection71Data`, or the explicit protocol-side `ProtocolTargetData`.
- **Protocol role**: Provides the typed boundary for Theorem 7 (Π_DEC is a reduction of knowledge); used in the folding composition (Section 7).

## Target Formulas

- \(\text{FinalStatement}(\text{ctx}) \equiv \pi_{\text{DEC}}\) knowledge statement (Definition 5).
- `ProtocolSection71Realization ctx → CE.Holds hReal.ce hReal.ceStatement hReal.ceWitness → FinalStatement ctx`
- `ProtocolSection71Provider ctx → FinalStatement ctx`
- `ProtocolSection71Specialization ctx hInst → FinalStatement ctx`
- `ProtocolSection71Setup ctx → FinalStatement ctx`
- `ProtocolSection71TheoremInstance ctx → FinalStatement ctx`
- `ProtocolSection71Context → FinalStatement target`
- `ProtocolSection71Data ctx → FinalStatement ctx`
- `ProtocolTargetData ctx → SumCheckTransitionWitness ctx → FinalStatement ctx`
- `ccsRelation ctx → SumCheckTransitionWitness ctx → FinalStatement ctx`
- \(\text{final\_of\_assumption}: \text{PiDECAssumptions}(\text{ctx}) \wedge \text{SumCheckTransitionWitness}(\text{ctx}) \to \text{FinalStatement}(\text{ctx})\).

## Paper Anchors

- **Source**: `./formal/superneo-lean/SuperNeo.pdf.md`
- Section 7.5 (Decomposition reduction – Π_DEC), lines 585–593.
- Theorem 7 (Π_DEC is a reduction of knowledge), line 593.

## Module Mapping

| Paper concept | Lean symbol | Role |
|---------------|-------------|--------|
| Π_DEC assumptions | `PiDECAssumptions` | Definitional (abbrev) |
| Final statement | `FinalStatement` | Definitional (abbrev) |
| Reduction of knowledge from realized CE | `final_of_section71_ce` | Theorem-Target |
| Reduction of knowledge from provider | `final_of_section71Provider` | Theorem-Target |
| Reduction of knowledge from specialization | `final_of_section71Specialization` | Theorem-Target |
| Reduction of knowledge from theorem-native setup | `final_of_section71Setup` | Theorem-Target |
| Reduction of knowledge from paper-faithful theorem instance | `final_of_section71TheoremInstance` | Theorem-Target |
| Reduction of knowledge from compact CCS | `final_of_ccsRelation` | Theorem-Target |
| Reduction-of-knowledge theorem | `final_of_assumption` | Theorem-Target (forwarded) |

## Contract Surface

| Group | Symbol | Guarantee | Role |
|-------|--------|-----------|--------|
| Assumptions | `PiDECAssumptions` | Forward to `SuperNeo.PiDECAssumptions` | Definitional |
| Statement | `FinalStatement` | Forward to `SuperNeo.piDECKnowledgeStatement` | Definitional |
| Theorem | `final_of_section71_ce` | Realized Section 7.1 CE instance → final statement | Theorem-Target |
| Theorem | `final_of_section71Provider` | Concrete Section 7.1 provider → final statement | Theorem-Target |
| Theorem | `final_of_section71Specialization` | Proof-system `Section71Instance` plus compact specialization → final statement | Theorem-Target |
| Theorem | `final_of_section71Setup` | Theorem-native Section 7.1 setup → final statement | Theorem-Target |
| Theorem | `final_of_section71TheoremInstance` | Paper-faithful Section 7.1 theorem instance → final statement | Theorem-Target |
| Theorem | `final_of_section71Context` | Theorem-native Section 7.1 context object → final statement | Theorem-Target |
| Theorem | `final_of_section71Data` | Protocol-side Section 7.1 Definition-14 data package → final statement | Theorem-Target |
| Theorem | `final_of_protocolTargetData` | Protocol-side Section 7.5 target-data owner + witness → final statement | Theorem-Target |
| Theorem | `final_of_ccsRelation` | Compact CCS relation + witness → final statement | Theorem-Target |
| Theorem | `final_of_assumption` | Assumptions + witness → final statement | Theorem-Target |

## Proof Obligations and Closure Plan

`final_of_section71_ce` forwards to `SuperNeo.piDEC_of_section71_ce`.
`final_of_section71Provider` forwards to `SuperNeo.piDEC_of_section71Provider`.
`final_of_section71Specialization` forwards to `SuperNeo.piDEC_of_section71Specialization`.
`final_of_section71Setup` forwards to `SuperNeo.piDEC_of_section71Setup`.
`final_of_section71TheoremInstance` forwards to `SuperNeo.piDEC_of_section71TheoremInstance`.
`final_of_section71Data` forwards to `SuperNeo.piDEC_of_section71Data`.
`final_of_protocolTargetData` forwards to `SuperNeo.piDEC_of_protocolTargetData`.
`final_of_ccsRelation` forwards to `SuperNeo.piDEC_of_ccsRelation`.
`final_of_assumption` forwards to `SuperNeo.piDEC_of_assumptions`.

## Assumption Ledger

No new theorem-level assumptions are introduced here. `PiDECAssumptions` is retained only as a compatibility surface; the theorem-native entrypoints consume explicit Section 7.1 CE data, one concrete provider, one proof-system `Section71Instance` plus compact specialization, the packaged theorem-native `ProtocolSection71Setup`, one paper-faithful `ProtocolSection71TheoremInstance`, the single-object `ProtocolSection71Context`, the explicit protocol-side `ProtocolSection71Data`, or the explicit protocol-side `ProtocolTargetData`.

## Dependency and Consumer Map

- **Dependencies**: imports `SuperNeo.PiDEC`, `SuperNeo.ProofSystem.Types`.
- **Consumers**:
  - `SuperNeo.ProofSystem.Folding`: imports PiDEC for barrel.
  - `SuperNeo.ProofSystem.Protocol`, `SuperNeo.FoldingProtocol`: depend on reduction-of-knowledge for composition.

## Implementation Plan

Keep as thin wrapper; proof work here is limited to forwarding the core `PiDEC`
theorem surfaces into the proof-system namespace.

## Quality Expectations

Wrapper stays minimal; interface docstring references spec and paper anchors.

## Acceptance Criteria

- `lake build` succeeds.
- Spec contains explicit paper anchors with line ranges.
- `final_of_assumption` proved.

## Out of Scope

- Core Π_DEC construction and compact-to-Section-7.1 specialization live in
  `SuperNeo.PiDEC` / `SuperNeo.ProtocolRelations`.
