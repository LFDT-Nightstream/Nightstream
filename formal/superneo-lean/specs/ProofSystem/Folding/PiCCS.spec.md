# ProofSystem/Folding/PiCCS.spec.md

## Purpose

- **What it is**: Proof-system-level wrapper that lifts the `SuperNeo.PiCCS` strong interactive reduction into the `ProofSystem.Folding` type framework.
- **Key property**: the wrapper exposes both the legacy compact assumption-bundle entrypoint and the paper-facing Section 7.1 / Section 7.5 entrypoints from an explicit realized CE instance, one concrete Section 7.1 provider, one generic proof-system `Section71Instance` plus compact specialization, the packaged theorem-native `ProtocolSection71Setup`, one paper-faithful `ProtocolSection71TheoremInstance`, the single-object `ProtocolSection71Context`, the explicit protocol-side `ProtocolSection71Data`, or the explicit protocol-side `ProtocolTargetData`.
- **Protocol role**: Provides the typed proof-system façade for Π_CCS consumed by the folding barrel and `ProtocolTheorem`.

## Target Formulas

- `ProtocolSection71Realization ctx → CE.Holds hReal.ce hReal.ceStatement hReal.ceWitness → StrongStatement ctx`
- `ProtocolSection71Provider ctx → StrongStatement ctx`
- `ProtocolSection71Specialization ctx hInst → StrongStatement ctx`
- `ProtocolSection71Setup ctx → StrongStatement ctx`
- `ProtocolSection71TheoremInstance ctx → StrongStatement ctx`
- `ProtocolSection71Context → StrongStatement target`
- `ProtocolSection71Data ctx → StrongStatement ctx`
- `ProtocolTargetData ctx → SumCheckTransitionWitness ctx → StrongStatement ctx`
- `ccsRelation ctx → SumCheckTransitionWitness ctx → StrongStatement ctx`
- `PiCCSAssumptions ctx → SumCheckTransitionWitness ctx → StrongStatement ctx`

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Section 7.3 (Π_CCS Strong Interactive Reduction), Lemma 3, lines 481-548.

## Module Mapping

| Paper concept | Lean symbol | Role |
|---------------|-------------|--------|
| Π_CCS assumptions | `PiCCSAssumptions` | Definitional (abbrev of `SuperNeo.PiCCSAssumptions`) |
| Strong statement | `StrongStatement` | Definitional (abbrev of `SuperNeo.piCCSStrongStatement`) |
| Strong soundness from realized CE | `soundness_relations_of_section71_ce` | Theorem-Target |
| Strong soundness from provider | `soundness_relations_of_section71Provider` | Theorem-Target |
| Strong soundness from specialization | `soundness_relations_of_section71Specialization` | Theorem-Target |
| Strong soundness from theorem-native setup | `soundness_relations_of_section71Setup` | Theorem-Target |
| Strong soundness from paper-faithful theorem instance | `soundness_relations_of_section71TheoremInstance` | Theorem-Target |
| Strong soundness from compact CCS | `soundness_relations_of_ccsRelation` | Theorem-Target |
| Strong soundness from assumptions | `soundness_relations` | Theorem-Target |

## Contract Surface

| Group | Symbol | Guarantee | Role |
|-------|--------|-----------|--------|
| Core | `PiCCSAssumptions` | Assumption bundle for Π_CCS | Definitional |
| Core | `StrongStatement` | Strong relation output | Definitional |
| Theorem | `soundness_relations_of_section71_ce` | Realized Section 7.1 CE instance → strong statement | Theorem-Target |
| Theorem | `soundness_relations_of_section71Provider` | Concrete Section 7.1 provider → strong statement | Theorem-Target |
| Theorem | `soundness_relations_of_section71Specialization` | Proof-system `Section71Instance` plus compact specialization → strong statement | Theorem-Target |
| Theorem | `soundness_relations_of_section71Setup` | Theorem-native Section 7.1 setup → strong statement | Theorem-Target |
| Theorem | `soundness_relations_of_section71TheoremInstance` | Paper-faithful Section 7.1 theorem instance → strong statement | Theorem-Target |
| Theorem | `soundness_relations_of_section71Context` | Theorem-native Section 7.1 context object → strong statement | Theorem-Target |
| Theorem | `soundness_relations_of_section71Data` | Protocol-side Section 7.1 Definition-14 data package → strong statement | Theorem-Target |
| Theorem | `soundness_relations_of_protocolTargetData` | Protocol-side Section 7.5 target-data owner + witness → strong statement | Theorem-Target |
| Theorem | `soundness_relations_of_ccsRelation` | Compact CCS relation + witness → strong statement | Theorem-Target |
| Theorem | `soundness_relations` | Compatibility assumptions + witness → strong statement | Theorem-Target |

## Proof Obligations and Closure Plan

- `soundness_relations_of_section71_ce`: forwards to `SuperNeo.piCCSStrong_of_section71_ce`.
- `soundness_relations_of_section71Provider`: forwards to `SuperNeo.piCCSStrong_of_section71Provider`.
- `soundness_relations_of_section71Specialization`: forwards to `SuperNeo.piCCSStrong_of_section71Specialization`.
- `soundness_relations_of_section71Setup`: forwards to `SuperNeo.piCCSStrong_of_section71Setup`.
- `soundness_relations_of_section71TheoremInstance`: forwards to `SuperNeo.piCCSStrong_of_section71TheoremInstance`.
- `soundness_relations_of_section71Data`: forwards to `SuperNeo.piCCSStrong_of_section71Data`.
- `soundness_relations_of_protocolTargetData`: forwards to `SuperNeo.piCCSStrong_of_protocolTargetData`.
- `soundness_relations_of_ccsRelation`: forwards to `SuperNeo.piCCSStrong_of_ccsRelation`.
- `soundness_relations`: forwards to `SuperNeo.piCCSStrong_of_assumptions`.

## Assumption Ledger

No new theorem-level assumptions are introduced here. `PiCCSAssumptions` is retained only as a compatibility surface; the theorem-native entrypoints consume explicit Section 7.1 CE data, one concrete provider, one proof-system `Section71Instance` plus compact specialization, the packaged theorem-native `ProtocolSection71Setup`, one paper-faithful `ProtocolSection71TheoremInstance`, the single-object `ProtocolSection71Context`, the explicit protocol-side `ProtocolSection71Data`, or the explicit protocol-side `ProtocolTargetData`.

## Dependency and Consumer Map

- **Dependencies**: `SuperNeo.PiCCS`, `SuperNeo.ProofSystem.Types`.
- **Consumers**:
  - `SuperNeo.ProofSystem.Folding`: imports PiCCS as part of the folding barrel.
  - `SuperNeo.ProofSystem.Protocol`: uses PiCCS soundness in the protocol capstone.

## Implementation Plan

- Stable thin wrapper. Proof work here is limited to forwarding the core `PiCCS`
  theorem surfaces into the proof-system namespace.

## Quality Expectations

- File stays under 25 lines.
- No logic duplication — delegates entirely to `SuperNeo.PiCCS`.

## Acceptance Criteria

- `lake build` succeeds.
- `soundness_relations` type-checks without sorry.

## Out of Scope

- Concrete sum-check round analysis and compact-to-Section-7.1 specialization
  live in `SuperNeo.PiCCS` / `SuperNeo.ProtocolRelations`.
