# PiRLC — Π_RLC Proof-System Wrapper

## Purpose

- **What it is**: Proof-system wrapper that lifts `SuperNeo.PiRLC` into proof-system types and re-exports both the compatibility assumption surface and the paper-facing Section 7.1 entrypoints.
- **Key property**: the wrapper exposes weak `Π_RLC` not only from the compact assumption bundle, but also from an explicit realized CE instance, one concrete Section 7.1 provider, one generic proof-system `Section71Instance` plus compact specialization, the packaged theorem-native `ProtocolSection71Setup`, one paper-faithful `ProtocolSection71TheoremInstance`, the single-object `ProtocolSection71Context`, the explicit protocol-side `ProtocolSection71Data`, or the explicit protocol-side `ProtocolTargetData`.
- **Protocol role**: Provides the typed boundary for Lemma 4 (Π_RLC is a weak interactive reduction); used in the folding composition (Section 7).

## Target Formulas

- \(\text{WeakStatement}(\text{ctx}) \equiv \pi_{\text{RLC}}\) weak relation (Definition 9).
- `ProtocolSection71Realization ctx → CE.Holds hReal.ce hReal.ceStatement hReal.ceWitness → WeakStatement ctx`
- `ProtocolSection71Provider ctx → WeakStatement ctx`
- `ProtocolSection71Specialization ctx hInst → WeakStatement ctx`
- `ProtocolSection71Setup ctx → WeakStatement ctx`
- `ProtocolSection71TheoremInstance ctx → WeakStatement ctx`
- `ProtocolSection71Context → WeakStatement target`
- `ProtocolSection71Data ctx → WeakStatement ctx`
- `ProtocolTargetData ctx → SumCheckTransitionWitness ctx → WeakStatement ctx`
- `ccsRelation ctx → SumCheckTransitionWitness ctx → WeakStatement ctx`
- \(\text{weak\_relaxed}: \text{PiRLCAssumptions}(\text{ctx}) \wedge \text{SumCheckTransitionWitness}(\text{ctx}) \to \text{WeakStatement}(\text{ctx})\).

## Paper Anchors

- **Source**: `./formal/superneo-lean/SuperNeo.pdf.md`
- Section 7.4 (Random linear combination reduction – Π_RLC), lines 549–583.
- Lemma 4 (Π_RLC is weak), line 581.

## Module Mapping

| Paper concept | Lean symbol | Role |
|---------------|-------------|--------|
| Π_RLC assumptions | `PiRLCAssumptions` | Definitional (abbrev) |
| Weak statement | `WeakStatement` | Definitional (abbrev) |
| Weak reduction from realized CE | `weak_relaxed_of_section71_ce` | Theorem-Target |
| Weak reduction from provider | `weak_relaxed_of_section71Provider` | Theorem-Target |
| Weak reduction from specialization | `weak_relaxed_of_section71Specialization` | Theorem-Target |
| Weak reduction from theorem-native setup | `weak_relaxed_of_section71Setup` | Theorem-Target |
| Weak reduction from paper-faithful theorem instance | `weak_relaxed_of_section71TheoremInstance` | Theorem-Target |
| Weak reduction from compact CCS | `weak_relaxed_of_ccsRelation` | Theorem-Target |
| Weak reduction theorem | `weak_relaxed` | Theorem-Target (forwarded) |

## Contract Surface

| Group | Symbol | Guarantee | Role |
|-------|--------|-----------|--------|
| Assumptions | `PiRLCAssumptions` | Forward to `SuperNeo.PiRLCAssumptions` | Definitional |
| Statement | `WeakStatement` | Forward to `SuperNeo.piRLCWeakStatement` | Definitional |
| Theorem | `weak_relaxed_of_section71_ce` | Realized Section 7.1 CE instance → weak statement | Theorem-Target |
| Theorem | `weak_relaxed_of_section71Provider` | Concrete Section 7.1 provider → weak statement | Theorem-Target |
| Theorem | `weak_relaxed_of_section71Specialization` | Proof-system `Section71Instance` plus compact specialization → weak statement | Theorem-Target |
| Theorem | `weak_relaxed_of_section71Setup` | Theorem-native Section 7.1 setup → weak statement | Theorem-Target |
| Theorem | `weak_relaxed_of_section71TheoremInstance` | Paper-faithful Section 7.1 theorem instance → weak statement | Theorem-Target |
| Theorem | `weak_relaxed_of_section71Context` | Theorem-native Section 7.1 context object → weak statement | Theorem-Target |
| Theorem | `weak_relaxed_of_section71Data` | Protocol-side Section 7.1 Definition-14 data package → weak statement | Theorem-Target |
| Theorem | `weak_relaxed_of_protocolTargetData` | Protocol-side Section 7.5 target-data owner + witness → weak statement | Theorem-Target |
| Theorem | `weak_relaxed_of_ccsRelation` | Compact CCS relation + witness → weak statement | Theorem-Target |
| Theorem | `weak_relaxed` | Assumptions + witness → weak statement | Theorem-Target |

## Proof Obligations and Closure Plan

`weak_relaxed_of_section71_ce` forwards to `SuperNeo.piRLCWeak_of_section71_ce`.
`weak_relaxed_of_section71Provider` forwards to `SuperNeo.piRLCWeak_of_section71Provider`.
`weak_relaxed_of_section71Specialization` forwards to `SuperNeo.piRLCWeak_of_section71Specialization`.
`weak_relaxed_of_section71Setup` forwards to `SuperNeo.piRLCWeak_of_section71Setup`.
`weak_relaxed_of_section71TheoremInstance` forwards to `SuperNeo.piRLCWeak_of_section71TheoremInstance`.
`weak_relaxed_of_section71Data` forwards to `SuperNeo.piRLCWeak_of_section71Data`.
`weak_relaxed_of_protocolTargetData` forwards to `SuperNeo.piRLCWeak_of_protocolTargetData`.
`weak_relaxed_of_ccsRelation` forwards to `SuperNeo.piRLCWeak_of_ccsRelation`.
`weak_relaxed` forwards to `SuperNeo.piRLCWeak_of_assumptions`.

## Assumption Ledger

No new theorem-level assumptions are introduced here. `PiRLCAssumptions` is retained only as a compatibility surface; the theorem-native entrypoints consume explicit Section 7.1 CE data, one concrete provider, one proof-system `Section71Instance` plus compact specialization, the packaged theorem-native `ProtocolSection71Setup`, one paper-faithful `ProtocolSection71TheoremInstance`, the single-object `ProtocolSection71Context`, the explicit protocol-side `ProtocolSection71Data`, or the explicit protocol-side `ProtocolTargetData`.

## Dependency and Consumer Map

- **Dependencies**: imports `SuperNeo.PiRLC`, `SuperNeo.ProofSystem.Types`.
- **Consumers**:
  - `SuperNeo.ProofSystem.Folding`: imports PiRLC for barrel.
  - `SuperNeo.ProofSystem.Protocol`, `SuperNeo.FoldingProtocol`: depend on weak reduction for composition.

## Implementation Plan

Keep as thin wrapper; proof work here is limited to forwarding the core `PiRLC`
theorem surfaces into the proof-system namespace.

## Quality Expectations

Wrapper stays minimal; interface docstring references spec and paper anchors.

## Acceptance Criteria

- `lake build` succeeds.
- Spec contains explicit paper anchors with line ranges.
- `weak_relaxed` proved.

## Out of Scope

- Core Π_RLC construction and compact-to-Section-7.1 specialization live in
  `SuperNeo.PiRLC` / `SuperNeo.ProtocolRelations`.
