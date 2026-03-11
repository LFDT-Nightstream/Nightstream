# PiCCS

## Purpose

- **What it is**: The strong interactive-reduction step Π_CCS. Defines `piCCSStrongStatement` as the conjunction of `ceRelation ctx` and `SumCheckClaimTrue (sumcheckInstanceOfContext ctx)`.
- **Key property**: `piCCSStrong_of_ce` gives the theorem-native entrypoint from `ceRelation ctx`; `piCCSStrong_of_section71_ce` bridges directly from a realized Section 7.1 proof-system CE instance; `piCCSStrong_of_section71Provider` takes one concrete Section 7.1 provider bundle; `piCCSStrong_of_section71Specialization` takes one generic proof-system `Section71Instance` plus compact specialization; `piCCSStrong_of_section71Setup` packages that theorem-native pair into one upstream owner; `piCCSStrong_of_section71TheoremInstance` consumes one paper-faithful specialized theorem instance; `piCCSStrong_of_section71Context` consumes the single-object compact theorem-native Section 7.1 owner; `piCCSStrong_of_section71Data` consumes one explicit protocol-side Definition-14 data package; `piCCSStrong_of_protocolTargetData` consumes the explicit protocol-side Section 7.5 owner plus a transition witness; `piCCSStrong_of_ccsRelation` bridges directly from CCS relation plus witness; and the active paper-facing/native paper-facing routes each have direct constructors.
- **Protocol role**: PiRLC depends on `piCCSStrongStatement` and `piCCSStrong_of_assumptions` for the strong→weak composition (Theorem 6). Section 7.3 (Π_CCS) reduces CCS instances to CE instances via sum-check.

## Target Formulas

- `piCCSStrongStatement(ctx) ↔ ceRelation(ctx) ∧ SumCheckClaimTrue(sumcheckInstanceOfContext ctx)`
- `piCCSStrong_of_ce`: `ceRelation ctx → piCCSStrongStatement ctx`
- `piCCSStrong_of_section71_ce`: realized proof-system `CE.Holds → piCCSStrongStatement ctx`
- `piCCSStrong_of_section71Provider`: one concrete Section 7.1 provider bundle → `piCCSStrongStatement ctx`
- `piCCSStrong_of_section71Specialization`: proof-system `Section71Instance` + compact specialization → `piCCSStrongStatement ctx`
- `piCCSStrong_of_section71Setup`: theorem-native Section 7.1 setup → `piCCSStrongStatement ctx`
- `piCCSStrong_of_section71TheoremInstance`: one paper-faithful Section 7.1 theorem instance → `piCCSStrongStatement ctx`
- `piCCSStrong_of_section71Context`: one theorem-native Section 7.1 context object → `piCCSStrongStatement target`
- `piCCSStrong_of_section71Data`: one protocol-side Section 7.1 Definition-14 data package → `piCCSStrongStatement ctx`
- `piCCSStrong_of_protocolTargetData`: one protocol-side Section 7.5 target-data owner + witness → `piCCSStrongStatement ctx`
- `piCCSStrong_of_ccsRelation`: `ccsRelation ctx → SumCheckTransitionWitness ctx → piCCSStrongStatement ctx`
- `piCCSStrong_of_paperCarrierDiff`: active paper-facing route data + witness → `piCCSStrongStatement ctx`
- `piCCSStrong_of_basisKernelAssumption`: finite basis-kernel Thm-3 witness + active paper-facing route data + witness → `piCCSStrongStatement ctx`
- `piCCSStrong_of_basisKernelCheck`: executable finite basis-kernel checker + active paper-facing route data + witness → `piCCSStrongStatement ctx`
- `piCCSStrong_of_native_paperCarrierDiff`: active native paper-facing route data + witness → `piCCSStrongStatement ctx`
- `piCCSStrong_of_assumptions`: `PiCCSAssumptions ctx → SumCheckTransitionWitness ctx → piCCSStrongStatement ctx`
- Strong reduction (Lemma 3): Π_CCS : CCS^K × CE^k → CE^{K+k} is strong for φ projecting commitments.

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Section 7.3 (Π_CCS), lines 481-548.
- Lemma 3 (Π_CCS is strong), lines 545-546.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/PiCCS.lean` | Section 7.3, Lemma 3 |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Assumptions | `PiCCSAssumptions` | abbrev | Definitional | Alias of `ProtocolTargetAssumptions ctx` |
| Statement | `piCCSStrongStatement` | def | Definitional | ceRelation ∧ SumCheckClaimTrue |
| Theorem | `piCCSStrong_of_ce` | theorem | Theorem-Target | CE relation → strong statement |
| Theorem | `piCCSStrong_of_section71_ce` | theorem | Theorem-Target | Realized proof-system CE membership → strong statement |
| Theorem | `piCCSStrong_of_section71Provider` | theorem | Theorem-Target | One concrete Section 7.1 provider bundle → strong statement |
| Theorem | `piCCSStrong_of_section71Specialization` | theorem | Theorem-Target | One proof-system `Section71Instance` plus compact specialization → strong statement |
| Theorem | `piCCSStrong_of_section71Setup` | theorem | Theorem-Target | One theorem-native Section 7.1 setup → strong statement |
| Theorem | `piCCSStrong_of_section71TheoremInstance` | theorem | Theorem-Target | One paper-faithful Section 7.1 theorem instance → strong statement |
| Theorem | `piCCSStrong_of_section71Context` | theorem | Theorem-Target | One theorem-native Section 7.1 context object → strong statement |
| Theorem | `piCCSStrong_of_section71Data` | theorem | Theorem-Target | One protocol-side Section 7.1 Definition-14 data package → strong statement |
| Theorem | `piCCSStrong_of_protocolTargetData` | theorem | Theorem-Target | One protocol-side Section 7.5 target-data owner + witness → strong statement |
| Theorem | `piCCSStrong_of_ccsRelation` | theorem | Theorem-Target | CCS relation + witness → strong statement |
| Theorem | `piCCSStrong_of_paperCarrierDiff` | theorem | Theorem-Target | Active paper-facing route data + witness → strong statement |
| Theorem | `piCCSStrong_of_basisKernelAssumption` | theorem | Theorem-Target | Finite basis-kernel Thm-3 witness + active paper-facing route data + witness → strong statement |
| Theorem | `piCCSStrong_of_basisKernelCheck` | theorem | Theorem-Target | Executable finite basis-kernel checker + active paper-facing route data + witness → strong statement |
| Theorem | `piCCSStrong_of_native_paperCarrierDiff` | theorem | Theorem-Target | Active native paper-facing route data + witness → strong statement |
| Theorem | `piCCSStrong_of_assumptions` | theorem | Theorem-Target | Assumptions + witness → strong statement |

## Proof Obligations and Closure Plan

- `piCCSStrong_of_ce` is the compact theorem-native entrypoint.
- `piCCSStrong_of_section71_ce` is the paper-facing proof-system entrypoint once a `ProtocolSection71Realization` is supplied upstream.
- `piCCSStrong_of_section71Provider` is the direct paper-facing entrypoint once one concrete Section 7.1 theorem instance is packaged upstream.
- `piCCSStrong_of_section71Specialization` is the generic proof-system entrypoint from one `Section71Instance` plus compact specialization.
- `piCCSStrong_of_section71Setup` is the smallest single-argument theorem-native entrypoint once that proof-system instance and specialization are packaged together upstream.
- `piCCSStrong_of_section71TheoremInstance` is the fully paper-faithful entrypoint once one specialized Section 7.1 theorem instance is available as a single object.
- `piCCSStrong_of_section71Context` packages that specialized theorem instance together with its compact target context as one upstream owner.
- `piCCSStrong_of_section71Data` exposes the same strong statement directly from the explicit protocol-side Definition-14 data, before collapsing it into `ProtocolSection71TheoremInstance` / `ProtocolSection71Context`.
- `piCCSStrong_of_protocolTargetData` is the direct protocol-side Section 7.5 entrypoint once the compact target data and accepted transition witness are available.
- `piCCSStrong_of_ccsRelation` and the direct active-route constructors must derive the strong statement without introducing an extra local SumCheck boundary bundle.
- `piCCSStrong_of_assumptions` remains the compatibility wrapper from the upstream protocol-target bundle.

## Assumption Ledger

- `PiCCSAssumptions` is a direct compatibility alias to `ProtocolTargetAssumptions` from upstream.
- SumCheck truth is discharged from the accepted transition witness rather than introduced as a separate local boundary here.
- `ProtocolTargetData` is a theorem-native owner, not a compatibility boundary; the assumptions alias remains only as a legacy convenience surface.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/ProtocolTarget.lean`: uses `protocolTargetProp_of_assumptions`, `ceRelation`, `SumCheckTransitionWitness`, `sumcheckInstanceOfContext`, `ProtocolTargetAssumptions`.
- `SuperNeo/SumCheck.lean`: uses `sumcheckSoundness_constructive`.

Downstream consumers:
- `SuperNeo/PiRLC.lean`: uses `piCCSStrongStatement`, `piCCSStrong_of_assumptions`, `PiCCSAssumptions`.
- `SuperNeo/ProofSystem/Folding/PiCCS.lean`: imports PiCCS for strong reduction step.

## Implementation Plan

1. Define `piCCSStrongStatement` as `ceRelation ctx ∧ SumCheckClaimTrue (sumcheckInstanceOfContext ctx)`.
2. Factor the theorem-native proof through `ceRelation`.
3. Expose direct bridges from Section 7.1 CE realizations, proof-system `Section71Instance` plus compact specialization, CCS relation plus witness, and active protocol routes.
4. Keep the assumptions-based theorem as a compatibility wrapper over the same compact relation proof.

## Quality Expectations

`piCCSStrongStatement` must match Lemma 3: CE relation plus sum-check claim truth. Derivation must thread assumptions correctly.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- `piCCSStrong_of_ce` proved.
- `piCCSStrong_of_assumptions` proved.

## Out of Scope

- Full protocol execution (ProofSystem layer).
- Probabilistic strong-reduction proof (Lemma 3 proof deferred to appendix).
