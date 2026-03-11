# PiRLC

## Purpose

- **What it is**: The weak interactive-reduction step Π_RLC. Defines `piRLCWeakStatement` as the conjunction of `ceRelaxedRelation ctx` and `SumCheckClaimTrue (sumcheckInstanceOfContext ctx)`.
- **Key property**: `piRLCWeak_of_ce` gives the theorem-native entrypoint from `ceRelation ctx`; `piRLCWeak_of_section71_ce` bridges directly from a realized Section 7.1 proof-system CE instance; `piRLCWeak_of_section71Provider` takes one concrete Section 7.1 provider bundle; `piRLCWeak_of_section71Specialization` takes one generic proof-system `Section71Instance` plus compact specialization; `piRLCWeak_of_section71Setup` packages that theorem-native pair into one upstream owner; `piRLCWeak_of_section71TheoremInstance` consumes one paper-faithful specialized theorem instance; `piRLCWeak_of_section71Context` consumes the single-object compact theorem-native Section 7.1 owner; `piRLCWeak_of_section71Data` consumes one explicit protocol-side Definition-14 data package; `piRLCWeak_of_protocolTargetData` consumes the explicit protocol-side Section 7.5 owner plus a transition witness; `piRLCWeak_of_ccsRelation` bridges directly from CCS relation plus witness; and the active paper-facing/native paper-facing routes each have direct constructors. The weak statement relaxes CE to ceRelaxed (CCS only).
- **Protocol role**: PiDEC depends on `piRLCWeakStatement` and `piRLCWeak_of_assumptions` for the weak→knowledge composition. Section 7.4 (Π_RLC) performs random linear combination of CE claims.

## Target Formulas

- `piRLCWeakStatement(ctx) ↔ ceRelaxedRelation(ctx) ∧ SumCheckClaimTrue(sumcheckInstanceOfContext ctx)`
- `piRLCWeak_of_ce`: `ceRelation ctx → piRLCWeakStatement ctx`
- `piRLCWeak_of_section71_ce`: realized proof-system `CE.Holds → piRLCWeakStatement ctx`
- `piRLCWeak_of_section71Provider`: one concrete Section 7.1 provider bundle → `piRLCWeakStatement ctx`
- `piRLCWeak_of_section71Specialization`: proof-system `Section71Instance` + compact specialization → `piRLCWeakStatement ctx`
- `piRLCWeak_of_section71Setup`: theorem-native Section 7.1 setup → `piRLCWeakStatement ctx`
- `piRLCWeak_of_section71TheoremInstance`: one paper-faithful Section 7.1 theorem instance → `piRLCWeakStatement ctx`
- `piRLCWeak_of_section71Context`: one theorem-native Section 7.1 context object → `piRLCWeakStatement target`
- `piRLCWeak_of_section71Data`: one protocol-side Section 7.1 Definition-14 data package → `piRLCWeakStatement ctx`
- `piRLCWeak_of_protocolTargetData`: one protocol-side Section 7.5 target-data owner + witness → `piRLCWeakStatement ctx`
- `piRLCWeak_of_ccsRelation`: `ccsRelation ctx → SumCheckTransitionWitness ctx → piRLCWeakStatement ctx`
- `piRLCWeak_of_paperCarrierDiff`: active paper-facing route data + witness → `piRLCWeakStatement ctx`
- `piRLCWeak_of_basisKernelAssumption`: finite basis-kernel Thm-3 witness + active paper-facing route data + witness → `piRLCWeakStatement ctx`
- `piRLCWeak_of_basisKernelCheck`: executable finite basis-kernel checker + active paper-facing route data + witness → `piRLCWeakStatement ctx`
- `piRLCWeak_of_native_paperCarrierDiff`: active native paper-facing route data + witness → `piRLCWeakStatement ctx`
- `piRLCWeak_of_assumptions`: `PiRLCAssumptions ctx → SumCheckTransitionWitness ctx → piRLCWeakStatement ctx`
- Weak reduction (Lemma 4): Π_RLC : CE^{K+k} → CE(B) is weak for φ projecting commitments.

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Section 7.4 (Π_RLC), lines 549-583.
- Lemma 4 (Π_RLC is weak), lines 582-583.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/PiRLC.lean` | Section 7.4, Lemma 4 |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Assumptions | `PiRLCAssumptions` | abbrev | Definitional | Alias of `ProtocolTargetAssumptions ctx` |
| Statement | `piRLCWeakStatement` | def | Definitional | ceRelaxedRelation ∧ SumCheckClaimTrue |
| Theorem | `piRLCWeak_of_ce` | theorem | Theorem-Target | CE relation → weak statement |
| Theorem | `piRLCWeak_of_section71_ce` | theorem | Theorem-Target | Realized proof-system CE membership → weak statement |
| Theorem | `piRLCWeak_of_section71Provider` | theorem | Theorem-Target | One concrete Section 7.1 provider bundle → weak statement |
| Theorem | `piRLCWeak_of_section71Specialization` | theorem | Theorem-Target | One proof-system `Section71Instance` plus compact specialization → weak statement |
| Theorem | `piRLCWeak_of_section71Setup` | theorem | Theorem-Target | One theorem-native Section 7.1 setup → weak statement |
| Theorem | `piRLCWeak_of_section71TheoremInstance` | theorem | Theorem-Target | One paper-faithful Section 7.1 theorem instance → weak statement |
| Theorem | `piRLCWeak_of_section71Context` | theorem | Theorem-Target | One theorem-native Section 7.1 context object → weak statement |
| Theorem | `piRLCWeak_of_section71Data` | theorem | Theorem-Target | One protocol-side Section 7.1 Definition-14 data package → weak statement |
| Theorem | `piRLCWeak_of_protocolTargetData` | theorem | Theorem-Target | One protocol-side Section 7.5 target-data owner + witness → weak statement |
| Theorem | `piRLCWeak_of_ccsRelation` | theorem | Theorem-Target | CCS relation + witness → weak statement |
| Theorem | `piRLCWeak_of_paperCarrierDiff` | theorem | Theorem-Target | Active paper-facing route data + witness → weak statement |
| Theorem | `piRLCWeak_of_basisKernelAssumption` | theorem | Theorem-Target | Finite basis-kernel Thm-3 witness + active paper-facing route data + witness → weak statement |
| Theorem | `piRLCWeak_of_basisKernelCheck` | theorem | Theorem-Target | Executable finite basis-kernel checker + active paper-facing route data + witness → weak statement |
| Theorem | `piRLCWeak_of_native_paperCarrierDiff` | theorem | Theorem-Target | Active native paper-facing route data + witness → weak statement |
| Theorem | `piRLCWeak_of_assumptions` | theorem | Theorem-Target | Assumptions + witness → weak statement |

## Proof Obligations and Closure Plan

- `piRLCWeak_of_ce` is the compact theorem-native entrypoint.
- `piRLCWeak_of_section71_ce` is the paper-facing proof-system entrypoint once a `ProtocolSection71Realization` is supplied upstream.
- `piRLCWeak_of_section71Provider` is the direct paper-facing entrypoint once one concrete Section 7.1 theorem instance is packaged upstream.
- `piRLCWeak_of_section71Specialization` is the generic proof-system entrypoint from one `Section71Instance` plus compact specialization.
- `piRLCWeak_of_section71Setup` is the smallest single-argument theorem-native entrypoint once that proof-system instance and specialization are packaged together upstream.
- `piRLCWeak_of_section71TheoremInstance` is the fully paper-faithful entrypoint once one specialized Section 7.1 theorem instance is available as a single object.
- `piRLCWeak_of_section71Context` packages that specialized theorem instance together with its compact target context as one upstream owner.
- `piRLCWeak_of_section71Data` exposes the same weak statement directly from the explicit protocol-side Definition-14 data, before collapsing it into `ProtocolSection71TheoremInstance` / `ProtocolSection71Context`.
- `piRLCWeak_of_protocolTargetData` is the direct protocol-side Section 7.5 entrypoint once the compact target data and accepted transition witness are available.
- `piRLCWeak_of_ccsRelation` and the direct active-route constructors must derive the weak statement without introducing a separate local protocol-relations bundle.
- `piRLCWeak_of_assumptions` remains the compatibility wrapper around the upstream protocol-target bundle.

## Assumption Ledger

- `PiRLCAssumptions` is a direct compatibility alias to `ProtocolTargetAssumptions` from upstream.
- No separate SumCheck or protocol-relations boundary bundle is introduced locally here.
- `ProtocolTargetData` is a theorem-native owner, not a compatibility boundary; the assumptions alias remains only as a legacy convenience surface.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/PiCCS.lean`: uses `piCCSStrongStatement`, `piCCSStrong_of_assumptions`, `PiCCSAssumptions`.
- `SuperNeo/ProtocolRelations.lean`: uses `ceRelaxedRelation`, `ceRelaxedRelation_of_ce`.

Downstream consumers:
- `SuperNeo/PiDEC.lean`: uses `piRLCWeakStatement`, `piRLCWeak_of_assumptions`, `PiRLCAssumptions`.
- `SuperNeo/ProofSystem/Folding/PiRLC.lean`: imports PiRLC for weak reduction step.

## Implementation Plan

1. Define `piRLCWeakStatement` as `ceRelaxedRelation ctx ∧ SumCheckClaimTrue (sumcheckInstanceOfContext ctx)`.
2. Factor the proof through `ceRelation`.
3. Expose direct bridges from Section 7.1 CE realizations, proof-system `Section71Instance` plus compact specialization, CCS relation plus witness, and active protocol routes.
4. Keep the assumptions-based theorem as a compatibility wrapper over the same compact proof.

## Quality Expectations

`piRLCWeakStatement` must match Lemma 4: relaxed CE (CCS only) plus sum-check claim truth. Derivation must compose correctly from strong statement.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- `piRLCWeak_of_assumptions` proved.

## Out of Scope

- Full protocol execution (ProofSystem layer).
- Probabilistic weak-reduction proof (Lemma 4 proof deferred to appendix).
