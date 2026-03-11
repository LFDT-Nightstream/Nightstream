# PiDEC

## Purpose

- **What it is**: The decomposition reduction step Π_DEC. Defines `piDECKnowledgeStatement` as the existence of `deltaInv` with `mulRq ctx.invDelta deltaInv = oneRq`, plus `ceRelaxedRelation ctx` and `SumCheckClaimTrue`.
- **Key property**: `piDEC_of_weak` gives the theorem-native entrypoint from the weak `Π_RLC` statement; `piDEC_of_ce` and `piDEC_of_ccsRelation` bridge directly from the compact relation layer; `piDEC_of_section71_ce` bridges directly from a realized Section 7.1 proof-system CE instance; `piDEC_of_section71Provider` takes one concrete Section 7.1 provider bundle; `piDEC_of_section71Specialization` takes one generic proof-system `Section71Instance` plus compact specialization; `piDEC_of_section71Setup` packages that theorem-native pair into one upstream owner; `piDEC_of_section71TheoremInstance` consumes one paper-faithful specialized theorem instance; `piDEC_of_section71Context` consumes the single-object compact theorem-native Section 7.1 owner; `piDEC_of_section71Data` consumes one explicit protocol-side Definition-14 data package; `piDEC_of_protocolTargetData` consumes the explicit protocol-side Section 7.5 owner plus a transition witness; and the active paper-facing/native paper-facing routes each have direct constructors. Invertibility is extracted directly from `protocolTargetProp` (via `ceRelaxedRelation`), not from a separate low-norm boundary input.
- **Protocol role**: ProtocolTheorem and FoldingProtocol depend on `piDECKnowledgeStatement` for the knowledge-soundness chain. Section 7.5 (Π_DEC) reduces norm from B to b via decomposition.

## Target Formulas

- `piDECKnowledgeStatement(ctx) ↔ ∃ deltaInv, mulRq ctx.invDelta deltaInv = oneRq ∧ ceRelaxedRelation(ctx) ∧ SumCheckClaimTrue(sumcheckInstanceOfContext ctx)`
- `piDEC_of_weak`: `piRLCWeakStatement ctx → piDECKnowledgeStatement ctx`
- `piDEC_of_ce`: `ceRelation ctx → piDECKnowledgeStatement ctx`
- `piDEC_of_section71_ce`: realized proof-system `CE.Holds → piDECKnowledgeStatement ctx`
- `piDEC_of_section71Provider`: one concrete Section 7.1 provider bundle → `piDECKnowledgeStatement ctx`
- `piDEC_of_section71Specialization`: proof-system `Section71Instance` + compact specialization → `piDECKnowledgeStatement ctx`
- `piDEC_of_section71Setup`: theorem-native Section 7.1 setup → `piDECKnowledgeStatement ctx`
- `piDEC_of_section71TheoremInstance`: one paper-faithful Section 7.1 theorem instance → `piDECKnowledgeStatement ctx`
- `piDEC_of_section71Context`: one theorem-native Section 7.1 context object → `piDECKnowledgeStatement target`
- `piDEC_of_section71Data`: one protocol-side Section 7.1 Definition-14 data package → `piDECKnowledgeStatement ctx`
- `piDEC_of_protocolTargetData`: one protocol-side Section 7.5 target-data owner + witness → `piDECKnowledgeStatement ctx`
- `piDEC_of_ccsRelation`: `ccsRelation ctx → SumCheckTransitionWitness ctx → piDECKnowledgeStatement ctx`
- `piDEC_of_paperCarrierDiff`: active paper-facing route data + witness → `piDECKnowledgeStatement ctx`
- `piDEC_of_basisKernelAssumption`: finite basis-kernel Thm-3 witness + active paper-facing route data + witness → `piDECKnowledgeStatement ctx`
- `piDEC_of_basisKernelCheck`: executable finite basis-kernel checker + active paper-facing route data + witness → `piDECKnowledgeStatement ctx`
- `piDEC_of_native_paperCarrierDiff`: active native paper-facing route data + witness → `piDECKnowledgeStatement ctx`
- `piDEC_of_assumptions`: `PiDECAssumptions ctx → SumCheckTransitionWitness ctx → piDECKnowledgeStatement ctx`
- Theorem 7: Π_DEC : CE(B) → CE(b)^k is a reduction of knowledge.

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Section 7.5 (Π_DEC), lines 585-593.
- Theorem 7 (Π_DEC is reduction of knowledge), lines 594-596.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/PiDEC.lean` | Section 7.5, Theorem 7 |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Assumptions | `PiDECAssumptions` | abbrev | Definitional | Alias of `ProtocolTargetAssumptions ctx` |
| Statement | `piDECKnowledgeStatement` | def | Definitional | ∃ deltaInv, inverse ∧ ceRelaxed ∧ claimTrue |
| Theorem | `piDEC_of_weak` | theorem | Theorem-Target | Weak statement → knowledge statement |
| Theorem | `piDEC_of_ce` | theorem | Theorem-Target | CE relation → knowledge statement |
| Theorem | `piDEC_of_section71_ce` | theorem | Theorem-Target | Realized proof-system CE membership → knowledge statement |
| Theorem | `piDEC_of_section71Provider` | theorem | Theorem-Target | One concrete Section 7.1 provider bundle → knowledge statement |
| Theorem | `piDEC_of_section71Specialization` | theorem | Theorem-Target | One proof-system `Section71Instance` plus compact specialization → knowledge statement |
| Theorem | `piDEC_of_section71Setup` | theorem | Theorem-Target | One theorem-native Section 7.1 setup → knowledge statement |
| Theorem | `piDEC_of_section71TheoremInstance` | theorem | Theorem-Target | One paper-faithful Section 7.1 theorem instance → knowledge statement |
| Theorem | `piDEC_of_section71Context` | theorem | Theorem-Target | One theorem-native Section 7.1 context object → knowledge statement |
| Theorem | `piDEC_of_section71Data` | theorem | Theorem-Target | One protocol-side Section 7.1 Definition-14 data package → knowledge statement |
| Theorem | `piDEC_of_protocolTargetData` | theorem | Theorem-Target | One protocol-side Section 7.5 target-data owner + witness → knowledge statement |
| Theorem | `piDEC_of_ccsRelation` | theorem | Theorem-Target | CCS relation + witness → knowledge statement |
| Theorem | `piDEC_of_paperCarrierDiff` | theorem | Theorem-Target | Active paper-facing route data + witness → knowledge statement |
| Theorem | `piDEC_of_basisKernelAssumption` | theorem | Theorem-Target | Finite basis-kernel Thm-3 witness + active paper-facing route data + witness → knowledge statement |
| Theorem | `piDEC_of_basisKernelCheck` | theorem | Theorem-Target | Executable finite basis-kernel checker + active paper-facing route data + witness → knowledge statement |
| Theorem | `piDEC_of_native_paperCarrierDiff` | theorem | Theorem-Target | Active native paper-facing route data + witness → knowledge statement |
| Theorem | `piDEC_of_assumptions` | theorem | Theorem-Target | Assumptions + witness → knowledge statement |

## Proof Obligations and Closure Plan

- `piDEC_of_weak` is the compact theorem-native entrypoint.
- `piDEC_of_section71_ce` is the paper-facing proof-system entrypoint once a `ProtocolSection71Realization` is supplied upstream.
- `piDEC_of_section71Provider` is the direct paper-facing entrypoint once one concrete Section 7.1 theorem instance is packaged upstream.
- `piDEC_of_section71Specialization` is the generic proof-system entrypoint from one `Section71Instance` plus compact specialization.
- `piDEC_of_section71Setup` is the smallest single-argument theorem-native entrypoint once that proof-system instance and specialization are packaged together upstream.
- `piDEC_of_section71TheoremInstance` is the fully paper-faithful entrypoint once one specialized Section 7.1 theorem instance is available as a single object.
- `piDEC_of_section71Context` packages that specialized theorem instance together with its compact target context as one upstream owner.
- `piDEC_of_section71Data` exposes the same knowledge statement directly from the explicit protocol-side Definition-14 data, before collapsing it into `ProtocolSection71TheoremInstance` / `ProtocolSection71Context`.
- `piDEC_of_protocolTargetData` is the direct protocol-side Section 7.5 entrypoint once the compact target data and accepted transition witness are available.
- The compact relation-layer bridges and active-route constructors must extract invertibility from `protocolTargetProp` rather than introducing a separate local invertibility boundary.
- `piDEC_of_assumptions` and `piDEC_of_native_assumptions` remain compatibility wrappers around the upstream protocol-target bundle.

## Assumption Ledger

- No extra invertibility boundary is threaded at `PiDEC` level; invertibility is already required upstream in `ProtocolTargetAssumptions`.
- No separate SumCheck bundle is introduced locally here.
- `ProtocolTargetData` is a theorem-native owner, not a compatibility boundary; the assumptions alias remains only as a legacy convenience surface.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/PiRLC.lean`: uses `piRLCWeakStatement`, `piRLCWeak_of_assumptions`, `PiRLCAssumptions`.
- `SuperNeo/ProtocolTarget.lean`: `protocolTargetProp` carries `invertibleRq ctx.invDelta`.

Downstream consumers:
- `SuperNeo/ProtocolTheorem.lean`: depends on PiDEC for knowledge reduction chain.
- `SuperNeo/ProofSystem/Folding/PiDEC.lean`: imports PiDEC for decomposition step.

## Implementation Plan

1. Define `piDECKnowledgeStatement` as inverse existence together with `ceRelaxedRelation` and `SumCheckClaimTrue`.
2. Factor the theorem-native proof through the weak `Π_RLC` statement.
3. Expose direct bridges from Section 7.1 CE realizations, proof-system `Section71Instance` plus compact specialization, compact relations, and active protocol routes.
4. Keep the assumptions-based theorems as compatibility wrappers over the same compact proof.

## Quality Expectations

`piDECKnowledgeStatement` must match Theorem 7: inverse existence plus relaxed CE and sum-check claim. Derivation must use invertibility assumption correctly.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- `piDEC_of_assumptions` proved.

## Out of Scope

- Proof of Theorem 7 (deferred to appendix).
- Concrete invertibility bound instantiation.
