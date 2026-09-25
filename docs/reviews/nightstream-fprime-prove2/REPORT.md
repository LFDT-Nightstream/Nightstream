# Nightstream F′ review using the Prove2Me method

Date: 2026-09-07. Reviewed source: `4fc02857c3aa4207f3290739cc62d794ba86f5f9`.

## Decision

Continue the current proof architecture, but judge progress by the missing proof from accepted assignments to the full step. The review found no basis for a broad rewrite. It also found no basis for a completion date or a numerical chance of success.

The project has substantial checked work. Its main remaining mathematical risk is the connection between the actual circuit assignment and the exact verifier computation. Its production path is also incomplete. More passing local proofs or diagnostics can leave both gaps open.

I applied Prove2Me's method locally: blind read-back of the theorem statements, an independent review of their dependencies, and a Lean-checked conditional root proof. **Prove2Me's hosted service did not run this review.** No project files were uploaded.

## Scope and assumptions

The review contract was to compare the implemented theorem statements with the required Stage 1 result, identify the missing connections, and give a concrete next action. Production code and owner files remain unchanged.

The architecture decision uses these assumptions:

| Assumption | Necessity check | Result |
|---|---|---|
| Stage 1 must finish the existing production F′ package, including its Rust caller and an approved proof backend. | Required by the owner goal. | Retained. |
| Soundness must cover every assignment accepted at the actual boundary. | Otherwise, a proof about the witness generator leaves other accepted assignments outside the theorem. | Retained. |
| Use the fixed Nightstream Goldilocks profile: `b = 2`, `k_rho = 16`, `B = 65536`. | Required by project policy. | Retained. |
| A new proof framework or broad compiler rewrite is needed. | No inspected counterexample or dependency establishes this. | Rejected. |
| A passing build, a file count, or a diagnostic count measures distance to completion. | None proves the final acceptance implication. | Rejected. |
| All cryptographic probability arguments must be newly formalized in Lean. | The owner goal permits explicit SuperNeo soundness assumptions for reductions not copied and proved. | Rejected. |

Sources: [Stage 1 outcome](/Users/nicarq/starstream/develop/nightstream-clean-up/FPRIME_STAGE1_GOAL.md:101), [security scope](/Users/nicarq/starstream/develop/nightstream-clean-up/FPRIME_STAGE1_GOAL.md:242), [current arbitrary-assignment target](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/CONSTRAINT_TREE.md:106).

## What Prove2Me contributes

Prove2Me checks that a submitted proof has exactly the target type. Its proof sketches show that a parent follows from named child statements. The parent remains open while required children are open. This is useful here because it can make a missing assumption visible at the root. A checked sketch does not establish that its children are true or easy to prove. [Prove2Me design](https://prove2.me/about).

Its auditor procedure uses a fresh agent to translate the Lean statement and its definitions back into plain mathematics without reading the intended claim first. I used that procedure for the three declarations below. A second agent independently traced the remaining connections. Their findings are saved in [READBACK.md](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-prove2/READBACK.md) and [CONNECTIONS.md](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-prove2/CONNECTIONS.md). [Auditor procedure](https://github.com/prove2me/prove2me_workspace/blob/main/references/mission_auditor.md).

The hosted service currently lists Lean 4.30.0 as supported. This package also uses Lean 4.30.0, with Mathlib revision `c5ea00351c28e24afc9f0f84379aa41082b1188f`. Hosted compatibility would still require the exact environment and project imports to match. Published statements and proofs are immutable. A full migration is therefore a separate task, with a submission scope and environment check. The site's example costs do not provide a Nightstream completion estimate. [Prove2Me FAQ](https://prove2.me/faq).

## Architecture and actual proof coverage

The paper-level requirement is clear: the augmented step performs the application transition, verifies the fold on recursive steps, and binds the complete next state into its digest. The project must connect accepted circuit rows to that same computation. See [HyperNova Construction 2](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/hypernova-paper/14_6_3_A_compiler_from_NIVC_compatible_folding_schemes_to_NIVC.md:27) and [SuperNeo v1.1 folding](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/superneo-paper-v1_1/07_superneo_folding_scheme_for_ccs.md:1).

The package has the intended layers: paper semantics in `Spec`; a circuit language and gadget contracts; concrete lifecycle phases; physical layout proofs; and circuit export. The eight-child Stage 1 assembler is already present. The main connection now under construction starts at the exported matrices and works back through values decoded from an arbitrary assignment.

| Entry or type | What it owns | Review result |
|---|---|---|
| [StepHoldsFor](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/Relation.lean:98) | The complete application, hash, base, and recursive step conditions. | The required semantic target. |
| [PerApplicationFixedPoint](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PerApplicationFixedPoint.lean:76) | The selected logical relation, recursive dimensions, and domain fit. | Checked structural results exist. |
| [RawValues](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PerApplicationCanonicalAssignment.lean:59) | Inputs to the canonical assignment constructor. | Useful for construction; narrower than all accepted assignments. |
| [ActualStep](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualStep.lean:28) | State, fresh input, and PiCCS proof values read from an arbitrary assignment. | Base step and exact PiCCS check proved; recursive PiDEC and output conditions remain. |
| [ActualPiDEC](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualPiDEC.lean:266) | PiDEC phase values read from the assignment. | Phase predicate proved; exact NIFS parent/proof/output transport remains. |
| [Poseidon2HashChainV1Package](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-fold-clean/src/stage1/mod.rs:24) | Rust package loading, matrix visits, and witness execution. | Adapter exists. A source search found no production lifecycle caller. |

### The three theorem read-backs

**The completed deterministic theorem has a restricted input domain.** `Poseidon2HashChainV1Closure.rowsZero_implies_stepHoldsFor` takes a raw packet, overwrites its context words with the selected verifier digest, builds its canonical encoded assignment, and assumes that assignment satisfies the rows. It proves the complete step for that packet. It does not quantify over an arbitrary accepted assignment. This is a real theorem, but it does not close the broader acceptance target. [Declaration](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/Poseidon2HashChainV1Closure.lean:35), [context overwrite](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PerApplicationVerifierBoundAssignment.lean:50).

**The arbitrary-assignment theorem exposes the recursive gap.** `ActualStep.selectedRowsAndPublic_step_iff_baseOrPiDec` assumes selected rows and the actual public digest boundary. It proves that the full decoded step is equivalent to the base case, or both exact PiDEC acceptance and exact computed output equality. It reads PiCCS proof fields from the assignment, but keeps PiDEC fields from a supplied proof template. Its context is the decoded context. [Declaration](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualStep.lean:350).

**The security theorem states a deterministic case split.** `rowsZero_implies_base_or_securityOutcome` has the same canonical assignment restriction and also assumes `LowNormInvertibility`. Its recursive result can be knowledge, a mixing failure, a sumcheck failure, a parent binding failure, failure to obtain an aligned PiRLC fork, or failure to obtain valid PiDEC child openings. It gives no probability bound. It must not be reported as an unconditional proof of cryptographic security. This does not require new probability research beyond the owner-approved assumption boundary. [Declaration](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/Poseidon2HashChainV1Closure.lean:64), [outcome definition](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Spec/Folding/Nifs/PaperSecurityComposition.lean:466).

## A concrete checked proof plan

[RootSketch.lean](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-prove2/tests/RootSketch.lean:1) is a new review artifact. It compiles against the unchanged package and passes the project's axiom audit.

It fixes the current hash-chain application, relation, and Ajtai key. It takes an **arbitrary assignment**. Its decoder reads PiCCS fields through `PiCCSAssignmentSoundness` and PiDEC child commitments and evaluations through `ActualPiDEC`. There is no caller-supplied proof template, raw packet, or `Encodes` premise.

The theorem proves this implication:

```text
selected rows + actual public digest boundary
  + equality to the selected verifier context
  + exact PiDEC acceptance when iteration is nonzero
  + exact computed output equality when iteration is nonzero
    => the unchanged full StepHoldsFor for the decoded input and output
```

The last three conditions remain explicit parameters. Lean checked that they are sufficient; it did not prove them from accepted rows. This is a conditional composition proof for one step. It is not the full Stage 1 goal, a completed security proof, or a hosted Prove2Me acceptance.

Context equality is a sufficient condition in this sketch. The review has not proved that the rows alone force it. The production acceptance path must justify the selected context, or use the permitted hash-collision argument at the correct outer boundary. A constructor that writes the expected context is not evidence about arbitrary accepted inputs. The existing map records an external context-or-collision draft, still awaiting review and a production caller. [Current context status](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/CONSTRAINT_TREE.md:70).

### Exact remaining connections

| Connection | Existing result to use | What still needs proof |
|---|---|---|
| Actual PiRLC challenge to verifier challenge | [piRlcChallenges_eq_key_of_initialState](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Layout/Stage1/AccumulatorSemantics.lean:461) | Derive its phase and initial-state premises from the same arbitrary accepted assignment. Connect each retained product challenge to that sampler result. |
| PiRLC sum to exact NIFS parent | [ActualPiRLCValues sums](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualPiRLCValues.lean:177) | Use the verifier challenges and exact typed PiCCS inputs, including commitment, public input, and both evaluation families. |
| Actual PiDEC phase to exact NIFS check | [ActualPiDEC phase](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualPiDEC.lean:266) | Identify its parent and decoded child messages with the attempt made by the production key. |
| PiDEC output to carried running state | [outputForAttempt_eq_accumulatorOutput](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Layout/Stage1/AccumulatorSemantics.lean:923) and [actual output state](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualRunningTransition.lean:181) | Show that the actual PiDEC decoder and the actual running-transition decoder read the same output. |
| Decoded context to verifier authority | [actual context preservation](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualPreimageFraming.lean:142) | Connect it to the actual verifier-owned acceptance boundary. Preserving a context does not select the correct context. |

These are proof connections, not evidence that each missing claim is true. The independent review found that the existing `Actual*` modules already take the appropriate direction. My earlier broad advice to separate construction and soundness needs this narrower form: complete these existing decoders and their equalities before proposing a new representation system.

### The next proof task

Prove that, for every assignment satisfying the selected rows and actual public boundary, the production key's PiRLC sampler returns a challenge vector whose coefficient at each source equals `ActualPiRLCValues.challenge` for every product descriptor of that source.

The proof must derive the sampler's initial state from the actual PiCCS outgoing transcript state. Its caller must not supply `RawValues`, `Encodes`, a separate environment with unproved agreement, or the sampler equality itself. Start from [ActualPiRLCValues.challenge](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualPiRLCValues.lean:35) and the existing [sampler connection theorem](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Layout/Stage1/AccumulatorSemantics.lean:461).

Success is a checked theorem with those inputs and that sampler conclusion. This resolves a required dependency of the PiDEC and computed-output conditions in the root sketch. If the proof exposes an unconstrained source cell, report the exact missing equality and show a row-satisfying counterexample before changing constraints. Do not respond by moving the same assumption into a new record.

The current owner order still applies: complete the required pilot/PiCCS reviews and approved conformance checks; keep PiRLC `InputBinding` frozen until PiCCS is Conformance-closed. This review grants no phase approval or permission to bypass that order. [Owner-ordered work](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/CONSTRAINT_TREE.md:1481).

## Why canonical construction cannot close the acceptance proof

The review includes a small Lean-checked example in [EncodingCheck.lean](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-prove2/tests/EncodingCheck.lean:1). Over the Goldilocks field, the integers `9223372034707292160` and `-9223372034707292161` differ by the modulus. Both have a 41-digit balanced ternary representation with coefficient norm below 2. They reconstruct the same field value, but the alternative digits differ from the canonical encoder output.

Thus, low norm and field reconstruction alone do not imply canonical encoding. An arbitrary-assignment proof must work with decoded values or derive every stronger representation property it needs. **This example does not show that the complete F′ rows accept a bad transition or even this alternative encoding.** The internal ternary encoding also does not change the fixed binary SuperNeo decomposition profile.

The earlier PiRLC/PiDEC parent-wiring defect is a separate, recorded row-assignment finding. The current source repairs it by sharing the actual forms. That record is not a reproduced cryptographic attack in this review. [Finding and repair scope](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/CONSTRAINT_TREE.md:1458).

## Completion risk

The source grew from 186 Lean files and 62,200 lines at `8efba072` on August 22 to 750 files and 254,090 lines at the reviewed commit. These counts include comments and blank lines. `Export` grew from 916 to 105,593 lines; `Spec` grew from 38,521 to 40,309. Most growth was in the implementation and proof connections, not new paper semantics. These measured sizes explain the maintenance burden, but do not prove that the approach cannot finish.

The current candidate has 6,377,559 active logical rows and 254,260,620 carrier coordinates. Its fixed-point and domain-fit results are useful completed work. They do not prove acceptable end-to-end production cost. [Current dimensions and proof owners](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/CONSTRAINT_TREE.md:418).

The project is not yet at final integration:

- The full arbitrary-assignment recursive step and verifier-context connection remain open.
- Pilot/PiCCS diagnostics are recorded, but required approved conformance evidence and independent reviews remain open. PiRLC and PiDEC still need the cumulative handoff on the same accepted package.
- The stored artifact and Rust identity pins precede the shared-value repair. They are not a validated replacement for the current candidate.
- The Rust package adapter has no production lifecycle caller in the searched source. Terminal and cumulative acceptance evidence remain part of the owner goal.
- The final production `prove → verify` run still needs a separately approved backend. A proof backend cannot establish the missing semantic or Lean–Rust conformance claims.

These are specific open requirements in the [current map](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/CONSTRAINT_TREE.md:17) and [Stage 1 outcome](/Users/nicarq/starstream/develop/nightstream-clean-up/FPRIME_STAGE1_GOAL.md:101).

My assessment is that the formal work has a credible route to completion because the exact phase proofs and useful arbitrary-assignment decoders already exist. Completion remains uncertain because those components are not yet connected through the actual acceptance boundary, and the production path is unfinished. The evidence supports neither “almost done” nor “the architecture cannot work.” The next sampler theorem gives a concrete test of whether the existing connection strategy continues to remove required assumptions.

## Validation and limits

| Check | Result | Scope |
|---|---|---|
| Package boundary, Lean library, and test/axiom library | Passed on this unchanged source earlier in this review. Warm incremental library check: 4 s; test library: 1 s. | Build and registered axiom checks. Not a cold-build or prover performance measurement. |
| Conditional root sketch | Passed in 2 s, including both declaration audits. | Sufficiency of the stated open conditions. |
| Encoding example | Passed in 1 s, including all four declaration audits. | Non-uniqueness of this low-norm field representation only. |
| Independent blind read-back | Completed without goal/paper text as interpretive input. | Literal statement scope; not a proof of source faithfulness by itself. |
| Independent dependency review | Completed, read-only. | Source connections and remaining premises; no execution. |
| Current Rust package and conformance archive | Not rerun. | Circuit JSON is a 134-byte Git LFS pointer; `../nightstream-stage1-evidence` is absent in this checkout. |
| Hosted Prove2Me check and production backend | Not run. | No hosted result or cryptographic acceptance claim. |

The root sketch initially failed due to a missing import and excessive definitional reduction in the review file. The encoding example initially failed because direct reduction exceeded Lean's default recursion depth. The final files use explicit proof structure, with no increased recursion or heartbeat limits. Their audits report only `propext`, `Classical.choice`, and `Quot.sound`, or subsets of those axioms.

See [validation record](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-prove2/VALIDATION.md), [root log](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-prove2/root-sketch.log), and [encoding log](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-prove2/encoding-check.log).

The report and checks are local artifacts under `docs/`, which the repository currently ignores. No production file, owner file, stored pin, or conformance status was changed. No commit was created.

## Rejected extra work

- Full migration to Prove2Me: not needed to perform this local review; it would add an environment and publication task.
- Broad architecture rewrite: no inspected fact makes it necessary to close the current proof gap.
- New cryptographic probability formalization: exceeds the explicit assumption policy unless a specific required claim demands it.
- New benchmark or deadline target: no measured evidence supports one; current counts cannot supply it.
