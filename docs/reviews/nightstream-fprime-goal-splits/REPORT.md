# Code review of the four goal splits

Reviewed commit: `4fc02857c3aa4207f3290739cc62d794ba86f5f9`, 2026-09-07.

## Revised decision: one compatible path

The user requires work to carry forward between the proposed goals. Under that requirement, the four proposals should not become separate implementation projects. **Use proposal 2 as Goal A, and complete the rest of the existing Stage 1 path as Goal B.** Native SuperNeo work from proposal 1 is shared implementation/conformance work inside those goals. The circuit and formal results from proposals 3 and 4 become later milestones on the same path.

**Goal A: close the existing pilot/PiCCS Compiler-closed and Conformance-closed tiers on the selected Nightstream package.** Keep the exact phase targets, pilot/hash observations, verifier-owned fixture context, input conversions, and registered reviews/checks. Its result is the accepted PiCCS phase, including its ordered outputs and outgoing transcript state, with current source/input evidence. It does not claim the complete F′ context/security theorem.

**Goal B: complete Stage 1 using those same definitions, package, and phase interfaces.** Close the actual PiRLC sampler/input connection and PiRLC conformance, then PiDEC input/output connection and conformance, then the full arbitrary-assignment F′ and selected-context/security/terminal composition, and finally the exclusive production implementation and approved backend run. Proof work follows the current owner order; each conformance extension consumes the exact preceding accepted output. [Required cumulative order](/Users/nicarq/starstream/develop/nightstream-clean-up/FPRIME_STAGE1_GOAL.md:517), [current work order](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/CONSTRAINT_TREE.md:1481).

This corrects an implication in the earlier advice: making native SuperNeo a separate secure product could add a public API, input-binding wrapper, or transcript variant that PiCCS closure does not need. That product direction is excluded from the recommended sequence. The existing native prover/verifier still supplies useful production calculations and conformance checks.

The assumptions behind this revision are the user's reuse requirement and the owner's single Lean authority, fixed profile, and one final package. They are necessary constraints. Separate releases, new packages, alternate statement formats, and a new proof framework are not required and are excluded. The final Stage 1 completion standard is unchanged.

| Earlier proposal | Place in the compatible plan | Work that carries forward |
|---|---|---|
| 1. Native SuperNeo | Support the active phase in A or B; no separate standalone product goal. | Existing prover/verifier, selected input conversions, exact phase results, valid openings, and cumulative native/Lean checks. |
| 2. PiCCS compiler/conformance | Goal A. | Exact selected-row proofs, validated package/inputs, review evidence, accepted PiCCS outputs, and outgoing state consumed by PiRLC. |
| 3. SuperNeo verifier circuit | A proof milestone inside B. | Use the same PiCCS result and actual assignment decoders to prove sampler, parent, child, and output agreement. |
| 4. Complete formal F′ package | A later milestone inside B, before final production closure. | Compose the same accumulator result into the unchanged full step, context/security/terminal boundary, then use the same emitted relation in Rust. |

### Concrete compatibility decisions

| Proposed work | Decision for this sequence |
|---|---|
| Repair a native PiCCS conversion that disagrees with the selected Lean input. | Required if the current checks establish the mismatch. Fix the conversion at the existing boundary; preserve Lean's transcript and public input. |
| Align the native prior digest with Lean's fresh-public-input digest. | Keep the existing state preimage, context and `encHash`. Derive or check the mapping through the current selected state/fixture boundary. Equal supplied digest fields alone are insufficient. |
| Reabsorb the full running statement or replace the state hash with a native accumulator hash. | Excluded. This would change the selected transcript or binding contract to create an independent native protocol. |
| Change Lean to read `running.fold_digest` because Rust currently does. | Excluded. The native implementation must conform to the selected Lean meaning. |
| Create a separate native relation, key, production package, or input schema. | Excluded. Convert existing values with exact checks and retain one semantic authority. |
| Redesign native PiRLC projection handling during A. | Excluded from A. It does not close PiCCS. Resolve its exact transcript endpoint during the authorized PiRLC work in B, against the existing Lean phase result. |
| Make every complete-NIFS definition executable before closing PiCCS. | Not a prerequisite for A. Use the existing PiCCS executable checker and equality theorems; close composed execution when its phase dependencies are ready. |
| Use the canonical witness generator as the final soundness boundary. | Excluded. Later proofs must still cover arbitrary accepted assignments. |

Source basis: [Lean prior digest and transcript blocks](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/ProductionKey.lean:100), [existing native state serializer](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-fold-clean/src/stage1/inputs.rs:152), [exact PiCCS accepted-row check](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualStep.lean:312), [existing phase composition](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Layout/Stage1/AccumulatorSemantics.lean:1101).

Goal A retains the existing pilot/PiCCS state and context checks and their precise proof assumptions. This does not move the still-open full arbitrary-assignment selected-context theorem into A. That theorem remains part of B's full F′ acceptance result. Keeping this distinction preserves both a useful early finish and the final security requirement.

The current evidence workflow already reuses matching approved runs. A goal split needs no duplicate checker or replacement obligation map. Reuse depends on recorded source/input/gate/policy/runtime applicability. A policy or relation change can invalidate broad evidence and exact-cut reviews, so zero future reruns cannot be promised. An unchanged theorem remains reusable while its assumptions and dependencies remain valid. [Run reuse](/Users/nicarq/starstream/develop/nightstream-clean-up/scripts/lean_graph/checkpoint.py:11), [applicability checks](/Users/nicarq/starstream/develop/nightstream-clean-up/scripts/lean_graph/records.py:14), [identity-change requirements](/Users/nicarq/starstream/develop/nightstream-clean-up/FPRIME_STAGE1_GOAL.md:534).

This recommendation was rechecked against the unchanged source, the exact PiCCS targets, native digest mapping, phase composition, and evidence applicability rules. No new implementation or test run was needed for this revision. Production code, owner goals, and approval status remain unchanged.

## Comparison of the original finish lines

The following inventory explains the scope of each original proposal. These are alternative completion boundaries over shared code, not four independent workstreams.

| Proposal | Position of Goal A today | Definite remaining work in A | Work left to B |
|---|---|---|---|
| 1. Native SuperNeo | Native fixed-arity prover/verifier, all reductions, exact sampler, and useful tests exist. Complete native/Lean conformance and authenticated input binding are not closed. | Define and enforce the native statement/context binding; align prior-digest sources; close the complete native interface against Lean on valid inputs/openings; settle transcript endpoints. | Arbitrary-assignment circuit proof, F′ lifecycle composition, selected package integration, terminal implementation and backend run. |
| 2. PiCCS compiler/conformance | Exact phase targets and all distinct registered check implementations exist. Required current artifact/input/review evidence cannot be verified here. | Recover or regenerate the exact payloads and input bundle; obtain the required current reviews and approved checks; repair only failures that those checks establish. | PiRLC/PiDEC cumulative conformance, remaining actual-assignment proof, full context/lifecycle/terminal connection, production integration. |
| 3. SuperNeo verifier circuit | An accumulator relation and conditional composition theorem exist. Actual PiCCS is proved; the selected complete accumulator proof remains open. | Actual sampler equality, typed 17-source parent equality, exact PiDEC acceptance, decoded child/proof/output agreement. | Complete application/state/context/F′ composition, final terminal acceptance and production integration. |
| 4. Full Lean F′ package | Most component builders, physical proofs, fixed point, domain proof, terminal definitions and package construction exist. The completed full-step theorem is canonical-assignment only. | All of proposal 3's proof work, plus full arbitrary-assignment StepHoldsFor, selected-context binding, and intended security/terminal composition. | Exact implementation links, selected package as the only production relation, and approved proof execution. |

This is a dependency assessment. It is not an estimate of remaining days or a completion percentage.

## Review contract and assumptions

The task was to determine where the current implementation stands for each proposed A/B split. Success means identifying concrete reusable code, missing proofs or callers, actual test results, and a completion condition for each Goal A.

The final Stage 1 target, fixed `b = 2`, `k_rho = 16`, `B = 65536` profile, Poseidon2 binding, and Lean authority remain the owner requirements. The split can define an earlier component milestone without claiming that all Stage 1 requirements are met. Existing assurance tiers already permit that distinction. [Goal tiers](/Users/nicarq/starstream/develop/nightstream-clean-up/FPRIME_STAGE1_GOAL.md:442).

No separate package, alternate relation, new transcript, or broad refactor was assumed necessary. The inspected code supplies existing boundaries for all four proposals. Those extra architectural changes were rejected unless a concrete missing claim requires one.

This review did not edit production code, goals, authority records, identities, or artifacts. It did not grant a phase review or conformance approval.

## Proposal 1: executable SuperNeo, then the recursive system

### What can be reused

The existing [fixed native interface](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-fold-clean/src/paper/nifs/fixed.rs:90) implements one fresh CCS input and a fixed-size running accumulator. Its prover calls PiCCS, PiRLC and PiDEC in order. Its verifier repeats those checks and returns claims without witnesses. It rechecks the carried parent cache through PiDEC before replaying the next proof. [Native verifier](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-fold-clean/src/paper/nifs/verifier.rs:33).

The current sampler implements the fixed 16-bit schedule, rejects candidate 65535, uses the first 54 accepted coefficients, and returns an error on shortfall. The older goal text about a three-bit padded sampler is stale. PiDEC's constructive Lean decision and strict parent-bound rejection are also already implemented.

These native calls do not need to prove that their input relation verifies its own folding verifier. The main arbitrary-assignment F′ proof can therefore remain in B.

### What prevents A from being closed now

**The native input contract needs authentication, not only shape checks.** Lean's `ProductionKey.priorDigest` decodes the digest from the fresh public input. Its public transcript blocks ignore the running statement because the pilot's state hash binds that statement elsewhere. Rust takes the prior digest from `running[0].fold_digest` and checks that the other running claims carry the same digest. These are different input sources. [Lean boundary](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/ProductionKey.lean:100), [Rust boundary](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-reductions/src/engines/pi_ccs_joint_protocol.rs:75).

Equal supplied digest fields do not establish that the digest hashes the complete ordered running state and selected context. The selected circuit has a proof connecting the actual prior state and running input to the fresh public hash. [Existing circuit connection](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualPiCCSInputs.lean:97). A native A must retain a corresponding verifier-checked boundary, or explicitly export a core whose caller must provide authenticated inputs. The latter is a valid component contract, but cannot be called a standalone secure system for arbitrary caller-selected statements.

**The current phase parity is narrower than complete native parity.** The PiRLC fixture exercises sampling, public combination and engine verification, rather than the complete native NIFS entrypoint. Native PiRLC also performs a projection-digest absorption and beta squeeze after sampling. The Lean parity result reports the sampler's final state. The code shows a schedule difference; this review did not reproduce different folded claims. PiDEC takes no transcript, and the next PiCCS resets it. Goal A needs one exact definition of which transcript endpoint belongs to its interface. [Native projection](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-fold-clean/src/paper/reductions/pi_rlc.rs:668), [Lean endpoint](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PiRLCParity.lean:191).

**Valid nonzero conformance remains distinct from synthetic verifier vectors.** Some stored Lean fixtures deliberately use synthetic opening/evaluation data. The positive-input checker and independent opening machinery exist, but their current evidence bundle is absent. The full composed native path must use the same validated relation, key, inputs, proof, and returned outputs as Lean.

### A completion condition and next action

Goal A closes when the fixed native interface has an explicit checked input-binding contract, the intended deterministic/security boundary, and complete Lean/native conformance on the required valid inputs and openings. Agreement must cover the PiCCS result, sampled challenges, combined parent, PiDEC children and returned accumulator, with exact transcript endpoints.

Within the compatible sequence, any native digest repair must be driven by the active phase's existing conformance contract. Use the current selected state serializer and context, establish the exact fresh-public/native-digest mapping, and feed that same value to both verifier semantics. Do not add a separate standalone verifier API or make this an independent prerequisite for PiCCS closure when its existing checks already supply the mapping.

**Assessment:** substantial reusable implementation; a viable earlier component goal, with real binding and integration work remaining. A secure native Goal A is larger than a native round-trip demonstration.

Full source audit: [NATIVE_SUPERNEO.md](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-goal-splits/NATIVE_SUPERNEO.md).

## Proposal 2: PiCCS compiler/conformance, then remaining Stage 1

### What can be reused

The exact phase target statements already exist in [EvidenceTargets.lean](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/tests/EvidenceTargets.lean:18), with closing calls in [EvidenceAcceptance.lean](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/tests/EvidenceAcceptance.lean:7). They cover pilot, PiCCS, and the actual public boundary. The public target proves PiCCS phase acceptance, decoded running-input equality, the fresh public hash, and the next hash. It does not require the full recursive StepHoldsFor result.

The separate full Stage 1 target remains `null` in the registry. It is not part of PiCCS phase closure. [Registry](/Users/nicarq/starstream/develop/nightstream-clean-up/scripts/lean_graph/obligations.json:297).

The audit traced actual check bodies for physical A/B/C equality, all 14 logical matrices, independent raw assignment evaluation, complete Lean/Rust PiCCS values, independent fresh and child openings, both child evaluation points, caller handoff, and registered mutations. No missing PiCCS protocol implementation was established by this audit.

### What remains

The registry defines seven PiCCS obligations: `pilot-assignment`, `piccs-assignment`, `piccs-public-assignment`, `compiler-coverage`, `pilot-conformance`, `piccs-conformance`, and `piccs-coverage`.

Together they select 95 distinct gates: 91 conformance gates plus four compiler/metadata gates. They use 22 named input keys and exact target/decomposition and formula/branch reviews. These counts are read from the current registry, not proposed limits. Matching approved gate results are reused; the overlapping coverage obligation does not require duplicate execution.

The missing work is definite at the evidence boundary:

- Restore or regenerate the full package, physical expansion, binding and other required payloads. Several local files are only Git LFS pointers.
- Restore or reconstruct the complete input bundle, including actual-child recursive data and independent opening caches. The sibling evidence archive is absent here.
- Select one current source/input/policy snapshot. Historical passes on different snapshots cannot be combined into current approval.
- Obtain the required independent reviews and approved current checks. Diagnostic runs alone do not close the status.
- Repair any code defect established by those current checks or required reviews. The audit cannot promise there will be none.

### A completion condition and next action

Goal A closes when the status calculation reports both `phase_statuses.PiCCS.Compiler-closed = true` and `phase_statuses.PiCCS.Conformance-closed = true` for the selected source and inputs. It does not require global Stage 1 closure. [Status calculation](/Users/nicarq/starstream/develop/nightstream-clean-up/scripts/lean_graph/records.py:162).

The first action is to recover the existing evidence archive and payloads, then determine which recorded results remain applicable. If they cannot be recovered or no longer apply, regenerate the needed bundle from the chosen source before approved checks and reviews. The existing check implementations and recorded diagnostic history are reusable work, not permission to mark A complete.

**Assessment:** the clearest existing first completion condition. Its known remaining work is chiefly artifact/evidence completion and review, with code repair conditional on actual failures. It is neither “reviews only” nor a claim that PiCCS is already production-complete.

Full source and gate audit: [PICCS_CONFORMANCE.md](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-goal-splits/PICCS_CONFORMANCE.md).

## Proposal 3: SuperNeo verifier circuit, then HyperNova lifecycle

### What can be reused

There is an existing mathematical boundary: `Accumulator.Holds` is exact NIFS verification returning the exact running output. The theorem `phases_imply_holds_of_wiring` composes the three phases when their proof, inputs, transcript state, PiDEC attempt and output agree. [Composition](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Layout/Stage1/AccumulatorSemantics.lean:1101).

The selected arbitrary-assignment results already prove the exact PiCCS check, product sums using actual assignment values, the actual PiDEC phase predicate, application transition, hash slots and base branch. These are useful completed pieces.

### What remains in A

Goal A still needs to prove:

- The product challenges decoded from the assignment equal the exact verifier sampler output, from the actual PiCCS outgoing transcript state.
- The 17-source sums use the exact typed PiCCS inputs and form the verifier-computed parent.
- The actual PiDEC child messages populate the same NIFS proof, and that exact PiDEC attempt passes.
- The verifier's returned running value equals the value carried by the actual running-transition decoder.

The current `ActualStep` equivalence exposes the last two conditions directly. It proves neither condition just by stating the equivalence. [Current reduction](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualStep.lean:350).

Existing canonical routes prove much of the transport for a constructed raw packet. They still take `Encodes` or related agreement facts. The selected arbitrary-assignment theorem must derive the needed value equalities from its actual rows and boundary. Scope assumptions such as variables being below an allocation offset are ordinary layout facts; their presence alone is not a security defect.

### A completion condition and next action

Goal A closes with a selected-row theorem that yields exact NIFS acceptance and computed-output equality for proof/input/output values decoded from that same arbitrary assignment, under only the explicit phase-interface boundary. It must not take canonical construction or missing representation facts as caller assumptions.

The first proof is the actual sampler agreement, using `ActualPiRLCValues.challenge` and `piRlcChallenges_eq_key_of_initialState`. The existing accumulator boundary can remain in place. Extracting a new package is not required.

Goal B retains the complete application, state, selected-context, F′ and terminal composition, plus production integration. Some existing rows already mix state binding with PiCCS, so the split should specify an interface contract rather than physically move those rows.

**Assessment:** a good proof ownership boundary, but it keeps the main present proof difficulty in A. It does not provide the same early relief as proposals 1 or 2.

Full theorem/assumption audit: [FORMAL_BOUNDARIES.md](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-goal-splits/FORMAL_BOUNDARIES.md).

## Proposal 4: complete Lean F′ package, then production execution

### What can be reused

The concrete F′ relation, phase assemblers, eight-child Stage 1 composition, completeness machinery, physical preservation, matrix-program exactness and canonical package construction exist. The selected recursive fixed point and domain bound are proved. They need not be invented again.

The outer terminal semantics also exist. They check the base case or the recursive public link, all 16 running CE openings and the fresh CCS opening against the selected relation. Terminal metadata adds no circuit rows. The missing work is not another terminal folding phase. [Terminal semantics](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PerApplicationTerminal.lean:35), [metadata](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/TerminalPackage.lean:20).

### What remains in A

The existing full-step closure is restricted to `(bound raw).assignment`. A must close every proposal 3 proof connection, compose the unchanged full StepHoldsFor for arbitrary accepted assignments, and establish the selected-context connection at its real acceptance boundary. The intended security and terminal use must be connected with the owner-permitted assumptions. A new probability formalization is not automatically required.

### What remains in B

The production package adapter has no public lifecycle caller. It exposes a header and row visits, but ordinary lifecycle preprocessing expects an actual matrix evaluator/cache. The selected package needs a verified evaluator connection, exact witness-to-logical-assignment transport, native proof input conversion, context/state binding, and terminal verification using the same package.

Existing terminal-induction preprocessing consumes a different native-CCS manifest. It does not consume `Poseidon2HashChainV1Package`. The file named `package_production.rs` tests loader pins and mutations; it never runs the complete lifecycle.

The final backend also requires separate owner approval. Existing backend code cannot replace these implementation links.

**Assessment:** the least disruptive formal/production split, but A keeps the full current proof problem and B remains a real integration project. Deferring all phase conformance to B would change the owner's current work order; that is a separate choice, not implied by this review.

Full implementation trace: [PRODUCTION_PATH.md](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-goal-splits/PRODUCTION_PATH.md).

## What was executed

All current Rust tests used release mode, locked offline dependencies, an empty `RUSTC_WRAPPER`, and the project-required 300-second cap. There was one build/test process at a time. No PaperExact test path was executed.

| Target | Result in this review | What it establishes |
|---|---|---|
| `nightstream-fprime / package_loader` | 1 passed, 7 failed; build 22.02 s, tests 0.01 s. | Package-dependent checks fail on the local LFS pointer. These are not seven demonstrated loader defects. |
| `neo-fold-clean / nifs_fixed` | 3 passed; tests 0.03 s. | Fixed-size native round trip and malformed accumulator rejection on the existing zero toy fixture. |
| `neo-fold-clean / nifs_r1cs_isolated` | 1 failed before folding: empty running input rejected. | This positive test uses a stale empty-accumulator convention. It does not establish a nonzero arithmetic defect. |
| `neo-fold-clean / nifs_round_trip` | 9 passed; tests 0.06 s. | Native replay and PiCCS/PiDEC/parent mutation rejection on the tested scope. |

The shared native release build took 49.45 s. The isolated R1CS failure caused Cargo to stop before the round-trip target; the latter was run separately. Its incremental build took 0.11 s. The shared toy constructor ignores its seed and returns a zero assignment, so the positive toy cases do not establish production nonzero conformance. [Fixture](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-fold-clean/tests/support/mod.rs:30).

The Lean library, registered axiom library, and review root sketch were checked earlier in this conversation on this unchanged commit. They were not rebuilt for this source-only comparison. The current review does not claim that the missing full root theorem has been proved.

Logs and exact commands: [VALIDATION.md](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-goal-splits/VALIDATION.md).

## How the split affects future changes

A phase theorem can remain reusable while Goal B continues. Conformance evidence is tied to its actual source, inputs and relation identity. If B changes those dependencies, applicable A checks must run again. Calling A closed cannot make its evidence valid for an unreviewed changed package.

To address the concern about never finishing while preserving reuse, Goal A uses the existing independently computable PiCCS status. Goal B extends that exact phase result through the remaining selected pipeline. Native conformance work supports those phase closures and is not a competing product project.

No proposal requires discarding the completed phase proofs, fixed-point results, or native folding implementation.
