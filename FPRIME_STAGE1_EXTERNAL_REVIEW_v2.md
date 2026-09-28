on course

# Nightstream F′ Stage 1 independent review v2

Review date: 2026-08-28

Reviewed snapshot:

- branch: `nico/f-prime-constraints-cuda-formal`;
- Git HEAD: `050a01ee24cd3d6f7ddc98f5dc2c77a1cafe61aa`;
- snapshot end: `2026-08-28T18:46:26Z`;
- permitted tracked review-scope diff SHA-256:
  `0bc03145e77f60524cd40255f7154234a3c62de027fcc64d93efa645c3c7703a`;
- 49 tracked modifications and four untracked files were present at that
  snapshot;
- compact `2^26` package SHA-256:
  `e0feeaa17f1a4acaded22eb15ff3af6c20aa4c8ee400b629004ba99691efe4b8`;
- expanded `2^26` package SHA-256:
  `c1c1c62975cfcf3c56bce5ab30dfda79b08c7b24e017a8b5e096de778e4fbad9`.

The owner changed the Stage 1 limit during this review. The active decision is
`decisions/fprime-stage1-domain-2p28.md:5-16`: a 28-variable cube, exactly 28
PiCCS SumCheck rounds, and a maximum joint domain of `2^28`. This decision
keeps `b = 2`, `k_rho = 16`, `B = 2^16`, 17 PiRLC inputs, 16 PiDEC children,
14 matrices, and Poseidon2-only binding.

The checked-in packages and parity fixtures still describe the retired
26-round cut. The active decision says that those artifacts and identities
are not evidence for the `2^28` profile. This review follows that rule.

## Verdict

The architecture is on course. The active `2^28` migration is not yet a
coherent compiler cut. It does not build, its axiom gate does not finish, and
its Rust package tests consume the old 26-round package. No active-profile
phase is **Compiler-closed**, **Conformance-closed**, or **Production-closed**.

The local phase design remains strong. It has one `FormalCircuit` type,
separate `Eval_K` and `Eval_A`, opaque child circuits, parent-owned wiring,
explicit physical owners, a 16-bit fail-closed PiRLC sampler, and strict PiDEC
parent rejection. These facts support the verdict. They do not close the
current profile.

## Material findings

### F1 — The `2^28` migration is not one buildable and testable cut

**Claim.** Lean, Rust, the owner files, the audit ledger, packages, and parity
tests do not yet use one 28-round profile.

**Direct evidence.**

- `decisions/fprime-stage1-domain-2p28.md:10-16` selects 28 rounds and retires
  all `2^25` and `2^26` artifacts.
- `formal/nightstream-fprime/NightstreamFPrime/Lifecycle/Types.lean:28-30`
  and `crates/nightstream-fprime/src/package/v1_1.rs:24-30` now select 28
  rounds. `crates/nightstream-fprime/src/package.rs:44-45` now permits a
  `2^28` joint domain.
- `FPRIME_STAGE1_GOAL.md:150` says `cubeVariables 28`, but
  `FPRIME_STAGE1_GOAL.md:186-187` still says 25 variables and 25 rounds. Its
  PiCCS tree still says a fixed 25-round chain at
  `FPRIME_STAGE1_GOAL.md:342-352`.
- `formal/nightstream-fprime/CONSTRAINT_TREE.md:34-58` still records 26
  rounds. Its package hashes and cumulative ledger are also the 26-round
  values.
- The active Lean build stopped in three successive places as the concurrent
  migration moved: first `Lifecycle/XOut.lean:128`, then
  `Lifecycle/PiCCS/v1_1/StatementAbsorption.lean:735` and
  `Layout/PilotProduction.lean:512,536,600`, and finally
  `Layout/Stage1/PiCCSInputs.lean:239,299,903`.
- The later axiom-gate run advanced further, but stopped in
  `Layout/Stage1/PiCCSStarts.lean:75,104,119,159,179` on old offsets and old
  start lists.
- The current strict loader passed 4 of 14 tests. Ten tests failed because the
  stored package has the old PiCCS input segments or transcript shape.
- The current exact matrix target failed before matrix comparison with
  `Invalid("PiCCS v1_1 input segments")`.
- `crates/neo-fold-clean/tests/nifs/pi_ccs_lean_nonzero_parity.rs:23` and
  `crates/neo-fold-clean/tests/nifs/pi_rlc_lean_nonzero_parity.rs:19` still
  hard-code 26. They pass the retired fixtures and cannot provide 28-round
  evidence. PiDEC imports the new round count and fails on the old package.

**Exact missing connection.** One unchanged `2^28` source cut must build and
must emit one new package and all dependent nonzero fixtures. The owner goal,
architecture contract, decision, constraint tree, Lean constants, Rust
constants, fixture decoders, expected identity, and cumulative ledgers must
name that same cut. The expected identity must be pinned only after every
identity-change gate passes.

**Blocks.** This blocks **Compiler-closed** and **Conformance-closed** status
for every active-profile phase. It also blocks full Stage 1.

### F2 — The emitted package is still a prefix, not the final F′ relation

**Claim.** The project has a good generic 14-matrix authority type, but it
does not construct the one final relation, application, key, and lifecycle
object required by Stage 1.

**Direct evidence.**

- `formal/nightstream-fprime/NightstreamFPrime/Layout/ProductionRelation.lean:226-321`
  defines `Plan`, all 14 matrices, zero slot 13, exact row-image evaluation,
  and `Plan.logicalRelation`. Its header at lines 9-11 says that a compiler
  must still construct one complete plan.
- The only other `ProductionRelation.Plan` construction is the generic
  ordinary-row program in
  `Layout/ProductionRelation/OrdinaryRow.lean:113-177`. It is not a complete
  low-norm Stage 1 plan.
- `Layout/ProductionRelation/SourceCompiler.lean:4-12` explicitly does not
  select the retained set or affine rewrite schedule.
- `Export/Stage1/Data.lean:426-450` emits `terminal := none`.
- There is no `Lifecycle/Stage1/Formal.lean`; the only Stage 1 lifecycle file
  is `Lifecycle/Stage1/RunningTransition.lean`.
- `Export/Stage1/VerifierContext.lean:8-18` says that the current prefix does
  not contain the final logical relation or application.
- `Export/Stage1/VerifierContextCandidate.lean:71-89` uses the package identity
  for both relation and application authority words.
- `Export/Stage1/PiDECNonzero.lean:32-48` uses a zero-matrix relation and a
  zero Ajtai key for the phase fixture. The file says final integration must
  replace them.
- `Lifecycle/Relation.lean:31-80` provides a generic chain from the NIFS key
  to `StepHolds`, but `relation`, `ajtai`, `vk`, and the application `F` remain
  separate parameters. No theorem fixes them to the final package object.
- The accepted owner order in
  `decisions/piccs-phase-local-conformance-order.md:9-26` keeps PiCCS status
  open until final package → logical relation → production key →
  application → `StepHolds` → recursive fixed point exists and all
  gates run again.

**Exact missing connection.** Construct the final retained assignment and
14-matrix `Plan`; connect it to `ProductionKey.key`, the exact application,
base and recursive behavior, terminal checks, `StepHolds`, the verifier
context, one emitted package, and the recursive fixed point.

**Blocks.** This blocks PiCCS **Compiler-closed** status under the owner
decision, every **Production-closed** status, and full Stage 1.

### F3 — Row-owner and column-owner mutation checks do not test rejection

**Claim.** The matrix target names 144 row-owner and 77 column-owner mutation
checks, but those checks only prove that a changed row vector differs from
the original row vector.

**Direct evidence.**

- `crates/nightstream-fprime/src/bin/check_package_conformance/owner_mutations.rs:96-126`
  selects one nonzero row per owner and side. It deletes or changes one term,
  then calls `assert_ne!(actual, changed)`. It does not pass the changed row to
  the package loader, exact comparator, or assignment evaluator.
- `owner_mutations.rs:159-205` changes one owned column and again calls only
  `assert_ne!`.
- `check_package_conformance/support.rs:1324-1329` counts these inequalities
  as mutation checks. The real semantic input mutations at lines 1363-1423
  do execute the witness program and require rejection; those checks are a
  different and valid class.
- The package loader has a real whole-package identity mutation rejection,
  but it is not one failure test for each row and column owner family.
- `FPRIME_STAGE1_GOAL.md:471-487` requires mutations to every authoritative
  row and column family to cause failure.

**Exact missing connection.** For every row and column owner family, feed the
mutated object to an independent checker that is expected to reject. A row
mutation must fail exact matrix equality or raw-assignment satisfaction. A
column mutation must fail the owner map, public map, matrix equality, or raw
assignment check. A simple inequality is not a rejection gate.

**Blocks.** This blocks **Conformance-closed** status for all phases, even
after the `2^28` artifacts are regenerated.

### F4 — The committed-statement and security links stop before raw authority

**Claim.** The deterministic collision lemmas are useful, but they do not yet
form the committed-statement reduction or the complete SuperNeo security
composition.

**Direct evidence.**

- `Lifecycle/VerifierContext.lean:46-86` hashes four raw authority word lists
  into four component digests.
- `Layout/Stage1/PiCCSSecurity.lean:23-27` defines
  `ContextDigestCollision` only for the fixed `Descriptor`. Two different raw
  word lists with the same component digest produce the same descriptor, so
  that raw collision is not named by this event.
- `PiCCSSecurity.lean:99-125` takes a context descriptor, a state preimage,
  and a transcript replay input as independent arguments. It has no premise
  that the replay input is the exact package-derived view of that state,
  proof, and output.
- The two committed-statement theorem names appear only in
  `formal/nightstream-fprime/tests/Axioms.lean:989-990`; no package or Stage 1
  theorem consumes them.
- `PiCCSSecurity.lean:8-12` states that the complete security composition is
  outside that module. No production theorem connects package acceptance to
  the full PiCCS, PiRLC, uniqueness, PiDEC, and named-failure chain.

**Exact missing connection.** Add domain-separated component-collision events
over raw relation, application, NIFS-key, and commitment-key words. Build one
package-derived replay view and prove that its prior state, fresh statement,
round messages, and output are the values constrained by the package rows.
Then compose this result with the exact SuperNeo reduction and the final F′
lifecycle theorem.

**Blocks.** This blocks the Stage 1 security-composition theorem and full
Stage 1. It does not block phase-local arithmetic work.

### F5 — The `2^28` headroom result is not a full fit theorem

**Claim.** The new limit removes the old direct-plan contradiction, but the
complete low-norm relation and cumulative fixed-point footprint remain open.

**Direct evidence.**

- `Export/Stage1/DirectPoseidonFootprint.lean:74-84` now states that
  108,068,374 retained S-box coordinates fit below `2^28`.
- `decisions/fprime-stage1-domain-2p28.md:22-31` says that this count excludes
  final outputs, non-Poseidon values, and remaining Stage 1 phases. It also
  says that it is not a complete fit proof.
- The active source has a local pilot theorem for 13,600,754 rows and a
  13,692,624 joint domain in `Layout/PilotProduction.lean:503-625`.
- The active PiCCS composition source states a standalone 5,281,269-row and
  5,281,051-column phase in
  `Layout/PiCCS/v1_1/Composition.lean:652-685`.
- The cumulative Stage 1 files still contain the old 26-round endpoints and
  old leaf arrays. The current build and axiom gate stop before those ledgers
  can be accepted for the 28-round cut.

**Exact missing connection.** Produce one accepted cumulative theorem for
the exact 28-round package and extend it through the final low-norm plan,
application, output hash, terminal relation, and recursive fixed point. The
final joint domain must be at most `2^28`.

**Blocks.** This blocks the full Stage 1 fixed-point and domain obligations.

### F6 — The package is not the sole production path, and CI checks the wrong Lean package

**Claim.** The validated package remains a bridge. A native lifecycle route
is public, and repository CI does not run the active F′ Lean package.

**Direct evidence.**

- `crates/neo-fold-clean/src/lib.rs:101-128` publicly exports native lifecycle
  entry points and native relation types. They remain reachable beside the
  package bridge.
- The package bridge exists under
  `crates/neo-fold-clean/src/frontends/r1cs_f_prime/ivc/package_v1_1.rs`, but
  there is no package-only public lifecycle theorem or reachability gate.
- `.github/workflows/ci.yml:57-81` runs Lean assurance in
  `formal/nightstream-lean`. `git ls-files formal/nightstream-lean` returns
  zero tracked files. CI does not run `formal/nightstream-fprime`.

**Exact missing connection.** Make the validated package relation the only
reachable production relation. Add the active package static, build, and
axiom gates to CI. Do this only after the final `2^28` identity passes the
conformance gates.

**Blocks.** This blocks every **Production-closed** status and full Stage 1.

The final proof-backend obligation remains open. No proof backend was run.
Backend acceptance cannot establish matrix equality, semantic correctness,
value equality, or assignment satisfaction.

## Formula coverage map

The following table is the independent formula trace for every current
top-level leaf. The path prefixes are:

- `LPC`: `formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiCCS/v1_1/`;
- `QPC`: `formal/nightstream-fprime/NightstreamFPrime/Layout/PiCCS/v1_1/`;
- `LPR`: `formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiRLC/v1_1/`;
- `QPR`: `formal/nightstream-fprime/NightstreamFPrime/Layout/PiRLC/v1_1/`;
- `LPD`: `formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiDEC/v1_1/`;
- `QPD`: `formal/nightstream-fprime/NightstreamFPrime/Layout/PiDEC/v1_1/`;
- `EXP`: `formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/`.

For PiCCS, `LPC/Formal.lean:1002-1222` connects the exact `Accepted` predicate
to the parent circuit, and lines 1224-1482 give conjunct coverage.
`LPC/Completeness.lean:221-296` gives the reverse witness direction. PiRLC
uses `LPR/Semantics.lean:428-650`, `LPR/Formal.lean:207-333`, and
`LPR/Completeness.lean:451-518`. PiDEC uses
`LPD/Semantics.lean:93-250`, `LPD/Formal.lean:169-398`, and
`LPD/Completeness.lean:249-276`.

| Paper obligation → Lean predicate → `FormalCircuit` | Theorem and parent wiring | Physical owner → package rows | Exact Rust matrix → independent assignment → parity → production |
|---|---|---|---|
| PiCCS statement and prior claim → `StateBinding.SpecHolds` / statement binding circuit | `LPC/StatementBinding.lean`; PiCCS parent coverage | `QPC/Leaves/StatementBinding.lean` → `EXP/PiCCSInvocations.lean` | Retired `2^26` evidence only → active `2^28` open → production open |
| PiCCS digest-only statement absorption → transcript prefix predicate → statement absorption circuit | `LPC/StatementAbsorption.lean`; exact parent state handoff | `QPC/Leaves/StatementAbsorption.lean` → PiCCS invocation rows | Retired evidence only → active open → production open |
| PiCCS `alpha`/`gamma` derivation → labelled Poseidon2 schedule → challenge circuit | `LPC/ChallengeDerivation.lean`; parent challenge wiring | `QPC/Leaves/ChallengeDerivation.lean` → PiCCS invocation rows | Retired evidence only; current source uses 29 samples → active open |
| PiCCS causal round transcript → message-before-challenge predicate → one indexed circuit | `LPC/RoundTranscript.lean`; parent state order | `QPC/Leaves/RoundTranscript.lean` → PiCCS invocation rows | Retired 26-round evidence only → 28-round evidence open |
| PiCCS initial claim → exact initial SumCheck claim → initial-claim child | `LPC/InitialClaim.lean`; parent claim wiring | `QPC/Leaves/InitialClaim.lean` → PiCCS arithmetic rows | Retired evidence only → active open |
| PiCCS SumCheck → one generic degree-9 round composed 28 times → chain circuit | `LPC/SumcheckChain.lean`; fixed-chain parent wiring | `QPC/Leaves/SumcheckChain.lean` → PiCCS arithmetic rows | Retired 26-round evidence only → active open |
| PiCCS separate `Eval_K` terminal → Pad-family terminal predicate → terminal circuit | `LPC/EvalKTerminal.lean`; separate parent field | `QPC/Leaves/EvalKTerminal.lean` → PiCCS arithmetic rows | Retired evidence only → active open |
| PiCCS separate `Eval_A` terminal → 14 matrix-family terminal predicates → terminal circuit | `LPC/EvalATerminal.lean`; indexed parent fields | `QPC/Leaves/EvalATerminal.lean` → PiCCS arithmetic rows | Retired evidence only → active open |
| PiCCS CCS terminal → exact CCS polynomial term → CCS child | `LPC/CcsTerminal.lean`; parent terminal wiring | `QPC/Leaves/CcsTerminal.lean` → PiCCS arithmetic rows | Retired evidence only → active open |
| PiCCS strict norm terminal → centered norm term → norm child | `LPC/NormTerminal.lean`; parent terminal wiring | `QPC/Leaves/NormTerminal.lean` → PiCCS arithmetic rows | Retired evidence only → active open |
| PiCCS final joint identity → separate terminal identities → final-identity circuit | `LPC/FinalIdentity.lean`; complete terminal conjunction | `QPC/Leaves/FinalIdentity.lean` → PiCCS arithmetic rows | Retired evidence only → active open |
| PiCCS 17 reduced outputs and outgoing state → output binding circuit | `LPC/OutputBinding.lean`; parent output coverage | `QPC/Leaves/OutputBinding.lean` → PiCCS package completeness | Retired evidence only → active open |
| PiRLC exact 17-input binding → `InputBinding.SpecHolds` → zero-row shared child | `LPR/InputBinding.lean`; parent source order | `QPR/Leaves/InputBinding.lean` → PiRLC package completeness | Retired prefix evidence only → active p28 handoff open |
| PiRLC Poseidon2 absorption, 16-bit bounded sampler, membership, and first 54 accepted values → sampler chain | `LPR/TranscriptAbsorption.lean`, `DigestWindow.lean`, `Sampler.lean`, `SamplerChain.lean`; parent state order | `QPR/Leaves/{TranscriptAbsorption,DigestWindow,DigestLane,First54}.lean` → sampler rows | Primitive sampler 2/2 passes; cumulative p28 package evidence open |
| PiRLC indexed commitment combination → 17-source Horner relation → combination family | `LPR/CombinationFamily.lean`; parent indexed wiring | `QPR/CombinationFamily.lean` → PiRLC combination rows | Retired evidence only → active p28 handoff open |
| PiRLC indexed public-input combination → 17-source relation → combination family | Same generic predicate and parent index map | `QPR/CombinationFamily.lean` → PiRLC combination rows | Retired evidence only → active open |
| PiRLC indexed separate `Eval_K` combination → 17-source relation | Same generic child; separate family wiring | `QPR/CombinationFamily.lean` → PiRLC combination rows | Retired evidence only → active open |
| PiRLC indexed `17 × 14 Eval_A` combination → separate matrix-family relation | Same generic child; matrix and source indices owned by parent | `QPR/CombinationFamily.lean` → PiRLC combination rows | Retired evidence only → active open |
| PiRLC output claim and outgoing state → output binding circuit | `LPR/OutputBinding.lean`; final partial equals output | `QPR/Leaves/OutputBinding.lean` → PiRLC package completeness | Retired evidence only → active open |
| PiDEC parent claim binding → `InputBinding.SpecHolds` → zero-row shared child | `LPD/InputBinding.lean`; parent claim wiring | `QPD/Leaves/InputBinding.lean` → PiDEC package completeness | Retired evidence only → active p28 package fails |
| PiDEC strict parent `< 2^16`, 16 signed digits, range, and recombination → public split circuit | `LPD/PublicInputSplit.lean`; `Spec/Folding/PiDEC/PaperVerifier.lean:81-137,222-317`; parent wiring | `QPD/Leaves/SignedSplitScalar.lean` and `QPD/PublicInputSplit.lean` → PiDEC arithmetic rows | Retired evidence only; current accepted path still proves no `fallbackDigit` use → package evidence open |
| PiDEC commitment recomposition → radix-two parent/children equality → recomposition child | `LPD/CommitmentRecomposition.lean`; parent index wiring | `QPD/CommitmentRecomposition.lean` → PiDEC arithmetic rows | Retired evidence only → active open |
| PiDEC separate `Eval_K` recomposition → 16-child equality → recomposition child | `LPD/EvalKRecomposition.lean`; separate parent field | `QPD/EvalKRecomposition.lean` → PiDEC arithmetic rows | Retired evidence only → active open |
| PiDEC separate `Eval_A` recomposition → `16 × 14` equality → recomposition child | `LPD/EvalARecomposition.lean`; matrix and child index wiring | `QPD/EvalARecomposition.lean` → PiDEC arithmetic rows | Retired evidence only → active open |
| PiDEC 16 child claims and unchanged outgoing state → output binding circuit | `LPD/OutputBinding.lean`; complete parent output coverage | `QPD/Leaves/OutputBinding.lean` → PiDEC package completeness | Retired evidence only → active open → production open |

Every leaf has a semantic, circuit, theorem, parent-wiring, and physical-owner
path. The open link for every row is the active `2^28` package and production
path. F3 adds a separate mutation-gate defect after package regeneration.

## Per-phase assurance

| Phase | Compiler-closed | Conformance-closed | Production-closed | Current reason |
|---|---|---|---|---|
| Pilot | No | No | No | Local `2^28` pilot counts exist, but the full active package does not build; pilot mutation target is 2/3 on the old fixture |
| PiCCS | No | No | No | Owner decision keeps status open; active 28-round starts, package, matrices, assignment, and parity are not closed |
| PiRLC | No on the active profile | No | No | Local phase design and sampler remain valid; its input must be the exact conformance-closed 28-round PiCCS output |
| PiDEC | No on the active profile | No | No | Local strict decision and recomposition design remain; the active package fails before PiDEC comparison |
| Running transition | No on the active profile | No | No | Its serialized running width changed from 45,893 to 45,897, but no accepted cumulative 28-round package exists |
| Full Stage 1 | No | No | No | Final plan, application, terminal, security composition, fixed point, final domain theorem, and package-only production route are open |

## Cumulative footprint ledger

There is no accepted cumulative `2^28` ledger on this snapshot. The active
source has these local declarations, but the cumulative build is red:

| Active source item | Rows | Columns or joint domain | Delta from retired source item | Lean evidence state |
|---|---:|---:|---:|---|
| Pilot | 13,600,754 | 13,692,624 | `+1,184` rows, `+1,192` columns | `PilotProduction.physicalRowCount_eq`, `physicalColumnCount_eq`, `jointDomain_le_twoPow28`; local module advanced through build |
| PiCCS standalone phase | 5,281,269 | 5,281,051 columns | `+45,074` rows, `+45,070` columns | `PiCCS.v1_1.Composition.physicalRowCount_eq_production`, `physicalColumnCount_eq_production`; cumulative Stage 1 offsets still fail |
| Direct retained Poseidon2 S-box coordinates | — | 108,068,374 | Old contradiction removed | `DirectPoseidonFootprint.directSboxCoordinates_le_cube`; not a full relation count |
| Pilot → PiCCS → PiRLC → PiDEC → running transition | Open | Open | Open | Existing cumulative files still contain retired endpoints; build and axiom gate stop before acceptance |
| Complete Stage 1 low-norm relation | Open | Open | Open | No complete `ProductionRelation.Plan`, fixed-point theorem, or final `≤ 2^28` theorem |

For comparison only, the last validated 26-round prefix had this ledger. It
is not active-profile evidence:

| Retired endpoint | Rows | Joint domain | Delta from preceding endpoint | Retired theorem |
|---|---:|---:|---:|---|
| Pilot | 13,599,570 | 13,691,432 | — | `PilotProduction` old cut |
| Pilot → PiCCS | 18,835,765 | 18,956,449 | `+5,236,195` rows, `+5,265,017` domain | `PilotPiCCS.cumulativeFootprints_eq` old cut |
| → PiRLC | 27,191,367 | 27,310,402 | `+8,355,602` rows, `+8,353,953` domain | `PilotPiCCSPiRLC.cumulativeFootprints_eq` old cut |
| → PiDEC | 27,216,639 | 27,374,284 | `+25,272` rows, `+63,882` domain | `PilotPiCCSPiRLCPiDEC.cumulativeFootprints_eq` old cut |
| → running transition | 27,537,894 | 27,649,646 | `+321,255` rows, `+275,362` domain | `PilotPiCCSPiRLCPiDECRunningTransition.cumulativeFootprints_eq` old cut |

The retired endpoint is below `2^28`, but it cannot prove that the 28-round
prefix or final relation fits.

## Rust migration surface

| Surface | Current state | Exact next evidence |
|---|---|---|
| Package profile and joint-domain guard | Rust constants are 28; stored package is 26 | Load a newly emitted 28-round package under a new expected identity |
| PiCCS transcript and proof shape | Loader rejects old vector; PiCCS parity test still hard-codes 26 | Replace hard-coded 26, regenerate nonzero Lean vector, compare all 28 rounds and states |
| PiCCS → PiRLC handoff | PiRLC test still hard-codes 26 and passes the retired pair | Consume the new PiCCS result and rerun all indexed PiRLC values and mutations |
| PiCCS → PiRLC → PiDEC handoff | PiDEC imports 28 and fails on old package, 0/3 | Regenerate all three fixtures under one package identity and rerun the cumulative handoff |
| 16-bit bounded PiRLC sampler | Current target passes 2/2; Rust uses the Lean schedule | Keep it fixed and rerun it with the final identity-dependent fixtures |
| Strict PiDEC parent bound and decision | Lean semantics and agreement theorem exist; accepted path excludes fallback | Rerun parent-bound, digit, range, recombination, output, and mutation cases on the 28-round package |
| Exact final matrices and raw assignment | Retired cut passed; active target fails before expansion | Compare every active A/B/C entry, then independently evaluate every active unpadded row and padded zero row |
| Row/column mutations | Current owner checks are inequalities only | Make every owner mutation cause checker rejection as described in F3 |
| Production lifecycle | Native route remains reachable | Make the final identity-bound package the only production relation |

## Bloat and policy sweep

- `scripts/validate.sh static` passes on the active dirty cut. It reports no
  forbidden old-package import, generated proof module, implicit root, mixed
  production profile, or file at or above 1,500 lines.
- `Circuit/Basic.lean:437` is the only `FormalCircuit` structure. The many
  circuit values all use this one type.
- The reviewed protocol-binding path uses Poseidon2. I found no second hash
  family in the package identity, verifier context, state, or transcript
  binding path.
- `Eval_K` and `Eval_A` remain separate in Lean, package inputs, Rust loader,
  and parity result types.
- PiRLC `InputBinding` remains frozen; its Git history has no migration edit
  after its initial phase work.
- PiDEC checks centered parent magnitude before the 16-child split. Its
  computable decision agrees with the predicate. The accepted path uses the
  bounded branch and cannot use `fallbackDigit`.
- The active decision file is untracked. The goal still contains stale
  25-round statements, and the constraint tree still contains 26-round
  statements. This is an authority and reproducibility defect, not code
  bloat.
- Three pre-existing untracked `.expected` files remain. They were preserved.
- `formal/nightstream-lean` has no status entry and was not built or changed.

## Executed validation record

All Lean commands used `formal/nightstream-fprime/scripts/validate.sh` and its
1,500-second cap. All Rust commands used release mode,
`RUSTC_WRAPPER=""`, and a 300-second cap. Only one Lean or Rust process was
started at a time. No proof backend ran.

The first group ran before the owner changed the profile. It is retained only
as a retired baseline:

| Retired 26-round target | Result |
|---|---|
| `validate.sh static` | Pass |
| focused `Export/Stage1/Package.lean` check | Pass |
| direct axiom audit | Pass; only `propext`, `Classical.choice`, `Quot.sound` |
| compact and expanded emit to `/private/tmp` | Pass; byte-identical to stored package; hashes shown above |
| strict package loader | 14/14 |
| compact-plan loader | 10/10 |
| exact matrices plus independent assignment | 1/1; all 27,537,894 unpadded rows checked |
| pilot / PiCCS / PiRLC / PiDEC comparisons | 3/3, 4/4, 3/3, 3/3 |
| 16-bit sampler | 2/2 |
| Poseidon2 primitive vectors | 2/2 |

Current 28-round migration results:

| Active target | Result |
|---|---|
| `validate.sh static` | Pass |
| focused `Export/Stage1/Package.lean` before dependency rebuild | Fail: missing current `Lifecycle/XOut.olean`; not treated as source evidence |
| `validate.sh build NightstreamFPrime`, round 1 | Fail: old state serialization length at `Lifecycle/XOut.lean:128` |
| same build, round 2 | Fail: old statement-absorption and pilot counts at `StatementAbsorption.lean:735` and `PilotProduction.lean:512,536,600` |
| same build, round 3 | Fail: old PiCCS input offsets and 26-round type at `PiCCSInputs.lean:239,299,903` |
| `validate.sh axioms` after more concurrent migration edits | Fail: old PiCCS start offsets and arrays at `PiCCSStarts.lean:75,104,119,159,179`; axiom audit did not finish |
| active emitter | Not reached because the active Lean package does not build; no repository artifact was written |
| strict package loader | Fail: 4/14 passed; ten fail on retired package shape or earlier validation errors |
| compact-plan loader | Fail: 5/10 passed; five fail before their intended checks because the package has old PiCCS segments |
| exact matrices plus independent assignment | Fail: 0/1; rejected old PiCCS input segments before comparison |
| pilot nonzero comparison and mutations | Fail: 2/3; mutation parser expected a 56-word point and read the old 52-word point |
| PiCCS comparison | 4/4, but invalid for the active profile because the test hard-codes 26 and reads the retired fixture |
| PiRLC comparison and PiCCS handoff | 3/3, but invalid for the active profile for the same hard-coded 26 reason |
| PiDEC cumulative comparison | Fail: 0/3; package rejected on old PiCCS segments |
| 16-bit sampler | Pass: 2/2 |
| Poseidon2 primitive vectors | Pass: 2/2 |

The three build rounds satisfy the project stop fuse. This review does not
chase later migration errors.

## Prior-review reconciliation

Independent findings were recorded before the prior reviews were read. The
following prior claims remain current:

- the complete 14-matrix plan is not instantiated;
- the final application, terminal relation, fixed point, security
  composition, and package-only production path are absent;
- raw verifier-context component collisions and the package-derived replay
  link are not in the security theorem;
- active CI does not run the current Lean package;
- an untracked source cut cannot be reproduced from HEAD alone.

The following prior claims are stale or need a narrower statement:

- The old `2^26` direct-footprint contradiction is stale. The 108,068,374
  retained S-box coordinates fit below `2^28`. The final fit is still open.
- The state layout is not the old 54-word public-input layout. It uses 270
  logical public words; the physical package has eight separate outer public
  columns.
- Lean does have a generic 14-matrix constructor. What is missing is the one
  final compiler plan and its package/key/lifecycle connection.
- Rust recomputes the verifier context from raw commitment setup words. The
  remaining defect is the missing final authority theorem and raw component
  collision composition.
- The Fable review's moving-cut concern applies again. The current cause is
  the owner-authorized `2^28` migration, not an error in the retired phase
  mathematics.

## Necessary next actions

- Finish the mechanical 28-round migration through every serialization
  length, offset, start table, footprint theorem, axiom root, Rust decoder,
  and parity test. Remove every active 25- and 26-round statement from the
  owner goal and audit ledger. Track the accepted `2^28` decision.
- Reach one unchanged cut where the Lean root and axiom gate pass. Emit the
  compact and expanded packages only to scratch first. Regenerate every
  identity-dependent nonzero fixture from that same source.
- Run exact A/B/C equality, the independent raw-assignment evaluator, the
  full PiCCS result, both cumulative handoffs, all rejection cases, and real
  owner-family mutation failures. Pin a new expected identity only after all
  these checks pass.
- Construct the complete low-norm retained assignment and one 14-matrix
  production plan. Extend the cumulative theorem through the application,
  terminal relation, recursive fixed point, and final `≤ 2^28` domain.
- Replace the placeholder relation/application context words and phase-local
  zero relation/key with the final package-selected objects. Prove the exact
  connection to `ProductionKey.key`, `StepHolds`, and the fixed point.
- Add the raw component-collision events, package-derived transcript replay
  theorem, and complete SuperNeo security composition.
- Make the validated package the only reachable production relation and run
  the active Lean gates in CI.
- Keep the final proof-backend obligation open until separate owner approval.

## Repository preservation record

- Initial status at `2026-08-28T18:00:40Z`: HEAD
  `050a01ee24cd3d6f7ddc98f5dc2c77a1cafe61aa`, no tracked modification, and
  three untracked `.expected` files.
- During the review, a concurrent process performed the owner-authorized
  `2^28` migration. The review did not make those source, test, decision,
  goal, specification, or prior-review changes.
- This review created only `FPRIME_STAGE1_EXTERNAL_REVIEW_v2.md`. It did not
  modify either prior review.
- Emitter output went only to `/private/tmp`. No checked-in package or fixture
  was written.
- `formal/nightstream-lean` remained unchanged and was not built.
- No commit, stage, reset, stash, removal, restore, or discard operation was
  performed.
- Review-snapshot status at `2026-08-28T18:46:26Z`: HEAD unchanged, 49 tracked
  modifications and four untracked files. The untracked files were the three
  pre-existing `.expected` files and `decisions/fprime-stage1-domain-2p28.md`.
- Final status observation for this review, at `2026-08-28T18:50:14Z`: HEAD
  unchanged, 53 tracked modifications and five
  untracked files. The fifth untracked file is this report. Concurrent
  migration edits after the named review snapshot were preserved and were not
  used to change the findings for that snapshot.
- The final SHA-256 values of the untouched prior reviews are
  `c46273217e7fc090a14e880edb3d6f4943959790918d2f843d9ebf29b81ecedf`
  and `b4280d2551a15abd2ac2d6d9a44ad21e4ef0b4a2da2c20fe377a3b3650b8acc6`.
