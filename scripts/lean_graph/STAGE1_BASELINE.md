# Current Stage 1 baseline closure

## Active goal: independent prover replay

The owner changed the active target on September 13: independently execute
the selected prover computations in Lean and Rust from the same setup,
public inputs and original witnesses. Rust results enter only comparison
checks. Preserve SuperNeo v1.2 and the selected Nightstream Goldilocks
profile (`b = 2`, `k_rho = 16`) and package. Finish and
measure **PiRLC first** before integrating PiDEC, PiCCS or HyperNova replay.
PiCCS starts with a measured first round on the actual witness when that
milestone begins. Full independence requires the composed phases.

The owner also requires a SuperNeo + HyperNova run with Nebula inactive.
Use the selected `Poseidon2HashChainV1Package` and plain commitments:
fresh, running and parent claims carry `adv = None`. The selected NIFS
resources contain no `LaneScheme`. Reject unexpected auxiliary commitments;
preserve the complete HyperNova state, transcript, witnesses and tails.
Nebula code being compiled into the crate is not Nebula relation execution.
The regression `selected_plain_step_rejects_auxiliary_commitments` passes:
the unchanged plain input is accepted, and zero/nonzero auxiliary tuples
on fresh, running and supplied-parent claims reject. See
`SELECTED_PLAIN_REPLAY.json` for the source hashes and bounded test command.

The local PiRLC checkpoint now passes: all 253,011,276 carrier coefficients
match the actual Rust parent, and a change to the last tail coefficient
rejects in its range. The prepared kernel and selected `honestResponse`
bridge pass the axiom audit and exact target check. Static/build/axioms and
the complete Rust test build pass. `PIRLC_WITNESS_REPLAY.json` in the
requirements review directory records the source hashes, commands, times
and memory. Release upload and fresh-download verification remain external.
PiDEC uses this Lean-computed parent.

PiDEC has two required results: every private digit, then every child
commitment and evaluation computed from those digits. The private target is
`LeanGraph.Targets.PiDECWitnessReplay`: total equality with the existing
scalar split, including rejection outside the strict bound and successful
output for every bounded input. The selected honest-witness bridge fixes
the message consumer. Preserve parent provenance from the PiRLC replay;
native children enter only the comparison. All 16 children and complete
carrier tails must match. Check signed boundary values and changed-target
rejection. Private digit agreement alone does not close the message result.

The local private PiDEC checkpoint passes: all 4,048,180,416 child
coefficients match freshly generated Rust children, and the final-tail
mutation rejects. The total kernel, selected consumer, exact target, audits,
boundary tests and Rust test build pass. See `PIDEC_WITNESS_REPLAY.json`.
Lean's two private ranges took 8.38 and 142.15 seconds. The child message
calculation uses these same digits. No PiCCS or HyperNova replay is claimed
by this private digit result.

The complete local PiDEC message replay now passes. Lean computes all
19,008 commitment field words from the same private digits. It also computes
all 1,728 Pad words and all 24,192 matrix-evaluation words. The 53 disjoint
matrix ranges cover every selected row in [0,6,377,559); Lean checks the
complete range coverage and adds the results. All 25,920 evaluation words
and all 56 common-point words match Rust. The final changed matrix
coefficient rejects at child 15, matrix 13, lane 53, imaginary component.
The earlier complete child comparison also checks all 4,320 verifier-derived
public words on this same trace. `PIDEC_COMPLETE_REPLAY.json` records the
sources, proof links, commands, coverage and scope. The range reports retain
the earlier measurements and compiler fixes.

The exact targets remain `LeanGraph.Targets.PiDECCommitmentReplay` and
`LeanGraph.Targets.PiDECChildEvaluationReplay`, with successful checked split
as the kernel premise. The public projection link retains its explicit
parent-public-opening premise. No claimed Rust commitment or evaluation
constructs the Lean expected values. The point remains conditional on
Lean-verified Rust PiCCS messages until independent PiCCS replay is complete.

The small `pidec-evaluation-comparison` gate rechecks the saved complete
values and target mutation. It does not establish source-generation
provenance, package identity, or full proof encoding. The original parent,
53 source-bound row runs and completed Pad supply the separate local
calculation evidence. Protected full-generation acceptance and release
upload/fresh-download verification remain external. Status stays
`Compiler-closed`.

The active computation milestone is PiCCS round one from the original
17 witnesses and public inputs. The exact graph target is
`LeanGraph.Targets.PiCCSFirstRoundReplayKernel`. It combines the complete
completion-sum formula, original and aggregated source constructors,
prepared coefficient equality, stored invocation rows, and total norm/selector
cache equality, and the complete prepared norm scan.
`piccs-first-round-kernel` checks its audit, literal target and
dependency graph. The producer input
contains no Rust rounds or newly claimed evaluations.

The first adjacent pair is measured. Reference polynomial construction took
60.04 seconds; shared gamma powers reduced it to 0.0247 seconds. All ten K
coefficients (20 field words) remain byte equal. Original-source images take
about 3.74 seconds per pair on that reference path. The retained source decoder also preserves the
complete prior PiRLC prefix. Six input rejection cases pass.
`PICCS_FIRST_ROUND_REPLAY.json` records the exact scope and source hashes.

Source/output weights now move before matrix evaluation. Reusing one complete
94-row invocation reduced the 47-pair run from 199.1 to 23.5 seconds, including
input loading. All 20 field words match the reference. Starting inside the
invocation and crossing its boundary also matches. A changed target coefficient
is rejected. The endpoint proof includes all original source lanes and both
canonical carried sums; stored-row equality retains the loader-success premise.

Weighted original blocks now share each requested read within an invocation.
The cache theorem covers repeated and missing keys. Prepared norm cubics and
suffix-selector weights also have complete coefficient equalities. Arbitrary
extension-field values use the original norm constructor on a cache miss.
The existing graph target includes the exact combined cached pair expression.

The norm contribution is now closed for the complete recorded input. The
prepared worker sum equals the original norm sum over all 2^27 pairs,
including the proved zero suffix beyond the complete carrier. The four inner
K coefficients match the actual Rust CPU norm function on all 17 original
sources; a changed coefficient rejects. The measured scan fell from 139.93
to 35.17 seconds, with 1,360,840 KiB peak RSS. `PICCS_NORM_REPLAY.json`
records inputs, exact scope and the checks. It is not full-Q or transcript
comparison.

The fresh contribution is also calculated across all 3,188,780 active pairs,
including row 6,377,558 paired with the first padded row. Exact coefficient
proofs skip zero monomials and exponent-zero factors. The selected positive
degree theorem closes the entire remaining Boolean suffix. The Option-preserving
source constructor is proved equal to the original-source reference sum; its
IO task and mutable cache loops remain implementation links. The complete run
took 623.73 seconds and 2,459,352 KiB peak RSS. `PICCS_FRESH_REPLAY.json`
records its exact scope and source hashes. This is not a full Rust Q comparison.

The complete first-round execution now matches Rust: all ten Q coefficients,
alpha, gamma, the challenge, both transcript states and the claims. A changed
coefficient rejects. `PICCS_FIRST_ROUND_REPLAY.json` binds the complete fresh,
norm and carried contributions to the original sources. The selected
source-to-whole-polynomial theorem remains an unchecked review patch;
its eleventh check still requires the requested owner approval.

Later rounds retain the existing `PrefixFold` authority. The generic
`PiCCSPrefixRound.roundPolynomial_evaluate` identifies the pair sum after any
challenge prefix with the original completion-sum specification. It does not
supply the stored endpoint arrays. `PiCCSSignedFirstFold.foldOne_prepare`
proves the signed-input cache equal to the original fold, including odd tails.
`PICCS_PREFIX_REPLAY.json` records the measured original-input prefix checks.
It also records the complete second-round inner norm: all 63,252,819 groups
and 17 sources match Rust's production fold and norm functions. A changed
coefficient rejects. The cached kernel is proved equal to the original
interpolation followed by the norm cubic; equal non-signed endpoints are
retained. The full Lean scan took 376.15 seconds at 1,362,264 KiB peak RSS.
The complete second-round fresh contribution is now computed in 49.69 seconds
from all 3,188,780 saved first-fold rows. Degrees 5–9 match Rust exactly.
`PICCS_FRESH_PREFIX_REPLAY.json` records the selected source, interpolation
and polynomial laws, measured runs and changed-challenge rejection. The
complete second round now matches Rust: all ten coefficients, alpha, gamma,
the challenge, both states and all claims; a changed coefficient rejects.
`PICCS_CARRIED_PREFIX_REPLAY.json` records the complete saved Pad prefix
(126,505,638 values), all 3,188,780 matrix prefix values, their second-round
moments, complete coverage and the final composition. The registered
`piccs-second-round-comparison` checks the saved complete result and mutation;
it does not prove generation or close the pending selected source theorem.
The retained round-one checkpoint leaves later rounds, final individual evaluations, HyperNova and
complete proof encoding. Generic
`program.row?` row caching was measured and removed: it expands a slow reference
path. Future sharing must retain the numeric invocation evaluator.

The complete ordered norm prefix after two Lean-derived challenges is now
retained for all 17 sources. Its 1,075,297,923 one-byte codes decode to the
original two-fold values. The complete comparison checks 17,204,766,768
canonical field bytes against the original scalar reader and the ordinary
fold on all 81 possible signed four-scalar inputs. A direct-array check also
passes on the retained small range. The final-tail change rejects at source
16, group 63,252,818. No zero source is removed or renumbered.
`PICCS_NORM_PREFIX_REPLAY.json` records the source hashes, complete coverage,
rejection cases and measured runs: 124.60 seconds to generate, 64.86 seconds
to compare, and about 1.3 GiB peak RSS. The exact target
`LeanGraph.Targets.PiCCSNormPrefixKernel` proves decoded-array equality with
two existing `PrefixFold` operations; file origin and IO remain separate
execution evidence. This supplies later-round norm inputs. It does not close
later rounds, final evaluations, encoding, or the pending selected-source proof.

The complete third round (Q[2]) now matches Rust: all ten coefficients,
alpha, gamma, the challenge, both transcript states and all claims. A changed
last coefficient rejects. The fresh, matrix and Pad prefixes were advanced
through the second Lean challenge; an independent indexed interpolation
compared all 1,394,698,704 output bytes, including every cross-file pair.
All 17 norm source streams remain separate. The norm scan validates all
1,075,297,923 stored codes and covers 31,626,410 pairs per source, including
the actual odd tail. Its cached/direct small range and tail agree exactly.
`PICCS_THIRD_ROUND_REPLAY.json` records sources, coverage and measurements.
The norm run took 181.05 seconds, fresh 33.36 seconds, matrix moments 6.32
seconds and Pad moments 13.63 seconds on the same Linux host with agents idle.
`LeanGraph.Targets.PiCCSPrefixNormAccumulation` proves exact chunked norm
coefficient accumulation and adjacent range composition. The saved-result
comparison and binary rejection gates remain separate from source provenance.
Rounds 3–27, individual final evaluations, complete encoding, HyperNova and
the selected-source proof remain open. Source check11 still needs approval.

All 28 PiCCS sum-check rounds now independently match the saved Rust result:
all 280 K coefficients, alpha/gamma, every challenge and transcript state,
and every claim. Each round's changed last coefficient rejects. The ordered
fresh, matrix, Pad and 17 separate norm arrays were advanced through all
28 challenges; every emitted field byte was compared with independent
indexed interpolation. Singleton prefixes still fold against zero.
The final 17 norm scalars and 14 fresh matrix constants also match Rust.
These 31 K values are coefficient-zero fields only. The other individual
Pad/matrix fields, complete encoding and selected-source proof remain open.
The pending carried-source check11 still needs owner approval.

`PICCS_ALL_ROUNDS_REPLAY.json` records complete coverage, source hashes,
commands and measured runs. `LeanGraph.Targets.PiCCSRetainedPrefixKernel`
states scalar MLE preservation with its exact cube-fit premise and per-port
fresh fold equality. It proves array interpolation, not saved-file origin
or complete polynomial-source equality. The full individual-evaluation
authority must cover all 17 `PaperAlgebra.evaluationFamily` results;
`messageAt` alone omits fresh nonconstant coefficients. Keep the full
families separate in the original-source evaluation pass.

For PiRLC, Lean checks the supplied PiCCS proof and derives all mixing
challenges. The PiCCS messages and final claims are Rust inputs until the
PiCCS prover replay is complete. The 17 source witnesses are original inputs;
the Rust combined witness is only a target. Compare every coefficient of
the complete carrier, including tails, and reject a changed target
coefficient with its block and lane. Missing source blocks mean exact zero,
not omitted comparison coverage.

Each plain executable kernel must have an audited equality with the
existing specification. No new `native_decide`, `implemented_by`, `extern`
or assumed comparison result can supply that equality. PiRLC uses
`PiRLCFinite.combineAssignments`, connected to the honest response, and
the existing ring action; later phases use their existing prover semantics.

Register each milestone's exact target, premises, dependencies, gates and
open requirements in `obligations.json`. A kernel proof does not close its
execution gate. Bind inputs and targets by path, SHA-256, producer, source
commit and regeneration command. Large inputs must be retrievable through
the selected GitHub release assets before the result is called reproducible.
Record time and peak memory per invocation. Derive chunk sizes from a
measured run; the Lean cap is 1,500 seconds and the native cap is 300 seconds.
Retain one build queue, the owner's ten-attempt rule, this checkout and
`nico/f-prime-constraints-cuda-formal` only. Replay gives evidence on tested
inputs; it does not prove universal Rust correctness.

## Original-source final evaluation checkpoint

The complete original-source Pad family now passes independently. Lean derives
the point from all 28 saved Lean rounds, validates all 17 original witnesses,
evaluates all 4,685,394 carrier blocks, and adds the 64 ordered ranges in Lean.
All 918 K coefficients (1,836 field words), all 56 point words, and the
consistent changed-target rejection match the required comparison.
Generation took 847.30 seconds with 1,359,656 KiB peak RSS on stock Lean
4.30.0 on the original Linux host.

The exact target is `LeanGraph.Targets.PiCCSOriginalEvaluationKernel`.
It fixes original-mask reads and complete Pad/canonical-matrix evaluation
against `PaperAlgebra.evaluationFamily` for every source and ring lane.
Loaded sparse and numeric invocation ranges retain exact loader, selected
block and bounds premises. Existing product and cached-row owners supply
the loader links. No split, magnitude or expected-value premise is added.

The three retained matrix cases keep all 77,112 field words byte equal.
Direct mask reads and dedicated workers reduce calculation time from 27.50
to 9.77 seconds, and command time from 48.60 to 31.51 seconds. The retained
Pad range `74272..148544` keeps all 1,836 output words equal to the scalar
reference: calculation time changes from 210.79 to 20.69 seconds and command
time from 227.67 to 37.73 seconds. These are retained cases, not a full matrix
speedup claim. All measured runs were sequential with agents idle; shared
initialization is recorded separately.

`PICCS_ORIGINAL_EVALUATIONS_REPLAY.json` records the source cut and evidence.
Synthetic merge checks cover 27,540 field sums and nine rejection cases;
they are assembly tests, not prover evidence. The complete original matrix
scan and PiCCS output encoding now pass as recorded below. Full NIFS encoding
and HyperNova remain open.
The separate PiCCSCarriedSource check 11 still needs owner approval.

The original matrix replay now checks all signed masks once and omits a
source's arithmetic only when that complete check proves it is zero. All
17 output positions remain present. The selected capture has seven active
sources and ten zero sources; these indices are not built into the code.
The sparse and numeric invocation guards are proved equal to the previous
complete batches for every source, port and lane. The exact graph target
includes both equalities.

On the same three retained production ranges, calculation time falls from
9.77 to 4.27 seconds (2.29 times faster), and command time from 31.51 to
26.01 seconds. Shared initialization takes 16.93 seconds, including the
0.50-second support check. All 77,112 field words and complete output bytes
match the previous executable. Synthetic tests also keep source 16 active,
check the last carrier block, compare nine complete ranges, and reject
malformed input before writing any result. These are source-selection tests,
not production proof evidence. `PICCS_ORIGINAL_MATRIX_SUPPORT.json` records
the exact source and measurements. Full matrix execution remains open.

The matrix replay now specializes its captured readers through both numeric
invocations and sparse row accumulation. The source change is 15 bare
`specialize` attributes and two `inline` attributes across eight files;
definition bodies and theorem statements are unchanged. Generated C confirms
direct reader calls in both hot sparse-entry loops.

On 150,400 Poseidon rows, the first 11 annotations reduce profiled calculation
time from 252.19 to 148.66 seconds and command time from 278.56 to 174.01 seconds.
On 80,750 product rows, the final six annotations reduce unprofiled calculation
time from 241.09 to 75.52 seconds and command time from 262.89 to 97.37 seconds,
compared with the saved 11-annotation executable. Each larger comparison
checks all 25,704 field words and complete file bytes, and rejects a changed
target. Peak memory is about 2.4 GiB and 2.9 GiB respectively. The three
retained cases also remain byte equal. All measured runs use the same host
with agents idle, and record initialization separately.

`PICCS_MATRIX_SPECIALIZATION.json` records the exact source cuts and scope.
These measurements do not establish a complete-matrix speedup. An initial
named-attribute experiment did not specialize the captured reader. A separate
zero-row accumulator change preserved bytes but did not establish a production
gain; that change was removed. Full matrix execution is recorded below; the
selected source-to-whole-polynomial proof remains open.

## Complete original PiCCS output execution

The complete matrix pass now covers all 6,377,559 active rows in 53 ordered
ranges. Lean combines these with the complete original Pad family. All
13,770 K values (27,540 field words), the complete point and changed-target
rejections pass against the two Rust comparison copies. This includes all
795 fresh nonconstant Pad/matrix values, all zero sources and the final row.

`finish-original` retains the causally validated 28 round vectors and uses
`PiCCSInputCheck.execute` and `PiCCSProofInputs.serializeProofInputs`. The
complete 657,063-byte PiCCS input and 450,952-byte phase result match Rust
byte for byte, including all terminal fields and the outgoing eight-word
transcript state. The existing package encoder emits exactly 29,288 words;
all words and their order match an independent flattening of those fields.
A consistent changed output target and a changed final proof word reject.

`PICCS_COMPLETE_OUTPUT_REPLAY.json` binds the producer sources, original
inputs, all saved ranges, merge, final encoding and comparison commands.
Rust proof values enter only the comparison tools. Earlier range outputs
are reused with their original source cuts and proved arithmetic equality;
this mixed execution record does not establish a full-matrix speedup.

The selected source-to-whole-polynomial theorem remains the unchanged
`PICCS_SOURCE_BRIDGE_PENDING.patch`. Check 11 still needs owner approval.
Full NIFS encoding, composed HyperNova next state/assignment/commitment and
protected fresh-checkout release reproduction remain separate open work.
This is complete local PiCCS output execution, not complete proof closure.

## Independent fresh witness execution

The standalone Lean replay now computes all 29,344,425 physical fields from
the independently derived recursive caller. Its 1,419,747 write events have
strictly increasing targets, and all 201,386 explicit assertions pass.
Stored recipe, hint and permutation procedures have audited equality proofs
against their existing Lean owners.

The fresh logical assignment uses the same canonical 30-block schedule.
Cached numeric widths and product metadata preserve the complete source
packet and schedule by Lean equality. All 253,011,231 logical coefficients
and 45 zero tail coefficients match the native successor witness. The
complete 107,246,512-byte witness and 39,448-byte fresh claim also match.
All 1,188 commitment coefficients use the unchanged production key.
Native artifacts enter comparison checks only.

The full command for all 30 blocks took 114.64 seconds: 17.12 seconds for
preparation and 96.99 seconds for block computation and output. Every logical
byte matched the earlier Lean output, and the complete carrier matched Rust
again. Peak RSS was 1,691,212 KiB.

The three fresh commitment commands took 281.32 seconds in total, with
277.148 seconds of recorded computation and 608,048 KiB peak RSS. This is
one fresh message; it is not a timing for the earlier sixteen-child PiDEC
scan. The measured range 74272..148544 took 5.695 seconds to compute and
7.12 seconds for its complete command. No other agents, builds or benchmarks
ran during those measurements. Initial preparation and compilation are not
charged to each block.

`FRESH_WITNESS_REPLAY.json` records the complete coverage, source links,
proof endpoints, command logs and 26 rejection cases. Large data stays
outside Git. The existing graph now has the literal `FreshWitnessKernels`
target and the `independent-fresh-witness` data flow, including caller
derivation, complete byte comparisons and all range guards.

The ordered compact-row lowering connection and whole selected physical
completion proof remain open. Runtime assertions and byte equality do not
supply those theorems. The existing PiCCS carried-source obligation still
needs owner approval for check11, with its ten earlier attempts preserved.
Protected source-bound reproduction and release delivery remain separate.

## Stored compact-row completion proof

The stored compact-row replay now uses the checked executor directly.
Its guarded Array execution equals the functional row executor, including
rejection, under the exact local-write bounds. The composed theorems connect
both canonical template families to the existing expression and constraint
completion functions after the physical output write. They retain explicit
array bounds, input/local separation and output/input separation premises.
All five PiRLC recipe families satisfy their output-scope bounds by structural
proofs. Ordinary stored instructions also equal their existing executor.

The full physical rerun preserves all 234,755,400 bytes from checkpoint
20bd388897dad5989008923bcd9b5f71f3c5cd0e. It covers 29,344,425 fields,
1,419,747 events and all 201,386 explicit assertions. The command took
29.76 seconds, with 23.04 seconds for computation and 2,513,200 KiB peak RSS.
Agents were idle. These are regression measurements, not a new speedup claim.
All 26 existing rejection cases pass with the changed executable.

The exact FreshWitnessKernels target includes both compact completion
statements and the ordinary instruction refinement. Leaf audits also cover
the five concrete recipe bounds. COMPACT_WITNESS_EXECUTION.json records the
source cut, attempts, complete byte comparison and validation logs.
The whole selected physical-plan connection remains open: actual invocation
geometry and preservation of every canonical row still need proofs.
The separate PiCCS source check11 remains pending with its prior budget intact.

## Complete final physical row coverage

The physical replay now checks every canonical row against the completed
array before it writes the result. Typed source records retain the exact
packet, block and permutation identities through task collection. The pure
plan constructor keeps canonical row events for checking and sorts a shared
copy for execution. The coverage proof therefore requires no sorting or
write-order premise.

StoredPhysicalPlan.ofSources_rowsHold proves that successful checks of the
concrete constructed arrays imply the selected package's complete RowsHold.
The FreshWitnessKernels target includes this theorem and exact strided-worker
coverage. The exported dependency record contains the actual ofSources and
assemble_sound definitions and the selected-package composition proof.
The target retains only positive worker count and successful final-check
premises; it assumes no new source identity, geometry or schedule property.

The executed check covers 29,024,343 event rows and 201,386 explicit
assertions: all 29,225,729 physical rows. Every one of the 234,755,400 output
bytes matches the earlier Lean result. The complete logical witness and
commitment comparisons are reused from that identical physical input.

Profiling found that only two ordinary task-pool threads performed the
integrated check. Dedicated tasks retain the hardware-derived worker count
and the same proved immutable predicate. The final check takes 7.63 seconds;
preparation takes 3.51 seconds, witness computation 24.81 seconds, assertions
0.15 seconds and output 3.73 seconds. Total command time is 40.22 seconds,
with 2,696,700 KiB peak RSS. All agents were idle during measured runs.
A serial implementation of the added check took 55.12 seconds, or 84.64
seconds for the complete command. These measurements concern physical row
validation; they are not PiDEC commitment timings.

The rotating-assignment experiment did not fix the thread-pool issue and was
removed. The final code keeps the simpler proved stride partition.
PHYSICAL_ROWS_REPLAY.json records the source cut, proofs, measurements,
rejection checks, and the independent canonical-coverage review. Caller
parsing and file-origin claims remain separate execution/custody evidence.
The PiCCS source check11 and external release/reproduction conditions remain
open. No change to protocol, package, production key, b=2 or k_rho=16 is made.

## Independent C/R/D and recursive caller execution

The complete independently generated C input now has exactly the bytes used
by the checked R witness and D range calculations. Those original-source
scans are retained. The new PiDEC `from-replay` mode loads only complete Lean
commitments and evaluations, recomputes C/R, checks the derived point and
bounded parent, derives child public inputs, and requires the existing D
check before it writes either result.

All 446,185 child bytes and all 4,722,709 complete C/R/D result bytes match
the retained results. The native comparison checks all 945,983 proof bytes
with its separate raw Lean-field encoder and passes 55 D mutation cases.
The existing recursive caller generator consumes these independent results
and the original iteration2 request. Its 1,446,131 bytes match, including
177,326 private words,278 public words and the point/transcript links.

`INDEPENDENT_NIFS_CALLER_REPLAY.json` records source and input custody,
producer commands, comparisons and rejection checks. Both final-output
commands resolve parent directories before checking distinct destinations;
the alias case `file.json` versus `./file.json` rejects before writes.

The caller packet is not the full fresh assignment. The earlier successor
record executes Rust physical/logical witness generation and a reference
commitment from Lean caller words. The next independent Lean step must
complete physical values, then reuse the proved canonical logical executor
and exact production-key commitment kernels. The selected-source Q proof
still needs owner approval for check11. Release reproduction remains open.

## Earlier baseline and retained evidence

Owner: the user's September 13 baseline goal. Work only on
`nico/f-prime-constraints-cuda-formal`, starting from `f714497c`.

The final result is the existing selected Goldilocks lifecycle, with
`b = 2`, `k_rho = 16`, its current package, and the approved FS/MSIS
boundaries. The public flow is `Stage1Envelope::initial(z0)`,
`package.extend(envelope, message)`, then
`package.verify(&expected_state, &envelope)`.

## Exact obligations

| Obligation | Required result | Closing evidence |
| --- | --- | --- |
| `stage1-assignment` | Arbitrary selected rows and their actual public input imply the full typed step at the decoded context. | Literal `LeanGraph.Targets.Stage1Assignment`, witnessed by `stage1Assignment` through `ActualPiDECOutput.selectedRowsAndPublic_imply_step`; exact acceptance and axiom audit. |
| Existing terminal and security targets | Bind the decoded step to the verifier-owned context or a named collision; retain the checked linear history bound. | `Stage1TerminalAssignment`, `Stage1TerminalParent`, and `HyperNovaLinearSecurity`, with their existing complete premises and required gates. |
| Public lifecycle | Construct the base assignment from actual state/advice and extend an active envelope through the existing native NIFS, caller packet and witness completion. | Public API execution, full Lean packet/assignment comparison, exact openings and terminal acceptance, with sampler/counter failures explicit. |
| Nonzero running input | Execute a further fold with actual nonzero running witnesses, then construct and verify its successor. | Complete C/R/D values and bytes, transcript/public/counter equality, exact next assignment and required mutation rejection against independent Lean checks. |
| Baseline delivery | Correct stale records and retain exact evidence for every technical link in scope. | Signed checkpoint, ordered static/build/axiom gates, affected identity/conformance checks, independent review and current requirements links. |

`stage1-baseline` tracks the combined result in the existing obligation map.
Its open requirements remain until their actual tests and review exist.
This registration does not change lean-graph's schema or acceptance rules.

The symbolic terminal false-acceptance target and six-record reconciliation
passed at `8084c256` and `5222c1d5`. The current staged nonzero C/R/D result,
complete proof bytes and mutations pass; see `NONZERO_NIFS_GATES.json` and
`NONZERO_NIFS_REVIEW.json` in `docs/reviews/nightstream-fprime-requirements`.
The complete later assignment and terminal checks also pass; see
`NONZERO_SUCCESSOR_GATES.json` and `NONZERO_SUCCESSOR_REVIEW.json`.
The aggregate stays compiler-closed while the two public active-call checks
await their specific execution allowance. Evidence delivery and external
production approval remain separate.

## Checkpoint requirements

The checkpoint must compile every `neo-fold-clean` test target with
`cargo test -p neo-fold-clean --release --no-run`, including the fixture
binary's test harness. Integration tests must stay in their integration
target; removing a failing harness is not the repair.

Commit reports and SHA-256 manifests. Store new evidence archives outside
Git; do not put generated witnesses or graph metadata in Git inside archives.
Existing committed archives remain historical evidence until verified
external copies and replacement references exist. Do not rewrite history.
Each named gate must have a documented fresh-checkout command and available
inputs. Small test inputs belong in the test fixtures directory; larger
external inputs need a retrieval location and checked manifest before the
gate is called reproducible. The owner selected GitHub release assets in
this repository on September 13. Record the release, asset URL and hash.

Before another expensive recursive run, record its measured stage costs,
memory measurements and the basis of its feasibility estimate. The existing
300-second native cap applies. An invocation that reaches it is a failed
slice. Report the stage and measurement; do not repeat an unchanged run or
silently convert failure into an ignored pass. The owner chose to keep the
existing project limits and report measured memory on September 13; the
review's proposed 64 GB ceiling is not adopted.

Only the coordinator runs Lean, Cargo and validation commands. Subagents
may read, draft and review. Reused Ironwood code must retain its verified
upstream license, authors and exact source commit. The reviewed snapshot is
Apache-2.0 OR MIT, Copyright (c) 2026 Zcash Protocol Developers, commit
`22dfee003b639eff660f68ea69a98a00409a9cb1`; preserve the notices described in
`external/ironwood/PROVENANCE.md`. A review citation is not code reuse.
Publish the requirements map only from committed inputs.

The public flow remains `Stage1Envelope::initial`, `package.extend` and
`package.verify`. Add no redundant initialization wrapper or public state
machine interface. Remove obsolete square-root assurance consumers only
after their uses and map references move to validated linear consumers.
General lemmas still required by those consumers are not obsolete.

## Named map dispositions

These are closure decisions and required evidence, not premature status changes.

| Record | Disposition | Required result or condition |
| --- | --- | --- |
| `N.security.error_budget` | Close the symbolic selected-terminal bound; numerical deployment choices are out of scope. | Connect terminal acceptance with no valid application history to the existing first-failure and linear visited bounds. Keep depth, queries, `g`, FS loss, marked hash collisions and actual adaptive MSIS advantage explicit. Do not substitute an extraction-success bound for false acceptance. |
| `L.language.expressions` | Close as a definition. | `Circuit.Basic.Expr` and `Env` supply the required evaluation semantics. Its `definition` status is accurate; no additional theorem is required by this record. |
| `L.language.contract` | Close as a definition. | `FormalCircuit` requires specification, footprint, soundness and completeness fields. Concrete production instances and their compiler gates remain the separate implementation evidence. |
| `L.encoding.actual` | Close the stale technical link. | `ActualPiDECOutput.selectedRowsAndPublic_imply_step`, the literal `Stage1Assignment` target and the terminal context/collision target cover arbitrary assignments. |
| `P.binding.context` | Close the selected technical link; production approval remains external. | Canonical descriptor plus `ActualContextSecurity.terminal_implies_matchingStepOrCollision`, exact assignment target and package-owned Rust context checks. |
| `P.delivery.terminal` | Close the selected terminal implementation; approved backend delivery remains external. | Selected `stage1::verify`, the typed Lean terminal target and reproducible acceptance/rejection checks on actual openings. Remove the stale citation to the generic verifier. |

## Premises and limits

The assignment target uses the actual public projection, a four-word digest,
and all selected rows. It assumes no honest encoder, NIFS acceptance,
sampler success or child-output match. Its context is decoded; selected
verifier authority remains the separate terminal/hash contract.

Reuse the approved cryptographic premises, mathematical invertibility,
declared clock and query conditions, conditional sampler success, and
canonical counter bounds. No new security model or numerical budget is
selected. The checked staged trace now reaches iteration 3 from actual
nonzero running witnesses. Capped stages supply their exact evidence and
terminal acceptance; they are not an uninterrupted public `extend` run.

One active obligation and one Lean/Rust build queue apply. The standing
owner override is ten rounds per obligation for this session; previous
attempts count. Native invocations retain the 300-second cap and Lean
commands the 1,500-second cap. Do not repeat an unchanged failed run.

This work excludes constraint reduction, new profiles, protocol changes,
new proof backends, Stage 2 and general cryptographic research. Required
correctness repairs must retain the appropriate preservation and package
checks. External production approval remains separate from local evidence.
