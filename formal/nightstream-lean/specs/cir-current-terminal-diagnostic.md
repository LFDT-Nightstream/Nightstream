# CIR-FPR-TERMINAL-LINK: current terminal diagnostic

```text
property_id: CIR-FPR-TERMINAL-LINK
assurance_state: artifact-checked three-matrix slice plus Rust-diagnostic
                 selected thirteen-matrix prefix
milestone_status: M4 historical-artifact complete/current-shipping open
profiles: direct-R1CS diagnostic: fixed-one, plain, carrier width 270,
          batch schedule [1, 1], fourteen CE claims, matrixCount = 3;
          selected fixed-point prefix: fixed-one, plain, carrier width 270,
          fourteen CE claims, matrixCount = 13
```

## Scope

This diagnostic follows one honest current fixed-one terminal synthesis from
the Rust entrypoint and captures the exact affine rows between the terminal
NIFS output and the terminal accumulator. It does not regenerate the complete
current program. It also records exact row ranges and compact column schedules
for the surrounding terminal owners so later work cannot infer placement from
historical rows or stage names.

The captured run is the full-history `minimal-supported-bit-carrier`
diagnostic fixture. It enters through `direct_ccs::preprocess_seeded`, whose
`R1cs::to_structure` uses the standard SuperNeo Definition-11 three-matrix
embedding `f(X0,X1,X2) = X0*X1-X2`. Its fourteen terminal CE claims each
contain exactly three evaluation rows. Fourteen is the source and CE-claim
count; it is not `matrixCount`.

The selected fixed-point production path instead enters through
`R1csIvcPreprocessing::new_seeded`, which calls
`R1csIvcRelation::compile_fixed_point`. Its selective CCS materializer owns
thirteen named ports and constructs a thirteen-matrix structure. The kernel theorem
`FPrimeProductionOwnerProgramBoundary.CurrentProfile.
diagnosticMatrixCount_ne_activeProduction` proves the matrix arities `t`
differ.
Consequently this capture is current-source evidence for its diagnostic
fixture, but it is not a capture of the selected production relation and
cannot decide production terminal refinement or furnish a production
counterexample.

The selected path is now independently selectable and exported by the existing
fixed-point terminal-placement test. The Rust-originated capture proves that
the live `R1csIvcPreprocessing` path has `matrixCount = 13`, fourteen terminal
CE claims, and thirteen evaluation rows per claim. It reaches row and column
frontiers `22,834,865` and `182,083,578`, then records the existing
24,185,169-row raw-old-block projection placement
`[22,834,865,47,020,034)`.

That exporter now uses the diagnostic-only
`R1csBuilder::new_witness_with_row_families` before the projection loop. It
records 286 lightweight nested row-family boundaries from the real selected
path while retaining no matrix coefficients. The terminal portion is:

| Selected terminal family | Exact rows | Count |
|---|---:|---:|
| Transcript | `[11306317,11306325)` | 8 |
| NIFS | `[11306317,19258244)` | 7,951,927 |
| Running digest link | `[19258244,19258248)` | 4 |
| Parent-authority link | `[19258248,19274957)` | 16,709 |
| Prior/latest link | `[19274957,19275227)` | 270 |
| Output accumulator | `[19275227,22597848)` | 3,322,621 |
| Complete pre-projection terminal block | `[11306317,22597848)` | 11,291,531 |

Within the NIFS family, PiCCS occupies 7,154,699 rows, PiRLC 785,327,
PiDEC 11,845, and point binding 48. These labels are capture selectors, not
semantic authority. The fourth diagnostic outcome still applies: the
production path is selectable and its exact family frontiers export
correctly, but the current capture has no coefficients from which to define a
row relation or prove the required full-rows-to-slice projection theorem.
This is a decoder/capture-detail gap, not a production counterexample.

The current Rust source emits no standalone owner named or typed as the frozen
unary `runningCheck` or `freshCheck`. The physical terminal instead consists
of a final NIFS fold, continuity/link rows, an output accumulator, public
image pins, and fourteen direct terminal-CE checks. This stops the direct
positive route. It is not yet a counterexample: the current NIFS and CE
equations could still imply the two frozen unary relations, modulo their
existing exact algebraic events.

## Three-matrix diagnostic source and row map

The direct-R1CS diagnostic synthesis has 13,396,704 rows and 12,601,679
columns. Column zero is the builder-owned constant one. The terminal owners are
emitted after host preflight; there is no row selector. This table is not the
selected thirteen-matrix row map.

| Semantic responsibility | Current Rust owner | Exact current rows | Exact captured columns | Existing Lean result | Missing refinement |
|---|---|---:|---|---|---|
| Terminal entry and profile | `decider::terminal::synthesize_last_step_terminal_r1cs_inner` | builder `[0,13396704)` | fixed-one, plain carrier 270; host requires a final fold, at least one step, and nonempty terminal fresh batch | historical whole-program theorems only | no whole-current theorem |
| Final transcript and NIFS fold | `emit_terminal_fold`; `terminal.transcript`, `terminal.nifs` | transcript `[4131804,4131812)`; NIFS `[4131804,9657286)` | range hashes and nonzero counts are captured; complete coefficients are not decoded in this slice | model-level NIFS and historical artifact theorems | current rows to the selected NIFS verifier or exact `BatchBadRoot` |
| Running digest continuity | `enforce_digest_eq`; `terminal.running_link` | `[9657286,9657290)` | `7769121..7769124 = 4085063..4085066` | `Captured.rows_iff_holds` | continuity is not running-relation validity |
| Parent-authority continuity | `enforce_parent_authority_equal`; `terminal.parent_link` | `[9657290,9673389)` | all 16,099 equality pairs are in generated `pinRuns`; column hull `[360229,4116205]` is non-contiguous | `Captured.rows_iff_holds` | continuity is not either frozen unary relation |
| Prior/latest public link | `enforce_terminal_latest_link`; `terminal.latest_link` | `[9673389,9673659)` | claim `4090877 = 1`; `4090878..4091133 = 16766..17021`; `4091134..4091146 = 0` | `Captured.priorLatest_iff_currentPlacementRows` | this is `priorLinkAccepted`, not `freshCheck` |
| Output materialization | `enforce_terminal_output_acc_digest`; `terminal.accumulator` | `[9673659,12937939)` | range hash and nonzero count are captured; coefficients are not decoded here | historical accumulator theorems only | current output semantics and binding |
| Child-to-running continuity | `enforce_child_core_equal_running`; `decider.terminal_continuity` | `[12937939,13161533)` | range hash and nonzero count are captured | historical continuity theorems only | current coefficient decoder and relation transport |
| Public output pins | `pin_public_image`; `decider.public_pins` | `[13161533,13165816)` | range hash and nonzero count are captured | historical public-pin theorems only | current output ownership refinement |
| Direct terminal CE | `enforce_final_ce_relations_with_pending_projection`; `decider.terminal_ce` | `[13165816,13396704)` | fourteen exact 16,492-row schedules, including every allocated witness, commitment, public, point, evaluation, constant-term, and NC-point column schedule | generic/model and historical direct-CE results | coefficient semantics for these current rows and implication to the frozen running/fresh checks |

The coefficient-complete bounded capture is the contiguous affine shell
`[9657286,9673659)`: four running-digest equalities, 16,099 parent-authority
equalities, and the 270-row prior/latest link. Its exact range hash is:

```text
019e66aac58486fd61e45bec572b1468afa75ae10447764b4782d63b1c2e1789
```

The complete 270-coordinate order is preserved. Coordinate 0 is the affine
one, coordinates 1–256 are linked to the producer's ordered `x_out` bits, and
coordinates 257–269 are the thirteen completion coordinates pinned to zero
for this fresh input. This capture does not assert that a running carrier's
completion tail is zero.

## Actual decoded relation

`FPrimeFullHistoryCurrentTerminalAffineShellSound.Captured.Holds` is defined
only from the imported row coefficients. For an assignment it requires:

1. every captured running-digest pin equation;
2. every captured parent-authority pin equation;
3. every captured prior/latest pin equation.

It contains no frozen validity proposition, `TerminalFacts`,
`SourceAuthority`, caller-supplied checker result, or generic failure branch.
Lean proves:

```lean
Captured.rows_iff_holds
Captured.priorLatest_iff_currentPlacementRows
```

The first theorem is soundness and completeness for all 16,373 exact captured
rows. The second identifies exactly the final 270 rows with the independent
current-placement artifact while explicitly refusing to identify that link
with frozen `freshCheck`. Both headline paths are guarded at
`[propext, Quot.sound]` and introduce no `Lean.trustCompiler` dependency.

## Evidence boundary and next decision

The three-matrix JSON diagnostic is proof-free raw current-source evidence. It records
source hashes, exact terminal range hashes, activation, the affine artifact
path, and all fourteen terminal-CE column schedules. Rust fails closed on any
byte drift in either the JSON or generated affine artifact.

The selected-prefix JSON is also proof-free Rust diagnostic evidence. Its
drift gate derives `matrixCount`, CE-claim count, evaluation count, row/column
frontiers, all 286 nested row-family boundaries, and projection placement from
the actual fixed-point path. It contains no hand-authored acceptance bit and
does not alter the fixture to obtain thirteen ports. Family names and
frontiers do not substitute for coefficient equations.

Phases 1–2 now establish both why the earlier capture had the wrong relation
and that the actual thirteen-matrix path is selectable. Phase 3 stops at a
precise decoder gap: the selected prefix retains witness values and family
frontiers but no coefficient records from which to define
`DecodedTerminalSlice.Holds` or prove

```text
FullTerminalRows.Holds assignment
→ DecodedTerminalSlice.Holds (assignment.restrict decodedColumns).
```

The family census rules out an undifferentiated dump of the 7,951,927-row NIFS
block as the scrappy next step. The next terminal-only task is to reuse the
existing PiCCS/PiRLC/PiDEC family compilers and semantic theorems, export the
actual selected terminal column joins, and retain coefficients only for
families or defining rows not already covered by those exact compilers. The
direct-CE side must likewise compose the existing raw-old-block artifact with
the post-projection claim families rather than recapture the complete
projection. This composition must yield the restricted-assignment projection
theorem above. `BatchBadRoot` must be transported as a distinct named event;
it is not a required sampled fixture. Decoding more of the three-matrix
diagnostic, constructing canonical terminal recipes, or beginning cost
arithmetic would answer a different question. Whole-current Route 2, artifact
regeneration, codecs, recipes, and Rust constraint changes remain out of
scope.

The read-only Step audit found no current overlap or unowned range, but no
checked whole-current Step range certificate exists. Historical Step ranges
remain historical. Step-only refinement may be viable after a bounded current
range certificate, but that work was not started.
