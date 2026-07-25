# REL-LIN-RUNNING-CLOSURE

```text
property_id: REL-LIN-RUNNING-CLOSURE
claim:
  The fixed-active F' public carrier owns exactly 270 = 54 * 5 paper
  coordinates, partitioned as 257 external logical coordinates plus 13
  completion coordinates, with an explicit two-sided ownership equivalence.
  A fresh input canonically fixes the 13 completion coordinates to zero and
  is decoded fail-closed. That zero-tail property is NOT preserved by the
  selected profile's Pi_RLC ring action: the sampler-valid challenge X maps
  external coordinate 256 (block 4, lane 40) to coordinate 257 (block 4,
  lane 41). Therefore the 13 coordinates are fresh-initialization data, not
  permanently inert padding, and a running L_in is authoritative on all 270
  coordinates. Consequently a 257-coordinate view of a 270-coordinate running
  carrier admits no total left inverse at the paper NIFS running type, the
  frozen fixed-one F' input, or the generic Construction 2 input.
assumptions:
  - The Phi81 profile fixes ringDegree = 54 and publicRingColumns = 5.
  - The production strong sampling set admits centered coefficients in
    [-2, 2]^54; challengeValid is the repository's own predicate.
  - Pi_RLC combines public inputs by ring-scalar multiplication per block,
    as in SuperNeo 7.4 step 1.
non_goals:
  - NOT a claim that the repository or production currently instantiates the
    F'/NIFS running interface at 257 coordinates. That interface is
    polymorphic in its public-input type and is not pinned anywhere; see
    REL-LIN-RUNNING-PIN below. The no-decoder theorems are conditional on the
    257 instantiation and do not by themselves exhibit a live defect.
  - NOT concrete Pi_CCS / Pi_RLC / Pi_DEC operational refinement.
  - NOT Fiat-Shamir, Poseidon2, transcript, commitment, or extraction
    refinement.
  - NOT Rust, R1CS, generated-row, or encoding conformance.
  - NOT a completed Pi_DEC . Pi_RLC . Pi_CCS verifier or F' equivalence.
paper_sources:
  - docs/superneo-paper/07-7-neo-s-folding-scheme-for-ccs.md:15-19 (Definition
    13 CE and L_in), :27-31 (Definition 14 global parameters)
  - docs/superneo-paper/07-7-neo-s-folding-scheme-for-ccs.md:111-133 (Pi_RLC
    combines commitments, public inputs, and evaluations with one coefficient
    vector), :135-137 (Lemma 4)
  - docs/superneo-paper/07-7-neo-s-folding-scheme-for-ccs.md:139-149 (Pi_DEC
    split_b and recomposition, Theorem 7)
  - docs/hypernova-paper/14_6_3_A_compiler_from_NIVC_compatible_folding_schemes_to_NIVC.md:7-22,
    :36-53, :55-61 (Construction 2 F', running vector in the public hash)
rust_surfaces:
  - none. This property is deliberately artifact independent and emits no
    rows. Do not cite it as Rust or R1CS evidence.
circuit_or_encoding_artifacts:
  - none. The Pi_RLC counterexample uses the symbolic quotient-ring product
    ringFMul_basis_basis, not a generated row artifact.
failure_class:
  A verifier treats the 13 completion coordinates as permanently zero, pins
  them to zero on a running (post-fold) carrier, or projects a running L_in
  to its first 257 coordinates. Any of these silently drops authoritative
  paper state or rejects an honest post-fold instance.
counterexample_or_witness:
  shift_enters_first_padding constructs the sampler-valid challenge X and a
  fresh input with value one at external coordinate 256, and proves the
  Pi_RLC image has value one at coordinate 257.
  freshImage_not_piRlcClosed converts this into refutation of closure.
  projectExternal_not_injective, eraseRunning_not_injective, and the three
  no_exact_*_decoder theorems are the necessity counterexamples.
lean_theorems:
  - ...FPrimeCarrier270.LogicalCarrier.coordinateEquiv_bijective
  - ...FPrimeCarrier270.LogicalCarrier.encodeFresh_externalOfLegacy_eq_expectedPublicInput
  - ...FPrimeCarrier270.LogicalCarrier.decodeFresh_sound
  - ...FPrimeCarrier270.LogicalCarrier.decodeFresh_complete
  - ...FPrimeCarrier270.LogicalCarrier.encodeFresh_injective
  - ...FPrimeCarrier270.LogicalCarrier.decodeFresh_rejects_nonzero_padding
  - ...FPrimeCarrier270.LogicalCarrier.piDecSplit_recompose
  - ...FPrimeCarrier270.LogicalCarrier.shiftChallenge_valid
  - ...FPrimeCarrier270.LogicalCarrier.shift_enters_first_padding
  - ...FPrimeCarrier270.LogicalCarrier.freshImage_not_piRlcClosed
  - ...FPrimeCarrier270.LogicalCarrier.projectExternal_not_injective
  - ...Frozen.FixedActiveCarrierObstruction.exactPaperVerifier_soundAndCompleteModulo
  - ...Frozen.FixedActiveCarrierObstruction.eraseRunning_not_injective
  - ...Frozen.FixedActiveCarrierObstruction.no_exact_paperNifs_running_decoder
  - ...Frozen.FixedActiveCarrierObstruction.no_exact_fixedOne_fprime_decoder
  - ...Frozen.FixedActiveCarrierObstruction.no_exact_construction2_fprime_decoder
axiom_report:
  The three no-decoder theorems and the whole carrier codec use
  [propext, Quot.sound]. shift_enters_first_padding,
  freshImage_not_piRlcClosed, and exactPaperVerifier_soundAndCompleteModulo
  use [propext, Classical.choice, Quot.sound]. No theorem in this property
  depends on Lean.trustCompiler; neither new module uses native_decide.
  Guarded fail-closed in tests/Axioms/FPrimeCarrier270LogicalCarrier.lean and
  tests/Axioms/FPrimeFixedActiveCarrierObstruction.lean.
proof_hash:
  LogicalCarrier.lean
    4d82d8be406893416f8c9f272a147a638323a95ad4dbbb84049199b849da83e7
  FixedActiveCarrierObstruction.lean
    bc2416566e07a08cbd8826fdd267db75f644818ceae25fcf3cb9d064ac86c9e2
conformance_status:
  model-proved. Not artifact-checked, not rust-conformant, not
  security-reduced. exactPaperVerifier_soundAndCompleteModulo is the first
  concrete discharge of the frozen Obligations.NifsSoundAndCompleteModulo
  target in this repository; every other occurrence is an assumed hypothesis.
retest_commands:
  - cd formal/nightstream-lean && lake build
      Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.LogicalCarrier
      Nightstream.Protocol.FPrime.Frozen.FixedActiveCarrierObstruction
      Nightstream.Protocol.FPrime.Frozen
      tests.FPrimeCarrier270LogicalCarrier
      tests.FPrimeFixedActiveCarrierObstruction
      tests.Axioms.FPrimeCarrier270LogicalCarrier
      tests.Axioms.FPrimeFixedActiveCarrierObstruction
```

## Open successor: REL-LIN-RUNNING-PIN

This property proves what a running `L_in` must be. It does not prove what the
repository's fixed-active F'/NIFS running interface currently is.

`Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Running`,
`Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Input`, and
`Nightstream.HyperNova.Construction2.Paper.Input` are all polymorphic in their
public-input type. Before this property, no module instantiated any of them at
a concrete carrier, and `Obligations.NifsSoundAndCompleteModulo` appeared only
as an assumed hypothesis in `Frozen.lean` and `CanonicalVerifier/NifsRefinement.lean`.

Independently, `FPrimeCarrier270/PiCcsSources.lean` states in its ownership
header that running assignments are already full-carrier values that "pass
through unchanged; the adapter does not truncate or reconstruct them", while
`FPrimeCarrier270/Assignment.lean`'s `expectedPublicInput` zero-fills every
column at or beyond 257 on the fresh path. Those two facts are consistent with
each other and with this property, but neither pins the F'/NIFS interface.

The decisive missing theorem is therefore:

```text
property_id: REL-LIN-RUNNING-PIN
claim:
  Along one typed path from the actually selected concrete setup to the frozen
  facade:
    1. its relation shape is the selected five-ring FPrimeCarrier270 shape;
    2. its public-input type is LIn dimensions, or explicitly equivalent to it;
    3. its running Pi_CCS statements preserve all 270 public coordinates;
    4. its selected NIFS Running type is the corresponding complete paper
       running type;
    5. the fixed-one or Construction 2 setup passed to the frozen checker uses
       exactly that type; and
    6. the concrete-to-paper projection preserves it coordinate for coordinate.
```

A new alias such as `ChosenRunning := ExactRunning ...` does **not** discharge
this. The pin is only real if it is tied to the setup actually consumed by the
checker; introducing another carrier name that nothing consumes replaces the
missing bridge with a second unproved relation of the same strength.

**This is a construction task, not a lookup.** Surveyed 2026-07-24: no module
instantiates the running carrier at a concrete type anywhere in the repository
except `FixedActiveCarrierObstruction` itself. In particular
`Implementation/Rust/CanonicalConformance/NativeStep/FixedOneCanonicalAdapter.lean`
— the nearest thing to a concrete consumer, and the module that discharges
`FixedOne.accepts_iff_transition` — takes `(Running : Type uRunning)` as an
opaque parameter with `DecidableEq`, and contains zero constraints relating
`Running` to any public width. `Implementation/Lowering/FPrimeFixedOne/Encoding/ProductionCallContext.lean`
likewise forwards `(Running := Running)` as a variable.

Consequently the pin cannot "resolve to 257 or 270" by inspection; there is
nothing yet to read off. Its real content is to *build* the concrete
instantiation and prove the six conditions thread through it. The live-defect
question — whether some layer forces a 257 truncation — only becomes answerable
after that construction exists. Budget it as the largest of the open items, not
the smallest, and note that the absence of any consuming instantiation is
exactly what makes the unused-alias shortcut tempting.

Its two outcomes are not symmetric:

- pinned at 270 - the no-decoder theorems become vacuous for production, and
  the concrete Pi_CCS / Pi_RLC / Pi_DEC phase work can proceed on the
  corrected carrier;
- pinned at 257 - `freshImage_not_piRlcClosed` upgrades from a design
  constraint to a live defect, because an honest post-fold running instance
  is then unrepresentable.

Do not begin concrete phase refinement (the B/C/D obligations of the
fixed-active NIFS task) before this successor is discharged.

### Sequencing context (2026-07-24)

`FOLD-PICCS-SPLIT` and `SUM-POLY-ENC` reached `model-proved` on the same day as
this property, which changes what the remaining gate is. Build-verified at this
tree state:

```text
lake build ...BlockLaneCombinedNc.ProductionRefinement
           ...BlockLaneCombinedNc.CausalSoundness
           ...DelayedPackedYZcol.Lifecycle
           ...CombinedNc.ProductionPaperNifs      -> 257 jobs, success
lake build tests.Axioms                           -> 2940 jobs, success
```

So the deterministic soundness and honest completeness of the production FE/NC
split against SuperNeo Section 7.3 is no longer the blocker. Note the scope
that `FOLD-PICCS-SPLIT` itself declares open: alpha/gamma mixing-root
probability, Fiat-Shamir, concrete Goldilocks/support instantiation, and
Rust/R1CS refinement. In particular the alpha/gamma mixing-root bound is the
probabilistic half of exactly the step D.4 Lemma 7 uses to separate its three
obligations (linear independence of powers of `C`), so `FOLD-PICCS-SPLIT`
closing does not close that argument.

The effective remaining chain is therefore:

```text
REL-LIN-RUNNING-PIN
  -> concrete accepted checker/decoder reaches PaperStepAccepted
  -> Nifs.PaperProfile.Transition or named events
  -> fixed-one Construction 2 transition
  -> concrete composition knowledge theorem   (FPR-NIFS-BRIDGE)
```

Do not expand the Lowering/R1CS surface further until that chain closes; per
`formal-verification.md` Section 15 the current selective-R1CS stage describes
a different compiler relation and is not a license to certify its rows.
