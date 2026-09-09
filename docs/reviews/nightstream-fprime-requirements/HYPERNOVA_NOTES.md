# HyperNova requirement coverage

Reviewed source: `4fc02857c3aa4207f3290739cc62d794ba86f5f9`.

This audit produced `hypernova.json`. It defines the H and O branches of the
combined requirement tree. It is a read-only source and paper audit. It runs no
build, test, emitter, or proof backend and makes no production change.

## Paper coverage

The following local files were read completely:

- `07_2_Preliminaries.md`: CCS, LCCS and committed-relation definitions.
- `08_3_Multi_folding_schemes.md`: completeness, knowledge, projection-succinctness,
  one-message syntax, and the explicit Fiat–Shamir knowledge boundary.
- `10_5_Non_uniform_incrementally_verifiable_computation.md`: Definition 11.
- `11_6_HyperNova_NIVC_from_multi_folding_schemes.md`: conditional theorem scope.
- `12_6_1_Overview_of_HyperNova.md`: complete running/fresh data flow and sizing.
- `13_6_2_NIVC_Compatible_multi_folding_schemes.md`: all six compatibility properties.
- `14_6_3_A_compiler_from_NIVC_compatible_folding_schemes_to_NIVC.md`: every step
  of Construction 2 key generation, augmented execution, prover, and verifier.
- `25_A_6_Incrementally_Verifiable_Computation.md`: uniform IVC and its scope.
- `39_H_2_Proof_of_Lemma_3_Folding_CCS_NIVC_compatibility.md`: fixed-circuit
  encoding, default opening, compact interface, and the recursive-size gap.
- `40_H_3_Proof_of_Theorem_4_HyperNova.md`: completeness and reverse extraction.

The scope statements of §§7–8, Appendix F, and Appendix G were inspected to
separate optional constructions from selected Stage 1. Their full internal
security proofs were not reviewed because those protocols are not selected.

## Coverage map

| Paper requirement | Tree branch |
|---|---|
| Definition 11 algorithm interfaces | H.prover, H.terminal |
| Definition 11 completeness | H.security.perfect_completeness |
| Definition 11 constant-step knowledge | H.security |
| Definition 11 proof size and runtime | H.size |
| Definition 12 properties 1–3 | H.compat.decode_arbitrary, encode_valid, structure_independent, instance_inverse, monotonicity; compiler details in L |
| Definition 12 default instances | H.compat.default_satisfies, H.base.default |
| Definition 12 compact recursive interface | H.compat.compact_projection |
| Definition 12 recursive-size closure | H.size |
| Construction 2 complete stateHash and encHash | H.state |
| Construction 2 augmented application/control | H.application |
| Construction 2 augmented base branch | H.base |
| Construction 2 augmented recursive branch | H.recursive |
| Construction 2 next-state output | H.output |
| Construction 2 external prover | H.prover |
| Construction 2 external verifier | H.terminal |
| Appendix H.3 local-to-history security | H.security |
| Optional/withdrawn constructions | O |

The selected profile has one HyperNova slot. Its running value is a bundle of
16 low-norm SuperNeo CE claims. This is not a 16-slot HyperNova program.
The fresh value has one CCS claim. The generic paper's one-running/one-fresh
interface is instantiated at this bundle boundary. The code owns this in
`Lifecycle.Relation`, `Lifecycle.Stage1.Accumulator`, and
`Lifecycle.Stage1.Terminal`.

All listed statement-preimage components are explicit: context, iteration, initial
state, current state, evaluation point, all 16 commitments/public inputs, separate
Eval_K and all 14 Eval_A families, and one-based pc. Primitive hash/word arithmetic
is owned by T/F. Encoding and matrix compiler details are owned by L. Folding
math and checks are owned by C/R/D/N. Production execution and evidence are owned
by P. H records the exact HyperNova consuming contracts.

## Status rules

`proof=proved` cites an actual theorem for the leaf's stated scope. It does not
mean the full system is complete. For example, a proved characterization of the
terminal predicate establishes its exact checks; it does not prove that Rust
executes that predicate or that accepted terminal proofs yield a complete history.

`connection=connected` means that the stated local Lean operation has a proved
connection to its present owner. It is independent of the `rust` status. A local
connection must not turn a partial higher-level theorem green.

`rust=open` refers to the selected Stage 1 package path. Existing native lifecycle
code is cited where it owns similar work, but it currently runs through other
preprocessing/representation paths. It receives no selected-package conformance
credit from its existence. No Rust test status was changed by this audit.

`not_reviewed` is used where this audit did not establish coverage. It is not a
claim that code or a theorem is absent. In particular, generic compiler
monotonicity is not automatically a new task for the one selected application.

## Important proved results

- `PilotZeroRunning.defaultRunning_holds` proves all 16 zero CE openings under
  every selected relation/Ajtai key. The default is more than a tuple of zero words.
- `ActualStep.selectedRowsAndPublic_imply_baseStep` proves the complete base step
  for arbitrary accepted assignments under the decoded context.
- `ActualPiCCSInputs.selectedRowsAndPublic_imply_phaseAndHashes` proves the exact
  PiCCS input and prior-state hash link on actual assignment values.
- `ActualNextPreimage.rowsZero_implies_decodedHeaders` proves field-encoded
  successor and all initial-state word equalities.
- `StateEncoding.serializePreimage_injective` proves complete state serialization
  injective on well-formed states.
- `XOut.decodeHash_encHash` and `encHash_injective_fixed` prove the selected
  public-instance inverse and injectivity.
- The selected fixed-point and joint-domain theorems exist. The local paper's
  missing recursive instantiation is not evidence that Nightstream lacks a fit.

## Exact remaining connections

The main F′ root still needs actual sampler/product agreement, exact PiDEC parent
and child messages, and equality of the computed running output. The current
`ActualStep` reduction leaves those recursive checks explicit.

Context preservation is proved. Equality with the verifier's selected context is
a different obligation. The full canonical closure overwrites the context words
before constructing its assignment. This does not establish the same result for
every accepted assignment.

Counter encoding and counter succession are separate leaves. The existing
`StateEncoding.WellFormed` requires `iteration < goldilocksModulus` and
`pc = 1`; under it, complete preimage injectivity is proved.
`StateDecoder.preimage_wellFormed` proves a decoded field counter is in range.
The actual next-preimage result explicitly proves only the field successor.
The final terminal/reverse-extraction connection must establish the successor's
admissible natural-number range before using equal hashes to derive `h+1=i`.
This audit did not produce a bad accepted proof. It did not invent a counter cap:
the modulus condition comes from the existing encoding theorem.

The terminal relation already has exact bottom/recursive syntax, full-state hash
check, all 16 running CE checks, and the fresh CCS check. It performs no new NIFS
fold. Its metadata covers the existing complete relation and adds no rows.

The reviewed selected security export is a one-step result. The HyperNova modules
contain local transition/terminal definitions and equivalence theorems. This audit
found no complete selected export for reverse predecessor reconstruction and
history extraction. Those uses must be connected or explicitly accounted for
under the owner's allowed assumption boundary.

## Theorem scope and exclusions

The supplied local HyperNova text contains corrections. Section 6 and Appendix
H.2 explicitly withdraw the instantiated equal-half CCS theorem because complete
recursive advice exceeds the private capacity for every positive capacity.
Only the conditional compiler theorem survives. Nightstream must supply its own
finite encoding; its current fixed-point/domain theorems are relevant evidence.

The surviving local paper theorem covers fixed constant iteration count independent
of the security parameter. It does not establish one uniform extractor for
polynomially growing iteration count. Its completeness result and scope are
recorded separately from the stronger optional claim.

The owner goal permits explicit remaining SuperNeo soundness assumptions.
A full new Lean adversary/probability framework is not required by this audit.
The knowledge/forking/hash assumptions must nevertheless be named and scoped.
A deterministic disjunction containing failure events is not a probability bound.

The O branch excludes multi-function support, the contradicted equal-half
instantiation, uniform polynomial-step extraction, a-la-carte efficiency,
authenticated external-memory optimization, blind/randomizer circuits, CycleFold,
nlookup, the outer-Nova alternative, PCD, Nebula, and Coral. These are not counted
as unfinished selected Stage 1 work.
