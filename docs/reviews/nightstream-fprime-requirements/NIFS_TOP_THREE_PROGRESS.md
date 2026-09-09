# NIFS top-three progress

Requested scope: four shared-profile records; the strong, weak and aligned-fork
interfaces; and interactive NIFS composition. This is eight proof records.

Base: `9e49e7fbe03d61b0e16eec1d6f839d33d57af9bb`, with uncommitted additions.

## Closed records

| Record | Checked owner |
|---|---|
| N.profile.relation | NifsProfile.selected_relation / selected_matrix |
| N.profile.shape | NifsProfile.selected_shape |
| N.profile.arity | NifsProfile.selected_arity |
| N.profile.setup | NifsProfile.shared_setup / phases_preserve_relation |
| N.security.strong | PaperStrongCompleteness.exists_honest_piCcs_prover; PaperStrongInterface.piRlcBatchForProbe_same_phi; Lifecycle.Nifs.StrongExtraction.probability_and_expected_work |
| N.security.aligned_fork | PaperAlignedExtraction.positive_return_implies_alignedFork |

The profile results preserve the exact selected matrices, polynomial, public
projection, commitment map and counts. They do not prove external context
selection or commitment hardness. The strong interface uses independent
interactive verifier coins, the actual returned witness and actual call/check/
access work. Alignment retains the observed probe and sampler equalities; it
does not turn Poseidon2 coins into independent uniform coins.

## Still open

`N.security.weak` remains partial/open pending the complete selected contract.
The resumed selected weak success wrapper now passes. The selected honest
C→R→D theorem derives all 16 child openings from the original source witness.
The exact same-phi binding connection is proved from the actual fork responses.
It has no arbitrary-ambient uniqueness premise.

`N.security.composition` remains partial/open. The selected probability theorem
now measures the actual returned source values, uses the fixed extraction
algebra, and requires suffix algorithms only at reachable positive-context
receipts. `SupportedExtraction.returned_source_bound_with_binding` combines
the original success rate, weak retry loss, PiCCS test error and actual binding
event. The selected runtime connection must still be assembled with it. The
probability theorem alone does not establish expected polynomial work.

The initial source stop was resumed on the owner's instruction. The weak
wrapper passed after rewriting the indicator with its proved equivalence.
The runtime connection then reached the formal project's three-round stop
rule. Its remaining elaboration failure is traced to a redundant finite-type
instance. An exception for that specific proof has been requested; no further
runtime repair has been applied while the answer is pending.

## Draft preservation

The original stopped drafts remain preserved outside the Lean package in:

`../nightstream-stage1-evidence/nifs-top-three-2026-09-09/drafts/`

`draft-manifest.json` records each original path and SHA-256. The files are
`WeakExtraction.lean.txt`, `InteractiveComposition.lean.txt` and
`InteractiveWork.lean.txt`. These originals are immutable evidence of the
earlier stop. Source copies were restored for the approved resume. The runtime
copy is preserved unchanged as `InteractiveWork.resumed-paused.lean.txt`, with
SHA-256 `c1b481517b655efbe1b803539977690efc796d473f09b43f09b3d190827be9ea`.
It is excluded from the proof package, root imports and axiom audit.
`resumed-runtime-pause.json` records the pending exception and exact open item.
`source-manifest.json` records the earlier validated cut;
`resumed-source-manifest.json` records the resumed source cut.

## Validation

Boundary gate passed. The full library passed: 3770 jobs, 371 seconds.
The full test/axiom library passed: 3808 jobs, 5 seconds, after the new audit
file was added to the explicit build roots. All 85 new audits use only
`propext`, `Classical.choice` and `Quot.sound`. Logs are `full-audit.log`
and `axiom-audit.log` in the evidence directory.
The requirements-site build and all seven existing export tests pass.
No Rust code, protocol layout, selected parameter profile, transcript or package
identity changed. Existing native conformance evidence was not rerun as proof
of this new security work. C/R/D and HyperNova status rows are unchanged.
The verified NIFS counts after that publication are Proof 12/21 and Link 16/26.

On resume, the selected weak, probability, source-return, reachable-state and
honest-completeness checks pass in 2–3 seconds each. The final boundary gate
passes. The full library passes: 3780 jobs, 367 seconds. The full test/axiom
library passes: 3818 jobs, 21 seconds. All 114 NIFS audit declarations use only
the three allowed axioms. `resumed-full-audit.log` records those gates; all 36
source hashes match `resumed-source-manifest.json` after validation. The site
build and seven existing export tests pass. These results do not yet justify
closing the two remaining requirement records.

## Publication

The public requirements site was updated and its live JSON was checked against
the validated source. Version 18, site source
`94e58ad442d814e491135de203f09597e91025a4`, deployment
`appgdep_6aa0e46a4bf48191a0d4191f1d6ca49f`. The two open entries now cite the
checked selected probability and binding proofs and name the remaining runtime
connection. The counts remain Proof 12/21 and Link 16/26.

https://nightstream-requirements.nicarq.chatgpt.site/#group-N
