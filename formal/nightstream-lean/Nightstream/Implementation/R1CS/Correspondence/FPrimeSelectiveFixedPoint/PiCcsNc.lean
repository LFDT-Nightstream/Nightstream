import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.Production
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitness
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessSourceProjection
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessCommitment
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessProduction
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RefinementBoundary
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveOpenedBoundary
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTrace
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTerminalRawProjection
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTraceRawProjection
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTraceRawProjectionRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact.FinalRoundFactorization
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact.ProductionHonestAssignment
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ActivePublicWriteExecution
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.SelectedRowsExecutionComposition

/-!
Public fixed-point `Pi_CCS`/NC refinement facade.

Owns: the public import boundary for the artifact-backed complete production
domain, full packed-witness layout contract, fixed-profile running-`X` public
decoder, delayed projection specialization, fixed raw-old-block row execution,
and active adjacent/terminal composition.

Does not own: the remaining combined-NC sparse rows, state or transcript rows,
terminal CE row refinement, commitment binding, or row-removal authority
outside the exact terminal raw-old-block family.

Emits constraints: none; facade only.

| Stable stage path | Obligation | Authority |
|---|---|---|
| `f_prime.pi_ccs_nc.production_domain` | Export the exact 25-row-variable and 25-round block×lane production shape | artifact-checked dimensions / model arithmetic |
| `f_prime.pi_ccs_nc.packed_witness` | Export the full `Z[lane,block]` to complete-assignment correspondence and exact generated fixed-profile coordinate bijection | artifact-checked geometry / model-level value refinement |
| `f_prime.pi_ccs_nc.packed_witness.commitment` | Identify Rust's row/block/lane Ajtai opening equation with the typed commitment and derive raw-child commitment authority from actual opened matrices | artifact-checked geometry / model-level commitment correspondence |
| `f_prime.pi_ccs_nc.packed_witness.source_table` | Connect 54 physical lanes to full `Z` cells and compute ten virtual lanes as zero | artifact-backed dimensions / model-level dataflow |
| `f_prime.pi_ccs_nc.fresh_public_x` | Export the exact 270-coordinate fresh public-`X` decoder and its explicit value/dataflow boundary | artifact-checked provenance / conditional model refinement |
| `f_prime.pi_ccs_nc.post_pidec.public_write` | Decode the exact live builder/normalized/packed 270-write execution artifact and reconstruct the independent typed public carrier | artifact-checked fixed-profile execution refinement |
| `f_prime.pi_ccs_nc.delayed_refinement` | Split the remaining public-input, source-product, opening, and key bridges into exact named failures | model-level refinement boundary |
| `f_prime.pi_ccs_nc.delayed_projection` | Export the concrete raw-child refinement contract | artifact/model composition |
| `f_prime.pi_ccs_nc.delayed_active.opened_children` | Derive the successor raw-child commitment premise from fourteen exact packed-matrix openings | artifact geometry / model-level commitment correspondence |
| `f_prime.pi_ccs_nc.delayed_active` | Exact canonical-parent commitment/norm and raw-child commitment predicates are case-partitioned into success or named binding failure; on success the raw block×lane identity propagates every packed equation backward across a nonempty digest-linked trace, with an explicit no-pending base and raw-matrix terminal anchor | model-level executable/refinement contract |
| `f_prime.pi_ccs_nc.delayed_active.raw_terminal_projection` | Use every fixed generated raw-old-block projection row over the same fourteen ordered `FinalWitnessWires` allocations joined to terminal Ajtai, construct honest derived assignments, then propagate every preceding equation back to an explicit no-pending base | artifact-checked row placement and allocation join / model-proved soundness and completeness; terminal CE row refinement remains separate |

The running-`X` public-prefix decoder, the live post-`Pi_DEC` 270-write
execution join, and compact full-`Z` geometry are artifact-checked for the
generated profile.  The live-write theorem eliminates the prior
`ActivePublicWritesBound` premise for this fixed public prefix; private
assignment and matrix decoding remain separate. The typed opening equation
now uses exactly those matrix coordinates. The direct terminal raw-witness
projection rows, their fixed runtime placement, and their exact allocation
join with the terminal Ajtai witness family now refine the active one-fold
delay trace without an implementation-failure branch. Native key
serialization, remaining combined-NC extraction, state/transcript rows,
terminal CE row refinement, and canonical parent opening enforcement remain
separate boundaries. The final-round factorization replaces the exact sound
baseline of 25,243,884 rows and 25,243,776 derived columns with 24,185,169
rows and 24,185,061 derived columns. Kernel-checked semantic equivalence and
honest assignment construction authorize exactly 1,058,715 fewer rows and
columns for this family only; this facade makes no broader removal or
security-reduction claim.

`OldPointBinding.OldPointSumcheckRelation` is the useful fixed-270, flat-column
diagnostic relation. It is not the active production statement: production has
19 block rounds plus six lane rounds over 11,437,038 aligned coordinates. The active
contract therefore concludes the exact 54-lane `PackedYZcolBoundAtBlock`
equation directly. Treating the 512-column diagnostic relation as that full
statement would be a dimension-changing semantic error.
-/
