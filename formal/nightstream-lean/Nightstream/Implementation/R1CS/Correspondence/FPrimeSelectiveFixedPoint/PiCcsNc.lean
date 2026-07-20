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
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder

/-!
Public fixed-point `Pi_CCS`/NC refinement facade.

Owns: the public import boundary for the artifact-backed complete production
domain, full packed-witness layout contract, fixed-profile running-`X` public
decoder, delayed projection specialization, and active adjacent/terminal
composition.

Does not own: combined-NC sparse rows, state or transcript rows, concrete
terminal-opening enforcement, commitment binding, costs, or row-removal
authority.

Emits constraints: none; facade only.

| Stable stage path | Obligation | Authority |
|---|---|---|
| `f_prime.pi_ccs_nc.production_domain` | Export the exact 25-row-variable and 25-round block×lane production shape | artifact-checked dimensions / model arithmetic |
| `f_prime.pi_ccs_nc.packed_witness` | Export the full `Z[lane,block]` to complete-assignment correspondence and exact generated fixed-profile coordinate bijection | artifact-checked geometry / model-level value refinement |
| `f_prime.pi_ccs_nc.packed_witness.commitment` | Identify Rust's row/block/lane Ajtai opening equation with the typed commitment and derive raw-child commitment authority from actual opened matrices | artifact-checked geometry / model-level commitment correspondence |
| `f_prime.pi_ccs_nc.packed_witness.source_table` | Connect 54 physical lanes to full `Z` cells and compute ten virtual lanes as zero | artifact-backed dimensions / model-level dataflow |
| `f_prime.pi_ccs_nc.fresh_public_x` | Export the exact 270-coordinate fresh public-`X` decoder and its explicit value/dataflow boundary | artifact-checked provenance / conditional model refinement |
| `f_prime.pi_ccs_nc.delayed_refinement` | Split the remaining public-input, source-product, opening, and key bridges into exact named failures | model-level refinement boundary |
| `f_prime.pi_ccs_nc.delayed_projection` | Export the concrete raw-child refinement contract | artifact/model composition |
| `f_prime.pi_ccs_nc.delayed_active.opened_children` | Derive the successor raw-child commitment premise from fourteen exact packed-matrix openings | artifact geometry / model-level commitment correspondence |
| `f_prime.pi_ccs_nc.delayed_active` | Exact canonical-parent commitment/norm and raw-child commitment predicates are case-partitioned into success or named binding failure; on success the raw block×lane identity propagates every packed equation backward across a nonempty digest-linked trace, with an explicit no-pending base and raw-matrix terminal anchor | model-level executable/refinement contract |

The running-`X` public-prefix decoder and compact full-`Z` geometry are
artifact-checked for the generated profile. The typed opening equation now
uses exactly those matrix coordinates, but native key serialization,
combined-NC extraction, state/transcript/terminal rows, and canonical parent
opening enforcement remain open before production authority or Rust
conformance can be claimed.

`OldPointBinding.OldPointSumcheckRelation` is the useful fixed-270, flat-column
diagnostic relation. It is not the active production statement: production has
19 block rounds plus six lane rounds over 14,338,890 coordinates. The active
contract therefore concludes the exact 54-lane `PackedYZcolBoundAtBlock`
equation directly. Treating the 512-column diagnostic relation as that full
statement would be a dimension-changing semantic error.
-/
