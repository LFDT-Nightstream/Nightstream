import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitness
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.SourceProjection

/-!
Full packed-witness cells at the production block×lane NC source table.

Assurance tier: artifact-checked dimensions and model-level dataflow. Rust
conformance remains open until the verifier passes actual `CcsWitness.Z` or
`CeWitness.Z` matrices through this exact contract.

Owns: equality of every live combined-NC running-source leaf with the
corresponding full packed-witness cell; verifier-computed zero for all ten
lane-padding positions; and zero for blocks outside the 265,535-block
production carrier.

Does not own: public `CeClaim.X`, prover-carried `CeClaim.y_zcol`, witness
commitment binding, generated sparse rows, transcript scheduling, Rust
integration, costs, or row-removal permission.

Emits constraints: none; direct dataflow/refinement theorem only.

| Stable stage path | Mathematical obligation | Authority class | Rust owner |
|---|---|---|---|
| `f_prime.pi_ccs_nc.raw_z.live` | each of 54 physical lanes reads `Z[lane, block]` | direct dataflow | production `CcsWitness.Z`/`CeWitness.Z` handoff |
| `f_prime.pi_ccs_nc.raw_z.lane_padding` | Boolean lanes 54 through 63 are zero | computed | combined-NC table construction |
| `f_prime.pi_ccs_nc.raw_z.block_padding` | blocks after the complete carrier are zero | computed | combined-NC table construction |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessSourceProjection

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open PackedWitness

/-- Every live block×lane leaf for a running source is exactly the supplied
full packed-witness cell. The result follows from the production source table,
not from a carried evaluation or digest. -/
theorem paddedValue_running_live_eq_witness
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (source : Fin shape.runningCount)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (lane : Fin ringDegree) :
    SourceProjection.paddedValue covers (decodedData template witnesses)
        (Data.runningIndex source) (domain.carrierBlock covers block)
        (domain.phi81Lane covers lane) =
      witnesses source lane (rustBlockOfSemantic block) := by
  rw [SourceProjection.paddedValue_live]
  unfold Semantics.Nc.BlockLane.value
  rw [Data.assignment_runningIndex]
  exact (decodedData_coordinatesAligned template witnesses source lane
    (rustBlockOfSemantic block)).symm

/-- Production specialization: the 54 physical lanes of every one of the
265,535 live blocks read the actual full running witness matrix. -/
theorem production_live_eq_witness
    (template : Data ProductionDomain.semanticShape)
    (witnesses : Fin ProductionDomain.semanticShape.runningCount ->
      Matrix ProductionDomain.semanticShape)
    (source : Fin ProductionDomain.semanticShape.runningCount)
    (block : Fin (Phi81ColumnLayout.blockCount
      ProductionDomain.semanticShape.carrierWidth))
    (lane : Fin ringDegree) :
    SourceProjection.paddedValue ProductionDomain.blockLaneDomain_covers
        (decodedData template witnesses) (Data.runningIndex source)
        (PiCcsDomains.production.nc.carrierBlock
          ProductionDomain.blockLaneDomain_covers block)
        (PiCcsDomains.production.nc.phi81Lane
          ProductionDomain.blockLaneDomain_covers lane) =
      witnesses source lane (rustBlockOfSemantic block) :=
  paddedValue_running_live_eq_witness
    ProductionDomain.blockLaneDomain_covers template witnesses source block lane

/-- Production lane padding is computed zero. These ten leaves do not read a
packed witness, public claim, sidecar, or physical assignment column. -/
theorem production_lane_padding_zero
    (template : Data ProductionDomain.semanticShape)
    (witnesses : Fin ProductionDomain.semanticShape.runningCount ->
      Matrix ProductionDomain.semanticShape)
    (source : Fin ProductionDomain.semanticShape.runningCount)
    (block : Fin PiCcsDomains.production.nc.blockCount)
    (lane : Fin PiCcsDomains.production.nc.laneCount)
    (padding : ringDegree <= lane.val) :
    SourceProjection.paddedValue ProductionDomain.blockLaneDomain_covers
        (decodedData template witnesses) (Data.runningIndex source)
        block lane = 0 :=
  SourceProjection.paddedValue_lane_padding
    ProductionDomain.blockLaneDomain_covers (decodedData template witnesses)
      (Data.runningIndex source) block lane padding

/-- Production block padding is computed zero after the complete full-`Z`
carrier; it cannot alias a packed witness cell. -/
theorem production_block_padding_zero
    (template : Data ProductionDomain.semanticShape)
    (witnesses : Fin ProductionDomain.semanticShape.runningCount ->
      Matrix ProductionDomain.semanticShape)
    (source : Fin ProductionDomain.semanticShape.runningCount)
    (block : Fin PiCcsDomains.production.nc.blockCount)
    (lane : Fin PiCcsDomains.production.nc.laneCount)
    (padding : Phi81ColumnLayout.blockCount
      ProductionDomain.semanticShape.carrierWidth <= block.val) :
    SourceProjection.paddedValue ProductionDomain.blockLaneDomain_covers
        (decodedData template witnesses) (Data.runningIndex source)
        block lane = 0 :=
  SourceProjection.paddedValue_block_padding
    ProductionDomain.blockLaneDomain_covers (decodedData template witnesses)
      (Data.runningIndex source) block lane padding

/-- The fixed production lane partition is exactly 54 witness-backed leaves
and ten computed-zero leaves. -/
theorem production_lane_partition :
    ProductionDomain.liveLaneCount = 54 /\
      ProductionDomain.virtualLaneCount = 10 /\
      ProductionDomain.liveLaneCount + ProductionDomain.virtualLaneCount =
        PiCcsDomains.production.nc.laneCount := by
  exact ⟨ProductionDomain.liveLaneCount_exact,
    ProductionDomain.virtualLaneCount_exact,
    ProductionDomain.live_add_virtual_lanes⟩

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessSourceProjection
