import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessSourceProjection

/-!
Generated full-`Z` decoder refinement into the production source table.

Assurance tier: artifact-checked for the fixed generated geometry and
model-level for the typed Rust-matrix/source-table correspondence.

Owns: alignment of every generated packed cell with `PackedWitness.unpack`;
the exact 14-child full-coordinate coverage; and specialization of the
production combined-NC source leaves to the generated 54-live/10-zero lane
partition.

Does not own: native `Mat` serialization, commitment binding, combined-NC
acceptance, delayed-projection R1CS rows, transcript scheduling, costs, or
row removal.

Emits constraints: none; direct dataflow/refinement theorem only.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `pi_ccs_nc.full_z_decoder.refinement.unpack` | generated affine address equals `PackedWitness.unpack` | direct dataflow |
| `pi_ccs_nc.full_z_decoder.refinement.live` | every live production source leaf reads its exact full-`Z` cell | derived |
| `pi_ccs_nc.full_z_decoder.refinement.padding` | every generated virtual lane is computed zero | checked/derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

namespace Artifact

export Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder
  (Child LiveLane Block LogicalColumn PaddingLane LaneSourceRecord laneSourceAt
    booleanLaneOfPadding laneSourceAt_exact paddingLane_source logicalColumnAt
    childLogicalColumnAt childLogicalColumnAt_bijective)

end Artifact

namespace GeneratedLayout

export Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder.Generated.Layout
  (logicalWidth childCount matrixRows matrixColumns booleanLaneCount)

end GeneratedLayout

namespace Production

export ProductionDomain (semanticShape blockLaneDomain_covers)

end Production

namespace Witness

export PackedWitness
  (Matrix unpack CoordinatesAligned coordinatesAligned_iff_unpack_eq
    decodedData semanticBlockOfRust)

end Witness

namespace Source

export PackedWitnessSourceProjection
  (production_live_eq_witness production_lane_padding_zero)

end Source

theorem generated_logicalWidth_eq_production :
    GeneratedLayout.logicalWidth = Production.semanticShape.carrierWidth := by
  decide

theorem generated_matrixRows_eq_ringDegree :
    GeneratedLayout.matrixRows = ringDegree := by
  decide

theorem generated_matrixColumns_eq_production :
    GeneratedLayout.matrixColumns =
      Phi81ColumnLayout.blockCount Production.semanticShape.logicalWidth := by
  decide

theorem generated_childCount_eq_production :
    GeneratedLayout.childCount = Production.semanticShape.runningCount := by
  decide

def semanticLane (lane : Artifact.LiveLane) : Fin ringDegree :=
  Fin.cast generated_matrixRows_eq_ringDegree lane

def rustBlock (block : Artifact.Block) :
    Fin (Phi81ColumnLayout.blockCount Production.semanticShape.logicalWidth) :=
  Fin.cast generated_matrixColumns_eq_production block

def semanticColumn (column : Artifact.LogicalColumn) :
    Fin Production.semanticShape.carrierWidth :=
  Fin.cast generated_logicalWidth_eq_production column

def productionChild (child : Artifact.Child) :
    Fin Production.semanticShape.runningCount :=
  Fin.cast generated_childCount_eq_production child

/-- Every generated affine decoder address is the exact coordinate read by
`PackedWitness.unpack`; this is a full-width quantified theorem, not a
sampled list comparison. -/
theorem unpack_at_generatedAddress
    (witness : Witness.Matrix Production.semanticShape)
    (address : Artifact.Block × Artifact.LiveLane) :
    Witness.unpack witness
        (semanticColumn (Artifact.logicalColumnAt address)) =
      witness (semanticLane address.2) (rustBlock address.1) := by
  have aligned : Witness.CoordinatesAligned witness (Witness.unpack witness) :=
    (Witness.coordinatesAligned_iff_unpack_eq witness
      (Witness.unpack witness)).2 rfl
  have cell := aligned (semanticLane address.2) (rustBlock address.1)
  symm
  rw [cell]
  congr 1

/-- The generated decoder's live source-table leaf is the corresponding
cell of the actual full packed witness. -/
theorem production_live_eq_generatedWitnessCell
    (template : Data Production.semanticShape)
    (witnesses : Fin Production.semanticShape.runningCount ->
      Witness.Matrix Production.semanticShape)
    (child : Artifact.Child)
    (block : Artifact.Block)
    (lane : Artifact.LiveLane) :
    SourceProjection.paddedValue Production.blockLaneDomain_covers
        (Witness.decodedData template witnesses)
        (Data.runningIndex (productionChild child))
        (PiCcsDomains.production.nc.carrierBlock
          Production.blockLaneDomain_covers
          (Witness.semanticBlockOfRust (rustBlock block)))
        (PiCcsDomains.production.nc.phi81Lane
          Production.blockLaneDomain_covers (semanticLane lane)) =
      witnesses (productionChild child) (semanticLane lane)
        (rustBlock block) := by
  simpa using Source.production_live_eq_witness template witnesses
    (productionChild child) (Witness.semanticBlockOfRust (rustBlock block))
      (semanticLane lane)

/-- Every generated padding-lane record is computed zero by the production
source table and reads no witness cell. -/
theorem production_padding_eq_zero
    (template : Data Production.semanticShape)
    (witnesses : Fin Production.semanticShape.runningCount ->
      Witness.Matrix Production.semanticShape)
    (child : Artifact.Child)
    (block : Fin PiCcsDomains.production.nc.blockCount)
    (padding : Artifact.PaddingLane) :
    Artifact.laneSourceAt (Artifact.booleanLaneOfPadding padding) =
        { booleanLane := (Artifact.booleanLaneOfPadding padding).val
          witnessLane := none } /\
      SourceProjection.paddedValue Production.blockLaneDomain_covers
          (Witness.decodedData template witnesses)
          (Data.runningIndex (productionChild child)) block
          (Fin.cast (by decide)
            (Artifact.booleanLaneOfPadding padding)) = 0 := by
  constructor
  · generalize currentEq :
      Artifact.laneSourceAt (Artifact.booleanLaneOfPadding padding) =
        current
    cases current with
    | mk booleanLane witnessLane =>
        have booleanEq : booleanLane =
            (Artifact.booleanLaneOfPadding padding).val := by
          simpa [currentEq] using (Artifact.laneSourceAt_exact
            (Artifact.booleanLaneOfPadding padding)).1
        have witnessEq : witnessLane = none := by
          simpa [currentEq] using Artifact.paddingLane_source padding
        subst booleanLane
        subst witnessLane
        rfl
  · apply Source.production_lane_padding_zero template witnesses
      (productionChild child) block
        (Fin.cast (by decide) (Artifact.booleanLaneOfPadding padding))
    change 54 <= 54 + padding.val
    omega

/-- Exact generated ownership summary: all fourteen full matrices are
covered bijectively, and the Boolean lane cube has exactly 54 witness-backed
plus ten computed-zero leaves. -/
theorem generated_full_decoder_and_lane_partition :
    (Function.Injective Artifact.childLogicalColumnAt /\
      Function.Surjective Artifact.childLogicalColumnAt) /\
      GeneratedLayout.matrixRows = 54 /\
      GeneratedLayout.booleanLaneCount - GeneratedLayout.matrixRows = 10 := by
  exact ⟨Artifact.childLogicalColumnAt_bijective, by decide, by decide⟩

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder
