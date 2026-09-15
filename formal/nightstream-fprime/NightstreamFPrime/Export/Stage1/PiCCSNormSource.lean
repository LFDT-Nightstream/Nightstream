import NightstreamFPrime.Export.Stage1.PiCCSNormCache
import NightstreamFPrime.Export.Stage1.PiCCSFirstRound
import NightstreamFPrime.Export.SignedUnitSourceInput
import NightstreamFPrime.Export.Stage1.PiCCSAggregatedImages
import NightstreamFPrime.Spec.Folding.PiCCS.CanonicalRowLayout

/-! Original signed-mask reads and complete-block coordinates for the norm
scan. No norm polynomial loop or protocol image table is constructed here. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSNormSource

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Lifecycle

abbrev PairCount : Nat := ringDegree / 2

/-- Preserve the scalar decoder's positive/negative priority and omitted
source behavior. Code order is the existing negative, zero, positive order. -/
def sourceCode (masks : Array (Nat × Nat)) (source : Fin productionShape.sourceCount)
    (lane : Fin ringDegree) : Fin 3 :=
  let pair := masks[source.val]?.getD (0, 0)
  if pair.1.testBit lane.val then ⟨2, by decide⟩
  else if pair.2.testBit lane.val then ⟨0, by decide⟩
  else ⟨1, by decide⟩

/-- The existing complete carrier assignment reconstructed by the pair runner.
Missing blocks and missing source entries use the decoder's exact zero. -/
def assignments (masks : Array (Array (Nat × Nat))) :
    Fin productionShape.sourceCount → Phi81Relation.Assignment PiCCSSourceImages.shape :=
  fun source column => SignedUnitSourceInput.scalar
    (masks[column.val / ringDegree]?.getD #[])
    ⟨source.val, source.isLt⟩ ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩

/-- The selected full-carrier column layout. The unit argument delays its
construction until the caller prepares the selected scan. -/
def canonicalLayout (_ : Unit) :
    ColumnLayout cubeVariables PiCCSSourceImages.shape.carrierWidth :=
  Folding.PiCCS.CanonicalRowLayout.layout cubeVariables PiCCSSourceImages.shape.carrierWidth
    (PerApplicationFixedPoint.relation Poseidon2HashChainV1Package.application
      Poseidon2HashChainV1Package.fits).cubeFits

/-- Ring degree 54 is even, so no adjacent pair crosses a complete block. -/
def lowLane (pair : Fin PairCount) : Fin ringDegree :=
  ⟨2 * pair.val, by
    have bound := pair.isLt
    change pair.val < 27 at bound
    change 2 * pair.val < 54
    omega⟩

def highLane (pair : Fin PairCount) : Fin ringDegree :=
  ⟨2 * pair.val + 1, by
    have bound := pair.isLt
    change pair.val < 27 at bound
    change 2 * pair.val + 1 < 54
    omega⟩

/-- Numeric suffix index after removing the first Boolean coordinate. -/
def pairIndex (block : Nat) (pair : Fin PairCount) : Nat :=
  block * PairCount + pair.val

/-- The scan code is exactly the original source decoder, including omitted
entries and its positive-before-negative branch order. -/
theorem signedValue_sourceCode (masks : Array (Nat × Nat))
    (source : Fin productionShape.sourceCount) (lane : Fin ringDegree) :
    PiCCSNormCache.signedValue (sourceCode masks source lane) =
      SignedUnitSourceInput.scalar masks ⟨source.val, source.isLt⟩ lane := by
  simp only [sourceCode, SignedUnitSourceInput.scalar]
  split_ifs <;> rfl

private theorem assignment_carrierColumn (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) (block : Fin PiCCSSourceImages.blockCount)
    (lane : Fin ringDegree) :
    assignments masks source
        (Phi81CarrierLayout.carrierColumn (logicalWidth := PiCCSSourceImages.logicalWidth)
          block lane) =
      SignedUnitSourceInput.scalar (masks[block.val]?.getD #[])
        ⟨source.val, source.isLt⟩ lane := by
  have decoded := Phi81CarrierLayout.decode_carrierColumn
    (logicalWidth := PiCCSSourceImages.logicalWidth) block lane
  have blockIndex :
      (Phi81CarrierLayout.carrierColumn (logicalWidth := PiCCSSourceImages.logicalWidth) block lane).val / ringDegree = block.val :=
    congrArg (fun value => value.1.val) decoded
  have laneIndex :
      (⟨(Phi81CarrierLayout.carrierColumn (logicalWidth := PiCCSSourceImages.logicalWidth) block lane).val % ringDegree,
        Nat.mod_lt _ (by decide)⟩ : Fin ringDegree) = lane :=
    congrArg Prod.snd decoded
  simp only [assignments, blockIndex, laneIndex]

/-- Each original mask coefficient is the actual nonlinear message's source
field at its canonical full-carrier vertex. The fresh vector is arbitrary:
this field never depends on a matrix image or a successful matrix lookup. -/
theorem sourceCode_sourceAssignment (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) (block : Fin PiCCSSourceImages.blockCount)
    (lane : Fin ringDegree) (fresh : Vector F ProductionRelation.matrixCount) :
    K.embed (PiCCSNormCache.signedValue
        (sourceCode (masks[block.val]?.getD #[]) source lane)) =
      (PiCCSAggregatedImages.nonlinearMessage (canonicalLayout ()) (assignments masks)
        ((canonicalLayout ()).toVertex
          (Phi81CarrierLayout.carrierColumn (logicalWidth := PiCCSSourceImages.logicalWidth)
            block lane)) fresh).sourceAssignment source := by
  rw [signedValue_sourceCode]
  change K.embed (SignedUnitSourceInput.scalar (masks[block.val]?.getD #[])
      ⟨source.val, source.isLt⟩ lane) =
    K.embed ((canonicalLayout ()).paddedValue (0 : F) (assignments masks source)
      ((canonicalLayout ()).toVertex (Phi81CarrierLayout.carrierColumn (logicalWidth := PiCCSSourceImages.logicalWidth) block lane)))
  rw [ColumnLayout.paddedValue_toVertex, assignment_carrierColumn]

/-- Both endpoints of every adjacent pair remain inside one 54-lane block. -/
theorem lowLane_global (block : Nat) (pair : Fin PairCount) :
    block * ringDegree + (lowLane pair).val = 2 * pairIndex block pair := by
  change block * 54 + 2 * pair.val = 2 * (block * 27 + pair.val)
  omega

theorem highLane_global (block : Nat) (pair : Fin PairCount) :
    block * ringDegree + (highLane pair).val = 2 * pairIndex block pair + 1 := by
  change block * 54 + (2 * pair.val + 1) = 2 * (block * 27 + pair.val) + 1
  omega

/-- The existing complete-carrier and cube bounds place each block pair in
the selected first-round suffix domain. No new range bound is assumed. -/
theorem pairIndex_bound (block : Fin PiCCSSourceImages.blockCount) (pair : Fin PairCount) :
    pairIndex block.val pair < 2 ^ (cubeVariables - 1) := by
  have live : block.val * ringDegree + (highLane pair).val <
      PiCCSSourceImages.shape.carrierWidth :=
    Phi81CarrierLayout.flatIndex_lt_carrierWidth
      (logicalWidth := PiCCSSourceImages.logicalWidth) block (highLane pair)
  rw [highLane_global] at live
  have bound := Nat.lt_of_lt_of_le live (canonicalLayout ()).columns_le
  change 2 * pairIndex block.val pair + 1 < 2 ^ 28 at bound
  change pairIndex block.val pair < 2 ^ 27
  omega

/-- The existing numeric Boolean vertex at this complete-block pair index. -/
def pairSuffix (block : Fin PiCCSSourceImages.blockCount) (pair : Fin PairCount) :
    BooleanVertex (cubeVariables - 1) :=
  NumericBooleanDomain.vertex (cubeVariables - 1)
    ⟨pairIndex block.val pair, pairIndex_bound block pair⟩

private theorem canonicalVertex_eq_of_index
    (column : Fin PiCCSSourceImages.shape.carrierWidth) (vertex : BooleanVertex cubeVariables)
    (same : NumericBooleanDomain.index vertex = column.val) :
    (canonicalLayout ()).toVertex column = vertex := by
  apply (canonicalLayout ()).toVertex_toColumn
  exact (Folding.PiCCS.CanonicalRowLayout.toColumn?_eq_some_iff cubeVariables
    PiCCSSourceImages.shape.carrierWidth (canonicalLayout ()).columns_le vertex column).2 same

/-- The low scalar code is the source field at numeric endpoint 2*pairIndex. -/
theorem sourceCode_low_sourceAssignment (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) (block : Fin PiCCSSourceImages.blockCount)
    (pair : Fin PairCount) (fresh : Vector F ProductionRelation.matrixCount) :
    K.embed (PiCCSNormCache.signedValue
        (sourceCode (masks[block.val]?.getD #[]) source (lowLane pair))) =
      (PiCCSAggregatedImages.nonlinearMessage (canonicalLayout ()) (assignments masks)
        (PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) false (pairSuffix block pair))
        fresh).sourceAssignment source := by
  have indexed :
      NumericBooleanDomain.index
          (PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) false (pairSuffix block pair)) =
        (Phi81CarrierLayout.carrierColumn (logicalWidth := PiCCSSourceImages.logicalWidth) block (lowLane pair)).val := by
    change 0 + 2 * NumericBooleanDomain.index (pairSuffix block pair) =
      block.val * ringDegree + (lowLane pair).val
    rw [Nat.zero_add, pairSuffix, NumericBooleanDomain.index_vertex]
    exact (lowLane_global block.val pair).symm
  have vertexEqual := canonicalVertex_eq_of_index
    (Phi81CarrierLayout.carrierColumn (logicalWidth := PiCCSSourceImages.logicalWidth) block (lowLane pair)) _ indexed
  have value := sourceCode_sourceAssignment masks source block (lowLane pair) fresh
  rw [vertexEqual] at value
  exact value

/-- The high scalar code is the source field at numeric endpoint 2*pairIndex+1. -/
theorem sourceCode_high_sourceAssignment (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) (block : Fin PiCCSSourceImages.blockCount)
    (pair : Fin PairCount) (fresh : Vector F ProductionRelation.matrixCount) :
    K.embed (PiCCSNormCache.signedValue
        (sourceCode (masks[block.val]?.getD #[]) source (highLane pair))) =
      (PiCCSAggregatedImages.nonlinearMessage (canonicalLayout ()) (assignments masks)
        (PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) true (pairSuffix block pair))
        fresh).sourceAssignment source := by
  have indexed :
      NumericBooleanDomain.index
          (PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) true (pairSuffix block pair)) =
        (Phi81CarrierLayout.carrierColumn (logicalWidth := PiCCSSourceImages.logicalWidth) block (highLane pair)).val := by
    change 1 + 2 * NumericBooleanDomain.index (pairSuffix block pair) =
      block.val * ringDegree + (highLane pair).val
    rw [pairSuffix, NumericBooleanDomain.index_vertex, highLane_global]
    change 1 + 2 * pairIndex block.val pair = 2 * pairIndex block.val pair + 1
    omega
  have vertexEqual := canonicalVertex_eq_of_index
    (Phi81CarrierLayout.carrierColumn (logicalWidth := PiCCSSourceImages.logicalWidth) block (highLane pair)) _ indexed
  have value := sourceCode_sourceAssignment masks source block (highLane pair) fresh
  rw [vertexEqual] at value
  exact value

/-- Every source field is exactly zero on the Boolean-domain suffix beyond
the complete carrier width. This does not discard any retained carrier tail. -/
theorem sourceAssignment_padding (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) (vertex : BooleanVertex cubeVariables)
    (padding : PiCCSSourceImages.shape.carrierWidth ≤ NumericBooleanDomain.index vertex)
    (fresh : Vector F ProductionRelation.matrixCount) :
    (PiCCSAggregatedImages.nonlinearMessage (canonicalLayout ()) (assignments masks)
        vertex fresh).sourceAssignment source = K.zero := by
  have decoded : (canonicalLayout ()).toColumn? vertex = none :=
    (Folding.PiCCS.CanonicalRowLayout.toColumn?_eq_none_iff cubeVariables
      PiCCSSourceImages.shape.carrierWidth (canonicalLayout ()).columns_le vertex).2 padding
  change K.embed ((canonicalLayout ()).paddedValue (0 : F) (assignments masks source) vertex) = K.zero
  rw [ColumnLayout.paddedValue, decoded]
  rfl

end NightstreamFPrime.Export.Stage1.PiCCSNormSource
