import NightstreamFPrime.Layout.PiRLC.v1_1.Sampler
import NightstreamFPrime.Layout.R1CS.Segments

/-!
Owns the constructive physical completeness proof for one complete PiRLC
scalar sampler. This proof is isolated from the normal sampler layout path
because only the axiom gate consumes it.
-/

namespace NightstreamFPrime.Layout.PiRLC.v1_1.Sampler

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout

set_option maxRecDepth 100000 in -- fixed-size: one scalar sampler
theorem physical_complete (interface : Logical.Interface)
    (coordinate offset : Nat) (env : Env)
    (inputs : ∀ current, InputsAffine interface current)
    (assumptions : Logical.Assumptions interface offset env)
    (relation : Logical.RelationHolds interface coordinate offset env) :
    ∃ completed,
      AgreesOutside env completed offset 59247 ∧
      PhysicalHolds interface coordinate offset completed := by
  rcases Logical.completeness interface coordinate env offset assumptions
      relation with ⟨logicalEnv, logicalAgrees, logicalRows⟩
  have logicalAgreesFixed : AgreesOutside env logicalEnv offset
      Logical.logicalPrivateCount := by
    rw [Logical.localLength_eq] at logicalAgrees
    exact logicalAgrees
  have logicalAssumptions : Logical.Assumptions interface offset logicalEnv :=
    assumptions
  have logicalSpecification : Logical.SpecHolds interface coordinate offset
      logicalEnv := by
    apply Logical.soundness interface coordinate logicalEnv offset
      logicalAssumptions
    exact holdsFlat_implies_holds logicalEnv _ logicalRows
  have scope : ∀ expression ∈
      logicalConstraints interface coordinate offset,
      expression.VarsBelow (offset + Logical.logicalPrivateCount) := by
    exact Logical.flatConstraints_varsBelow interface coordinate offset
      logicalEnv logicalAssumptions logicalSpecification
  have logicalConstraintsHold : ConstraintsHold logicalEnv
      (logicalConstraints interface coordinate offset) := logicalRows
  have segmentScope : ∀ expression ∈
      (childConstraintLists interface coordinate offset).flatten,
      expression.VarsBelow (offset + Logical.logicalPrivateCount) := by
    rw [← logicalConstraints_eq_flatten]
    exact scope
  have segmentLogical : ConstraintsHold logicalEnv
      (childConstraintLists interface coordinate offset).flatten := by
    rw [← logicalConstraints_eq_flatten]
    exact logicalConstraintsHold
  rcases R1CS.lowerSegments_complete logicalEnv
      (childConstraintLists interface coordinate offset)
    (offset + Logical.logicalPrivateCount) segmentScope segmentLogical with
    ⟨completed, physicalAgrees, segmentRows⟩
  refine ⟨completed, ?_, segmentRows⟩
  have combined := logicalAgreesFixed.append physicalAgrees
  rw [← logicalConstraints_eq_flatten interface coordinate offset,
    totalFreshCount_eq interface coordinate offset inputs] at combined
  simpa [Logical.logicalPrivateCount] using combined

end NightstreamFPrime.Layout.PiRLC.v1_1.Sampler
