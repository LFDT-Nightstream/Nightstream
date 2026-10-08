import NightstreamFPrime.Layout.PiRLC.v1_2.Sampler
import NightstreamFPrime.Layout.R1CS.Completeness

/-! Constructive physical completeness for the total scalar sampler. The
logical constructor supplies checked rows; generic R1CS lowering supplies
only its fresh interval. No sampler-success premise is needed. -/

namespace NightstreamFPrime.Layout.PiRLC.v1_2.Sampler

open NightstreamFPrime.Circuit NightstreamFPrime.Layout

theorem physical_complete (interface : Logical.Interface)
    (coordinate offset : Nat) (env : Env)
    (inputs : ∀ current, InputsAffine interface current)
    (assumptions : Logical.Assumptions interface offset) :
    ∃ completed,
      AgreesOutside env completed offset 4411 ∧
      PhysicalHolds interface coordinate offset completed := by
  obtain ⟨logicalEnv, logicalAgrees, logicalRows⟩ :=
    Lifecycle.PiRLC.v1_2.Sampler.complete interface coordinate env offset assumptions
  have logicalAgreesFixed : AgreesOutside env logicalEnv offset Logical.logicalPrivateCount := by
    rwa [Lifecycle.PiRLC.v1_2.Sampler.localLength_eq] at logicalAgrees
  have scope : ∀ expression ∈ logicalConstraints interface coordinate offset,
      expression.VarsBelow (offset + Logical.logicalPrivateCount) :=
    Lifecycle.PiRLC.v1_2.Sampler.scope interface coordinate offset assumptions
  obtain ⟨completed, physicalAgrees, physicalRows⟩ :=
    R1CS.lowerConstraints_complete logicalEnv (logicalConstraints interface coordinate offset)
      (offset + Logical.logicalPrivateCount) scope logicalRows
  refine ⟨completed, ?_, physicalRows⟩
  have combined := logicalAgreesFixed.append physicalAgrees
  rw [totalFreshCount_eq interface coordinate offset inputs] at combined
  exact combined

end NightstreamFPrime.Layout.PiRLC.v1_2.Sampler
