import NightstreamFPrime.Layout.SumCheck.FixedChain
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness

/-!
Paper authority: SuperNeo v1_1, section 7.3, `SumCheck(T; Q)`.
Obligation: Enforce the 24 degree-4 round equations and export the final
verifier claim for the separate `Q(r')` identity.

Inputs:
- the Initial-claim child output;
- 24 prover degree-4 polynomial messages;
- 24 transcript-derived challenges.

Outputs:
- the final child-owned SumCheck claim.

Constraint groups:
- one generic round equality pair;
- indexed composition over 24 rounds;
- no logical witness or terminal-copy row.

Parent coverage:
- `Formal.opsAt`, child `piccs.v1_1.sumcheck_chain`.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.SumcheckChain

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.SumCheck
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Layout.Polynomial.Horner
open NightstreamFPrime.Layout.SumCheck.FixedChain
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Stable physical wire shape for the initial claim, prover coefficients,
and verifier-derived challenges. -/
structure InputsLinear
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.Interface 4)
    (offset : Nat) : Prop where
  initial : KExprLinear (interface.initial offset)
  coefficient : ∀ roundIndex coefficientIndex,
    KExprLinear
      ((interface.round offset roundIndex).coefficient coefficientIndex)
  challenge : ∀ roundIndex,
    KExprLinear (interface.round offset roundIndex).challenge

private theorem coreRounds_linear
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.Interface 4)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    ∀ round ∈
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.coreInterface
        interface offset).rounds,
      RoundLinear round := by
  intro round member
  rw [NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned.Interface.rounds,
    List.mem_ofFn'] at member
  rcases member with ⟨roundIndex, rfl⟩
  exact ⟨inputs.coefficient roundIndex, inputs.challenge roundIndex⟩

private theorem coreRounds_length
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.Interface 4)
    (offset : Nat) :
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.coreInterface
      interface offset).rounds.length = 24 := by
  simp [NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.coreInterface,
    NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned.Interface.rounds,
    productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

private theorem coreRounds_nonempty
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.Interface 4)
    (offset : Nat) :
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.coreInterface
      interface offset).rounds ≠ [] := by
  intro empty
  have length := congrArg List.length empty
  rw [coreRounds_length interface offset] at length
  simp at length

private theorem flatConstraints_eq_coreConstraints
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.Interface 4)
    (offset : Nat) :
    flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.circuit interface
        ).main offset) =
      NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned.constraints
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.coreInterface
          interface offset) := by
  unfold NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.circuit
  exact NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned.flatConstraints_eq
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.coreInterface
      interface offset) offset

private theorem core_totalFreshCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.Interface 4)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalFreshCount
      (NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned.constraints
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.coreInterface
          interface offset)) = 11053 := by
  let core :=
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.coreInterface
      interface offset
  unfold NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned.constraints
  rw [constraintsFrom_totalFreshCount_of_nonempty core.initial core.rounds
    (coreRounds_nonempty interface offset)
    (coreRounds_linear interface offset inputs)]
  have initialCount : KExprMulCount core.initial = 0 := by
    change R1CS.mulCount (interface.initial offset).c0 +
      R1CS.mulCount (interface.initial offset).c1 = 0
    rw [inputs.initial.c0_mulCount, inputs.initial.c1_mulCount]
  rw [initialCount, coreRounds_length interface offset]

private theorem core_totalRowCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.Interface 4)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalRowCount
      (NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned.constraints
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.coreInterface
          interface offset)) = 11101 := by
  let core :=
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.coreInterface
      interface offset
  unfold NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned.constraints
  rw [constraintsFrom_totalRowCount_of_nonempty core.initial core.rounds
    (coreRounds_nonempty interface offset)
    (coreRounds_linear interface offset inputs)]
  have initialCount : KExprMulCount core.initial = 0 := by
    change R1CS.mulCount (interface.initial offset).c0 +
      R1CS.mulCount (interface.initial offset).c1 = 0
    rw [inputs.initial.c0_mulCount, inputs.initial.c1_mulCount]
  rw [initialCount, coreRounds_length interface offset]

/-- Exact parent-facing physical footprint for the fixed 24-round chain. -/
def footprint
    (interface : Formal.Interface logicalWidth 4 publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.sumcheckInterface interface) offset) :
    R1CS.CircuitFootprint (Formal.sumcheckCircuit interface) where
  freshColumnCount := fun _ => 11053
  physicalRowCount := fun _ => 11101
  freshColumnCount_eq := by
    intro offset
    unfold Formal.sumcheckCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    rw [flatConstraints_eq_coreConstraints]
    exact core_totalFreshCount _ offset (inputs offset)
  physicalRowCount_eq := by
    intro offset
    unfold Formal.sumcheckCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    rw [flatConstraints_eq_coreConstraints]
    exact core_totalRowCount _ offset (inputs offset)

theorem freshColumnCount_eq
    (interface : Formal.Interface logicalWidth 4 publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.sumcheckInterface interface) offset)
    (offset : Nat) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Formal.sumcheckCircuit interface).main offset)) = 11053 :=
  (footprint interface inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (interface : Formal.Interface logicalWidth 4 publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.sumcheckInterface interface) offset)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Formal.sumcheckCircuit interface).main offset)) = 11101 :=
  (footprint interface inputs).physicalRowCount_eq offset

theorem physicalPrivateColumnCount_eq
    (interface : Formal.Interface logicalWidth 4 publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.sumcheckInterface interface) offset)
    (offset : Nat) :
    localLength (Circuit.ops (Formal.sumcheckCircuit interface).main offset) +
      R1CS.totalFreshCount (flatConstraints (Circuit.ops
        (Formal.sumcheckCircuit interface).main offset)) = 11053 := by
  have noLogicalColumns :
      localLength (Circuit.ops (Formal.sumcheckCircuit interface).main
        offset) = 0 := by
    unfold Formal.sumcheckCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.localLength_eq
      (Formal.sumcheckInterface interface) offset
  rw [noLogicalColumns, freshColumnCount_eq interface inputs offset]

end NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.SumcheckChain
