import NightstreamFPrime.Layout.SumCheck.FixedChain
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness

/-!
Paper authority: SuperNeo v1_1, section 7.3, `SumCheck(T; Q)`.
Obligation: Enforce the 25 degree-9 round equations and export the final
verifier claim for the separate `Q(r')` identity.

Inputs:
- the Initial-claim child output;
- 25 prover degree-9 polynomial messages;
- 25 transcript-derived challenges.

Outputs:
- the final child-owned SumCheck claim.

Constraint groups:
- one generic round equality pair;
- indexed composition over 25 rounds;
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
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.Interface 9)
    (offset : Nat) : Prop where
  initial : KExprLinear (interface.initial offset)
  coefficient : ∀ roundIndex coefficientIndex,
    KExprLinear
      ((interface.round offset roundIndex).coefficient coefficientIndex)
  challenge : ∀ roundIndex,
    KExprLinear (interface.round offset roundIndex).challenge

private theorem coreRounds_linear
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.Interface 9)
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
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.Interface 9)
    (offset : Nat) :
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.coreInterface
      interface offset).rounds.length = 25 := by
  simp [NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.coreInterface,
    NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned.Interface.rounds,
    productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

private theorem coreRounds_nonempty
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.Interface 9)
    (offset : Nat) :
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.coreInterface
      interface offset).rounds ≠ [] := by
  intro empty
  have length := congrArg List.length empty
  rw [coreRounds_length interface offset] at length
  simp at length

private theorem flatConstraints_eq_coreConstraints
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.Interface 9)
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
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.Interface 9)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalFreshCount
      (NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned.constraints
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.coreInterface
        interface offset)) = 378560 := by
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
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.Interface 9)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalRowCount
      (NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned.constraints
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.coreInterface
        interface offset)) = 378610 := by
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

/-- Exact parent-facing physical footprint for the fixed 25-round chain. -/
def footprint
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.sumcheckInterface interface) offset) :
    R1CS.CircuitFootprint (Formal.sumcheckCircuit interface) where
  freshColumnCount := fun _ => 378560
  physicalRowCount := fun _ => 378610
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
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.sumcheckInterface interface) offset)
    (offset : Nat) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Formal.sumcheckCircuit interface).main offset)) = 378560 :=
  (footprint interface inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.sumcheckInterface interface) offset)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Formal.sumcheckCircuit interface).main offset)) = 378610 :=
  (footprint interface inputs).physicalRowCount_eq offset

theorem physicalPrivateColumnCount_eq
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.sumcheckInterface interface) offset)
    (offset : Nat) :
    localLength (Circuit.ops (Formal.sumcheckCircuit interface).main offset) +
      R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Formal.sumcheckCircuit interface).main offset)) = 378560 := by
  have noLogicalColumns :
      localLength (Circuit.ops (Formal.sumcheckCircuit interface).main
        offset) = 0 := by
    unfold Formal.sumcheckCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.localLength_eq
      (Formal.sumcheckInterface interface) offset
  rw [noLogicalColumns, freshColumnCount_eq interface inputs offset]

/-- The final SumCheck claim is in causal scope at the final-identity child. -/
theorem output_varsBelow_finalIdentity
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (parentOffset : Nat)
    (env : Env)
    (assumptions :
      (Formal.sumcheckCircuit (Formal.atOffset interface parentOffset)
        ).assumptions (Formal.sumcheckOffset interface parentOffset) env) :
    (Formal.sumcheckOutput (Formal.atOffset interface parentOffset)
      (Formal.finalIdentityOffset relation interface parentOffset)).VarsBelow
        (Formal.finalIdentityOffset relation interface parentOffset) := by
  let frozen := Formal.atOffset interface parentOffset
  let sumcheckAt := Formal.sumcheckOffset interface parentOffset
  let evalKAt := Formal.evalKOffset interface parentOffset
  let evalAAt := Formal.evalAOffset interface parentOffset
  let ccsAt := Formal.ccsOffset interface parentOffset
  let normAt := Formal.normOffset relation interface parentOffset
  let finalAt := Formal.finalIdentityOffset relation interface parentOffset
  have childAssumption :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.Assumptions
        (Formal.sumcheckInterface frozen) sumcheckAt (fun _ => 0) := by
    exact assumptions
  have canonicalAssumption :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.Assumptions
        (Formal.sumcheckInterface frozen) (Formal.sumcheckStart frozen)
          (fun _ => 0) := by
    rw [Formal.sumcheckStart_atOffset interface parentOffset]
    exact childAssumption
  have below := Formal.sumcheckOutput_varsBelow_start frozen finalAt
    canonicalAssumption
  have sumcheckLeEvalK : sumcheckAt ≤ evalKAt := by
    dsimp [evalKAt]
    unfold Formal.evalKOffset Formal.nextOffset
    omega
  have evalKLeEvalA : evalKAt ≤ evalAAt := by
    dsimp [evalAAt]
    unfold Formal.evalAOffset Formal.nextOffset
    omega
  have evalALeCcs : evalAAt ≤ ccsAt := by
    dsimp [ccsAt]
    unfold Formal.ccsOffset Formal.nextOffset
    omega
  have ccsLeNorm : ccsAt ≤ normAt := by
    dsimp [normAt]
    unfold Formal.normOffset Formal.nextOffset
    omega
  have normLeFinal : normAt ≤ finalAt := by
    dsimp [finalAt]
    unfold Formal.finalIdentityOffset Formal.nextOffset
    omega
  have sumcheckLeFinal : sumcheckAt ≤ finalAt := by
    exact Nat.le_trans sumcheckLeEvalK
      (Nat.le_trans evalKLeEvalA
        (Nat.le_trans evalALeCcs (Nat.le_trans ccsLeNorm normLeFinal)))
  have startLeFinal : Formal.sumcheckStart frozen ≤ finalAt := by
    rw [Formal.sumcheckStart_atOffset interface parentOffset]
    exact sumcheckLeFinal
  exact KExpr.varsBelow_mono _ below startLeFinal

/-- Exact syntactic shape exported to the separate final-identity leaf. -/
theorem output_mulCounts
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (offset : Nat)
    (inputs : InputsLinear (Formal.sumcheckInterface interface)
      (Formal.sumcheckStart interface)) :
    R1CS.mulCount (Formal.sumcheckOutput interface offset).c0 = 2558 ∧
      R1CS.mulCount (Formal.sumcheckOutput interface offset).c1 = 2557 := by
  unfold Formal.sumcheckOutput
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.output
    NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned.output
  apply outputFrom_mulCounts_of_nonempty
  · exact coreRounds_nonempty _ _
  · exact coreRounds_linear _ _ inputs

end NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.SumcheckChain
