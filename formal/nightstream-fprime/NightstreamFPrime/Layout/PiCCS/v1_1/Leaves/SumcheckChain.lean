import NightstreamFPrime.Layout.SumCheck.FixedChain
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness

/-!
Paper authority: SuperNeo v1_1, section 7.3, `SumCheck(T; Q)`.
Obligation: Enforce the 28 degree-9 round equations and export the final
verifier claim for the separate `Q(r')` identity.

Inputs:
- the Initial-claim child output;
- 28 prover degree-9 polynomial messages;
- 28 transcript-derived challenges.

Outputs:
- the final child-owned SumCheck claim.

Constraint groups:
- nine stored Horner products for each round evaluation;
- one generic round equality pair;
- indexed composition over 28 rounds;
- no terminal-copy row.

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
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.Interface 8)
    (offset : Nat) : Prop where
  initial : KExprLinear (interface.initial offset)
  coefficient : ∀ roundIndex coefficientIndex,
    KExprLinear
      ((interface.round offset roundIndex).coefficient coefficientIndex)
  challenge : ∀ roundIndex,
    KExprLinear (interface.round offset roundIndex).challenge

private theorem coreRounds_linear
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.Interface 8)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    ∀ roundIndex, RoundLinear
      ((NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.coreInterface
        interface offset).round roundIndex) :=
  fun roundIndex =>
    ⟨inputs.coefficient roundIndex, inputs.challenge roundIndex⟩

private theorem core_totalFreshCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.Interface 8)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.circuit interface
        ).main offset)) = 0 := by
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.circuit_ops,
    ← NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned.circuit_ops,
    NightstreamFPrime.Layout.SumCheck.FixedChain.ownedCircuit_totalFreshCount _ offset inputs.initial
      (coreRounds_linear interface offset inputs)]

private theorem core_totalRowCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.Interface 8)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.circuit interface
        ).main offset)) = 728 := by
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.circuit_ops,
    ← NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned.circuit_ops,
    NightstreamFPrime.Layout.SumCheck.FixedChain.ownedCircuit_totalRowCount _ offset inputs.initial
      (coreRounds_linear interface offset inputs)]
  norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

/-- Exact parent-facing physical footprint for the fixed 28-round chain. -/
def footprint
    (interface : Formal.Interface logicalWidth 8 publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.sumcheckInterface interface) offset) :
    R1CS.CircuitFootprint (Formal.sumcheckCircuit interface) where
  freshColumnCount := fun _ => 0
  physicalRowCount := fun _ => 728
  freshColumnCount_eq := by
    intro offset
    unfold Formal.sumcheckCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact core_totalFreshCount _ offset (inputs offset)
  physicalRowCount_eq := by
    intro offset
    unfold Formal.sumcheckCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact core_totalRowCount _ offset (inputs offset)

theorem freshColumnCount_eq
    (interface : Formal.Interface logicalWidth 8 publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.sumcheckInterface interface) offset)
    (offset : Nat) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Formal.sumcheckCircuit interface).main offset)) = 0 :=
  (footprint interface inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (interface : Formal.Interface logicalWidth 8 publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.sumcheckInterface interface) offset)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Formal.sumcheckCircuit interface).main offset)) = 728 :=
  (footprint interface inputs).physicalRowCount_eq offset

theorem physicalPrivateColumnCount_eq
    (interface : Formal.Interface logicalWidth 8 publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.sumcheckInterface interface) offset)
    (offset : Nat) :
    localLength (Circuit.ops (Formal.sumcheckCircuit interface).main offset) +
      R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Formal.sumcheckCircuit interface).main offset)) = 672 := by
  have storedColumns :
      localLength (Circuit.ops (Formal.sumcheckCircuit interface).main
        offset) = 672 := by
    unfold Formal.sumcheckCircuit
    rw [FormalCircuit.withConstantFootprint_main,
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.localLength_eq]
    norm_num [NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.privateCount,
      NightstreamFPrime.Gadgets.SumCheck.FixedChain.Owned.privateCount,
      productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]
  rw [storedColumns, freshColumnCount_eq interface inputs offset]

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
  have below := Formal.sumcheckOutput_varsBelow_end frozen finalAt
    canonicalAssumption
  rw [← Formal.evalKStart, Formal.evalKStart_atOffset interface
    parentOffset] at below
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
  exact KExpr.varsBelow_mono _ below
    (Nat.le_trans evalKLeEvalA
      (Nat.le_trans evalALeCcs (Nat.le_trans ccsLeNorm normLeFinal)))

/-- The final claim exported to the final-identity leaf is a sum of stored
wires. -/
theorem output_linear
    (interface : Formal.Interface logicalWidth 8 publicFits)
    (offset : Nat)
    (inputs : InputsLinear (Formal.sumcheckInterface interface)
      (Formal.sumcheckStart interface)) :
    KExprLinear (Formal.sumcheckOutput interface offset) :=
  NightstreamFPrime.Layout.SumCheck.FixedChain.ownedOutput_linear _ (Formal.sumcheckStart interface) inputs.initial
    (coreRounds_linear _ _ inputs)

end NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.SumcheckChain
