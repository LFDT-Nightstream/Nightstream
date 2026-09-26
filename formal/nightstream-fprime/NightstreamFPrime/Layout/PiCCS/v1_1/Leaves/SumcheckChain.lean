import NightstreamFPrime.Layout.SumCheck.CompactChain
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
- one generic round equality pair;
- indexed composition over 28 rounds;
- one shared materialized Horner trace and no terminal-copy row.

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

private theorem coreCosts
    (interface : NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.Interface 9)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    localLength (Circuit.ops (NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.circuit
      interface).main offset) = 504 ∧
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.circuit interface).main offset)) = 1764 ∧
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.circuit interface).main offset)) = 2324 := by
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.circuit_ops]
  exact Layout.SumCheck.CompactChain.production_costs
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.coreInterface interface offset)
    offset inputs.initial (fun index => ⟨inputs.coefficient index, inputs.challenge index⟩)

/-- Exact parent-facing footprint of the materialized 28-round chain. -/
def footprint
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (inputs : ∀ offset, InputsLinear (Formal.sumcheckInterface interface) offset) :
    R1CS.CircuitFootprint (Formal.sumcheckCircuit interface) where
  freshColumnCount := fun _ => 1764
  physicalRowCount := fun _ => 2324
  freshColumnCount_eq := by
    intro offset
    unfold Formal.sumcheckCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact (coreCosts _ offset (inputs offset)).2.1
  physicalRowCount_eq := by
    intro offset
    unfold Formal.sumcheckCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact (coreCosts _ offset (inputs offset)).2.2

theorem freshColumnCount_eq
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (inputs : ∀ offset, InputsLinear (Formal.sumcheckInterface interface) offset)
    (offset : Nat) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Formal.sumcheckCircuit interface).main offset)) = 1764 :=
  (footprint interface inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (inputs : ∀ offset, InputsLinear (Formal.sumcheckInterface interface) offset)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Formal.sumcheckCircuit interface).main offset)) = 2324 :=
  (footprint interface inputs).physicalRowCount_eq offset

theorem physicalPrivateColumnCount_eq
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (inputs : ∀ offset, InputsLinear (Formal.sumcheckInterface interface) offset)
    (offset : Nat) :
    localLength (Circuit.ops (Formal.sumcheckCircuit interface).main offset) +
      R1CS.totalFreshCount (flatConstraints (Circuit.ops
        (Formal.sumcheckCircuit interface).main offset)) = 2268 := by
  rw [freshColumnCount_eq interface inputs offset]
  have logical := (coreCosts _ offset (inputs offset)).1
  simpa only [Formal.sumcheckCircuit, FormalCircuit.withConstantFootprint_main]
    using congrArg (fun count => count + 1764) logical

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
  have sumcheckLeEvalK : sumcheckAt +
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.privateCount
        (ProductionKey.degreeBound relation) ≤ evalKAt := by
    dsimp [evalKAt]
    unfold Formal.evalKOffset Formal.nextOffset Formal.childLength Formal.sumcheckCircuit
    rw [FormalCircuit.withConstantFootprint_main,
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.localLength_eq]
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
  have sumcheckLeFinal : sumcheckAt +
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.privateCount
        (ProductionKey.degreeBound relation) ≤ finalAt := by
    exact Nat.le_trans sumcheckLeEvalK
      (Nat.le_trans evalKLeEvalA
        (Nat.le_trans evalALeCcs (Nat.le_trans ccsLeNorm normLeFinal)))
  have startLeFinal : Formal.sumcheckStart frozen +
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.privateCount
        (ProductionKey.degreeBound relation) ≤ finalAt := by
    rw [Formal.sumcheckStart_atOffset interface parentOffset]
    exact sumcheckLeFinal
  exact KExpr.varsBelow_mono _ below startLeFinal

/-- The final identity receives an affine view of the owned Horner result. -/
theorem output_mulCounts
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (offset : Nat)
    (inputs : InputsLinear (Formal.sumcheckInterface interface)
      (Formal.sumcheckStart interface)) :
    R1CS.mulCount (Formal.sumcheckOutput interface offset).c0 = 0 ∧
      R1CS.mulCount (Formal.sumcheckOutput interface offset).c1 = 0 := by
  have linear := Layout.SumCheck.CompactChain.compile_output_linear
    (Formal.sumcheckStart interface)
    ((Formal.sumcheckInterface interface).initial (Formal.sumcheckStart interface))
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.coreInterface
      (Formal.sumcheckInterface interface) (Formal.sumcheckStart interface)).rounds
    inputs.initial (by
      intro round member
      rw [FixedChain.Owned.Interface.rounds, List.mem_ofFn'] at member
      obtain ⟨index, rfl⟩ := member
      exact ⟨inputs.coefficient index, inputs.challenge index⟩)
  exact ⟨linear.c0_mulCount, linear.c1_mulCount⟩

end NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.SumcheckChain
