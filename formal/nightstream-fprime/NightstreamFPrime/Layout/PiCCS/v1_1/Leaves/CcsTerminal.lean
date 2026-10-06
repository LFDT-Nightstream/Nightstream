import NightstreamFPrime.Layout.Polynomial.Sparse
import NightstreamFPrime.Layout.R1CS.Completeness
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness

/-!
Paper authority: SuperNeo v1.2, section 7.3, Step 4, `F`.
Obligation: Lower the materialized evaluation of the fixed 8-term selective
constraint polynomial over all 7 `Eval_A` matrix images.

Inputs:
- 7 selective matrix images;
- the relation-owned sparse polynomial. Pad and `Eval_K` do not enter.

Outputs:
- two child-owned residual wires consumed by final identity.

Constraint groups:
- C1: three rank-one recipes for each of the 18 stored extension products;
- C2: two rank-one recipes for the result components.

Parent coverage:
- `Formal.opsAt`, child `piccs.v1_1.ccs_terminal`.

The structural sparse cost model proves no fresh lowering column and 56
physical rows. No proof evaluates emitted circuit data.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.CcsTerminal

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Layout.Polynomial.Horner
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable

variable {logicalWidth degreeBound : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

structure InputsLinear
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.Interface)
    (offset : Nat) : Prop where
  freshMatrix : ∀ matrix,
    KExprLinear (interface.freshMatrix offset matrix)

/-- The two child-owned CCS residual wires lie below the canonical norm child
start. -/
theorem output_varsBelow_norm
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (parentOffset : Nat) :
    (Formal.ccsOutput relation (Formal.atOffset interface parentOffset)
      (Formal.normOffset relation interface parentOffset)).VarsBelow
        (Formal.normOffset relation interface parentOffset) := by
  have normOffsetEq : Formal.normOffset relation interface parentOffset =
      Formal.ccsOffset interface parentOffset + 56 := by
    calc
      Formal.normOffset relation interface parentOffset =
          Formal.normStart (Formal.atOffset interface parentOffset) :=
        (Formal.normStart_atOffset relation interface parentOffset).symm
      _ = _ := by
        unfold Formal.normStart
          NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.privateCount
        rw [Formal.ccsStart_atOffset interface parentOffset]
  unfold Formal.ccsOutput
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.output
    NightstreamFPrime.Gadgets.Polynomial.Sparse.Owned.output
    KExpr.VarsBelow Expr.VarsBelow
  rw [Formal.ccsStart_atOffset interface parentOffset, normOffsetEq,
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.productCount_eq]
  omega

private theorem core_totalFreshCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat)
    (inputs : InputsLinear (Formal.ccsInterface relation interface) offset) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Formal.ccsCircuit relation interface).main offset)) = 0 := by
  unfold Formal.ccsCircuit
  rw [FormalCircuit.withConstantFootprint_main]
  exact NightstreamFPrime.Layout.Polynomial.Sparse.ownedCircuit_totalFreshCount
    _ _ offset (fun matrix => (inputs.freshMatrix matrix).kAffine)

private theorem core_totalRowCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat)
    (inputs : InputsLinear (Formal.ccsInterface relation interface) offset) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Formal.ccsCircuit relation interface).main offset)) = 56 := by
  unfold Formal.ccsCircuit
  rw [FormalCircuit.withConstantFootprint_main]
  have rows :=
    NightstreamFPrime.Layout.Polynomial.Sparse.ownedCircuit_totalRowCount
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.polynomial relation)
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.sparseInterface
        (Formal.ccsInterface relation interface))
      offset (fun matrix => (inputs.freshMatrix matrix).kAffine)
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.productCount_eq]
    at rows
  exact rows

def footprint
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.ccsInterface relation interface) offset) :
    R1CS.CircuitFootprint (Formal.ccsCircuit relation interface) where
  freshColumnCount := fun _ => 0
  physicalRowCount := fun _ => 56
  freshColumnCount_eq := by
    intro offset
    exact core_totalFreshCount relation interface offset (inputs offset)
  physicalRowCount_eq := by
    intro offset
    exact core_totalRowCount relation interface offset (inputs offset)

theorem freshColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.ccsInterface relation interface) offset)
    (offset : Nat) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Formal.ccsCircuit relation interface).main offset)) = 0 :=
  (footprint relation interface inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.ccsInterface relation interface) offset)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Formal.ccsCircuit relation interface).main offset)) = 56 :=
  (footprint relation interface inputs).physicalRowCount_eq offset

theorem physicalPrivateColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.ccsInterface relation interface) offset)
    (offset : Nat) :
    localLength (Circuit.ops (Formal.ccsCircuit relation interface).main
        offset) +
      R1CS.totalFreshCount (flatConstraints (Circuit.ops
        (Formal.ccsCircuit relation interface).main offset)) = 56 := by
  have logicalColumns :
      localLength (Circuit.ops (Formal.ccsCircuit relation interface).main
        offset) = 56 := by
    unfold Formal.ccsCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.localLength_eq
        relation (Formal.ccsInterface relation interface) offset
  rw [logicalColumns, freshColumnCount_eq relation interface inputs offset]

theorem output_linear
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    KExprLinear
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.output relation
        (Formal.ccsInterface relation interface) offset) := by
  refine ⟨rfl, rfl, ?_, ?_⟩
  · intro value equality
    cases equality
  · intro value equality
    cases equality

def logicalConstraints
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : List Expr :=
  flatConstraints (Circuit.ops
    (Formal.ccsCircuit relation interface).main offset)

def plan
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : R1CS.LoweringPlan where
  constraints := logicalConstraints relation interface offset
  firstFresh := offset +
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.privateCount

def physicalRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : List R1CS.Row :=
  (plan relation interface offset).rows

def PhysicalHolds
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (env : Env) : Prop :=
  R1CS.RowsHold env (physicalRows relation interface offset)

private theorem logicalConstraints_varsBelow
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (env : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.Assumptions relation
        (Formal.ccsInterface relation interface) offset env) :
    ∀ expression ∈ logicalConstraints relation interface offset,
      expression.VarsBelow (offset +
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.privateCount) := by
  have scope :=
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.flatConstraints_varsBelow
      relation (Formal.ccsInterface relation interface) offset assumptions
  unfold logicalConstraints Formal.ccsCircuit
  rw [FormalCircuit.withConstantFootprint_main]
  exact scope

theorem physical_implies_logicalConstraints
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (env : Env)
    (physical : PhysicalHolds relation interface offset env) :
    ConstraintsHold env (logicalConstraints relation interface offset) := by
  unfold PhysicalHolds physicalRows at physical
  exact R1CS.lowerConstraints_sound env
    (logicalConstraints relation interface offset)
    (offset +
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.privateCount)
    physical

theorem physical_implies_spec
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (env : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.Assumptions relation
        (Formal.ccsInterface relation interface) offset env)
    (physical : PhysicalHolds relation interface offset env) :
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.SpecHolds relation
      (Formal.ccsInterface relation interface) offset env := by
  apply NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.soundness relation
    (Formal.ccsInterface relation interface) env offset assumptions
  apply holdsFlat_implies_holds
  change ConstraintsHold env (logicalConstraints relation interface offset)
  exact physical_implies_logicalConstraints relation interface offset env
    physical

theorem physical_complete
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (env : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.Assumptions relation
        (Formal.ccsInterface relation interface) offset env)
    (specification :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.SpecHolds relation
        (Formal.ccsInterface relation interface) offset env) :
    ∃ completed,
      AgreesOutside env completed offset
          (NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.privateCount +
            R1CS.totalFreshCount
              (logicalConstraints relation interface offset)) ∧
        PhysicalHolds relation interface offset completed := by
  rcases NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.completeness
      relation (Formal.ccsInterface relation interface) env offset assumptions
      specification with ⟨logicalEnv, logicalAgrees, logicalRows⟩
  have logicalAgreesFixed : AgreesOutside env logicalEnv offset
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.privateCount := by
    rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.localLength_eq]
      at logicalAgrees
    exact logicalAgrees
  have scope := logicalConstraints_varsBelow relation interface offset
    logicalEnv assumptions
  have logicalHolds : ConstraintsHold logicalEnv
      (logicalConstraints relation interface offset) := by
    exact logicalRows
  rcases R1CS.lowerConstraints_complete logicalEnv
      (logicalConstraints relation interface offset)
      (offset +
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.privateCount)
      scope logicalHolds with
    ⟨completed, physicalAgrees, physicalRowsHold⟩
  refine ⟨completed, logicalAgreesFixed.append physicalAgrees, ?_⟩
  exact physicalRowsHold

end NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.CcsTerminal
