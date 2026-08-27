import NightstreamFPrime.Layout.PiCCS.v1_1.Lowering
import NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.StatementBinding
import NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.StatementAbsorption
import NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.ChallengeDerivation
import NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.RoundTranscript
import NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.InitialClaim
import NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.SumcheckChain
import NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.EvalKTerminal
import NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.EvalATerminal
import NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.CcsTerminal
import NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.NormTerminal
import NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.FinalIdentity
import NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.OutputBinding

/-!
Paper authority: SuperNeo v1_1, section 7.3, PiCCS Steps 1--5.
Obligation: Assemble the twelve physical leaf owners in the exact order of
the sole logical PiCCS circuit.

This module owns only the parent constraint and footprint decomposition. It
does not unfold a child operation or replace the canonical `Formal` circuit.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_1

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth degreeBound : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Layout-only shape evidence for the fixed interfaces of all nonlinear or
transcript PiCCS children. These fields do not supply protocol values or
challenges. -/
structure InputShapes
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat) : Prop where
  statementBinding : ∀ childOffset,
    Leaves.StatementBinding.InputsAffine
      (Formal.atOffset interface parentOffset) childOffset
  statementAbsorption : ∀ childOffset,
    Leaves.StatementAbsorption.InputsAffine
      (Formal.statementAbsorptionInterface
        (Formal.atOffset interface parentOffset)) childOffset
  challengeDerivation : ∀ childOffset,
    Leaves.ChallengeDerivation.InputsAffine
      (Formal.challengeInterface (Formal.atOffset interface parentOffset)
        parentOffset) childOffset
  roundTranscript : ∀ childOffset,
    Leaves.RoundTranscript.InputsAffine
      (Formal.roundTranscriptInterface
        (Formal.atOffset interface parentOffset)) childOffset
  initialClaim : ∀ childOffset,
    Leaves.InitialClaim.InputsLinear
      (Formal.initialClaimInterface
        (Formal.atOffset interface parentOffset)) childOffset
  sumcheck : ∀ childOffset,
    Leaves.SumcheckChain.InputsLinear
      (Formal.sumcheckInterface
        (Formal.atOffset interface parentOffset)) childOffset
  eval_K : ∀ childOffset,
    Leaves.EvalKTerminal.InputsLinear
      (Formal.evalKInterface
        (Formal.atOffset interface parentOffset)) childOffset
  eval_A : ∀ childOffset,
    Leaves.EvalATerminal.InputsLinear
      (Formal.evalAInterface
        (Formal.atOffset interface parentOffset)) childOffset
  ccs : ∀ childOffset,
    Leaves.CcsTerminal.InputsLinear
      (Formal.ccsInterface relation
        (Formal.atOffset interface parentOffset)) childOffset
  norm : ∀ childOffset,
    Leaves.NormTerminal.InputsLinear
      (Formal.normInterface relation
        (Formal.atOffset interface parentOffset)) childOffset
  finalIdentity : ∀ childOffset,
    Leaves.FinalIdentity.InputsLinear
      (Formal.finalIdentityInterface relation
        (Formal.atOffset interface parentOffset)) childOffset
  outputBinding : ∀ childOffset,
    Leaves.OutputBinding.InputsAffine
      (Formal.outputBindingInterface
        (Formal.atOffset interface parentOffset)) childOffset

/-- Relation-owned fresh-column cost of the two final identity assertions. -/
def terminalFreshCost
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat) : Nat :=
  Leaves.FinalIdentity.terminalFreshColumnCount
    (Formal.finalIdentityInterface relation
      (Formal.atOffset interface parentOffset))
    (Formal.finalIdentityOffset relation interface parentOffset)

/-- Relation-owned physical-row cost of the two final identity assertions. -/
def terminalRowCost
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat) : Nat :=
  Leaves.FinalIdentity.terminalPhysicalRowCount
    (Formal.finalIdentityInterface relation
      (Formal.atOffset interface parentOffset))
    (Formal.finalIdentityOffset relation interface parentOffset)

/-- The final identity receives exactly the output shapes proved by the five
preceding arithmetic children. No child operations are unfolded here. -/
private theorem terminalInputShapes
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat)
    (inputs : InputShapes relation interface parentOffset) :
    Leaves.FinalIdentity.TerminalInputShapes
      (Formal.finalIdentityInterface relation
        (Formal.atOffset interface parentOffset))
      (Formal.finalIdentityOffset relation interface parentOffset) := by
  let frozen := Formal.atOffset interface parentOffset
  let finalOffset := Formal.finalIdentityOffset relation interface parentOffset
  have terminalShape := Leaves.SumcheckChain.output_mulCounts frozen finalOffset
    (inputs.sumcheck (Formal.sumcheckStart frozen))
  have evalKShape := Leaves.EvalKTerminal.output_shape frozen finalOffset
    (inputs.eval_K (Formal.evalKStart frozen))
  have evalAShape := Leaves.EvalATerminal.output_shape frozen finalOffset
    (inputs.eval_A (Formal.evalAStart frozen))
  have ccsShape := Leaves.CcsTerminal.output_linear relation frozen
    (Formal.ccsStart frozen)
  have normShape := Leaves.NormTerminal.output_shape relation frozen finalOffset
    (inputs.norm (Formal.normStart frozen))
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa [frozen, finalOffset, Formal.finalIdentityInterface] using
      terminalShape.1
  · simpa [frozen, finalOffset, Formal.finalIdentityInterface] using
      terminalShape.2
  · simpa [frozen, finalOffset, Formal.finalIdentityInterface] using
      evalKShape.c0_mulCount
  · simpa [frozen, finalOffset, Formal.finalIdentityInterface] using
      evalKShape.c1_mulCount
  · simpa [frozen, finalOffset, Formal.finalIdentityInterface] using
      evalKShape.c0_nonAffine
  · simpa [frozen, finalOffset, Formal.finalIdentityInterface] using
      evalKShape.c1_nonAffine
  · simpa [frozen, finalOffset, Formal.finalIdentityInterface] using
      evalAShape.c0_mulCount
  · simpa [frozen, finalOffset, Formal.finalIdentityInterface] using
      evalAShape.c1_mulCount
  · simpa [frozen, finalOffset, Formal.finalIdentityInterface,
      Formal.ccsOutput] using ccsShape
  · simpa [frozen, finalOffset, Formal.finalIdentityInterface] using
      normShape.c0_mulCount
  · simpa [frozen, finalOffset, Formal.finalIdentityInterface] using
      normShape.c1_mulCount

theorem terminalFreshCost_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat)
    (inputs : InputShapes relation interface parentOffset) :
    terminalFreshCost relation interface parentOffset = 5324 := by
  unfold terminalFreshCost
  exact Leaves.FinalIdentity.terminalFreshColumnCount_eq _ _
    (inputs.finalIdentity
      (Formal.finalIdentityOffset relation interface parentOffset))
    (terminalInputShapes relation interface parentOffset inputs)

theorem terminalRowCost_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat)
    (inputs : InputShapes relation interface parentOffset) :
    terminalRowCost relation interface parentOffset = 5326 := by
  unfold terminalRowCost
  exact Leaves.FinalIdentity.terminalPhysicalRowCount_eq _ _
    (inputs.finalIdentity
      (Formal.finalIdentityOffset relation interface parentOffset))
    (terminalInputShapes relation interface parentOffset inputs)

def childConstraints (child : FormalCircuit) (offset : Nat) : List Expr :=
  flatConstraints (Circuit.ops child.main offset)

private def appendAll : List (List Expr) → List Expr
  | [] => []
  | [constraints] => constraints
  | constraints :: next :: rest =>
      constraints ++ appendAll (next :: rest)

/-- The twelve opaque child constraint lists in canonical phase order. -/
def childConstraintLists
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : List (List Expr) :=
  [childConstraints
    (Formal.statementBindingCircuit (Formal.atOffset interface offset)) offset,
   childConstraints
    (Formal.statementAbsorptionCircuit (Formal.atOffset interface offset))
    (Formal.statementAbsorptionOffset interface offset),
   childConstraints (Formal.challengeCircuit interface offset)
    (Formal.challengeOffset interface offset),
   childConstraints
    (Formal.roundTranscriptCircuit (Formal.atOffset interface offset))
    (Formal.roundTranscriptOffset interface offset),
   childConstraints
    (Formal.initialClaimCircuit (Formal.atOffset interface offset))
    (Formal.initialClaimOffset interface offset),
   childConstraints (Formal.sumcheckCircuit (Formal.atOffset interface offset))
    (Formal.sumcheckOffset interface offset),
   childConstraints (Formal.evalKCircuit (Formal.atOffset interface offset))
    (Formal.evalKOffset interface offset),
   childConstraints (Formal.evalACircuit (Formal.atOffset interface offset))
    (Formal.evalAOffset interface offset),
   childConstraints
    (Formal.ccsCircuit relation (Formal.atOffset interface offset))
    (Formal.ccsOffset interface offset),
   childConstraints
    (Formal.normCircuit relation (Formal.atOffset interface offset))
    (Formal.normOffset relation interface offset),
   childConstraints
    (Formal.finalIdentityCircuit relation (Formal.atOffset interface offset))
    (Formal.finalIdentityOffset relation interface offset),
   childConstraints
    (Formal.outputBindingCircuit (Formal.atOffset interface offset))
    (Formal.outputBindingOffset relation interface offset)]

/-- Exact twelve-child constraint order of `Formal.opsAt`. -/
def orderedConstraints
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : List Expr :=
  appendAll (childConstraintLists relation interface offset)

theorem logicalConstraints_eq_ordered
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    logicalConstraints relation interface offset =
      orderedConstraints relation interface offset := by
  unfold logicalConstraints
  rw [Formal.main_ops]
  unfold Formal.opsAt orderedConstraints childConstraintLists
  simp only [appendAll, childConstraints, flatConstraints, List.flatMap_cons,
    List.flatMap_nil, Formal.childOp, Op.flatConstraints,
    FormalCircuit.asSubcircuit_constraints, List.append_nil]

/-- Fresh-column delta for each child in canonical phase order. -/
def physicalFreshDeltas
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : List Nat :=
  (childConstraintLists relation interface offset).map R1CS.totalFreshCount

/-- Physical-row delta for each child in canonical phase order. -/
def physicalRowDeltas
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : List Nat :=
  (childConstraintLists relation interface offset).map R1CS.totalRowCount

/-- Logical private-column delta of each child, read from the canonical
parent operation list rather than restated by the layout. -/
def logicalPrivateDeltas
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : List Nat :=
  (Formal.opsAt relation interface offset).map Op.localLength

/-- Each physical column delta is the child's logical private interval plus
the fresh columns introduced by lowering that child's constraints. -/
def physicalColumnDeltas
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : List Nat :=
  List.zipWith (· + ·)
    (logicalPrivateDeltas relation interface offset)
    (physicalFreshDeltas relation interface offset)

/-- Running totals after each child, excluding the initial zero prefix. -/
def cumulativeFrom : Nat → List Nat → List Nat
  | _, [] => []
  | total, delta :: rest =>
      let next := total + delta
      next :: cumulativeFrom next rest

def cumulativePhysicalRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : List Nat :=
  cumulativeFrom 0 (physicalRowDeltas relation interface offset)

def cumulativePhysicalColumns
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : List Nat :=
  cumulativeFrom 0 (physicalColumnDeltas relation interface offset)

def cumulativeJointDomains
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : List Nat :=
  List.zipWith max
    (cumulativePhysicalRows relation interface offset)
    (cumulativePhysicalColumns relation interface offset)

private theorem totalFreshCount_appendAll (lists : List (List Expr)) :
    R1CS.totalFreshCount (appendAll lists) =
      (lists.map R1CS.totalFreshCount).sum := by
  induction lists with
  | nil => rfl
  | cons first rest ih =>
      cases rest with
      | nil =>
          simp only [appendAll, List.map_cons, List.map_nil, List.sum_cons,
            List.sum_nil, Nat.add_zero]
      | cons second tail =>
          simp only [appendAll, R1CS.totalFreshCount_append, List.map_cons,
            List.sum_cons, ih]

private theorem totalRowCount_appendAll (lists : List (List Expr)) :
    R1CS.totalRowCount (appendAll lists) =
      (lists.map R1CS.totalRowCount).sum := by
  induction lists with
  | nil => rfl
  | cons first rest ih =>
      cases rest with
      | nil =>
          simp only [appendAll, List.map_cons, List.map_nil, List.sum_cons,
            List.sum_nil, Nat.add_zero]
      | cons second tail =>
          simp only [appendAll, R1CS.totalRowCount_append, List.map_cons,
            List.sum_cons, ih]

theorem totalFreshCount_eq_deltas
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    R1CS.totalFreshCount (logicalConstraints relation interface offset) =
      (physicalFreshDeltas relation interface offset).sum := by
  rw [logicalConstraints_eq_ordered]
  change R1CS.totalFreshCount
      (appendAll (childConstraintLists relation interface offset)) =
    ((childConstraintLists relation interface offset).map
      R1CS.totalFreshCount).sum
  exact totalFreshCount_appendAll _

theorem totalRowCount_eq_deltas
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) :
    R1CS.totalRowCount (logicalConstraints relation interface offset) =
      (physicalRowDeltas relation interface offset).sum := by
  rw [logicalConstraints_eq_ordered]
  change R1CS.totalRowCount
      (appendAll (childConstraintLists relation interface offset)) =
    ((childConstraintLists relation interface offset).map
      R1CS.totalRowCount).sum
  exact totalRowCount_appendAll _

/-- Exact fresh-column delta of every child at the fixed production degree. -/
theorem physicalFreshDeltas_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat)
    (inputs : InputShapes relation interface parentOffset) :
    physicalFreshDeltas relation interface parentOffset =
      [0, 0, 0, 0, 90713, 378560, 6634, 85258, 20792, 720,
        97347 + terminalFreshCost relation interface parentOffset, 0] := by
  unfold physicalFreshDeltas childConstraintLists childConstraints
  simp only [List.map_cons, List.map_nil]
  rw [Leaves.StatementBinding.freshColumnCount_eq
      (Formal.atOffset interface parentOffset) inputs.statementBinding
      parentOffset,
    Leaves.StatementAbsorption.freshColumnCount_eq
      (Formal.atOffset interface parentOffset) inputs.statementAbsorption
      (Formal.statementAbsorptionOffset interface parentOffset),
    Leaves.ChallengeDerivation.freshColumnCount_eq interface parentOffset
      inputs.challengeDerivation
      (Formal.challengeOffset interface parentOffset),
    Leaves.RoundTranscript.freshColumnCount_eq
      (Formal.atOffset interface parentOffset) inputs.roundTranscript
      (Formal.roundTranscriptOffset interface parentOffset),
    Leaves.InitialClaim.freshColumnCount_eq
      (Formal.atOffset interface parentOffset) inputs.initialClaim
      (Formal.initialClaimOffset interface parentOffset),
    Leaves.SumcheckChain.freshColumnCount_eq
      (Formal.atOffset interface parentOffset) inputs.sumcheck
      (Formal.sumcheckOffset interface parentOffset),
    Leaves.EvalKTerminal.freshColumnCount_eq
      (Formal.atOffset interface parentOffset) inputs.eval_K
      (Formal.evalKOffset interface parentOffset),
    Leaves.EvalATerminal.freshColumnCount_eq
      (Formal.atOffset interface parentOffset) inputs.eval_A
      (Formal.evalAOffset interface parentOffset),
    Leaves.CcsTerminal.freshColumnCount_eq relation
      (Formal.atOffset interface parentOffset)
      inputs.ccs
      (Formal.ccsOffset interface parentOffset),
    Leaves.NormTerminal.freshColumnCount_eq relation
      (Formal.atOffset interface parentOffset) inputs.norm
      (Formal.normOffset relation interface parentOffset),
    Leaves.FinalIdentity.freshColumnCount_eq relation
      (Formal.atOffset interface parentOffset) inputs.finalIdentity
      (Formal.finalIdentityOffset relation interface parentOffset),
    Leaves.OutputBinding.freshColumnCount_eq
      (Formal.atOffset interface parentOffset) inputs.outputBinding
      (Formal.outputBindingOffset relation interface parentOffset)]
  rfl

/-- Exact physical-row delta of every child at the fixed production degree. -/
theorem physicalRowDeltas_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat)
    (inputs : InputShapes relation interface parentOffset) :
    physicalRowDeltas relation interface parentOffset =
      [160, 192400, 46176, 133200, 116631, 378610, 8458, 109546,
        20794, 752, 125093 + terminalRowCost relation interface parentOffset,
        4076512] := by
  unfold physicalRowDeltas childConstraintLists childConstraints
  simp only [List.map_cons, List.map_nil]
  rw [Leaves.StatementBinding.physicalRowCount_eq
      (Formal.atOffset interface parentOffset) inputs.statementBinding
      parentOffset,
    Leaves.StatementAbsorption.physicalRowCount_eq
      (Formal.atOffset interface parentOffset) inputs.statementAbsorption
      (Formal.statementAbsorptionOffset interface parentOffset),
    Leaves.ChallengeDerivation.physicalRowCount_eq interface parentOffset
      inputs.challengeDerivation
      (Formal.challengeOffset interface parentOffset),
    Leaves.RoundTranscript.physicalRowCount_eq_of_degreeBound_eq_nine
      (Formal.atOffset interface parentOffset) inputs.roundTranscript
      (Formal.roundTranscriptOffset interface parentOffset) rfl,
    Leaves.InitialClaim.physicalRowCount_eq
      (Formal.atOffset interface parentOffset) inputs.initialClaim
      (Formal.initialClaimOffset interface parentOffset),
    Leaves.SumcheckChain.physicalRowCount_eq
      (Formal.atOffset interface parentOffset) inputs.sumcheck
      (Formal.sumcheckOffset interface parentOffset),
    Leaves.EvalKTerminal.physicalRowCount_eq
      (Formal.atOffset interface parentOffset) inputs.eval_K
      (Formal.evalKOffset interface parentOffset),
    Leaves.EvalATerminal.physicalRowCount_eq
      (Formal.atOffset interface parentOffset) inputs.eval_A
      (Formal.evalAOffset interface parentOffset),
    Leaves.CcsTerminal.physicalRowCount_eq relation
      (Formal.atOffset interface parentOffset)
      inputs.ccs
      (Formal.ccsOffset interface parentOffset),
    Leaves.NormTerminal.physicalRowCount_eq relation
      (Formal.atOffset interface parentOffset) inputs.norm
      (Formal.normOffset relation interface parentOffset),
    Leaves.FinalIdentity.physicalRowCount_eq relation
      (Formal.atOffset interface parentOffset) inputs.finalIdentity
      (Formal.finalIdentityOffset relation interface parentOffset),
    Leaves.OutputBinding.physicalRowCount_eq
      (Formal.atOffset interface parentOffset) inputs.outputBinding
      (Formal.outputBindingOffset relation interface parentOffset)]
  rfl

/-- Exact logical private interval of every child at the production degree. -/
theorem logicalPrivateDeltas_eq_production
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat) :
    logicalPrivateDeltas relation interface parentOffset =
      [0, 192400, 46176, 133200, 25918, 0, 1824, 24288, 2, 32,
        27746, 4076512] := by
  unfold logicalPrivateDeltas Formal.opsAt
  simp only [List.map_cons, List.map_nil, Formal.childOp_privateCount]
  unfold Formal.statementBindingCircuit Formal.statementAbsorptionCircuit
    Formal.challengeCircuit Formal.roundTranscriptCircuit
    Formal.initialClaimCircuit Formal.sumcheckCircuit Formal.evalKCircuit
    Formal.evalACircuit Formal.ccsCircuit Formal.normCircuit
    Formal.finalIdentityCircuit Formal.outputBindingCircuit
  simp only [FormalCircuit.withConstantFootprint_privateCount]
  norm_num [RoundTranscript.perRoundRecipeCount, CcsTerminal.privateCount]

/-- Exact row deltas after discharging the relation-owned terminal cost. -/
theorem physicalRowDeltas_eq_production
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat)
    (inputs : InputShapes relation interface parentOffset) :
    physicalRowDeltas relation interface parentOffset =
      [160, 192400, 46176, 133200, 116631, 378610, 8458, 109546,
        20794, 752, 130419, 4076512] := by
  rw [physicalRowDeltas_eq relation interface parentOffset inputs,
    terminalRowCost_eq relation interface parentOffset inputs]

/-- Exact physical-column deltas after each child's lowering. -/
theorem physicalColumnDeltas_eq_production
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat)
    (inputs : InputShapes relation interface parentOffset) :
    physicalColumnDeltas relation interface parentOffset =
      [0, 192400, 46176, 133200, 116631, 378560, 8458, 109546,
        20794, 752, 130417, 4076512] := by
  unfold physicalColumnDeltas
  rw [logicalPrivateDeltas_eq_production,
    physicalFreshDeltas_eq relation interface parentOffset inputs,
    terminalFreshCost_eq relation interface parentOffset inputs]
  rfl

/-- Per-leaf cumulative rows, columns, and joint domains. This is the
canonical twelve-leaf footprint ledger required by the owner goal. -/
theorem cumulativeFootprints_eq_production
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat)
    (inputs : InputShapes relation interface parentOffset) :
    cumulativePhysicalRows relation interface parentOffset =
        [160, 192560, 238736, 371936, 488567, 867177, 875635, 985181,
          1005975, 1006727, 1137146, 5213658] ∧
      cumulativePhysicalColumns relation interface parentOffset =
        [0, 192400, 238576, 371776, 488407, 866967, 875425, 984971,
          1005765, 1006517, 1136934, 5213446] ∧
      cumulativeJointDomains relation interface parentOffset =
        [160, 192560, 238736, 371936, 488567, 867177, 875635, 985181,
          1005975, 1006727, 1137146, 5213658] := by
  rw [cumulativePhysicalRows, physicalRowDeltas_eq_production
      relation interface parentOffset inputs,
    cumulativePhysicalColumns, physicalColumnDeltas_eq_production
      relation interface parentOffset inputs]
  norm_num [cumulativeFrom, cumulativeJointDomains, cumulativePhysicalRows,
    cumulativePhysicalColumns, physicalRowDeltas_eq_production relation
      interface parentOffset inputs,
    physicalColumnDeltas_eq_production relation interface parentOffset inputs]

private theorem freshDeltaSum_eq (terminal : Nat) :
    [0, 0, 0, 0, 90713, 378560, 6634, 85258, 20792, 720,
      97347 + terminal, 0].sum = 680024 + terminal := by
  simp only [List.sum_cons, List.sum_nil, Nat.add_zero]
  omega

private theorem rowFixedCost_eq :
    160 + 192400 + 46176 + 133200 + 116631 + 378610 + 8458 + 109546 +
      20794 + 752 + 125093 + 4076512 = 5208332 := by
  norm_num

private theorem rowDeltaSum_reassociate
    (a b c d e f g h i j k l terminal : Nat) :
    [a, b, c, d, e, f, g, h, i, j, k + terminal, l].sum =
      (a + b + c + d + e + f + g + h + i + j + k + l) + terminal := by
  simp only [List.sum_cons, List.sum_nil, Nat.add_zero]
  ac_rfl

private theorem rowDeltaSum_eq (terminal : Nat) :
    [160, 192400, 46176, 133200, 116631, 378610, 8458, 109546,
      20794, 752, 125093 + terminal, 4076512].sum =
        5208332 + terminal := by
  calc
    _ = (160 + 192400 + 46176 + 133200 + 116631 + 378610 + 8458 + 109546 +
          20794 + 752 + 125093 + 4076512) + terminal :=
      rowDeltaSum_reassociate _ _ _ _ _ _ _ _ _ _ _ _ _
    _ = _ := by rw [rowFixedCost_eq]

theorem totalFreshCount_eq_fixed
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat)
    (inputs : InputShapes relation interface parentOffset) :
    R1CS.totalFreshCount
        (logicalConstraints relation interface parentOffset) =
      680024 + terminalFreshCost relation interface parentOffset := by
  calc
    _ = (physicalFreshDeltas relation interface parentOffset).sum :=
      totalFreshCount_eq_deltas relation interface parentOffset
    _ = [0, 0, 0, 0, 90713, 378560, 6634, 85258, 20792, 720,
          97347 + terminalFreshCost relation interface parentOffset, 0].sum :=
      congrArg List.sum
        (physicalFreshDeltas_eq relation interface parentOffset inputs)
    _ = _ := freshDeltaSum_eq _

theorem totalRowCount_eq_fixed
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat)
    (inputs : InputShapes relation interface parentOffset) :
    R1CS.totalRowCount (logicalConstraints relation interface parentOffset) =
      5208332 + terminalRowCost relation interface parentOffset := by
  calc
    _ = (physicalRowDeltas relation interface parentOffset).sum :=
      totalRowCount_eq_deltas relation interface parentOffset
    _ = [160, 192400, 46176, 133200, 116631, 378610, 8458, 109546,
          20794, 752,
          125093 + terminalRowCost relation interface parentOffset,
          4076512].sum :=
      congrArg List.sum
        (physicalRowDeltas_eq relation interface parentOffset inputs)
    _ = _ := rowDeltaSum_eq _

theorem physicalFreshColumnCount_eq_fixed
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat)
    (inputs : InputShapes relation interface parentOffset) :
    physicalFreshColumnCount relation interface parentOffset =
      680024 + terminalFreshCost relation interface parentOffset := by
  exact totalFreshCount_eq_fixed relation interface parentOffset inputs

theorem physicalRowCount_eq_fixed
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat)
    (inputs : InputShapes relation interface parentOffset) :
    physicalRowCount relation interface parentOffset =
      5208332 + terminalRowCost relation interface parentOffset := by
  rw [physicalRowCount_eq]
  exact totalRowCount_eq_fixed relation interface parentOffset inputs

theorem physicalColumnCount_eq_fixed
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat)
    (inputs : InputShapes relation interface parentOffset) :
    physicalColumnCount relation interface parentOffset =
      parentOffset + 5208122 +
        terminalFreshCost relation interface parentOffset := by
  rw [physicalColumnCount_eq,
    logicalColumnCount_eq_of_degreeBound_eq_nine relation interface
      parentOffset rfl,
    physicalFreshColumnCount_eq_fixed relation interface parentOffset inputs]
  omega

/-- Local PiCCS row/column domain when the phase owns the zero-based layout. -/
def jointDomain
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits) : Nat :=
  max (physicalRowCount relation interface 0)
    (physicalColumnCount relation interface 0)

theorem jointDomain_eq_fixed
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (inputs : InputShapes relation interface 0) :
    jointDomain relation interface =
      max (5208332 + terminalRowCost relation interface 0)
        (5208122 + terminalFreshCost relation interface 0) := by
  unfold jointDomain
  rw [physicalRowCount_eq_fixed relation interface 0 inputs,
    physicalColumnCount_eq_fixed relation interface 0 inputs]

theorem physicalFreshColumnCount_eq_production
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat)
    (inputs : InputShapes relation interface parentOffset) :
    physicalFreshColumnCount relation interface parentOffset = 685348 := by
  rw [physicalFreshColumnCount_eq_fixed relation interface parentOffset inputs,
    terminalFreshCost_eq relation interface parentOffset inputs]

theorem physicalRowCount_eq_production
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat)
    (inputs : InputShapes relation interface parentOffset) :
    physicalRowCount relation interface parentOffset = 5213658 := by
  rw [physicalRowCount_eq_fixed relation interface parentOffset inputs,
    terminalRowCost_eq relation interface parentOffset inputs]

theorem physicalColumnCount_eq_production
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat)
    (inputs : InputShapes relation interface parentOffset) :
    physicalColumnCount relation interface parentOffset =
      parentOffset + 5213446 := by
  rw [physicalColumnCount_eq_fixed relation interface parentOffset inputs,
    terminalFreshCost_eq relation interface parentOffset inputs]

theorem jointDomain_eq_production
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (inputs : InputShapes relation interface 0) :
    jointDomain relation interface = 5213658 := by
  unfold jointDomain
  rw [physicalRowCount_eq_production relation interface 0 inputs,
    physicalColumnCount_eq_production relation interface 0 inputs]
  norm_num

theorem jointDomain_le_twoPow25
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (inputs : InputShapes relation interface 0) :
    jointDomain relation interface ≤ 2 ^ 25 := by
  rw [jointDomain_eq_production relation interface inputs]
  norm_num

end NightstreamFPrime.Layout.PiCCS.v1_1
