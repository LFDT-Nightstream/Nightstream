import NightstreamFPrime.Lifecycle.PiCCS.v1_2.StateBinding
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.StatementAbsorption

/-!
Owns the expression form of the hashed running section of the state preimage:
commitments, `Eval_K`, `Eval_A`, the point, then the packed parent public
input, in `XOut.serializeRunning` order. Each packed word recomposes the
sixteen child public inputs with the shared Π_DEC radix weights.

The running transition consumes this encoding. This module adds no row and
owns no layout.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_2.RunningWords

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_2.StatementAbsorption

/-- Parent coordinate `column` of the sixteen child public-input expressions,
with the production radix weights. -/
def parentPublicExpr {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : RunningExpr logicalWidth publicFits)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) : Expr :=
  PiDEC.v1_2.SignedSplitScalar.recomposeDigits fun child =>
    running.publicInput
      (Fin.cast NightstreamFPrime.Lifecycle.runningCount_eq_radixChildCount.symm child)
      column

/-- The packed parent public input of the sixteen children. -/
def serializeParentPublicExpr {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : RunningExpr logicalWidth publicFits) : List Expr :=
  (List.finRange NightstreamFPrime.Lifecycle.packedParentWords).map fun word =>
    StateBinding.packWordExpr
      (parentPublicExpr running (NightstreamFPrime.Lifecycle.packedColumn word 0))
      (parentPublicExpr running (NightstreamFPrime.Lifecycle.packedColumn word 1))
      (parentPublicExpr running (NightstreamFPrime.Lifecycle.packedColumn word 2))

/-- Canonical expression form of the hashed running section in
`XOut.serializeRunning` order: commitments, `Eval_K`, `Eval_A`, the point,
then the packed parent public input. -/
def serializeRunningExpr {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : RunningExpr logicalWidth publicFits) : List Expr :=
  ((List.finRange productionShape.runningCount).flatMap fun index =>
      serializeCommitmentExpr (running.commitment index)) ++
    ((List.finRange productionShape.runningCount).flatMap fun index =>
      serializeEvalKExpr (running.evaluation index)) ++
    ((List.finRange productionShape.runningCount).flatMap fun index =>
      serializeEvalAExpr (running.evaluation index)) ++
    serializePointExpr running.point ++
    serializeParentPublicExpr running

theorem serializeRunningExpr_length {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : RunningExpr logicalWidth publicFits) :
    (serializeRunningExpr running).length = 27794 := by
  simp [serializeRunningExpr, serializeParentPublicExpr, serializePointExpr_length,
    serializeCommitmentExpr_length, serializeEvalKExpr_length,
    serializeEvalAExpr_length, productionShape, productionProfile,
    Phi81MatrixSource.phi81Shape, NightstreamFPrime.Lifecycle.packedParentWords]

/-- Every hashed running word is one commitment coefficient, one `Eval_K` or
`Eval_A` limb, one point limb, or one packed parent word. -/
theorem serializeRunningExpr_mem {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {running : RunningExpr logicalWidth publicFits} {expression : Expr}
    (member : expression ∈ serializeRunningExpr running) :
    (∃ source row coefficient, expression = running.commitment source row coefficient) ∨
      (∃ source coefficient,
        expression ∈ serializeKExpr ((running.evaluation source).eval_K coefficient)) ∨
      (∃ source matrix coefficient,
        expression ∈ serializeKExpr ((running.evaluation source).eval_A matrix coefficient)) ∨
      (∃ coordinate, expression ∈ serializeKExpr (running.point coordinate)) ∨
      (∃ word, expression = StateBinding.packWordExpr
        (parentPublicExpr running (NightstreamFPrime.Lifecycle.packedColumn word 0))
        (parentPublicExpr running (NightstreamFPrime.Lifecycle.packedColumn word 1))
        (parentPublicExpr running (NightstreamFPrime.Lifecycle.packedColumn word 2))) := by
  simp only [serializeRunningExpr, List.mem_append, List.mem_flatMap] at member
  rcases member with (((⟨source, _, commitment⟩ | ⟨source, _, evalK⟩) |
      ⟨source, _, evalA⟩) | point) | parent
  · simp only [serializeCommitmentExpr, List.mem_flatMap, List.mem_map] at commitment
    rcases commitment with ⟨row, _, coefficient, _, rfl⟩
    exact Or.inl ⟨source, row, coefficient, rfl⟩
  · simp only [serializeEvalKExpr, List.mem_flatMap] at evalK
    rcases evalK with ⟨coefficient, _, member⟩
    exact Or.inr (Or.inl ⟨source, coefficient, member⟩)
  · simp only [serializeEvalAExpr, List.mem_flatMap] at evalA
    rcases evalA with ⟨matrix, _, coefficient, _, member⟩
    exact Or.inr (Or.inr (Or.inl ⟨source, matrix, coefficient, member⟩))
  · simp only [serializePointExpr, List.mem_flatMap] at point
    rcases point with ⟨coordinate, _, member⟩
    exact Or.inr (Or.inr (Or.inr (Or.inl ⟨coordinate, member⟩)))
  · simp only [serializeParentPublicExpr, List.mem_map] at parent
    rcases parent with ⟨word, _, rfl⟩
    exact Or.inr (Or.inr (Or.inr (Or.inr ⟨word, rfl⟩)))

/-- A property closed under constants, sums and constant scaling holds on
every packed parent word whose child public inputs satisfy it. -/
theorem packWordExpr_parent_closed {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (Holds : Expr → Prop)
    (constant : ∀ value, Holds (Expr.const value))
    (add : ∀ left right, Holds left → Holds right → Holds (left + right))
    (scale : ∀ weight value, Holds value → Holds (Expr.const weight * value))
    (running : RunningExpr logicalWidth publicFits)
    (publicInput : ∀ source column, Holds (running.publicInput source column))
    (word : Fin NightstreamFPrime.Lifecycle.packedParentWords) :
    Holds (StateBinding.packWordExpr
      (parentPublicExpr running (NightstreamFPrime.Lifecycle.packedColumn word 0))
      (parentPublicExpr running (NightstreamFPrime.Lifecycle.packedColumn word 1))
      (parentPublicExpr running (NightstreamFPrime.Lifecycle.packedColumn word 2))) := by
  exact StateBinding.packWordExpr_closed Holds add scale
    (PiDEC.v1_2.SignedSplitScalar.recomposeDigits_closed Holds constant add scale _ fun _ => publicInput _ _)
    (PiDEC.v1_2.SignedSplitScalar.recomposeDigits_closed Holds constant add scale _ fun _ => publicInput _ _)
    (PiDEC.v1_2.SignedSplitScalar.recomposeDigits_closed Holds constant add scale _ fun _ => publicInput _ _)

theorem parentPublicExpr_eval {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : RunningExpr logicalWidth publicFits) (env : Env)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    (parentPublicExpr running column).eval env =
      NightstreamFPrime.Lifecycle.parentPublic (evalRunning running env) column := by
  unfold parentPublicExpr NightstreamFPrime.Lifecycle.parentPublic
  rw [PiDEC.v1_2.SignedSplitScalar.recomposeDigits_eval]
  rfl

theorem serializeRunningExpr_eval {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (running : RunningExpr logicalWidth publicFits) (env : Env) :
    Hash.evalList env (serializeRunningExpr running) =
      NightstreamFPrime.Lifecycle.serializeRunning
        (publicFits := publicFits) (evalRunning running env) := by
  unfold Hash.evalList serializeRunningExpr
    NightstreamFPrime.Lifecycle.serializeRunning
    NightstreamFPrime.Lifecycle.serializeRunningFields
    NightstreamFPrime.Lifecycle.serializeCommitments
    NightstreamFPrime.Lifecycle.serializeEvalKs
    NightstreamFPrime.Lifecycle.serializeEvalAs
  have pointEq : (serializePointExpr running.point).map (Expr.eval env) =
      NightstreamFPrime.Lifecycle.serializePoint (evalPoint running.point env) :=
    serializePointExpr_eval running.point env
  have parentEq : (serializeParentPublicExpr running).map (Expr.eval env) =
      NightstreamFPrime.Lifecycle.serializeParentPublic (evalRunning running env) := by
    unfold serializeParentPublicExpr NightstreamFPrime.Lifecycle.serializeParentPublic
    rw [List.map_map]
    apply List.map_congr_left
    intro word _
    simp only [Function.comp_apply, StateBinding.packWordExpr_eval, parentPublicExpr_eval]
  simp only [List.map_append]
  rw [map_flatMap_congr _ _ _ _ fun index => serializeCommitmentExpr_eval _ env,
    map_flatMap_congr _ _ _ _ fun index => serializeEvalKExpr_eval _ env,
    map_flatMap_congr _ _ _ _ fun index => serializeEvalAExpr_eval _ env,
    pointEq, parentEq]
  rfl

end NightstreamFPrime.Lifecycle.PiCCS.v1_2.RunningWords
