import Nightstream.Implementation.R1CS.Correspondence.Projection.ProjectionSound

/-!
Witness-completeness for one production PiRLC projection identity.

`ExecutionWitness` carries an interpreter source and its actual `Program.run`
output, together with the decoded sampled identity.  It never carries
definition-row satisfaction under a renamed predicate.  Those native facts
are sufficient to reconstruct the exact definition and assertion rows.
-/

namespace Nightstream.Implementation.R1CS.ProjectionProgram

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

private theorem fadd_assoc' (left middle right : F) :
    (left + middle) + right = left + (middle + right) := by
  apply Fin.ext
  simp only [Fin.val_add]
  rw [Nat.mod_add_mod, Nat.add_mod_mod, Nat.add_assoc]

private theorem fadd_comm' (left right : F) : left + right = right + left := by
  apply Fin.ext
  simp only [Fin.val_add, Nat.add_comm]

local instance : Std.Associative (fun (left right : F) => left + right) :=
  ⟨fadd_assoc'⟩
local instance : Std.Commutative (fun (left right : F) => left + right) :=
  ⟨fadd_comm'⟩

private theorem rawLcEval_append' (assignment : Nat → Nat)
    (left right : List (Nat × Nat)) :
    rawLcEval assignment (left ++ right) =
      rawLcEval assignment left + rawLcEval assignment right := by
  induction left with
  | nil => simp [rawLcEval]
  | cons head tail inductionHypothesis =>
      simp [rawLcEval, inductionHypothesis, Nat.add_assoc]

private theorem termsValue_append' (assignment : Nat → Nat)
    (left right : List (Nat × Nat)) :
    residue (lcEval assignment (left ++ right)) =
      residue (lcEval assignment left) + residue (lcEval assignment right) := by
  apply Fin.ext
  simp only [residue, Fin.val_add]
  rw [lcEval_eq_raw_mod, lcEval_eq_raw_mod, lcEval_eq_raw_mod,
    rawLcEval_append']
  simp [Nat.add_mod]

private theorem termsValue_columns' (assignment : Nat → Nat)
    (columns : List Nat) :
    residue (lcEval assignment (columns.map fun column => (column, 1))) =
      columns.foldr (fun column suffix =>
        baseAt assignment column + suffix) 0 := by
  induction columns with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      apply Fin.ext
      simp only [List.map_cons, List.foldr_cons, Fin.val_add, residue]
      rw [lcEval_eq_raw_mod]
      simp only [rawLcEval, Nat.one_mul, Nat.mod_mod]
      have valueHypothesis := congrArg Fin.val inductionHypothesis
      simp only [residue] at valueHypothesis
      rw [lcEval_eq_raw_mod] at valueHypothesis
      simp only [Nat.mod_mod] at valueHypothesis
      rw [← valueHypothesis]
      simp only [baseAt, residue]
      rw [← Nat.add_mod]

private theorem negOne_mul_add_self' (value : F) :
    residue (goldilocksP - 1) * value + value = 0 := by
  apply Fin.ext
  simp only [Fin.val_add, Fin.val_mul, residue]
  have raw : (goldilocksP - 1) * value.val + value.val =
      goldilocksP * value.val := by
    unfold goldilocksP
    omega
  have modular : ((goldilocksP - 1) * value.val + value.val) %
      goldilocksP = 0 := by
    rw [raw, Nat.mul_mod_right]
  simpa only [Nat.add_mod, Nat.mul_mod, Nat.mod_mod,
    Nat.mod_eq_of_lt value.isLt] using modular

private theorem projectionCheckLimb_complete
    (assignment : Nat → Nat) (constantOne : assignment 0 = 1)
    (outputs : List Nat) (quotientPhi output : Nat)
    (equality : outputs.foldr (fun column suffix =>
        baseAt assignment column + suffix) 0 =
      baseAt assignment quotientPhi + baseAt assignment output) :
    RowHolds assignment
      ⟨outputs.map (fun column => (column, 1)) ++
          [(quotientPhi, goldilocksP - 1),
           (output, goldilocksP - 1)],
       [(0, 1)], []⟩ := by
  let positive := outputs.map fun column => (column, 1)
  let negative :=
    [(quotientPhi, goldilocksP - 1), (output, goldilocksP - 1)]
  have positiveValue : residue (lcEval assignment positive) =
      outputs.foldr (fun column suffix =>
        baseAt assignment column + suffix) 0 := by
    simpa [positive] using termsValue_columns' assignment outputs
  have negativeValue : residue (lcEval assignment negative) =
      residue (goldilocksP - 1) * baseAt assignment quotientPhi +
        residue (goldilocksP - 1) * baseAt assignment output := by
    apply Fin.ext
    simp [negative, baseAt, residue, lcEval, Fin.val_add, Fin.val_mul]
  have fieldZero :
      outputs.foldr (fun column suffix =>
          baseAt assignment column + suffix) 0 +
        (residue (goldilocksP - 1) * baseAt assignment quotientPhi +
          residue (goldilocksP - 1) * baseAt assignment output) = 0 := by
    rw [equality]
    have leftCancel := negOne_mul_add_self'
      (baseAt assignment quotientPhi)
    have rightCancel := negOne_mul_add_self'
      (baseAt assignment output)
    calc
      (baseAt assignment quotientPhi + baseAt assignment output) +
          (residue (goldilocksP - 1) * baseAt assignment quotientPhi +
            residue (goldilocksP - 1) * baseAt assignment output) =
        (residue (goldilocksP - 1) * baseAt assignment quotientPhi +
            baseAt assignment quotientPhi) +
          (residue (goldilocksP - 1) * baseAt assignment output +
            baseAt assignment output) := by ac_rfl
      _ = 0 := by rw [leftCancel, rightCancel, Fin.zero_add]
  have split := termsValue_append' assignment positive negative
  rw [positiveValue, negativeValue, fieldZero] at split
  have rawZero := congrArg Fin.val split
  change lcEval assignment (positive ++ negative) % goldilocksP = 0 at rawZero
  simpa [RowHolds, positive, negative, lcEval, constantOne] using rawZero

private theorem foldKValues' (values : List K) :
    values.foldr K.add K.zero =
      ⟨(values.map K.c0).foldr (· + ·) 0,
       (values.map K.c1).foldr (· + ·) 0⟩ := by
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.foldr_cons, List.map_cons]
      rw [inductionHypothesis]
      rfl

/-- Verifier inputs read before the projection trace's deterministic SSA
execution. -/
def ProjectionTrace.inputColumns (trace : ProjectionTrace) : List Nat :=
  [0, trace.ladder.beta.c0, trace.ladder.beta.c1] ++
    trace.pairs.flatMap (fun pair => pair.rhoColumns ++ pair.inputColumns) ++
    trace.outputColumns ++ trace.quotientColumns

/-- Honest compiler execution witness.  The public premise is an actual source
state and the output of `Program.run`, never definition-row satisfaction under
another name. -/
structure ProjectionTrace.ExecutionWitness
    (trace : ProjectionTrace) (assignment : Nat → Nat) where
  source : Nat → Nat
  sourceCanonical : ∀ column, source column < goldilocksP
  sourceOne : source 0 = 1
  executed : run source trace.definitions = assignment
  sampledWireEquation :
    (trace.pairProductValues assignment).foldr K.add K.zero =
      K.add (trace.quotientPhiProduct.output.value assignment)
        (trace.outputEvaluation.output.value assignment)

def ProjectionTrace.nativeCheck (trace : ProjectionTrace)
    (assignment : Nat → Nat) : Bool :=
  (trace.definitions.all fun definition =>
    decide (Definition.Holds assignment definition)) &&
  decide
    ((trace.pairProductValues assignment).foldr K.add K.zero =
      K.add (trace.quotientPhiProduct.output.value assignment)
        (trace.outputEvaluation.output.value assignment))

theorem ProjectionTrace.nativeCheck_eq_true_iff
    (trace : ProjectionTrace) (assignment : Nat → Nat) :
    trace.nativeCheck assignment = true ↔
      DefinitionsHold assignment trace.definitions ∧
      (trace.pairProductValues assignment).foldr K.add K.zero =
        K.add (trace.quotientPhiProduct.output.value assignment)
          (trace.outputEvaluation.output.value assignment) := by
  simp [ProjectionTrace.nativeCheck, DefinitionsHold, List.all_eq_true,
    Bool.and_eq_true, decide_eq_true_eq]

theorem ProjectionTrace.checks_complete
    (trace : ProjectionTrace) (assignment : Nat → Nat)
    (constantOne : assignment 0 = 1)
    (sampledWireEquation :
      (trace.pairProductValues assignment).foldr K.add K.zero =
        K.add (trace.quotientPhiProduct.output.value assignment)
          (trace.outputEvaluation.output.value assignment)) :
    Satisfies trace.checks assignment := by
  let outputC0 := trace.pairs.map fun pair => pair.product.output.c0
  let outputC1 := trace.pairs.map fun pair => pair.product.output.c1
  have wire := sampledWireEquation
  rw [foldKValues'] at wire
  have low := congrArg K.c0 wire
  have high := congrArg K.c1 wire
  simp only [K.add] at low high
  have low' : outputC0.foldr (fun column suffix =>
        baseAt assignment column + suffix) 0 =
      baseAt assignment trace.quotientPhiProduct.output.c0 +
        baseAt assignment trace.outputEvaluation.output.c0 := by
    simpa [outputC0, ProjectionTrace.pairProductValues, KColumns.value,
      List.map_map, Function.comp_def, List.foldr_map] using low
  have high' : outputC1.foldr (fun column suffix =>
        baseAt assignment column + suffix) 0 =
      baseAt assignment trace.quotientPhiProduct.output.c1 +
        baseAt assignment trace.outputEvaluation.output.c1 := by
    simpa [outputC1, ProjectionTrace.pairProductValues, KColumns.value,
      List.map_map, Function.comp_def, List.foldr_map] using high
  intro row member
  simp [ProjectionTrace.checks] at member
  rcases member with rfl | rfl
  · simpa [outputC0] using
      projectionCheckLimb_complete assignment constantOne outputC0
        trace.quotientPhiProduct.output.c0
        trace.outputEvaluation.output.c0 low'
  · simpa [outputC1] using
      projectionCheckLimb_complete assignment constantOne outputC1
        trace.quotientPhiProduct.output.c1
        trace.outputEvaluation.output.c1 high'

/-- Native SSA execution plus the sampled wire equation reconstruct every
exact definition and assertion row for this projection identity. -/
theorem ProjectionTrace.native_complete
    (trace : ProjectionTrace) (assignment : Nat → Nat)
    (wellFormed : WellFormed trace.inputColumns trace.definitions)
    (definitionsCanonical : ∀ definition ∈ trace.definitions,
      definition.Canonical)
    (witness : trace.ExecutionWitness assignment) :
    Satisfies
      (trace.definitions.map Definition.builderRow ++ trace.checks)
      assignment := by
  have canonical : ∀ column, assignment column < goldilocksP := by
    rw [← witness.executed]
    exact run_canonical witness.sourceCanonical
  have preserves := run_preserves_known wellFormed witness.source
  have constantOne : assignment 0 = 1 := by
    rw [← witness.executed]
    exact (preserves 0 (by simp [ProjectionTrace.inputColumns])).trans
      witness.sourceOne
  have definitionsHold := run_definitions_hold wellFormed witness.source
  rw [witness.executed] at definitionsHold
  intro row member
  rw [List.mem_append] at member
  rcases member with definitionRow | checkRow
  · exact builderDefinitions_complete canonical constantOne
      definitionsCanonical definitionsHold row definitionRow
  · exact trace.checks_complete assignment constantOne
      witness.sampledWireEquation row checkRow

end Nightstream.Implementation.R1CS.ProjectionProgram
