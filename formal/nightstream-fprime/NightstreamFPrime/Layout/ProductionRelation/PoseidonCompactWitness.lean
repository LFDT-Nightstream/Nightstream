import NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxSourceCompleteness
import NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxFamilyPlan

/-! Construct retained S-box values from the proved canonical permutation executor. -/

namespace NightstreamFPrime.Layout.ProductionRelation.PoseidonCompactWitness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def inputEnv (initial : Fin 8 → F) : Env :=
  fun index => if bound : index < 8 then initial ⟨index, bound⟩ else 0

def recipes : List Expr :=
  (Permutation.compile PoseidonScheduleTrace.inputCount
    PoseidonScheduleTrace.canonicalState Permutation.schedule).recipes

def source (initial : Fin 8 → F) : Env :=
  executeRecipes (inputEnv initial) PoseidonScheduleTrace.inputCount recipes

theorem source_input (initial : Fin 8 → F) (lane : Fin 8) :
    source initial lane.val = initial lane := by
  have same := executeRecipes_agrees_below (inputEnv initial)
    PoseidonScheduleTrace.inputCount recipes lane.val lane.isLt
  exact same.trans (by simp [inputEnv, lane.isLt])

theorem source_rows (initial : Fin 8 → F) :
    ConstraintsHold (source initial)
      (recipeConstraints PoseidonScheduleTrace.inputCount recipes) := by
  apply executeRecipes_holds_recipeConstraints
  exact Permutation.compile_schedule_causal _ _ (fun lane => lane.isLt)

def retained (initial : Fin 8 → F) (row : Fin PoseidonRetainedSlots.rows.length) : F :=
  source initial (PoseidonRetainedSlots.rows.get row).step.output.val

def output (initial : Fin 8 → F) : Fin 8 → F :=
  Layer.evalState (source initial) (Permutation.scheduleOutput PoseidonScheduleTrace.inputCount)

theorem output_eq_permute (initial : Fin 8 → F) :
    List.ofFn (output initial) = Poseidon2.permute (List.ofFn initial) := by
  have result := Permutation.compile_schedule_sound (source initial)
    PoseidonScheduleTrace.inputCount PoseidonScheduleTrace.canonicalState (source_rows initial)
  rw [← Permutation.scheduleOutput_eq_compile] at result
  have input : Layer.evalState (source initial) PoseidonScheduleTrace.canonicalState = initial := by
    funext lane
    dsimp only [Layer.evalState, PoseidonScheduleTrace.canonicalState, Expr.eval]
    exact source_input initial lane
  rw [input] at result
  exact result

/-- Retaining the executor's 86 selected values satisfies the compact equations. -/
theorem equations {columns : Nat} (interface : PoseidonSboxPlan.Interface columns)
    (assignment : Assignment F columns) (initial : Fin 8 → F)
    (one : assignment interface.oneColumn = 1)
    (inputs : SparseLayer.evalState assignment interface.input = initial)
    (values : ∀ row, (interface.sboxOutput row).eval assignment = retained initial row) :
    PoseidonSboxPlan.SboxEquations interface assignment := by
  apply PoseidonSboxSourceCompleteness.equations_of_sourceRows interface assignment
    (source initial) one _ values (source_rows initial)
  rw [inputs]
  funext lane
  dsimp only [Layer.evalState, PoseidonScheduleTrace.canonicalState, Expr.eval]
  exact (source_input initial lane).symm

/-- For an output derived from the final S-box values, source encoding proves
both acceptance and equality with the constructive permutation output. -/
theorem family_member {columns count : Nat}
    (interface : PoseidonSboxFamilyPlan.Interface columns count)
    (invocation : Fin count) (assignment : Assignment F columns)
    (initial : Fin 8 → F) (one : assignment interface.oneColumn = 1)
    (inputs : SparseLayer.evalState assignment (interface.input invocation) = initial)
    (values : ∀ row, (interface.sboxOutput invocation row).eval assignment = retained initial row) :
    PoseidonSboxPlan.RowsZero (PoseidonSboxFamilyPlan.invocationInterface interface invocation) assignment ∧
      SparseLayer.evalState assignment (interface.output invocation) = output initial := by
  have sboxes := equations (PoseidonSboxFamilyPlan.invocationInterface interface invocation)
    assignment initial one inputs values
  have outputs : PoseidonSboxPlan.OutputEquations
      (PoseidonSboxFamilyPlan.invocationInterface interface invocation) assignment := by
    intro lane
    rw [← PoseidonSboxFamilyPlan.invocation_output_eq_trace]
  have rows := PoseidonSboxPlan.rowsZero_of_equations _ assignment one sboxes outputs
  refine ⟨rows, ?_⟩
  have result := PoseidonSboxPlan.rowsZero_implies_permute _ assignment one rows
  rw [show SparseLayer.evalState assignment
    (PoseidonSboxFamilyPlan.invocationInterface interface invocation).input = initial from inputs,
    ← output_eq_permute initial] at result
  exact List.ofFn_inj.mp result

end NightstreamFPrime.Layout.ProductionRelation.PoseidonCompactWitness
