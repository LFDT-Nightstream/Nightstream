import NightstreamFPrime.Export.Generic.Pieces
import NightstreamFPrime.Gadgets.Poseidon2.Permutation.Mark

/-! The rows of a marked permutation are the canonical template with its lanes and locals
(SPEC.md section 4.4). The canonical template is the compiler's permutation of the lanes
`var 0, …, var 15` from local 16 (`PilotData.canonicalConstraints`). The compiler builds every
recipe from the input lanes, constants and its own locals, so a substitution that moves the locals
from 16 to `start` and puts the mark's lanes in place of the input lanes gives the compiler's
permutation of those lanes from `start` (`compile_subst`). Hence the exporter's window of a mark
is its compiled rows (`window_eq_markWindow`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Generic.PermutationWindow

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Export.Generic.Pieces

/-- `wires` moves each local from `start` to `target`. -/
def Moves (wires : Nat → Expr) (start target : Nat) : Prop :=
  ∀ index, wires (start + index) = .var (target + index)

theorem Moves.shift {wires : Nat → Expr} {start target : Nat} (moves : Moves wires start target)
    (offset : Nat) : Moves wires (start + offset) (target + offset) := by
  intro index
  have := moves (offset + index)
  rwa [← Nat.add_assoc, ← Nat.add_assoc] at this

/-- The state with `wires` in place of each column. -/
def substState (wires : Nat → Expr) (state : Layer.EState) : Layer.EState :=
  fun lane => subst wires (state lane)

@[simp] theorem subst_add (wires : Nat → Expr) (left right : Expr) :
    subst wires (left + right) = subst wires left + subst wires right :=
  rfl

@[simp] theorem subst_mul (wires : Nat → Expr) (left right : Expr) :
    subst wires (left * right) = subst wires left * subst wires right :=
  rfl

@[simp] theorem subst_sub (wires : Nat → Expr) (left right : Expr) :
    subst wires (left - right) = subst wires left - subst wires right :=
  rfl

@[simp] theorem subst_const (wires : Nat → Expr) (value : Spec.F) :
    subst wires (.const value) = .const value :=
  rfl

@[simp] theorem subst_ofNat (wires : Nat → Expr) (value : Nat) :
    subst wires (OfNat.ofNat value : Expr) = OfNat.ofNat value :=
  rfl

@[simp] theorem subst_var (wires : Nat → Expr) (column : Nat) :
    subst wires (.var column) = wires column :=
  rfl

theorem subst_getE (wires : Nat → Expr) (state : Layer.EState) (index : Nat) :
    subst wires (Layer.getE state index) = Layer.getE (substState wires state) index := by
  unfold Layer.getE
  split <;> rfl

theorem subst_mat4E (wires : Nat → Expr) (state : Layer.EState) (base lane : Nat) :
    subst wires (Layer.mat4E state base lane) = Layer.mat4E (substState wires state) base lane := by
  unfold Layer.mat4E
  split <;> simp only [subst_add, subst_mul, subst_getE] <;> rfl

theorem subst_foldl_add (wires : Nat → Expr) :
    ∀ (values : List Expr) (initial : Expr),
      subst wires (values.foldl (· + ·) initial) =
        (values.map (subst wires)).foldl (· + ·) (subst wires initial)
  | [], _ => rfl
  | value :: values, initial => by
      simp only [List.foldl_cons, List.map_cons]
      rw [subst_foldl_add wires values (initial + value), subst_add]

theorem subst_blockE (wires : Nat → Expr) (state : Layer.EState) (index : Nat) :
    subst wires (Layer.blockE state index) = Layer.blockE (substState wires state) index :=
  subst_mat4E wires state _ _

theorem subst_columnE (wires : Nat → Expr) (state : Layer.EState) (index : Nat) :
    subst wires (Layer.columnE state index) = Layer.columnE (substState wires state) index := by
  unfold Layer.columnE
  rw [subst_foldl_add, List.map_map]
  simp only [Function.comp_def, subst_blockE]
  rfl

theorem substState_externalE (wires : Nat → Expr) (state : Layer.EState) :
    substState wires (Layer.externalE state) = Layer.externalE (substState wires state) := by
  funext lane
  simp only [substState, Layer.externalE, subst_add, subst_blockE, subst_columnE]

theorem subst_sumE (wires : Nat → Expr) (state : Layer.EState) :
    subst wires (Layer.sumE state) = Layer.sumE (substState wires state) := by
  unfold Layer.sumE
  rw [subst_foldl_add, List.map_map]
  simp only [Function.comp_def, subst_getE]
  rfl

theorem substState_internalE (wires : Nat → Expr) (state : Layer.EState) :
    substState wires (Layer.internalE state) = Layer.internalE (substState wires state) := by
  funext lane
  simp only [substState, Layer.internalE, subst_add, subst_mul, subst_const, subst_sumE]

theorem substState_freshState {wires : Nat → Expr} {start target : Nat}
    (moves : Moves wires start target) :
    substState wires (Permutation.freshState start) = Permutation.freshState target := by
  funext lane
  exact moves lane.val

theorem compileSboxes_subst (wires : Nat → Expr) :
    ∀ (values : List Expr) (start target : Nat), Moves wires start target →
      (Permutation.compileSboxes start values).recipes.map (subst wires) =
          (Permutation.compileSboxes target (values.map (subst wires))).recipes ∧
        (Permutation.compileSboxes start values).outputs.map (subst wires) =
          (Permutation.compileSboxes target (values.map (subst wires))).outputs
  | [], _, _, _ => ⟨rfl, rfl⟩
  | value :: values, start, target, moves => by
      obtain ⟨recipes, outputs⟩ := compileSboxes_subst wires values (start + 4) (target + 4)
        (moves.shift 4)
      have zero := moves 0
      have one := moves 1
      have two := moves 2
      have three := moves 3
      simp only [Nat.add_zero] at zero
      simp only [Permutation.compileSboxes, List.map_append, List.map_cons, recipes, outputs,
        Permutation.sboxRecipes, Permutation.sboxOutput, List.map_nil, subst_mul, subst_var,
        zero, one, two, three, and_self]

theorem fullInputs_subst (wires : Nat → Expr) (rows : List (List Nat)) (round : Nat)
    (state : Layer.EState) :
    (Permutation.fullInputs rows round state).map (subst wires) =
      Permutation.fullInputs rows round (substState wires state) := by
  simp only [Permutation.fullInputs, List.map_ofFn]
  rfl

theorem subst_getD (wires : Nat → Expr) (values : List Expr) (index : Nat) :
    subst wires (values.getD index 0) = (values.map (subst wires)).getD index 0 := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
  cases values[index]? <;> rfl

theorem substState_fullSboxState {wires : Nat → Expr} {start target : Nat}
    (moves : Moves wires start target) (rows : List (List Nat)) (round : Nat)
    (state : Layer.EState) :
    substState wires (Permutation.fullSboxState start rows round state) =
      Permutation.fullSboxState target rows round (substState wires state) := by
  funext lane
  simp only [substState, Permutation.fullSboxState, subst_getD]
  rw [(compileSboxes_subst wires _ start target moves).2, fullInputs_subst]

theorem subst_partialInput (wires : Nat → Expr) (round : Nat) (state : Layer.EState) :
    subst wires (Permutation.partialInput round state) =
      Permutation.partialInput round (substState wires state) :=
  rfl

theorem substState_partialSboxState {wires : Nat → Expr} {start target : Nat}
    (moves : Moves wires start target) (round : Nat) (state : Layer.EState) :
    substState wires (Permutation.partialSboxState start round state) =
      Permutation.partialSboxState target round (substState wires state) := by
  funext lane
  by_cases first : lane.val = 0
  · simp only [substState, Permutation.partialSboxState, first, ↓reduceIte]
    rw [subst_getD, (compileSboxes_subst wires _ start target moves).2, List.map_cons,
      List.map_nil, subst_partialInput]
  · simp only [substState, Permutation.partialSboxState, first, ↓reduceIte]

theorem ofFn_externalE (wires : Nat → Expr) (state : Layer.EState) :
    (List.ofFn (Layer.externalE state)).map (subst wires) =
      List.ofFn (Layer.externalE (substState wires state)) := by
  rw [List.map_ofFn, ← substState_externalE]
  rfl

theorem ofFn_internalE (wires : Nat → Expr) (state : Layer.EState) :
    (List.ofFn (Layer.internalE state)).map (subst wires) =
      List.ofFn (Layer.internalE (substState wires state)) := by
  rw [List.map_ofFn, ← substState_internalE]
  rfl

theorem stepRecipes_subst {wires : Nat → Expr} {start target : Nat}
    (moves : Moves wires start target) (step : Permutation.Step) (state : Layer.EState) :
    (Permutation.stepRecipes start step state).map (subst wires) =
      Permutation.stepRecipes target step (substState wires state) := by
  cases step with
  | initialLayer =>
      simp only [Permutation.stepRecipes, ofFn_externalE]
  | initialFullRound round =>
      simp only [Permutation.stepRecipes, List.map_append, ofFn_externalE]
      rw [(compileSboxes_subst wires _ start target moves).1, fullInputs_subst,
        substState_fullSboxState moves]
  | terminalFullRound round =>
      simp only [Permutation.stepRecipes, List.map_append, ofFn_externalE]
      rw [(compileSboxes_subst wires _ start target moves).1, fullInputs_subst,
        substState_fullSboxState moves]
  | partialRound round =>
      simp only [Permutation.stepRecipes, List.map_append, ofFn_internalE]
      rw [(compileSboxes_subst wires _ start target moves).1, List.map_cons, List.map_nil,
        subst_partialInput, substState_partialSboxState moves]

theorem substState_stepOutput {wires : Nat → Expr} {start target : Nat}
    (moves : Moves wires start target) (step : Permutation.Step) :
    substState wires (Permutation.stepOutput start step) = Permutation.stepOutput target step := by
  cases step with
  | initialLayer => exact substState_freshState moves
  | initialFullRound => exact substState_freshState (moves.shift 64)
  | terminalFullRound => exact substState_freshState (moves.shift 64)
  | partialRound => exact substState_freshState (moves.shift 4)

/-- The compiler commutes with a substitution that moves its locals. -/
theorem compile_subst (wires : Nat → Expr) :
    ∀ (steps : List Permutation.Step) (start target : Nat) (state : Layer.EState),
      Moves wires start target →
        (Permutation.compile start state steps).recipes.map (subst wires) =
          (Permutation.compile target (substState wires state) steps).recipes
  | [], _, _, _, _ => rfl
  | step :: steps, start, target, state, moves => by
      simp only [Permutation.compile, List.map_append]
      rw [stepRecipes_subst moves, compile_subst wires steps _ _ _ (moves.shift _),
        substState_stepOutput moves]

theorem recipeConstraints_subst (wires : Nat → Expr) :
    ∀ (recipes : List Expr) (start target : Nat), Moves wires start target →
      (recipeConstraints start recipes).map (subst wires) =
        recipeConstraints target (recipes.map (subst wires))
  | [], _, _, _ => rfl
  | recipe :: recipes, start, target, moves => by
      have head := moves 0
      simp only [Nat.add_zero] at head
      simp only [recipeConstraints, List.map_cons, subst_sub, subst_var, head]
      rw [recipeConstraints_subst wires recipes (start + 1) (target + 1) (moves.shift 1)]

theorem laneSubst_moves (start : Nat) (lanes : List Expr) :
    Moves (laneSubst 16 start lanes) 16 start :=
  fun index => laneSubst_local 16 start index lanes

theorem substState_canonical (start : Nat) (lanes : List Expr) :
    substState (laneSubst 16 start lanes) PilotData.canonicalState =
      Permutation.laneState lanes := by
  funext lane
  simp only [substState, PilotData.canonicalState, subst_var, Permutation.laneState]
  exact laneSubst_input lanes lane.isLt

/-- The exporter's window of a mark is the compiler's rows of the permutation of its lanes. -/
theorem window_eq_markWindow (mark : PermutationMark) :
    window (PilotData.canonicalConstraints ()) mark = Permutation.markWindow mark := by
  unfold window PilotData.canonicalConstraints PilotData.canonicalRecipes Permutation.markWindow
  rw [recipeConstraints_subst _ _ 16 mark.start (laneSubst_moves mark.start mark.lanes),
    compile_subst _ _ 16 mark.start _ (laneSubst_moves mark.start mark.lanes),
    substState_canonical]

end NightstreamFPrime.Export.Generic.PermutationWindow
