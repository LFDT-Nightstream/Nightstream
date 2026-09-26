import NightstreamFPrime.Export.Stage1.Wide.PlanSupport

/-! Read support follows the Poseidon compiler's state and row constructors.
The proof is structural in the round program, not in the invocation count. -/

namespace NightstreamFPrime.Export.Stage1.Wide.FormSupport

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation

theorem internal {columns : Nat} {predicate : Fin columns → Prop} (state : SparseLayer.State columns)
    (supported : ∀ lane, Supported predicate (state lane)) (lane : Fin 8) :
    Supported predicate (SparseLayer.internal state lane) := by
  apply add (scale _ (supported lane))
  unfold SparseLayer.sum SparseLayer.add
  repeat' first | apply add | exact get state supported _

def SboxSupported {columns : Nat} (predicate : Fin columns → Prop) (forms : SboxRow.Forms columns) : Prop :=
  Supported predicate forms.selector ∧ Supported predicate forms.input ∧ Supported predicate forms.output

theorem sbox_form {columns : Nat} {predicate : Fin columns → Prop} (forms : SboxRow.Forms columns)
    (supported : SboxSupported predicate forms) (port : Fin ProductionRelation.meaningfulPortCount) :
    Supported predicate (forms.meaningfulForm port) := by
  unfold SboxRow.Forms.meaningfulForm
  split
  · exact supported.1
  · exact supported.2.2
  · exact supported.2.1
  · exact empty _

theorem pin_form {columns : Nat} {predicate : Fin columns → Prop} (forms : PinRow.Forms columns)
    (selector : Supported predicate forms.selector) (value : Supported predicate forms.value)
    (port : Fin ProductionRelation.meaningfulPortCount) : Supported predicate (forms.meaningfulForm port) := by
  unfold PinRow.Forms.meaningfulForm
  split
  · exact selector
  · exact value
  · exact empty _

theorem pin_plan {columns rows : Nat} {predicate : Fin columns → Prop}
    (interface : PinFamilyPlan.Interface columns rows) (fits)
    (one : predicate interface.oneColumn) (values : ∀ row, Supported predicate (interface.value row)) :
    PlanSupported predicate (PinFamilyPlan.plan interface fits) :=
  fun row port => pin_form _ (singleton interface.oneColumn 1 one) (values row) port

private theorem outputAt {columns : Nat} {predicate : Fin columns → Prop}
    (interface : PoseidonSboxPlan.Interface columns)
    (outputs : ∀ index, Supported predicate (interface.sboxOutput index)) (index : Nat) :
    Supported predicate (PoseidonSboxPlan.sboxOutputAt interface index) := by
  unfold PoseidonSboxPlan.sboxOutputAt
  split
  · exact outputs _
  · exact empty _

private theorem fullRows {columns : Nat} {predicate : Fin columns → Prop}
    (interface : PoseidonSboxPlan.Interface columns) (one : predicate interface.oneColumn)
    (outputs : ∀ index, Supported predicate (interface.sboxOutput index))
    (constants : List (List Nat)) (round next : Nat) (state : SparseLayer.State columns)
    (supported : ∀ lane, Supported predicate (state lane)) :
    ∀ forms ∈ PoseidonSboxPlan.fullRows interface constants round next state, SboxSupported predicate forms := by
  intro forms member
  obtain ⟨lane, rfl⟩ := List.mem_ofFn.mp member
  exact ⟨singleton interface.oneColumn 1 one,
    add (supported lane) (singleton interface.oneColumn _ one), outputAt interface outputs _⟩

private theorem partialRows {columns : Nat} {predicate : Fin columns → Prop}
    (interface : PoseidonSboxPlan.Interface columns) (one : predicate interface.oneColumn)
    (outputs : ∀ index, Supported predicate (interface.sboxOutput index))
    (round next : Nat) (state : SparseLayer.State columns)
    (supported : ∀ lane, Supported predicate (state lane)) :
    ∀ forms ∈ PoseidonSboxPlan.partialRows interface round next state, SboxSupported predicate forms := by
  intro forms member
  obtain rfl := List.mem_singleton.mp member
  exact ⟨singleton interface.oneColumn 1 one,
    add (supported 0) (singleton interface.oneColumn _ one), outputAt interface outputs _⟩

private theorem step {columns : Nat} {predicate : Fin columns → Prop}
    (interface : PoseidonSboxPlan.Interface columns) (one : predicate interface.oneColumn)
    (outputs : ∀ index, Supported predicate (interface.sboxOutput index))
    (next : Nat) (state : SparseLayer.State columns) (supported : ∀ lane, Supported predicate (state lane))
    (action : Gadgets.Poseidon2.Permutation.Step) :
    (∀ lane, Supported predicate ((PoseidonSboxPlan.compileStep interface next state action).state lane)) ∧
    (∀ forms ∈ (PoseidonSboxPlan.compileStep interface next state action).rows, SboxSupported predicate forms) := by
  cases action with
  | initialLayer => exact ⟨external state supported, by simp [PoseidonSboxPlan.compileStep]⟩
  | initialFullRound round =>
      exact ⟨external _ (fun _ => outputAt interface outputs _), fullRows interface one outputs _ _ _ _ supported⟩
  | terminalFullRound round =>
      exact ⟨external _ (fun _ => outputAt interface outputs _), fullRows interface one outputs _ _ _ _ supported⟩
  | partialRound round =>
      refine ⟨internal _ ?_, partialRows interface one outputs _ _ _ supported⟩
      intro lane
      unfold PoseidonSboxPlan.partialState
      split
      · exact outputAt interface outputs _
      · exact supported lane

theorem poseidon_compile {columns : Nat} {predicate : Fin columns → Prop}
    (interface : PoseidonSboxPlan.Interface columns) (one : predicate interface.oneColumn)
    (outputs : ∀ index, Supported predicate (interface.sboxOutput index))
    (actions : List Gadgets.Poseidon2.Permutation.Step) (next : Nat) (state : SparseLayer.State columns)
    (supported : ∀ lane, Supported predicate (state lane)) :
    (∀ lane, Supported predicate ((PoseidonSboxPlan.compile interface next state actions).state lane)) ∧
    (∀ forms ∈ (PoseidonSboxPlan.compile interface next state actions).rows, SboxSupported predicate forms) := by
  induction actions generalizing next state with
  | nil => exact ⟨supported, by simp [PoseidonSboxPlan.compile]⟩
  | cons action rest ih =>
      have head := step interface one outputs next state supported action
      have tail := ih (PoseidonSboxPlan.compileStep interface next state action).nextSbox _ head.1
      refine ⟨tail.1, ?_⟩
      intro forms member
      rcases List.mem_append.mp member with h | h
      · exact head.2 forms h
      · exact tail.2 forms h

theorem poseidon_family {columns count : Nat} {predicate : Fin columns → Prop}
    (interface : PoseidonSboxFamilyPlan.Interface columns count) (fits)
    (one : predicate interface.oneColumn)
    (inputs : ∀ invocation lane, Supported predicate (interface.input invocation lane))
    (outputs : ∀ invocation slot, Supported predicate (interface.sboxOutput invocation slot)) :
    PlanSupported predicate (PoseidonSboxFamilyPlan.plan interface fits) := by
  apply indexed
  intro invocation row port
  have supported := (poseidon_compile (PoseidonSboxFamilyPlan.invocationInterface interface invocation)
    one (outputs invocation) Gadgets.Poseidon2.Permutation.schedule 0 _ (inputs invocation)).2
  have member := List.get_mem
    (PoseidonRetainedRows.rows (PoseidonSboxFamilyPlan.invocationInterface interface invocation))
    ⟨row.val, by rw [PoseidonRetainedRows.rows_length]; exact row.isLt⟩
  obtain ⟨forms, member, same⟩ := List.mem_map.mp member
  change Supported predicate ((PoseidonSboxFamilyPlan.rowAt interface invocation row).meaningfulForm port)
  change PoseidonSboxPlan.Row.sbox forms = PoseidonSboxFamilyPlan.rowAt interface invocation row at same
  rw [← same]
  exact sbox_form forms (supported forms member) port

end NightstreamFPrime.Export.Stage1.Wide.FormSupport
