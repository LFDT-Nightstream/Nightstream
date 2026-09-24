import NightstreamFPrime.Export.Stage1.Wide.PlanSupport
import NightstreamFPrime.Layout.MatrixProgram.Program

namespace NightstreamFPrime.Export.Stage1.Wide.FormSupport

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation MatrixProgram MatrixProgram.AffineGrid

def WireSupported {columns : Nat} (predicate : Fin columns → Prop) (wire : RetainedBlock) : Prop :=
  ∀ column, wire.start ≤ column.val → column.val < wire.start + wire.coordinateCount → predicate column

def TermSupported {columns : Nat} (predicate : Fin columns → Prop) : Term → Prop
  | .retained wire _ _ _ _ _ => WireSupported predicate wire
  | .constant _ => True

def AffineSupported {columns : Nat} (predicate : Fin columns → Prop) (program : AffineGrid.Program) : Prop :=
  ∀ rule ∈ program.rules, TermSupported predicate rule.term

theorem wire_form {columns : Nat} {predicate : Fin columns → Prop} (wire : RetainedBlock)
    (supported : WireSupported predicate wire) (slot : Nat) (form : SparseForm columns)
    (loaded : wire.form? columns slot = some form) : Supported predicate form := by
  unfold RetainedBlock.form? at loaded
  split at loaded
  · split at loaded
    · cases Option.some.inj loaded
      exact block _ _ _ _ supported
    · contradiction
  · contradiction

theorem affine_term {columns : Nat} {predicate : Fin columns → Prop} (term : Term)
    (oneColumn : Nat) (one : ∀ column : Fin columns, column.val = oneColumn → predicate column)
    (coordinate : Coordinate) (supported : TermSupported predicate term)
    (form : SparseForm columns) (loaded : term.form? columns oneColumn coordinate = some form) :
    Supported predicate form := by
  cases term with
  | retained wire slot major middle minor coefficient =>
    simp only [Term.form?] at loaded
    obtain ⟨value, valueLoaded, rest⟩ := Option.bind_eq_some_iff.mp loaded
    split at rest
    · cases Option.some.inj rest
      unfold applyCoefficient
      split
      · exact wire_form wire supported _ _ valueLoaded
      · exact scale _ (wire_form wire supported _ _ valueLoaded)
    · contradiction
  | constant coefficient =>
    simp only [Term.form?] at loaded
    split at loaded
    · split at loaded
      · cases Option.some.inj loaded
        exact FormSupport.singleton _ _ (one _ rfl)
      · contradiction
    · contradiction

theorem affine_rule {columns : Nat} {predicate : Fin columns → Prop} (rule : Rule)
    (oneColumn : Nat) (one : ∀ column : Fin columns, column.val = oneColumn → predicate column)
    (coordinate : Coordinate) (supported : TermSupported predicate rule.term)
    (form : SparseForm columns) (loaded : rule.form? columns oneColumn coordinate = some (some form)) :
    Supported predicate form := by
  unfold Rule.form? at loaded
  split at loaded
  · simp at loaded
  · obtain ⟨value, valueLoaded, same⟩ := Option.bind_eq_some_iff.mp loaded
    have equal : value = form := Option.some.inj (Option.some.inj same)
    exact equal ▸ affine_term _ oneColumn one _ supported _ valueLoaded

theorem affine_program {columns : Nat} {predicate : Fin columns → Prop} (program : AffineGrid.Program)
    (oneColumn : Nat) (one : ∀ column : Fin columns, column.val = oneColumn → predicate column)
    (coordinate : Coordinate) (supported : AffineSupported predicate program)
    (form : SparseForm columns) (loaded : program.form? columns oneColumn coordinate = some form) :
    Supported predicate form := by
  apply AffineGrid.Program.form?_property program oneColumn coordinate (Supported predicate)
    (empty _) (fun _ _ => add) _ form loaded
  intro rule member value equal
  exact affine_rule rule oneColumn one coordinate (supported rule member) value equal

theorem multiplication_grid {columns : Nat} {predicate : Fin columns → Prop} (grid : MultiplicationGrid.Block)
    (one : ∀ column : Fin columns, column.val = grid.oneColumn → predicate column)
    (left : AffineSupported predicate grid.left) (right : AffineSupported predicate grid.right)
    (output : AffineSupported predicate grid.output)
    (ordinal : Nat) (forms : OrdinaryRow.Forms columns)
    (loaded : grid.row? columns ordinal = some forms) : OrdinarySupported predicate forms := by
  unfold MultiplicationGrid.Block.row? at loaded
  split at loaded
  · obtain ⟨coordinate, _, rest⟩ := Option.bind_eq_some_iff.mp loaded
    obtain ⟨a, aLoaded, rest⟩ := Option.bind_eq_some_iff.mp rest
    obtain ⟨b, bLoaded, rest⟩ := Option.bind_eq_some_iff.mp rest
    obtain ⟨c, cLoaded, same⟩ := Option.bind_eq_some_iff.mp rest
    cases Option.some.inj same
    exact ⟨FormSupport.singleton _ _ (one _ rfl),
      affine_program _ _ one coordinate left _ aLoaded,
      affine_program _ _ one coordinate right _ bLoaded,
      affine_program _ _ one coordinate output _ cLoaded⟩
  · contradiction

theorem matrix_program {columns : Nat} {predicate : Fin columns → Prop}
    (program : MatrixProgram.Program) (sourceRow : Nat → Option R1CS.Row)
    (supported : ∀ block ∈ program.blocks, ∀ ordinal forms,
      block.row? columns sourceRow ordinal = some forms → ∀ port, Supported predicate (forms port))
    (ordinal : Nat) (forms : RowForms columns)
    (loaded : program.row? columns sourceRow ordinal = some forms) :
    ∀ port, Supported predicate (forms port) := by
  rcases program with ⟨blocks⟩
  induction blocks generalizing ordinal with
  | nil => contradiction
  | cons block rest ih =>
    by_cases first : ordinal < block.rowCount
    · rw [MatrixProgram.Program.cons_first_row? _ _ _ _ _ first] at loaded
      exact supported _ List.mem_cons_self ordinal forms loaded
    · rw [MatrixProgram.Program.cons_rest_row? _ _ _ _ _ (Nat.le_of_not_gt first)] at loaded
      exact ih _ (fun next member => supported next (List.mem_cons_of_mem _ member)) loaded

end NightstreamFPrime.Export.Stage1.Wide.FormSupport
