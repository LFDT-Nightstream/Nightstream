import NightstreamFPrime.Export.Stage1.Wide.FormSupport

/-! Structural read support for the compact row constructors. These proofs
quantify over a row index; they do not enumerate a plan's rows. -/

namespace NightstreamFPrime.Export.Stage1.Wide.FormSupport

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation

theorem mono {columns : Nat} {p q : Fin columns → Prop} {form : SparseForm columns}
    (supported : Supported p form) (implies : ∀ column, p column → q column) : Supported q form :=
  fun entry member => implies entry.column (supported entry member)

def PlanSupported {columns : Nat} (predicate : Fin columns → Prop) (plan : ProductionRelation.Plan columns) : Prop :=
  ∀ row port, Supported predicate (plan.forms row port)

theorem append {columns : Nat} {predicate : Fin columns → Prop}
    (left right : ProductionRelation.Plan columns) (fits)
    (hl : PlanSupported predicate left) (hr : PlanSupported predicate right) :
    PlanSupported predicate (Plan.append left right fits) := by
  intro row port
  dsimp only [Plan.append]
  cases Plan.splitIndex left.rowCount right.rowCount row with
  | inl index => exact hl index port
  | inr index => exact hr index port

theorem indexed {columns count rows : Nat} {predicate : Fin columns → Prop}
    (forms : Fin count → Fin rows → Fin ProductionRelation.meaningfulPortCount → SparseForm columns)
    (fits) (supported : ∀ block row port, Supported predicate (forms block row port)) :
    PlanSupported predicate (Plan.indexed forms fits) := by
  intro row port
  exact supported (Fin.decodeProd row).1 (Fin.decodeProd row).2 port

def OrdinarySupported {columns : Nat} (predicate : Fin columns → Prop) (forms : OrdinaryRow.Forms columns) : Prop :=
  Supported predicate forms.selector ∧ Supported predicate forms.a ∧
    Supported predicate forms.b ∧ Supported predicate forms.c

theorem ordinary_form {columns : Nat} {predicate : Fin columns → Prop}
    (forms : OrdinaryRow.Forms columns) (supported : OrdinarySupported predicate forms)
    (port : Fin ProductionRelation.meaningfulPortCount) : Supported predicate (forms.meaningfulForm port) := by
  unfold OrdinaryRow.Forms.meaningfulForm
  split
  · exact supported.1
  · exact supported.2.1
  · exact supported.2.2.1
  · exact supported.2.2.2
  · exact empty _

theorem ordinary_plan {columns rows : Nat} {predicate : Fin columns → Prop}
    (forms : Fin rows → OrdinaryRow.Forms columns) (fits)
    (supported : ∀ row, OrdinarySupported predicate (forms row)) :
    PlanSupported predicate (OrdinaryRow.planOfForms fits forms) :=
  fun row port => ordinary_form (forms row) (supported row) port

theorem source_terms {sourceWidth columns : Nat} {predicate : Fin columns → Prop}
    (source : SourceCompiler.SourceMap sourceWidth columns)
    (supported : ∀ column, Supported predicate (source.form column))
    (terms : List (Nat × F)) (bounded) :
    Supported predicate (SourceCompiler.compileTerms source terms bounded) := by
  induction terms with
  | nil => exact empty _
  | cons head tail ih => exact add (scale _ (supported _)) (ih _)

theorem source_combination {sourceWidth columns : Nat} {predicate : Fin columns → Prop}
    (source : SourceCompiler.SourceMap sourceWidth columns) (oneColumn : Fin columns)
    (supported : ∀ column, Supported predicate (source.form column)) (one : predicate oneColumn)
    (combination : R1CS.LinearCombination) (bounded) :
    Supported predicate (SourceCompiler.compileCombination source oneColumn combination bounded) :=
  add (singleton oneColumn _ one) (source_terms source supported _ _)

theorem source_row {sourceWidth columns : Nat} {predicate : Fin columns → Prop}
    (source : SourceCompiler.SourceMap sourceWidth columns) (oneColumn : Fin columns)
    (supported : ∀ column, Supported predicate (source.form column)) (one : predicate oneColumn)
    (row : R1CS.Row) (bounded) :
    OrdinarySupported predicate (SourceCompiler.compileRow source oneColumn row bounded) :=
  ⟨singleton oneColumn 1 one, source_combination source oneColumn supported one _ _,
    source_combination source oneColumn supported one _ _,
    source_combination source oneColumn supported one _ _⟩

theorem source_plan {sourceWidth columns : Nat} {predicate : Fin columns → Prop}
    (program : OrdinarySourcePlan.Program sourceWidth) (inputs : program.Inputs columns)
    (one : predicate inputs.oneColumn)
    (supported : ∀ row column, Supported predicate ((inputs.sourceMap row).form column)) :
    PlanSupported predicate (program.compile inputs).toPlan := by
  intro row port
  exact ordinary_form _ (source_row (inputs.sourceMap row) inputs.oneColumn (supported row) one
    (program.row row) (program.bounded row)) port

end NightstreamFPrime.Export.Stage1.Wide.FormSupport
