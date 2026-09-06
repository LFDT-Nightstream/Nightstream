import NightstreamFPrime.Export.Stage1.PiCCSPayloadWiring
import NightstreamFPrime.Export.MatrixProgram.Affine
import NightstreamFPrime.Export.RowSemantics

/-!
Owns the compact affine payload table for the PiCCS matrix program. Its source
indices use the physical permutation expected by the parent substitution.
The proofs preserve ordered entries before any field evaluation.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSPayloadMatrix

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec

def combination (index : Fin PiCCSActionPayloadBlock.payloadCount) : R1CS.LinearCombination :=
  Package.mapCombinationColumns Spartan.sourceToSpartan
    (PiCCSPayloadWiring.lowering index).combination

private def compileWord? (expression : Circuit.Expr) : Option Affine.Form := do
  let lowered ← SourceCompiler.lowerAffine? Spartan.SourceColumnCount expression
  pure <| Affine.Form.ofSemantic <|
    Package.mapCombinationColumns Spartan.sourceToSpartan lowered.combination

private theorem compileWord?_payload (index : Fin PiCCSActionPayloadBlock.payloadCount) :
    compileWord? (PiCCSActionPayloadBlock.payloadExpression index) =
      some (Affine.Form.ofSemantic (combination index)) := by
  rw [compileWord?, PiCCSPayloadWiring.lowering?_eq_some_lowering]
  rfl

private theorem mapM_ofFn_some {count : Nat}
    (source : Fin count → Circuit.Expr) (expected : Fin count → Affine.Form)
    (agrees : ∀ index, compileWord? (source index) = some (expected index)) :
    (List.ofFn source).mapM compileWord? = some (List.ofFn expected) := by
  induction count with
  | zero =>
      simp only [List.ofFn_zero, List.mapM_nil]
      rfl
  | succ count inductionHypothesis =>
      rw [List.ofFn_succ, List.mapM_cons, agrees 0,
        inductionHypothesis (fun index => source index.succ)
          (fun index => expected index.succ) (fun index => agrees index.succ),
        List.ofFn_succ]
      rfl

/-- Compile the ordered payload once; an unrecognized word rejects the table. -/
private def table? (_delay : Unit := ()) : Option Affine.Table := do
  let words ← (PiCCSActionPayloadBlock.materializedPayloadExpressions ()).mapM compileWord?
  pure ⟨words.toArray⟩

private theorem table?_eq_some :
    table? () = some (Affine.Table.ofSemantic combination) := by
  rw [table?, PiCCSActionPayloadBlock.materializedPayloadExpressions_eq,
    mapM_ofFn_some PiCCSActionPayloadBlock.payloadExpression
      (fun index => Affine.Form.ofSemantic (combination index)) compileWord?_payload]
  change some ⟨(List.ofFn (fun index => Affine.Form.ofSemantic (combination index))).toArray⟩ =
    some (Affine.Table.ofSemantic combination)
  rw [List.toArray_ofFn]
  rfl

/-- Total emission follows from recognition of every canonical payload word. -/
def table (_delay : Unit := ()) : Affine.Table :=
  (table? ()).get (by rw [table?_eq_some]; rfl)

/-- Ordered traversal changes no word, coefficient, or position. -/
theorem table_eq_ofSemantic : table () = Affine.Table.ofSemantic combination := by
  simp only [table, table?_eq_some, Option.get_some]

private theorem compileMappedTerms
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (terms : List (Nat × F))
    (bounded : ∀ term ∈ terms, term.1 < Spartan.SourceColumnCount)
    (supported : ∀ term ∈ terms, PiCCSOrdinarySourceSupport.Source term.1) :
    Ordinary.compileTerms? (PiCCSOrdinaryMatrixProgram.substitution program) logicalWidth
        (terms.map fun term => (Spartan.sourceToSpartan term.1, term.2)) =
      some (SourceCompiler.compileTerms (PiCCSPayloadWiring.sourceMap geometry) terms bounded) := by
  induction terms with
  | nil => rfl
  | cons term rest inductionHypothesis =>
      have headBound := bounded term (by simp)
      have tailBound : ∀ value ∈ rest, value.1 < Spartan.SourceColumnCount := by
        intro value member
        exact bounded value (by simp [member])
      have tailSupport : ∀ value ∈ rest, PiCCSOrdinarySourceSupport.Source value.1 := by
        intro value member
        exact supported value (by simp [member])
      let column : Fin Spartan.spartanColumnCount :=
        ⟨Spartan.sourceToSpartan term.1, Spartan.sourceToSpartan_lt _ headBound⟩
      have head := PiCCSOrdinaryMatrixProgram.substitution_agrees_on_target geometry column
        (PiCCSOrdinarySourceSupport.source_target term.1 (supported term (by simp)))
      rw [List.map_cons, Ordinary.compileTerms?, head,
        inductionHypothesis tailBound tailSupport]
      rfl

/-- Exact remapping and substitution preserve the selected logical form. -/
theorem compileCombination_eq
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (index : Fin PiCCSActionPayloadBlock.payloadCount) :
    Ordinary.compileCombination? (PiCCSOrdinaryMatrixProgram.substitution program)
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry) (combination index) =
      some (PiCCSPayloadWiring.form geometry index) := by
  rw [PiCCSPayloadWiring.form_eq_compileCombination]
  unfold Ordinary.compileCombination? combination Package.mapCombinationColumns
  rw [compileMappedTerms geometry (PiCCSPayloadWiring.lowering index).combination.terms
    (PiCCSPayloadWiring.lowering index).bounded (PiCCSPayloadWiring.lowering_supported index)]
  rfl

theorem table_compile_eq
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (index : Fin PiCCSActionPayloadBlock.payloadCount) :
    (table ()).compile? (PiCCSOrdinaryMatrixProgram.substitution program) logicalWidth
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val index.val =
      some (PiCCSPayloadWiring.form geometry index) := by
  rw [table_eq_ofSemantic, Affine.Table.compile?_ofSemantic]
  exact compileCombination_eq geometry index

end NightstreamFPrime.Export.Stage1.PiCCSPayloadMatrix
