import NightstreamFPrime.Export.MatrixProgram.Ordinary

/-!
Owns canonical affine source words and their indexed substitution into final
sparse forms. Source columns are resolved by the existing substitution; field
coefficients are checked before evaluation. Table proofs are structural.
-/

namespace NightstreamFPrime.Export.MatrixProgram.Affine

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec

structure Form where
  constant : Nat
  terms : WireForm
deriving Repr, DecidableEq

def Form.format : Format Form where
  encode := fun form => .array [.atom form.constant, WireForm.format.encode form.terms]
  decode
    | .array [.atom constant, terms] => do
        pure ⟨constant, ← WireForm.format.decode terms⟩
    | _ => .error "invalid affine source form"
  decode_encode := by
    intro form
    cases form
    simp [Format.decode_encode]

private def readTerms : List WireEntry → Option (List (Nat × F))
  | [] => some []
  | term :: rest =>
      if canonical : term.coefficient < goldilocksModulus then do
        let tail ← readTerms rest
        pure ((term.column, ⟨term.coefficient, canonical⟩) :: tail)
      else none

def Form.semantic? (form : Form) : Option R1CS.LinearCombination :=
  if canonical : form.constant < goldilocksModulus then do
    let terms ← readTerms form.terms.entries
    pure ⟨⟨form.constant, canonical⟩, terms⟩
  else none

def Form.ofSemantic (combination : R1CS.LinearCombination) : Form where
  constant := combination.constant.val
  terms := ⟨combination.terms.map fun term => ⟨term.1, term.2.val⟩⟩

private theorem readTerms_ofSemantic (terms : List (Nat × F)) :
    readTerms (terms.map fun term => ⟨term.1, term.2.val⟩) = some terms := by
  induction terms with
  | nil => rfl
  | cons term rest inductionHypothesis =>
      simp only [List.map_cons, readTerms, dif_pos term.2.isLt, inductionHypothesis]
      rfl

/-- Decoding retains every source index, coefficient, and term position. -/
theorem Form.semantic?_ofSemantic (combination : R1CS.LinearCombination) :
    (Form.ofSemantic combination).semantic? = some combination := by
  cases combination with
  | mk constant terms =>
      simp [Form.semantic?, Form.ofSemantic, constant.isLt, readTerms_ofSemantic]

structure Table where
  values : Array Form
deriving Repr, DecidableEq

def Table.format : Format Table where
  encode := fun table => (list Form.format).encode table.values.toList
  decode := fun value => do
    pure ⟨(← (list Form.format).decode value).toArray⟩
  decode_encode := by
    intro table
    cases table
    simp [Format.decode_encode]

def Table.combination? (table : Table) (index : Nat) :
    Option R1CS.LinearCombination := do
  let encoded ← table.values[index]?
  encoded.semantic?

def Table.ofSemantic {count : Nat}
    (combination : Fin count → R1CS.LinearCombination) : Table where
  values := Array.ofFn fun index => Form.ofSemantic (combination index)

theorem Table.combination?_ofSemantic {count : Nat}
    (combination : Fin count → R1CS.LinearCombination) (index : Fin count) :
    (Table.ofSemantic combination).combination? index.val = some (combination index) := by
  simp [Table.combination?, Table.ofSemantic, Form.semantic?_ofSemantic]

def Table.compile? (table : Table) (substitution : SourceSubstitution)
    (logicalWidth oneColumn index : Nat) : Option (SparseForm logicalWidth) :=
  if oneBound : oneColumn < logicalWidth then do
    let combination ← table.combination? index
    Ordinary.compileCombination? substitution ⟨oneColumn, oneBound⟩ combination
  else none

/-- The wire table applies the same ordered affine substitution as the
ordinary matrix compiler. Missing columns and invalid encodings fail closed. -/
theorem Table.compile?_ofSemantic {count logicalWidth : Nat}
    (combination : Fin count → R1CS.LinearCombination)
    (substitution : SourceSubstitution) (oneColumn : Fin logicalWidth)
    (index : Fin count) :
    (Table.ofSemantic combination).compile? substitution logicalWidth oneColumn.val index.val =
      Ordinary.compileCombination? substitution oneColumn (combination index) := by
  simp only [Table.compile?, dif_pos oneColumn.isLt, Table.combination?_ofSemantic]
  rfl

end NightstreamFPrime.Export.MatrixProgram.Affine
