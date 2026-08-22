import NightstreamFPrime.Export.Package
import NightstreamFPrime.Layout.R1CS

/-!
Owns the Lean semantics of the compact circuit-package row program. Template
rows instantiate to ordinary `Layout.R1CS.Row` values. Hash-chain substitution
uses only package fields. Rust may execute this program, but it does not define
its meaning.
-/

namespace NightstreamFPrime.Export.Package

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout

def fieldValue (value : Nat) : F := Poseidon2.ofNat value

def ColumnRef.eval (value : ColumnRef → F) (column : ColumnRef) : F :=
  value column

def TemplateCombination.eval (combination : TemplateCombination)
    (value : ColumnRef → F) : F :=
  fieldValue combination.constant +
    (combination.terms.map fun term =>
      fieldValue term.coefficient * term.column.eval value).sum

def TemplateRow.Holds (row : TemplateRow) (value : ColumnRef → F) : Prop :=
  row.a.eval value * row.b.eval value = row.c.eval value

private def sumCombinations : List R1CS.LinearCombination →
    R1CS.LinearCombination
  | [] => R1CS.LinearCombination.zero
  | combination :: rest =>
      R1CS.LinearCombination.add combination (sumCombinations rest)

private theorem sumCombinations_eval (env : Env)
    (combinations : List R1CS.LinearCombination) :
    (sumCombinations combinations).eval env =
      (combinations.map fun combination => combination.eval env).sum := by
  induction combinations with
  | nil => rfl
  | cons combination rest ih =>
      simp [sumCombinations, ih]

/-- The affine input to one permutation invocation. Invocation zero starts
from zero. Later invocations use the previous permutation output. Absorb
invocations add one rate-width input block; the final invocation adds the
lane-zero `+1` padding marker. -/
def invocationInput (package : CircuitPackage) (chain : HashChain)
    (invocation lane : Nat) : R1CS.LinearCombination :=
  let previous :=
    if invocation = 0 then
      R1CS.LinearCombination.zero
    else
      R1CS.LinearCombination.ofVar
        (chain.witnessStart +
          (invocation - 1) * package.permutation.localColumnCount +
          package.permutation.outputLocalStart + lane)
  if invocation < chain.absorbCount then
    let inputOffset := invocation * package.poseidon.rate + lane
    if lane < package.poseidon.rate ∧ inputOffset < chain.inputLength then
      R1CS.LinearCombination.add previous
        (R1CS.LinearCombination.ofVar (chain.inputStart + inputOffset))
    else
      previous
  else if lane = 0 then
    R1CS.LinearCombination.add previous R1CS.LinearCombination.one
  else
    previous

def invocationLocalStart (package : CircuitPackage) (chain : HashChain)
    (invocation : Nat) : Nat :=
  chain.witnessStart + invocation * package.permutation.localColumnCount

def instantiateColumn (package : CircuitPackage) (chain : HashChain)
    (invocation : Nat) : ColumnRef → R1CS.LinearCombination
  | .input lane => invocationInput package chain invocation lane
  | .local index =>
      R1CS.LinearCombination.ofVar
        (invocationLocalStart package chain invocation + index)

def instantiateCombination (package : CircuitPackage) (chain : HashChain)
    (invocation : Nat) (combination : TemplateCombination) :
    R1CS.LinearCombination :=
  R1CS.LinearCombination.add
    (R1CS.LinearCombination.const (fieldValue combination.constant))
    (sumCombinations (combination.terms.map fun term =>
      R1CS.LinearCombination.scale (fieldValue term.coefficient)
        (instantiateColumn package chain invocation term.column)))

def instantiateRow (package : CircuitPackage) (chain : HashChain)
    (invocation : Nat) (row : TemplateRow) : R1CS.Row :=
  ⟨instantiateCombination package chain invocation row.a,
    instantiateCombination package chain invocation row.b,
    instantiateCombination package chain invocation row.c⟩

/-- Instantiation has exactly the template-row semantics under the package's
chain substitution. -/
theorem instantiateRow_holds (package : CircuitPackage) (chain : HashChain)
    (invocation : Nat) (row : TemplateRow) (env : Env) :
    (instantiateRow package chain invocation row).Holds env ↔
      row.Holds (fun column =>
        (instantiateColumn package chain invocation column).eval env) := by
  have combinationEval (combination : TemplateCombination) :
      (instantiateCombination package chain invocation combination).eval env =
        combination.eval (fun column =>
          (instantiateColumn package chain invocation column).eval env) := by
    unfold instantiateCombination TemplateCombination.eval
    simp [sumCombinations_eval, List.map_map, Function.comp_def,
      ColumnRef.eval]
  simp [R1CS.Row.Holds, instantiateRow, TemplateRow.Holds,
    combinationEval]

def TemplateInvocationHolds (package : CircuitPackage) (chain : HashChain)
    (invocation : Nat) (env : Env) : Prop :=
  ∀ row ∈ package.permutation.rows,
    (instantiateRow package chain invocation row).Holds env

def HashChainHolds (package : CircuitPackage) (chain : HashChain)
    (env : Env) : Prop :=
  ∀ invocation, invocation ≤ chain.absorbCount →
    TemplateInvocationHolds package chain invocation env

def SparseCombination.eval (combination : SparseCombination) (env : Env) : F :=
  fieldValue combination.constant +
    (combination.terms.map fun term =>
      fieldValue term.coefficient * env term.column).sum

def SparseRow.Holds (row : SparseRow) (env : Env) : Prop :=
  row.a.eval env * row.b.eval env = row.c.eval env

def SparseCombination.toR1CS (combination : SparseCombination) :
    R1CS.LinearCombination :=
  ⟨fieldValue combination.constant,
    combination.terms.map fun term =>
      (term.column, fieldValue term.coefficient)⟩

def SparseRow.toR1CS (row : SparseRow) : R1CS.Row :=
  ⟨row.a.toR1CS, row.b.toR1CS, row.c.toR1CS⟩

theorem sparseRow_holds (row : SparseRow) (env : Env) :
    row.toR1CS.Holds env ↔ row.Holds env := by
  have combinationEval (combination : SparseCombination) :
      combination.toR1CS.eval env = combination.eval env := by
    simp [SparseCombination.toR1CS, SparseCombination.eval,
      R1CS.LinearCombination.eval, List.map_map, Function.comp_def]
  simp [R1CS.Row.Holds, SparseRow.toR1CS, SparseRow.Holds,
    combinationEval]

def AssertionsHold (package : CircuitPackage) (env : Env) : Prop :=
  ∀ row ∈ package.assertionRows, row.Holds env

/-- Authoritative row semantics of a loaded circuit package. -/
def CircuitPackage.RowsHold (package : CircuitPackage) (env : Env) : Prop :=
  (∀ chain ∈ package.hashChains, HashChainHolds package chain env) ∧
    AssertionsHold package env

end NightstreamFPrime.Export.Package
