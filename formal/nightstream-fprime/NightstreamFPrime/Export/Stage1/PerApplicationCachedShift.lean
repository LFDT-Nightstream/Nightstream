import NightstreamFPrime.Export.Stage1.PerApplicationPackage

/-!
Owns the cached executable form of per-application column shifting.

The generic package definitions remain semantic authority. A `Context`
computes their application-private delta once, then reuses it for every
serialized term. Each cached structure has an equality theorem to the
generic shift.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationCachedShift

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Lifecycle

structure Context where
  program : Lifecycle.Stage1.Application.Program
  privateDelta : Nat
  privateDelta_eq :
    privateDelta = PerApplicationPackage.directAddedPrivateColumnCount program

def Context.ofProgram
    (program : Lifecycle.Stage1.Application.Program) : Context where
  program := program
  privateDelta := PerApplicationPackage.directAddedPrivateColumnCount program
  privateDelta_eq := rfl

def Context.column (context : Context) (column : Nat) : Nat :=
  if column < Data.physicalLayout.constantColumn then column
  else column + context.privateDelta

theorem Context.column_eq_directShiftColumn (context : Context)
    (column : Nat) :
    context.column column =
      PerApplicationPackage.directShiftColumn context.program column := by
  simp [Context.column, PerApplicationPackage.directShiftColumn,
    context.privateDelta_eq]

theorem Context.column_eq_shiftColumn (context : Context)
    (column : Nat) :
    context.column column =
      PerApplicationPackage.shiftColumn context.program column :=
  (context.column_eq_directShiftColumn column).trans
    (PerApplicationPackage.directShiftColumn_eq_shiftColumn
      context.program column)

theorem Context.column_function_eq (context : Context) :
    context.column = PerApplicationPackage.shiftColumn context.program := by
  funext column
  exact context.column_eq_shiftColumn column

def shiftExpr (context : Context) (value : Expr) : Expr :=
  CompactRows.renameExpr context.column value

theorem shiftExpr_eq (context : Context) (value : Expr) :
    shiftExpr context value =
      PerApplicationPackage.shiftExpr context.program value := by
  rw [shiftExpr, PerApplicationPackage.shiftExpr,
    context.column_function_eq]

def shiftHint (context : Context) : Hint → Hint
  | .bit source index => .bit (shiftExpr context source) index
  | .inverseOrZero source => .inverseOrZero (shiftExpr context source)
  | .quotientFive source => .quotientFive (shiftExpr context source)
  | .remainderFive source => .remainderFive (shiftExpr context source)

theorem shiftHint_eq (context : Context) (hint : Hint) :
    shiftHint context hint =
      PerApplicationPackage.shiftHint context.program hint := by
  cases hint <;> simp [shiftHint, PerApplicationPackage.shiftHint,
    shiftExpr_eq]

def shiftBatch (context : Context) (batch : WitnessBatch) : WitnessBatch where
  start := context.column batch.start
  recipes := batch.recipes.map (shiftExpr context)
  hints := batch.hints.map (shiftHint context)

theorem shiftBatch_eq (context : Context) (batch : WitnessBatch) :
    shiftBatch context batch =
      PerApplicationPackage.shiftBatch context.program batch := by
  cases batch
  simp [shiftBatch, PerApplicationPackage.shiftBatch,
    context.column_eq_shiftColumn, shiftExpr_eq, shiftHint_eq]

def shiftSparseTerm (context : Context) (term : SparseTerm) : SparseTerm :=
  ⟨context.column term.column, term.coefficient⟩

theorem shiftSparseTerm_eq (context : Context) (term : SparseTerm) :
    shiftSparseTerm context term =
      PerApplicationPackage.shiftSparseTerm context.program term := by
  cases term
  simp [shiftSparseTerm, PerApplicationPackage.shiftSparseTerm,
    context.column_eq_shiftColumn]

def shiftSparseCombination (context : Context)
    (combination : SparseCombination) : SparseCombination :=
  ⟨combination.constant, combination.terms.map (shiftSparseTerm context)⟩

theorem shiftSparseCombination_eq (context : Context)
    (combination : SparseCombination) :
    shiftSparseCombination context combination =
      PerApplicationPackage.shiftSparseCombination context.program
        combination := by
  cases combination
  simp [shiftSparseCombination,
    PerApplicationPackage.shiftSparseCombination, shiftSparseTerm_eq]

def shiftWitnessInstruction (context : Context)
    (instruction : WitnessInstruction) : WitnessInstruction where
  rowIndex := instruction.rowIndex
  target := context.column instruction.target
  a := shiftSparseCombination context instruction.a
  b := shiftSparseCombination context instruction.b

theorem shiftWitnessInstruction_eq (context : Context)
    (instruction : WitnessInstruction) :
    shiftWitnessInstruction context instruction =
      PerApplicationPackage.shiftWitnessInstruction context.program
        instruction := by
  cases instruction
  simp [shiftWitnessInstruction,
    PerApplicationPackage.shiftWitnessInstruction,
    context.column_eq_shiftColumn, shiftSparseCombination_eq]

def shiftSparseRow (context : Context) (row : SparseRow) : SparseRow where
  rowIndex := row.rowIndex
  a := shiftSparseCombination context row.a
  b := shiftSparseCombination context row.b
  c := shiftSparseCombination context row.c

theorem shiftSparseRow_eq (context : Context) (row : SparseRow) :
    shiftSparseRow context row =
      PerApplicationPackage.shiftSparseRow context.program row := by
  cases row
  simp [shiftSparseRow, PerApplicationPackage.shiftSparseRow,
    shiftSparseCombination_eq]

def shiftHashChain (context : Context) (chain : HashChain) : HashChain where
  phase := chain.phase
  rowStart := chain.rowStart
  rowCount := chain.rowCount
  inputStart := context.column chain.inputStart
  inputLength := chain.inputLength
  witnessStart := context.column chain.witnessStart
  witnessLength := chain.witnessLength
  absorbCount := chain.absorbCount
  digestLength := chain.digestLength
  digestStart := context.column chain.digestStart

theorem shiftHashChain_eq (context : Context) (chain : HashChain) :
    shiftHashChain context chain =
      PerApplicationPackage.shiftHashChain context.program chain := by
  cases chain
  simp [shiftHashChain, PerApplicationPackage.shiftHashChain,
    context.column_eq_shiftColumn]

def shiftPermutationInvocation (context : Context)
    (invocation : PermutationInvocation) : PermutationInvocation where
  phase := invocation.phase
  rowStart := invocation.rowStart
  witnessStart := context.column invocation.witnessStart
  inputs := invocation.inputs.map (shiftSparseCombination context)

theorem shiftPermutationInvocation_eq (context : Context)
    (invocation : PermutationInvocation) :
    shiftPermutationInvocation context invocation =
      PerApplicationPackage.shiftPermutationInvocation context.program
        invocation := by
  cases invocation
  simp [shiftPermutationInvocation,
    PerApplicationPackage.shiftPermutationInvocation,
    context.column_eq_shiftColumn, shiftSparseCombination_eq]

def shiftCompactInputRange (context : Context)
    (range : CompactInputRange) : CompactInputRange where
  inputStart := range.inputStart
  inputCount := range.inputCount
  columnStart := context.column range.columnStart
  columnStride := range.columnStride

theorem shiftCompactInputRange_eq (context : Context)
    (range : CompactInputRange) :
    shiftCompactInputRange context range =
      PerApplicationPackage.shiftCompactInputRange context.program range := by
  cases range
  simp [shiftCompactInputRange,
    PerApplicationPackage.shiftCompactInputRange,
    context.column_eq_shiftColumn]

def shiftCompactRowInvocation (context : Context)
    (invocation : CompactRowInvocation) : CompactRowInvocation where
  phase := invocation.phase
  templateIndex := invocation.templateIndex
  rowStart := invocation.rowStart
  localStart := context.column invocation.localStart
  inputRanges := invocation.inputRanges.map (shiftCompactInputRange context)

theorem shiftCompactRowInvocation_eq (context : Context)
    (invocation : CompactRowInvocation) :
    shiftCompactRowInvocation context invocation =
      PerApplicationPackage.shiftCompactRowInvocation context.program
        invocation := by
  cases invocation
  simp [shiftCompactRowInvocation,
    PerApplicationPackage.shiftCompactRowInvocation,
    context.column_eq_shiftColumn, shiftCompactInputRange_eq]

end NightstreamFPrime.Export.Stage1.PerApplicationCachedShift
