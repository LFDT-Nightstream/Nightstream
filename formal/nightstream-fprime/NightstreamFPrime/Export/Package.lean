import NightstreamFPrime.Circuit.Basic
import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Spec.Poseidon2
import NightstreamFPrime.Spec.ProductionRelation

/-!
Owns the canonical circuit-package schema, its lossless structured codec, and
the Poseidon2 relation identifier. Concrete pilot data belongs to `Pilot`.
The schema stores sparse matrix templates and their exact instantiations; Rust
may expand them but does not choose rows or circuit structure.
-/

namespace NightstreamFPrime.Export.Package

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Codec

/-- The one fixed Nightstream Goldilocks production profile. -/
structure Profile where
  fieldModulus : Nat
  decompositionBase : Nat
  decompositionDigits : Nat
  decompositionBound : Nat
  freshSources : Nat
  runningSources : Nat
  piRlcInputs : Nat
  piDecChildren : Nat
  ccsMatrices : Nat
  cubeVariables : Nat
deriving Repr

def Profile.format : Format Profile where
  encode := fun value => .array [
    .atom value.fieldModulus,
    .atom value.decompositionBase,
    .atom value.decompositionDigits,
    .atom value.decompositionBound,
    .atom value.freshSources,
    .atom value.runningSources,
    .atom value.piRlcInputs,
    .atom value.piDecChildren,
    .atom value.ccsMatrices,
    .atom value.cubeVariables]
  decode
    | .array [.atom fieldModulus, .atom decompositionBase,
        .atom decompositionDigits, .atom decompositionBound,
        .atom freshSources, .atom runningSources, .atom piRlcInputs,
        .atom piDecChildren, .atom ccsMatrices, .atom cubeVariables] =>
      .ok ⟨fieldModulus, decompositionBase, decompositionDigits,
        decompositionBound, freshSources, runningSources, piRlcInputs,
        piDecChildren, ccsMatrices, cubeVariables⟩
    | _ => .error "invalid production profile"
  decode_encode := by
    intro value
    cases value
    rfl

/-- Poseidon2 parameters that every hash-chain template uses. -/
structure PoseidonSchedule where
  width : Nat
  rate : Nat
  digestLength : Nat
  initialFullRounds : Nat
  partialRounds : Nat
  terminalFullRounds : Nat
  recipesPerPermutation : Nat
  outputLocalStart : Nat
deriving Repr

def PoseidonSchedule.format : Format PoseidonSchedule where
  encode := fun value => .array [
    .atom value.width,
    .atom value.rate,
    .atom value.digestLength,
    .atom value.initialFullRounds,
    .atom value.partialRounds,
    .atom value.terminalFullRounds,
    .atom value.recipesPerPermutation,
    .atom value.outputLocalStart]
  decode
    | .array [.atom width, .atom rate, .atom digestLength,
        .atom initialFullRounds, .atom partialRounds,
        .atom terminalFullRounds, .atom recipesPerPermutation,
        .atom outputLocalStart] =>
      .ok ⟨width, rate, digestLength, initialFullRounds, partialRounds,
        terminalFullRounds, recipesPerPermutation, outputLocalStart⟩
    | _ => .error "invalid Poseidon2 schedule"
  decode_encode := by
    intro value
    cases value
    rfl

/-- One named contiguous segment in the final Spartan column order. -/
structure Segment where
  role : Nat
  start : Nat
  length : Nat
deriving Repr

def Segment.format : Format Segment where
  encode := fun value => .array [
    .atom value.role, .atom value.start, .atom value.length]
  decode
    | .array [.atom role, .atom start, .atom length] =>
      .ok ⟨role, start, length⟩
    | _ => .error "invalid layout segment"
  decode_encode := by
    intro value
    cases value
    rfl

/-- Complete physical layout after the proved Lean-to-Spartan permutation. -/
structure PhysicalLayout where
  rowCount : Nat
  privateColumnCount : Nat
  constantColumn : Nat
  publicColumnCount : Nat
  totalColumnCount : Nat
  privateSegments : List Segment
  publicSegments : List Segment
deriving Repr

def PhysicalLayout.format : Format PhysicalLayout where
  encode := fun value => .array [
    .atom value.rowCount,
    .atom value.privateColumnCount,
    .atom value.constantColumn,
    .atom value.publicColumnCount,
    .atom value.totalColumnCount,
    (list Segment.format).encode value.privateSegments,
    (list Segment.format).encode value.publicSegments]
  decode
    | .array [.atom rowCount, .atom privateColumnCount,
        .atom constantColumn, .atom publicColumnCount,
        .atom totalColumnCount, privateSegments, publicSegments] => do
      pure ⟨rowCount, privateColumnCount, constantColumn,
        publicColumnCount, totalColumnCount,
        ← (list Segment.format).decode privateSegments,
        ← (list Segment.format).decode publicSegments⟩
    | _ => .error "invalid physical layout"
  decode_encode := by
    intro value
    cases value
    simp [Format.decode_encode] <;> rfl

/-- The physical R1CS matrix selected for one logical CCS matrix slot. Pad is
not a source: SuperNeo v1_1 carries its evaluations in `Eval_K`. -/
inductive CcsMatrixSource where
  | bit
  | generalSelector
  | a
  | b
  | c
  | sboxInput
  | centeredUnit
  | evalSelector
  | class0
  | class1
  | class2
  | class3
  | class4
  | zero
deriving Repr

def CcsMatrixSource.format : Format CcsMatrixSource where
  encode
    | .bit => .atom 0
    | .generalSelector => .atom 1
    | .a => .atom 2
    | .b => .atom 3
    | .c => .atom 4
    | .sboxInput => .atom 5
    | .centeredUnit => .atom 6
    | .evalSelector => .atom 7
    | .class0 => .atom 8
    | .class1 => .atom 9
    | .class2 => .atom 10
    | .class3 => .atom 11
    | .class4 => .atom 12
    | .zero => .atom 13
  decode
    | .atom 0 => .ok .bit
    | .atom 1 => .ok .generalSelector
    | .atom 2 => .ok .a
    | .atom 3 => .ok .b
    | .atom 4 => .ok .c
    | .atom 5 => .ok .sboxInput
    | .atom 6 => .ok .centeredUnit
    | .atom 7 => .ok .evalSelector
    | .atom 8 => .ok .class0
    | .atom 9 => .ok .class1
    | .atom 10 => .ok .class2
    | .atom 11 => .ok .class3
    | .atom 12 => .ok .class4
    | .atom 13 => .ok .zero
    | _ => .error "invalid CCS matrix source"
  decode_encode := by
    intro value
    cases value <;> rfl

/-- One sparse term of the logical CCS constraint polynomial. Exponents use
the same canonical matrix-slot order as `matrixSources`. -/
structure CcsPolynomialTerm where
  coefficient : Nat
  exponents : List Nat
deriving Repr

def CcsPolynomialTerm.format : Format CcsPolynomialTerm where
  encode := fun value => .array [
    .atom value.coefficient,
    (list nat).encode value.exponents]
  decode
    | .array [.atom coefficient, exponents] => do
      pure ⟨coefficient, ← (list nat).decode exponents⟩
    | _ => .error "invalid CCS polynomial term"
  decode_encode := by
    intro value
    cases value
    simp [Format.decode_encode] <;> rfl

/-- Logical CCS relation carried by the package beside its physical row
program. Rust decodes this record; it does not choose matrix slots or the
constraint polynomial. -/
structure CcsRelation where
  rowCount : Nat
  columnCount : Nat
  cubeVariables : Nat
  matrixSources : List CcsMatrixSource
  degreeBound : Nat
  terms : List CcsPolynomialTerm
deriving Repr

def CcsRelation.format : Format CcsRelation where
  encode := fun value => .array [
    .atom value.rowCount,
    .atom value.columnCount,
    .atom value.cubeVariables,
    (list CcsMatrixSource.format).encode value.matrixSources,
    .atom value.degreeBound,
    (list CcsPolynomialTerm.format).encode value.terms]
  decode
    | .array [.atom rowCount, .atom columnCount, .atom cubeVariables,
        matrixSources, .atom degreeBound, terms] => do
      pure ⟨rowCount, columnCount, cubeVariables,
        ← (list CcsMatrixSource.format).decode matrixSources,
        degreeBound, ← (list CcsPolynomialTerm.format).decode terms⟩
    | _ => .error "invalid CCS relation"
  decode_encode := by
    intro value
    cases value
    simp [Format.decode_encode] <;> rfl

/-- Exact selective matrix-slot order. Pad remains outside this list. -/
def productionMatrixSources : List CcsMatrixSource :=
  [.bit, .generalSelector, .a, .b, .c, .sboxInput, .centeredUnit,
    .evalSelector, .class0, .class1, .class2, .class3, .class4, .zero]

def encodeProductionTerm
    (term : NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable.Monomial
      F Spec.ProductionRelation.matrixCount) : CcsPolynomialTerm where
  coefficient := term.coefficient.val
  exponents := List.ofFn term.exponents

def productionTerms : List CcsPolynomialTerm :=
  Spec.ProductionRelation.polynomial.terms.map encodeProductionTerm

/-- Wire form of the exact production relation from `Spec.ProductionRelation`.
Only its physical dimensions are supplied by the selected layout. -/
def productionCcsRelation
    (rowCount columnCount cubeVariables : Nat) : CcsRelation where
  rowCount := rowCount
  columnCount := columnCount
  cubeVariables := cubeVariables
  matrixSources := productionMatrixSources
  degreeBound := Spec.ProductionRelation.polynomial.degreeBound
  terms := productionTerms

@[simp] theorem productionCcsRelation_matrixSources
    (rowCount columnCount cubeVariables : Nat) :
    (productionCcsRelation rowCount columnCount cubeVariables).matrixSources =
      productionMatrixSources := by
  rfl

@[simp] theorem productionCcsRelation_terms
    (rowCount columnCount cubeVariables : Nat) :
    (productionCcsRelation rowCount columnCount cubeVariables).terms =
      productionTerms := by
  rfl

@[simp] theorem productionMatrixSources_length :
    productionMatrixSources.length = 14 := by
  rfl

@[simp] theorem productionTerms_length : productionTerms.length = 74 := by
  unfold productionTerms
  rw [List.length_map, Spec.ProductionRelation.polynomial_terms]
  exact Spec.ProductionRelation.SelectivePolynomial.terms_length

/-- A sparse template column is one of eight invocation inputs or one local
witness column. -/
inductive ColumnRef where
  | input (index : Nat)
  | local (index : Nat)
deriving Repr

def ColumnRef.format : Format ColumnRef where
  encode
    | .input index => .array [.atom 0, .atom index]
    | .local index => .array [.atom 1, .atom index]
  decode
    | .array [.atom 0, .atom index] => .ok (.input index)
    | .array [.atom 1, .atom index] => .ok (.local index)
    | _ => .error "invalid template column reference"
  decode_encode := by
    intro value
    cases value <;> rfl

structure TemplateTerm where
  column : ColumnRef
  coefficient : Nat
deriving Repr

def TemplateTerm.format : Format TemplateTerm where
  encode := fun value => .array [
    ColumnRef.format.encode value.column, .atom value.coefficient]
  decode
    | .array [column, .atom coefficient] => do
      pure ⟨← ColumnRef.format.decode column, coefficient⟩
    | _ => .error "invalid template term"
  decode_encode := by
    intro value
    cases value
    simp [ColumnRef.format.decode_encode] <;> rfl

structure TemplateCombination where
  constant : Nat
  terms : List TemplateTerm
deriving Repr

def TemplateCombination.format : Format TemplateCombination where
  encode := fun value => .array [
    .atom value.constant, (list TemplateTerm.format).encode value.terms]
  decode
    | .array [.atom constant, terms] => do
      pure ⟨constant, ← (list TemplateTerm.format).decode terms⟩
    | _ => .error "invalid template linear combination"
  decode_encode := by
    intro value
    cases value
    simp [Format.decode_encode] <;> rfl

/-- One sparse template equation. `outputLocal` is the witness instruction:
the interpreter writes `A · B` there, and the same row checks the result. -/
structure TemplateRow where
  outputLocal : Nat
  a : TemplateCombination
  b : TemplateCombination
  c : TemplateCombination
deriving Repr

def TemplateRow.format : Format TemplateRow where
  encode := fun value => .array [
    .atom value.outputLocal,
    TemplateCombination.format.encode value.a,
    TemplateCombination.format.encode value.b,
    TemplateCombination.format.encode value.c]
  decode
    | .array [.atom outputLocal, a, b, c] => do
      pure ⟨outputLocal,
        ← TemplateCombination.format.decode a,
        ← TemplateCombination.format.decode b,
        ← TemplateCombination.format.decode c⟩
    | _ => .error "invalid template row"
  decode_encode := by
    intro value
    cases value
    simp [TemplateCombination.format.decode_encode] <;> rfl

/-- Sparse matrix data for one Poseidon2 permutation. -/
structure PermutationTemplate where
  inputCount : Nat
  localColumnCount : Nat
  outputLocalStart : Nat
  rows : List TemplateRow
deriving Repr

def PermutationTemplate.format : Format PermutationTemplate where
  encode := fun value => .array [
    .atom value.inputCount,
    .atom value.localColumnCount,
    .atom value.outputLocalStart,
    (list TemplateRow.format).encode value.rows]
  decode
    | .array [.atom inputCount, .atom localColumnCount,
        .atom outputLocalStart, rows] => do
      pure ⟨inputCount, localColumnCount, outputLocalStart,
        ← (list TemplateRow.format).decode rows⟩
    | _ => .error "invalid permutation template"
  decode_encode := by
    intro value
    cases value
    simp [Format.decode_encode] <;> rfl

/-- One exact instantiation chain. Absorb permutations precede the one final
padding permutation. `digestLength` records how many caller-owned digest
columns are bound by ordinary rows after the permutation rows. -/
structure HashChain where
  phase : Nat
  rowStart : Nat
  rowCount : Nat
  inputStart : Nat
  inputLength : Nat
  witnessStart : Nat
  witnessLength : Nat
  absorbCount : Nat
  digestLength : Nat
  digestStart : Nat
deriving Repr

def HashChain.format : Format HashChain where
  encode := fun value => .array [
    .atom value.phase,
    .atom value.rowStart,
    .atom value.rowCount,
    .atom value.inputStart,
    .atom value.inputLength,
    .atom value.witnessStart,
    .atom value.witnessLength,
    .atom value.absorbCount,
    .atom value.digestLength,
    .atom value.digestStart]
  decode
    | .array [.atom phase, .atom rowStart, .atom rowCount,
        .atom inputStart, .atom inputLength, .atom witnessStart,
        .atom witnessLength, .atom absorbCount, .atom digestLength,
        .atom digestStart] =>
      .ok ⟨phase, rowStart, rowCount, inputStart, inputLength,
        witnessStart, witnessLength, absorbCount, digestLength, digestStart⟩
    | _ => .error "invalid hash chain"
  decode_encode := by
    intro value
    cases value
    rfl

structure SparseTerm where
  column : Nat
  coefficient : Nat
deriving Repr

def SparseTerm.format : Format SparseTerm where
  encode := fun value => .array [.atom value.column, .atom value.coefficient]
  decode
    | .array [.atom column, .atom coefficient] => .ok ⟨column, coefficient⟩
    | _ => .error "invalid sparse term"
  decode_encode := by
    intro value
    cases value
    rfl

structure SparseCombination where
  constant : Nat
  terms : List SparseTerm
deriving Repr

def SparseCombination.format : Format SparseCombination where
  encode := fun value => .array [
    .atom value.constant, (list SparseTerm.format).encode value.terms]
  decode
    | .array [.atom constant, terms] => do
      pure ⟨constant, ← (list SparseTerm.format).decode terms⟩
    | _ => .error "invalid sparse linear combination"
  decode_encode := by
    intro value
    cases value
    simp [Format.decode_encode] <;> rfl

/-- One explicit use of the canonical Poseidon2 permutation template.
`inputs` contains one absolute sparse combination for each of the eight input
lanes. The template owns the adjacent witness and row intervals. -/
structure PermutationInvocation where
  phase : Nat
  rowStart : Nat
  witnessStart : Nat
  inputs : List SparseCombination
deriving Repr

def PermutationInvocation.format : Format PermutationInvocation where
  encode := fun value => .array [
    .atom value.phase,
    .atom value.rowStart,
    .atom value.witnessStart,
    (list SparseCombination.format).encode value.inputs]
  decode
    | .array [.atom phase, .atom rowStart, .atom witnessStart, inputs] => do
      pure ⟨phase, rowStart, witnessStart,
        ← (list SparseCombination.format).decode inputs⟩
    | _ => .error "invalid permutation invocation"
  decode_encode := by
    intro value
    cases value
    simp [Format.decode_encode] <;> rfl

/-- One row in a generic compact template. A witness row names the local
column that receives `A * B`. An assertion row has no output target and only
checks its exact `A * B = C` equation. -/
structure CompactTemplateRow where
  outputLocal : Option Nat
  a : TemplateCombination
  b : TemplateCombination
  c : TemplateCombination
deriving Repr

def encodeOptionalNat : Option Nat → Value
  | none => .array [.atom 0]
  | some value => .array [.atom 1, .atom value]

def decodeOptionalNat : Value → Except String (Option Nat)
  | .array [.atom 0] => .ok none
  | .array [.atom 1, .atom value] => .ok (some value)
  | _ => .error "invalid optional natural"

@[simp] theorem decodeOptionalNat_encode (value : Option Nat) :
    decodeOptionalNat (encodeOptionalNat value) = .ok value := by
  cases value <;> rfl

def CompactTemplateRow.format : Format CompactTemplateRow where
  encode := fun value => .array [
    encodeOptionalNat value.outputLocal,
    TemplateCombination.format.encode value.a,
    TemplateCombination.format.encode value.b,
    TemplateCombination.format.encode value.c]
  decode
    | .array [outputLocal, a, b, c] => do
      pure ⟨← decodeOptionalNat outputLocal,
        ← TemplateCombination.format.decode a,
        ← TemplateCombination.format.decode b,
        ← TemplateCombination.format.decode c⟩
    | _ => .error "invalid compact template row"
  decode_encode := by
    intro value
    cases value
    simp [decodeOptionalNat_encode,
      TemplateCombination.format.decode_encode,
      Format.decode_encode] <;> rfl

/-- One Lean-authored generic row template. `outputRecipe` computes the one
logical output column selected by `outputInput`. The rows then compute the
template-local R1CS intermediates and check the final assertion. -/
structure CompactRowTemplate where
  inputCount : Nat
  localColumnCount : Nat
  outputInput : Nat
  outputRecipe : Expr
  rows : List CompactTemplateRow
deriving Repr

/-- One contiguous affine binding from template input slots to final package
columns. Slot `inputStart + i` reads `columnStart + i * columnStride`. -/
structure CompactInputRange where
  inputStart : Nat
  inputCount : Nat
  columnStart : Nat
  columnStride : Nat
deriving Repr

def CompactInputRange.format : Format CompactInputRange where
  encode := fun value => .array [
    .atom value.inputStart, .atom value.inputCount,
    .atom value.columnStart, .atom value.columnStride]
  decode
    | .array [.atom inputStart, .atom inputCount, .atom columnStart,
        .atom columnStride] =>
      .ok ⟨inputStart, inputCount, columnStart, columnStride⟩
    | _ => .error "invalid compact input range"
  decode_encode := by
    intro value
    cases value
    rfl

/-- One exact use of a compact row template. The invocation owns one logical
output column, one contiguous local interval, and one contiguous row interval.
Its input ranges contain only final package column numbers. -/
structure CompactRowInvocation where
  phase : Nat
  templateIndex : Nat
  rowStart : Nat
  localStart : Nat
  inputRanges : List CompactInputRange
deriving Repr

def CompactRowInvocation.format : Format CompactRowInvocation where
  encode := fun value => .array [
    .atom value.phase,
    .atom value.templateIndex,
    .atom value.rowStart,
    .atom value.localStart,
    (list CompactInputRange.format).encode value.inputRanges]
  decode
    | .array [.atom phase, .atom templateIndex, .atom rowStart,
        .atom localStart, inputRanges] => do
      pure ⟨phase, templateIndex, rowStart, localStart,
        ← (list CompactInputRange.format).decode inputRanges⟩
    | _ => .error "invalid compact row invocation"
  decode_encode := by
    intro value
    cases value
    simp [Format.decode_encode] <;> rfl

/-- One generic straight-line witness instruction and its authoritative row.
The interpreter writes `a * b` to `target`; the same package row checks that
value. `rowIndex` fixes its position in the physical circuit. -/
structure WitnessInstruction where
  rowIndex : Nat
  target : Nat
  a : SparseCombination
  b : SparseCombination
deriving Repr

def WitnessInstruction.format : Format WitnessInstruction where
  encode := fun value => .array [
    .atom value.rowIndex,
    .atom value.target,
    SparseCombination.format.encode value.a,
    SparseCombination.format.encode value.b]
  decode
    | .array [.atom rowIndex, .atom target, a, b] => do
      pure ⟨rowIndex, target,
        ← SparseCombination.format.decode a,
        ← SparseCombination.format.decode b⟩
    | _ => .error "invalid witness instruction"
  decode_encode := by
    intro value
    cases value
    simp [SparseCombination.format.decode_encode] <;> rfl

/-- An assertion-only absolute sparse row. Witness rows live in the template. -/
structure SparseRow where
  rowIndex : Nat
  a : SparseCombination
  b : SparseCombination
  c : SparseCombination
deriving Repr

def SparseRow.format : Format SparseRow where
  encode := fun value => .array [
    .atom value.rowIndex,
    SparseCombination.format.encode value.a,
    SparseCombination.format.encode value.b,
    SparseCombination.format.encode value.c]
  decode
    | .array [.atom rowIndex, a, b, c] => do
      pure ⟨rowIndex, ← SparseCombination.format.decode a,
        ← SparseCombination.format.decode b,
        ← SparseCombination.format.decode c⟩
    | _ => .error "invalid sparse row"
  decode_encode := by
    intro value
    cases value
    simp [SparseCombination.format.decode_encode] <;> rfl

/-- Terminal metadata becomes present when the terminal phase is added to the
same package path. -/
structure TerminalLayout where
  rowStart : Nat
  rowCount : Nat
  runningClaims : Nat
  freshClaims : Nat
deriving Repr

def TerminalLayout.format : Format TerminalLayout where
  encode := fun value => .array [.atom value.rowStart, .atom value.rowCount,
    .atom value.runningClaims, .atom value.freshClaims]
  decode
    | .array [.atom rowStart, .atom rowCount, .atom runningClaims,
        .atom freshClaims] => .ok ⟨rowStart, rowCount, runningClaims,
          freshClaims⟩
    | _ => .error "invalid terminal layout"
  decode_encode := by
    intro value
    cases value
    rfl

/-- Lossless wire encoding of the existing symbolic witness-expression IR. -/
def encodeExpr : Expr → Value
  | .var index => .array [.atom 0, .atom index]
  | .const value => .array [.atom 1, .atom value.val]
  | .add left right => .array [.atom 2, encodeExpr left, encodeExpr right]
  | .mul left right => .array [.atom 3, encodeExpr left, encodeExpr right]

def decodeExpr : Value → Except String Expr
  | .array [.atom 0, .atom index] => .ok (.var index)
  | .array [.atom 1, .atom value] =>
      if bound : value < goldilocksModulus then
        .ok (.const ⟨value, bound⟩)
      else
        .error "noncanonical witness-expression constant"
  | .array [.atom 2, left, right] => do
      pure (.add (← decodeExpr left) (← decodeExpr right))
  | .array [.atom 3, left, right] => do
      pure (.mul (← decodeExpr left) (← decodeExpr right))
  | _ => .error "invalid witness expression"

theorem decodeExpr_encode (expression : Expr) :
    decodeExpr (encodeExpr expression) = .ok expression := by
  induction expression with
  | var index => rfl
  | const value => simp [encodeExpr, decodeExpr, value.isLt]
  | add left right leftIH rightIH =>
      simp only [encodeExpr, decodeExpr]
      rw [leftIH, rightIH]
      rfl
  | mul left right leftIH rightIH =>
      simp only [encodeExpr, decodeExpr]
      rw [leftIH, rightIH]
      rfl

def exprFormat : Format Expr where
  encode := encodeExpr
  decode := decodeExpr
  decode_encode := decodeExpr_encode

def CompactRowTemplate.format : Format CompactRowTemplate where
  encode := fun value => .array [
    .atom value.inputCount,
    .atom value.localColumnCount,
    .atom value.outputInput,
    exprFormat.encode value.outputRecipe,
    (list CompactTemplateRow.format).encode value.rows]
  decode
    | .array [.atom inputCount, .atom localColumnCount, .atom outputInput,
        outputRecipe, rows] => do
      pure ⟨inputCount, localColumnCount, outputInput,
        ← exprFormat.decode outputRecipe,
        ← (list CompactTemplateRow.format).decode rows⟩
    | _ => .error "invalid compact row template"
  decode_encode := by
    intro value
    cases value
    simp [exprFormat.decode_encode, Format.decode_encode] <;> rfl

/-- Lossless wire encoding of non-authoritative witness hints. -/
def encodeHint : Hint → Value
  | .bit source index =>
      .array [.atom 0, encodeExpr source, .atom index]
  | .inverseOrZero source => .array [.atom 1, encodeExpr source]
  | .quotientFive source => .array [.atom 2, encodeExpr source]
  | .remainderFive source => .array [.atom 3, encodeExpr source]

def decodeHint : Value → Except String Hint
  | .array [.atom 0, source, .atom index] => do
      pure (.bit (← decodeExpr source) index)
  | .array [.atom 1, source] => do
      pure (.inverseOrZero (← decodeExpr source))
  | .array [.atom 2, source] => do
      pure (.quotientFive (← decodeExpr source))
  | .array [.atom 3, source] => do
      pure (.remainderFive (← decodeExpr source))
  | _ => .error "invalid witness hint"

theorem decodeHint_encode (hint : Hint) :
    decodeHint (encodeHint hint) = .ok hint := by
  cases hint <;> simp [encodeHint, decodeHint, decodeExpr_encode]

def hintFormat : Format Hint where
  encode := encodeHint
  decode := decodeHint
  decode_encode := decodeHint_encode

def WitnessBatch.format : Format WitnessBatch where
  encode := fun value => .array [
    .atom value.start, (list exprFormat).encode value.recipes,
    (list hintFormat).encode value.hints]
  decode
    | .array [.atom start, recipes, hints] => do
      let decodedRecipes ← (list exprFormat).decode recipes
      let decodedHints ← (list hintFormat).decode hints
      pure { start := start, recipes := decodedRecipes, hints := decodedHints }
    | _ => .error "invalid witness batch"
  decode_encode := by
    intro value
    cases value
    simp [Format.decode_encode] <;> rfl

/-- The only circuit-package type accepted by the Stage 1 Rust loader. -/
structure CircuitPackage where
  schemaVersion : Nat
  profile : Profile
  poseidon : PoseidonSchedule
  layout : PhysicalLayout
  relation : CcsRelation
  permutation : PermutationTemplate
  hashChains : List HashChain
  permutationInvocations : List PermutationInvocation
  compactRowTemplates : List CompactRowTemplate
  compactRowInvocations : List CompactRowInvocation
  witnessBatches : List WitnessBatch
  witnessInstructions : List WitnessInstruction
  assertionRows : List SparseRow
  terminal : Option TerminalLayout
deriving Repr

def CircuitPackage.format : Format CircuitPackage where
  encode := fun value => .array [
    .atom value.schemaVersion,
    Profile.format.encode value.profile,
    PoseidonSchedule.format.encode value.poseidon,
    PhysicalLayout.format.encode value.layout,
    CcsRelation.format.encode value.relation,
    PermutationTemplate.format.encode value.permutation,
    (list HashChain.format).encode value.hashChains,
    (list PermutationInvocation.format).encode value.permutationInvocations,
    (list CompactRowTemplate.format).encode value.compactRowTemplates,
    (list CompactRowInvocation.format).encode value.compactRowInvocations,
    (list WitnessBatch.format).encode value.witnessBatches,
    (list WitnessInstruction.format).encode value.witnessInstructions,
    (list SparseRow.format).encode value.assertionRows,
    (option TerminalLayout.format).encode value.terminal]
  decode
    | .array [.atom schemaVersion, profile, poseidon, layout, relation,
        permutation, hashChains, permutationInvocations, compactRowTemplates,
        compactRowInvocations, witnessBatches, witnessInstructions,
        assertionRows, terminal] => do
      pure ⟨schemaVersion,
        ← Profile.format.decode profile,
        ← PoseidonSchedule.format.decode poseidon,
        ← PhysicalLayout.format.decode layout,
        ← CcsRelation.format.decode relation,
        ← PermutationTemplate.format.decode permutation,
        ← (list HashChain.format).decode hashChains,
        ← (list PermutationInvocation.format).decode permutationInvocations,
        ← (list CompactRowTemplate.format).decode compactRowTemplates,
        ← (list CompactRowInvocation.format).decode compactRowInvocations,
        ← (list WitnessBatch.format).decode witnessBatches,
        ← (list WitnessInstruction.format).decode witnessInstructions,
        ← (list SparseRow.format).decode assertionRows,
        ← (option TerminalLayout.format).decode terminal⟩
    | _ => .error "invalid circuit package"
  decode_encode := by
    intro value
    cases value
    simp [Profile.format.decode_encode, PoseidonSchedule.format.decode_encode,
      PhysicalLayout.format.decode_encode,
      CcsRelation.format.decode_encode,
      PermutationTemplate.format.decode_encode, Format.decode_encode] <;> rfl

theorem decode_encode (value : CircuitPackage) :
    CircuitPackage.format.decode (CircuitPackage.format.encode value) =
      .ok value :=
  CircuitPackage.format.decode_encode value

def limbBase : Nat := 4294967296

def prependIdentityNode (tag value : Nat) (tail : List F) : List F :=
  Poseidon2.ofNat (value / (limbBase * limbBase)) ::
    Poseidon2.ofNat ((value / limbBase) % limbBase) ::
    Poseidon2.ofNat (value % limbBase) :: Poseidon2.ofNat tag :: tail

/-- Reverse-accumulating traversal of the canonical numeric-array codec.

An atom contributes tag `0` and three base-`2^32` limbs. An array contributes
tag `1`, three limbs for its length, and then its children in order. The tags
and array lengths make the token stream prefix-free. `List.foldl` keeps the
cost linear in the encoded package size and avoids one Poseidon2 call per
codec node. -/
def valuePreimageRev : Value → List F → List F
  | .atom value, tail => prependIdentityNode 0 value tail
  | .array values, tail =>
      values.foldl (fun state child => valuePreimageRev child state)
        (prependIdentityNode 1 values.length tail)

def valuePreimage (value : Value) : List F :=
  (valuePreimageRev value []).reverse

@[simp] theorem valuePreimage_atom (value : Nat) :
    valuePreimage (.atom value) =
      [Poseidon2.ofNat 0,
       Poseidon2.ofNat (value % limbBase),
       Poseidon2.ofNat ((value / limbBase) % limbBase),
       Poseidon2.ofNat (value / (limbBase * limbBase))] := by
  simp [valuePreimage, valuePreimageRev, prependIdentityNode]

@[simp] theorem valuePreimage_emptyArray :
    valuePreimage (.array []) =
      [Poseidon2.ofNat 1, Poseidon2.ofNat 0,
       Poseidon2.ofNat 0, Poseidon2.ofNat 0] := by
  simp [valuePreimage, valuePreimageRev, prependIdentityNode]

def identityDomain : List F :=
  [78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47,
    70, 80, 114, 105, 109, 101, 47, 112, 97, 99, 107, 97, 103,
    101, 47, 118, 50]

/-- Verifier-bound Poseidon2 identity of one canonical codec value. -/
def relationIdentifierValue (value : Value) : List F :=
  Poseidon2.hash (identityDomain ++
    valuePreimage value)

/-- Verifier-bound Poseidon2 identity of every canonical package field.

Version 2 hashes one prefix-free token stream. It replaces the version-1
per-node digest tree, whose executable cost was one Poseidon2 hash for every
codec node. -/
def relationIdentifier (value : CircuitPackage) : List F :=
  relationIdentifierValue (CircuitPackage.format.encode value)

def render (value : CircuitPackage) : String :=
  (CircuitPackage.format.encode value).render

/-- Emitted package envelope. It carries no prover-claimed digest. The Rust
verifier recomputes `relationIdentifier` from this package and compares it
with its verifier-owned expected identity. -/
structure Artifact where
  package : CircuitPackage
deriving Repr

def Artifact.format : Format Artifact where
  encode := fun value => CircuitPackage.format.encode value.package
  decode := fun value => do
    pure ⟨← CircuitPackage.format.decode value⟩
  decode_encode := by
    intro value
    cases value
    simp [CircuitPackage.format.decode_encode] <;> rfl

def sealPackage (value : CircuitPackage) : Artifact where
  package := value

theorem sealPackage_package (value : CircuitPackage) :
    (sealPackage value).package = value := by
  rfl

theorem artifact_decode_encode (value : Artifact) :
    Artifact.format.decode (Artifact.format.encode value) = .ok value :=
  Artifact.format.decode_encode value

def Artifact.render (value : Artifact) : String :=
  (Artifact.format.encode value).render

end NightstreamFPrime.Export.Package
