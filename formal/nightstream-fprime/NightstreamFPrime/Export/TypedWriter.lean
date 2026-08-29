import NightstreamFPrime.Export.Package

/-!
Writes high-volume package records directly in the canonical numeric-array
syntax. `Package.Format` remains the schema and decode authority; this module
only avoids constructing transient `Codec.Value` trees during emission.
-/

namespace NightstreamFPrime.Export.TypedWriter

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Package

@[inline] def writeNat (handle : IO.FS.Handle) (value : Nat) : IO Unit :=
  handle.putStr (toString value)

@[inline] private def openArray (handle : IO.FS.Handle) : IO Unit :=
  writeByte handle 91

@[inline] private def closeArray (handle : IO.FS.Handle) : IO Unit :=
  writeByte handle 93

@[inline] private def separator (handle : IO.FS.Handle) : IO Unit :=
  writeByte handle 44

def writeExpr (handle : IO.FS.Handle) : Expr → IO Unit
  | .var index => do
      openArray handle
      writeNat handle 0
      separator handle
      writeNat handle index
      closeArray handle
  | .const value => do
      openArray handle
      writeNat handle 1
      separator handle
      writeNat handle value.val
      closeArray handle
  | .add left right => do
      openArray handle
      writeNat handle 2
      separator handle
      writeExpr handle left
      separator handle
      writeExpr handle right
      closeArray handle
  | .mul left right => do
      openArray handle
      writeNat handle 3
      separator handle
      writeExpr handle left
      separator handle
      writeExpr handle right
      closeArray handle

def writeHint (handle : IO.FS.Handle) : Hint → IO Unit
  | .bit source index => do
      openArray handle
      writeNat handle 0
      separator handle
      writeExpr handle source
      separator handle
      writeNat handle index
      closeArray handle
  | .inverseOrZero source => do
      openArray handle
      writeNat handle 1
      separator handle
      writeExpr handle source
      closeArray handle
  | .quotientFive source => do
      openArray handle
      writeNat handle 2
      separator handle
      writeExpr handle source
      closeArray handle
  | .remainderFive source => do
      openArray handle
      writeNat handle 3
      separator handle
      writeExpr handle source
      closeArray handle

private def writeSparseTerm (handle : IO.FS.Handle)
    (term : SparseTerm) : IO Unit := do
  openArray handle
  writeNat handle term.column
  separator handle
  writeNat handle term.coefficient
  closeArray handle

def writeSparseCombination (handle : IO.FS.Handle)
    (combination : SparseCombination) : IO Unit := do
  openArray handle
  writeNat handle combination.constant
  separator handle
  writeListWith handle (writeSparseTerm handle) combination.terms
  closeArray handle

def writeWitnessInstruction (handle : IO.FS.Handle)
    (instruction : WitnessInstruction) : IO Unit := do
  openArray handle
  writeNat handle instruction.rowIndex
  separator handle
  writeNat handle instruction.target
  separator handle
  writeSparseCombination handle instruction.a
  separator handle
  writeSparseCombination handle instruction.b
  closeArray handle

def writeSparseRow (handle : IO.FS.Handle) (row : SparseRow) : IO Unit := do
  openArray handle
  writeNat handle row.rowIndex
  separator handle
  writeSparseCombination handle row.a
  separator handle
  writeSparseCombination handle row.b
  separator handle
  writeSparseCombination handle row.c
  closeArray handle

def writeWitnessBatch (handle : IO.FS.Handle)
    (batch : WitnessBatch) : IO Unit := do
  openArray handle
  writeNat handle batch.start
  separator handle
  writeListWith handle (writeExpr handle) batch.recipes
  separator handle
  writeListWith handle (writeHint handle) batch.hints
  closeArray handle

def writePermutationInvocation (handle : IO.FS.Handle)
    (invocation : PermutationInvocation) : IO Unit := do
  openArray handle
  writeNat handle invocation.phase
  separator handle
  writeNat handle invocation.rowStart
  separator handle
  writeNat handle invocation.witnessStart
  separator handle
  writeListWith handle (writeSparseCombination handle) invocation.inputs
  closeArray handle

private def writeCompactInputRange (handle : IO.FS.Handle)
    (inputRange : CompactInputRange) : IO Unit := do
  openArray handle
  writeNat handle inputRange.inputStart
  separator handle
  writeNat handle inputRange.inputCount
  separator handle
  writeNat handle inputRange.columnStart
  separator handle
  writeNat handle inputRange.columnStride
  closeArray handle

def writeCompactRowInvocation (handle : IO.FS.Handle)
    (invocation : CompactRowInvocation) : IO Unit := do
  openArray handle
  writeNat handle invocation.phase
  separator handle
  writeNat handle invocation.templateIndex
  separator handle
  writeNat handle invocation.rowStart
  separator handle
  writeNat handle invocation.localStart
  separator handle
  writeListWith handle (writeCompactInputRange handle) invocation.inputRanges
  closeArray handle

private def writeColumnRef (handle : IO.FS.Handle) : ColumnRef → IO Unit
  | .input index => do
      openArray handle
      writeNat handle 0
      separator handle
      writeNat handle index
      closeArray handle
  | .local index => do
      openArray handle
      writeNat handle 1
      separator handle
      writeNat handle index
      closeArray handle

private def writeTemplateTerm (handle : IO.FS.Handle)
    (term : TemplateTerm) : IO Unit := do
  openArray handle
  writeColumnRef handle term.column
  separator handle
  writeNat handle term.coefficient
  closeArray handle

private def writeTemplateCombination (handle : IO.FS.Handle)
    (combination : TemplateCombination) : IO Unit := do
  openArray handle
  writeNat handle combination.constant
  separator handle
  writeListWith handle (writeTemplateTerm handle) combination.terms
  closeArray handle

def writeTemplateRow (handle : IO.FS.Handle) (row : TemplateRow) : IO Unit := do
  openArray handle
  writeNat handle row.outputLocal
  separator handle
  writeTemplateCombination handle row.a
  separator handle
  writeTemplateCombination handle row.b
  separator handle
  writeTemplateCombination handle row.c
  closeArray handle

private def writeOptionalNat (handle : IO.FS.Handle) : Option Nat → IO Unit
  | none => do
      openArray handle
      writeNat handle 0
      closeArray handle
  | some value => do
      openArray handle
      writeNat handle 1
      separator handle
      writeNat handle value
      closeArray handle

def writeCompactTemplateRow (handle : IO.FS.Handle)
    (row : CompactTemplateRow) : IO Unit := do
  openArray handle
  writeOptionalNat handle row.outputLocal
  separator handle
  writeTemplateCombination handle row.a
  separator handle
  writeTemplateCombination handle row.b
  separator handle
  writeTemplateCombination handle row.c
  closeArray handle

end NightstreamFPrime.Export.TypedWriter
