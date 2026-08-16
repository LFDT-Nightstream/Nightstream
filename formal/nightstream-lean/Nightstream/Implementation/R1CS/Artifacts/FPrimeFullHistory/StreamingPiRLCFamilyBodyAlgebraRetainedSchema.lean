/-!
Schema for the compact normalized PiRLC algebra retained-row scan receipt.

This file owns only inert artifact data types. It does not validate matrix
content, row semantics, assignments, selectors, or lifecycle state.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyAlgebraRetainedSchema

def supportedSchemaVersion : Nat := 1

structure RawAudit where
  schemaVersion : Nat
  sourceRows : Nat
  localColumns : Nat
  sourceColumnShift : Nat
  finalRows : Nat
  finalColumns : Nat
  selectorColumns : List Nat
  emittedStarts : List Nat
  sourceStarts : List Nat
  finalStarts : List Nat
  widths : List Nat
  radices : List Nat
  sourceNnz : List Nat
  finalPortNnz : List Nat
  deriving DecidableEq, Repr

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyAlgebraRetainedSchema
