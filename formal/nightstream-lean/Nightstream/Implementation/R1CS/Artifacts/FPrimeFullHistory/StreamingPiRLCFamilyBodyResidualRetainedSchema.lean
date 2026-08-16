/-!
Schema for the compact normalized PiRLC residual retained-row scan receipt.

This file owns inert artifact data only. It does not validate matrix content,
row semantics, assignments, selectors, the local commitment, or lifecycle
state.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyResidualRetainedSchema

def supportedSchemaVersion : Nat := 1

structure RawAudit where
  schemaVersion : Nat
  sourceRowStart : Nat
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

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyResidualRetainedSchema
