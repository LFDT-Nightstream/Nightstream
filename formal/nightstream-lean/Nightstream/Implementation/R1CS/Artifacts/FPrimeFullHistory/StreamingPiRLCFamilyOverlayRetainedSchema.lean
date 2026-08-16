/-!
Schema for the compact normalized PiRLC family-overlay retained-row receipt.

This file owns inert artifact data only. It does not validate matrix content,
seed expansion, row semantics, assignments, links, selectors, or lifecycle
state.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyOverlayRetainedSchema

def supportedSchemaVersion : Nat := 1

structure RawAudit where
  schemaVersion : Nat
  familyCount : Nat
  sourceRows : Nat
  sourceColumns : Nat
  finalRows : Nat
  finalColumns : Nat
  selectorStart : Nat
  selectorCount : Nat
  retainedStart : Nat
  retainedStride : Nat
  sourceStarts : List Nat
  finalStarts : List Nat
  widths : List Nat
  radices : List Nat
  chunkSize : Nat
  chunkSeedsByRow : List (List (List Nat))
  sourceExplicitNnz : List Nat
  finalBlockCounts : List Nat
  finalExplicitPortNnz : List Nat
deriving DecidableEq, Repr

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyOverlayRetainedSchema
