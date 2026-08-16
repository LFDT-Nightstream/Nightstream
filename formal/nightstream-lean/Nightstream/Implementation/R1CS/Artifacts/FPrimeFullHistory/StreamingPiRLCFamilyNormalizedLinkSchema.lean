/-!
Schema for the compact normalized PiRLC body-overlay link receipt.

This file owns inert artifact data only. It does not validate source or final
slots, equality-row semantics, selectors, assignments, or lifecycle state.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyNormalizedLinkSchema

def supportedSchemaVersion : Nat := 1

structure RawRun where
  bodySourceStart : Nat
  overlaySourceStart : Nat
  outerCount : Nat
  bodySourceStride : Nat
  overlaySourceStride : Nat
  fieldCount : Nat
  bodyFinalStart : Nat
  overlayFinalStart : Nat
  finalOuterStride : Nat
  finalFieldStride : Nat
  width : Nat
  radix : Nat
deriving DecidableEq, Repr

structure RawAudit where
  schemaVersion : Nat
  familyCount : Nat
  parityCount : Nat
  publicOutputCount : Nat
  bodyFinalColumns : Nat
  overlayFinalColumns : Nat
  linkCountPerFamily : Nat
  totalLinkCount : Nat
  phaseKinds : List Nat
  runs : List RawRun
deriving DecidableEq, Repr

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyNormalizedLinkSchema
