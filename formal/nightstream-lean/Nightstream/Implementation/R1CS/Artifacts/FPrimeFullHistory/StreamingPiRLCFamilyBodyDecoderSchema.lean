/-!
Wire schema for the compact production PiRLC parity-body source decoder.

This file owns inert generated-data types only. It does not validate a source
range, a final slot, a compiler trace, or a matrix.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoderSchema

def supportedSchemaVersion : Nat := 1

inductive RawResolutionRun where
  | direct (start startStride width : Nat) (centered : Bool)
  | decompositionAlias
      (source sourceStride digit digitStride start startStride : Nat)
      (centered : Bool)
  | equalityAlias
      (source sourceStride start startStride width : Nat)
      (centered : Bool)
  | linearDefinition
  | traceEliminated
deriving DecidableEq, Repr

structure RawRun where
  sourceStart : Nat
  length : Nat
  resolution : RawResolutionRun
deriving DecidableEq, Repr

structure RawStridedRun where
  sourceStart : Nat
  count : Nat
  sourceStride : Nat
  resolution : RawResolutionRun
deriving DecidableEq, Repr

structure RawTemplateInstances where
  sourceStart : Nat
  count : Nat
  sourceStride : Nat
  finalStart : Nat
  finalStride : Nat
  referenceStart : Nat
  referenceStride : Nat
  referenceFinalStart : Nat
  referenceFinalStride : Nat
deriving DecidableEq, Repr

structure RawTemplate where
  sourceWidth : Nat
  relativeRuns : List RawRun
  instances : List RawTemplateInstances
deriving DecidableEq, Repr

structure RawArm where
  schemaVersion : Nat
  arm : Nat
  sourceStart : Nat
  sourceEnd : Nat
  finalColumns : Nat
  templates : List RawTemplate
  residualRuns : List RawStridedRun
deriving DecidableEq, Repr

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoderSchema
