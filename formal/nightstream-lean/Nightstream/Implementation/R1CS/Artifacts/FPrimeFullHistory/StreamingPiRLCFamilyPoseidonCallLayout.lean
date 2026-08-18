import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout

/-!
Artifact facade for the exact production PiRLC replay-call layout.

Owns the stable handwritten import boundary for four compact Rust-emitted
call runs. It does not own Poseidon2 semantics, row satisfaction, lifecycle
authority, or permission to remove constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout

abbrev RawScope :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout.RawScope
abbrev RawFirstClass :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout.RawFirstClass
abbrev RawRun :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout.RawRun

def schemaVersion : Nat :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout.schemaVersion
def rowTemplateSource : String :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout.rowTemplateSource
def sourceCallStride : Nat :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout.sourceCallStride
def emittedCallRows : Nat :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout.emittedCallRows
def slotWidth : Nat :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout.slotWidth
def localFinalStride : Nat :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout.localFinalStride
def evenSourceRows : Nat :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout.evenSourceRows
def evenSourceColumns : Nat :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout.evenSourceColumns
def oddSourceRows : Nat :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout.oddSourceRows
def oddSourceColumns : Nat :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout.oddSourceColumns
def finalRows : Nat :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout.finalRows
def finalColumns : Nat :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout.finalColumns

def rawRun0 : RawRun :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout.rawRun0
def rawRun1 : RawRun :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout.rawRun1
def rawRun2 : RawRun :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout.rawRun2
def rawRun3 : RawRun :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout.rawRun3
def rawRuns : List RawRun :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallLayout.rawRuns

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout
