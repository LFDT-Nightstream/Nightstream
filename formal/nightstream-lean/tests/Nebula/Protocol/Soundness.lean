import Nightstream.Protocol.Nebula

set_option autoImplicit false

namespace tests.NebulaSoundness

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ApplicationTrace
open Nightstream.Protocol.Nebula.Completion
open Nightstream.Protocol.Nebula.Memory
open Nightstream.Protocol.Nebula.Ports
open Nightstream.Protocol.Nebula.Soundness

def zeroImage : Fin scannedCells → Nat := fun _ => 0

def zeroImageInRange : Snapshot.ImageInRange zeroImage := by
  intro _
  change 0 < valueLimit
  decide

def zeroSnapshot : Snapshot := Snapshot.ofImage zeroImage

def zeroSnapshotValid : zeroSnapshot.ValidAt 0 :=
  Snapshot.ofImage_validAt_zero zeroImageInRange

def emptySegment : ValidSegment zeroSnapshot zeroSnapshot 0 [] 0 where
  initialValid := zeroSnapshotValid
  finalValid := zeroSnapshotValid
  ordered := .nil 0
  balanced := by simp [Balanced, readTuples, writeTuples]

def oneSegmentChain :
    ValidChain zeroSnapshot 0 [[]] zeroSnapshot 0 :=
  .cons emptySegment (.nil zeroSnapshot 0)

def result : ExecutionResult Unit Nat :=
  { realApplicationRowCount := 1
    finalApplicationState := ()
    outcome := .returned none
    finalMemoryRoot := 9 }

def rows : List RowKind := canonicalRows result 1

def completion : ValidCompletedTrace result 1 rows where
  segmentCountPositive := by decide
  segmentCountBound := by decide
  realRowCountPositive := by decide
  realRowCountBound := by decide
  fitsDeclaredSegments := by decide
  smallestSegmentCount := by decide
  rowsCanonical := rfl

def statement : PublicStatement Unit Unit Nat where
  identity :=
    { profile := Profile.v2
      verifierKey :=
        { digest := 1
          relationManifestDigest := 2
          laneLayoutDigest := 3
          setupManifestDigest := 4
          transcriptManifestDigest := 5
          codecManifestDigest := 6
          terminalManifestDigest := 7
          applicationStateSchemaDigest := 8 }
      applicationRelationDigest := 9
      programDigest := 10
      memoryPlanDigest := 11 }
  program := ()
  initialApplicationState := ()
  initialImage := zeroImage
  initialImageInRange := zeroImageInRange
  segmentCount := 1
  finalGlobalTimestamp := 0
  expectedResult := result

def differentIdentity : StatementIdentity Nat :=
  { statement.identity with
      verifierKey := { statement.identity.verifierKey with digest := 12 } }

theorem verifier_key_identity_is_authoritative :
    differentIdentity ≠ statement.identity := by
  intro equal
  have digestEqual := congrArg
    (fun identity => identity.verifierKey.digest) equal
  change 12 = 1 at digestEqual
  omega

theorem statement_identity_encoding_is_injective :
    Function.Injective
      (StatementIdentity.encode :
        StatementIdentity Nat → List (IdentityAtom Nat)) :=
  StatementIdentity.encode_injective

def snapshotRoot (_snapshot : Snapshot) : Nat := 9

def applicationSemantics : ApplicationTrace.Semantics Unit Unit where
  active := fun _ _ _ _ => False
  returned := fun _ before _ output after =>
    output = none ∧ after = before
  trapped := fun _ _ _ _ _ => False

def realApplicationExecution :
    ApplicationTrace.RealExecution applicationSemantics () () ()
      (.returned none) where
  activeRows := []
  beforeTerminal := ()
  activeTrace := .nil ()
  terminalRow := NormalizedRow.inactive
  terminal := .returned none ⟨rfl, rfl⟩

def completedApplicationExecution :
    ApplicationTrace.CompletedExecution applicationSemantics () () result 1 where
  real := realApplicationExecution
  realRowCountExact := rfl
  segmentCountPositive := by decide
  segmentCountBound := by decide
  realRowCountBound := by decide
  fitsDeclaredSegments := by decide
  smallestSegmentCount := by decide

theorem completedApplicationCoverage :
    completedApplicationExecution.CoversMemory [[]] := by
  simp [ApplicationTrace.CompletedExecution.CoversMemory,
    ApplicationTrace.CompletedExecution.segmentAccesses,
    ApplicationTrace.CompletedExecution.fixedSegmentRows,
    ApplicationTrace.RealExecution.rows, NormalizedRow.inactive,
    NormalizedRow.accesses, Completion.applicationRowsPerSegment,
    Completion.applicationRowsPerClaim, Lifecycle.claimsPerSegment,
    completedApplicationExecution, realApplicationExecution]

theorem honest_semantics_has_sound_execution :
    HasSoundExecution applicationSemantics statement snapshotRoot := by
  exact
    ⟨zeroSnapshot, [[]], completedApplicationExecution,
      oneSegmentChain.executes, rfl, completedApplicationCoverage, rfl⟩

def noApplicationExecution : ApplicationTrace.Semantics Unit Unit where
  active := fun _ _ _ _ => False
  returned := fun _ _ _ _ _ => False
  trapped := fun _ _ _ _ _ => False

/- Memory balance and terminal row form alone cannot establish application
execution. The top-level conclusion keeps this independent premise. -/
theorem application_semantics_cannot_be_omitted :
    ¬ HasSoundExecution noApplicationExecution statement snapshotRoot := by
  rintro ⟨_finalSnapshot, _segmentAccesses, applicationExecution,
    _memoryExecution, _segmentCount, _coverage, _finalRoot⟩
  cases applicationExecution.real.terminal with
  | returned _ impossible => exact impossible

end tests.NebulaSoundness
