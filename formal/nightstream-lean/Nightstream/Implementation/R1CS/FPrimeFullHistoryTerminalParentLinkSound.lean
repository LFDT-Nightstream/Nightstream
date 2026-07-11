import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalParentLinkArtifact
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound

/-!
Contract: semantic correspondence for the exact terminal parent-authority
link in the supported two-step full-history profile.

The terminal decider re-serializes the previous accumulator parent and links
all 15,397 coordinates to the current terminal accumulator input.  The
semantic claim below is reconstructed from those equality rows; neither side
is represented by an accepted flag or an artifact hash.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalParentLinkSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound

namespace Artifact

abbrev pairs := FPrimeFullHistoryTerminalParentLink.pairs
abbrev rows := FPrimeFullHistoryTerminalParentLink.rows

end Artifact

/-- The verifier-derived previous parent is exactly the current accumulator
parent, coordinate for coordinate in the production CE serialization. -/
def Holds (assignment : Nat → Nat) : Prop :=
  verifierChild assignment Artifact.pairs =
    runningChild assignment Artifact.pairs

/-- Every exact parent-link row derives the typed parent-authority identity. -/
theorem sound {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies Artifact.rows assignment) :
    Holds assignment :=
  claimImage_sound canonical one satisfies

/-- Parent-authority identity directly satisfies every exact equality row;
this owner has no auxiliary witness columns. -/
theorem complete {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Holds assignment) :
    Satisfies Artifact.rows assignment :=
  claimImage_complete canonical one holds

theorem satisfies_iff_holds {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) :
    Satisfies Artifact.rows assignment ↔ Holds assignment :=
  ⟨sound canonical one, complete canonical one⟩

/-- The named CE components inherited from the common serialization all
cross the parent boundary unchanged. -/
theorem components {assignment : Nat → Nat}
    (holds : Holds assignment) :
    ComponentsEqual
      (verifierChild assignment Artifact.pairs)
      (runningChild assignment Artifact.pairs) := by
  have equal := holds
  exact
    ⟨ congrArg ClaimImage.commitment equal
    , congrArg ClaimImage.publicMatrix equal
    , congrArg ClaimImage.shape equal
    , congrArg ClaimImage.evaluationPoints equal
    , congrArg ClaimImage.constantTerms equal
    , congrArg ClaimImage.ringEvaluations equal
    , congrArg ClaimImage.ncEvaluation equal
    , congrArg ClaimImage.foldDigest equal ⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalParentLinkSound
