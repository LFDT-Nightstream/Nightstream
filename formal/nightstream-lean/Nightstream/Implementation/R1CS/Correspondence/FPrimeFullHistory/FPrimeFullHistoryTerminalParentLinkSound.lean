import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Links
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryTerminalContinuitySound

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

set_option maxRecDepth 1048576

namespace Artifact

abbrev pairs := FPrimeFullHistoryTerminalParentLink.pairs
abbrev rows := FPrimeFullHistoryTerminalParentLink.rows

end Artifact

/-- Full parent-authority image.  Unlike terminal child/running continuity,
the parent link retains the 128-coordinate `y_zcol` projection. -/
structure ParentAuthorityImage where
  serialized : List Nat
deriving DecidableEq, Repr

namespace ParentAuthorityImage

def commitment (claim : ParentAuthorityImage) : List Nat :=
  claim.serialized.take 972
def publicMatrix (claim : ParentAuthorityImage) : List Nat :=
  (claim.serialized.drop 972).take 13878
def shape (claim : ParentAuthorityImage) : List Nat :=
  (claim.serialized.drop 14850).take 5
def evaluationPoints (claim : ParentAuthorityImage) : List Nat :=
  (claim.serialized.drop 14855).take 20
def constantTerms (claim : ParentAuthorityImage) : List Nat :=
  (claim.serialized.drop 14875).take 6
def ringEvaluations (claim : ParentAuthorityImage) : List Nat :=
  (claim.serialized.drop 14881).take 384
def foldDigest (claim : ParentAuthorityImage) : List Nat :=
  (claim.serialized.drop 15265).take 4
def yZcol (claim : ParentAuthorityImage) : List Nat :=
  (claim.serialized.drop 15269).take 128

end ParentAuthorityImage

def previousParent (assignment : Nat → Nat) : ParentAuthorityImage :=
  ⟨verifierChildValues assignment Artifact.pairs⟩

def currentParent (assignment : Nat → Nat) : ParentAuthorityImage :=
  ⟨runningChildValues assignment Artifact.pairs⟩

/-- The verifier-derived previous parent is exactly the current accumulator
parent, coordinate for coordinate in the production CE serialization. -/
def Holds (assignment : Nat → Nat) : Prop :=
  previousParent assignment = currentParent assignment

/-- Every exact parent-link row derives the typed parent-authority identity. -/
theorem sound {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies Artifact.rows assignment) :
    Holds assignment := by
  unfold Holds previousParent currentParent
  congr 1
  exact (child_values_eq_iff assignment Artifact.pairs).mpr
    (EqualityPins.rows_sound canonical one satisfies)

/-- Parent-authority identity directly satisfies every exact equality row;
this owner has no auxiliary witness columns. -/
theorem complete {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Holds assignment) :
    Satisfies Artifact.rows assignment := by
  apply EqualityPins.rows_complete canonical one
  exact (child_values_eq_iff assignment Artifact.pairs).mp
    (congrArg ParentAuthorityImage.serialized holds)

theorem satisfies_iff_holds {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) :
    Satisfies Artifact.rows assignment ↔ Holds assignment :=
  ⟨sound canonical one, complete canonical one⟩

/-- Named components of the full parent serialization all
cross the parent boundary unchanged. -/
structure ParentComponentsEqual (left right : ParentAuthorityImage) : Prop where
  commitment : left.commitment = right.commitment
  publicMatrix : left.publicMatrix = right.publicMatrix
  shape : left.shape = right.shape
  evaluationPoints : left.evaluationPoints = right.evaluationPoints
  constantTerms : left.constantTerms = right.constantTerms
  ringEvaluations : left.ringEvaluations = right.ringEvaluations
  foldDigest : left.foldDigest = right.foldDigest
  yZcol : left.yZcol = right.yZcol

theorem components {assignment : Nat → Nat}
    (holds : Holds assignment) :
    ParentComponentsEqual
      (previousParent assignment)
      (currentParent assignment) := by
  have equal := holds
  exact
    ⟨ congrArg ParentAuthorityImage.commitment equal
    , congrArg ParentAuthorityImage.publicMatrix equal
    , congrArg ParentAuthorityImage.shape equal
    , congrArg ParentAuthorityImage.evaluationPoints equal
    , congrArg ParentAuthorityImage.constantTerms equal
    , congrArg ParentAuthorityImage.ringEvaluations equal
    , congrArg ParentAuthorityImage.foldDigest equal
    , congrArg ParentAuthorityImage.yZcol equal ⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalParentLinkSound
