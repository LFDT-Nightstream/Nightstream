import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuityArtifact

/-!
Contract: semantic correspondence for the complete terminal child/running
continuity owner in the supported two-step full-history profile.

The terminal NIFS consumes fourteen serialized running CE children.  The
decider emits one exact equality shard per child, linking every serialized
coordinate of the verifier-owned child to the corresponding running input.
This module proves both directions for all 215,558 emitted rows.  The semantic
conclusion is reconstructed from the equality rows; it is not stored in the
generated artifact or inferred from its range hash.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuity

/-- Exact per-child partitions, in the production terminal fold order. -/
def childPairs : List (List (Nat × Nat)) :=
  [ Generated0.pairs
  , Generated1.pairs
  , Generated2.pairs
  , Generated3.pairs
  , Generated4.pairs
  , Generated5.pairs
  , Generated6.pairs
  , Generated7.pairs
  , Generated8.pairs
  , Generated9.pairs
  , Generated10.pairs
  , Generated11.pairs
  , Generated12.pairs
  , Generated13.pairs ]

/-- Typed image of the CE-claim data carried across the terminal boundary.
The accessors below follow the production `CeClaimWires` /
`SplitNcPiCcsOutputWires` serialization order. -/
structure ClaimImage where
  serialized : List Nat
deriving DecidableEq, Repr

namespace ClaimImage

def commitment (claim : ClaimImage) : List Nat := claim.serialized.take 972
def publicMatrix (claim : ClaimImage) : List Nat :=
  (claim.serialized.drop 972).take 13878
def shape (claim : ClaimImage) : List Nat :=
  (claim.serialized.drop 14850).take 5
def evaluationPoints (claim : ClaimImage) : List Nat :=
  (claim.serialized.drop 14855).take 20
def constantTerms (claim : ClaimImage) : List Nat :=
  (claim.serialized.drop 14875).take 6
def ringEvaluations (claim : ClaimImage) : List Nat :=
  (claim.serialized.drop 14881).take 384
def ncEvaluation (claim : ClaimImage) : List Nat :=
  (claim.serialized.drop 15265).take 128
def foldDigest (claim : ClaimImage) : List Nat :=
  (claim.serialized.drop 15393).take 4

end ClaimImage

/-- Serialized verifier-child coordinates selected by one continuity shard. -/
def verifierChildValues
    (assignment : Nat → Nat) (pairs : List (Nat × Nat)) : List Nat :=
  pairs.map fun pair => assignment pair.1

/-- Serialized running-input coordinates selected by one continuity shard. -/
def runningChildValues
    (assignment : Nat → Nat) (pairs : List (Nat × Nat)) : List Nat :=
  pairs.map fun pair => assignment pair.2

def verifierChild
    (assignment : Nat → Nat) (pairs : List (Nat × Nat)) : ClaimImage :=
  ⟨verifierChildValues assignment pairs⟩

def runningChild
    (assignment : Nat → Nat) (pairs : List (Nat × Nat)) : ClaimImage :=
  ⟨runningChildValues assignment pairs⟩

/-- Semantic authority condition for the complete terminal continuity owner:
every one of the fourteen verifier-derived children is coordinate-for-
coordinate identical to the corresponding running CE input. -/
def Holds (assignment : Nat → Nat) : Prop :=
  ∀ pairs ∈ childPairs,
    verifierChild assignment pairs = runningChild assignment pairs

/-- Named CE fields transported by one child/running authority equality. -/
structure ComponentsEqual (left right : ClaimImage) : Prop where
  commitment : left.commitment = right.commitment
  publicMatrix : left.publicMatrix = right.publicMatrix
  shape : left.shape = right.shape
  evaluationPoints : left.evaluationPoints = right.evaluationPoints
  constantTerms : left.constantTerms = right.constantTerms
  ringEvaluations : left.ringEvaluations = right.ringEvaluations
  ncEvaluation : left.ncEvaluation = right.ncEvaluation
  foldDigest : left.foldDigest = right.foldDigest

theorem Holds.components
    {assignment : Nat → Nat} (holds : Holds assignment)
    {pairs : List (Nat × Nat)} (member : pairs ∈ childPairs) :
    ComponentsEqual (verifierChild assignment pairs)
      (runningChild assignment pairs) := by
  have equal := holds pairs member
  exact
    ⟨ congrArg ClaimImage.commitment equal
    , congrArg ClaimImage.publicMatrix equal
    , congrArg ClaimImage.shape equal
    , congrArg ClaimImage.evaluationPoints equal
    , congrArg ClaimImage.constantTerms equal
    , congrArg ClaimImage.ringEvaluations equal
    , congrArg ClaimImage.ncEvaluation equal
    , congrArg ClaimImage.foldDigest equal ⟩

/-- The generated per-child shards are exactly the complete owner pair list. -/
theorem pairs_partition :
    FPrimeFullHistoryTerminalContinuity.pairs = childPairs.flatten := by
  simp [FPrimeFullHistoryTerminalContinuity.pairs, childPairs]

theorem child_count : childPairs.length = 14 := by
  decide

theorem child_widths :
    childPairs.map List.length = List.replicate 14 15397 := by
  simp [childPairs, Generated0.pairs_length, Generated1.pairs_length,
    Generated2.pairs_length, Generated3.pairs_length,
    Generated4.pairs_length, Generated5.pairs_length,
    Generated6.pairs_length, Generated7.pairs_length,
    Generated8.pairs_length, Generated9.pairs_length,
    Generated10.pairs_length, Generated11.pairs_length,
    Generated12.pairs_length, Generated13.pairs_length,
    Generated0.rowCount, Generated1.rowCount, Generated2.rowCount,
    Generated3.rowCount, Generated4.rowCount, Generated5.rowCount,
    Generated6.rowCount, Generated7.rowCount, Generated8.rowCount,
    Generated9.rowCount, Generated10.rowCount, Generated11.rowCount,
    Generated12.rowCount, Generated13.rowCount]

private theorem child_values_eq_iff
    (assignment : Nat → Nat) (pairs : List (Nat × Nat)) :
    verifierChildValues assignment pairs = runningChildValues assignment pairs ↔
      ∀ pair ∈ pairs, assignment pair.1 = assignment pair.2 := by
  induction pairs with
  | nil => simp [verifierChildValues, runningChildValues]
  | cons head tail inductionHypothesis =>
      constructor
      · intro valuesEqual pair pairMember
        have split :
            assignment head.1 = assignment head.2 ∧
              verifierChildValues assignment tail =
                runningChildValues assignment tail := by
          simpa [verifierChildValues, runningChildValues] using valuesEqual
        rw [List.mem_cons] at pairMember
        rcases pairMember with pairEqual | pairMember
        · subst pair
          exact split.1
        · exact (inductionHypothesis.mp split.2) pair pairMember
      · intro pointwise
        have headEqual : assignment head.1 = assignment head.2 :=
          pointwise head (by simp)
        have tailEqual : verifierChildValues assignment tail =
            runningChildValues assignment tail := by
          apply inductionHypothesis.mpr
          intro pair pairMember
          exact pointwise pair (by simp [pairMember])
        simpa [verifierChildValues, runningChildValues, headEqual] using tailEqual

/-- Reusable compiler rule for one exact child/running equality block. -/
theorem claimImage_sound
    {assignment : Nat → Nat} {pairs : List (Nat × Nat)}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (EqualityPins.rows pairs) assignment) :
    verifierChild assignment pairs = runningChild assignment pairs := by
  have equalities := EqualityPins.rows_sound canonical one satisfies
  unfold verifierChild runningChild
  congr 1
  exact (child_values_eq_iff assignment pairs).mpr equalities

/-- Reverse compiler rule for one exact child/running equality block. -/
theorem claimImage_complete
    {assignment : Nat → Nat} {pairs : List (Nat × Nat)}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (equal : verifierChild assignment pairs = runningChild assignment pairs) :
    Satisfies (EqualityPins.rows pairs) assignment := by
  apply EqualityPins.rows_complete canonical one
  exact (child_values_eq_iff assignment pairs).mp
    (congrArg ClaimImage.serialized equal)

/-- `CIR-SOUND` for the exact 215,558-row terminal continuity owner. -/
theorem sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies FPrimeFullHistoryTerminalContinuity.rows assignment) :
    Holds assignment := by
  have equalities := FPrimeFullHistoryTerminalContinuity.sound
    canonical one satisfies
  intro child childMember
  unfold verifierChild runningChild
  congr 1
  apply (child_values_eq_iff assignment child).mpr
  intro pair pairMember
  apply equalities pair
  rw [pairs_partition]
  exact List.mem_flatten.mpr ⟨child, childMember, pairMember⟩

/-- `CIR-COMPLETE` for the same owner.  The equality family has no auxiliary
witness columns: semantic child/running identity directly satisfies every
emitted row. -/
theorem complete
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Holds assignment) :
    Satisfies FPrimeFullHistoryTerminalContinuity.rows assignment := by
  apply EqualityPins.rows_complete canonical one
  intro pair pairMember
  have flattenedMember : pair ∈ childPairs.flatten := by
    rw [← pairs_partition]
    exact pairMember
  rcases List.mem_flatten.mp flattenedMember with
    ⟨child, childMember, memberInChild⟩
  have childValuesEqual : verifierChildValues assignment child =
      runningChildValues assignment child := by
    exact congrArg ClaimImage.serialized (holds child childMember)
  exact (child_values_eq_iff assignment child).mp
    childValuesEqual pair memberInChild

/-- Exact semantic characterization of all rows in the owner. -/
theorem satisfies_iff_holds
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) :
    Satisfies FPrimeFullHistoryTerminalContinuity.rows assignment ↔
      Holds assignment :=
  ⟨sound canonical one, complete canonical one⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound
