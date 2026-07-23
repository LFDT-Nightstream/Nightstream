import Nightstream.SuperNeo.InteractiveReduction.CoordinateForking

/-!
Kernel counterexample to the denominator rendered in SuperNeo Appendix C,
Theorem 10.

Owns: one exact finite challenge space with alphabet cardinality three and two
coordinates; its explicit `Fin 9` codec; the three accepting challenges; and
the proof that none can be the base of an accepted coordinate fork.

Does not own: the corrected finite-uniform coordinate-fork theorem, the
Appendix-D.5 extractor, `Pi_RLC`, commitment binding, Fiat--Shamir, Rust,
R1CS, or costs.

The rendered loss `ell / |C^ell|` is `2 / 9` here.  The adversary succeeds on
exactly `3 / 9` challenges, so that rendering predicts the strictly positive
lower bound `(3 - 2) / 9`.  Nevertheless, acceptance fixes coordinate zero to
zero, making the required distinct coordinate-zero fork impossible.
-/

namespace Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.RenderedBoundObstruction

def challengeAlphabetCardinality : Nat := 3

def coordinateCount : Nat := 2

def challengeSpaceCardinality : Nat :=
  challengeAlphabetCardinality ^ coordinateCount

abbrev Scalar := Fin challengeAlphabetCardinality

abbrev Challenge := ChallengeVector Scalar coordinateCount

def coordinateZero : Fin coordinateCount := ⟨0, by decide⟩

def coordinateOne : Fin coordinateCount := ⟨1, by decide⟩

def scalarZero : Scalar := ⟨0, by decide⟩

/-! ## Exact nine-point challenge codec -/

/-- Quotient/remainder decoding of `Fin 9` into two base-three digits. -/
def challengeOfIndex (index : Fin challengeSpaceCardinality) : Challenge :=
  fun coordinate =>
    if coordinate.val = 0 then
      ⟨index.val / challengeAlphabetCardinality, by
        have indexLt : index.val < 9 := by
          simpa [challengeSpaceCardinality, challengeAlphabetCardinality,
            coordinateCount] using index.isLt
        change index.val / 3 < 3
        omega⟩
    else
      ⟨index.val % challengeAlphabetCardinality,
        Nat.mod_lt _ (by decide)⟩

/-- Base-three encoding of a two-coordinate challenge. -/
def challengeIndex (challenge : Challenge) :
    Fin challengeSpaceCardinality :=
  ⟨(challenge coordinateZero).val * challengeAlphabetCardinality +
      (challenge coordinateOne).val, by
    have firstLt : (challenge coordinateZero).val < 3 := by
      simpa [challengeAlphabetCardinality] using
        (challenge coordinateZero).isLt
    have secondLt : (challenge coordinateOne).val < 3 := by
      simpa [challengeAlphabetCardinality] using
        (challenge coordinateOne).isLt
    change
      (challenge coordinateZero).val * 3 +
        (challenge coordinateOne).val < 9
    omega⟩

theorem challengeOfIndex_challengeIndex (challenge : Challenge) :
    challengeOfIndex (challengeIndex challenge) = challenge := by
  funext coordinate
  have coordinateLt : coordinate.val < 2 := by
    simpa [coordinateCount] using coordinate.isLt
  by_cases isZero : coordinate.val = 0
  · have coordinateEq : coordinate = coordinateZero := by
      apply Fin.ext
      simpa [coordinateZero] using isZero
    subst coordinate
    apply Fin.ext
    change
      ((challenge coordinateZero).val * 3 +
          (challenge coordinateOne).val) / 3 =
        (challenge coordinateZero).val
    have secondLt : (challenge coordinateOne).val < 3 := by
      simpa [challengeAlphabetCardinality] using
        (challenge coordinateOne).isLt
    omega
  · have coordinateIsOne : coordinate.val = 1 := by
      omega
    have coordinateEq : coordinate = coordinateOne := by
      apply Fin.ext
      simpa [coordinateOne] using coordinateIsOne
    subst coordinate
    apply Fin.ext
    change
      ((challenge coordinateZero).val * 3 +
          (challenge coordinateOne).val) % 3 =
        (challenge coordinateOne).val
    have secondLt : (challenge coordinateOne).val < 3 := by
      simpa [challengeAlphabetCardinality] using
        (challenge coordinateOne).isLt
    omega

theorem challengeIndex_challengeOfIndex
    (index : Fin challengeSpaceCardinality) :
    challengeIndex (challengeOfIndex index) = index := by
  apply Fin.ext
  change index.val / 3 * 3 + index.val % 3 = index.val
  simpa [Nat.mul_comm] using Nat.div_add_mod index.val 3

theorem challengeOfIndex_injective : Function.Injective challengeOfIndex := by
  intro left right equal
  have encodedEqual := congrArg challengeIndex equal
  simpa only [challengeIndex_challengeOfIndex] using encodedEqual

theorem challengeOfIndex_surjective : Function.Surjective challengeOfIndex := by
  intro challenge
  exact ⟨challengeIndex challenge, challengeOfIndex_challengeIndex challenge⟩

/-! ## The accepting set and the impossible fork -/

def valid (_ : Scalar) : Prop := True

def oracle (_ : Challenge) : Unit := ()

def verifies (challenge : Challenge) (_ : Unit) : Prop :=
  challenge coordinateZero = scalarZero

def verifiesBool (challenge : Challenge) : Bool :=
  decide ((challenge coordinateZero).val = 0)

theorem verifiesBool_eq_true_iff (challenge : Challenge) :
    verifiesBool challenge = true ↔ verifies challenge () := by
  simp only [verifiesBool, decide_eq_true_eq, verifies]
  constructor
  · intro valueZero
    apply Fin.ext
    exact valueZero
  · intro equalZero
    exact congrArg Fin.val equalZero

def allChallengeIndices : List (Fin challengeSpaceCardinality) :=
  List.ofFn fun index => index

def acceptedChallengeCount : Nat :=
  (allChallengeIndices.filter fun index =>
    verifiesBool (challengeOfIndex index)).length

theorem acceptedChallengeCount_eq_three : acceptedChallengeCount = 3 := by
  decide

/-- Acceptance fixes coordinate zero to zero in both the base and its named
coordinate-zero fork, contradicting the required changed-coordinate fact. -/
theorem noAcceptedCoordinateFork (sample : ForkSample Scalar coordinateCount) :
    ¬ AcceptedCoordinateFork valid verifies oracle sample := by
  intro accepted
  have baseZero : sample.base coordinateZero = scalarZero := by
    simpa [verifies, oracle] using accepted.baseAccepted
  have forkZero :
      sample.forks coordinateZero coordinateZero = scalarZero := by
    simpa [verifies, oracle] using accepted.forkAccepted coordinateZero
  exact accepted.changed coordinateZero (baseZero.trans forkZero.symm)

/-! ## Rendered positive lower numerator -/

/-- In the rendered theorem the loss numerator is `ell`, while its
denominator is the complete challenge-space cardinality. -/
def renderedLossNumerator : Nat := coordinateCount

/-- Both success and the rendered loss have denominator nine in this finite
instance, so subtraction happens directly on their numerators. -/
def renderedClaimedLowerNumerator : Nat :=
  acceptedChallengeCount - renderedLossNumerator

theorem renderedClaimedLowerNumerator_eq_one :
    renderedClaimedLowerNumerator = 1 := by
  decide

/-- The exact finite counterexample: the rendered bound predicts positive
fork success, while the special-set event is empty. -/
theorem rendered_denominator_bound_counterexample :
    (forall sample : ForkSample (Fin 3) 2,
      ¬ AcceptedCoordinateFork
        (fun _ : Fin 3 => True)
        (fun challenge (_ : Unit) =>
          challenge ⟨0, by decide⟩ = ⟨0, by decide⟩)
        (fun _ => ()) sample) ∧
      challengeAlphabetCardinality = 3 ∧
      coordinateCount = 2 ∧
      challengeSpaceCardinality = 9 ∧
      acceptedChallengeCount = 3 ∧
      renderedLossNumerator = 2 ∧
      renderedClaimedLowerNumerator = 1 ∧
      0 < renderedClaimedLowerNumerator := by
  refine ⟨?_, rfl, rfl, by decide, acceptedChallengeCount_eq_three,
    rfl, renderedClaimedLowerNumerator_eq_one, ?_⟩
  · intro sample
    simpa [valid, verifies, oracle, coordinateZero, scalarZero,
      challengeAlphabetCardinality, coordinateCount] using
      noAcceptedCoordinateFork sample
  · rw [renderedClaimedLowerNumerator_eq_one]
    decide

end Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.RenderedBoundObstruction
