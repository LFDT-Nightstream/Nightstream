import Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform.Experiment.BadProbability

/-! Semantic coordinate-fork loss and actual expected-query headline. -/

set_option autoImplicit false

namespace Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform

open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

universe uScalar uAnswer

variable {Scalar : Type uScalar} {Answer : Type uAnswer}

theorem finite_uniform_coordinate_forking
    (alphabet : Support Scalar)
    (coordinates : Nat)
    (valid : Scalar -> Prop)
    (verify : ChallengeVector Scalar coordinates -> Answer -> Prop)
    (oracle : Oracle Scalar Answer coordinates)
    (accepts : ChallengeVector Scalar coordinates -> Bool)
    (acceptsIff : forall challenge,
      accepts challenge = true <-> verify challenge (oracle challenge))
    (alphabetValid : forall scalar, scalar ∈ alphabet.values -> valid scalar) :
    (uniformChallenges alphabet coordinates).probability
          (fun challenge => verify challenge (oracle challenge)) -
        ratio coordinates alphabet.cardinality ≤
      (forkExperiment alphabet coordinates).probability
        (fun seed => AcceptedCoordinateFork valid verify oracle
          (run accepts seed.val).sample) ∧
    (uniformChallenges alphabet coordinates).probability
          (fun challenge => verify challenge (oracle challenge)) -
        ratio (coordinates + 1) alphabet.cardinality ≤
      (forkExperiment alphabet coordinates).probability
        (fun seed => AcceptedCoordinateFork valid verify oracle
          (run accepts seed.val).sample) ∧
    (forkExperiment alphabet coordinates).ExpectedQueriesAtMost
      (fun seed => (run accepts seed.val).trace) (coordinates + 1) := by
  have executableLower :
      (uniformChallenges alphabet coordinates).probabilityBool accepts -
          ratio coordinates alphabet.cardinality ≤
        (forkExperiment alphabet coordinates).probabilityBool
          (fun seed => (run accepts seed.val).successBool accepts) := by
    rw [← base_probability_eq_uniform alphabet accepts]
    exact (forkExperiment alphabet coordinates).probabilityBool_sub_le_of_cover
      (fun seed => accepts (decodeWord seed.val))
      (fun seed => (run accepts seed.val).successBool accepts)
      (fun seed => wordBad accepts seed.val)
      (ratio coordinates alphabet.cardinality)
      (fun seed accepted => base_success_or_wordBad accepts seed.val accepted)
      (bad_probability_le_sharp alphabet accepts)
  rw [← (uniformChallenges alphabet coordinates).probability_bool_event
    accepts] at executableLower
  have uniformEvent :
      (fun challenge => accepts challenge = true) =
        (fun challenge => verify challenge (oracle challenge)) := by
    funext challenge
    exact propext (acceptsIff challenge)
  rw [uniformEvent] at executableLower
  have semanticMonotone :
      (forkExperiment alphabet coordinates).probability
          (fun seed => (run accepts seed.val).successBool accepts = true) ≤
        (forkExperiment alphabet coordinates).probability
          (fun seed => AcceptedCoordinateFork valid verify oracle
            (run accepts seed.val).sample) := by
    apply Experiment.probability_mono
    intro seed success
    exact successBool_implies_acceptedCoordinateFork alphabet valid verify
      oracle accepts acceptsIff alphabetValid seed.property success
  rw [(forkExperiment alphabet coordinates).probability_bool_event
    (fun seed => (run accepts seed.val).successBool accepts)] at semanticMonotone
  have sharp := Rat.le_trans executableLower semanticMonotone
  have lossOrdered :
      ratio coordinates alphabet.cardinality ≤
        ratio (coordinates + 1) alphabet.cardinality := by
    unfold ratio
    apply div_le_div_of_le
    · exact Rat.natCast_le_natCast.mpr (Nat.le_succ coordinates)
    · exact Rat.natCast_pos.mpr alphabet.cardinality_pos
  have shifted :
      (uniformChallenges alphabet coordinates).probability
            (fun challenge => verify challenge (oracle challenge)) -
          ratio (coordinates + 1) alphabet.cardinality ≤
        (uniformChallenges alphabet coordinates).probability
            (fun challenge => verify challenge (oracle challenge)) -
          ratio coordinates alphabet.cardinality := by
    simpa [Rat.sub_eq_add_neg] using
      (Rat.add_le_add_left
        (c := (uniformChallenges alphabet coordinates).probability
          (fun challenge => verify challenge (oracle challenge)))).mpr
        (Rat.neg_le_neg lossOrdered)
  exact ⟨sharp, Rat.le_trans shifted sharp,
    expected_queries_at_most alphabet accepts⟩

end Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform
