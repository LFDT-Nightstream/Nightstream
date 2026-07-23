import Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform

/-! Finite support, probability, and expected-cost adapter for the concrete
coordinate-fork runner. The imported module owns the runner and counting. -/

set_option autoImplicit false

namespace Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform

open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords

universe uScalar uAnswer

variable {Scalar : Type uScalar} {Answer : Type uAnswer} {coordinates : Nat}

def pointWordSupport
    (alphabet : Support Scalar)
    (coordinates : Nat) : Support (PointWord Scalar coordinates) :=
  Support.challengeVectors (cyclicPointSupport alphabet) coordinates

abbrev ForkSeed
    (alphabet : Support Scalar)
    (coordinates : Nat) :=
  { word // word ∈ (pointWordSupport alphabet coordinates).values }

def forkSeedSupport
    (alphabet : Support Scalar)
    (coordinates : Nat) : Support (ForkSeed alphabet coordinates) where
  values := (pointWordSupport alphabet coordinates).values.attach
  nodup := by
    have mappedNodup :
        (((pointWordSupport alphabet coordinates).values.attach).map
          Subtype.val).Nodup := by
      rw [List.attach_map_subtype_val]
      exact (pointWordSupport alphabet coordinates).nodup
    exact List.Pairwise.of_map Subtype.val
      (fun _ _ valuesDistinct equal =>
        valuesDistinct (congrArg Subtype.val equal)) mappedNodup
  nonempty := List.attach_ne_nil_iff.mpr
    (pointWordSupport alphabet coordinates).nonempty

def uniformChallenges
    (alphabet : Support Scalar)
    (coordinates : Nat) :
    Experiment (ChallengeVector Scalar coordinates) :=
  (Support.challengeVectors alphabet coordinates).uniform

def forkExperiment
    (alphabet : Support Scalar)
    (coordinates : Nat) : Experiment (ForkSeed alphabet coordinates) :=
  (forkSeedSupport alphabet coordinates).uniform

private theorem decode_vectors
    (points : List (CyclicPoint Scalar))
    (coordinates : Nat) :
    (vectors points coordinates).map decodeWord =
      vectors (points.map CyclicPoint.value) coordinates := by
  induction coordinates with
  | zero =>
      simp only [vectors, List.map_singleton]
      congr 1
      funext index
      exact Fin.elim0 index
  | succ coordinates inductionHypothesis =>
      simp only [vectors, List.map_flatMap, List.map_map,
        Function.comp_def, decodeWord_prepend, List.flatMap_map]
      apply congrArg (fun mapping => points.flatMap mapping)
      funext point
      calc
        (vectors points coordinates).map
              (fun tail => prepend point.value (decodeWord tail)) =
            ((vectors points coordinates).map decodeWord).map
              (prepend point.value) := by
          simpa only [List.map_map, Function.comp_def]
        _ = (vectors (points.map CyclicPoint.value) coordinates).map
              (prepend point.value) := by
          rw [inductionHypothesis]

theorem base_probability_eq_uniform
    (alphabet : Support Scalar)
    (accepts : ChallengeVector Scalar coordinates -> Bool) :
    (forkExperiment alphabet coordinates).probabilityBool
        (fun seed => accepts (decodeWord seed.val)) =
      (uniformChallenges alphabet coordinates).probabilityBool accepts := by
  let words := (pointWordSupport alphabet coordinates).values
  have attachedCount :
      words.attach.countP
          (fun seed => accepts (decodeWord seed.val)) =
        words.countP (fun word => accepts (decodeWord word)) := by
    exact List.countP_attach
      (l := words) (p := fun word => accepts (decodeWord word))
  have decodedWords :
      words.map decodeWord = vectors alphabet.values coordinates := by
    dsimp [words, pointWordSupport]
    rw [decode_vectors]
    change vectors ((cyclicPoints alphabet.values).map CyclicPoint.value)
        coordinates = vectors alphabet.values coordinates
    rw [cyclicPoints_values]
  have baseCount :
      words.attach.countP
          (fun seed => accepts (decodeWord seed.val)) =
        (vectors alphabet.values coordinates).countP accepts := by
    rw [attachedCount]
    calc
      _ = (words.map decodeWord).countP accepts := by
        symm
        simpa only [Function.comp_apply] using
          (List.countP_map
            (l := words) (f := decodeWord) (p := accepts))
      _ = _ := by rw [decodedWords]
  unfold Experiment.probabilityBool Experiment.countBool forkExperiment
    forkSeedSupport uniformChallenges Support.uniform
  simp only [Support.challengeVectors_values, Support.cardinality,
    List.length_attach, id_eq]
  rw [baseCount]
  have receiptLength :
      (cyclicPointSupport alphabet).values.length = alphabet.values.length :=
    cyclicPoints_length alphabet.values
  have pointWordLength :
      (pointWordSupport alphabet coordinates).values.length =
        alphabet.values.length ^ coordinates := by
    rw [show (pointWordSupport alphabet coordinates).values =
      vectors (cyclicPointSupport alphabet).values coordinates by rfl,
      vectors_length, receiptLength]
  rw [pointWordLength, vectors_length]

theorem expected_queries_at_most
    (alphabet : Support Scalar)
    (accepts : ChallengeVector Scalar coordinates -> Bool) :
    (forkExperiment alphabet coordinates).ExpectedQueriesAtMost
      (fun seed => (run accepts seed.val).trace) (coordinates + 1) := by
  have traceMap :
      (pointWordSupport alphabet coordinates).values.attach.map
          (fun seed => (run accepts seed.val).trace.length) =
        (pointWordSupport alphabet coordinates).values.map
          (fun word => (run accepts word).trace.length) := by
    simpa only using
      (List.attach_map_val
        (l := (pointWordSupport alphabet coordinates).values)
        (f := fun word => (run accepts word).trace.length))
  unfold Experiment.ExpectedQueriesAtMost Experiment.ExpectedCostAtMost
    Experiment.totalCost forkExperiment forkSeedSupport
  simp only [id_eq]
  change
    ((pointWordSupport alphabet coordinates).values.attach.map
      (fun seed => (run accepts seed.val).trace.length)).sum ≤
      (coordinates + 1) *
        (pointWordSupport alphabet coordinates).values.attach.length
  rw [traceMap]
  simp only [List.map_map, Function.comp_def, Support.cardinality,
    List.length_attach]
  rw [show (pointWordSupport alphabet coordinates).values =
    vectors (cyclicPointSupport alphabet).values coordinates by rfl,
    vectors_length]
  have receipt :
      (cyclicPointSupport alphabet).values.length = alphabet.cardinality := by
    simpa [Support.cardinality] using
      cyclicPointSupport_cardinality alphabet
  rw [receipt]
  exact total_trace_length_le alphabet accepts

end Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform
