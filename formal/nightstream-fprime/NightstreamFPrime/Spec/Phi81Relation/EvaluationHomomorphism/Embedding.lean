import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingKAction

namespace NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.Embedding

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem ringKCoeff_embedChallenge
    (value : RingF) (degree : Nat) :
    ringKCoeff (RingKAction.embedChallenge value) degree =
      K.embed (ringFCoeff value degree) := by
  unfold ringKCoeff ringFCoeff RingKAction.embedChallenge
  split <;> rfl

private theorem embed_add (left right : F) :
    K.embed (left + right) = K.add (K.embed left) (K.embed right) := by
  simpa only [ConcreteCarrier.baseOps, ConcreteCarrier.extensionOps] using
    ConcreteCarrier.embed_add left right

private theorem embed_mul (left right : F) :
    K.embed (left * right) = K.mul (K.embed left) (K.embed right) := by
  simpa only [ConcreteCarrier.baseOps, ConcreteCarrier.extensionOps] using
    ConcreteCarrier.embed_mul left right

private theorem foldl_raw_embedChallenge
    (indices : List Nat) (left right : RingF) (degree : Nat) (initial : F) :
    indices.foldl
        (fun accumulated index =>
          if index <= degree ∧ degree - index < ringDegree then
            K.add accumulated
              (K.mul
                (ringKCoeff (RingKAction.embedChallenge left) index)
                (ringKCoeff (RingKAction.embedChallenge right)
                  (degree - index)))
          else accumulated)
        (K.embed initial) =
      K.embed
        (indices.foldl
          (fun accumulated index =>
            if index <= degree ∧ degree - index < ringDegree then
              accumulated +
                ringFCoeff left index * ringFCoeff right (degree - index)
            else accumulated)
          initial) := by
  induction indices generalizing initial with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      simp only [List.foldl_cons]
      by_cases active : index <= degree ∧ degree - index < ringDegree
      · simp only [ite_eq_left active, ringKCoeff_embedChallenge]
        have initialStep :
            K.add (K.embed initial)
                (K.mul
                  (K.embed (ringFCoeff left index))
                  (K.embed (ringFCoeff right (degree - index)))) =
              K.embed
                (initial +
                  ringFCoeff left index *
                    ringFCoeff right (degree - index)) := by
          rw [← embed_mul, ← embed_add]
        rw [initialStep]
        simpa only [ringKCoeff_embedChallenge, ← embed_mul] using
          inductionHypothesis
            (initial +
              ringFCoeff left index * ringFCoeff right (degree - index))
      · simp only [ite_eq_right active]
        exact inductionHypothesis initial

theorem rawMulCoeffK_embedChallenge
    (left right : RingF) (degree : Nat) :
    rawMulCoeffK
        (RingKAction.embedChallenge left)
        (RingKAction.embedChallenge right) degree =
      K.embed (rawMulCoeffF left right degree) := by
  unfold rawMulCoeffK rawMulCoeffF
  exact foldl_raw_embedChallenge (List.range ringDegree) left right degree 0

private theorem embed_sub (left right : F) :
    K.embed (left - right) = K.sub (K.embed left) (K.embed right) := by
  simp only [K.embed, K.sub, Fin.sub_self]

private theorem embed_reduction (raw folded twice : F) :
    K.embed (raw - folded + twice) =
      K.add (K.sub (K.embed raw) (K.embed folded)) (K.embed twice) := by
  rw [embed_add, embed_sub]

theorem embedChallenge_ringFMul (left right : RingF) :
    RingKAction.embedChallenge (ringFMul left right) =
      ringKMul
        (RingKAction.embedChallenge left)
        (RingKAction.embedChallenge right) := by
  funext output
  by_cases foldedLow : output.val < ringMiddleDegree
  · by_cases hasTwice :
      output.val + (ringDegree + ringMiddleDegree) <= 2 * (ringDegree - 1)
    · simp only [RingKAction.embedChallenge, ringFMul, ringKMul,
        ite_eq_left foldedLow, ite_eq_left hasTwice,
        rawMulCoeffK_embedChallenge]
      exact embed_reduction _ _ _
    · simp only [RingKAction.embedChallenge, ringFMul, ringKMul,
        ite_eq_left foldedLow, ite_eq_right hasTwice,
        rawMulCoeffK_embedChallenge]
      exact embed_reduction _ _ _
  · by_cases hasTwice :
      output.val + (ringDegree + ringMiddleDegree) <= 2 * (ringDegree - 1)
    · simp only [RingKAction.embedChallenge, ringFMul, ringKMul,
        ite_eq_right foldedLow, ite_eq_left hasTwice,
        rawMulCoeffK_embedChallenge]
      exact embed_reduction _ _ _
    · simp only [RingKAction.embedChallenge, ringFMul, ringKMul,
        ite_eq_right foldedLow, ite_eq_right hasTwice,
        rawMulCoeffK_embedChallenge]
      exact embed_reduction _ _ _

end NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.Embedding
