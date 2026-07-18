import Nightstream.Implementation.R1CS.Correspondence.Projection.Phi81.Carrier
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLCFinite

/-!
Profile-neutral pairing of two Phi81 base rings into one typed `RingK` value.

Assurance tier: model-level. No profile, generated artifact, column layout, or
R1CS row is imported or accepted as authority.

| Stage family | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `projection.pair.action` | an embedded `RingF` challenge acts on both limbs | derived | `pairRingF_action` |
| `projection.pair.combine` | two list folds equal one typed `RingK` fold | derived | `pairRings_phi81Combine` |

Owns: the mathematical two-limb pairing and its finite product-sum theorem.
Does not own: limb column binding, challenge derivation, projection rows, or
cost ownership. Emits constraints: no.
-/

namespace Nightstream.Implementation.R1CS.ProjectionPhi81

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism

/-- Pair two base-field Phi81 rings as the two coefficients of one
quadratic-extension Phi81 ring. -/
def pairRingF (low high : RingF) : RingK :=
  fun lane => ⟨low lane, high lane⟩

/-- Decode two coefficient lists into one typed extension-ring value. -/
def pairRings (low high : Ring) : RingK :=
  pairRingF (ringOfList low) (ringOfList high)

private def extensionUnit : K := ⟨0, 1⟩

private theorem pairRingF_decompose (low high : RingF) :
    pairRingF low high =
      ringKAdd
        (RingKAction.embedChallenge low)
        (RingKAction.scale extensionUnit
          (RingKAction.embedChallenge high)) := by
  funext lane
  simp only [pairRingF, ringKAdd, RingKAction.embedChallenge,
    RingKAction.scale, extensionUnit, K.add, K.mul, K.embed, K.mk.injEq]
  constructor
  · rw [Fin.zero_mul, Fin.mul_zero, Fin.zero_add, Fin.add_zero]
  · rw [Fin.one_mul, Fin.zero_mul]
    simp only [Fin.zero_add]

private theorem pairRingF_zero :
    pairRingF ringFZero ringFZero = ringKZero := by
  rfl

private theorem pairRingF_add
    (lowLeft lowRight highLeft highRight : RingF) :
    pairRingF
        (ringFAdd lowLeft lowRight)
        (ringFAdd highLeft highRight) =
      ringKAdd
        (pairRingF lowLeft highLeft)
        (pairRingF lowRight highRight) := by
  rfl

/-- Multiplication by an embedded base-ring challenge acts independently on
the two extension limbs. -/
theorem pairRingF_action (challenge low high : RingF) :
    pairRingF
        (ringFMul challenge low)
        (ringFMul challenge high) =
      ringKMul
        (RingKAction.embedChallenge challenge)
        (pairRingF low high) := by
  symm
  calc
    ringKMul (RingKAction.embedChallenge challenge) (pairRingF low high) =
        ringKMul (RingKAction.embedChallenge challenge)
          (ringKAdd
            (RingKAction.embedChallenge low)
            (RingKAction.scale extensionUnit
              (RingKAction.embedChallenge high))) := by
      rw [pairRingF_decompose]
    _ = ringKAdd
          (ringKMul
            (RingKAction.embedChallenge challenge)
            (RingKAction.embedChallenge low))
          (ringKMul
            (RingKAction.embedChallenge challenge)
            (RingKAction.scale extensionUnit
              (RingKAction.embedChallenge high))) :=
      RingKAction.ringKMul_right_add _ _ _
    _ = ringKAdd
          (RingKAction.embedChallenge (ringFMul challenge low))
          (RingKAction.scale extensionUnit
            (RingKAction.embedChallenge (ringFMul challenge high))) := by
      rw [← Embedding.embedChallenge_ringFMul,
        RingKAction.ringKMul_right_scale,
        ← Embedding.embedChallenge_ringFMul]
    _ = pairRingF
          (ringFMul challenge low)
          (ringFMul challenge high) :=
      (pairRingF_decompose _ _).symm

private theorem pairRingF_productSum
    {count : Nat} (challenges lows highs : Fin count -> RingF) :
    pairRingF
        (productSum challenges lows)
        (productSum challenges highs) =
      PiRLCFinite.combineEvaluation challenges
        (fun index => pairRingF (lows index) (highs index)) := by
  induction count with
  | zero => exact pairRingF_zero
  | succ count inductionHypothesis =>
      rw [productSum, productSum, PiRLCFinite.combineEvaluation,
        pairRingF_add, pairRingF_action]
      rw [inductionHypothesis
        (fun index => challenges index.succ)
        (fun index => lows index.succ)
        (fun index => highs index.succ)]

/-- Pairing the two list combinations gives exactly the independent typed
`RingK` finite combination. -/
theorem pairRings_phi81Combine
    {count : Nat} (challenges lows highs : Fin count -> Ring) :
    pairRings
        (phi81Combine challenges lows)
        (phi81Combine challenges highs) =
      PiRLCFinite.combineEvaluation
        (fun index => ringOfList (challenges index))
        (fun index => pairRings (lows index) (highs index)) := by
  unfold pairRings
  rw [ringOfList_phi81Combine challenges lows,
    ringOfList_phi81Combine challenges highs]
  exact pairRingF_productSum
    (fun index => ringOfList (challenges index))
    (fun index => ringOfList (lows index))
    (fun index => ringOfList (highs index))

end Nightstream.Implementation.R1CS.ProjectionPhi81
