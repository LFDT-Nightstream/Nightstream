import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.RingTransport
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLCFinite

/-!
Typed, matrix-indexed evaluation refinement for `Pi_RLC`.

Assurance tier: model-level. These theorems compare independent Lean public
computations; they do not establish Rust-conformant row emission.

Protocol: SuperNeo `Pi_RLC` inside the fixed F' NIFS.
Phase: two-limb evaluation decoding and public ring action.
Constraint family: two public `y_ring` identities per semantic matrix, grouped
into one typed `RingK` evaluation per matrix; this file emits no rows.

Owns: pairing two base-field Phi81 rings into one extension-field evaluation;
proof that an embedded `RingF` challenge acts on both limbs exactly as
`ringKMul`; transport of the list-level `phi81Combine` into the independent
finite `RingK` fold; and the shape-indexed array equation.

Does not own: evaluation-column serialization or padding; matrix semantics;
the output point; transcript challenges; projection-row soundness; source or
parent authority; costs; or row removal.

Emits constraints: no.

Authority boundary: every output limb is computed from explicit challenge and
input rings. No claimed evaluation, digest, private opening, multiplication
law, or native-verifier result is accepted as a premise.

Axiom boundary: the ring-action theorem uses only `propext` and `Quot.sound`;
array packaging additionally inherits `Classical.choice` from the independent
array-level evaluation interface. The fail-closed guards distinguish them.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.identities.y_ring.decode` | two base-field rings decode to one `RingK` evaluation | computed | `pairRings`, `decodeYRingRings` |
| `nifs.pi_rlc.verify.identities.y_ring.action` | embedded `RingF` multiplication acts independently on both extension limbs | derived | `pairRingF_action` |
| `nifs.pi_rlc.verify.identities.y_ring.combine` | list `phi81Combine` equals the independent typed evaluation fold | derived | `pairRings_phi81Combine` |
| `nifs.pi_rlc.verify.identities.y_ring.array` | every shape-indexed output evaluation uses the same finite fold | derived | `decodeYRingRings_phi81Combine` |
| `nifs.pi_rlc.verify.identities.y_ring.equation` | exact projection identities imply the typed evaluation equation | derived | `typedEvaluationEquation_of_refinement` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.EvaluationBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RingTransport

/-- Pair two typed base-field Phi81 rings as the two coefficients of one
quadratic-extension Phi81 ring. -/
def pairRingF (low high : RingF) : RingK :=
  fun lane => ⟨low lane, high lane⟩

/-- Decode two production coefficient lists into one typed evaluation. -/
def pairRings (low high : Ring) : Evaluation :=
  pairRingF (ringOfList low) (ringOfList high)

/-- Decode every matrix-indexed evaluation row in canonical order. -/
def decodeYRingRings {matrixCount : Nat}
    (rings : YRingRings matrixCount) : Array Evaluation :=
  Array.ofFn fun row : Fin matrixCount =>
    pairRings
      (rings row ⟨0, by decide⟩)
      (rings row ⟨1, by decide⟩)

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
  · rw [Fin.zero_mul,
      Fin.mul_zero,
      Fin.zero_add,
      Fin.add_zero]
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

/-- Multiplication by an embedded base-ring challenge acts on the two
extension limbs exactly as two independent `ringFMul` operations. -/
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

/-- Pairing the two production list combinations gives exactly the independent
typed `RingK` evaluation combination. -/
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

@[simp] theorem decodeYRingRings_size {matrixCount : Nat}
    (rings : YRingRings matrixCount) :
    (decodeYRingRings rings).size = matrixCount := by
  simp [decodeYRingRings]

theorem decodeYRingRings_getD {matrixCount : Nat}
    (rings : YRingRings matrixCount) (row : Fin matrixCount) :
    (decodeYRingRings rings).getD row.val BaseLinear.evaluationZero =
      pairRings
        (rings row ⟨0, by decide⟩)
        (rings row ⟨1, by decide⟩) := by
  rw [Array.getD_eq_getD_getElem?,
    Array.getElem?_eq_getElem (by simp [decodeYRingRings])]
  simp [decodeYRingRings]

/-- The shape-indexed evaluation carrier uses the independent typed finite
`RingK` combination at every matrix. -/
theorem decodeYRingRings_phi81Combine
    {shape : Shape} {count : Nat}
    (challenges : Fin count -> Ring)
    (inputs : Fin count -> YRingRings shape.matrixCount) :
    decodeYRingRings
        (fun row limb =>
          phi81Combine challenges (fun index => inputs index row limb)) =
      PiRLCFinite.combineEvaluations (shape := shape)
        (fun index => ringOfList (challenges index))
        (fun index => decodeYRingRings (inputs index)) := by
  apply Array.ext
  · simp [decodeYRingRings, PiRLCFinite.combineEvaluations]
  · intro index leftLt _rightLt
    have indexLt : index < shape.matrixCount := by
      simpa [decodeYRingRings] using leftLt
    let row : Fin shape.matrixCount := ⟨index, indexLt⟩
    simp only [PiRLCFinite.combineEvaluations, Array.getElem_ofFn]
    calc
      (decodeYRingRings
          (fun outputRow limb =>
            phi81Combine challenges
              (fun source => inputs source outputRow limb)))[index] =
          pairRings
            (phi81Combine challenges
              (fun source => inputs source row ⟨0, by decide⟩))
            (phi81Combine challenges
              (fun source => inputs source row ⟨1, by decide⟩)) := by
        simp [decodeYRingRings, row]
      _ = PiRLCFinite.combineEvaluation
            (fun source => ringOfList (challenges source))
            (fun source => pairRings
              (inputs source row ⟨0, by decide⟩)
              (inputs source row ⟨1, by decide⟩)) :=
        pairRings_phi81Combine challenges
          (fun source => inputs source row ⟨0, by decide⟩)
          (fun source => inputs source row ⟨1, by decide⟩)
      _ = PiRLCFinite.combineEvaluation
            (fun source => ringOfList (challenges source))
            (fun source =>
              (decodeYRingRings (inputs source)).getD index
                BaseLinear.evaluationZero) := by
        apply congrArg (PiRLCFinite.combineEvaluation
          (fun source => ringOfList (challenges source)))
        funext source
        rw [show index = row.val by rfl]
        exact (decodeYRingRings_getD (inputs source) row).symm

/-- Exact shape-indexed projection identities imply the independent typed
evaluation combination. All values are derived from the equation refinement. -/
theorem typedEvaluationEquation_of_refinement
    {params : GlobalParams} {arity : BatchArity params}
    {shape : Shape}
    (assignment : Nat -> Nat)
    (columns : BatchColumns params arity shape.matrixCount)
    (tree : TraceTree arity shape.matrixCount)
    (refinement : EquationRefinement assignment columns tree) :
    decodeYRingRings
        (decodeOpening assignment columns.output).yRing =
      PiRLCFinite.combineEvaluations (shape := shape)
        (fun index =>
          ringOfList (values assignment (columns.challenges index)))
        (fun index =>
          decodeYRingRings
            (decodeOpening assignment (columns.inputs index)).yRing) := by
  have outputEquation :
      (decodeOpening assignment columns.output).yRing =
        fun row limb =>
          phi81Combine
            (fun index => values assignment (columns.challenges index))
            (fun index =>
              (decodeOpening assignment (columns.inputs index)).yRing row limb) := by
    funext row limb
    exact phi81_reduction_at refinement.challengeWiring refinement.wiring
      refinement.reduction (.yRing row limb)
  rw [outputEquation]
  exact decodeYRingRings_phi81Combine
    (fun index => values assignment (columns.challenges index))
    (fun index => (decodeOpening assignment (columns.inputs index)).yRing)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.EvaluationBridge
