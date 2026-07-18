import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.RingTransport
import Nightstream.Implementation.R1CS.Correspondence.Projection.Phi81.Pairing
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLCFinite

/-!
Fixed-history evaluation-carrier packaging over the profile-neutral Phi81
two-limb theorem.

Assurance tier: model-level.

Owns: diagnostic matrix-array decoding and its shape-indexed composition with
the neutral two-limb fold. Does not own: the base-ring pairing proof, active
profiles, generated rows, transcript authority, source authority, costs, or
row removal. Emits constraints: no.

| Stage family | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `identities.y_ring.pair` | two base limbs form one `RingK` value | derived | `ProjectionPhi81.pairRings_phi81Combine` |
| `identities.y_ring.array` | package every diagnostic matrix evaluation | derived | `decodeYRingRings_phi81Combine` |
| `identities.y_ring.equation` | exact diagnostic equation uses the typed array fold | derived | `typedEvaluationEquation_of_refinement` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.EvaluationBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc

abbrev pairRingF := ProjectionPhi81.pairRingF
abbrev pairRings := ProjectionPhi81.pairRings

theorem pairRingF_action (challenge low high : RingF) :
    pairRingF (ringFMul challenge low) (ringFMul challenge high) =
      ringKMul (RingKAction.embedChallenge challenge)
        (pairRingF low high) :=
  ProjectionPhi81.pairRingF_action challenge low high

theorem pairRings_phi81Combine
    {count : Nat} (challenges lows highs : Fin count -> Ring) :
    pairRings
        (phi81Combine challenges lows)
        (phi81Combine challenges highs) =
      PiRLCFinite.combineEvaluation
        (fun index => ringOfList (challenges index))
        (fun index => pairRings (lows index) (highs index)) :=
  ProjectionPhi81.pairRings_phi81Combine challenges lows highs

/-- Decode every matrix-indexed evaluation row in canonical order. -/
def decodeYRingRings {matrixCount : Nat}
    (rings : YRingRings matrixCount) : Array Evaluation :=
  Array.ofFn fun row : Fin matrixCount =>
    pairRings
      (rings row ⟨0, by decide⟩)
      (rings row ⟨1, by decide⟩)

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

/-- The shape-indexed diagnostic carrier uses the independent typed finite
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

/-- Exact diagnostic projection identities imply the independent typed
evaluation combination. -/
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
