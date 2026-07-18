import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc
import Nightstream.Implementation.R1CS.Correspondence.Projection.Phi81.Carrier

/-!
Diagnostic-carrier compatibility surface for the profile-neutral Phi81 list
transport.

Assurance tier: model-level.

Owns: only the legacy namespace expected by fixed-profile diagnostic modules.
Does not own: the mathematical proofs, active profiles, generated artifacts,
transcript authority, costs, or row removal. Emits constraints: no.

| Legacy name | Neutral mathematical owner | Authority class |
|---|---|---|
| scalar/product sums | `ProjectionPhi81` | computed |
| list-to-`RingF` transport | `ProjectionPhi81.ringOfList_phi81Combine` | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RingTransport

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc

abbrev scalarSum := ProjectionPhi81.scalarSum

theorem phi81Combine_eq_scalarSum
    {count : Nat} (challenges inputs : Fin count -> Ring) :
    phi81Combine challenges inputs =
      List.ofFn fun coefficient : Fin ringDegree =>
        scalarSum (List.ofFn fun index : Fin count =>
          ringFMul (ringOfList (challenges index))
            (ringOfList (inputs index)) coefficient) :=
  ProjectionPhi81.phi81Combine_eq_scalarSum challenges inputs

theorem phi81Combine_coefficient
    {count : Nat} (challenges inputs : Fin count -> Ring)
    (coefficient : Fin ringDegree) :
    ringOfList (phi81Combine challenges inputs) coefficient =
      scalarSum (List.ofFn fun index : Fin count =>
        ringFMul (ringOfList (challenges index))
          (ringOfList (inputs index)) coefficient) :=
  ProjectionPhi81.phi81Combine_coefficient challenges inputs coefficient

def productSum : {count : Nat} ->
    (Fin count -> RingF) -> (Fin count -> RingF) -> RingF
  | 0, _, _ => ringFZero
  | _ + 1, challenges, inputs =>
      ringFAdd
        (ringFMul (challenges 0) (inputs 0))
        (productSum
          (fun index => challenges index.succ)
          (fun index => inputs index.succ))

theorem productSum_eq_projectionPhi81
    {count : Nat} (challenges inputs : Fin count -> RingF) :
    productSum challenges inputs =
      ProjectionPhi81.productSum challenges inputs := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [productSum, ProjectionPhi81.productSum]
      rw [inductionHypothesis
        (fun index => challenges index.succ)
        (fun index => inputs index.succ)]

theorem ringOfList_phi81Combine
    {count : Nat} (challenges inputs : Fin count -> Ring) :
    ringOfList (phi81Combine challenges inputs) =
      productSum
        (fun index => ringOfList (challenges index))
        (fun index => ringOfList (inputs index)) := by
  change ProjectionPhi81.ringOfList
      (ProjectionPhi81.phi81Combine challenges inputs) = _
  rw [ProjectionPhi81.ringOfList_phi81Combine]
  exact (productSum_eq_projectionPhi81
    (fun index => ringOfList (challenges index))
    (fun index => ringOfList (inputs index))).symm

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RingTransport
