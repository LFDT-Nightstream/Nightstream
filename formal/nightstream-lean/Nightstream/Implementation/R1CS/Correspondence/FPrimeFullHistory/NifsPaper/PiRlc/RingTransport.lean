import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-!
List-to-`RingF` transport for the production `Pi_RLC` public ring action.

Assurance tier: model-level. This file compares two independently typed Lean
computations; it does not establish Rust-conformant row emission.

Protocol: SuperNeo `Pi_RLC` inside the fixed F' NIFS.
Phase: public projection-ring representation refinement.
Constraint family: shared arithmetic beneath commitment, `X`, and evaluation
identities; this file emits no rows.

Owns: the canonical scalar sum used by production's coefficient lists; exact
interpretation of `phi81Combine` at one coefficient; and equality between the
complete list operation and a typed head-first `RingF` product sum.

Does not own: commitment, public-input, or evaluation carrier layouts;
transcript challenges; strong-set security; projection-row soundness; costs;
or row removal.

Emits constraints: no.

Authority boundary: both sides are computed from the same explicit challenge
and input functions. No digest, output witness, algebra law, or native-verifier
result is accepted as a premise.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.identities.public.product_sum.scalar` | production's left fold equals the canonical head-first scalar sum | derived | `phi81Combine_eq_scalarSum`, `phi81Combine_coefficient` |
| `nifs.pi_rlc.verify.identities.public.product_sum.ring` | decoding `phi81Combine` equals the typed `RingF` product sum | derived | `ringOfList_phi81Combine` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RingTransport

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc

/-- Canonical head-first sum on the base-field list carrier. -/
def scalarSum : List Scalar -> Scalar
  | [] => 0
  | value :: rest => value + scalarSum rest

private theorem foldl_eq_add_scalarSum
    (items : List Scalar) (initial : Scalar) :
    items.foldl (fun sum item => sum + item) initial =
      initial + scalarSum items := by
  induction items generalizing initial with
  | nil => exact (ConcreteCarrier.baseLaws.add_zero initial).symm
  | cons value rest inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis]
      exact ConcreteCarrier.baseLaws.add_assoc _ _ _

private theorem foldl_zero_eq_scalarSum (items : List Scalar) :
    items.foldl (fun sum item => sum + item) 0 = scalarSum items := by
  rw [foldl_eq_add_scalarSum]
  exact ConcreteCarrier.baseLaws.zero_add _

/-- Production's complete coefficient list is the canonical scalar sum at
every Phi81 coefficient. -/
theorem phi81Combine_eq_scalarSum
    {count : Nat} (challenges inputs : Fin count -> Ring) :
    phi81Combine challenges inputs =
      List.ofFn fun coefficient : Fin ringDegree =>
        scalarSum (List.ofFn fun index : Fin count =>
          ringFMul (ringOfList (challenges index))
            (ringOfList (inputs index)) coefficient) := by
  unfold phi81Combine
  apply congrArg List.ofFn
  funext coefficient
  exact foldl_zero_eq_scalarSum _

/-- Reading one coefficient of production's list result exposes exactly the
canonical sum of the corresponding typed Phi81 product coefficients. -/
theorem phi81Combine_coefficient
    {count : Nat} (challenges : Fin count -> Ring)
    (inputs : Fin count -> Ring) (coefficient : Fin ringDegree) :
    ringOfList (phi81Combine challenges inputs) coefficient =
      scalarSum (List.ofFn fun index : Fin count =>
        ringFMul (ringOfList (challenges index))
          (ringOfList (inputs index)) coefficient) := by
  unfold ringOfList phi81Combine
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by simp), List.getElem_ofFn]
  simp only [Option.getD_some]
  exact foldl_zero_eq_scalarSum _

/-- Canonical head-first sum of typed challenge-times-input products. -/
def productSum : {count : Nat} ->
    (Fin count -> RingF) -> (Fin count -> RingF) -> RingF
  | 0, _, _ => ringFZero
  | _ + 1, challenges, inputs =>
      ringFAdd
        (ringFMul (challenges 0) (inputs 0))
        (productSum
          (fun index => challenges index.succ)
          (fun index => inputs index.succ))

private theorem productSum_coefficient
    {count : Nat} (challenges inputs : Fin count -> RingF)
    (coefficient : Fin ringDegree) :
    productSum challenges inputs coefficient =
      scalarSum (List.ofFn fun index : Fin count =>
        ringFMul (challenges index) (inputs index) coefficient) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [productSum, List.ofFn_succ, scalarSum, ringFAdd]
      rw [inductionHypothesis
        (fun index => challenges index.succ)
        (fun index => inputs index.succ)]

/-- The complete production list operation decodes to the canonical typed
Phi81 product sum, for every finite input count and every coefficient. -/
theorem ringOfList_phi81Combine
    {count : Nat} (challenges inputs : Fin count -> Ring) :
    ringOfList (phi81Combine challenges inputs) =
      productSum
        (fun index => ringOfList (challenges index))
        (fun index => ringOfList (inputs index)) := by
  funext coefficient
  rw [phi81Combine_coefficient]
  exact (productSum_coefficient
    (fun index => ringOfList (challenges index))
    (fun index => ringOfList (inputs index)) coefficient).symm

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RingTransport
