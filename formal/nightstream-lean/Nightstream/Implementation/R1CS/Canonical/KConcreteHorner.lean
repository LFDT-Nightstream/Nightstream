import Nightstream.Implementation.R1CS.Canonical.KPointEquality

/-!
Contract: direct soundness bridge from canonical Horner rows to the concrete
quadratic-extension evaluator used by paper PiCCS.

Owns no rows.  It proves that the row layer's `hornerValue` is exactly
`Message.evaluateCoefficients ConcreteCarrier.extensionOps.toOps`, then
packages the result for carried coefficients decoded from one assignment.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KConcreteHorner

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner

abbrev ConcreteK := Nightstream.SuperNeo.Concrete.K

theorem ofConcrete_evaluateCoefficients (point : ConcreteK) :
    ∀ coefficients : List ConcreteK,
      KConcreteBridge.ofConcrete
          (Message.evaluateCoefficients ConcreteCarrier.extensionOps.toOps
            point coefficients) =
        hornerValue (KConcreteBridge.ofConcrete point)
          (coefficients.map KConcreteBridge.ofConcrete)
  | [] => KConcreteBridge.ofConcrete_zero
  | [coefficient] => by
      simp only [Message.evaluateCoefficients, List.map_cons, List.map_nil]
      rw [ConcreteCarrier.extensionLaws.mul_zero,
        ConcreteCarrier.extensionLaws.add_zero]
      rfl
  | coefficient :: next :: rest => by
      change
        KConcreteBridge.ofConcrete
            (Nightstream.SuperNeo.Concrete.K.add coefficient
              (Nightstream.SuperNeo.Concrete.K.mul point
                (Message.evaluateCoefficients
                  ConcreteCarrier.extensionOps.toOps point
                  (next :: rest)))) =
          addPair (KConcreteBridge.ofConcrete coefficient)
            (mulPair (KConcreteBridge.ofConcrete point)
              (hornerValue (KConcreteBridge.ofConcrete point)
                ((next :: rest).map KConcreteBridge.ofConcrete)))
      rw [KConcreteBridge.ofConcrete_add, KConcreteBridge.ofConcrete_mul,
        ofConcrete_evaluateCoefficients point (next :: rest)]

def decoded (assignment : Nat → Nat) (value : Carried) : ConcreteK :=
  KPointEquality.decoded assignment value

theorem rows_sound
    (assignment : Nat → Nat) (point : Carried) (frames : Nat → Frame)
    (coefficients : List Carried) (step : Nat)
    (satisfied :
      Satisfies (hornerRows point frames coefficients step) assignment) :
    decoded assignment (hornerCarried point frames coefficients step) =
      Message.evaluateCoefficients ConcreteCarrier.extensionOps.toOps
        (decoded assignment point)
        (coefficients.map (decoded assignment)) := by
  unfold decoded
  apply KConcreteBridge.ofConcrete_injective
  rw [KPointEquality.ofConcrete_decoded]
  have computed :=
    hornerRows_sound assignment point frames coefficients step satisfied
  rw [computed, ← KPointEquality.ofConcrete_decoded assignment point]
  have coefficientMap :
      coefficients.map (carriedValue assignment) =
        (coefficients.map
          (KPointEquality.decoded assignment)).map
            KConcreteBridge.ofConcrete := by
    rw [List.map_map]
    apply List.map_congr_left
    intro coefficient _
    exact (KPointEquality.ofConcrete_decoded assignment coefficient).symm
  rw [coefficientMap]
  exact
    (ofConcrete_evaluateCoefficients
      (KPointEquality.decoded assignment point)
      (coefficients.map (KPointEquality.decoded assignment))).symm

end Nightstream.Implementation.R1CS.Canonical.KConcreteHorner
