import Nightstream.Implementation.R1CS.Canonical.KBooleanMle
import Nightstream.Implementation.R1CS.Canonical.KConcreteBridge
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra

/-!
Contract: representation bridge from the canonical Boolean-MLE rows to the
existing semantic `BooleanTable.evaluate`.

The bridge decodes every table leaf and every point coordinate from the
satisfying assignment.  The dimension proof is used only to rule out the
totalized missing-coordinate branch; it does not supply an evaluation
equation.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KBooleanMleSemantics

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Decode one carried row-layer value into the semantic extension field. -/
def decodeCarried (assignment : Nat → Nat) (value : Carried) : K where
  c0 := ⟨lcEval assignment value.low, by
    unfold lcEval
    exact Nat.mod_lt _ (by decide)⟩
  c1 := ⟨lcEval assignment value.high, by
    unfold lcEval
    exact Nat.mod_lt _ (by decide)⟩

@[simp] theorem ofConcrete_decodeCarried
    (assignment : Nat → Nat) (value : Carried) :
    KConcreteBridge.ofConcrete (decodeCarried assignment value) =
      carriedValue assignment value := by
  rfl

/-- Decode every leaf without changing the typed low/high tree. -/
def decodeTable (assignment : Nat → Nat) :
    {variables : Nat} →
      BooleanTable Carried variables → BooleanTable K variables
  | 0, .leaf value => .leaf (decodeCarried assignment value)
  | _ + 1, .branch low high =>
      .branch (decodeTable assignment low) (decodeTable assignment high)

/-- Decode an exact-length coordinate list into the semantic cube point. -/
def decodePoint
    (assignment : Nat → Nat)
    {variables : Nat}
    (coordinates : List Carried)
    (dimension : coordinates.length = variables) :
    CubePoint K variables where
  coordinates := coordinates.map (decodeCarried assignment)
  dimension := by rw [List.length_map, dimension]

private theorem tail_dimension
    {variables : Nat}
    {coordinate : Carried} {coordinates : List Carried}
    (dimension : (coordinate :: coordinates).length = variables + 1) :
    coordinates.length = variables := by
  simpa using Nat.succ.inj dimension

/-- The independently decoded recursion is exactly the semantic MLE. -/
theorem evaluate_decodes
    (assignment : Nat → Nat) :
    ∀ {variables : Nat}
      (table : BooleanTable Carried variables)
      (coordinates : List Carried)
      (dimension : coordinates.length = variables),
      KConcreteBridge.ofConcrete
          (BooleanTable.evaluate ConcreteCarrier.extensionOps
            (decodeTable assignment table)
            (decodePoint assignment coordinates dimension)) =
        KBooleanMle.decodedValue assignment table coordinates
  | 0, .leaf _, [], _ => rfl
  | 0, .leaf _, _ :: _, dimension => by simp at dimension
  | variables + 1, .branch low high, [], dimension => by simp at dimension
  | variables + 1, .branch low high, coordinate :: coordinates, dimension => by
      have tailDimension := tail_dimension dimension
      have lowInduction :=
        evaluate_decodes assignment low coordinates tailDimension
      have highInduction :=
        evaluate_decodes assignment high coordinates tailDimension
      simp only [BooleanTable.evaluate, BooleanTable.evaluateCoordinates,
        decodeTable, decodePoint, List.map_cons,
        KBooleanMle.decodedValue, KBooleanMle.headCoordinate,
        KBooleanMle.tailCoordinates]
      simp only [BooleanTable.evaluate, decodePoint] at lowInduction
      simp only [BooleanTable.evaluate, decodePoint] at highInduction
      rw [ConcreteCarrier.derived_sub_eq_concrete_sub]
      simp only [ConcreteCarrier.extensionOps]
      simp only [ConcreteCarrier.extensionOps] at lowInduction
      simp only [ConcreteCarrier.extensionOps] at highInduction
      rw [KConcreteBridge.ofConcrete_add, KConcreteBridge.ofConcrete_mul,
        KConcreteBridge.ofConcrete_sub, ofConcrete_decodeCarried,
        lowInduction, highInduction]

/-- Headline soundness in the semantic carrier. -/
theorem rows_compute_evaluate
    (assignment : Nat → Nat)
    (base : Nat)
    {variables : Nat}
    (table : BooleanTable Carried variables)
    (coordinates : List Carried)
    (dimension : coordinates.length = variables)
    (satisfied :
      Satisfies
        (KBooleanMle.rows (KFrames.frameAt base) table coordinates 0)
        assignment) :
    carriedValue assignment
        (KBooleanMle.carried (KFrames.frameAt base) table coordinates 0) =
      KConcreteBridge.ofConcrete
        (BooleanTable.evaluate ConcreteCarrier.extensionOps
          (decodeTable assignment table)
          (decodePoint assignment coordinates dimension)) := by
  rw [KBooleanMle.rows_sound assignment (KFrames.frameAt base)
    table coordinates 0 satisfied]
  exact (evaluate_decodes assignment table coordinates dimension).symm

end Nightstream.Implementation.R1CS.Canonical.KBooleanMleSemantics
