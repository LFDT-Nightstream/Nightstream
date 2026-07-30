import Nightstream.Implementation.R1CS.Canonical.KEquality
import Nightstream.Implementation.R1CS.Canonical.KHornerOwnership
import Nightstream.Implementation.R1CS.Canonical.KBridge

/-!
Contract: one projection identity, checked at a challenge, as emitted rows.

Owns: the row program that evaluates two coefficient lists at one challenge and
asserts the results agree, its derived row count, and the proof that
satisfaction forces the two evaluations to be equal.

Does not own: the PiRLC batch, the challenge's derivation, or any NIFS
structure. In particular this says nothing about *where* `beta` came from —
binding it to a verifier transcript is a separate obligation and is the
difference between a checked identity and a sound one.

## The composite, and its cost

Two Horner evaluations over disjoint frame blocks, plus one `K`-equality:

```
3·(|lhs| − 1)  +  3·(|rhs| − 1)  +  2
```

The trailing `2` is the equality, one row per extension coordinate. Earlier
cycles of this project wrote `+ 1` here; `KEquality.rows_length` derives the
two.

## What soundness gives, and what it does not

`identityRows_sound` says a satisfying assignment forces the two evaluations to
agree at `beta`. Composed with `KBridge.toPair_eval` that is exactly
`ProjectionCheck.Accepted`'s evaluation component.

It is **not** `Identity.Exact`. Agreement at one challenge is what the rows can
check; coefficient equality is what the protocol wants, and the gap between
them is the `BadRoot` event that `NifsRecipeShape.badRoot_at_production_ops`
shows is non-vacuous. That gap is intrinsic to the projection optimization, not
an artifact of this encoding.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KIdentity

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner

/-- The value the left-hand evaluation carries. -/
def leftCarried (beta : Carried) (leftBase : Nat) (left : List Carried) :
    Carried :=
  hornerCarried beta (KFrames.frameAt leftBase) left 0

/-- The value the right-hand evaluation carries. -/
def rightCarried (beta : Carried) (rightBase : Nat) (right : List Carried) :
    Carried :=
  hornerCarried beta (KFrames.frameAt rightBase) right 0

/-- **The emitted identity check.**  Two evaluations over disjoint frame
blocks, then a coordinatewise equality of their results. -/
def identityRows
    (beta : Carried) (leftBase rightBase : Nat) (left right : List Carried) :
    List Row :=
  hornerRows beta (KFrames.frameAt leftBase) left 0
    ++ hornerRows beta (KFrames.frameAt rightBase) right 0
    ++ KEquality.rows (leftCarried beta leftBase left)
        (rightCarried beta rightBase right)

/-- **The derived row count.**  Three per multiplication on each side, plus two
for the equality — not one. -/
theorem identityRows_length
    (beta : Carried) (leftBase rightBase : Nat) (left right : List Carried) :
    (identityRows beta leftBase rightBase left right).length
      = 3 * (left.length - 1) + 3 * (right.length - 1) + 2 := by
  unfold identityRows
  rw [List.length_append, List.length_append, hornerRows_length,
    hornerRows_length, KEquality.rows_length]

/-- A degree-`d` identity on both sides costs `6d + 2` rows. -/
theorem identityRows_length_of_degree
    (beta : Carried) (leftBase rightBase : Nat) (left right : List Carried)
    (degree : Nat) (leftSized : left.length = degree + 1)
    (rightSized : right.length = degree + 1) :
    (identityRows beta leftBase rightBase left right).length = 6 * degree + 2 := by
  rw [identityRows_length, leftSized, rightSized]
  omega

/-! ## Soundness -/

/-- **Satisfaction forces the two evaluations to agree at the challenge.**

This is `ProjectionCheck.Accepted`'s evaluation component, not
`Identity.Exact`. The gap between them is the projection-root event. -/
theorem identityRows_sound
    (z : Nat → Nat) (beta : Carried) (leftBase rightBase : Nat)
    (left right : List Carried) (constantWire : z 0 = 1)
    (satisfied : Satisfies (identityRows beta leftBase rightBase left right) z) :
    hornerValue (carriedValue z beta) (left.map (carriedValue z))
      = hornerValue (carriedValue z beta) (right.map (carriedValue z)) := by
  have leftSat : Satisfies (hornerRows beta (KFrames.frameAt leftBase) left 0) z :=
    fun row member => satisfied row
      (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inl member))))
  have rightSat :
      Satisfies (hornerRows beta (KFrames.frameAt rightBase) right 0) z :=
    fun row member => satisfied row
      (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inr member))))
  have equalSat : Satisfies (KEquality.rows (leftCarried beta leftBase left)
      (rightCarried beta rightBase right)) z :=
    fun row member => satisfied row (List.mem_append.2 (Or.inr member))
  have leftValue := hornerRows_sound z beta (KFrames.frameAt leftBase) left 0
    leftSat
  have rightValue := hornerRows_sound z beta (KFrames.frameAt rightBase) right 0
    rightSat
  rcases KEquality.rows_sound z _ _ constantWire equalSat with ⟨lowEq, highEq⟩
  rw [← leftValue, ← rightValue]
  show carriedValue z (leftCarried beta leftBase left)
    = carriedValue z (rightCarried beta rightBase right)
  unfold carriedValue
  simp only [Pair.mk.injEq]
  exact ⟨lowEq, highEq⟩

/-- **The identity is the frozen `ProjectionCheck` evaluation.**  Composing
with `KBridge.toPair_eval`: when the carried values denote extension elements,
the agreement this program forces is agreement of `ProjectionCheck.eval`. -/
theorem identityRows_is_projection_eval
    (z : Nat → Nat) (beta : Carried) (leftBase rightBase : Nat)
    (left right : List Carried) (constantWire : z 0 = 1)
    (point : ProjectionProgram.K) (leftK rightK : List ProjectionProgram.K)
    (betaDenotes : carriedValue z beta = KBridge.toPair point)
    (leftDenotes : left.map (carriedValue z) = leftK.map KBridge.toPair)
    (rightDenotes : right.map (carriedValue z) = rightK.map KBridge.toPair)
    (satisfied : Satisfies (identityRows beta leftBase rightBase left right) z) :
    KBridge.toPair (SuperNeo.ProjectionCheck.eval ProjectionProgram.K.ops leftK point)
      = KBridge.toPair
          (SuperNeo.ProjectionCheck.eval ProjectionProgram.K.ops rightK point) := by
  rw [KBridge.toPair_eval, KBridge.toPair_eval, ← betaDenotes, ← leftDenotes,
    ← rightDenotes]
  exact identityRows_sound z beta leftBase rightBase left right constantWire
    satisfied

end Nightstream.Implementation.R1CS.Canonical.KIdentity
