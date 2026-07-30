import Nightstream.Implementation.R1CS.Canonical.KMul

/-!
Contract: Horner evaluation of a coefficient list at a challenge, as emitted
rows.

Owns: the row program for one polynomial evaluation over the Goldilocks
quadratic extension, its derived row count, and the proof that a satisfying
assignment computes the reference Horner value.

Does not own: honest completeness, ownership, or conservation — all three need
frame disjointness, which soundness does not. Nor the projection identity (two
evaluations plus one equality row), the PiRLC batch, or any NIFS structure.

## Shape

`ProjectionCheck.eval` is `coefficients.foldr (fun c suffix => c + point *
suffix) 0`. The row program mirrors that `foldr` exactly, which is what makes
the induction close structurally rather than by hand — the same reason the
sponge's `chainValues` was written to mirror `absorb`.

Each recursion step is one `K` multiplication and one `K` addition. The
multiplication emits `KMul`'s three rows; the addition is concatenation of
combinations and emits nothing. So `n + 1` coefficients cost `n`
multiplications and `3n` rows.

The last coefficient is a base case rather than a multiply-by-zero. Writing it
as `foldr`'s literal `c + point * 0` would emit three rows whose product is
known to vanish — a wasted multiplication per evaluation, and the projection
batch performs many. `hornerValue`'s `[c]` case agrees with `eval` because
`K.add c 0 = c`, so nothing is lost by special-casing it.

## Why the reference is on coordinate pairs

The canonical track evaluates through `lcEval : (Nat → Nat) → LinComb → Nat`.
The reference here is therefore stated on `Pair`, two `Nat` coordinates, rather
than on `ProjectionProgram.K`, which is a pair of `Fin goldilocksP`. Both
describe the same arithmetic; relating them is a representation bridge that
belongs with the identity check, not here, and it is named
`KHORNER-K-BRIDGE`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KHorner

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul

/-! ## The reference, on coordinate pairs -/

/-- A quadratic-extension value as two canonical coordinates. -/
structure Pair where
  low : Nat
  high : Nat

/-- Extension multiplication, `X² = 7`. -/
def mulPair (x y : Pair) : Pair where
  low := (x.low * y.low + 7 * (x.high * y.high)) % goldilocksP
  high := (x.low * y.high + x.high * y.low) % goldilocksP

/-- Extension addition, coordinatewise. -/
def addPair (x y : Pair) : Pair where
  low := (x.low + y.low) % goldilocksP
  high := (x.high + y.high) % goldilocksP

/-- **The reference Horner value.**  Mirrors `ProjectionCheck.eval`'s `foldr`,
with the final coefficient as a base case instead of `c + point * 0`. -/
def hornerValue (beta : Pair) : List Pair → Pair
  | [] => ⟨0, 0⟩
  | [c] => c
  | c :: rest => addPair c (mulPair beta (hornerValue beta rest))

/-! ## The emitted program

`frames` supplies one `KMul.Frame` per multiplication, indexed by position, so
distinct steps allocate distinct columns. Frame disjointness is the caller's
obligation and is what a whole-program conservation proof will need; it is not
assumed here. -/

/-- The value each recursion step carries, as combinations. -/
def hornerCarried (beta : Carried) (frames : Nat → Frame) :
    List Carried → Nat → Carried
  | [], _ => ⟨[], []⟩
  | [c], _ => c
  | c :: _ :: rest, step =>
      ⟨c.low ++ outLow (frames step), c.high ++ outHigh (frames step)⟩

/-- **The emitted rows.**  Three per multiplication, none for the additions. -/
def hornerRows (beta : Carried) (frames : Nat → Frame) :
    List Carried → Nat → List Row
  | [], _ => []
  | [_], _ => []
  | _ :: next :: rest, step =>
      KMul.rows beta (hornerCarried beta frames (next :: rest) (step + 1))
          (frames step)
        ++ hornerRows beta frames (next :: rest) (step + 1)

/-- **The derived row count.**  Three rows per multiplication, and one
multiplication fewer than there are coefficients. -/
theorem hornerRows_length
    (beta : Carried) (frames : Nat → Frame) :
    ∀ (coefficients : List Carried) (step : Nat),
      (hornerRows beta frames coefficients step).length
        = 3 * (coefficients.length - 1)
  | [], _ => rfl
  | [_], _ => rfl
  | _ :: next :: rest, step => by
      have tail := hornerRows_length beta frames (next :: rest) (step + 1)
      simp only [hornerRows, List.length_append, KMul.rows, List.length_cons,
        List.length_nil, tail]
      omega

/-- A degree-`d` polynomial has `d + 1` coefficients, so it costs `3d` rows. -/
theorem hornerRows_length_of_degree
    (beta : Carried) (frames : Nat → Frame)
    (coefficients : List Carried) (degree : Nat)
    (sized : coefficients.length = degree + 1) (step : Nat) :
    (hornerRows beta frames coefficients step).length = 3 * degree := by
  rw [hornerRows_length, sized]
  omega

/-! ## Soundness

The reference values are *derived from the assignment* rather than supplied
alongside it, so no agreement hypothesis is needed and no two-list relation has
to be carried through the induction.

Note what this does **not** need: frame disjointness. Each step's
`outLow_sound` uses only that step's own rows, so overlapping frames would
over-constrain the system without breaking this direction. Disjointness is
required for honest completeness and for ownership, not here. -/

/-- The pair a carried value denotes under an assignment. -/
def carriedValue (z : Nat → Nat) (value : Carried) : Pair where
  low := lcEval z value.low
  high := lcEval z value.high

theorem lcEval_append (z : Nat → Nat) (left right : LinComb) :
    lcEval z (left ++ right)
      = (lcEval z left + lcEval z right) % goldilocksP := by
  rw [lcEval_eq_rawSum, rawSum_append, lcEval_eq_rawSum, lcEval_eq_rawSum,
    Nat.add_mod]

theorem lcEval_nil (z : Nat → Nat) : lcEval z [] = 0 := rfl

/-- **A satisfying assignment computes the reference Horner value.** -/
theorem hornerRows_sound
    (z : Nat → Nat) (beta : Carried) (frames : Nat → Frame) :
    ∀ (coefficients : List Carried) (step : Nat),
      Satisfies (hornerRows beta frames coefficients step) z →
      carriedValue z (hornerCarried beta frames coefficients step)
        = hornerValue (carriedValue z beta)
            (coefficients.map (carriedValue z))
  | [], _, _ => rfl
  | [_], _, _ => rfl
  | c :: next :: rest, step, satisfied => by
      have splitLeft : Satisfies (KMul.rows beta
          (hornerCarried beta frames (next :: rest) (step + 1))
          (frames step)) z :=
        fun row member => satisfied row (List.mem_append.2 (Or.inl member))
      have splitRight : Satisfies
          (hornerRows beta frames (next :: rest) (step + 1)) z :=
        fun row member => satisfied row (List.mem_append.2 (Or.inr member))
      have tail := hornerRows_sound z beta frames (next :: rest) (step + 1)
        splitRight
      have low := KMul.outLow_sound z beta
        (hornerCarried beta frames (next :: rest) (step + 1)) (frames step)
        splitLeft
      have high := KMul.outHigh_sound z beta
        (hornerCarried beta frames (next :: rest) (step + 1)) (frames step)
        splitLeft
      simp only [hornerCarried, hornerValue, carriedValue, addPair, mulPair,
        List.map_cons] at *
      rw [lcEval_append, lcEval_append, low, high, ← tail]

end Nightstream.Implementation.R1CS.Canonical.KHorner
