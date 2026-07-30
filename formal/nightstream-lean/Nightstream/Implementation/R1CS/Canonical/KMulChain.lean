import Nightstream.Implementation.R1CS.Canonical.KFrames

/-!
Contract: sequential multiplication of quadratic-extension values.

Owns: the emitted row list for a left-to-right product, its exact row and
column counts, the carried result, and soundness against the corresponding
`K`-value recursion.

Does not own: honest witness construction or positional ownership. Those need
the canonical allocator's separation facts and live in companion modules.

Each factor costs exactly one `KMul` frame: three rows and three auxiliary
columns. The recursion is deliberately left-to-right because that is the
shape needed by point-equality products, sparse CCS monomials, and the strict
norm cubic in the paper PiCCS terminal formula.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KMulChain

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner

/-- The carried output of one multiplication frame. -/
def frameOutput (frame : Frame) : Carried :=
  ⟨outLow frame, outHigh frame⟩

/-- The reference left-to-right product on decoded coordinate pairs. -/
def productValue (initial : Pair) : List Pair → Pair
  | [] => initial
  | factor :: rest => productValue (mulPair initial factor) rest

/-- The carried output after multiplying all factors. -/
def productCarried (initial : Carried) (frames : Nat → Frame) :
    List Carried → Nat → Carried
  | [], _ => initial
  | _ :: rest, step =>
      productCarried (frameOutput (frames step)) frames rest (step + 1)

/-- Three multiplication rows per factor. -/
def rows (initial : Carried) (frames : Nat → Frame) :
    List Carried → Nat → List Row
  | [], _ => []
  | factor :: rest, step =>
      KMul.rows initial factor (frames step) ++
        rows (frameOutput (frames step)) frames rest (step + 1)

/-- Exact row count, derived from the emitted list. -/
theorem rows_length (initial : Carried) (frames : Nat → Frame) :
    ∀ (factors : List Carried) (step : Nat),
      (rows initial frames factors step).length = 3 * factors.length
  | [], _ => rfl
  | factor :: rest, step => by
      rw [rows, List.length_append, KMul.rows_length,
        rows_length (frameOutput (frames step)) frames rest (step + 1)]
      simp only [List.length_cons]
      omega

/-- Exact auxiliary allocation for a chain placed at `base`. -/
def columns (base factorCount : Nat) : List Nat :=
  KFrames.frameColumns base factorCount

theorem columns_length (base factorCount : Nat) :
    (columns base factorCount).length = 3 * factorCount :=
  KFrames.frameColumns_length _ _

theorem columns_nodup (base factorCount : Nat) :
    (columns base factorCount).Nodup :=
  KFrames.frameColumns_nodup _ _

/-- One frame computes one decoded `K` multiplication. -/
theorem frameOutput_sound
    (assignment : Nat → Nat) (left right : Carried) (frame : Frame)
    (satisfied : Satisfies (KMul.rows left right frame) assignment) :
    carriedValue assignment (frameOutput frame) =
      mulPair (carriedValue assignment left) (carriedValue assignment right) := by
  have low := KMul.outLow_sound assignment left right frame satisfied
  have high := KMul.outHigh_sound assignment left right frame satisfied
  unfold frameOutput carriedValue mulPair
  simp only [Pair.mk.injEq]
  exact ⟨low, high⟩

/-- Satisfying rows compute the exact left-to-right reference product. -/
theorem rows_sound
    (assignment : Nat → Nat) (frames : Nat → Frame) :
    ∀ (initial : Carried) (factors : List Carried) (step : Nat),
      Satisfies (rows initial frames factors step) assignment →
      carriedValue assignment
          (productCarried initial frames factors step) =
        productValue (carriedValue assignment initial)
          (factors.map (carriedValue assignment))
  | _, [], _, _ => rfl
  | initial, factor :: rest, step, satisfied => by
      have headSatisfied :
          Satisfies (KMul.rows initial factor (frames step)) assignment :=
        fun row member =>
          satisfied row (List.mem_append.2 (Or.inl member))
      have tailSatisfied :
          Satisfies
            (rows (frameOutput (frames step)) frames rest (step + 1))
            assignment :=
        fun row member =>
          satisfied row (List.mem_append.2 (Or.inr member))
      have head :=
        frameOutput_sound assignment initial factor (frames step) headSatisfied
      have tail :=
        rows_sound assignment frames (frameOutput (frames step)) rest
          (step + 1) tailSatisfied
      simp only [productCarried, productValue, List.map_cons]
      rw [tail, head]

end Nightstream.Implementation.R1CS.Canonical.KMulChain
