import Nightstream.Implementation.R1CS.Canonical.KStrictNorm
import Nightstream.Implementation.R1CS.Canonical.KMulChainHonest

/-!
Honest completeness for the exact strict-`b = 2` norm residual.

The two rows blocks are one ordinary multiplication chain with factors
`value` and `value - 1`.  The witness therefore comes directly from the
canonical chain witness; this module only proves that the source value and the
constant wire lie below the fresh two-frame allocation.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KStrictNormHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KStrictNorm

def factors (input : Input) : List Carried :=
  [input.value, KLinear.subCarried input.value KLinear.oneCarried]

def initial (input : Input) : Carried :=
  KLinear.addCarried input.value KLinear.oneCarried

theorem rows_eq_chain (input : Input) :
    rows input =
      KMulChain.rows (initial input) (KFrames.frameAt input.frameBase)
        (factors input) 0 := by
  rfl

def honestAssignment (input : Input) (assignment : Nat → Nat) : Nat → Nat :=
  KMulChainHonest.witness assignment (initial input) (factors input)
    input.frameBase 0

private theorem one_below (base : Nat) (positive : 0 < base) :
    KHornerHonest.BelowBase KLinear.oneCarried.low base ∧
      KHornerHonest.BelowBase KLinear.oneCarried.high base := by
  constructor <;> intro column mentioned
  · simp only [KLinear.oneCarried, Mentions, List.map_cons, List.map_nil,
      List.mem_singleton] at mentioned
    subst column
    exact positive
  · simp [KLinear.oneCarried, Mentions] at mentioned

private theorem append_below (left right : LinComb) (base : Nat)
    (leftBelow : KHornerHonest.BelowBase left base)
    (rightBelow : KHornerHonest.BelowBase right base) :
    KHornerHonest.BelowBase (left ++ right) base := by
  intro column mentioned
  rcases List.mem_append.1
      (by simpa only [Mentions, List.map_append] using mentioned) with
    inLeft | inRight
  · exact leftBelow column inLeft
  · exact rightBelow column inRight

private theorem initial_below (input : Input) (positive : 0 < input.frameBase)
    (valueBelow :
      KHornerHonest.BelowBase input.value.low input.frameBase ∧
        KHornerHonest.BelowBase input.value.high input.frameBase) :
    KHornerHonest.BelowBase (initial input).low input.frameBase ∧
      KHornerHonest.BelowBase (initial input).high input.frameBase := by
  have one := one_below input.frameBase positive
  exact ⟨append_below _ _ _ valueBelow.1 one.1,
    append_below _ _ _ valueBelow.2 one.2⟩

private theorem sub_below (input : Input) (positive : 0 < input.frameBase)
    (valueBelow :
      KHornerHonest.BelowBase input.value.low input.frameBase ∧
        KHornerHonest.BelowBase input.value.high input.frameBase) :
    KHornerHonest.BelowBase
        (KLinear.subCarried input.value KLinear.oneCarried).low
        input.frameBase ∧
      KHornerHonest.BelowBase
        (KLinear.subCarried input.value KLinear.oneCarried).high
        input.frameBase := by
  have one := one_below input.frameBase positive
  have scale_below : ∀ combination,
      KHornerHonest.BelowBase combination input.frameBase →
      KHornerHonest.BelowBase
        (Nightstream.Implementation.R1CS.LinearSubstitution.scaleTerms
          (goldilocksP - 1) combination) input.frameBase := by
    intro combination below column mentioned
    apply below column
    simpa [Nightstream.Implementation.R1CS.LinearSubstitution.scaleTerms,
      Mentions] using mentioned
  unfold KLinear.subCarried KLinear.addCarried KLinear.scaleCarried
  exact ⟨append_below _ _ _ valueBelow.1 (scale_below _ one.1),
    append_below _ _ _ valueBelow.2 (scale_below _ one.2)⟩

/-- Every authoritative assignment extends to a satisfying strict-norm
execution.  No semantic residual is supplied: the row program computes it. -/
theorem rows_honest
    (input : Input) (assignment : Nat → Nat)
    (basePositive : 0 < input.frameBase)
    (valueBelow :
      KHornerHonest.BelowBase input.value.low input.frameBase ∧
        KHornerHonest.BelowBase input.value.high input.frameBase) :
    Satisfies (rows input) (honestAssignment input assignment) := by
  rw [rows_eq_chain]
  exact
    KMulChainHonest.witness_satisfies_from_base assignment
      (initial input) (factors input) input.frameBase
      (initial_below input basePositive valueBelow).1
      (initial_below input basePositive valueBelow).2
      (by
        intro factor member
        simp only [factors, List.mem_cons, List.not_mem_nil, or_false] at member
        rcases member with rfl | rfl
        · exact valueBelow
        · exact sub_below input basePositive valueBelow)

theorem honestAssignment_preserves_below
    (input : Input) (assignment : Nat → Nat)
    (column : Nat) (below : column < input.frameBase) :
    honestAssignment input assignment column = assignment column :=
  KMulChainHonest.witness_off_before assignment
    (initial input) input.frameBase (factors input) 0 column (by simpa)

end Nightstream.Implementation.R1CS.Canonical.KStrictNormHonest
