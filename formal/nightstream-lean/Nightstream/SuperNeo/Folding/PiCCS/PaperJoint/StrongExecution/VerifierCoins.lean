import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
import Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Product

/-!
Finite verifier-coin seeds for causal paper `Pi_CCS`.

Owns: one explicit product seed for the alpha word, gamma scalar, and SumCheck
round word; its exact finite support; conversion to `PublicCoins`; cardinality;
membership; and the three uniform product marginals.

Does not own: a prover, acceptance, target witnesses, rejection conditioning,
root bounds, Fiat--Shamir, Rust, R1CS, or costs.

Emits constraints: no.

The seed support is derived from one verifier-owned scalar alphabet. No prover
message, witness, or protocol predicate can affect the enumerated coins.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.VerifierCoins

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

universe uExtension

/-- One fixed-length verifier word. -/
abbrev Word (Extension : Type uExtension) (variables : Nat) :=
  Fin variables -> Extension

/-- All paper `Pi_CCS` verifier randomness, before conversion to lists:
alpha, then gamma, then the independent SumCheck round word. -/
abbrev Seed (Extension : Type uExtension) (variables : Nat) :=
  Word Extension variables × (Extension × Word Extension variables)

def alphaWord
    {Extension : Type uExtension}
    {variables : Nat}
    (seed : Seed Extension variables) : Word Extension variables :=
  seed.1

def gamma
    {Extension : Type uExtension}
    {variables : Nat}
    (seed : Seed Extension variables) : Extension :=
  seed.2.1

def roundWord
    {Extension : Type uExtension}
    {variables : Nat}
    (seed : Seed Extension variables) : Word Extension variables :=
  seed.2.2

/-- Exact uniform seed support induced by one scalar alphabet. -/
def support
    {Extension : Type uExtension}
    (alphabet : Support Extension)
    (variables : Nat) : Support (Seed Extension variables) :=
  let words := FiniteWords.Support.challengeVectors alphabet variables
  words.product (alphabet.product words)

@[simp] theorem support_values
    {Extension : Type uExtension}
    (alphabet : Support Extension)
    (variables : Nat) :
    (support alphabet variables).values =
      (FiniteWords.Support.challengeVectors alphabet variables).values.flatMap
        (fun alpha =>
          (alphabet.product
            (FiniteWords.Support.challengeVectors alphabet variables)).values.map
              fun gammaAndRounds => (alpha, gammaAndRounds)) := by
  rfl

theorem support_nodup
    {Extension : Type uExtension}
    (alphabet : Support Extension)
    (variables : Nat) :
    (support alphabet variables).values.Nodup :=
  (support alphabet variables).nodup

theorem support_nonempty
    {Extension : Type uExtension}
    (alphabet : Support Extension)
    (variables : Nat) :
    (support alphabet variables).values ≠ [] :=
  (support alphabet variables).nonempty

/-- Exact factorized support size. -/
@[simp] theorem support_cardinality
    {Extension : Type uExtension}
    (alphabet : Support Extension)
    (variables : Nat) :
    (support alphabet variables).cardinality =
      alphabet.cardinality ^ variables *
        (alphabet.cardinality * alphabet.cardinality ^ variables) := by
  simp [support]

/-- The same support size in the paper-readable `|K|^(2*ell+1)` form. -/
theorem support_cardinality_pow
    {Extension : Type uExtension}
    (alphabet : Support Extension)
    (variables : Nat) :
    (support alphabet variables).cardinality =
      alphabet.cardinality ^ (2 * variables + 1) := by
  rw [support_cardinality]
  calc
    alphabet.cardinality ^ variables *
          (alphabet.cardinality * alphabet.cardinality ^ variables) =
        alphabet.cardinality ^ variables *
          (alphabet.cardinality ^ variables * alphabet.cardinality) := by
            ac_rfl
    _ = alphabet.cardinality ^ (variables + (variables + 1)) := by
      rw [Nat.pow_add, Nat.pow_succ]
    _ = alphabet.cardinality ^ (2 * variables + 1) := by
      congr 1
      omega

/-- Support membership is exactly coordinate-wise scalar-alphabet membership
for both words, plus scalar membership for gamma. -/
theorem mem_support_iff
    {Extension : Type uExtension}
    (alphabet : Support Extension)
    (variables : Nat)
    (seed : Seed Extension variables) :
    seed ∈ (support alphabet variables).values ↔
      (forall index, alphaWord seed index ∈ alphabet.values) ∧
      gamma seed ∈ alphabet.values ∧
      (forall index, roundWord seed index ∈ alphabet.values) := by
  unfold support
  rw [Support.mem_product_iff, Support.mem_product_iff,
    FiniteWords.Support.mem_challengeVectors_iff,
    FiniteWords.Support.mem_challengeVectors_iff]
  rfl

private def wordPoint
    {Extension : Type uExtension}
    {variables : Nat}
    (word : Word Extension variables) : CubePoint Extension variables where
  coordinates := List.ofFn word
  dimension := by simp

/-- Convert the explicit seed to the exact public-coin record consumed by
`StrongExecution.execute`. The other protocol-shape coordinates do not affect
this conversion. -/
def toPublicCoins
    {Extension : Type uExtension}
    {shape : Shape}
    (seed : Seed Extension shape.cubeVariables) :
    PublicCoins Extension shape where
  alpha := wordPoint (alphaWord seed)
  gamma := gamma seed
  roundPoint := wordPoint (roundWord seed)

@[simp] theorem toPublicCoins_alpha_coordinates
    {Extension : Type uExtension}
    {shape : Shape}
    (seed : Seed Extension shape.cubeVariables) :
    (toPublicCoins seed).alpha.coordinates =
      List.ofFn (alphaWord seed) :=
  rfl

@[simp] theorem toPublicCoins_gamma
    {Extension : Type uExtension}
    {shape : Shape}
    (seed : Seed Extension shape.cubeVariables) :
    (toPublicCoins seed).gamma = gamma seed :=
  rfl

@[simp] theorem toPublicCoins_round_coordinates
    {Extension : Type uExtension}
    {shape : Shape}
    (seed : Seed Extension shape.cubeVariables) :
    (toPublicCoins seed).roundPoint.coordinates =
      List.ofFn (roundWord seed) :=
  rfl

/-- Alpha-word marginal of the exact product support. -/
theorem alphaWord_marginal
    {Extension : Type uExtension}
    (alphabet : Support Extension)
    (variables : Nat)
    (event : Word Extension variables -> Bool) :
    ((support alphabet variables).uniform).probabilityBool
        (fun seed => event (alphaWord seed)) =
      ((FiniteWords.Support.challengeVectors alphabet variables).uniform).probabilityBool
        event := by
  let words := FiniteWords.Support.challengeVectors alphabet variables
  simpa [support, alphaWord, words] using
    Support.product_uniform_probabilityBool_first
      words (alphabet.product words) event

/-- Gamma marginal of the exact product support. -/
theorem gamma_marginal
    {Extension : Type uExtension}
    (alphabet : Support Extension)
    (variables : Nat)
    (event : Extension -> Bool) :
    ((support alphabet variables).uniform).probabilityBool
        (fun seed => event (gamma seed)) =
      alphabet.uniform.probabilityBool event := by
  let words := FiniteWords.Support.challengeVectors alphabet variables
  calc
    ((support alphabet variables).uniform).probabilityBool
          (fun seed => event (gamma seed)) =
        ((alphabet.product words).uniform).probabilityBool
          (fun gammaAndRounds => event gammaAndRounds.1) := by
            simpa [support, gamma, words] using
              Support.product_uniform_probabilityBool_second
                words (alphabet.product words)
                  (fun gammaAndRounds => event gammaAndRounds.1)
    _ = alphabet.uniform.probabilityBool event :=
      Support.product_uniform_probabilityBool_first alphabet words event

/-- SumCheck-round-word marginal of the exact product support. -/
theorem roundWord_marginal
    {Extension : Type uExtension}
    (alphabet : Support Extension)
    (variables : Nat)
    (event : Word Extension variables -> Bool) :
    ((support alphabet variables).uniform).probabilityBool
        (fun seed => event (roundWord seed)) =
      ((FiniteWords.Support.challengeVectors alphabet variables).uniform).probabilityBool
        event := by
  let words := FiniteWords.Support.challengeVectors alphabet variables
  calc
    ((support alphabet variables).uniform).probabilityBool
          (fun seed => event (roundWord seed)) =
        ((alphabet.product words).uniform).probabilityBool
          (fun gammaAndRounds => event gammaAndRounds.2) := by
            simpa [support, roundWord, words] using
              Support.product_uniform_probabilityBool_second
                words (alphabet.product words)
                  (fun gammaAndRounds => event gammaAndRounds.2)
    _ = words.uniform.probabilityBool event :=
      Support.product_uniform_probabilityBool_second alphabet words event

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.VerifierCoins
