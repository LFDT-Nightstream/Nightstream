import Nightstream.Implementation.R1CS.Core.Semantics

/-!
Contract: the kernel-checked Lucas certificate data for the Goldilocks modulus.

Owns: a kernel-evaluable modular exponentiation, the full factorisation of
`q - 1`, and the seven exponentiation facts witnessing that `7` has
multiplicative order exactly `q - 1` modulo `q`.

Does **not** own, and does not prove: that `q` is prime, or
`EuclidPrime goldilocksP`. Those need Lucas's theorem — order `q - 1` implies
every nonzero residue is a unit implies primality — and that derivation is not
written here. This module supplies only the arithmetic the derivation will
consume.

`GoldilocksField` consumes these facts through Lucas's theorem and closes the
end-to-end `EuclidPrime goldilocksP` and inverse-witness boundary.

## Why the exponentiation carries fuel

Kernel reduction needs structural recursion, so `powMod` recurses on a fuel
counter rather than on the exponent. Fuel exhaustion returns `0`, which is a
**poison** value, not a plausible one.

That choice is load-bearing and it is why every condition below is stated as an
exact value rather than as `≠ 1`. With a `≠ 1` formulation an exhausted fuel
budget returns `0`, and `0 ≠ 1` holds — an under-fuelled call would *pass* the
test while computing nothing. Stating the exact residue makes the poison value
fail, so the certificate is fail-closed against its own fuel parameter.

This is not hypothetical: the first draft of this probe had the arguments
swapped, ran with fuel `7` against a 64-bit exponent, and returned `1` from the
exhaustion branch — exactly the value a Fermat test accepts. The disagreement
that exposed it was `#eval` reporting `7` a quadratic residue when it is not.

## No `native_decide`

Every theorem here is closed by `decide`, so the reduction happens in the
kernel. `#print axioms` reports no axioms at all for these.
-/

set_option autoImplicit false
-- The certificate exponents are 64-bit, so the fuel recursion is ~65 deep.
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.GoldilocksCertificate

open Nightstream.Implementation.R1CS

/-- Modular exponentiation by repeated squaring, structural on `fuel` so the
kernel can reduce it. Fuel exhaustion returns the poison value `0`. -/
def powMod (modulus : Nat) : Nat → Nat → Nat → Nat
  | 0, _, _ => 0
  | _, _, 0 => 1 % modulus
  | fuel + 1, base, exponent =>
      let half := powMod modulus fuel (base * base % modulus) (exponent / 2)
      if exponent % 2 = 1 then half * base % modulus else half

/-- Fuel large enough for any exponent below `2 ^ 70`. -/
def certificateFuel : Nat := 70

/-- The proposed generator. -/
def generator : Nat := 7

/-! ## The factorisation of `q - 1`

Fully known, which is what makes a Lucas certificate available at all. -/

theorem order_factorisation :
    goldilocksP - 1 = 2 ^ 32 * 3 * 5 * 17 * 257 * 65537 := by
  decide

/-! ## The certificate

`7 ^ (q - 1) = 1`, and `7 ^ ((q - 1) / p)` is a specific residue other than `1`
for each prime `p` dividing `q - 1`. Together these say the order of `7` is
exactly `q - 1`.

Each is stated as an exact residue, not as `≠ 1`, so that a poisoned fuel value
cannot satisfy it. -/

theorem fermat :
    powMod goldilocksP certificateFuel generator (goldilocksP - 1) = 1 := by
  decide

theorem order_not_halved :
    powMod goldilocksP certificateFuel generator ((goldilocksP - 1) / 2)
      = 18446744069414584320 := by
  decide

theorem order_not_thirded :
    powMod goldilocksP certificateFuel generator ((goldilocksP - 1) / 3)
      = 18446744065119617025 := by
  decide

theorem order_not_fifthed :
    powMod goldilocksP certificateFuel generator ((goldilocksP - 1) / 5)
      = 1373043270956696022 := by
  decide

theorem order_not_seventeenthed :
    powMod goldilocksP certificateFuel generator ((goldilocksP - 1) / 17)
      = 16301593560560007290 := by
  decide

theorem order_not_two_five_seventhed :
    powMod goldilocksP certificateFuel generator ((goldilocksP - 1) / 257)
      = 995085315851368103 := by
  decide

theorem order_not_sixtyfive_five_three_seventhed :
    powMod goldilocksP certificateFuel generator ((goldilocksP - 1) / 65537)
      = 8478886009461009681 := by
  decide

/-! ## The six residues are not one

Immediate from the exact values, and stated separately because it is the form
Lucas's theorem consumes. -/

theorem certificate_residues_ne_one :
    (18446744069414584320 : Nat) ≠ 1
      ∧ (18446744065119617025 : Nat) ≠ 1
      ∧ (1373043270956696022 : Nat) ≠ 1
      ∧ (16301593560560007290 : Nat) ≠ 1
      ∧ (995085315851368103 : Nat) ≠ 1
      ∧ (8478886009461009681 : Nat) ≠ 1 := by
  decide

end Nightstream.Implementation.R1CS.Canonical.GoldilocksCertificate
