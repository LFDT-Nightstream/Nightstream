import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Nightstream.Implementation.NebulaV2.FPrime.State.SeedSchedule

/-!
Contract: explicit seven-role hybrid budget for the verifier-key-seeded V2
Ajtai matrices.

Assurance tier: cryptographic-assumption boundary plus proved union arithmetic.

Owns the exact partition of the seven setup roles, a per-role uniform
Module-SIS advantage, a per-role ChaCha8 distinguishing advantage, the hybrid
inequality for the selected manifest seed, and the exact union bound.

Does not prove Module-SIS hardness, ChaCha8 pseudorandomness, or that final
rank-18 bundle widths meet an estimator. Those are named fields of
`HybridAssumption`, not conclusions hidden in an execution theorem.

The setup event is global for one verifier key. It is not multiplied by the
number of folds or segments.
-/

set_option autoImplicit false

namespace Nightstream.Assurance.NebulaV2.SeededSetupSecurity

open scoped BigOperators
open Nightstream.Implementation.NebulaV2.SeedSchedule

def rawRoleBits : Nat := 131
def postUnionBits : Nat := 127

def dyadic (bits : Nat) : ℚ :=
  1 / (2 ^ bits : ℚ)

theorem exact_role_partition :
    (Finset.univ.filter fun role : Role => role.rows = 18).card = 3 ∧
      (Finset.univ.filter fun role : Role => role.rows = 2).card = 2 ∧
      (Finset.univ.filter fun role : Role => role.rows = 1).card = 2 := by
  decide

/-- Computational assumptions for the exact matrices selected by one
manifest. `uniformKernelAdvantage` refers to the uniform-matrix Module-SIS
game at that role's exact rows, columns, ring, and norm. `prgAdvantage` refers
to replacing that uniform matrix with the manifest's pure ChaCha8 expansion.
The hybrid inequality must be proved by the concrete game reduction. -/
structure HybridAssumption (manifest : Manifest) where
  seededKernelAdvantage : Role → ℚ
  uniformKernelAdvantage : Role → ℚ
  prgAdvantage : Role → ℚ
  seededNonnegative : ∀ role, 0 ≤ seededKernelAdvantage role
  uniformNonnegative : ∀ role, 0 ≤ uniformKernelAdvantage role
  prgNonnegative : ∀ role, 0 ≤ prgAdvantage role
  hybrid : ∀ role,
    seededKernelAdvantage role ≤
      uniformKernelAdvantage role + prgAdvantage role
  uniformBound : ∀ role,
    uniformKernelAdvantage role ≤ dyadic rawRoleBits
  prgBound : ∀ role,
    prgAdvantage role ≤ dyadic rawRoleBits

namespace HybridAssumption

def totalSeededAdvantage
    {manifest : Manifest} (assumption : HybridAssumption manifest) : ℚ :=
  ∑ role : Role, assumption.seededKernelAdvantage role

theorem per_role_le_two_raw
    {manifest : Manifest} (assumption : HybridAssumption manifest)
    (role : Role) :
    assumption.seededKernelAdvantage role ≤ 2 * dyadic rawRoleBits := by
  calc
    assumption.seededKernelAdvantage role ≤
        assumption.uniformKernelAdvantage role +
          assumption.prgAdvantage role :=
      assumption.hybrid role
    _ ≤ dyadic rawRoleBits + dyadic rawRoleBits :=
      add_le_add (assumption.uniformBound role) (assumption.prgBound role)
    _ = 2 * dyadic rawRoleBits := by ring

/-- Seven exact roles, each with one uniform-hardness term and one PRG term,
consume less than `2^-127` total setup advantage. -/
theorem total_lt_post_union
    {manifest : Manifest} (assumption : HybridAssumption manifest) :
    assumption.totalSeededAdvantage < dyadic postUnionBits := by
  calc
    assumption.totalSeededAdvantage ≤
        ∑ _role : Role, (2 * dyadic rawRoleBits) := by
      apply Finset.sum_le_sum
      intro role _
      exact assumption.per_role_le_two_raw role
    _ = 7 * (2 * dyadic rawRoleBits) := by
      rw [Finset.sum_const, Finset.card_univ, role_count]
      norm_num
    _ < dyadic postUnionBits := by
      norm_num [dyadic, rawRoleBits, postUnionBits]

theorem total_nonnegative
    {manifest : Manifest} (assumption : HybridAssumption manifest) :
    0 ≤ assumption.totalSeededAdvantage := by
  apply Finset.sum_nonneg
  intro role _
  exact assumption.seededNonnegative role

end HybridAssumption

end Nightstream.Assurance.NebulaV2.SeededSetupSecurity
