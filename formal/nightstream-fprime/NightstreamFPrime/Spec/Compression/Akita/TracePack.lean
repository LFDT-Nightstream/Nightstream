import NightstreamFPrime.Spec.Compression.Akita.Prover
import NightstreamFPrime.Spec.Compression.Akita.Trace
import NightstreamFPrime.Spec.Compression.Akita.TraceFold

/-! The layout of the scalar row of a level of Akita with the evaluation-trace opening method, for
`k = 2` (ePrint 2026/1983, revision of 2026-10-15, Section 3.1 with (21), Section 5.3
with (65) and (66), and Section 7.1 with (95) to (98)).

A trace level commits to ring elements `F_{i,x} = ψ(f̄_{i,x})`: each has `d_A/k = 2·m` values of
the packed polynomial `f̄` of Akita Section 3.7 in the extension field.

- `unpack d m u` is entry `u` of `ψ^{-1}`: `Z ↦ (k/d)·Tr_H(Z·σ_{−1}(X^{p(u)}))`, and `unpack_psi`
  states that it returns entry `u` of the input of `ψ`.
- `packFunctional` is `T_{ρ_pack}` of Akita (97), `packFunctional_apply` gives its value
  `Σ_u ẽq(ρ_pack, u)·ψ^{-1}(Z)_u` on each ring element `Z`, and `packFunctional_subfield_mul`
  states that it is linear over the fixed subfield.
- `unpackTable` is the table of the packed polynomial that a table of ring coefficients has:
  cell `(i·M + x)·(d_A/k) + u` is entry `u` of `ψ^{-1}(F_{i,x})`.
- `traceScalar_eq_mle` states that the left side of the row (98) for the partials
  `e_i = ⟨a, G_{b,M}·s_i⟩` of (66) is the multilinear evaluation of this table at the point
  `(ρ_pack, ρ_pos, ρ_blk)` of (65), with the first variable on the lowest bit of the cell. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib (eqAt mle mle_eq_cubeSum cubeSum_eq_sum_range)

variable {E : Type} [CommRing E] [Algebra Zq E] [Degree2 E]

section Unpack

variable {d : ℕ} [NeZero d] {m : ℕ}

/-- Entry `u` of `ψ^{-1}` for `k = 2` and `d = 4·m`: `Z ↦ (k/d)·Tr_H(Z·σ_{−1}(X^{p(u)}))`, for
the position `p(u)` of entry `u` in Akita (21). -/
noncomputable def unpack (d m : ℕ) [NeZero d] (u : ℕ) : Rq d →ₗ[Zq] E :=
  ((2 * m : ℕ) : Zq)⁻¹ •
    (traceForm d m ∘ₗ LinearMap.mulRight Zq (conj (AdjoinRoot.root (modulus d) ^ slot m u)))

/-- `T_{ρ_pack}` of Akita (97) for the packed weights `χ̌_pack = ψ(weights)` of (96):
`Z ↦ (k/d)·Tr_H(Z·σ_{−1}(χ̌_pack))`. -/
noncomputable def packFunctional (quarter : 4 * m = d) (weights : ℕ → E) :
    Rq d →ₗ[Zq] E :=
  ((2 * m : ℕ) : Zq)⁻¹ • traceFunctional quarter weights

theorem traceFunctional_eq_sum (quarter : 4 * m = d) (weights : ℕ → E) (Z : Rq d) :
    traceFunctional quarter weights Z = ∑ u ∈ Finset.range (2 * m),
      weights u * traceForm d m (Z * conj (AdjoinRoot.root (modulus d) ^ slot m u)) := by
  show traceForm d m (Z * conj (psi quarter weights)) = _
  rw [psi, map_sum, Finset.mul_sum, map_sum]
  refine Finset.sum_congr rfl fun u _ => ?_
  rw [map_mul (conj (d := d)), conj_subfield, ← traceForm_subfield_mul quarter]
  congr 1
  ring

/-- The value of `T_{ρ_pack}` on each ring element: `Σ_u weights_u·ψ^{-1}(Z)_u`. For the weights
`ẽq(ρ_pack, u)` it is the evaluation at `ρ_pack` of the `d/k` values that `Z` packs
(Akita (98)). -/
theorem packFunctional_apply (quarter : 4 * m = d) (weights : ℕ → E) (Z : Rq d) :
    packFunctional quarter weights Z =
      ∑ u ∈ Finset.range (2 * m), weights u * unpack d m u Z := by
  simp only [packFunctional, unpack, LinearMap.smul_apply, LinearMap.comp_apply,
    LinearMap.mulRight_apply, traceFunctional_eq_sum quarter, Finset.smul_sum, mul_smul_comm]

/-- `T_{ρ_pack}` is linear over the fixed subfield: `T(c·Z) = c·T(Z)` for `c ∈ E ⊆ R_{q,d}`. -/
theorem packFunctional_subfield_mul (quarter : 4 * m = d) (weights : ℕ → E)
    (c : E) (Z : Rq d) :
    packFunctional quarter weights (subfield quarter c * Z) =
      c * packFunctional quarter weights Z := by
  simp only [packFunctional, traceFunctional, LinearMap.smul_apply, LinearMap.comp_apply,
    LinearMap.mulRight_apply]
  rw [mul_assoc, traceForm_subfield_mul quarter, mul_smul_comm]

/-- `unpack` is the inverse of `ψ` on its `d/k = 2·m` entries, when `d/k` is a unit of
`Z_q`. -/
theorem unpack_psi (quarter : 4 * m = d) (unit : IsUnit ((2 * m : ℕ) : Zq)) (f : ℕ → E)
    {u : ℕ} (small : u < 2 * m) : unpack d m u (psi quarter f) = f u := by
  have single : psi quarter (fun v => if v = u then (1 : E) else 0) =
      AdjoinRoot.root (modulus d) ^ slot m u := by
    rw [psi, Finset.sum_eq_single u]
    · rw [ite_eq_left rfl, map_one, mul_one]
    · intro v _ other
      rw [ite_eq_right other, map_zero, mul_zero]
    · intro absent
      exact absurd (Finset.mem_range.mpr small) absent
  have value := traceForm_psi_mul_conj quarter f fun v => if v = u then (1 : E) else 0
  rw [single, Finset.sum_eq_single u (fun v _ other => by rw [ite_eq_right other, mul_zero])
    (fun absent => absurd (Finset.mem_range.mpr small) absent), ite_eq_left rfl, mul_one] at value
  simp only [unpack, LinearMap.smul_apply, LinearMap.comp_apply, LinearMap.mulRight_apply]
  rw [value, ← Nat.cast_smul_eq_nsmul Zq, smul_smul, ZMod.inv_mul_of_unit _ unit, one_smul]

/-- The table of the packed polynomial `f̄` that a table of ring coefficients has: cell
`r·(d/k) + u` is entry `u` of `ψ^{-1}` of the ring element `r` of the table. -/
noncomputable def unpackTable (d m : ℕ) [NeZero d] (table : Table) (cell : ℕ) : E :=
  unpack d m (cell % (2 * m)) (ofCoeffs fun k => table (cell / (2 * m) * d + k))

end Unpack

section Layout

variable {S : Shape} {m : ℕ}

/-- A sum over the cells of a table as a sum over the block `i`, the position `x` and the
entry `u`. -/
theorem sum_range_slots {M : Type} [AddCommMonoid M] (B L W : ℕ) (g : ℕ → M) :
    ∑ cell ∈ Finset.range (B * (L * W)), g cell =
      ∑ i : Fin B, ∑ x : Fin L, ∑ u : Fin W, g (u + W * x + L * W * i) := by
  rw [Finset.sum_range, sum_fin_mul]
  exact Finset.sum_congr rfl fun i _ => sum_fin_mul L W fun n => g (n + L * W * i)

/-- The layout of the trace row (Akita (98), with (65), (66), (95), (96) and (97)): for the
partials `e_i = ⟨a, G_{b,M}·s_i⟩` with `a_x = ẽq(ρ_pos, x)`, the sum
`Σ_i ẽq(ρ_blk, i)·T_{ρ_pack}(e_i)` is the multilinear evaluation at `(ρ_pack, ρ_pos, ρ_blk)` of
the packed polynomial of the table that the source blocks decode to. `M` and `d_A/k` are the
powers of two of the lengths of their segments. The number `B` of blocks is at most `2^{r_blk}`,
and the decoded table is 0 at each absent block (Akita Section 4.5, (40) to (42)). -/
theorem traceScalar_eq_mle (quarter : 4 * m = S.innerDegree) (block position pack : List E)
    (blocks : S.blocks ≤ 2 ^ block.length) (length : S.blockLength = 2 ^ position.length)
    (width : 2 * m = 2 ^ pack.length) (source : Fin S.blocks → S.Source) :
    ∑ i : Fin S.blocks, eqAt block i * packFunctional quarter (eqAt pack)
        (traceContract (subfield quarter).toRingHom (fun x : Fin S.blockLength => eqAt position x)
          (gadget S.sourceBase (source i))) =
      mle (pack ++ (position ++ block)) (unpackTable S.innerDegree m (S.decode source)) := by
  have positive : 0 < 2 * m := by
    have := NeZero.pos S.innerDegree
    omega
  have total : 2 ^ (pack ++ (position ++ block)).length =
      2 ^ block.length * (S.blockLength * (2 * m)) := by
    rw [length, width]
    simp only [List.length_append, pow_add]
    ring
  have live : S.blocks * (S.blockLength * (2 * m)) ≤ 2 ^ (pack ++ (position ++ block)).length := by
    rw [total]
    exact Nat.mul_le_mul_right _ blocks
  have zero : ∀ cell, S.blocks * (S.blockLength * (2 * m)) ≤ cell →
      (unpackTable S.innerDegree m (S.decode source) cell : E) = 0 := fun cell absent => by
    have ring : S.blocks * S.blockLength ≤ cell / (2 * m) :=
      (Nat.le_div_iff_mul_le positive).mpr (by rw [Nat.mul_assoc]; exact absent)
    have empty : (ofCoeffs fun k => S.decode source (cell / (2 * m) * S.innerDegree + k) :
        Rq S.innerDegree) = 0 := ext_coeff fun k small => by
      rw [coeff_ofCoeffs _ small, coeff_zero, S.decode_of_le source
        (by rw [length]; exact Nat.pow_pos Nat.two_pos)]
      calc S.blocks * (S.blockLength * S.innerDegree)
          = S.blocks * S.blockLength * S.innerDegree := (Nat.mul_assoc _ _ _).symm
        _ ≤ cell / (2 * m) * S.innerDegree := Nat.mul_le_mul_right _ ring
        _ ≤ cell / (2 * m) * S.innerDegree + k := Nat.le_add_right _ _
    rw [unpackTable, empty, map_zero]
  have left (i : Fin S.blocks) : packFunctional quarter (eqAt pack)
      (traceContract (subfield quarter).toRingHom (fun x : Fin S.blockLength => eqAt position x)
        (gadget S.sourceBase (source i))) =
      ∑ x : Fin S.blockLength, ∑ u : Fin (2 * m), eqAt position x *
        (eqAt pack u * unpack S.innerDegree m u (gadget S.sourceBase (source i) x)) := by
    rw [traceContract, map_sum]
    refine Finset.sum_congr rfl fun x _ => ?_
    rw [AlgHom.toRingHom_eq_coe, RingHom.coe_coe, packFunctional_subfield_mul,
      packFunctional_apply, Finset.sum_range, Finset.mul_sum]
  rw [mle_eq_sum_live _ _ live zero, sum_range_slots]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [left i, Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun u _ => ?_
  have nested : (u : ℕ) + 2 * m * x + S.blockLength * (2 * m) * i =
      u + 2 ^ pack.length * (x + 2 ^ position.length * i) := by
    rw [← width, ← length]
    ring
  have rest : ((u : ℕ) + 2 * m * (x + S.blockLength * i)) % (2 * m) = u := by
    rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt u.isLt]
  have quotient : ((u : ℕ) + 2 * m * (x + S.blockLength * i)) / (2 * m) =
      x + S.blockLength * i := by
    rw [Nat.add_mul_div_left _ _ positive, Nat.div_eq_of_lt u.isLt, zero_add]
  have cell : (unpackTable S.innerDegree m (S.decode source)
      ((u : ℕ) + 2 * m * x + S.blockLength * (2 * m) * i) : E) =
        unpack S.innerDegree m u (gadget S.sourceBase (source i) x) := by
    rw [show (u : ℕ) + 2 * m * x + S.blockLength * (2 * m) * i =
      u + 2 * m * (x + S.blockLength * i) by ring, unpackTable, rest, quotient]
    congr 1
    refine ext_coeff fun k small => ?_
    rw [coeff_ofCoeffs _ small, show ((x : ℕ) + S.blockLength * i) * S.innerDegree + k =
      (i * S.blockLength + x) * S.innerDegree + k by ring]
    exact S.decode_cell source i x small
  rw [cell, nested, eqAt_append _ _ (u.isLt.trans_eq width),
    eqAt_append _ _ (x.isLt.trans_eq length)]
  ring

end Layout

end NightstreamFPrime.Spec.Compression.Akita
