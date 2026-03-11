import SuperNeo.Field

/-!
Contract interface for `SuperNeo.Field`.

Spec: `specs/Field.spec.md`

Paper anchors:
- Definition 1, Section 4, lines 275-282: `F` is a finite field of prime order `q`.
- Definition 3, Section 4, lines 290-291: centered representative for `‖a‖_∞`.
-/

namespace SuperNeo

namespace FieldInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `F`. -/
abbrev F := SuperNeo.F

/-- [Role: Theorem-Target] Curated re-export of `F.ofNat`. -/
abbrev F_ofNat := SuperNeo.F.ofNat

/-- [Role: Theorem-Target] Curated re-export of `F.zero`. -/
abbrev F_zero := SuperNeo.F.zero

/-- [Role: Theorem-Target] Curated re-export of `F.one`. -/
abbrev F_one := SuperNeo.F.one

/-- [Role: Theorem-Target] Curated re-export of `F.pow`. -/
abbrev F_pow := SuperNeo.F.pow

/-- [Role: Theorem-Target] Curated re-export of `F.inv`. -/
abbrev F_inv := SuperNeo.F.inv

/-- [Role: Theorem-Target] Curated re-export of `F.canonicalRep`. -/
abbrev F_canonicalRep := SuperNeo.F.canonicalRep

/-- [Role: Theorem-Target] Curated re-export of `F.isCanonical`. -/
abbrev F_isCanonical := SuperNeo.F.isCanonical

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Curated theorem surface `F.canonical`. -/
abbrev F_canonical := SuperNeo.F.canonical

/-- [Role: Theorem-Target] Curated theorem surface `F.canonicalCheck_true`. -/
abbrev F_canonicalCheck_true := SuperNeo.F.canonicalCheck_true

/-- [Role: Theorem-Target] Curated theorem surface `F.centeredRep_eq_of_le_halfQ`. -/
theorem F_centeredRep_eq_of_le_halfQ {a : F}
    (h : a.val ≤ SuperNeo.Goldilocks.halfQ) :
    SuperNeo.F.centeredRep a = Int.ofNat a.val :=
  SuperNeo.F.centeredRep_eq_of_le_halfQ h

/-- [Role: Theorem-Target] Curated theorem surface `F.centeredRep_eq_sub_q_of_halfQ_lt`. -/
theorem F_centeredRep_eq_sub_q_of_halfQ_lt {a : F}
    (h : SuperNeo.Goldilocks.halfQ < a.val) :
    SuperNeo.F.centeredRep a = Int.ofNat a.val - Int.ofNat SuperNeo.Goldilocks.q :=
  SuperNeo.F.centeredRep_eq_sub_q_of_halfQ_lt h

/-- [Role: Theorem-Target] Total branch split for centered representatives. -/
abbrev F_centeredRep_cases := SuperNeo.F.centeredRep_cases

/-- [Role: Theorem-Target] Non-dependent centered representative cover theorem. -/
abbrev F_centeredRep_cover := SuperNeo.F.centeredRep_cover

/-- [Role: Theorem-Target] Curated theorem surface `F.ofNat_val`. -/
abbrev F_ofNat_val := SuperNeo.F.ofNat_val

/-- [Role: Theorem-Target] Curated theorem surface `F.canonicalRep_ofNat`. -/
abbrev F_canonicalRep_ofNat := SuperNeo.F.canonicalRep_ofNat

/-- [Role: Theorem-Target] Curated theorem surface `F.ofNat_canonicalRep`. -/
abbrev F_ofNat_canonicalRep := SuperNeo.F.ofNat_canonicalRep

/-- [Role: Theorem-Target] Curated theorem surface `F.ofNat_val_eq_of_canonical`. -/
theorem F_ofNat_val_eq_of_canonical
    {n : Nat}
    (h : n < SuperNeo.Goldilocks.q) :
    (SuperNeo.F.ofNat n).val = n :=
  SuperNeo.F.ofNat_val_eq_of_canonical h

/-- [Role: Theorem-Target] Curated theorem surface `F.canonicalRep_ofNat_eq_of_lt`. -/
theorem F_canonicalRep_ofNat_eq_of_lt
    {n : Nat}
    (h : n < SuperNeo.Goldilocks.q) :
    SuperNeo.F.canonicalRep (SuperNeo.F.ofNat n) = n :=
  SuperNeo.F.canonicalRep_ofNat_eq_of_lt h

/-- [Role: Theorem-Target] Curated theorem surface `F.canonicalCheck_iff`. -/
abbrev F_canonicalCheck_iff := SuperNeo.F.canonicalCheck_iff

/-- [Role: Theorem-Target] Curated theorem surfaces for canonical representatives of `0/1`. -/
abbrev F_canonicalRep_zero := SuperNeo.F.canonicalRep_zero
abbrev F_canonicalRep_one := SuperNeo.F.canonicalRep_one

/-- [Role: Theorem-Target] Curated theorem surfaces for centered representatives of `0/1`. -/
abbrev F_centeredRep_zero := SuperNeo.F.centeredRep_zero
abbrev F_centeredRep_one := SuperNeo.F.centeredRep_one

/-- [Role: Theorem-Target] Curated rewrite surfaces for field operation representatives. -/
abbrev F_val_add := SuperNeo.F.val_add
abbrev F_val_sub := SuperNeo.F.val_sub
abbrev F_val_mul := SuperNeo.F.val_mul
abbrev F_val_neg := SuperNeo.F.val_neg
abbrev F_canonicalRep_add := SuperNeo.F.canonicalRep_add
abbrev F_canonicalRep_mul := SuperNeo.F.canonicalRep_mul
abbrev F_canonicalRep_neg := SuperNeo.F.canonicalRep_neg

end FieldInterface

end SuperNeo
