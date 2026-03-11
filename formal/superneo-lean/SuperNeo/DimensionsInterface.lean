import SuperNeo.Dimensions

/-!
Contract interface for `SuperNeo.Dimensions`.

Spec: `specs/Dimensions.spec.md`

Paper anchors:
- Definition 1, Section 4, lines 275-282: `d` = deg(Φ), `nF = d · nR`.
- Appendix B.2, lines 709-727: concrete `η = 81`, `d = 54`.
-/

namespace SuperNeo

namespace DimensionsInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `eta`. -/
abbrev eta := SuperNeo.eta

/-- [Role: Theorem-Target] Curated re-export of `d`. -/
abbrev d := SuperNeo.d

/-- [Role: Theorem-Target] Curated re-export of `nF`. -/
abbrev nF := SuperNeo.nF

/-- [Role: Theorem-Target] Curated re-export of `nFIn`. -/
abbrev nFIn := SuperNeo.nFIn

/-- [Role: Theorem-Target] Curated re-export of `goldilocksShapeSanity`. -/
abbrev goldilocksShapeSanity := SuperNeo.goldilocksShapeSanity

/-- [Role: Theorem-Target] Curated re-export of `goldilocksShapeProp`. -/
abbrev goldilocksShapeProp := SuperNeo.goldilocksShapeProp

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Curated theorem surface `goldilocksShapeSanity_sound`. -/
abbrev goldilocksShapeSanity_sound := SuperNeo.goldilocksShapeSanity_sound

/-- [Role: Theorem-Target] Curated theorem surface `goldilocksShape`. -/
abbrev goldilocksShape := SuperNeo.goldilocksShape

/-- [Role: Theorem-Target] Curated theorem surface `eta_eq_81`. -/
abbrev eta_eq_81 := SuperNeo.eta_eq_81

/-- [Role: Theorem-Target] Curated theorem surface `d_eq_54`. -/
abbrev d_eq_54 := SuperNeo.d_eq_54

/-- [Role: Theorem-Target] Curated theorem surfaces for positivity. -/
abbrev eta_pos := SuperNeo.eta_pos
abbrev d_pos := SuperNeo.d_pos

/-- [Role: Theorem-Target] Curated theorem surface `nF_def`. -/
abbrev nF_def := SuperNeo.nF_def

/-- [Role: Theorem-Target] Curated theorem surface `nFIn_def`. -/
abbrev nFIn_def := SuperNeo.nFIn_def

/-- [Role: Theorem-Target] Curated rewrite surfaces for `nF`/`nFIn`. -/
abbrev nF_zero := SuperNeo.nF_zero
abbrev nF_one := SuperNeo.nF_one
abbrev nF_two := SuperNeo.nF_two
abbrev nF_add := SuperNeo.nF_add
abbrev nF_mul := SuperNeo.nF_mul
abbrev nFIn_zero := SuperNeo.nFIn_zero
abbrev nFIn_one := SuperNeo.nFIn_one
abbrev nFIn_add := SuperNeo.nFIn_add
abbrev nFIn_mul := SuperNeo.nFIn_mul
abbrev nF_eq_54_mul := SuperNeo.nF_eq_54_mul
abbrev nFIn_eq_54_mul := SuperNeo.nFIn_eq_54_mul

theorem nF_pos_of_pos {nR : Nat} (h : 0 < nR) : 0 < nF nR :=
  SuperNeo.nF_pos_of_pos h

theorem nFIn_pos_of_pos {nRIn : Nat} (h : 0 < nRIn) : 0 < nFIn nRIn :=
  SuperNeo.nFIn_pos_of_pos h

theorem nF_mono {a b : Nat} (h : a ≤ b) : nF a ≤ nF b :=
  SuperNeo.nF_mono h

theorem nFIn_mono {a b : Nat} (h : a ≤ b) : nFIn a ≤ nFIn b :=
  SuperNeo.nFIn_mono h

end DimensionsInterface

end SuperNeo
