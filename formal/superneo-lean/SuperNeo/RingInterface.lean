import SuperNeo.Ring

/-!
Contract interface for `SuperNeo.Ring`.

Spec: `specs/Ring.spec.md`

Paper anchors:
- Definition 1, Section 4, lines 275-282: `R_F = F[X]/Φ(X)`, degree `d`.
- Section 7.3, lines 440-470: folding linear combination `z' = ρ₁·z₁ + ρ₂·z₂`.
-/

namespace SuperNeo

namespace RingInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `d`. -/
abbrev d := SuperNeo.d

/-- [Role: Theorem-Target] Curated re-export of `Coeffs`. -/
abbrev Coeffs := SuperNeo.Coeffs

/-- [Role: Theorem-Target] Curated re-export of `vecAdd`. -/
abbrev vecAdd := SuperNeo.vecAdd

/-- [Role: Theorem-Target] Curated re-export of `vecScale`. -/
abbrev vecScale := SuperNeo.vecScale

/-- [Role: Theorem-Target] Curated re-export of `linComb2Vec`. -/
abbrev linComb2Vec := SuperNeo.linComb2Vec

/-- [Role: Theorem-Target] Curated re-export of `ct`. -/
abbrev ct := SuperNeo.ct

/-- [Role: Theorem-Target] Curated re-export of `coeffAt`. -/
abbrev coeffAt := SuperNeo.coeffAt

/-- [Role: Theorem-Target] Curated re-export of `mulRq`. -/
abbrev mulRq := SuperNeo.mulRq

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Curated theorem surface `d_pos`. -/
theorem d_pos : 0 < d := SuperNeo.d_pos

/-- [Role: Theorem-Target] Curated theorem surface `vecScale_size`. -/
theorem vecScale_size (s : F) (a : Array F) :
    (vecScale s a).size = a.size := SuperNeo.vecScale_size s a

/-- [Role: Theorem-Target] Curated theorem surface `vecAdd_size_of_eq`. -/
theorem vecAdd_size_of_eq {a b : Array F} (h : a.size = b.size) :
    (vecAdd a b).size = a.size := SuperNeo.vecAdd_size_of_eq h

/-- [Role: Theorem-Target] Curated theorem surface `vecAdd_size_of_ne`. -/
theorem vecAdd_size_of_ne {a b : Array F} (h : a.size ≠ b.size) :
    (vecAdd a b).size = 0 := SuperNeo.vecAdd_size_of_ne h

/-- [Role: Theorem-Target] Curated theorem surface `linComb2Vec_size_of_eq`. -/
theorem linComb2Vec_size_of_eq
    {ρ1 ρ2 : F} {z1 z2 : Array F} (h : z1.size = z2.size) :
    (linComb2Vec ρ1 ρ2 z1 z2).size = z1.size :=
  SuperNeo.linComb2Vec_size_of_eq h

/-- [Role: Theorem-Target] Curated theorem surface `mulRq_size`. -/
theorem mulRq_size (a b : Coeffs) : (mulRq a b).size = d := SuperNeo.mulRq_size a b

/-- [Role: Theorem-Target] Curated theorem surface `mulRq_vecAdd_right`. -/
theorem mulRq_vecAdd_right
    (a b c : Coeffs)
    (hb : b.size = d)
    (hc : c.size = d) :
    mulRq a (vecAdd b c) = vecAdd (mulRq a b) (mulRq a c) :=
  SuperNeo.mulRq_vecAdd_right a b c hb hc

/-- [Role: Theorem-Target] Curated theorem surface `mulRq_vecAdd_left`. -/
theorem mulRq_vecAdd_left
    (a b c : Coeffs)
    (hb : b.size = d)
    (hc : c.size = d) :
    mulRq (vecAdd b c) a = vecAdd (mulRq b a) (mulRq c a) :=
  SuperNeo.mulRq_vecAdd_left a b c hb hc

/-- [Role: Theorem-Target] Curated theorem surface `hasRingDegreeShape_zeroRq`. -/
theorem hasRingDegreeShape_zeroRq : SuperNeo.hasRingDegreeShape SuperNeo.zeroRq :=
  SuperNeo.hasRingDegreeShape_zeroRq

/-- [Role: Theorem-Target] Curated theorem surface `hasRingDegreeShape_oneRq`. -/
theorem hasRingDegreeShape_oneRq : SuperNeo.hasRingDegreeShape SuperNeo.oneRq :=
  SuperNeo.hasRingDegreeShape_oneRq

/-- [Role: Theorem-Target] Curated theorem surface `hasRingDegreeShape_mulRq`. -/
theorem hasRingDegreeShape_mulRq (a b : Coeffs) :
    SuperNeo.hasRingDegreeShape (mulRq a b) := SuperNeo.hasRingDegreeShape_mulRq a b

/-- [Role: Theorem-Target] Curated theorem surface `ct_zeroRq`. -/
theorem ct_zeroRq : ct SuperNeo.zeroRq = 0 := SuperNeo.ct_zeroRq

/-- [Role: Theorem-Target] Curated theorem surface `ct_oneRq`. -/
theorem ct_oneRq : ct SuperNeo.oneRq = 1 := SuperNeo.ct_oneRq

/-- [Role: Theorem-Target] Curated theorem surface `coeffAt_zeroRq`. -/
theorem coeffAt_zeroRq (i : Nat) : coeffAt SuperNeo.zeroRq i = 0 := SuperNeo.coeffAt_zeroRq i

/-- [Role: Theorem-Target] Curated theorem surface `ringMulShapeProp_of_shapes`. -/
theorem ringMulShapeProp_of_shapes
    {a b : Coeffs}
    (ha : SuperNeo.hasRingDegreeShape a)
    (hb : SuperNeo.hasRingDegreeShape b) :
    SuperNeo.ringMulShapeProp a b :=
  SuperNeo.ringMulShapeProp_of_shapes ha hb

/-- [Role: Theorem-Target] Curated theorem surface `ringMulShapeProp_left`. -/
theorem ringMulShapeProp_left
    {a b : Coeffs}
    (h : SuperNeo.ringMulShapeProp a b) :
    SuperNeo.hasRingDegreeShape a := SuperNeo.ringMulShapeProp_left h

/-- [Role: Theorem-Target] Curated theorem surface `ringMulShapeProp_right`. -/
theorem ringMulShapeProp_right
    {a b : Coeffs}
    (h : SuperNeo.ringMulShapeProp a b) :
    SuperNeo.hasRingDegreeShape b := SuperNeo.ringMulShapeProp_right h

end RingInterface

end SuperNeo
