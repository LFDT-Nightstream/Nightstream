import SuperNeo.MatrixTransform

/-!
Contract interface for `SuperNeo.MatrixTransform`.

Spec: `specs/MatrixTransform.spec.md`

Paper anchors:
- Theorem 4 (Matrix-Vector Product Transform), Section 5, lines 384-386.
- Appendix D.1: proof via block decomposition of Theorem 3.
-/

namespace SuperNeo

namespace MatrixTransformInterface

/-! ## Core Surfaces -/

/-- [Role: Definitional] Curated re-export of `dotVec`. -/
abbrev dotVec := SuperNeo.dotVec

/-- [Role: Definitional] Curated re-export of `matrixVecDirect`. -/
abbrev matrixVecDirect := SuperNeo.matrixVecDirect

/-- [Role: Definitional] Curated re-export of `ctBarDot`. -/
abbrev ctBarDot := SuperNeo.ctBarDot

/-- [Role: Definitional] Curated re-export of `extractBlock`. -/
abbrev extractBlock := SuperNeo.extractBlock

/-- [Role: Definitional] Curated re-export of `ringBlockDot`. -/
abbrev ringBlockDot := SuperNeo.ringBlockDot

/-- [Role: Definitional] Curated re-export of `matrixVecCtBar`. -/
abbrev matrixVecCtBar := SuperNeo.matrixVecCtBar

/-- [Role: Definitional] Curated re-export of `MatrixRowsCompatible`. -/
abbrev MatrixRowsCompatible := SuperNeo.MatrixRowsCompatible

/-- [Role: Definitional] Curated re-export of `matrixTransformIdentity`. -/
abbrev matrixTransformIdentity := SuperNeo.matrixTransformIdentity

/-- [Role: Definitional] Curated re-export of `matrixTransformIdentityProp`. -/
abbrev matrixTransformIdentityProp := SuperNeo.matrixTransformIdentityProp

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Curated theorem surface `matrixVecDirect_size`. -/
abbrev matrixVecDirect_size := SuperNeo.matrixVecDirect_size

/-- [Role: Theorem-Target] Curated theorem surface `matrixVecCtBar_size`. -/
abbrev matrixVecCtBar_size := SuperNeo.matrixVecCtBar_size

/-- [Role: Theorem-Target] Dot/inner-product equivalence used by theorem-native P12 derivation. -/
abbrev dotVec_eq_innerProduct := SuperNeo.dotVec_eq_innerProduct

/-- [Role: Theorem-Target] Theorem-native P12 derivation from Theorem-3. -/
theorem matrixTransformEq_of_thm3CoreAssumption
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hThm3 : thm3CoreAssumption bar)
  (hRows : MatrixRowsCompatible m z) :
  matrixVecDirect m z = matrixVecCtBar bar m z :=
  SuperNeo.matrixTransformEq_of_thm3CoreAssumption hThm3 hRows

/-- [Role: Theorem-Target] Curated theorem surface `matrixTransformIdentity_sound`. -/
theorem matrixTransformIdentity_sound
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hOk : matrixTransformIdentity bar m z = true) :
  matrixTransformIdentityProp bar m z :=
  SuperNeo.matrixTransformIdentity_sound hOk

/-- [Role: Theorem-Target] Curated theorem surface `matrixTransformIdentity_complete`. -/
theorem matrixTransformIdentity_complete
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hProp : matrixTransformIdentityProp bar m z) :
  matrixTransformIdentity bar m z = true :=
  SuperNeo.matrixTransformIdentity_complete hProp

/-- [Role: Theorem-Target] Curated theorem surface `matrixTransformIdentity_iff_prop`. -/
theorem matrixTransformIdentity_iff_prop
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F} :
  matrixTransformIdentity bar m z = true ↔ matrixTransformIdentityProp bar m z :=
  SuperNeo.matrixTransformIdentity_iff_prop

theorem matrixTransformIdentity_sound_full
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hOk : matrixTransformIdentity bar m z = true) :
  MatrixRowsCompatible m z ∧ matrixVecDirect m z = matrixVecCtBar bar m z :=
  SuperNeo.matrixTransformIdentity_sound_full hOk

theorem matrixTransformIdentity_complete_of_rowsCompatible
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hRows : MatrixRowsCompatible m z)
  (hEq : matrixVecDirect m z = matrixVecCtBar bar m z) :
  matrixTransformIdentity bar m z = true :=
  SuperNeo.matrixTransformIdentity_complete_of_rowsCompatible hRows hEq

/-! ## Boundary Surfaces -/

/-- [Role: Boundary] Theorem-facing matrix-transform boundary surface. -/
abbrev matrixTransformAssumption := SuperNeo.matrixTransformAssumption

/-- [Role: Boundary] Check-facing matrix-transform boundary surface. -/
abbrev matrixTransformCheckAssumption := SuperNeo.matrixTransformCheckAssumption

/-- [Role: Theorem-Target] Theorem-native P12 boundary constructor from Theorem-3. -/
theorem matrixTransformAssumption_of_thm3CoreAssumption
  {bar : Array (Array F)} {m : Array (Array F)}
  (hThm3 : thm3CoreAssumption bar) :
  matrixTransformAssumption bar m :=
  SuperNeo.matrixTransformAssumption_of_thm3CoreAssumption hThm3

/-- [Role: Theorem-Target] Theorem-native P12 boundary constructor from `P10` only. -/
theorem matrixTransformAssumption_of_p10
  {bar : Array (Array F)} {m : Array (Array F)}
  (hThm3 : thm3CoreAssumption bar) :
  matrixTransformAssumption bar m :=
  SuperNeo.matrixTransformAssumption_of_p10 hThm3

/-- [Role: Theorem-Target] Theorem-native P12 constructor from the finite basis-kernel witness. -/
theorem matrixTransformAssumption_of_basisKernelAssumption
  {bar : Array (Array F)} {m : Array (Array F)}
  (hBasis : thm3BasisKernelAssumption bar) :
  matrixTransformAssumption bar m :=
  SuperNeo.matrixTransformAssumption_of_basisKernelAssumption hBasis

/-- [Role: Theorem-Target] Theorem-native P12 constructor from the finite basis-kernel checker. -/
theorem matrixTransformAssumption_of_basisKernelCheck
  {bar : Array (Array F)} {m : Array (Array F)}
  (hCheck : thm3BasisKernelCheck bar = true) :
  matrixTransformAssumption bar m :=
  SuperNeo.matrixTransformAssumption_of_basisKernelCheck hCheck

/-- [Role: Theorem-Target] Theorem-native P12 boundary constructor from `(P10 + P11)`. -/
theorem matrixTransformAssumption_of_p10_p11
  {bar : Array (Array F)} {m : Array (Array F)}
  (hThm3 : thm3CoreAssumption bar)
  (hLift : barLiftLinearityAssumption bar) :
  matrixTransformAssumption bar m :=
  SuperNeo.matrixTransformAssumption_of_p10_p11 hThm3 hLift

/-- [Role: Theorem-Target] Conversion from theorem-facing to check-facing boundary. -/
theorem matrixTransformCheckAssumption_of_assumption
  {bar : Array (Array F)} {m : Array (Array F)}
  (hAssm : matrixTransformAssumption bar m) :
  matrixTransformCheckAssumption bar m :=
  SuperNeo.matrixTransformCheckAssumption_of_assumption hAssm

/-- [Role: Theorem-Target] Conversion from check-facing to theorem-facing boundary. -/
theorem matrixTransformAssumption_of_checkAssumption
  {bar : Array (Array F)} {m : Array (Array F)}
  (hCheck : matrixTransformCheckAssumption bar m) :
  matrixTransformAssumption bar m :=
  SuperNeo.matrixTransformAssumption_of_checkAssumption hCheck

/-- [Role: Theorem-Target] Equivalence between theorem and check boundaries. -/
theorem matrixTransformAssumption_iff_checkAssumption
  {bar : Array (Array F)} {m : Array (Array F)} :
  matrixTransformAssumption bar m ↔ matrixTransformCheckAssumption bar m :=
  SuperNeo.matrixTransformAssumption_iff_checkAssumption

end MatrixTransformInterface

end SuperNeo
