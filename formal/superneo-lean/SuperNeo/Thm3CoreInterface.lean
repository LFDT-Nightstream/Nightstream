import SuperNeo.Thm3Core

/-!
Contract interface for `SuperNeo.Thm3Core`.

Spec: `specs/Thm3Core.spec.md`

Paper anchors:
- Theorem 3 (Inner Product Transform), Section 5, lines 368-372.
-/

namespace SuperNeo

namespace Thm3CoreInterface

/-! ## Core Surfaces -/

/-- [Role: Definitional] Curated re-export of `innerProduct`. -/
abbrev innerProduct := SuperNeo.innerProduct

/-- [Role: Definitional] Curated re-export of `IsDVec`. -/
abbrev IsDVec := SuperNeo.IsDVec

/-- [Role: Definitional] Curated re-export of `IsDBarMatrix`. -/
abbrev IsDBarMatrix := SuperNeo.IsDBarMatrix

/-- [Role: Definitional] Curated re-export of `p10CoreProp`. -/
abbrev p10CoreProp := SuperNeo.p10CoreProp

/-- [Role: Definitional] Curated re-export of `p10CoreCheck`. -/
abbrev p10CoreCheck := SuperNeo.p10CoreCheck

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Curated theorem surface `p10CoreCheck_sound`. -/
theorem p10CoreCheck_sound
    {bar : Array (Array F)} {a b : Array F}
    (hOk : p10CoreCheck bar a b = true) :
    p10CoreProp bar a b :=
  SuperNeo.p10CoreCheck_sound hOk

/-- [Role: Theorem-Target] Curated theorem surface `p10CoreCheck_complete`. -/
theorem p10CoreCheck_complete
    {bar : Array (Array F)} {a b : Array F}
    (hProp : p10CoreProp bar a b) :
    p10CoreCheck bar a b = true :=
  SuperNeo.p10CoreCheck_complete hProp

/-- [Role: Theorem-Target] Curated theorem surface `p10Core_of_preconditions`. -/
theorem p10Core_of_preconditions
    {bar : Array (Array F)} {a b : Array F}
    (hBar : IsDBarMatrix bar)
    (hA : IsDVec a)
    (hB : IsDVec b)
    (hCheck : p10CoreCheck bar a b = true) :
    p10CoreProp bar a b :=
  SuperNeo.p10Core_of_preconditions hBar hA hB hCheck

/-- [Role: Theorem-Target] Curated theorem surface `p10Core_of_preconditions_props`. -/
theorem p10Core_of_preconditions_props
    {bar : Array (Array F)} {a b : Array F}
    (hBar : IsDBarMatrix bar)
    (hA : IsDVec a)
    (hB : IsDVec b)
    (hThm3 : thm3CoreAssumption bar) :
    p10CoreProp bar a b :=
  SuperNeo.p10Core_of_preconditions_props hBar hA hB hThm3

/-! ## Boundary Surfaces -/

/-- [Role: Boundary] Theorem-3 boundary: ct(mulRqPhi(bar(a), b)) = ⟨a, b⟩ for d-sized blocks. -/
abbrev thm3CoreAssumption := SuperNeo.thm3CoreAssumption

/-- [Role: Boundary] Finite basis-kernel criterion equivalent to `thm3CoreAssumption`. -/
abbrev thm3BasisKernelAssumption := SuperNeo.thm3BasisKernelAssumption

/-- [Role: Theorem-Target] Native basis-kernel closure on the canonical bar matrix. -/
theorem thm3BasisKernelAssumption_native :
    thm3BasisKernelAssumption SuperNeo.nativeBarMatrix :=
  SuperNeo.thm3BasisKernelAssumption_native

/-- [Role: Theorem-Target] Lift finite basis-kernel criterion to full Theorem-3 boundary. -/
theorem thm3CoreAssumption_of_basisKernelAssumption
    {bar : Array (Array F)}
    (hBasis : thm3BasisKernelAssumption bar) :
    thm3CoreAssumption bar :=
  SuperNeo.thm3CoreAssumption_of_basisKernelAssumption hBasis

/-- [Role: Theorem-Target] Restrict full Theorem-3 boundary to finite basis-kernel criterion. -/
theorem thm3BasisKernelAssumption_of_thm3CoreAssumption
    {bar : Array (Array F)}
    (hThm3 : thm3CoreAssumption bar) :
    thm3BasisKernelAssumption bar :=
  SuperNeo.thm3BasisKernelAssumption_of_thm3CoreAssumption hThm3

/-- [Role: Theorem-Target] Equivalence between full Theorem-3 boundary and finite basis-kernel criterion. -/
theorem thm3CoreAssumption_iff_basisKernelAssumption
    {bar : Array (Array F)} :
    thm3CoreAssumption bar ↔ thm3BasisKernelAssumption bar :=
  SuperNeo.thm3CoreAssumption_iff_basisKernelAssumption

/-- [Role: Definitional] Executable finite checker for `thm3BasisKernelAssumption`. -/
noncomputable abbrev thm3BasisKernelCheck := SuperNeo.thm3BasisKernelCheck

/-- [Role: Theorem-Target] Boolean checker correctness for `thm3BasisKernelAssumption`. -/
theorem thm3BasisKernelCheck_eq_true_iff
    {bar : Array (Array F)} :
    thm3BasisKernelCheck bar = true ↔ thm3BasisKernelAssumption bar :=
  SuperNeo.thm3BasisKernelCheck_eq_true_iff

/-- [Role: Theorem-Target] Recover basis-kernel assumption from checker success. -/
theorem thm3BasisKernelAssumption_of_check
    {bar : Array (Array F)}
    (hCheck : thm3BasisKernelCheck bar = true) :
    thm3BasisKernelAssumption bar :=
  SuperNeo.thm3BasisKernelAssumption_of_check hCheck

/-- [Role: Theorem-Target] Recover full Theorem-3 boundary from checker success. -/
theorem thm3CoreAssumption_of_basisKernelCheck
    {bar : Array (Array F)}
    (hCheck : thm3BasisKernelCheck bar = true) :
    thm3CoreAssumption bar :=
  SuperNeo.thm3CoreAssumption_of_basisKernelCheck hCheck

/-- [Role: Theorem-Target] Equivalence between Theorem-3 boundary and finite checker success. -/
theorem thm3CoreAssumption_iff_basisKernelCheck
    {bar : Array (Array F)} :
    thm3CoreAssumption bar ↔ thm3BasisKernelCheck bar = true :=
  SuperNeo.thm3CoreAssumption_iff_basisKernelCheck

/-- [Role: Theorem-Target] Native checker closure on the canonical bar matrix. -/
theorem thm3BasisKernelCheck_native :
    thm3BasisKernelCheck SuperNeo.nativeBarMatrix = true :=
  SuperNeo.thm3BasisKernelCheck_native

/-- [Role: Boundary] Compatibility alias for Theorem-3 wrapper naming (refs [36,64]). -/
abbrev thm3CoreAssumption_ref36_64 := SuperNeo.thm3CoreAssumption_ref36_64

/-- [Role: Theorem-Target] Bridge from ref-named compatibility alias to canonical Theorem-3 boundary. -/
theorem thm3CoreAssumption_of_ref36_64
    {bar : Array (Array F)}
    (hRef : thm3CoreAssumption_ref36_64 bar) :
    thm3CoreAssumption bar :=
  SuperNeo.thm3CoreAssumption_of_ref36_64 hRef

/-- [Role: Theorem-Target] View canonical Theorem-3 boundary via the ref-named compatibility alias. -/
theorem thm3CoreAssumption_ref36_64_of_assumption
    {bar : Array (Array F)}
    (hThm3 : thm3CoreAssumption bar) :
    thm3CoreAssumption_ref36_64 bar :=
  SuperNeo.thm3CoreAssumption_ref36_64_of_assumption hThm3

/-- [Role: Theorem-Target] Derive P10 from Theorem-3 boundary and vector shape. -/
theorem p10Core_of_assumption
    {bar : Array (Array F)} {a b : Array F}
    (hThm3 : thm3CoreAssumption bar)
    (hA : IsDVec a)
    (hB : IsDVec b) :
    p10CoreProp bar a b :=
  SuperNeo.p10Core_of_assumption hThm3 hA hB

end Thm3CoreInterface

end SuperNeo
