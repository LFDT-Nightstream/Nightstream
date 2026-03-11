import SuperNeo.MatrixTransform

/-!
Evaluation-link layer (Remark 2 style, P13 core).

Reading guide:
1. `evalLinkIdentity` is the executable check surface.
2. `evalLinkIdentityProp` is the proposition it checks.
3. `evalLinkAssumption` is the theorem-facing contract for downstream proofs.
4. `_of_assumption`, `_of_checkAssumption`, and `_iff_...` bridge check/prop.
-/

namespace SuperNeo

/-- Check surface for eval-link identity. -/
def evalLinkIdentity (bar : Array (Array F)) (m : Array (Array F)) (z : Array F) : Bool :=
  matrixTransformIdentity bar m z

/-- Proposition-level counterpart for eval-link identity. -/
def evalLinkIdentityProp (bar : Array (Array F)) (m : Array (Array F)) (z : Array F) : Prop :=
  matrixTransformIdentityProp bar m z

/--
Theorem-facing eval-link contract.

Downstream theorem files should consume this Prop boundary directly.
-/
def evalLinkAssumption (bar : Array (Array F)) (m : Array (Array F)) : Prop :=
  ∀ z : Array F, MatrixRowsCompatible m z -> evalLinkIdentityProp bar m z

/-- Check-facing eval-link contract retained for executable compatibility. -/
def evalLinkCheckAssumption (bar : Array (Array F)) (m : Array (Array F)) : Prop :=
  ∀ z : Array F, MatrixRowsCompatible m z -> evalLinkIdentity bar m z = true

theorem evalLinkIdentity_sound
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hOk : evalLinkIdentity bar m z = true) :
  evalLinkIdentityProp bar m z := by
  exact matrixTransformIdentity_sound hOk

theorem evalLinkIdentity_complete
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F}
  (hProp : evalLinkIdentityProp bar m z) :
  evalLinkIdentity bar m z = true := by
  exact matrixTransformIdentity_complete hProp

theorem evalLinkIdentity_iff_prop
  {bar : Array (Array F)} {m : Array (Array F)} {z : Array F} :
  evalLinkIdentity bar m z = true ↔ evalLinkIdentityProp bar m z := by
  exact matrixTransformIdentity_iff_prop

/-- Convert check-facing eval-link contract into theorem-facing form. -/
theorem evalLinkAssumption_of_checkAssumption
  {bar : Array (Array F)} {m : Array (Array F)}
  (hCheck : evalLinkCheckAssumption bar m) :
  evalLinkAssumption bar m := by
  intro z hRows
  exact evalLinkIdentity_sound (hCheck z hRows)

/-- Convert theorem-facing eval-link contract into check-facing form. -/
theorem evalLinkCheckAssumption_of_assumption
  {bar : Array (Array F)} {m : Array (Array F)}
  (hAssm : evalLinkAssumption bar m) :
  evalLinkCheckAssumption bar m := by
  intro z hRows
  exact evalLinkIdentity_complete (hAssm z hRows)

/-- Equivalence between theorem-facing and check-facing eval-link contracts. -/
theorem evalLinkAssumption_iff_checkAssumption
  {bar : Array (Array F)} {m : Array (Array F)} :
  evalLinkAssumption bar m ↔ evalLinkCheckAssumption bar m := by
  constructor
  · exact evalLinkCheckAssumption_of_assumption
  · exact evalLinkAssumption_of_checkAssumption

/-- Native eval-link assumption from native matrix-transform assumption. -/
theorem evalLinkAssumption_of_matrixTransformAssumption
  {bar : Array (Array F)} {m : Array (Array F)}
  (hMatrix : matrixTransformAssumption bar m) :
  evalLinkAssumption bar m := by
  intro z hRows
  exact ⟨hRows, hMatrix z hRows⟩

/-- Theorem-native `P13` constructor from Theorem-3 (`P10`) through `P12`. -/
theorem evalLinkAssumption_of_thm3CoreAssumption
  {bar : Array (Array F)} {m : Array (Array F)}
  (hThm3 : thm3CoreAssumption bar) :
  evalLinkAssumption bar m := by
  exact evalLinkAssumption_of_matrixTransformAssumption
    (matrixTransformAssumption_of_thm3CoreAssumption hThm3)

/-- Theorem-native `P13` constructor from `P10` only. -/
theorem evalLinkAssumption_of_p10
  {bar : Array (Array F)} {m : Array (Array F)}
  (hThm3 : thm3CoreAssumption bar) :
  evalLinkAssumption bar m := by
  exact evalLinkAssumption_of_matrixTransformAssumption
    (matrixTransformAssumption_of_p10 hThm3)

/--
Theorem-native `P13` constructor from `(P10 + P11)` through `P12`.

`P11` is carried explicitly to keep theorem dependency accounting aligned.
-/
theorem evalLinkAssumption_of_p10_p11
  {bar : Array (Array F)} {m : Array (Array F)}
  (hThm3 : thm3CoreAssumption bar)
  (_hLift : barLiftLinearityAssumption bar) :
  evalLinkAssumption bar m := by
  exact evalLinkAssumption_of_p10 hThm3

/--
Native Theorem-3 path for P13: build eval-link assumption directly from the
canonical native bar matrix, without explicit `hThm3`.
-/
theorem evalLinkAssumption_native
  (m : Array (Array F)) :
  evalLinkAssumption nativeBarMatrix m := by
  exact evalLinkAssumption_of_matrixTransformAssumption
    (matrixTransformAssumption_native m)

/-- Rewrite helper for deriving native P13 under `bar = nativeBarMatrix`. -/
theorem evalLinkAssumption_of_bar_eq_native
  {bar : Array (Array F)} {m : Array (Array F)}
  (hBar : bar = nativeBarMatrix) :
  evalLinkAssumption bar m := by
  subst hBar
  exact evalLinkAssumption_native m


end SuperNeo
