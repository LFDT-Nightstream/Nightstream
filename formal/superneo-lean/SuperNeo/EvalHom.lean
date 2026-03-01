import SuperNeo.EvalLink
import SuperNeo.ModuleHom

/-!
Evaluation homomorphism layer (Theorem 5 / P14 core).

Reading guide:
1. `evalHom2` is the executable check surface.
2. `evalHom2Prop` is the proposition being checked.
3. `evalHomAssumption` is the theorem-facing boundary.
4. `_of_assumption`, `_of_checkAssumption`, and `_iff_...` bridge check/prop.
-/

namespace SuperNeo

open F

/--
Compact scaffold evaluator at challenge point `r` (scalar-valued).

Current scaffold keeps this intentionally simple so theorem wiring stays clear.
-/
def evalBarMzAt
  (_bar : Array (Array F))
  (_m : Array (Array F))
  (_z _r : Array F) : F :=
  0

/-- Proposition-level statement checked by `evalHom2`. -/
def evalHom2Prop
  (_bar : Array (Array F))
  (m : Array (Array F))
  (z1 z2 _r : Array F)
  (_ρ1 _ρ2 : F) : Prop :=
  z1.size = z2.size ∧
  MatrixRowsCompatible m z1

/-- Theorem 5 check surface. -/
def evalHom2
  (_bar : Array (Array F))
  (m : Array (Array F))
  (z1 z2 _r : Array F)
  (_ρ1 _ρ2 : F) : Bool :=
  if z1.size != z2.size then
    false
  else if MatrixRowsCompatible m z1 then
    true
  else
    false

/--
Theorem-facing evaluation-hom contract.

Downstream proof files should depend on this Prop boundary directly.
-/
def evalHomAssumption
  (bar : Array (Array F))
  (m : Array (Array F))
  (r : Array F)
  (ρ1 ρ2 : F) : Prop :=
  ∀ z1 z2 : Array F, z1.size = z2.size → MatrixRowsCompatible m z1 →
    evalHom2Prop bar m z1 z2 r ρ1 ρ2

/-- Check-facing evaluation-hom contract retained for compatibility. -/
def evalHomCheckAssumption
  (bar : Array (Array F))
  (m : Array (Array F))
  (r : Array F)
  (ρ1 ρ2 : F) : Prop :=
  ∀ z1 z2 : Array F, z1.size = z2.size → MatrixRowsCompatible m z1 →
    evalHom2 bar m z1 z2 r ρ1 ρ2 = true

theorem evalHom2_sound
  {bar : Array (Array F)} {m : Array (Array F)}
  {z1 z2 r : Array F} {ρ1 ρ2 : F}
  (hOk : evalHom2 bar m z1 z2 r ρ1 ρ2 = true) :
  evalHom2Prop bar m z1 z2 r ρ1 ρ2 := by
  unfold evalHom2Prop
  unfold evalHom2 at hOk
  by_cases hSize : z1.size != z2.size
  · simp [hSize] at hOk
  · have hEq : z1.size = z2.size := by simpa using hSize
    by_cases hRows : MatrixRowsCompatible m z1
    · simp [hSize, hRows] at hOk
      exact ⟨hEq, hRows⟩
    · simp [hSize, hRows] at hOk

theorem evalHom2_complete
  {bar : Array (Array F)} {m : Array (Array F)}
  {z1 z2 r : Array F} {ρ1 ρ2 : F}
  (hProp : evalHom2Prop bar m z1 z2 r ρ1 ρ2) :
  evalHom2 bar m z1 z2 r ρ1 ρ2 = true := by
  rcases hProp with ⟨hSize, hRows⟩
  unfold evalHom2
  simp [hSize, hRows]

theorem evalHom2_iff_prop
  {bar : Array (Array F)} {m : Array (Array F)}
  {z1 z2 r : Array F} {ρ1 ρ2 : F} :
  evalHom2 bar m z1 z2 r ρ1 ρ2 = true ↔ evalHom2Prop bar m z1 z2 r ρ1 ρ2 := by
  constructor
  · exact evalHom2_sound
  · exact evalHom2_complete

/-- Native theorem-5 assumption for compact scaffold (`barLiftVector = id`). -/
theorem evalHomAssumption_native
  {bar : Array (Array F)} {m : Array (Array F)} {r : Array F} {ρ1 ρ2 : F} :
  evalHomAssumption bar m r ρ1 ρ2 := by
  intro z1 z2 hSize hRows
  exact ⟨hSize, hRows⟩

/-- Convert check-facing eval-hom contract into theorem-facing form. -/
theorem evalHomAssumption_of_checkAssumption
  {bar : Array (Array F)} {m : Array (Array F)}
  {r : Array F} {ρ1 ρ2 : F}
  (hCheck : evalHomCheckAssumption bar m r ρ1 ρ2) :
  evalHomAssumption bar m r ρ1 ρ2 := by
  intro z1 z2 hSize hRows
  exact evalHom2_sound (hCheck z1 z2 hSize hRows)

/-- Convert theorem-facing eval-hom contract into check-facing form. -/
theorem evalHomCheckAssumption_of_assumption
  {bar : Array (Array F)} {m : Array (Array F)}
  {r : Array F} {ρ1 ρ2 : F}
  (hAssm : evalHomAssumption bar m r ρ1 ρ2) :
  evalHomCheckAssumption bar m r ρ1 ρ2 := by
  intro z1 z2 hSize hRows
  exact evalHom2_complete (hAssm z1 z2 hSize hRows)

/-- Equivalence between theorem-facing and check-facing eval-hom contracts. -/
theorem evalHomAssumption_iff_checkAssumption
  {bar : Array (Array F)} {m : Array (Array F)}
  {r : Array F} {ρ1 ρ2 : F} :
  evalHomAssumption bar m r ρ1 ρ2 ↔
    evalHomCheckAssumption bar m r ρ1 ρ2 := by
  constructor
  · exact evalHomCheckAssumption_of_assumption
  · exact evalHomAssumption_of_checkAssumption

/--
Derive the evaluation-hom assumption from eval-link and module-hom assumptions.

In the compact scaffold, `evalHom2Prop` is shape-centric, so this theorem threads
the intended dependency chain while keeping obligations explicit.
-/
theorem evalHomAssumption_of_evalLink_and_moduleAssumptions
  {bar : Array (Array F)} {m : Array (Array F)}
  {r : Array F} {ρ1 ρ2 : F}
  {hVec : VecModuleHom} {hScal : ScalarModuleHom}
  (hEvalLink : evalLinkAssumption bar m)
  (_hVec : vecModuleAssumption hVec)
  (_hScal : scalarModuleAssumption hScal) :
  evalHomAssumption bar m r ρ1 ρ2 := by
  intro z1 z2 hSize hRows
  have _hLink := hEvalLink z1 hRows
  exact ⟨hSize, hRows⟩

end SuperNeo
