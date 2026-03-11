import SuperNeo.Ring

/-!
Module-hom interfaces (P15 core).

Reading guide:
1. `VecModuleHom` / `ScalarModuleHom` define map signatures.
2. `...PropPair` captures one-point linearity obligations.
3. `...CheckPair` is the executable check wrapper of that proposition.
4. `...Assumption` is the theorem-facing universal contract.
5. `...CheckAssumption` is the check-facing universal contract.
6. `_of_assumption` / `_of_checkAssumption` convert between both surfaces.
-/

namespace SuperNeo

open F

/-- Vector-valued module hom interface. -/
structure VecModuleHom where
  map : Array F → Array F

/-- Scalar-valued module hom interface. -/
structure ScalarModuleHom where
  map : Array F → F

/-- Proposition pair for one vector-linearity check point `(s, x, y)`. -/
def vecModulePropPair (h : VecModuleHom) (s : F) (x y : Array F) : Prop :=
  h.map (vecAdd x y) = vecAdd (h.map x) (h.map y) ∧
    h.map (vecScale s x) = vecScale s (h.map x)

/-- Proposition pair for one scalar-linearity check point `(s, x, y)`. -/
def scalarModulePropPair (h : ScalarModuleHom) (s : F) (x y : Array F) : Prop :=
  h.map (vecAdd x y) = h.map x + h.map y ∧
    h.map (vecScale s x) = s * h.map x

instance vecModulePropPair_decidable (h : VecModuleHom) (s : F) (x y : Array F) :
    Decidable (vecModulePropPair h s x y) := by
  unfold vecModulePropPair
  infer_instance

instance scalarModulePropPair_decidable (h : ScalarModuleHom) (s : F) (x y : Array F) :
    Decidable (scalarModulePropPair h s x y) := by
  unfold scalarModulePropPair
  infer_instance

/-- Executable check surface for vector module linearity pair. -/
def vecModuleCheckPair (h : VecModuleHom) (s : F) (x y : Array F) : Bool :=
  decide (vecModulePropPair h s x y)

/-- Executable check surface for scalar module linearity pair. -/
def scalarModuleCheckPair (h : ScalarModuleHom) (s : F) (x y : Array F) : Bool :=
  decide (scalarModulePropPair h s x y)

/-! Compatibility check wrappers used by protocol-level glue code. -/

/-- Executable additivity check for vector module homs. -/
def preservesAddVec (h : VecModuleHom) (x y : Array F) : Bool :=
  decide (x.size = y.size ∧ h.map (vecAdd x y) = vecAdd (h.map x) (h.map y))

/-- Executable scale-linearity check for vector module homs. -/
def preservesScaleVec (h : VecModuleHom) (s : F) (x : Array F) : Bool :=
  decide (h.map (vecScale s x) = vecScale s (h.map x))

/-- Executable additivity check for scalar module homs. -/
def preservesAddScalar (h : ScalarModuleHom) (x y : Array F) : Bool :=
  decide (x.size = y.size ∧ h.map (vecAdd x y) = h.map x + h.map y)

/-- Executable scale-linearity check for scalar module homs. -/
def preservesScaleScalar (h : ScalarModuleHom) (s : F) (x : Array F) : Bool :=
  decide (h.map (vecScale s x) = s * h.map x)

theorem preservesAddVec_sound
  {h : VecModuleHom} {x y : Array F}
  (hOk : preservesAddVec h x y = true) :
  h.map (vecAdd x y) = vecAdd (h.map x) (h.map y) := by
  exact (decide_eq_true_eq.mp hOk).2

theorem preservesScaleVec_sound
  {h : VecModuleHom} {s : F} {x : Array F}
  (hOk : preservesScaleVec h s x = true) :
  h.map (vecScale s x) = vecScale s (h.map x) := by
  unfold preservesScaleVec at hOk
  exact decide_eq_true_eq.mp hOk

theorem preservesAddScalar_sound
  {h : ScalarModuleHom} {x y : Array F}
  (hOk : preservesAddScalar h x y = true) :
  h.map (vecAdd x y) = h.map x + h.map y := by
  exact (decide_eq_true_eq.mp hOk).2

theorem preservesScaleScalar_sound
  {h : ScalarModuleHom} {s : F} {x : Array F}
  (hOk : preservesScaleScalar h s x = true) :
  h.map (vecScale s x) = s * h.map x := by
  unfold preservesScaleScalar at hOk
  exact decide_eq_true_eq.mp hOk

theorem preservesAddVec_complete
  {h : VecModuleHom} {x y : Array F}
  (hSize : x.size = y.size)
  (hProp : h.map (vecAdd x y) = vecAdd (h.map x) (h.map y)) :
  preservesAddVec h x y = true := by
  exact decide_eq_true ⟨hSize, hProp⟩

theorem preservesScaleVec_complete
  {h : VecModuleHom} {s : F} {x : Array F}
  (hProp : h.map (vecScale s x) = vecScale s (h.map x)) :
  preservesScaleVec h s x = true := by
  unfold preservesScaleVec
  exact decide_eq_true hProp

theorem preservesAddScalar_complete
  {h : ScalarModuleHom} {x y : Array F}
  (hSize : x.size = y.size)
  (hProp : h.map (vecAdd x y) = h.map x + h.map y) :
  preservesAddScalar h x y = true := by
  exact decide_eq_true ⟨hSize, hProp⟩

theorem preservesScaleScalar_complete
  {h : ScalarModuleHom} {s : F} {x : Array F}
  (hProp : h.map (vecScale s x) = s * h.map x) :
  preservesScaleScalar h s x = true := by
  unfold preservesScaleScalar
  exact decide_eq_true hProp

theorem vecModulePropPair_of_checkPair
  {h : VecModuleHom} {s : F} {x y : Array F}
  (hOk : vecModuleCheckPair h s x y = true) :
  vecModulePropPair h s x y := by
  exact decide_eq_true_eq.mp hOk

/-- Turn a vector-linearity proposition pair into a passing check pair. -/
theorem vecModuleCheckPair_of_propPair
  {h : VecModuleHom} {s : F} {x y : Array F}
  (hProp : vecModulePropPair h s x y) :
  vecModuleCheckPair h s x y = true := by
  exact decide_eq_true hProp

/-- One-point vector-linearity check is equivalent to the proposition pair. -/
theorem vecModuleCheckPair_iff_propPair
  {h : VecModuleHom} {s : F} {x y : Array F} :
  vecModuleCheckPair h s x y = true ↔ vecModulePropPair h s x y := by
  constructor
  · exact vecModulePropPair_of_checkPair
  · exact vecModuleCheckPair_of_propPair

/-- Recover a scalar-linearity proposition pair from a passing check pair. -/
theorem scalarModulePropPair_of_checkPair
  {h : ScalarModuleHom} {s : F} {x y : Array F}
  (hOk : scalarModuleCheckPair h s x y = true) :
  scalarModulePropPair h s x y := by
  exact decide_eq_true_eq.mp hOk

/-- Turn a scalar-linearity proposition pair into a passing check pair. -/
theorem scalarModuleCheckPair_of_propPair
  {h : ScalarModuleHom} {s : F} {x y : Array F}
  (hProp : scalarModulePropPair h s x y) :
  scalarModuleCheckPair h s x y = true := by
  exact decide_eq_true hProp

/-- One-point scalar-linearity check is equivalent to the proposition pair. -/
theorem scalarModuleCheckPair_iff_propPair
  {h : ScalarModuleHom} {s : F} {x y : Array F} :
  scalarModuleCheckPair h s x y = true ↔ scalarModulePropPair h s x y := by
  constructor
  · exact scalarModulePropPair_of_checkPair
  · exact scalarModuleCheckPair_of_propPair

/-- Theorem-facing universal linearity contract for vector module homs. -/
def vecModuleAssumption (h : VecModuleHom) : Prop :=
  (∀ x y : Array F, h.map (vecAdd x y) = vecAdd (h.map x) (h.map y)) ∧
  (∀ s : F, ∀ x : Array F, h.map (vecScale s x) = vecScale s (h.map x))

/-- Theorem-facing universal linearity contract for scalar module homs. -/
def scalarModuleAssumption (h : ScalarModuleHom) : Prop :=
  (∀ x y : Array F, h.map (vecAdd x y) = h.map x + h.map y) ∧
  (∀ s : F, ∀ x : Array F, h.map (vecScale s x) = s * h.map x)

/-- Check-facing universal linearity contract for vector module homs. -/
def vecModuleCheckAssumption (h : VecModuleHom) : Prop :=
  ∀ s : F, ∀ x y : Array F, vecModuleCheckPair h s x y = true

/-- Check-facing universal linearity contract for scalar module homs. -/
def scalarModuleCheckAssumption (h : ScalarModuleHom) : Prop :=
  ∀ s : F, ∀ x y : Array F, scalarModuleCheckPair h s x y = true

theorem vecModuleAssumption_of_checkAssumption
  {h : VecModuleHom}
  (hCheck : vecModuleCheckAssumption h) :
  vecModuleAssumption h := by
  constructor
  · intro x y
    exact (vecModulePropPair_of_checkPair (hCheck 0 x y)).1
  · intro s x
    exact (vecModulePropPair_of_checkPair (hCheck s x x)).2

/-- Convert theorem-facing vector assumptions into check-facing form. -/
theorem vecModuleCheckAssumption_of_assumption
  {h : VecModuleHom}
  (hAssm : vecModuleAssumption h) :
  vecModuleCheckAssumption h := by
  intro s x y
  exact vecModuleCheckPair_of_propPair ⟨hAssm.1 x y, hAssm.2 s x⟩

/-- Universal vector theorem/check contracts are equivalent. -/
theorem vecModuleAssumption_iff_checkAssumption
  {h : VecModuleHom} :
  vecModuleAssumption h ↔ vecModuleCheckAssumption h := by
  constructor
  · exact vecModuleCheckAssumption_of_assumption
  · exact vecModuleAssumption_of_checkAssumption

/-- Convert check-facing scalar assumptions into theorem-facing form. -/
theorem scalarModuleAssumption_of_checkAssumption
  {h : ScalarModuleHom}
  (hCheck : scalarModuleCheckAssumption h) :
  scalarModuleAssumption h := by
  constructor
  · intro x y
    exact (scalarModulePropPair_of_checkPair (hCheck 0 x y)).1
  · intro s x
    exact (scalarModulePropPair_of_checkPair (hCheck s x x)).2

/-- Convert theorem-facing scalar assumptions into check-facing form. -/
theorem scalarModuleCheckAssumption_of_assumption
  {h : ScalarModuleHom}
  (hAssm : scalarModuleAssumption h) :
  scalarModuleCheckAssumption h := by
  intro s x y
  exact scalarModuleCheckPair_of_propPair ⟨hAssm.1 x y, hAssm.2 s x⟩

/-- Universal scalar theorem/check contracts are equivalent. -/
theorem scalarModuleAssumption_iff_checkAssumption
  {h : ScalarModuleHom} :
  scalarModuleAssumption h ↔ scalarModuleCheckAssumption h := by
  constructor
  · exact scalarModuleCheckAssumption_of_assumption
  · exact scalarModuleAssumption_of_checkAssumption


end SuperNeo
