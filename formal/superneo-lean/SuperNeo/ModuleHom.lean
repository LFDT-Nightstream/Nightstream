import SuperNeo.EvalHom

/-! Module-hom interfaces and theorem/check bridges (P15). -/


namespace SuperNeo

open F

private def dotF (a b : Array F) : F :=
  if a.size != b.size then
    0
  else
    Id.run do
      let mut acc : F := 0
      for i in [0:a.size] do
        acc := acc + a[i]! * b[i]!
      return acc

/-- Lightweight computational interface for R-module homomorphisms on vectors. -/
structure VecModuleHom where
  map : Array F -> Array F

/-- Lightweight computational interface for scalar-valued module homomorphisms. -/
structure ScalarModuleHom where
  map : Array F -> F

def preservesAddVec (h : VecModuleHom) (x y : Array F) : Bool :=
  if x.size != y.size then
    false
  else
    decide (h.map (vecAdd x y) = vecAdd (h.map x) (h.map y))

def preservesScaleVec (h : VecModuleHom) (s : F) (x : Array F) : Bool :=
  decide (h.map (vecScale s x) = vecScale s (h.map x))

def preservesAddScalar (h : ScalarModuleHom) (x y : Array F) : Bool :=
  if x.size != y.size then
    false
  else
    decide (h.map (vecAdd x y) = h.map x + h.map y)

def preservesScaleScalar (h : ScalarModuleHom) (s : F) (x : Array F) : Bool :=
  decide (h.map (vecScale s x) = s * h.map x)

def idHom : VecModuleHom := { map := fun x => x }

def scaleHom (s : F) : VecModuleHom := { map := fun x => vecScale s x }

def dotHom (w : Array F) : ScalarModuleHom := { map := fun x => dotF x w }

def moduleHomSanity : Bool :=
  let x := #[1, 2, 3, 4]
  let y := #[5, 6, 7, 8]
  let s : F := 9
  let h1 := idHom
  let h2 := scaleHom 3
  let h3 := dotHom #[2, 1, 0, 4]
  preservesAddVec h1 x y &&
    preservesScaleVec h1 s x &&
    preservesAddVec h2 x y &&
    preservesScaleVec h2 s x &&
    preservesAddScalar h3 x y &&
    preservesScaleScalar h3 s x

theorem preservesAddVec_sound
  {h : VecModuleHom} {x y : Array F}
  (hOk : preservesAddVec h x y = true) :
  h.map (vecAdd x y) = vecAdd (h.map x) (h.map y) := by
  unfold preservesAddVec at hOk
  by_cases hsz : x.size != y.size
  · simp [hsz] at hOk
  · simp [hsz] at hOk
    exact hOk

theorem preservesAddVec_complete
  {h : VecModuleHom} {x y : Array F}
  (hSize : x.size = y.size)
  (hProp : h.map (vecAdd x y) = vecAdd (h.map x) (h.map y)) :
  preservesAddVec h x y = true := by
  unfold preservesAddVec
  simp [hSize, decide_eq_true hProp]

theorem preservesScaleVec_sound
  {h : VecModuleHom} {s : F} {x : Array F}
  (hOk : preservesScaleVec h s x = true) :
  h.map (vecScale s x) = vecScale s (h.map x) := by
  unfold preservesScaleVec at hOk
  exact decide_eq_true_eq.mp hOk

theorem preservesScaleVec_complete
  {h : VecModuleHom} {s : F} {x : Array F}
  (hProp : h.map (vecScale s x) = vecScale s (h.map x)) :
  preservesScaleVec h s x = true := by
  unfold preservesScaleVec
  exact decide_eq_true hProp

theorem preservesAddScalar_sound
  {h : ScalarModuleHom} {x y : Array F}
  (hOk : preservesAddScalar h x y = true) :
  h.map (vecAdd x y) = h.map x + h.map y := by
  unfold preservesAddScalar at hOk
  by_cases hsz : x.size != y.size
  · simp [hsz] at hOk
  · simp [hsz] at hOk
    exact hOk

theorem preservesAddScalar_complete
  {h : ScalarModuleHom} {x y : Array F}
  (hSize : x.size = y.size)
  (hProp : h.map (vecAdd x y) = h.map x + h.map y) :
  preservesAddScalar h x y = true := by
  unfold preservesAddScalar
  simp [hSize, decide_eq_true hProp]

theorem preservesScaleScalar_sound
  {h : ScalarModuleHom} {s : F} {x : Array F}
  (hOk : preservesScaleScalar h s x = true) :
  h.map (vecScale s x) = s * h.map x := by
  unfold preservesScaleScalar at hOk
  exact decide_eq_true_eq.mp hOk

theorem preservesScaleScalar_complete
  {h : ScalarModuleHom} {s : F} {x : Array F}
  (hProp : h.map (vecScale s x) = s * h.map x) :
  preservesScaleScalar h s x = true := by
  unfold preservesScaleScalar
  exact decide_eq_true hProp

def vecModuleCheckPair (h : VecModuleHom) (s : F) (x y : Array F) : Prop :=
  preservesAddVec h x y = true ∧ preservesScaleVec h s x = true

def vecModulePropPair (h : VecModuleHom) (s : F) (x y : Array F) : Prop :=
  h.map (vecAdd x y) = vecAdd (h.map x) (h.map y) ∧
    h.map (vecScale s x) = vecScale s (h.map x)

def scalarModuleCheckPair (h : ScalarModuleHom) (s : F) (x y : Array F) : Prop :=
  preservesAddScalar h x y = true ∧ preservesScaleScalar h s x = true

def scalarModulePropPair (h : ScalarModuleHom) (s : F) (x y : Array F) : Prop :=
  h.map (vecAdd x y) = h.map x + h.map y ∧
    h.map (vecScale s x) = s * h.map x

theorem vecModulePropPair_of_checkPair
  {h : VecModuleHom} {s : F} {x y : Array F}
  (hCheck : vecModuleCheckPair h s x y) :
  vecModulePropPair h s x y := by
  exact ⟨preservesAddVec_sound hCheck.1, preservesScaleVec_sound hCheck.2⟩

theorem vecModuleCheckPair_of_propPair
  {h : VecModuleHom} {s : F} {x y : Array F}
  (hSize : x.size = y.size)
  (hProp : vecModulePropPair h s x y) :
  vecModuleCheckPair h s x y := by
  exact ⟨
    preservesAddVec_complete hSize hProp.1,
    preservesScaleVec_complete hProp.2
  ⟩

theorem vecModuleCheckPair_iff_propPair
  {h : VecModuleHom} {s : F} {x y : Array F}
  (hSize : x.size = y.size) :
  vecModuleCheckPair h s x y ↔ vecModulePropPair h s x y := by
  constructor
  · exact vecModulePropPair_of_checkPair
  · exact vecModuleCheckPair_of_propPair hSize

theorem scalarModulePropPair_of_checkPair
  {h : ScalarModuleHom} {s : F} {x y : Array F}
  (hCheck : scalarModuleCheckPair h s x y) :
  scalarModulePropPair h s x y := by
  exact ⟨preservesAddScalar_sound hCheck.1, preservesScaleScalar_sound hCheck.2⟩

theorem scalarModuleCheckPair_of_propPair
  {h : ScalarModuleHom} {s : F} {x y : Array F}
  (hSize : x.size = y.size)
  (hProp : scalarModulePropPair h s x y) :
  scalarModuleCheckPair h s x y := by
  exact ⟨
    preservesAddScalar_complete hSize hProp.1,
    preservesScaleScalar_complete hProp.2
  ⟩

theorem scalarModuleCheckPair_iff_propPair
  {h : ScalarModuleHom} {s : F} {x y : Array F}
  (hSize : x.size = y.size) :
  scalarModuleCheckPair h s x y ↔ scalarModulePropPair h s x y := by
  constructor
  · exact scalarModulePropPair_of_checkPair
  · exact scalarModuleCheckPair_of_propPair hSize

/-
P15 theorem-native assumption boundaries + bridges.

Goal: expose a theorem-native linearity interface that downstream theorems can
consume directly, without routing through boolean check wrappers.
-/

/-- Theorem-native assumption boundary for vector-valued module homomorphisms. -/
def p15VecModuleAssumption (h : VecModuleHom) : Prop :=
  (∀ x y : Array F, x.size = y.size →
      h.map (vecAdd x y) = vecAdd (h.map x) (h.map y)) ∧
  (∀ s : F, ∀ x : Array F,
      h.map (vecScale s x) = vecScale s (h.map x))

/-- Theorem-native assumption boundary for scalar-valued module homomorphisms. -/
def p15ScalarModuleAssumption (h : ScalarModuleHom) : Prop :=
  (∀ x y : Array F, x.size = y.size →
      h.map (vecAdd x y) = h.map x + h.map y) ∧
  (∀ s : F, ∀ x : Array F,
      h.map (vecScale s x) = s * h.map x)

theorem p15VecModuleAssumption_add
  {h : VecModuleHom}
  (hAssm : p15VecModuleAssumption h)
  {x y : Array F}
  (hSize : x.size = y.size) :
  h.map (vecAdd x y) = vecAdd (h.map x) (h.map y) := by
  exact hAssm.1 x y hSize

theorem p15VecModuleAssumption_scale
  {h : VecModuleHom}
  (hAssm : p15VecModuleAssumption h)
  (s : F) (x : Array F) :
  h.map (vecScale s x) = vecScale s (h.map x) := by
  exact hAssm.2 s x

theorem p15ScalarModuleAssumption_add
  {h : ScalarModuleHom}
  (hAssm : p15ScalarModuleAssumption h)
  {x y : Array F}
  (hSize : x.size = y.size) :
  h.map (vecAdd x y) = h.map x + h.map y := by
  exact hAssm.1 x y hSize

theorem p15ScalarModuleAssumption_scale
  {h : ScalarModuleHom}
  (hAssm : p15ScalarModuleAssumption h)
  (s : F) (x : Array F) :
  h.map (vecScale s x) = s * h.map x := by
  exact hAssm.2 s x

/--
Check-oriented universal assumption for `VecModuleHom`, still supported as a
regression bridge.

This keeps the existing executable checks usable, while enabling a theorem-first
path via `p15VecModuleAssumption`.
-/
def p15VecModuleCheckAssumption (h : VecModuleHom) : Prop :=
  ∀ s : F, ∀ x y : Array F, x.size = y.size → vecModuleCheckPair h s x y

/-- Check-oriented universal assumption for `ScalarModuleHom`. -/
def p15ScalarModuleCheckAssumption (h : ScalarModuleHom) : Prop :=
  ∀ s : F, ∀ x y : Array F, x.size = y.size → scalarModuleCheckPair h s x y

theorem p15VecModuleAssumption_of_checkAssumption
  {h : VecModuleHom}
  (hCheck : p15VecModuleCheckAssumption h) :
  p15VecModuleAssumption h := by
  refine ⟨?_, ?_⟩
  · intro x y hSize
    have hPair : vecModuleCheckPair h (0 : F) x y :=
      hCheck (0 : F) x y hSize
    have hProp : vecModulePropPair h (0 : F) x y :=
      vecModulePropPair_of_checkPair hPair
    exact hProp.1
  · intro s x
    have hPair : vecModuleCheckPair h s x x :=
      hCheck s x x rfl
    have hProp : vecModulePropPair h s x x :=
      vecModulePropPair_of_checkPair hPair
    exact hProp.2

theorem p15VecModuleCheckAssumption_of_assumption
  {h : VecModuleHom}
  (hAssm : p15VecModuleAssumption h) :
  p15VecModuleCheckAssumption h := by
  intro s x y hSize
  have hProp : vecModulePropPair h s x y := by
    refine ⟨hAssm.1 x y hSize, hAssm.2 s x⟩
  exact vecModuleCheckPair_of_propPair hSize hProp

theorem p15ScalarModuleAssumption_of_checkAssumption
  {h : ScalarModuleHom}
  (hCheck : p15ScalarModuleCheckAssumption h) :
  p15ScalarModuleAssumption h := by
  refine ⟨?_, ?_⟩
  · intro x y hSize
    have hPair : scalarModuleCheckPair h (0 : F) x y :=
      hCheck (0 : F) x y hSize
    have hProp : scalarModulePropPair h (0 : F) x y :=
      scalarModulePropPair_of_checkPair hPair
    exact hProp.1
  · intro s x
    have hPair : scalarModuleCheckPair h s x x :=
      hCheck s x x rfl
    have hProp : scalarModulePropPair h s x x :=
      scalarModulePropPair_of_checkPair hPair
    exact hProp.2

theorem p15ScalarModuleCheckAssumption_of_assumption
  {h : ScalarModuleHom}
  (hAssm : p15ScalarModuleAssumption h) :
  p15ScalarModuleCheckAssumption h := by
  intro s x y hSize
  have hProp : scalarModulePropPair h s x y := by
    refine ⟨hAssm.1 x y hSize, hAssm.2 s x⟩
  exact scalarModuleCheckPair_of_propPair hSize hProp

theorem p15VecModuleAssumption_map_linComb2Vec
  {h : VecModuleHom} {ρ1 ρ2 : F} {z1 z2 : Array F}
  (hAssm : p15VecModuleAssumption h)
  (hSize : z1.size = z2.size) :
  h.map (linComb2Vec ρ1 ρ2 z1 z2) = linComb2Vec ρ1 ρ2 (h.map z1) (h.map z2) := by
  unfold linComb2Vec
  have hSizeScaled : (vecScale ρ1 z1).size = (vecScale ρ2 z2).size := by
    simp [vecScale_size, hSize]
  have hAdd :=
    hAssm.1 (vecScale ρ1 z1) (vecScale ρ2 z2) hSizeScaled
  have hS1 : h.map (vecScale ρ1 z1) = vecScale ρ1 (h.map z1) := hAssm.2 ρ1 z1
  have hS2 : h.map (vecScale ρ2 z2) = vecScale ρ2 (h.map z2) := hAssm.2 ρ2 z2
  simpa [hS1, hS2] using hAdd

/-- Package `evalBarMzAt` as a `VecModuleHom` in the `z` argument. -/
def evalBarMzAtVecHom
  (bar : Array (Array F)) (m : Array (Array F)) (r : Array F) : VecModuleHom :=
  { map := fun z => evalBarMzAt bar m z r }

/-- P15-style linearity assumption specialized to `evalBarMzAt` as a module map in `z`. -/
def p15EvalBarMzAtAssumption
  (bar : Array (Array F)) (m : Array (Array F)) (r : Array F) : Prop :=
  p15VecModuleAssumption (evalBarMzAtVecHom bar m r)

theorem evalHom2Prop_of_p15EvalBarMzAtAssumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z1 z2 r : Array F}
  {ρ1 ρ2 : F}
  (hLin : p15EvalBarMzAtAssumption bar m r)
  (hSize : z1.size = z2.size)
  (hRows : MatrixRowsCompatible m z1) :
  evalHom2Prop bar m z1 z2 r ρ1 ρ2 := by
  unfold p15EvalBarMzAtAssumption at hLin
  unfold evalHom2Prop
  have hSizeBool : (z1.size != z2.size) = false := by
    simp [hSize]
  refine ⟨hSizeBool, hRows, ?_⟩
  dsimp [evalBarMzAtVecHom]
  -- y-linearity from the VecModuleHom assumption (no check wrapper)
  have hMap :
      evalBarMzAt bar m (linComb2Vec ρ1 ρ2 z1 z2) r
        = linComb2Vec ρ1 ρ2 (evalBarMzAt bar m z1 r) (evalBarMzAt bar m z2 r) := by
    simpa [evalBarMzAtVecHom] using
      (p15VecModuleAssumption_map_linComb2Vec
        (h := evalBarMzAtVecHom bar m r)
        (ρ1 := ρ1) (ρ2 := ρ2) (z1 := z1) (z2 := z2)
        hLin hSize)
  have hYEq :
      vecAdd (vecScale ρ1 (evalBarMzAt bar m z1 r))
        (vecScale ρ2 (evalBarMzAt bar m z2 r))
        = evalBarMzAt bar m (linComb2Vec ρ1 ρ2 z1 z2) r := by
    -- `linComb2Vec` is definitional to `vecAdd (vecScale ..) (vecScale ..)`
    simpa [linComb2Vec] using hMap.symm
  -- constant-term linearity is theorem-native (`ct_vecAdd_of_size_eq`, `ct_vecScale`)
  have hSz1 : (evalBarMzAt bar m z1 r).size = d :=
    evalBarMzAt_size bar m z1 r
  have hSz2 : (evalBarMzAt bar m z2 r).size = d :=
    evalBarMzAt_size bar m z2 r
  have hSizeY :
      (vecScale ρ1 (evalBarMzAt bar m z1 r)).size
        = (vecScale ρ2 (evalBarMzAt bar m z2 r)).size := by
    simp [vecScale_size, hSz1, hSz2]
  have hCtEq :
      ct (vecAdd (vecScale ρ1 (evalBarMzAt bar m z1 r))
            (vecScale ρ2 (evalBarMzAt bar m z2 r)))
        = ρ1 * ct (evalBarMzAt bar m z1 r) + ρ2 * ct (evalBarMzAt bar m z2 r) := by
    calc
      ct (vecAdd (vecScale ρ1 (evalBarMzAt bar m z1 r))
            (vecScale ρ2 (evalBarMzAt bar m z2 r)))
          = ct (vecScale ρ1 (evalBarMzAt bar m z1 r))
              + ct (vecScale ρ2 (evalBarMzAt bar m z2 r)) := by
              simpa using
                (ct_vecAdd_of_size_eq
                  (a := vecScale ρ1 (evalBarMzAt bar m z1 r))
                  (b := vecScale ρ2 (evalBarMzAt bar m z2 r))
                  hSizeY)
      _ = ρ1 * ct (evalBarMzAt bar m z1 r) + ρ2 * ct (evalBarMzAt bar m z2 r) := by
              simp [ct_vecScale]
  exact ⟨hYEq, hCtEq⟩

/--
Theorem-native bridge: P15 linearity of `evalBarMzAt` (as a module map in `z`)
implies the P14/P5 evaluation-hom proposition (`evalHom2Prop`), without routing
through `evalHom2` boolean checks.
-/
theorem p14EvalHomAssumption_of_p15EvalBarMzAtAssumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {r : Array F}
  {ρ1 ρ2 : F}
  (hLin : p15EvalBarMzAtAssumption bar m r) :
  p14EvalHomAssumption bar m r ρ1 ρ2 := by
  intro z1 z2 hSize hRows
  exact evalHom2Prop_of_p15EvalBarMzAtAssumption
    (bar := bar) (m := m) (z1 := z1) (z2 := z2) (r := r) (ρ1 := ρ1) (ρ2 := ρ2)
    hLin hSize hRows

theorem p14EvalHomCheckAssumption_of_p15EvalBarMzAtAssumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {r : Array F}
  {ρ1 ρ2 : F}
  (hLin : p15EvalBarMzAtAssumption bar m r) :
  p14EvalHomCheckAssumption bar m r ρ1 ρ2 := by
  intro z1 z2 hSize hRows
  exact evalHom2_complete
    (evalHom2Prop_of_p15EvalBarMzAtAssumption
      (bar := bar) (m := m) (z1 := z1) (z2 := z2) (r := r) (ρ1 := ρ1) (ρ2 := ρ2)
      hLin hSize hRows)

end SuperNeo
