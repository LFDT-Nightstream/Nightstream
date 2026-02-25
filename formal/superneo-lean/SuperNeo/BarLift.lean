import SuperNeo.Ring
import SuperNeo.Dimensions

namespace SuperNeo

open F

def vecAdd (a b : Array F) : Array F :=
  if _h : a.size = b.size then
    Array.ofFn (fun i : Fin a.size => a[i.1]! + b[i.1]!)
  else
    #[]

def vecScale (s : F) (a : Array F) : Array F :=
  a.map (fun x => s * x)

theorem vecAdd_size_of_eq
  {a b : Array F}
  (hSize : a.size = b.size) :
  (vecAdd a b).size = a.size := by
  unfold vecAdd
  simp [hSize]

theorem vecScale_size
  (s : F) (a : Array F) :
  (vecScale s a).size = a.size := by
  unfold vecScale
  simp

private def chunkBlocks (v : Array F) : Array Coeffs :=
  Array.ofFn (fun t : Fin (v.size / d) => v.extract (t.1 * d) (t.1 * d + d))

private def flattenBlocks (bs : Array Coeffs) : Array F :=
  bs.foldl (fun acc blk => acc ++ blk) #[]

theorem flattenBlocks_singleton (blk : Coeffs) :
  flattenBlocks #[blk] = blk := by
  unfold flattenBlocks
  simp

/-- Definition 8 vector lifting: apply bar transform blockwise on d-coefficient chunks. -/
def barLiftVec (bar : Array (Array F)) (v : Array F) : Array F :=
  if v.size % d != 0 then
    #[]
  else
    let chunks := chunkBlocks v
    flattenBlocks (chunks.map (fun blk => superneoBarBlock bar blk))

/-- Definition 8 matrix lifting: row-wise lifting of vector transform. -/
def barLiftMatrix (bar : Array (Array F)) (m : Array (Array F)) : Array (Array F) :=
  m.map (barLiftVec bar)

def barLiftAddLinear (bar : Array (Array F)) (v w : Array F) : Bool :=
  if v.size = w.size then
    if v.size % d = 0 then
      let lhs := barLiftVec bar (vecAdd v w)
      let rhs := vecAdd (barLiftVec bar v) (barLiftVec bar w)
      decide (lhs = rhs)
    else
      false
  else
    false

def barLiftScaleLinear (bar : Array (Array F)) (s : F) (v : Array F) : Bool :=
  if v.size % d = 0 then
    let lhs := barLiftVec bar (vecScale s v)
    let rhs := vecScale s (barLiftVec bar v)
    decide (lhs = rhs)
  else
    false

theorem chunkBlocks_size (v : Array F) :
  (chunkBlocks v).size = v.size / d := by
  unfold chunkBlocks
  simp

theorem barLiftVec_singleBlock
  {bar : Array (Array F)} {v : Array F}
  (hSize : v.size = d) :
  barLiftVec bar v = superneoBarBlock bar v := by
  have hMod : v.size % d = 0 := by
    simpa [hSize]
  have hChunk : chunkBlocks v = #[v.extract 0 d] := by
    unfold chunkBlocks
    have hDiv : v.size / d = 1 := by
      have hdpos : 0 < d := by
        unfold d
        decide
      calc
        v.size / d = d / d := by simpa [hSize]
        _ = 1 := by simpa using Nat.div_self hdpos
    rw [hDiv]
    ext i <;> simp
  have hExtract : v.extract 0 d = v := by
    simpa [hSize]
  unfold barLiftVec
  simp [hMod, hChunk, hExtract, flattenBlocks_singleton]

def p11BarLiftAddProp (bar : Array (Array F)) (v w : Array F) : Prop :=
  v.size = w.size ∧
    v.size % d = 0 ∧
    barLiftVec bar (vecAdd v w) = vecAdd (barLiftVec bar v) (barLiftVec bar w)

def p11BarLiftScaleProp (bar : Array (Array F)) (s : F) (v : Array F) : Prop :=
  v.size % d = 0 ∧
    barLiftVec bar (vecScale s v) = vecScale s (barLiftVec bar v)

def p11BarLiftLinearProp (bar : Array (Array F)) (s : F) (v w : Array F) : Prop :=
  p11BarLiftAddProp bar v w ∧ p11BarLiftScaleProp bar s v

def p11BarLiftMatrixProp (bar : Array (Array F)) (m : Array (Array F)) : Prop :=
  barLiftMatrix bar m = m.map (barLiftVec bar)

/--
Theorem-native assumption boundary for Definition 8 additivity:
for shape-valid inputs, lifting commutes with vector addition.
-/
def p11AdditivityAssumption (bar : Array (Array F)) : Prop :=
  ∀ v w : Array F, v.size = w.size -> v.size % d = 0 ->
    barLiftVec bar (vecAdd v w) = vecAdd (barLiftVec bar v) (barLiftVec bar w)

/--
Theorem-native assumption boundary for Definition 8 homogeneity:
for shape-valid inputs, lifting commutes with scalar multiplication.
-/
def p11HomogeneityAssumption (bar : Array (Array F)) : Prop :=
  ∀ s : F, ∀ v : Array F, v.size % d = 0 ->
    barLiftVec bar (vecScale s v) = vecScale s (barLiftVec bar v)

/--
Stronger (check-oriented) universal assumption forms used as a regression bridge.
These are intentionally stronger than the theorem-native assumptions above.
-/
def p11AdditivityCheckAssumption (bar : Array (Array F)) : Prop :=
  ∀ v w : Array F, barLiftAddLinear bar v w = true

def p11HomogeneityCheckAssumption (bar : Array (Array F)) : Prop :=
  ∀ s : F, ∀ v : Array F, barLiftScaleLinear bar s v = true

/--
Block-level theorem assumptions for size-`d` vectors. These isolate the algebraic
core needed to derive Definition 8 at lift level.
-/
def p11BlockAdditivityAssumption (bar : Array (Array F)) : Prop :=
  ∀ v w : Array F, v.size = d -> w.size = d ->
    superneoBarBlock bar (vecAdd v w) = vecAdd (superneoBarBlock bar v) (superneoBarBlock bar w)

def p11BlockHomogeneityAssumption (bar : Array (Array F)) : Prop :=
  ∀ s : F, ∀ v : Array F, v.size = d ->
    superneoBarBlock bar (vecScale s v) = vecScale s (superneoBarBlock bar v)

theorem barLiftAddLinear_sound
  {bar : Array (Array F)} {v w : Array F}
  (hOk : barLiftAddLinear bar v w = true) :
  p11BarLiftAddProp bar v w := by
  unfold barLiftAddLinear at hOk
  by_cases hSize : v.size = w.size
  · by_cases hMod : v.size % d = 0
    · have hRes :
        w.size % d = 0 ∧ barLiftVec bar (vecAdd v w) = vecAdd (barLiftVec bar v) (barLiftVec bar w) := by
          simpa [hSize, hMod] using hOk
      exact ⟨hSize, hMod, hRes.2⟩
    · have hRes :
        w.size % d = 0 ∧ barLiftVec bar (vecAdd v w) = vecAdd (barLiftVec bar v) (barLiftVec bar w) := by
          simpa [hSize, hMod] using hOk
      have hModTrue : v.size % d = 0 := by
        simpa [hSize] using hRes.1
      exact False.elim (hMod hModTrue)
  · simp [hSize] at hOk

theorem barLiftAddLinear_complete
  {bar : Array (Array F)} {v w : Array F}
  (hProp : p11BarLiftAddProp bar v w) :
  barLiftAddLinear bar v w = true := by
  rcases hProp with ⟨hSize, hMod, hEq⟩
  have hModW : w.size % d = 0 := by
    simpa [hSize] using hMod
  unfold barLiftAddLinear
  simp [hSize, hModW, decide_eq_true hEq]

theorem barLiftScaleLinear_sound
  {bar : Array (Array F)} {s : F} {v : Array F}
  (hOk : barLiftScaleLinear bar s v = true) :
  p11BarLiftScaleProp bar s v := by
  unfold barLiftScaleLinear at hOk
  by_cases hMod : v.size % d = 0
  · have hRes :
      v.size % d = 0 ∧ barLiftVec bar (vecScale s v) = vecScale s (barLiftVec bar v) := by
        simpa [hMod] using hOk
    exact ⟨hRes.1, hRes.2⟩
  · have hRes :
      v.size % d = 0 ∧ barLiftVec bar (vecScale s v) = vecScale s (barLiftVec bar v) := by
        simp at hOk
        exact hOk
    exact False.elim (hMod hRes.1)

theorem barLiftScaleLinear_complete
  {bar : Array (Array F)} {s : F} {v : Array F}
  (hProp : p11BarLiftScaleProp bar s v) :
  barLiftScaleLinear bar s v = true := by
  rcases hProp with ⟨hMod, hEq⟩
  unfold barLiftScaleLinear
  simp [hMod, decide_eq_true hEq]

theorem p11BarLiftLinear_of_checks
  {bar : Array (Array F)} {s : F} {v w : Array F}
  (hAdd : barLiftAddLinear bar v w = true)
  (hScale : barLiftScaleLinear bar s v = true) :
  p11BarLiftLinearProp bar s v w := by
  exact ⟨barLiftAddLinear_sound hAdd, barLiftScaleLinear_sound hScale⟩

theorem p11BarLiftAddProp_of_assumption
  {bar : Array (Array F)} {v w : Array F}
  (hAdd : p11AdditivityAssumption bar)
  (hSize : v.size = w.size)
  (hMod : v.size % d = 0) :
  p11BarLiftAddProp bar v w := by
  exact ⟨hSize, hMod, hAdd v w hSize hMod⟩

theorem p11BarLiftScaleProp_of_assumption
  {bar : Array (Array F)} {s : F} {v : Array F}
  (hScale : p11HomogeneityAssumption bar)
  (hMod : v.size % d = 0) :
  p11BarLiftScaleProp bar s v := by
  exact ⟨hMod, hScale s v hMod⟩

theorem p11BarLiftLinearProp_of_assumptions
  {bar : Array (Array F)} {s : F} {v w : Array F}
  (hAdd : p11AdditivityAssumption bar)
  (hScale : p11HomogeneityAssumption bar)
  (hSize : v.size = w.size)
  (hMod : v.size % d = 0) :
  p11BarLiftLinearProp bar s v w := by
  exact ⟨
    p11BarLiftAddProp_of_assumption hAdd hSize hMod,
    p11BarLiftScaleProp_of_assumption hScale hMod
  ⟩

theorem p11BarLiftChecks_of_assumptions
  {bar : Array (Array F)} {s : F} {v w : Array F}
  (hAdd : p11AdditivityAssumption bar)
  (hScale : p11HomogeneityAssumption bar)
  (hSize : v.size = w.size)
  (hMod : v.size % d = 0) :
  barLiftAddLinear bar v w = true ∧ barLiftScaleLinear bar s v = true := by
  exact ⟨
    barLiftAddLinear_complete (p11BarLiftAddProp_of_assumption hAdd hSize hMod),
    barLiftScaleLinear_complete (p11BarLiftScaleProp_of_assumption hScale hMod)
  ⟩

theorem p11AdditivityAssumption_of_checkAssumption
  {bar : Array (Array F)}
  (hCheck : p11AdditivityCheckAssumption bar) :
  p11AdditivityAssumption bar := by
  intro v w _hSize _hMod
  exact (barLiftAddLinear_sound (hCheck v w)).2.2

theorem p11HomogeneityAssumption_of_checkAssumption
  {bar : Array (Array F)}
  (hCheck : p11HomogeneityCheckAssumption bar) :
  p11HomogeneityAssumption bar := by
  intro s v _hMod
  exact (barLiftScaleLinear_sound (hCheck s v)).2

theorem p11BarLiftLinearProp_of_checkAssumptions
  {bar : Array (Array F)} {s : F} {v w : Array F}
  (hAddCheck : p11AdditivityCheckAssumption bar)
  (hScaleCheck : p11HomogeneityCheckAssumption bar)
  (hSize : v.size = w.size)
  (hMod : v.size % d = 0) :
  p11BarLiftLinearProp bar s v w := by
  exact p11BarLiftLinearProp_of_assumptions
    (p11AdditivityAssumption_of_checkAssumption hAddCheck)
    (p11HomogeneityAssumption_of_checkAssumption hScaleCheck)
    hSize hMod

theorem p11BarLiftAddProp_singleBlock_of_blockAssumption
  {bar : Array (Array F)} {v w : Array F}
  (hBlock : p11BlockAdditivityAssumption bar)
  (hv : v.size = d)
  (hw : w.size = d) :
  p11BarLiftAddProp bar v w := by
  have hSize : v.size = w.size := by
    calc
      v.size = d := hv
      _ = w.size := hw.symm
  have hMod : v.size % d = 0 := by
    simpa [hv]
  have hAddSize : (vecAdd v w).size = d := by
    calc
      (vecAdd v w).size = v.size := vecAdd_size_of_eq hSize
      _ = d := hv
  refine ⟨hSize, hMod, ?_⟩
  calc
    barLiftVec bar (vecAdd v w)
      = superneoBarBlock bar (vecAdd v w) := by
          exact barLiftVec_singleBlock hAddSize
    _ = vecAdd (superneoBarBlock bar v) (superneoBarBlock bar w) := by
          exact hBlock v w hv hw
    _ = vecAdd (barLiftVec bar v) (barLiftVec bar w) := by
          simp [barLiftVec_singleBlock hv, barLiftVec_singleBlock hw]

theorem p11BarLiftScaleProp_singleBlock_of_blockAssumption
  {bar : Array (Array F)} {s : F} {v : Array F}
  (hBlock : p11BlockHomogeneityAssumption bar)
  (hv : v.size = d) :
  p11BarLiftScaleProp bar s v := by
  have hMod : v.size % d = 0 := by
    simpa [hv]
  have hScaleSize : (vecScale s v).size = d := by
    simpa [hv] using vecScale_size s v
  refine ⟨hMod, ?_⟩
  calc
    barLiftVec bar (vecScale s v)
      = superneoBarBlock bar (vecScale s v) := by
          exact barLiftVec_singleBlock hScaleSize
    _ = vecScale s (superneoBarBlock bar v) := by
          exact hBlock s v hv
    _ = vecScale s (barLiftVec bar v) := by
          simp [barLiftVec_singleBlock hv]

theorem p11BarLiftLinearProp_singleBlock_of_blockAssumptions
  {bar : Array (Array F)} {s : F} {v w : Array F}
  (hBlockAdd : p11BlockAdditivityAssumption bar)
  (hBlockScale : p11BlockHomogeneityAssumption bar)
  (hv : v.size = d)
  (hw : w.size = d) :
  p11BarLiftLinearProp bar s v w := by
  exact ⟨
    p11BarLiftAddProp_singleBlock_of_blockAssumption hBlockAdd hv hw,
    p11BarLiftScaleProp_singleBlock_of_blockAssumption hBlockScale hv
  ⟩

theorem p11BarLiftChecks_of_props
  {bar : Array (Array F)} {s : F} {v w : Array F}
  (hProp : p11BarLiftLinearProp bar s v w) :
  barLiftAddLinear bar v w = true ∧ barLiftScaleLinear bar s v = true := by
  exact ⟨barLiftAddLinear_complete hProp.1, barLiftScaleLinear_complete hProp.2⟩

theorem p11BarLiftMatrix_theorem
  (bar : Array (Array F)) (m : Array (Array F)) :
  p11BarLiftMatrixProp bar m := by
  rfl

end SuperNeo
