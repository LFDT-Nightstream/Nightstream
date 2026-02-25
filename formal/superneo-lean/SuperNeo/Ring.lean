import SuperNeo.Field
import SuperNeo.Dimensions

namespace SuperNeo

open F

def D : Nat := 54

theorem D_eq_d : D = d := rfl
theorem D_pos : 0 < D := by decide

abbrev Coeffs := Array F

private def addAt (arr : Array F) (idx : Nat) (delta : F) : Array F :=
  arr.set! idx (arr[idx]! + delta)

private def subAt (arr : Array F) (idx : Nat) (delta : F) : Array F :=
  arr.set! idx (arr[idx]! - delta)

@[simp] theorem addAt_size (arr : Array F) (idx : Nat) (delta : F) :
  (addAt arr idx delta).size = arr.size := by
  unfold addAt
  simp

@[simp] theorem subAt_size (arr : Array F) (idx : Nat) (delta : F) :
  (subAt arr idx delta).size = arr.size := by
  unfold subAt
  simp

theorem setAt_allCanonical
  (arr : Array F) (idx : Nat) (val : F)
  (hVal : F.Canonical val)
  (hArr : arr.all F.Canonical = true) :
  (arr.set! idx val).all F.Canonical = true := by
  apply (Array.all_eq_true).2
  intro j hj
  have hjArr : j < arr.size := by
    simpa [Array.set!_eq_setIfInBounds] using hj
  by_cases hidx : idx = j
  · subst hidx
    have hji : idx < (arr.set! idx val).size := by
      simpa using hj
    have hSet :
        (arr.set! idx val)[idx] = val := by
      simpa [Array.set!_eq_setIfInBounds] using
        (Array.getElem_setIfInBounds_self (xs := arr) (i := idx) (a := val) hji)
    rw [hSet]
    exact decide_eq_true hVal
  · have hSet :
      (arr.set! idx val)[j] = arr[j] := by
      simpa [Array.set!_eq_setIfInBounds] using
        (Array.getElem_setIfInBounds_ne (xs := arr) (i := idx) (a := val) (j := j) hjArr hidx)
    have hArrjDec : decide (F.Canonical arr[j]) = true := (Array.all_eq_true.mp hArr) j hjArr
    have hArrj : F.Canonical arr[j] := decide_eq_true_eq.mp hArrjDec
    rw [hSet]
    exact decide_eq_true hArrj

theorem addAt_allCanonical
  (arr : Array F) (idx : Nat) (delta : F)
  (hArr : arr.all F.Canonical = true) :
  (addAt arr idx delta).all F.Canonical = true := by
  unfold addAt
  exact setAt_allCanonical arr idx (arr[idx]! + delta) (F.canonical_add _ _) hArr

theorem subAt_allCanonical
  (arr : Array F) (idx : Nat) (delta : F)
  (hArr : arr.all F.Canonical = true) :
  (subAt arr idx delta).all F.Canonical = true := by
  unfold subAt
  exact setAt_allCanonical arr idx (arr[idx]! - delta) (F.canonical_sub _ _) hArr

theorem replicate_zero_allCanonical (n : Nat) :
  (Array.replicate n (0 : F)).all F.Canonical = true := by
  apply (Array.all_eq_true).2
  intro i hi
  have hZero : (Array.replicate n (0 : F))[i] = (0 : F) := by
    simp [Array.getElem_replicate]
  rw [hZero]
  exact decide_eq_true F.canonical_zero

/-- Constant-term extraction ct : R_q -> F_q. -/
def ct (a : Coeffs) : F :=
  if a.isEmpty then
    0
  else
    a[0]!

theorem ct_of_isEmpty {a : Coeffs} (h : a.isEmpty = true) : ct a = 0 := by
  simp [ct, h]

theorem ct_of_not_isEmpty {a : Coeffs} (h : a.isEmpty = false) : ct a = a[0]! := by
  simp [ct, h]

theorem ct_canonical_of_all
  {a : Coeffs}
  (hAll : a.all F.Canonical = true) :
  F.Canonical (ct a) := by
  unfold ct
  by_cases hEmpty : a.isEmpty = true
  · simpa [hEmpty] using F.canonical_zero
  · have hGet : F.Canonical (a[0]!) := by
      exact F.canonical_getElem!_of_all (arr := a) (hArr := by simpa using hAll) 0
    simpa [hEmpty] using hGet

private def schoolbookRaw (a b : Coeffs) : Array F :=
  Id.run do
    let mut tmp := Array.replicate (2 * D - 1) (0 : F)
    for i in [0:D] do
      let ai := a[i]!
      for j in [0:D] do
        tmp := addAt tmp (i + j) (ai * b[j]!)
    return tmp

theorem schoolbookRaw_size (a b : Coeffs) :
  (schoolbookRaw a b).size = 2 * D - 1 := by
  unfold schoolbookRaw
  have hInner :
      ∀ (i : Nat) (inner : List Nat) (tmp : Array F),
        (List.foldl (fun acc j => addAt acc (i + j) (a[i]! * b[j]!)) tmp inner).size =
          tmp.size := by
    intro i inner tmp
    induction inner generalizing tmp with
    | nil =>
        simp
    | cons j js ih =>
        simp [List.foldl_cons, ih, addAt_size]
  have hOuter :
      ∀ (outer : List Nat) (tmp : Array F),
        (List.foldl
            (fun acc i =>
              List.foldl (fun acc' j => addAt acc' (i + j) (a[i]! * b[j]!)) acc (List.range' 0 D))
            tmp
            outer).size =
          tmp.size := by
    intro outer tmp
    induction outer generalizing tmp with
    | nil =>
        simp
    | cons i is ih =>
        simp [List.foldl_cons, hInner i (List.range' 0 D), ih]
  simpa using hOuter (List.range' 0 D) (Array.replicate (2 * D - 1) (0 : F))

private def takeFirstD (arr : Array F) : Coeffs :=
  Array.ofFn (fun i : Fin D => arr[i.1]!)

theorem takeFirstD_size (arr : Array F) : (takeFirstD arr).size = D := by
  unfold takeFirstD
  simp

theorem takeFirstD_allCanonical
  (arr : Array F)
  (hAll : arr.all F.Canonical = true) :
  (takeFirstD arr).all F.Canonical = true := by
  apply (Array.all_eq_true).2
  intro i hi
  have hCanon : F.Canonical (arr[i]!) := by
    exact F.canonical_getElem!_of_all (arr := arr) (hArr := by simpa using hAll) i
  simpa [takeFirstD] using decide_eq_true hCanon

theorem takeFirstD_not_isEmpty (arr : Array F) : (takeFirstD arr).isEmpty = false := by
  simp [takeFirstD, D]

theorem ct_takeFirstD (arr : Array F) : ct (takeFirstD arr) = arr[0]! := by
  simp [ct, takeFirstD, D]

theorem ct_takeFirstD_canonical
  (arr : Array F)
  (hAll : arr.all F.Canonical = true) :
  F.Canonical (ct (takeFirstD arr)) := by
  exact ct_canonical_of_all (takeFirstD_allCanonical arr hAll)

/-- Reduce modulo Phi_81(X)=X^54 + X^27 + 1, matching Rust logic. -/
private def reducePhi81 (coeffsIn : Array F) : Coeffs :=
  Id.run do
    let mut coeffs := coeffsIn
    for off in [0:(D - 1)] do
      let i := (2 * D - 2) - off
      let t := coeffs[i]!
      coeffs := coeffs.set! i (0 : F)
      coeffs := subAt coeffs (i - D) t
      let idx27 := i - 27
      if idx27 < D then
        coeffs := subAt coeffs idx27 t
      else
        coeffs := addAt coeffs (idx27 - D) t
        let idx2727 := idx27 - 27
        if idx2727 < D then
          coeffs := addAt coeffs idx2727 t
    return takeFirstD coeffs

theorem reducePhi81_size
  (coeffsIn : Array F) :
  (reducePhi81 coeffsIn).size = D := by
  unfold reducePhi81
  simp [takeFirstD_size]

/-- Ring multiplication in R_q = F_q[X]/(X^54 + X^27 + 1). -/
def mulRq (a b : Coeffs) : Coeffs :=
  reducePhi81 (schoolbookRaw a b)

def hasRingDegreeShape (a : Coeffs) : Prop := a.size = D

def ringMulShapeProp (a b : Coeffs) : Prop :=
  hasRingDegreeShape a ∧ hasRingDegreeShape b ∧ (mulRq a b).size = D

instance hasRingDegreeShape_decidable (a : Coeffs) : Decidable (hasRingDegreeShape a) := by
  unfold hasRingDegreeShape
  infer_instance

instance ringMulShapeProp_decidable (a b : Coeffs) : Decidable (ringMulShapeProp a b) := by
  unfold ringMulShapeProp
  infer_instance

def ringMulShapeCheck (a b : Coeffs) : Bool :=
  decide (ringMulShapeProp a b)

theorem ringMulShapeCheck_sound
  {a b : Coeffs}
  (hOk : ringMulShapeCheck a b = true) :
  ringMulShapeProp a b := by
  unfold ringMulShapeCheck at hOk
  exact decide_eq_true_eq.mp hOk

theorem ringMulShapeCheck_complete
  {a b : Coeffs}
  (hProp : ringMulShapeProp a b) :
  ringMulShapeCheck a b = true := by
  unfold ringMulShapeCheck
  exact decide_eq_true hProp

theorem mulRq_size (a b : Coeffs) : (mulRq a b).size = D := by
  unfold mulRq
  exact reducePhi81_size (schoolbookRaw a b)

theorem hasRingDegreeShape_mulRq (a b : Coeffs) : hasRingDegreeShape (mulRq a b) := by
  unfold hasRingDegreeShape
  exact mulRq_size a b

theorem ringMulShape_of_shapes
  {a b : Coeffs}
  (ha : hasRingDegreeShape a)
  (hb : hasRingDegreeShape b) :
  ringMulShapeProp a b := by
  exact ⟨ha, hb, mulRq_size a b⟩

theorem ringMulShapeCheck_true_of_shapes
  {a b : Coeffs}
  (ha : hasRingDegreeShape a)
  (hb : hasRingDegreeShape b) :
  ringMulShapeCheck a b = true := by
  unfold ringMulShapeCheck
  exact decide_eq_true (ringMulShape_of_shapes ha hb)

/-- Dot product over F_q^d. -/
def dot (a b : Coeffs) : F :=
  Id.run do
    let mut acc : F := 0
    for i in [0:D] do
      acc := acc + a[i]! * b[i]!
    return acc

/-- Apply SuperNeo bar transform for one D-sized block. -/
def superneoBarBlock (bar : Array (Array F)) (v : Coeffs) : Coeffs :=
  Id.run do
    let mut out := Array.replicate D (0 : F)
    for row in [0:D] do
      let mut acc : F := 0
      let barRow := bar[row]!
      for col in [0:D] do
        acc := acc + barRow[col]! * v[col]!
      out := out.set! row acc
    return out

theorem superneoBarBlock_size (bar : Array (Array F)) (v : Coeffs) :
  (superneoBarBlock bar v).size = D := by
  unfold superneoBarBlock
  have hOuter :
      ∀ (rows : List Nat) (out : Array F),
        (List.foldl
            (fun acc row =>
              acc.setIfInBounds row
                (List.foldl (fun acc' col => acc' + bar[row]![col]! * v[col]!) 0 (List.range' 0 D)))
            out
            rows).size =
          out.size := by
    intro rows out
    induction rows generalizing out with
    | nil =>
        simp
    | cons row rows ih =>
        simp [List.foldl_cons, ih]
  simpa using hOuter (List.range' 0 D) (Array.replicate D (0 : F))

theorem hasRingDegreeShape_superneoBarBlock (bar : Array (Array F)) (v : Coeffs) :
  hasRingDegreeShape (superneoBarBlock bar v) := by
  unfold hasRingDegreeShape
  exact superneoBarBlock_size bar v

end SuperNeo
