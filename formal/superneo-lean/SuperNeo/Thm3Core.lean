import SuperNeo.BarLift
import SuperNeo.Generated.Vectors

/-!
Theorem-3 inner-product transform (paper-faithful).

This file defines the Theorem-3 boundary: for d-sized field blocks,
`ct(mulRqPhi(bar(a), b)) = ⟨a, b⟩` — the constant term of the ring product
between the bar-transformed left block and the right block equals the
field inner product.

Paper anchor: Theorem 3 (Inner Product Transform), Section 5, lines 368-372.
-/

namespace SuperNeo

open F
local instance : NeZero Goldilocks.q := ⟨Nat.ne_of_gt Goldilocks.q_pos⟩

/-- Dot/inner product with an explicit size guard. -/
def innerProduct (a b : Array F) : F :=
  if _h : a.size = b.size then
    (List.range a.size).foldl (fun acc i => acc + a[i]! * b[i]!) 0
  else
    0

/--
Theorem-3 boundary (paper-faithful): for d-sized blocks,
`ct(mulRqPhi(bar(a), b)) = ⟨a, b⟩`.

This is a pure boundary assumption. Closure requires a bar transform matrix
that encodes the field inner product via ring multiplication for the
cyclotomic Φ(X) = X^d + X^(d/2) + 1.
-/
def thm3CoreAssumption (bar : Array (Array F)) : Prop :=
  ∀ a b : Array F,
    a.size = d → b.size = d →
    ct (mulRqPhi (superneoBarBlock bar a) b) = innerProduct a b

/-! ### Canonical Native Bar Matrix (paper instance) -/

/-- Convert a machine-word literal into field element in `F`. -/
def thm3ToF (x : Nat) : F := F.ofNat x

/-- Convert a Nat vector into field vector in `F`. -/
def thm3ToFArray (xs : Array Nat) : Array F :=
  xs.map thm3ToF

/-- Convert a Nat matrix into field matrix in `F`. -/
def thm3ToFMatrix (m : Array (Array Nat)) : Array (Array F) :=
  m.map thm3ToFArray

/-- Closed-form entry formula for the canonical native bar matrix. -/
def nativeBarEntry (i j : Nat) : F :=
  if hi0 : i = 0 then
    if hj0 : j = 0 then (1 : F) else 0
  else if hiLt : i < d / 2 then
    if hj : j = (d / 2 - i) ∨ j = (d - i) then (-1 : F) else 0
  else
    if hj : j = (d - i) then (-1 : F) else 0

/-- Canonical native bar matrix used by theorem-native closures. -/
def nativeBarMatrix : Array (Array F) :=
  Array.ofFn (fun i : Fin d =>
    Array.ofFn (fun j : Fin d => nativeBarEntry i.1 j.1))

/-- The closed-form native bar matrix matches the generated matrix artifact. -/
theorem nativeBarMatrix_eq_generated :
    nativeBarMatrix = thm3ToFMatrix SuperNeo.Generated.barMatrixU64 := by
  native_decide

@[simp] theorem nativeBarMatrix_size : nativeBarMatrix.size = d := by
  simp [nativeBarMatrix]

theorem nativeBarMatrix_row_size
    (i : Nat)
    (hi : i < nativeBarMatrix.size) :
    (nativeBarMatrix[i]'hi).size = d := by
  simp [nativeBarMatrix] at hi ⊢

/-! ### Native Basis-Level Closure Evidence -/

/-- Canonical basis vector in `F^d`. -/
def basisVec (i : Fin d) : Array F :=
  Array.ofFn (fun j : Fin d => if j = i then (1 : F) else 0)

@[simp] theorem basisVec_size (i : Fin d) : (basisVec i).size = d := by
  simp [basisVec]

/--
Basis-level Theorem-3 identity for the canonical native bar matrix.

This is a fully constructive finite check over `d × d` basis pairs and serves
as executable algebraic evidence for the native transform kernel.
-/
theorem thm3Core_native_on_basis :
    ∀ i j : Fin d,
      ct (mulRqPhi (superneoBarBlock nativeBarMatrix (basisVec i)) (basisVec j)) =
        (if i = j then (1 : F) else 0) := by
  native_decide

/-! ### Algebraic linearity helpers for native closure -/

private theorem list_foldl_congr
  {α β : Type}
  (l : List α)
  (init : β)
  (f g : β → α → β)
  (hfg : ∀ acc x, x ∈ l → f acc x = g acc x) :
  l.foldl f init = l.foldl g init := by
  induction l generalizing init with
  | nil =>
      rfl
  | cons x xs ih =>
      have hHead : f init x = g init x := hfg init x (by simp)
      calc
        (x :: xs).foldl f init = xs.foldl f (f init x) := by rfl
        _ = xs.foldl f (g init x) := by rw [hHead]
        _ = xs.foldl g (g init x) := by
              apply ih
              intro acc y hy
              exact hfg acc y (by simp [hy])
        _ = (x :: xs).foldl g init := by rfl

private theorem innerProduct_vecAdd_right_of_size_d
    (a b c : Array F)
    (ha : a.size = d)
    (hb : b.size = d)
    (hc : c.size = d) :
    innerProduct a (vecAdd b c) = innerProduct a b + innerProduct a c := by
  have hbc : b.size = c.size := by simpa [hb, hc]
  have hVec : (vecAdd b c).size = d := by
    simpa [hb] using (vecAdd_size_of_eq hbc)
  have hEq : a.size = (vecAdd b c).size := by simpa [ha, hVec]
  unfold innerProduct
  have hMain :
      (List.range d).foldl
          (fun acc i => acc + a[i]! * (vecAdd b c)[i]!)
          0
        =
      (List.range d).foldl (fun acc i => acc + a[i]! * b[i]!) 0 +
      (List.range d).foldl (fun acc i => acc + a[i]! * c[i]!) 0 := by
    let t1 : Nat → F := fun i => a[i]! * b[i]!
    let t2 : Nat → F := fun i => a[i]! * c[i]!
    have hFold :
        (List.range d).foldl (fun acc i => acc + (t1 i + t2 i)) (0 + 0) =
          (List.range d).foldl (fun acc i => acc + t1 i) 0 +
            (List.range d).foldl (fun acc i => acc + t2 i) 0 := by
      simpa using (foldl_add_linearity (l := List.range d) (t1 := t1) (t2 := t2) (acc1 := 0) (acc2 := 0))
    have hPointwise :
        (List.range d).foldl
            (fun acc i => acc + a[i]! * (vecAdd b c)[i]!)
            0
          =
        (List.range d).foldl (fun acc i => acc + (t1 i + t2 i)) 0 := by
      apply list_foldl_congr
        (l := List.range d)
        (init := (0 : F))
        (f := fun acc i => acc + a[i]! * (vecAdd b c)[i]!)
        (g := fun acc i => acc + (t1 i + t2 i))
      intro acc i hiMem
      have hi : i < d := by simpa [List.mem_range] using hiMem
      have hiA : i < a.size := by simpa [ha] using hi
      have hiB : i < b.size := by simpa [hb] using hi
      have hiC : i < c.size := by simpa [hc] using hi
      have hVecIdx :
          (vecAdd b c)[i]! = b[i]! + c[i]! := by
        have hCoeff :
            coeffAt (vecAdd b c) i = coeffAt b i + coeffAt c i :=
          coeffAt_vecAdd_of_size_d b c hb hc i hi
        have hGetD :
            (vecAdd b c).getD i 0 = b.getD i 0 + c.getD i 0 := by
          simpa [coeffAt, hi] using hCoeff
        have hL : (vecAdd b c)[i]! = (vecAdd b c).getD i 0 := by
          simpa using (Array.getElem!_eq_getD (xs := (vecAdd b c)) (i := i))
        have hB : b[i]! = b.getD i 0 := by
          simpa using (Array.getElem!_eq_getD (xs := b) (i := i))
        have hC : c[i]! = c.getD i 0 := by
          simpa using (Array.getElem!_eq_getD (xs := c) (i := i))
        calc
          (vecAdd b c)[i]!
              = (vecAdd b c).getD i 0 := hL
          _ = b.getD i 0 + c.getD i 0 := hGetD
          _ = b[i]! + c[i]! := by simpa [hB, hC]
      have hDistrib :
          a[i]! * (b[i]! + c[i]!) = t1 i + t2 i := by
        simp [t1, t2, Lean.Grind.Fin.left_distrib]
      simpa [hVecIdx] using congrArg (fun x => acc + x) hDistrib
    calc
      (List.range d).foldl
          (fun acc i => acc + a[i]! * (vecAdd b c)[i]!)
          0
          = (List.range d).foldl (fun acc i => acc + (t1 i + t2 i)) 0 := hPointwise
      _ = (List.range d).foldl (fun acc i => acc + t1 i) 0 +
            (List.range d).foldl (fun acc i => acc + t2 i) 0 := by
            simpa using hFold
  calc
    innerProduct a (vecAdd b c)
        = (List.range d).foldl
            (fun acc i => acc + a[i]! * (vecAdd b c)[i]!)
            0 := by
              simp [innerProduct, hEq, hVec]
    _ = (List.range d).foldl (fun acc i => acc + a[i]! * b[i]!) 0 +
          (List.range d).foldl (fun acc i => acc + a[i]! * c[i]!) 0 := hMain
    _ = innerProduct a b + innerProduct a c := by
          simp [innerProduct, ha, hb, hc]

private theorem innerProduct_vecScale_right_of_size_d
    (a b : Array F)
    (s : F)
    (ha : a.size = d)
    (hb : b.size = d) :
    innerProduct a (vecScale s b) = s * innerProduct a b := by
  have hVec : (vecScale s b).size = d := by simpa [hb] using (vecScale_size s b)
  have hEq : a.size = (vecScale s b).size := by simpa [ha, hVec]
  calc
    innerProduct a (vecScale s b)
        = (List.range d).foldl
        (fun acc i => acc + a[i]! * (vecScale s b)[i]!)
        0 := by
          simp [innerProduct, hEq, hVec]
    _ 
        =
      (List.range d).foldl (fun acc i => acc + s * (a[i]! * b[i]!)) 0 := by
        apply list_foldl_congr
          (l := List.range d)
          (init := (0 : F))
          (f := fun acc i => acc + a[i]! * (vecScale s b)[i]!)
          (g := fun acc i => acc + s * (a[i]! * b[i]!))
        intro acc i hiMem
        have hi : i < d := by simpa [List.mem_range] using hiMem
        have hiB : i < b.size := by simpa [hb] using hi
        have hVS :
            (vecScale s b)[i]! = s * b[i]! := by
          have hGetD :
              (vecScale s b).getD i 0 = s * b.getD i 0 := by
            simpa [coeffAt, hi] using (coeffAt_vecScale_of_size_d s b hb i hi)
          have hL : (vecScale s b)[i]! = (vecScale s b).getD i 0 := by
            simpa using (Array.getElem!_eq_getD (xs := (vecScale s b)) (i := i))
          have hR : b[i]! = b.getD i 0 := by
            simpa using (Array.getElem!_eq_getD (xs := b) (i := i))
          calc
            (vecScale s b)[i]! = (vecScale s b).getD i 0 := hL
            _ = s * b.getD i 0 := hGetD
            _ = s * b[i]! := by simpa [hR]
        have hTerm :
            a[i]! * (vecScale s b)[i]! = s * (a[i]! * b[i]!) := by
          calc
            a[i]! * (vecScale s b)[i]!
                = a[i]! * (s * b[i]!) := by simp [hVS]
            _ = (a[i]! * s) * b[i]! := by
                  simpa using (Lean.Grind.Fin.mul_assoc (n := Goldilocks.q) (a[i]!) s (b[i]!)).symm
            _ = (s * a[i]!) * b[i]! := by
                  simpa using congrArg (fun t => t * b[i]!)
                    (Lean.Grind.Fin.mul_comm (n := Goldilocks.q) (a[i]!) s)
            _ = s * (a[i]! * b[i]!) := by
                  simpa using (Lean.Grind.Fin.mul_assoc (n := Goldilocks.q) s (a[i]!) (b[i]!))
        simpa [hTerm]
    _ = s * (List.range d).foldl (fun acc i => acc + a[i]! * b[i]!) 0 := by
          have hFold := foldl_mul_left_distrib
            (l := List.range d)
            (t := fun i => a[i]! * b[i]!)
            (acc := (0 : F))
            (c := s)
          simpa using hFold.symm
    _ = s * innerProduct a b := by
          simp [innerProduct, ha, hb]

private theorem innerProduct_comm_of_size_d
    (a b : Array F)
    (ha : a.size = d)
    (hb : b.size = d) :
    innerProduct a b = innerProduct b a := by
  have hEq : a.size = b.size := by simpa [ha, hb]
  unfold innerProduct
  have hFold :
      (List.range d).foldl (fun acc i => acc + a[i]! * b[i]!) 0 =
        (List.range d).foldl (fun acc i => acc + b[i]! * a[i]!) 0 := by
    apply list_foldl_congr
      (l := List.range d)
      (init := (0 : F))
      (f := fun acc i => acc + a[i]! * b[i]!)
      (g := fun acc i => acc + b[i]! * a[i]!)
    intro acc i _hiMem
    simpa [Lean.Grind.Fin.mul_comm] using
      congrArg (fun z => acc + z)
        (Lean.Grind.Fin.mul_comm (n := Goldilocks.q) (a[i]!) (b[i]!))
  simp [hEq, ha, hb, hFold]

private theorem innerProduct_vecAdd_left_of_size_d
    (a b c : Array F)
    (ha : a.size = d)
    (hb : b.size = d)
    (hc : c.size = d) :
    innerProduct (vecAdd a b) c = innerProduct a c + innerProduct b c := by
  have hab : a.size = b.size := by simpa [ha, hb]
  have hVec : (vecAdd a b).size = d := by
    simpa [ha] using (vecAdd_size_of_eq hab)
  calc
    innerProduct (vecAdd a b) c = innerProduct c (vecAdd a b) := by
      symm
      exact innerProduct_comm_of_size_d c (vecAdd a b) hc hVec
    _ = innerProduct c a + innerProduct c b :=
      innerProduct_vecAdd_right_of_size_d c a b hc ha hb
    _ = innerProduct a c + innerProduct b c := by
      rw [innerProduct_comm_of_size_d c a hc ha, innerProduct_comm_of_size_d c b hc hb]

private theorem innerProduct_vecScale_left_of_size_d
    (a b : Array F)
    (s : F)
    (ha : a.size = d)
    (hb : b.size = d) :
    innerProduct (vecScale s a) b = s * innerProduct a b := by
  have hVec : (vecScale s a).size = d := by simpa [ha] using (vecScale_size s a)
  calc
    innerProduct (vecScale s a) b = innerProduct b (vecScale s a) := by
      symm
      exact innerProduct_comm_of_size_d b (vecScale s a) hb hVec
    _ = s * innerProduct b a :=
      innerProduct_vecScale_right_of_size_d b a s hb ha
    _ = s * innerProduct a b := by
      rw [innerProduct_comm_of_size_d b a hb ha]

private theorem dotBySize_vecAdd_right_of_size_d
    (row x y : Array F)
    (hRow : row.size = d)
    (hx : x.size = d)
    (hy : y.size = d) :
    dotBySize row (vecAdd x y) = dotBySize row x + dotBySize row y := by
  have hxy : x.size = y.size := by simpa [hx, hy]
  have hVec : (vecAdd x y).size = d := by
    simpa [hx] using (vecAdd_size_of_eq hxy)
  have hEq : row.size = (vecAdd x y).size := by simpa [hRow, hVec]
  calc
    dotBySize row (vecAdd x y)
        = (List.range d).foldl
        (fun acc i => acc + row[i]! * (vecAdd x y)[i]!)
        0 := by
          simp [dotBySize, hEq, hVec]
    _
        =
      (List.range d).foldl (fun acc i => acc + row[i]! * x[i]!) 0 +
      (List.range d).foldl (fun acc i => acc + row[i]! * y[i]!) 0 := by
        let t1 : Nat → F := fun i => row[i]! * x[i]!
        let t2 : Nat → F := fun i => row[i]! * y[i]!
        have hFold :
            (List.range d).foldl (fun acc i => acc + (t1 i + t2 i)) (0 + 0) =
              (List.range d).foldl (fun acc i => acc + t1 i) 0 +
                (List.range d).foldl (fun acc i => acc + t2 i) 0 := by
          simpa using (foldl_add_linearity (l := List.range d) (t1 := t1) (t2 := t2) (acc1 := 0) (acc2 := 0))
        have hPointwise :
            (List.range d).foldl
                (fun acc i => acc + row[i]! * (vecAdd x y)[i]!)
                0
              =
            (List.range d).foldl (fun acc i => acc + (t1 i + t2 i)) 0 := by
          apply list_foldl_congr
            (l := List.range d)
            (init := (0 : F))
            (f := fun acc i => acc + row[i]! * (vecAdd x y)[i]!)
            (g := fun acc i => acc + (t1 i + t2 i))
          intro acc i hiMem
          have hi : i < d := by simpa [List.mem_range] using hiMem
          have hCoeff :
              coeffAt (vecAdd x y) i = coeffAt x i + coeffAt y i :=
            coeffAt_vecAdd_of_size_d x y hx hy i hi
          have hGetD :
              (vecAdd x y).getD i 0 = x.getD i 0 + y.getD i 0 := by
            simpa [coeffAt, hi] using hCoeff
          have hL : (vecAdd x y)[i]! = (vecAdd x y).getD i 0 := by
            simpa using (Array.getElem!_eq_getD (xs := (vecAdd x y)) (i := i))
          have hX : x[i]! = x.getD i 0 := by
            simpa using (Array.getElem!_eq_getD (xs := x) (i := i))
          have hY : y[i]! = y.getD i 0 := by
            simpa using (Array.getElem!_eq_getD (xs := y) (i := i))
          have hVecIdx : (vecAdd x y)[i]! = x[i]! + y[i]! := by
            calc
              (vecAdd x y)[i]! = (vecAdd x y).getD i 0 := hL
              _ = x.getD i 0 + y.getD i 0 := hGetD
              _ = x[i]! + y[i]! := by simpa [hX, hY]
          have hDistrib :
              row[i]! * (x[i]! + y[i]!) = t1 i + t2 i := by
            simp [t1, t2, Lean.Grind.Fin.left_distrib]
          simpa [hVecIdx] using congrArg (fun z => acc + z) hDistrib
        calc
          (List.range d).foldl
              (fun acc i => acc + row[i]! * (vecAdd x y)[i]!)
              0
              = (List.range d).foldl (fun acc i => acc + (t1 i + t2 i)) 0 := hPointwise
          _ = (List.range d).foldl (fun acc i => acc + t1 i) 0 +
                (List.range d).foldl (fun acc i => acc + t2 i) 0 := by
                simpa using hFold
    _ = dotBySize row x + dotBySize row y := by
          simp [dotBySize, hRow, hx, hy]

private theorem dotBySize_vecScale_right_of_size_d
    (row x : Array F)
    (s : F)
    (hRow : row.size = d)
    (hx : x.size = d) :
    dotBySize row (vecScale s x) = s * dotBySize row x := by
  have hVec : (vecScale s x).size = d := by simpa [hx] using (vecScale_size s x)
  have hEq : row.size = (vecScale s x).size := by simpa [hRow, hVec]
  calc
    dotBySize row (vecScale s x)
        = (List.range d).foldl
        (fun acc i => acc + row[i]! * (vecScale s x)[i]!)
        0 := by
          simp [dotBySize, hEq, hVec]
    _
        =
      (List.range d).foldl (fun acc i => acc + s * (row[i]! * x[i]!)) 0 := by
        apply list_foldl_congr
          (l := List.range d)
          (init := (0 : F))
          (f := fun acc i => acc + row[i]! * (vecScale s x)[i]!)
          (g := fun acc i => acc + s * (row[i]! * x[i]!))
        intro acc i hiMem
        have hi : i < d := by simpa [List.mem_range] using hiMem
        have hCoeff :
            coeffAt (vecScale s x) i = s * coeffAt x i :=
          coeffAt_vecScale_of_size_d s x hx i hi
        have hGetD :
            (vecScale s x).getD i 0 = s * x.getD i 0 := by
          simpa [coeffAt, hi] using hCoeff
        have hL : (vecScale s x)[i]! = (vecScale s x).getD i 0 := by
          simpa using (Array.getElem!_eq_getD (xs := (vecScale s x)) (i := i))
        have hX : x[i]! = x.getD i 0 := by
          simpa using (Array.getElem!_eq_getD (xs := x) (i := i))
        have hVS : (vecScale s x)[i]! = s * x[i]! := by
          calc
            (vecScale s x)[i]! = (vecScale s x).getD i 0 := hL
            _ = s * x.getD i 0 := hGetD
            _ = s * x[i]! := by simpa [hX]
        have hTerm :
            row[i]! * (vecScale s x)[i]! = s * (row[i]! * x[i]!) := by
          calc
            row[i]! * (vecScale s x)[i]!
                = row[i]! * (s * x[i]!) := by simp [hVS]
            _ = (row[i]! * s) * x[i]! := by
                  simpa using (Lean.Grind.Fin.mul_assoc (n := Goldilocks.q) (row[i]!) s (x[i]!)).symm
            _ = (s * row[i]!) * x[i]! := by
                  simpa using congrArg (fun t => t * x[i]!)
                    (Lean.Grind.Fin.mul_comm (n := Goldilocks.q) (row[i]!) s)
            _ = s * (row[i]! * x[i]!) := by
                  simpa using (Lean.Grind.Fin.mul_assoc (n := Goldilocks.q) s (row[i]!) (x[i]!))
        simpa [hTerm]
    _ = s * (List.range d).foldl (fun acc i => acc + row[i]! * x[i]!) 0 := by
          have hFold := foldl_mul_left_distrib
            (l := List.range d)
            (t := fun i => row[i]! * x[i]!)
            (acc := (0 : F))
            (c := s)
          simpa using hFold.symm
    _ = s * dotBySize row x := by
          simp [dotBySize, hRow, hx]

private theorem superneoBarBlock_native_vecAdd
    (x y : Array F)
    (hx : x.size = d)
    (hy : y.size = d) :
    superneoBarBlock nativeBarMatrix (vecAdd x y) =
      vecAdd (superneoBarBlock nativeBarMatrix x) (superneoBarBlock nativeBarMatrix y) := by
  have hxy : x.size = y.size := by simpa [hx, hy]
  have hVec : (vecAdd x y).size = d := by
    simpa [hx] using (vecAdd_size_of_eq hxy)
  have hOutEq :
      (superneoBarBlock nativeBarMatrix x).size =
        (superneoBarBlock nativeBarMatrix y).size := by
    simp [superneoBarBlock, nativeBarMatrix_size, hx, hy]
  have hRSize :
      (vecAdd (superneoBarBlock nativeBarMatrix x) (superneoBarBlock nativeBarMatrix y)).size = d := by
    calc
      (vecAdd (superneoBarBlock nativeBarMatrix x) (superneoBarBlock nativeBarMatrix y)).size
          = (superneoBarBlock nativeBarMatrix x).size := vecAdd_size_of_eq hOutEq
      _ = d := by simp [superneoBarBlock, nativeBarMatrix_size, hx]
  have hLSize :
      (superneoBarBlock nativeBarMatrix (vecAdd x y)).size = d := by
    simp [superneoBarBlock, nativeBarMatrix_size, hVec]
  apply Array.ext
  · exact hLSize.trans hRSize.symm
  · intro i hiL hiR
    have hi : i < d := by simpa [superneoBarBlock, nativeBarMatrix_size, hx, hy, hVec] using hiL
    have hRowSize : (nativeBarMatrix[i]'(by simpa [nativeBarMatrix_size] using hi)).size = d := by
      exact nativeBarMatrix_row_size i (by simpa [nativeBarMatrix_size] using hi)
    have hDot :
        dotBySize (nativeBarMatrix[i]'(by simpa [nativeBarMatrix_size] using hi)) (vecAdd x y) =
          dotBySize (nativeBarMatrix[i]'(by simpa [nativeBarMatrix_size] using hi)) x +
            dotBySize (nativeBarMatrix[i]'(by simpa [nativeBarMatrix_size] using hi)) y := by
      exact dotBySize_vecAdd_right_of_size_d
        (row := (nativeBarMatrix[i]'(by simpa [nativeBarMatrix_size] using hi)))
        (x := x) (y := y) hRowSize hx hy
    simpa [superneoBarBlock, nativeBarMatrix_size, hx, hy, hVec, vecAdd, hxy, hi] using hDot

private theorem superneoBarBlock_native_vecScale
    (x : Array F)
    (s : F)
    (hx : x.size = d) :
    superneoBarBlock nativeBarMatrix (vecScale s x) =
      vecScale s (superneoBarBlock nativeBarMatrix x) := by
  have hVec : (vecScale s x).size = d := by simpa [hx] using (vecScale_size s x)
  apply Array.ext
  · simp [superneoBarBlock, nativeBarMatrix_size, hx, hVec, vecScale]
  · intro i hiL hiR
    have hi : i < d := by simpa [superneoBarBlock, nativeBarMatrix_size, hx, hVec, vecScale] using hiL
    have hRowSize : (nativeBarMatrix[i]'(by simpa [nativeBarMatrix_size] using hi)).size = d := by
      exact nativeBarMatrix_row_size i (by simpa [nativeBarMatrix_size] using hi)
    have hDot :
        dotBySize (nativeBarMatrix[i]'(by simpa [nativeBarMatrix_size] using hi)) (vecScale s x) =
          s * dotBySize (nativeBarMatrix[i]'(by simpa [nativeBarMatrix_size] using hi)) x := by
      exact dotBySize_vecScale_right_of_size_d
        (row := (nativeBarMatrix[i]'(by simpa [nativeBarMatrix_size] using hi)))
        (x := x) (s := s) hRowSize hx
    simpa [superneoBarBlock, nativeBarMatrix_size, hx, hVec, vecScale, hi] using hDot

/-! ### Basis expansion for constructive native closure -/

/-- Basis vector indexed by a natural coordinate. -/
private def basisVecNat (i : Nat) : Array F :=
  Array.ofFn (fun j : Fin d => if j.1 = i then (1 : F) else 0)

@[simp] private theorem basisVecNat_size (i : Nat) : (basisVecNat i).size = d := by
  simp [basisVecNat]

private theorem basisVecNat_eq_basisVec
    (i : Nat)
    (hi : i < d) :
    basisVecNat i = basisVec ⟨i, hi⟩ := by
  apply Array.ext
  · simp [basisVecNat, basisVec]
  · intro j hj1 hj2
    have hj : j < d := by
      simpa [basisVecNat] using hj1
    have hEq : ((⟨j, hj⟩ : Fin d) = (⟨i, hi⟩ : Fin d)) ↔ j = i := by
      constructor
      · intro h
        exact congrArg Fin.val h
      · intro h
        apply Fin.ext
        simpa using h
    simp [basisVecNat, basisVec, hEq]

/--
Prefix basis expansion:
`n` terms of `Σ_i a[i] * e_i`, built with vector add/scale over arrays.
-/
private def basisExpandPrefix (a : Array F) : Nat → Array F
  | 0 => Array.replicate d 0
  | n + 1 =>
      vecAdd (basisExpandPrefix a n) (vecScale a[n]! (basisVecNat n))

/-- Full basis expansion over all `d` coordinates. -/
private def basisExpand (a : Array F) : Array F :=
  basisExpandPrefix a d

@[simp] private theorem basisExpand_eq_prefix (a : Array F) :
    basisExpand a = basisExpandPrefix a d := by
  rfl

@[simp] private theorem basisExpandPrefix_zero_eq (a : Array F) :
    basisExpandPrefix a 0 = Array.replicate d (0 : F) := by
  rfl

@[simp] private theorem basisExpandPrefix_succ_eq (a : Array F) (n : Nat) :
    basisExpandPrefix a (n + 1) =
      vecAdd (basisExpandPrefix a n) (vecScale a[n]! (basisVecNat n)) := by
  rfl

private theorem basisExpandPrefix_size
    (a : Array F) :
    ∀ n : Nat, (basisExpandPrefix a n).size = d := by
  intro n
  induction n with
  | zero =>
      simp [basisExpandPrefix]
  | succ n ih =>
      have hScaled : (vecScale a[n]! (basisVecNat n)).size = d := by
        simp [basisVecNat]
      calc
        (basisExpandPrefix a (n + 1)).size
            = (vecAdd (basisExpandPrefix a n) (vecScale a[n]! (basisVecNat n))).size := by
                simp [basisExpandPrefix]
        _ = (basisExpandPrefix a n).size := by
              exact vecAdd_size_of_eq (by simpa [ih, hScaled])
        _ = d := ih

private theorem coeffAt_basisVecNat_of_lt
    (i j : Nat)
    (hj : j < d) :
    coeffAt (basisVecNat i) j = (if j = i then (1 : F) else 0) := by
  unfold coeffAt basisVecNat
  simp [hj]

private theorem coeffAt_basisExpandPrefix
    (a : Array F)
    (ha : a.size = d) :
    ∀ n : Nat, n ≤ d →
      ∀ j : Nat, j < d →
        coeffAt (basisExpandPrefix a n) j = (if j < n then a[j]! else 0) := by
  intro n hn
  induction n with
  | zero =>
      intro j hj
      simp [basisExpandPrefix, coeffAt, hj]
  | succ n ih =>
      intro j hj
      have hnLe : n ≤ d := Nat.le_trans (Nat.le_succ n) hn
      have hnLt : n < d := Nat.lt_of_lt_of_le (Nat.lt_succ_self n) hn
      have hPrefSize : (basisExpandPrefix a n).size = d := basisExpandPrefix_size a n
      have hScaledSize : (vecScale a[n]! (basisVecNat n)).size = d := by
        simp [basisVecNat]
      have hAdd :
          coeffAt (basisExpandPrefix a (n + 1)) j =
            coeffAt (basisExpandPrefix a n) j +
              coeffAt (vecScale a[n]! (basisVecNat n)) j := by
        calc
          coeffAt (basisExpandPrefix a (n + 1)) j
              = coeffAt (vecAdd (basisExpandPrefix a n) (vecScale a[n]! (basisVecNat n))) j := by
                  simp [basisExpandPrefix]
          _ = coeffAt (basisExpandPrefix a n) j + coeffAt (vecScale a[n]! (basisVecNat n)) j := by
                exact coeffAt_vecAdd_of_size_d
                  (x := basisExpandPrefix a n)
                  (y := vecScale a[n]! (basisVecNat n))
                  hPrefSize hScaledSize j hj
      have hScale :
          coeffAt (vecScale a[n]! (basisVecNat n)) j =
            a[n]! * coeffAt (basisVecNat n) j := by
        exact coeffAt_vecScale_of_size_d
          (s := a[n]!)
          (x := basisVecNat n)
          (hx := by simp [basisVecNat])
          (k := j)
          hj
      have hPref :
          coeffAt (basisExpandPrefix a n) j = (if j < n then a[j]! else 0) := by
        exact ih hnLe j hj
      have hBasis :
          coeffAt (basisVecNat n) j = (if j = n then (1 : F) else 0) := by
        exact coeffAt_basisVecNat_of_lt n j hj
      calc
        coeffAt (basisExpandPrefix a (n + 1)) j
            = coeffAt (basisExpandPrefix a n) j +
                coeffAt (vecScale a[n]! (basisVecNat n)) j := hAdd
        _ = (if j < n then a[j]! else 0) + (a[n]! * (if j = n then (1 : F) else 0)) := by
              simp [hScale, hPref, hBasis]
        _ = (if j < n + 1 then a[j]! else 0) := by
              by_cases hjn : j < n
              · have hjnNe : j ≠ n := Nat.ne_of_lt hjn
                have hlt : j < n + 1 := Nat.lt_trans hjn (Nat.lt_succ_self n)
                have hMulZero : a[n]! * (0 : F) = 0 := by
                  simpa using (Fin.mul_zero (n := Goldilocks.q) (a[n]!))
                calc
                  (if j < n then a[j]! else 0) + (a[n]! * (if j = n then (1 : F) else 0))
                      = a[j]! + (a[n]! * (0 : F)) := by simp [hjn, hjnNe]
                  _ = a[j]! + 0 := by simp [hMulZero]
                  _ = a[j]! := by
                        simpa using (Fin.add_zero (n := Goldilocks.q) (a[j]!))
                  _ = (if j < n + 1 then a[j]! else 0) := by simp [hlt]
              · by_cases hjeq : j = n
                · subst hjeq
                  have hMulOne : a[j]! * (1 : F) = a[j]! := by
                    simpa using (Fin.mul_one (n := Goldilocks.q) (a[j]!))
                  have hZeroAdd : (0 : F) + a[j]! = a[j]! := by
                    simpa using (Fin.zero_add (n := Goldilocks.q) (a[j]!))
                  calc
                    (if j < j then a[j]! else 0) + (a[j]! * (if j = j then (1 : F) else 0))
                        = 0 + (a[j]! * (1 : F)) := by simp
                    _ = 0 + a[j]! := by simp [hMulOne]
                    _ = a[j]! := hZeroAdd
                    _ = (if j < j + 1 then a[j]! else 0) := by simp
                · have hngt : n < j := by
                    omega
                  have hNot : ¬ j < n + 1 := Nat.not_lt_of_ge (Nat.succ_le_of_lt hngt)
                  have hMulZero : a[n]! * (0 : F) = 0 := by
                    simpa using (Fin.mul_zero (n := Goldilocks.q) (a[n]!))
                  have hZeroAdd : (0 : F) + 0 = 0 := by
                    simpa using (Fin.zero_add (n := Goldilocks.q) (0 : F))
                  calc
                    (if j < n then a[j]! else 0) + (a[n]! * (if j = n then (1 : F) else 0))
                        = 0 + (a[n]! * (0 : F)) := by simp [hjn, hjeq]
                    _ = 0 + 0 := by simp [hMulZero]
                    _ = 0 := hZeroAdd
                    _ = (if j < n + 1 then a[j]! else 0) := by simp [hNot]

attribute [irreducible] basisExpandPrefix basisExpand

set_option maxRecDepth 4096 in
set_option maxHeartbeats 1200000 in
private theorem basisExpand_eq
    (a : Array F)
    (ha : a.size = d) :
    basisExpand a = a := by
  have hSize : (basisExpand a).size = d := by
    unfold basisExpand
    exact basisExpandPrefix_size a d
  apply Array.ext
  · exact hSize.trans ha.symm
  · intro i hiL hiR
    have hiExpand : i < (basisExpand a).size := by
      simpa [basisExpand_eq_prefix] using hiL
    have hi : i < d := lt_of_lt_of_eq hiExpand hSize
    have hLGet : (basisExpand a)[i]'hiL = (basisExpand a).getD i 0 := by
      exact Array.getElem_eq_getD
        (xs := basisExpand a) (i := i) (h := hiL) (fallback := (0 : F))
    have hCoeffL :
        coeffAt (basisExpand a) i = a[i]! := by
      have h := coeffAt_basisExpandPrefix a ha d (Nat.le_refl d) i hi
      simpa [basisExpand, hi] using h
    have hRGet : a[i]'hiR = a.getD i 0 := by
      exact Array.getElem_eq_getD (xs := a) (i := i) (h := hiR) (fallback := (0 : F))
    have hRBang : a[i]! = a.getD i 0 := by
      simpa using (Array.getElem!_eq_getD (xs := a) (i := i))
    calc
      (basisExpand a)[i]'hiL
          = (basisExpand a).getD i 0 := hLGet
      _ = coeffAt (basisExpand a) i := by
            simp [coeffAt, hi]
      _ = a[i]! := hCoeffL
      _ = a.getD i 0 := hRBang
      _ = a[i]'hiR := hRGet.symm

/-! ### Native-kernel bilinearity and basis extension -/

/-- Native Theorem-3 kernel specialized to `nativeBarMatrix`. -/
private def nativeKernel (a b : Array F) : F :=
  ct (mulRqPhi (superneoBarBlock nativeBarMatrix a) b)

private theorem nativeKernel_vecAdd_right_of_size_d
    (a b c : Array F)
    (ha : a.size = d)
    (hb : b.size = d)
    (hc : c.size = d) :
    nativeKernel a (vecAdd b c) =
      nativeKernel a b + nativeKernel a c := by
  have hBarA : (superneoBarBlock nativeBarMatrix a).size = d := by
    simp [superneoBarBlock, nativeBarMatrix_size, ha]
  simpa [nativeKernel] using
    ct_mulRqPhi_vecAdd_right_of_size_d
      (a := superneoBarBlock nativeBarMatrix a)
      (x := b)
      (y := c)
      hb hc

private theorem nativeKernel_vecScale_right_of_size_d
    (a x : Array F)
    (s : F)
    (ha : a.size = d)
    (hx : x.size = d) :
    nativeKernel a (vecScale s x) =
      s * nativeKernel a x := by
  have hBarA : (superneoBarBlock nativeBarMatrix a).size = d := by
    simp [superneoBarBlock, nativeBarMatrix_size, ha]
  simpa [nativeKernel] using
    ct_mulRqPhi_vecScale_right_of_size_d
      (s := s)
      (a := superneoBarBlock nativeBarMatrix a)
      (x := x)
      hx

private theorem nativeKernel_vecAdd_left_of_size_d
    (a b c : Array F)
    (ha : a.size = d)
    (hb : b.size = d)
    (hc : c.size = d) :
    nativeKernel (vecAdd a b) c =
      nativeKernel a c + nativeKernel b c := by
  have hab : a.size = b.size := by simpa [ha, hb]
  have hVec : (vecAdd a b).size = d := by
    simpa [ha] using vecAdd_size_of_eq hab
  have hBarA : (superneoBarBlock nativeBarMatrix a).size = d := by
    simp [superneoBarBlock, nativeBarMatrix_size, ha]
  have hBarB : (superneoBarBlock nativeBarMatrix b).size = d := by
    simp [superneoBarBlock, nativeBarMatrix_size, hb]
  have hBarAdd :
      superneoBarBlock nativeBarMatrix (vecAdd a b) =
        vecAdd (superneoBarBlock nativeBarMatrix a) (superneoBarBlock nativeBarMatrix b) := by
    exact superneoBarBlock_native_vecAdd a b ha hb
  calc
    nativeKernel (vecAdd a b) c
        = ct (mulRqPhi
            (vecAdd (superneoBarBlock nativeBarMatrix a) (superneoBarBlock nativeBarMatrix b))
            c) := by
              simp [nativeKernel, hBarAdd]
    _ = ct (mulRqPhi (superneoBarBlock nativeBarMatrix a) c) +
          ct (mulRqPhi (superneoBarBlock nativeBarMatrix b) c) := by
            simpa using
              ct_mulRqPhi_vecAdd_left_of_size_d
                (x := superneoBarBlock nativeBarMatrix a)
                (y := superneoBarBlock nativeBarMatrix b)
                (b := c)
                hBarA hBarB
    _ = nativeKernel a c + nativeKernel b c := by
          simp [nativeKernel]

private theorem nativeKernel_vecScale_left_of_size_d
    (a c : Array F)
    (s : F)
    (ha : a.size = d)
    (hc : c.size = d) :
    nativeKernel (vecScale s a) c =
      s * nativeKernel a c := by
  have hVec : (vecScale s a).size = d := by
    simpa [ha] using vecScale_size s a
  have hBarA : (superneoBarBlock nativeBarMatrix a).size = d := by
    simp [superneoBarBlock, nativeBarMatrix_size, ha]
  have hBarScale :
      superneoBarBlock nativeBarMatrix (vecScale s a) =
        vecScale s (superneoBarBlock nativeBarMatrix a) := by
    exact superneoBarBlock_native_vecScale a s ha
  calc
    nativeKernel (vecScale s a) c
        = ct (mulRqPhi (vecScale s (superneoBarBlock nativeBarMatrix a)) c) := by
              simp [nativeKernel, hBarScale]
    _ = s * ct (mulRqPhi (superneoBarBlock nativeBarMatrix a) c) := by
          simpa using
            ct_mulRqPhi_vecScale_left_of_size_d
              (s := s)
              (x := superneoBarBlock nativeBarMatrix a)
              (b := c)
              hBarA
    _ = s * nativeKernel a c := by
          simp [nativeKernel]

private theorem innerProduct_basis_basis
    (i j : Fin d) :
    innerProduct (basisVec i) (basisVec j) =
      (if i = j then (1 : F) else 0) := by
  native_decide +revert

private theorem nativeKernel_basis_basisNat
    (i : Fin d)
    (n : Nat)
    (hn : n < d) :
    nativeKernel (basisVec i) (basisVecNat n) =
      (if i.1 = n then (1 : F) else 0) := by
  have hBasisNat : basisVecNat n = basisVec ⟨n, hn⟩ :=
    basisVecNat_eq_basisVec n hn
  have hEq : (i = (⟨n, hn⟩ : Fin d)) ↔ i.1 = n := by
    constructor
    · intro h
      exact congrArg Fin.val h
    · intro h
      apply Fin.ext
      simpa using h
  calc
    nativeKernel (basisVec i) (basisVecNat n)
        = nativeKernel (basisVec i) (basisVec ⟨n, hn⟩) := by
            simp [hBasisNat]
    _ = (if i = (⟨n, hn⟩ : Fin d) then (1 : F) else 0) := by
          simpa [nativeKernel] using thm3Core_native_on_basis i ⟨n, hn⟩
    _ = (if i.1 = n then (1 : F) else 0) := by
          simp [hEq]

private theorem innerProduct_basis_basisNat
    (i : Fin d)
    (n : Nat)
    (hn : n < d) :
    innerProduct (basisVec i) (basisVecNat n) =
      (if i.1 = n then (1 : F) else 0) := by
  have hBasisNat : basisVecNat n = basisVec ⟨n, hn⟩ :=
    basisVecNat_eq_basisVec n hn
  have hEq : (i = (⟨n, hn⟩ : Fin d)) ↔ i.1 = n := by
    constructor
    · intro h
      exact congrArg Fin.val h
    · intro h
      apply Fin.ext
      simpa using h
  calc
    innerProduct (basisVec i) (basisVecNat n)
        = innerProduct (basisVec i) (basisVec ⟨n, hn⟩) := by
            simp [hBasisNat]
    _ = (if i = (⟨n, hn⟩ : Fin d) then (1 : F) else 0) := by
          exact innerProduct_basis_basis i ⟨n, hn⟩
    _ = (if i.1 = n then (1 : F) else 0) := by
          simp [hEq]

private theorem zeroVec_eq_vecScale_zero_basisNat
    (n : Nat) :
    Array.replicate d (0 : F) = vecScale (0 : F) (basisVecNat n) := by
  apply Array.ext
  · simp [vecScale, basisVecNat]
  · intro i hiL hiR
    have hMul : (0 : F) * (if i = n then (1 : F) else 0) = 0 := by
      simpa using (Fin.zero_mul (n := Goldilocks.q) (if i = n then (1 : F) else 0))
    simpa [vecScale, basisVecNat, hMul]

private theorem nativeKernel_basis_left_prefix
    (i : Fin d)
    (b : Array F)
    (hb : b.size = d) :
    ∀ n : Nat, n ≤ d →
      nativeKernel (basisVec i) (basisExpandPrefix b n) =
        innerProduct (basisVec i) (basisExpandPrefix b n) := by
  intro n hn
  induction n with
  | zero =>
      have hZeroScale : basisExpandPrefix b 0 = vecScale (0 : F) (basisVecNat 0) := by
        simpa [basisExpandPrefix] using (zeroVec_eq_vecScale_zero_basisNat 0)
      have hBasisSize : (basisVec i).size = d := by simp
      have hBasisNatSize : (basisVecNat 0).size = d := by simp [basisVecNat]
      calc
        nativeKernel (basisVec i) (basisExpandPrefix b 0)
            = nativeKernel (basisVec i) (vecScale (0 : F) (basisVecNat 0)) := by
                simp [hZeroScale]
        _ = (0 : F) * nativeKernel (basisVec i) (basisVecNat 0) := by
              simpa using
                nativeKernel_vecScale_right_of_size_d
                  (a := basisVec i)
                  (x := basisVecNat 0)
                  (s := (0 : F))
                  hBasisSize hBasisNatSize
        _ = 0 := by
              have h0mul : (0 : F) * nativeKernel (basisVec i) (basisVecNat 0) = 0 := by
                simpa using (Fin.zero_mul (n := Goldilocks.q) (nativeKernel (basisVec i) (basisVecNat 0)))
              simpa [h0mul]
        _ = (0 : F) * innerProduct (basisVec i) (basisVecNat 0) := by
              have h0mul : (0 : F) * innerProduct (basisVec i) (basisVecNat 0) = 0 := by
                simpa using (Fin.zero_mul (n := Goldilocks.q) (innerProduct (basisVec i) (basisVecNat 0)))
              simpa [h0mul]
        _ = innerProduct (basisVec i) (vecScale (0 : F) (basisVecNat 0)) := by
              symm
              simpa using
                innerProduct_vecScale_right_of_size_d
                  (a := basisVec i)
                  (b := basisVecNat 0)
                  (s := (0 : F))
                  hBasisSize hBasisNatSize
        _ = innerProduct (basisVec i) (basisExpandPrefix b 0) := by
              simp [hZeroScale]
  | succ n ih =>
      have hnLe : n ≤ d := Nat.le_trans (Nat.le_succ n) hn
      have hnLt : n < d := Nat.lt_of_lt_of_le (Nat.lt_succ_self n) hn
      have hPrefSize : (basisExpandPrefix b n).size = d := basisExpandPrefix_size b n
      have hBasisSize : (basisVec i).size = d := by simp
      have hBasisNatSize : (basisVecNat n).size = d := by simp [basisVecNat]
      have hBasisEq :
          nativeKernel (basisVec i) (basisVecNat n) =
            innerProduct (basisVec i) (basisVecNat n) := by
        calc
          nativeKernel (basisVec i) (basisVecNat n)
              = (if i.1 = n then (1 : F) else 0) :=
                nativeKernel_basis_basisNat i n hnLt
          _ = innerProduct (basisVec i) (basisVecNat n) := by
                symm
                exact innerProduct_basis_basisNat i n hnLt
      calc
        nativeKernel (basisVec i) (basisExpandPrefix b (n + 1))
            = nativeKernel (basisVec i)
                (vecAdd (basisExpandPrefix b n) (vecScale b[n]! (basisVecNat n))) := by
                  simp [basisExpandPrefix]
        _ = nativeKernel (basisVec i) (basisExpandPrefix b n) +
              nativeKernel (basisVec i) (vecScale b[n]! (basisVecNat n)) := by
              exact nativeKernel_vecAdd_right_of_size_d
                (a := basisVec i)
                (b := basisExpandPrefix b n)
                (c := vecScale b[n]! (basisVecNat n))
                hBasisSize
                hPrefSize
                (by simp [basisVecNat])
        _ = innerProduct (basisVec i) (basisExpandPrefix b n) +
              nativeKernel (basisVec i) (vecScale b[n]! (basisVecNat n)) := by
              rw [ih hnLe]
        _ = innerProduct (basisVec i) (basisExpandPrefix b n) +
              (b[n]!) * nativeKernel (basisVec i) (basisVecNat n) := by
              simp [nativeKernel_vecScale_right_of_size_d, hBasisSize, hBasisNatSize]
        _ = innerProduct (basisVec i) (basisExpandPrefix b n) +
              (b[n]!) * innerProduct (basisVec i) (basisVecNat n) := by
              simp [hBasisEq]
        _ = innerProduct (basisVec i) (basisExpandPrefix b n) +
              innerProduct (basisVec i) (vecScale b[n]! (basisVecNat n)) := by
              have hScale :
                  innerProduct (basisVec i) (vecScale b[n]! (basisVecNat n)) =
                    (b[n]!) * innerProduct (basisVec i) (basisVecNat n) := by
                exact innerProduct_vecScale_right_of_size_d
                  (a := basisVec i)
                  (b := basisVecNat n)
                  (s := b[n]!)
                  hBasisSize hBasisNatSize
              rw [hScale]
        _ = innerProduct (basisVec i)
              (vecAdd (basisExpandPrefix b n) (vecScale b[n]! (basisVecNat n))) := by
              symm
              exact innerProduct_vecAdd_right_of_size_d
                (a := basisVec i)
                (b := basisExpandPrefix b n)
                (c := vecScale b[n]! (basisVecNat n))
                hBasisSize
                hPrefSize
                (by simp [basisVecNat])
        _ = innerProduct (basisVec i) (basisExpandPrefix b (n + 1)) := by
              simp [basisExpandPrefix_succ_eq]

private theorem nativeKernel_basis_left_all
    (i : Fin d)
    (b : Array F)
    (hb : b.size = d) :
    nativeKernel (basisVec i) b =
      innerProduct (basisVec i) b := by
  have hPrefix :
      nativeKernel (basisVec i) (basisExpandPrefix b d) =
        innerProduct (basisVec i) (basisExpandPrefix b d) :=
    nativeKernel_basis_left_prefix i b hb d (Nat.le_refl d)
  have hExpand : basisExpand b = b := basisExpand_eq b hb
  calc
    nativeKernel (basisVec i) b
        = nativeKernel (basisVec i) (basisExpand b) := by
            rw [hExpand]
    _ = nativeKernel (basisVec i) (basisExpandPrefix b d) := by
          simp [basisExpand_eq_prefix]
    _ = innerProduct (basisVec i) (basisExpandPrefix b d) := hPrefix
    _ = innerProduct (basisVec i) (basisExpand b) := by
          simp [basisExpand_eq_prefix]
    _ = innerProduct (basisVec i) b := by
          rw [hExpand]

private theorem nativeKernel_basisNat_left_all
    (n : Nat)
    (hn : n < d)
    (b : Array F)
    (hb : b.size = d) :
    nativeKernel (basisVecNat n) b =
      innerProduct (basisVecNat n) b := by
  have hBasis : basisVecNat n = basisVec ⟨n, hn⟩ :=
    basisVecNat_eq_basisVec n hn
  calc
    nativeKernel (basisVecNat n) b
        = nativeKernel (basisVec ⟨n, hn⟩) b := by
            simp [hBasis]
    _ = innerProduct (basisVec ⟨n, hn⟩) b :=
          nativeKernel_basis_left_all ⟨n, hn⟩ b hb
    _ = innerProduct (basisVecNat n) b := by
          simp [hBasis]

private theorem nativeKernel_prefix_left_all
    (a b : Array F)
    (ha : a.size = d)
    (hb : b.size = d) :
    ∀ n : Nat, n ≤ d →
      nativeKernel (basisExpandPrefix a n) b =
        innerProduct (basisExpandPrefix a n) b := by
  intro n hn
  induction n with
  | zero =>
      have hZeroScale : basisExpandPrefix a 0 = vecScale (0 : F) (basisVecNat 0) := by
        simpa [basisExpandPrefix] using (zeroVec_eq_vecScale_zero_basisNat 0)
      have hBasisNatSize : (basisVecNat 0).size = d := by simp [basisVecNat]
      calc
        nativeKernel (basisExpandPrefix a 0) b
            = nativeKernel (vecScale (0 : F) (basisVecNat 0)) b := by
                simp [hZeroScale]
        _ = (0 : F) * nativeKernel (basisVecNat 0) b := by
              simpa using
                nativeKernel_vecScale_left_of_size_d
                  (a := basisVecNat 0)
                  (c := b)
                  (s := (0 : F))
                  hBasisNatSize hb
        _ = 0 := by
              have h0mul : (0 : F) * nativeKernel (basisVecNat 0) b = 0 := by
                simpa using (Fin.zero_mul (n := Goldilocks.q) (nativeKernel (basisVecNat 0) b))
              simpa [h0mul]
        _ = (0 : F) * innerProduct (basisVecNat 0) b := by
              have h0mul : (0 : F) * innerProduct (basisVecNat 0) b = 0 := by
                simpa using (Fin.zero_mul (n := Goldilocks.q) (innerProduct (basisVecNat 0) b))
              simpa [h0mul]
        _ = innerProduct (vecScale (0 : F) (basisVecNat 0)) b := by
              symm
              simpa using
                innerProduct_vecScale_left_of_size_d
                  (a := basisVecNat 0)
                  (b := b)
                  (s := (0 : F))
                  hBasisNatSize hb
        _ = innerProduct (basisExpandPrefix a 0) b := by
              simp [hZeroScale]
  | succ n ih =>
      have hnLe : n ≤ d := Nat.le_trans (Nat.le_succ n) hn
      have hnLt : n < d := Nat.lt_of_lt_of_le (Nat.lt_succ_self n) hn
      have hPrefSize : (basisExpandPrefix a n).size = d := basisExpandPrefix_size a n
      have hBasisNatSize : (basisVecNat n).size = d := by simp [basisVecNat]
      have hBasisEq :
          nativeKernel (basisVecNat n) b =
            innerProduct (basisVecNat n) b :=
        nativeKernel_basisNat_left_all n hnLt b hb
      calc
        nativeKernel (basisExpandPrefix a (n + 1)) b
            = nativeKernel
                (vecAdd (basisExpandPrefix a n) (vecScale a[n]! (basisVecNat n)))
                b := by
                  simp [basisExpandPrefix]
        _ = nativeKernel (basisExpandPrefix a n) b +
              nativeKernel (vecScale a[n]! (basisVecNat n)) b := by
              exact nativeKernel_vecAdd_left_of_size_d
                (a := basisExpandPrefix a n)
                (b := vecScale a[n]! (basisVecNat n))
                (c := b)
                hPrefSize
                (by simp [basisVecNat])
                hb
        _ = innerProduct (basisExpandPrefix a n) b +
              nativeKernel (vecScale a[n]! (basisVecNat n)) b := by
              rw [ih hnLe]
        _ = innerProduct (basisExpandPrefix a n) b +
              (a[n]!) * nativeKernel (basisVecNat n) b := by
              simp [nativeKernel_vecScale_left_of_size_d, hBasisNatSize, hb]
        _ = innerProduct (basisExpandPrefix a n) b +
              (a[n]!) * innerProduct (basisVecNat n) b := by
              simp [hBasisEq]
        _ = innerProduct (basisExpandPrefix a n) b +
              innerProduct (vecScale a[n]! (basisVecNat n)) b := by
              have hScale :
                  innerProduct (vecScale a[n]! (basisVecNat n)) b =
                    (a[n]!) * innerProduct (basisVecNat n) b := by
                exact innerProduct_vecScale_left_of_size_d
                  (a := basisVecNat n)
                  (b := b)
                  (s := a[n]!)
                  hBasisNatSize hb
              rw [hScale]
        _ = innerProduct
              (vecAdd (basisExpandPrefix a n) (vecScale a[n]! (basisVecNat n)))
              b := by
              symm
              exact innerProduct_vecAdd_left_of_size_d
                (a := basisExpandPrefix a n)
                (b := vecScale a[n]! (basisVecNat n))
                (c := b)
                hPrefSize
                (by simp [basisVecNat])
                hb
        _ = innerProduct (basisExpandPrefix a (n + 1)) b := by
              simp [basisExpandPrefix_succ_eq]

/-- Canonical native Theorem-3 closure theorem. -/
theorem thm3CoreAssumption_native :
    thm3CoreAssumption nativeBarMatrix := by
  intro a b ha hb
  have hPrefix :
      nativeKernel (basisExpandPrefix a d) b =
        innerProduct (basisExpandPrefix a d) b :=
    nativeKernel_prefix_left_all a b ha hb d (Nat.le_refl d)
  have hExpand : basisExpand a = a := basisExpand_eq a ha
  calc
    ct (mulRqPhi (superneoBarBlock nativeBarMatrix a) b)
        = nativeKernel a b := by rfl
    _ = nativeKernel (basisExpand a) b := by
          rw [hExpand]
    _ = nativeKernel (basisExpandPrefix a d) b := by
          simp [basisExpand_eq_prefix]
    _ = innerProduct (basisExpandPrefix a d) b := hPrefix
    _ = innerProduct (basisExpand a) b := by
          simp [basisExpand_eq_prefix]
    _ = innerProduct a b := by
          rw [hExpand]

/-! ### Generic bar closure via finite basis kernel -/

/--
Finite basis-kernel criterion for Theorem-3 on a fixed bar matrix.

If this holds on all basis pairs, bilinearity lifts it to all d-sized vectors.
-/
def thm3BasisKernelAssumption (bar : Array (Array F)) : Prop :=
  ∀ i j : Fin d,
    ct (mulRqPhi (superneoBarBlock bar (basisVec i)) (basisVec j)) =
      (if i = j then (1 : F) else 0)

theorem thm3BasisKernelAssumption_native :
    thm3BasisKernelAssumption nativeBarMatrix := by
  intro i j
  exact thm3Core_native_on_basis i j

private theorem dotBySize_vecAdd_right_of_size_d_anyRow
    (row x y : Array F)
    (hx : x.size = d)
    (hy : y.size = d) :
    dotBySize row (vecAdd x y) = dotBySize row x + dotBySize row y := by
  by_cases hRow : row.size = d
  · exact dotBySize_vecAdd_right_of_size_d row x y hRow hx hy
  · have hxy : x.size = y.size := by simpa [hx, hy]
    have hVec : (vecAdd x y).size = d := by
      simpa [hx] using (vecAdd_size_of_eq hxy)
    have hNeL : row.size ≠ (vecAdd x y).size := by
      simpa [hVec] using hRow
    have hNeX : row.size ≠ x.size := by
      simpa [hx] using hRow
    have hNeY : row.size ≠ y.size := by
      simpa [hy] using hRow
    simp [dotBySize, hNeL, hNeX, hNeY]

private theorem dotBySize_vecScale_right_of_size_d_anyRow
    (row x : Array F)
    (s : F)
    (hx : x.size = d) :
    dotBySize row (vecScale s x) = s * dotBySize row x := by
  by_cases hRow : row.size = d
  · exact dotBySize_vecScale_right_of_size_d row x s hRow hx
  · have hVec : (vecScale s x).size = d := by
      simpa [hx] using (vecScale_size s x)
    have hNeL : row.size ≠ (vecScale s x).size := by
      simpa [hVec] using hRow
    have hNeX : row.size ≠ x.size := by
      simpa [hx] using hRow
    have hs0 : s * (0 : F) = 0 := by
      simpa using (Fin.mul_zero (n := Goldilocks.q) s)
    simpa [dotBySize, hNeL, hNeX, hs0]

private theorem superneoBarBlock_size_of_size_d
    (bar : Array (Array F))
    (a : Array F)
    (ha : a.size = d) :
    (superneoBarBlock bar a).size = d := by
  by_cases hBar : bar.size = d
  · simp [superneoBarBlock, hBar, ha]
  · simp [superneoBarBlock, hBar, ha]

private theorem superneoBarBlock_vecAdd
    (bar : Array (Array F))
    (x y : Array F)
    (hx : x.size = d)
    (hy : y.size = d) :
    superneoBarBlock bar (vecAdd x y) =
      vecAdd (superneoBarBlock bar x) (superneoBarBlock bar y) := by
  have hxy : x.size = y.size := by simpa [hx, hy]
  have hVec : (vecAdd x y).size = d := by
    simpa [hx] using (vecAdd_size_of_eq hxy)
  by_cases hBar : bar.size = d
  · have hOutEq :
      (superneoBarBlock bar x).size =
        (superneoBarBlock bar y).size := by
      simp [superneoBarBlock, hBar, hx, hy]
    have hRSize :
        (vecAdd (superneoBarBlock bar x) (superneoBarBlock bar y)).size = d := by
      calc
        (vecAdd (superneoBarBlock bar x) (superneoBarBlock bar y)).size
            = (superneoBarBlock bar x).size := vecAdd_size_of_eq hOutEq
        _ = d := by simp [superneoBarBlock, hBar, hx]
    have hLSize :
        (superneoBarBlock bar (vecAdd x y)).size = d := by
      simp [superneoBarBlock, hBar, hVec]
    apply Array.ext
    · exact hLSize.trans hRSize.symm
    · intro i hiL hiR
      have hi : i < d := by
        simpa [superneoBarBlock, hBar, hVec] using hiL
      have hDot :
          dotBySize (bar[i]'(by simpa [hBar] using hi)) (vecAdd x y) =
            dotBySize (bar[i]'(by simpa [hBar] using hi)) x +
              dotBySize (bar[i]'(by simpa [hBar] using hi)) y := by
        exact dotBySize_vecAdd_right_of_size_d_anyRow
          (row := bar[i]'(by simpa [hBar] using hi))
          (x := x) (y := y) hx hy
      simpa [superneoBarBlock, hBar, hx, hy, hVec, vecAdd, hxy, hi] using hDot
  · simp [superneoBarBlock, hBar]

private theorem superneoBarBlock_vecScale
    (bar : Array (Array F))
    (x : Array F)
    (s : F)
    (hx : x.size = d) :
    superneoBarBlock bar (vecScale s x) =
      vecScale s (superneoBarBlock bar x) := by
  have hVec : (vecScale s x).size = d := by
    simpa [hx] using (vecScale_size s x)
  by_cases hBar : bar.size = d
  · apply Array.ext
    · simp [superneoBarBlock, hBar, hx, hVec, vecScale]
    · intro i hiL hiR
      have hi : i < d := by
        simpa [superneoBarBlock, hBar, hx, hVec, vecScale] using hiL
      have hDot :
          dotBySize (bar[i]'(by simpa [hBar] using hi)) (vecScale s x) =
            s * dotBySize (bar[i]'(by simpa [hBar] using hi)) x := by
        exact dotBySize_vecScale_right_of_size_d_anyRow
          (row := bar[i]'(by simpa [hBar] using hi))
          (x := x) (s := s) hx
      simpa [superneoBarBlock, hBar, hx, hVec, vecScale, hi] using hDot
  · simp [superneoBarBlock, hBar]

/-- Generic Theorem-3 kernel for a fixed bar matrix. -/
private def kernel (bar : Array (Array F)) (a b : Array F) : F :=
  ct (mulRqPhi (superneoBarBlock bar a) b)

private theorem kernel_vecAdd_right_of_size_d
    (bar : Array (Array F))
    (a b c : Array F)
    (ha : a.size = d)
    (hb : b.size = d)
    (hc : c.size = d) :
    kernel bar a (vecAdd b c) =
      kernel bar a b + kernel bar a c := by
  simpa [kernel] using
    ct_mulRqPhi_vecAdd_right_of_size_d
      (a := superneoBarBlock bar a)
      (x := b)
      (y := c)
      hb hc

private theorem kernel_vecScale_right_of_size_d
    (bar : Array (Array F))
    (a x : Array F)
    (s : F)
    (ha : a.size = d)
    (hx : x.size = d) :
    kernel bar a (vecScale s x) =
      s * kernel bar a x := by
  simpa [kernel] using
    ct_mulRqPhi_vecScale_right_of_size_d
      (s := s)
      (a := superneoBarBlock bar a)
      (x := x)
      hx

private theorem kernel_vecAdd_left_of_size_d
    (bar : Array (Array F))
    (a b c : Array F)
    (ha : a.size = d)
    (hb : b.size = d)
    (hc : c.size = d) :
    kernel bar (vecAdd a b) c =
      kernel bar a c + kernel bar b c := by
  have hBarA : (superneoBarBlock bar a).size = d := by
    exact superneoBarBlock_size_of_size_d (bar := bar) (a := a) ha
  have hBarB : (superneoBarBlock bar b).size = d := by
    exact superneoBarBlock_size_of_size_d (bar := bar) (a := b) hb
  have hBarAdd :
      superneoBarBlock bar (vecAdd a b) =
        vecAdd (superneoBarBlock bar a) (superneoBarBlock bar b) := by
    exact superneoBarBlock_vecAdd bar a b ha hb
  calc
    kernel bar (vecAdd a b) c
        = ct (mulRqPhi
            (vecAdd (superneoBarBlock bar a) (superneoBarBlock bar b))
            c) := by
              simp [kernel, hBarAdd]
    _ = ct (mulRqPhi (superneoBarBlock bar a) c) +
          ct (mulRqPhi (superneoBarBlock bar b) c) := by
            simpa using
              ct_mulRqPhi_vecAdd_left_of_size_d
                (x := superneoBarBlock bar a)
                (y := superneoBarBlock bar b)
                (b := c)
                hBarA hBarB
    _ = kernel bar a c + kernel bar b c := by
          simp [kernel]

private theorem kernel_vecScale_left_of_size_d
    (bar : Array (Array F))
    (a c : Array F)
    (s : F)
    (ha : a.size = d)
    (hc : c.size = d) :
    kernel bar (vecScale s a) c =
      s * kernel bar a c := by
  have hBarA : (superneoBarBlock bar a).size = d := by
    exact superneoBarBlock_size_of_size_d (bar := bar) (a := a) ha
  have hBarScale :
      superneoBarBlock bar (vecScale s a) =
        vecScale s (superneoBarBlock bar a) := by
    exact superneoBarBlock_vecScale bar a s ha
  calc
    kernel bar (vecScale s a) c
        = ct (mulRqPhi (vecScale s (superneoBarBlock bar a)) c) := by
              simp [kernel, hBarScale]
    _ = s * ct (mulRqPhi (superneoBarBlock bar a) c) := by
          simpa using
            ct_mulRqPhi_vecScale_left_of_size_d
              (s := s)
              (x := superneoBarBlock bar a)
              (b := c)
              hBarA
    _ = s * kernel bar a c := by
          simp [kernel]

private theorem kernel_basis_basisNat
    (bar : Array (Array F))
    (hBasis : thm3BasisKernelAssumption bar)
    (i : Fin d)
    (n : Nat)
    (hn : n < d) :
    kernel bar (basisVec i) (basisVecNat n) =
      (if i.1 = n then (1 : F) else 0) := by
  have hBasisNat : basisVecNat n = basisVec ⟨n, hn⟩ :=
    basisVecNat_eq_basisVec n hn
  have hEq : (i = (⟨n, hn⟩ : Fin d)) ↔ i.1 = n := by
    constructor
    · intro h
      exact congrArg Fin.val h
    · intro h
      apply Fin.ext
      simpa using h
  calc
    kernel bar (basisVec i) (basisVecNat n)
        = kernel bar (basisVec i) (basisVec ⟨n, hn⟩) := by
            simp [hBasisNat]
    _ = (if i = (⟨n, hn⟩ : Fin d) then (1 : F) else 0) := by
          simpa [kernel, thm3BasisKernelAssumption] using hBasis i ⟨n, hn⟩
    _ = (if i.1 = n then (1 : F) else 0) := by
          simp [hEq]

private theorem kernel_basis_left_prefix
    (bar : Array (Array F))
    (hBasis : thm3BasisKernelAssumption bar)
    (i : Fin d)
    (b : Array F)
    (hb : b.size = d) :
    ∀ n : Nat, n ≤ d →
      kernel bar (basisVec i) (basisExpandPrefix b n) =
        innerProduct (basisVec i) (basisExpandPrefix b n) := by
  intro n hn
  induction n with
  | zero =>
      have hZeroScale : basisExpandPrefix b 0 = vecScale (0 : F) (basisVecNat 0) := by
        simpa [basisExpandPrefix] using (zeroVec_eq_vecScale_zero_basisNat 0)
      have hBasisSize : (basisVec i).size = d := by simp
      have hBasisNatSize : (basisVecNat 0).size = d := by simp [basisVecNat]
      calc
        kernel bar (basisVec i) (basisExpandPrefix b 0)
            = kernel bar (basisVec i) (vecScale (0 : F) (basisVecNat 0)) := by
                simp [hZeroScale]
        _ = (0 : F) * kernel bar (basisVec i) (basisVecNat 0) := by
              simpa using
                kernel_vecScale_right_of_size_d
                  (bar := bar)
                  (a := basisVec i)
                  (x := basisVecNat 0)
                  (s := (0 : F))
                  hBasisSize hBasisNatSize
        _ = 0 := by
              have h0mul : (0 : F) * kernel bar (basisVec i) (basisVecNat 0) = 0 := by
                simpa using (Fin.zero_mul (n := Goldilocks.q) (kernel bar (basisVec i) (basisVecNat 0)))
              simpa [h0mul]
        _ = (0 : F) * innerProduct (basisVec i) (basisVecNat 0) := by
              have h0mul : (0 : F) * innerProduct (basisVec i) (basisVecNat 0) = 0 := by
                simpa using (Fin.zero_mul (n := Goldilocks.q) (innerProduct (basisVec i) (basisVecNat 0)))
              simpa [h0mul]
        _ = innerProduct (basisVec i) (vecScale (0 : F) (basisVecNat 0)) := by
              symm
              simpa using
                innerProduct_vecScale_right_of_size_d
                  (a := basisVec i)
                  (b := basisVecNat 0)
                  (s := (0 : F))
                  hBasisSize hBasisNatSize
        _ = innerProduct (basisVec i) (basisExpandPrefix b 0) := by
              simp [hZeroScale]
  | succ n ih =>
      have hnLe : n ≤ d := Nat.le_trans (Nat.le_succ n) hn
      have hnLt : n < d := Nat.lt_of_lt_of_le (Nat.lt_succ_self n) hn
      have hPrefSize : (basisExpandPrefix b n).size = d := basisExpandPrefix_size b n
      have hBasisSize : (basisVec i).size = d := by simp
      have hBasisNatSize : (basisVecNat n).size = d := by simp [basisVecNat]
      have hBasisEq :
          kernel bar (basisVec i) (basisVecNat n) =
            innerProduct (basisVec i) (basisVecNat n) := by
        calc
          kernel bar (basisVec i) (basisVecNat n)
              = (if i.1 = n then (1 : F) else 0) :=
                kernel_basis_basisNat bar hBasis i n hnLt
          _ = innerProduct (basisVec i) (basisVecNat n) := by
                symm
                exact innerProduct_basis_basisNat i n hnLt
      calc
        kernel bar (basisVec i) (basisExpandPrefix b (n + 1))
            = kernel bar (basisVec i)
                (vecAdd (basisExpandPrefix b n) (vecScale b[n]! (basisVecNat n))) := by
                  simp [basisExpandPrefix]
        _ = kernel bar (basisVec i) (basisExpandPrefix b n) +
              kernel bar (basisVec i) (vecScale b[n]! (basisVecNat n)) := by
              exact kernel_vecAdd_right_of_size_d
                (bar := bar)
                (a := basisVec i)
                (b := basisExpandPrefix b n)
                (c := vecScale b[n]! (basisVecNat n))
                hBasisSize
                hPrefSize
                (by simp [basisVecNat])
        _ = innerProduct (basisVec i) (basisExpandPrefix b n) +
              kernel bar (basisVec i) (vecScale b[n]! (basisVecNat n)) := by
              rw [ih hnLe]
        _ = innerProduct (basisVec i) (basisExpandPrefix b n) +
              (b[n]!) * kernel bar (basisVec i) (basisVecNat n) := by
              simp [kernel_vecScale_right_of_size_d, hBasisSize, hBasisNatSize]
        _ = innerProduct (basisVec i) (basisExpandPrefix b n) +
              (b[n]!) * innerProduct (basisVec i) (basisVecNat n) := by
              simp [hBasisEq]
        _ = innerProduct (basisVec i) (basisExpandPrefix b n) +
              innerProduct (basisVec i) (vecScale b[n]! (basisVecNat n)) := by
              have hScale :
                  innerProduct (basisVec i) (vecScale b[n]! (basisVecNat n)) =
                    (b[n]!) * innerProduct (basisVec i) (basisVecNat n) := by
                exact innerProduct_vecScale_right_of_size_d
                  (a := basisVec i)
                  (b := basisVecNat n)
                  (s := b[n]!)
                  hBasisSize hBasisNatSize
              rw [hScale]
        _ = innerProduct (basisVec i)
              (vecAdd (basisExpandPrefix b n) (vecScale b[n]! (basisVecNat n))) := by
              symm
              exact innerProduct_vecAdd_right_of_size_d
                (a := basisVec i)
                (b := basisExpandPrefix b n)
                (c := vecScale b[n]! (basisVecNat n))
                hBasisSize
                hPrefSize
                (by simp [basisVecNat])
        _ = innerProduct (basisVec i) (basisExpandPrefix b (n + 1)) := by
              simp [basisExpandPrefix_succ_eq]

private theorem kernel_basis_left_all
    (bar : Array (Array F))
    (hBasis : thm3BasisKernelAssumption bar)
    (i : Fin d)
    (b : Array F)
    (hb : b.size = d) :
    kernel bar (basisVec i) b =
      innerProduct (basisVec i) b := by
  have hPrefix :
      kernel bar (basisVec i) (basisExpandPrefix b d) =
        innerProduct (basisVec i) (basisExpandPrefix b d) :=
    kernel_basis_left_prefix bar hBasis i b hb d (Nat.le_refl d)
  have hExpand : basisExpand b = b := basisExpand_eq b hb
  calc
    kernel bar (basisVec i) b
        = kernel bar (basisVec i) (basisExpand b) := by
            rw [hExpand]
    _ = kernel bar (basisVec i) (basisExpandPrefix b d) := by
          simp [basisExpand_eq_prefix]
    _ = innerProduct (basisVec i) (basisExpandPrefix b d) := hPrefix
    _ = innerProduct (basisVec i) (basisExpand b) := by
          simp [basisExpand_eq_prefix]
    _ = innerProduct (basisVec i) b := by
          rw [hExpand]

private theorem kernel_basisNat_left_all
    (bar : Array (Array F))
    (hBasis : thm3BasisKernelAssumption bar)
    (n : Nat)
    (hn : n < d)
    (b : Array F)
    (hb : b.size = d) :
    kernel bar (basisVecNat n) b =
      innerProduct (basisVecNat n) b := by
  have hBasisVec : basisVecNat n = basisVec ⟨n, hn⟩ :=
    basisVecNat_eq_basisVec n hn
  calc
    kernel bar (basisVecNat n) b
        = kernel bar (basisVec ⟨n, hn⟩) b := by
            simp [hBasisVec]
    _ = innerProduct (basisVec ⟨n, hn⟩) b :=
          kernel_basis_left_all bar hBasis ⟨n, hn⟩ b hb
    _ = innerProduct (basisVecNat n) b := by
          simp [hBasisVec]

private theorem kernel_prefix_left_all
    (bar : Array (Array F))
    (hBasis : thm3BasisKernelAssumption bar)
    (a b : Array F)
    (ha : a.size = d)
    (hb : b.size = d) :
    ∀ n : Nat, n ≤ d →
      kernel bar (basisExpandPrefix a n) b =
        innerProduct (basisExpandPrefix a n) b := by
  intro n hn
  induction n with
  | zero =>
      have hZeroScale : basisExpandPrefix a 0 = vecScale (0 : F) (basisVecNat 0) := by
        simpa [basisExpandPrefix] using (zeroVec_eq_vecScale_zero_basisNat 0)
      have hBasisNatSize : (basisVecNat 0).size = d := by simp [basisVecNat]
      calc
        kernel bar (basisExpandPrefix a 0) b
            = kernel bar (vecScale (0 : F) (basisVecNat 0)) b := by
                simp [hZeroScale]
        _ = (0 : F) * kernel bar (basisVecNat 0) b := by
              simpa using
                kernel_vecScale_left_of_size_d
                  (bar := bar)
                  (a := basisVecNat 0)
                  (c := b)
                  (s := (0 : F))
                  hBasisNatSize hb
        _ = 0 := by
              have h0mul : (0 : F) * kernel bar (basisVecNat 0) b = 0 := by
                simpa using (Fin.zero_mul (n := Goldilocks.q) (kernel bar (basisVecNat 0) b))
              simpa [h0mul]
        _ = (0 : F) * innerProduct (basisVecNat 0) b := by
              have h0mul : (0 : F) * innerProduct (basisVecNat 0) b = 0 := by
                simpa using (Fin.zero_mul (n := Goldilocks.q) (innerProduct (basisVecNat 0) b))
              simpa [h0mul]
        _ = innerProduct (vecScale (0 : F) (basisVecNat 0)) b := by
              symm
              simpa using
                innerProduct_vecScale_left_of_size_d
                  (a := basisVecNat 0)
                  (b := b)
                  (s := (0 : F))
                  hBasisNatSize hb
        _ = innerProduct (basisExpandPrefix a 0) b := by
              simp [hZeroScale]
  | succ n ih =>
      have hnLe : n ≤ d := Nat.le_trans (Nat.le_succ n) hn
      have hnLt : n < d := Nat.lt_of_lt_of_le (Nat.lt_succ_self n) hn
      have hPrefSize : (basisExpandPrefix a n).size = d := basisExpandPrefix_size a n
      have hBasisNatSize : (basisVecNat n).size = d := by simp [basisVecNat]
      have hBasisEq :
          kernel bar (basisVecNat n) b =
            innerProduct (basisVecNat n) b :=
        kernel_basisNat_left_all bar hBasis n hnLt b hb
      calc
        kernel bar (basisExpandPrefix a (n + 1)) b
            = kernel bar
                (vecAdd (basisExpandPrefix a n) (vecScale a[n]! (basisVecNat n)))
                b := by
                  simp [basisExpandPrefix]
        _ = kernel bar (basisExpandPrefix a n) b +
              kernel bar (vecScale a[n]! (basisVecNat n)) b := by
              exact kernel_vecAdd_left_of_size_d
                (bar := bar)
                (a := basisExpandPrefix a n)
                (b := vecScale a[n]! (basisVecNat n))
                (c := b)
                hPrefSize
                (by simp [basisVecNat])
                hb
        _ = innerProduct (basisExpandPrefix a n) b +
              kernel bar (vecScale a[n]! (basisVecNat n)) b := by
              rw [ih hnLe]
        _ = innerProduct (basisExpandPrefix a n) b +
              (a[n]!) * kernel bar (basisVecNat n) b := by
              simp [kernel_vecScale_left_of_size_d, hBasisNatSize, hb]
        _ = innerProduct (basisExpandPrefix a n) b +
              (a[n]!) * innerProduct (basisVecNat n) b := by
              simp [hBasisEq]
        _ = innerProduct (basisExpandPrefix a n) b +
              innerProduct (vecScale a[n]! (basisVecNat n)) b := by
              have hScale :
                  innerProduct (vecScale a[n]! (basisVecNat n)) b =
                    (a[n]!) * innerProduct (basisVecNat n) b := by
                exact innerProduct_vecScale_left_of_size_d
                  (a := basisVecNat n)
                  (b := b)
                  (s := a[n]!)
                  hBasisNatSize hb
              rw [hScale]
        _ = innerProduct
              (vecAdd (basisExpandPrefix a n) (vecScale a[n]! (basisVecNat n)))
              b := by
              symm
              exact innerProduct_vecAdd_left_of_size_d
                (a := basisExpandPrefix a n)
                (b := vecScale a[n]! (basisVecNat n))
                (c := b)
                hPrefSize
                (by simp [basisVecNat])
                hb
        _ = innerProduct (basisExpandPrefix a (n + 1)) b := by
              simp [basisExpandPrefix_succ_eq]

/--
Generic closure: basis-kernel identity implies Theorem-3 boundary for all d-sized vectors.
-/
theorem thm3CoreAssumption_of_basisKernelAssumption
    {bar : Array (Array F)}
    (hBasis : thm3BasisKernelAssumption bar) :
    thm3CoreAssumption bar := by
  intro a b ha hb
  have hPrefix :
      kernel bar (basisExpandPrefix a d) b =
        innerProduct (basisExpandPrefix a d) b :=
    kernel_prefix_left_all bar hBasis a b ha hb d (Nat.le_refl d)
  have hExpand : basisExpand a = a := basisExpand_eq a ha
  calc
    ct (mulRqPhi (superneoBarBlock bar a) b)
        = kernel bar a b := by rfl
    _ = kernel bar (basisExpand a) b := by
          rw [hExpand]
    _ = kernel bar (basisExpandPrefix a d) b := by
          simp [basisExpand_eq_prefix]
    _ = innerProduct (basisExpandPrefix a d) b := hPrefix
    _ = innerProduct (basisExpand a) b := by
          simp [basisExpand_eq_prefix]
    _ = innerProduct a b := by
          rw [hExpand]

/-- Any Theorem-3 boundary instance implies its finite basis-kernel form. -/
theorem thm3BasisKernelAssumption_of_thm3CoreAssumption
    {bar : Array (Array F)}
    (hThm3 : thm3CoreAssumption bar) :
    thm3BasisKernelAssumption bar := by
  intro i j
  have hi : (basisVec i).size = d := by simp [basisVec]
  have hj : (basisVec j).size = d := by simp [basisVec]
  calc
    ct (mulRqPhi (superneoBarBlock bar (basisVec i)) (basisVec j))
        = innerProduct (basisVec i) (basisVec j) := hThm3 (basisVec i) (basisVec j) hi hj
    _ = (if i = j then (1 : F) else 0) := innerProduct_basis_basis i j

/-- Theorem-3 boundary is equivalent to its finite basis-kernel criterion. -/
theorem thm3CoreAssumption_iff_basisKernelAssumption
    {bar : Array (Array F)} :
    thm3CoreAssumption bar ↔ thm3BasisKernelAssumption bar := by
  constructor
  · intro h
    exact thm3BasisKernelAssumption_of_thm3CoreAssumption h
  · intro h
    exact thm3CoreAssumption_of_basisKernelAssumption h

/--
Executable finite checker for the basis-kernel criterion.

Because `d` is concrete and finite, this is a finite boolean witness for the
generic Theorem-3 boundary over a fixed `bar`.
-/
noncomputable def thm3BasisKernelCheck (bar : Array (Array F)) : Bool := by
  classical
  exact decide (thm3BasisKernelAssumption bar)

theorem thm3BasisKernelCheck_eq_true_iff
    {bar : Array (Array F)} :
    thm3BasisKernelCheck bar = true ↔ thm3BasisKernelAssumption bar := by
  classical
  simp [thm3BasisKernelCheck]

theorem thm3BasisKernelAssumption_of_check
    {bar : Array (Array F)}
    (hCheck : thm3BasisKernelCheck bar = true) :
    thm3BasisKernelAssumption bar := by
  exact (thm3BasisKernelCheck_eq_true_iff.mp hCheck)

theorem thm3CoreAssumption_of_basisKernelCheck
    {bar : Array (Array F)}
    (hCheck : thm3BasisKernelCheck bar = true) :
    thm3CoreAssumption bar := by
  exact thm3CoreAssumption_of_basisKernelAssumption
    (thm3BasisKernelAssumption_of_check hCheck)

theorem thm3CoreAssumption_iff_basisKernelCheck
    {bar : Array (Array F)} :
    thm3CoreAssumption bar ↔ thm3BasisKernelCheck bar = true := by
  calc
    thm3CoreAssumption bar ↔ thm3BasisKernelAssumption bar :=
      thm3CoreAssumption_iff_basisKernelAssumption
    _ ↔ thm3BasisKernelCheck bar = true := by
      constructor
      · intro h
        exact (thm3BasisKernelCheck_eq_true_iff.mpr h)
      · intro h
        exact (thm3BasisKernelCheck_eq_true_iff.mp h)

theorem thm3BasisKernelCheck_native :
    thm3BasisKernelCheck nativeBarMatrix = true := by
  exact (thm3BasisKernelCheck_eq_true_iff.mpr thm3BasisKernelAssumption_native)

/-- Alternate native closure route through the finite basis-kernel criterion. -/
theorem thm3CoreAssumption_native_of_basis :
    thm3CoreAssumption nativeBarMatrix := by
  exact thm3CoreAssumption_of_basisKernelAssumption thm3BasisKernelAssumption_native

/--
Compatibility alias for Theorem-3 boundary references.

The native in-repo constructive closure is provided by
`thm3CoreAssumption_native`. This alias exists so downstream surfaces that
still mention `[36,64]` naming can remain source-compatible.
-/
abbrev thm3CoreAssumption_ref36_64 (bar : Array (Array F)) : Prop :=
  thm3CoreAssumption bar

/-- Bridge from the ref-backed wrapper into the canonical Theorem-3 boundary. -/
theorem thm3CoreAssumption_of_ref36_64
    {bar : Array (Array F)}
    (hRef : thm3CoreAssumption_ref36_64 bar) :
    thm3CoreAssumption bar := hRef

/-- Bridge from canonical Theorem-3 boundary into the ref-backed wrapper. -/
theorem thm3CoreAssumption_ref36_64_of_assumption
    {bar : Array (Array F)}
    (hThm3 : thm3CoreAssumption bar) :
    thm3CoreAssumption_ref36_64 bar := hThm3

/-! ### P10 Compatibility Surface -/

/-- Dimension-shape predicate for vectors used by P10 compatibility wrappers. -/
def IsDVec (a : Array F) : Prop :=
  a.size = d

/-- Shape predicate for bar matrices used by P10 compatibility wrappers. -/
def IsDBarMatrix (bar : Array (Array F)) : Prop :=
  bar.size = d ∧ bar.all (fun row => row.size == d) = true

/-- Compact P10 proposition surface on concrete vectors. -/
def p10CoreProp (bar : Array (Array F)) (a b : Array F) : Prop :=
  a.size = d ∧ b.size = d ∧
    ct (mulRqPhi (superneoBarBlock bar a) b) = innerProduct a b

instance p10CoreProp_decidable (bar : Array (Array F)) (a b : Array F) :
    Decidable (p10CoreProp bar a b) := by
  unfold p10CoreProp
  infer_instance

/-- Executable P10 check surface on concrete vectors. -/
def p10CoreCheck (bar : Array (Array F)) (a b : Array F) : Bool :=
  decide (p10CoreProp bar a b)

theorem p10CoreCheck_sound
  {bar : Array (Array F)} {a b : Array F}
  (hOk : p10CoreCheck bar a b = true) :
  p10CoreProp bar a b := by
  unfold p10CoreCheck at hOk
  exact decide_eq_true_eq.mp hOk

theorem p10CoreCheck_complete
  {bar : Array (Array F)} {a b : Array F}
  (hProp : p10CoreProp bar a b) :
  p10CoreCheck bar a b = true := by
  unfold p10CoreCheck
  exact decide_eq_true hProp

/-- Build P10 proposition from shape preconditions and passing P10 check. -/
theorem p10Core_of_preconditions
  {bar : Array (Array F)} {a b : Array F}
  (_hBar : IsDBarMatrix bar)
  (_hA : IsDVec a)
  (_hB : IsDVec b)
  (hCheck : p10CoreCheck bar a b = true) :
  p10CoreProp bar a b := by
  exact p10CoreCheck_sound hCheck

/-- Build P10 proposition directly from Theorem-3 assumption plus shape preconditions. -/
theorem p10Core_of_preconditions_props
  {bar : Array (Array F)} {a b : Array F}
  (_hBar : IsDBarMatrix bar)
  (hA : IsDVec a)
  (hB : IsDVec b)
  (hThm3 : thm3CoreAssumption bar) :
  p10CoreProp bar a b := by
  exact ⟨hA, hB, hThm3 a b hA hB⟩

/-- Build P10 proposition from Theorem-3 assumption and vector shape assumptions. -/
theorem p10Core_of_assumption
  {bar : Array (Array F)} {a b : Array F}
  (hThm3 : thm3CoreAssumption bar)
  (hA : IsDVec a)
  (hB : IsDVec b) :
  p10CoreProp bar a b := by
  exact ⟨hA, hB, hThm3 a b hA hB⟩

/-- Native P10 constructor without explicitly threading `thm3CoreAssumption`. -/
theorem p10Core_native
  {a b : Array F}
  (hA : IsDVec a)
  (hB : IsDVec b) :
  p10CoreProp nativeBarMatrix a b := by
  exact p10Core_of_assumption thm3CoreAssumption_native hA hB


end SuperNeo
