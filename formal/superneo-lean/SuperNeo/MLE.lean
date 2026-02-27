import SuperNeo.EqPoly

/-! Multilinear extension utilities and theorem/check bridges. -/


namespace SuperNeo

open F

/-- χ_r(j) = Π_i (r_i if bit_i(j)=1 else (1-r_i)). -/
def chiWeight (r : Array F) (j : Nat) : F :=
  Id.run do
    let mut w : F := 1
    for i in [0:r.size] do
      let bit := (j / (2 ^ i)) % 2
      let ri := r[i]!
      let term := if bit = 1 then ri else (1 : F) - ri
      w := w * term
    return w

def rHat (r : Array F) (n : Nat) : Array F :=
  Id.run do
    let mut out := Array.replicate n (0 : F)
    for j in [0:n] do
      out := out.set! j (chiWeight r j)
    return out

theorem rHat_size (r : Array F) (n : Nat) : (rHat r n).size = n := by
  unfold rHat
  have hFold :
      ∀ (l : List Nat) (acc : Array F),
        (List.foldl (fun b a => b.setIfInBounds a (chiWeight r a)) acc l).size = acc.size := by
    intro l
    induction l with
    | nil =>
        intro acc
        simp
    | cons x xs ih =>
        intro acc
        simp [ih, Array.size_setIfInBounds]
  simpa using hFold (List.range' 0 n) (Array.replicate n (0 : F))

def dotVec (a b : Array F) : F :=
  if a.size != b.size then
    0
  else
    Id.run do
      let mut acc : F := 0
      for i in [0:a.size] do
        acc := acc + a[i]! * b[i]!
      return acc

/-- MLE via inner product identity: v~(r) = <v, r_hat>. -/
def mleByInnerProduct (v r : Array F) : F :=
  dotVec v (rHat r v.size)

private def foldLayer (vals : Array F) (ri : F) : Array F :=
  Id.run do
    let pairs := vals.size / 2
    let mut out := Array.replicate pairs (0 : F)
    for i in [0:pairs] do
      let a := vals[2 * i]!
      let b := vals[2 * i + 1]!
      out := out.set! i (a * ((1 : F) - ri) + b * ri)
    return out

/-- MLE via iterative multilinear folding across coordinates. -/
def mleByFolding (v r : Array F) : F :=
  Id.run do
    let mut cur := v
    for i in [0:r.size] do
      cur := foldLayer cur r[i]!
    if cur.isEmpty then
      0
    else
      cur[0]!

def mleIdentity (v r : Array F) : Bool :=
  if v.size != 2 ^ r.size then
    false
  else
    decide (mleByInnerProduct v r = mleByFolding v r)

/-- Proposition-level counterpart of `mleIdentity`. -/
def mleIdentityProp (v r : Array F) : Prop :=
  (v.size != 2 ^ r.size) = false ∧
    mleByInnerProduct v r = mleByFolding v r

def mleIdentityPropEq (v r : Array F) : Prop :=
  v.size = 2 ^ r.size ∧
    mleByInnerProduct v r = mleByFolding v r

theorem mleIdentity_sound
  {v r : Array F}
  (hOk : mleIdentity v r = true) :
  mleIdentityProp v r := by
  unfold mleIdentity at hOk
  cases hSize : (v.size != 2 ^ r.size) with
  | true =>
      simp [hSize] at hOk
  | false =>
      simp [hSize] at hOk
      exact ⟨hSize, hOk⟩

theorem mleIdentity_complete
  {v r : Array F}
  (hProp : mleIdentityProp v r) :
  mleIdentity v r = true := by
  rcases hProp with ⟨hSize, hEq⟩
  unfold mleIdentity
  simp [hSize, decide_eq_true hEq]

theorem mleIdentityProp_size_eq
  {v r : Array F}
  (hProp : mleIdentityProp v r) :
  v.size = 2 ^ r.size := by
  have hSizeFalse : (v.size != 2 ^ r.size) = false := hProp.1
  by_cases hEq : v.size = 2 ^ r.size
  · exact hEq
  · have hNeTrue : (v.size != 2 ^ r.size) = true := by simp [hEq]
    rw [hNeTrue] at hSizeFalse
    cases hSizeFalse

theorem mleIdentityProp_eval_eq
  {v r : Array F}
  (hProp : mleIdentityProp v r) :
  mleByInnerProduct v r = mleByFolding v r := by
  exact hProp.2

theorem mleIdentity_iff_prop
  {v r : Array F} :
  mleIdentity v r = true ↔ mleIdentityProp v r := by
  constructor
  · exact mleIdentity_sound
  · exact mleIdentity_complete

theorem mleIdentityPropEq_iff_prop
  {v r : Array F} :
  mleIdentityPropEq v r ↔ mleIdentityProp v r := by
  constructor
  · intro h
    refine ⟨?_, h.2⟩
    simp [h.1]
  · intro h
    refine ⟨?_, h.2⟩
    have hSizeFalse : (v.size != 2 ^ r.size) = false := h.1
    by_cases hSize : v.size = 2 ^ r.size
    · exact hSize
    · have hNeTrue : (v.size != 2 ^ r.size) = true := by simp [hSize]
      rw [hNeTrue] at hSizeFalse
      cases hSizeFalse

theorem mleIdentity_sound_eq
  {v r : Array F}
  (hOk : mleIdentity v r = true) :
  mleIdentityPropEq v r := by
  exact (mleIdentityPropEq_iff_prop).2 (mleIdentity_sound hOk)

theorem mleIdentity_complete_eq
  {v r : Array F}
  (hProp : mleIdentityPropEq v r) :
  mleIdentity v r = true := by
  exact mleIdentity_complete ((mleIdentityPropEq_iff_prop).1 hProp)

theorem mleIdentityPropEq_size_eq
  {v r : Array F}
  (hProp : mleIdentityPropEq v r) :
  v.size = 2 ^ r.size := by
  exact hProp.1

theorem mleIdentityPropEq_eval_eq
  {v r : Array F}
  (hProp : mleIdentityPropEq v r) :
  mleByInnerProduct v r = mleByFolding v r := by
  exact hProp.2

theorem mleIdentity_size_eq
  {v r : Array F}
  (hOk : mleIdentity v r = true) :
  v.size = 2 ^ r.size := by
  exact (mleIdentity_sound_eq hOk).1

theorem mleIdentity_eval_eq
  {v r : Array F}
  (hOk : mleIdentity v r = true) :
  mleByInnerProduct v r = mleByFolding v r := by
  exact (mleIdentity_sound hOk).2

def mleSanity : Bool :=
  let v := #[3, 5, 7, 9]
  let r := #[F.ofNat 2, F.ofNat 1]
  mleIdentity v r

end SuperNeo
