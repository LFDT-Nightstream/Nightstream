import SuperNeo.Norm

/-! Balanced digit decomposition and round-trip properties (P6). -/


namespace SuperNeo

open F

/-- Centered integer representative in [-(q-1)/2, (q-1)/2]. -/
def centeredInt (a : F) : Int :=
  if a.val <= halfQ then
    Int.ofNat a.val
  else
    Int.ofNat a.val - Int.ofNat q

private def balancedResidue (a : Int) (b : Nat) : Int :=
  let bi := Int.ofNat b
  let half := Int.ofNat (b / 2)
  let q0 :=
    if a >= 0 then
      a / bi
    else
      - ((-a) / bi)
  let r0 := a - q0 * bi
  let r1 := if r0 > half then r0 - bi else r0
  if r1 < -half then r1 + bi else r1

/-- Balanced base-b split of one field element into k digits. -/
def splitBalancedScalar (a : F) (b k : Nat) : Array F :=
  if b < 2 then
    Array.replicate k (0 : F)
  else
    Id.run do
      let mut out : Array F := #[]
      let mut cur := centeredInt a
      let bi := Int.ofNat b
      for _ in [0:k] do
        let r := balancedResidue cur b
        out := out.push (F.ofInt r)
        cur :=
          if cur - r >= 0 then
            (cur - r) / bi
          else
            - ((-(cur - r)) / bi)
      return out

/-- Balanced base-b split of a vector into k digit-vectors. -/
def splitBalancedVec (z : Array F) (b k : Nat) : Array (Array F) :=
  if b < 2 then
    Array.replicate k (Array.replicate z.size (0 : F))
  else
    Id.run do
      let mut digits := Array.replicate k (Array.replicate z.size (0 : F))
      for j in [0:z.size] do
        let ds := splitBalancedScalar z[j]! b k
        for i in [0:k] do
          let row := digits[i]!
          digits := digits.set! i (row.set! j ds[i]!)
      return digits

/-- Internal invariant: all rows in `digits` have width `m`. -/
private def rowsHaveSize (digits : Array (Array F)) (m : Nat) : Prop :=
  ∀ i (hi : i < digits.size), (digits[i]'hi).size = m

private theorem rowsHaveSize_replicate (k m : Nat) :
  rowsHaveSize (Array.replicate k (Array.replicate m (0 : F))) m := by
  intro i hi
  simp

private theorem rowsHaveSize_setIfInBounds
  {digits : Array (Array F)} {m i j : Nat} {x : F}
  (hRows : rowsHaveSize digits m) :
  rowsHaveSize (digits.setIfInBounds i (digits[i]!.setIfInBounds j x)) m := by
  intro t ht
  by_cases hti : t = i
  · have hiOld : i < digits.size := by
      simpa [hti, Array.size_setIfInBounds] using ht
    have hiNew : i < (digits.setIfInBounds i (digits[i]!.setIfInBounds j x)).size := by
      simpa [Array.size_setIfInBounds] using hiOld
    have hSelf :
        (digits.setIfInBounds i (digits[i]!.setIfInBounds j x))[i]'hiNew =
          digits[i]!.setIfInBounds j x := by
      simpa using
        (Array.getElem_setIfInBounds_self
          (xs := digits) (i := i) (a := digits[i]!.setIfInBounds j x) hiNew)
    have hEq :
        (digits.setIfInBounds i (digits[i]!.setIfInBounds j x))[t]'ht =
          digits[i]!.setIfInBounds j x := by
      simpa [hti] using hSelf
    have hSizeOld : (digits[i]!).size = m := by
      simpa [hiOld] using hRows i hiOld
    calc
      ((digits.setIfInBounds i (digits[i]!.setIfInBounds j x))[t]'ht).size
          = (digits[i]!.setIfInBounds j x).size := by
            simpa using congrArg Array.size hEq
      _ = (digits[i]!).size := by simp [Array.size_setIfInBounds]
      _ = m := hSizeOld
  · have htOld : t < digits.size := by
      simpa [Array.size_setIfInBounds] using ht
    have hit : i ≠ t := Ne.symm hti
    have hEq :
        (digits.setIfInBounds i (digits[i]!.setIfInBounds j x))[t]'ht =
          digits[t]'htOld := by
      simpa using
        (Array.getElem_setIfInBounds_ne
          (xs := digits) (i := i) (a := digits[i]!.setIfInBounds j x) (j := t) htOld hit)
    calc
      ((digits.setIfInBounds i (digits[i]!.setIfInBounds j x))[t]'ht).size
          = (digits[t]'htOld).size := by
            simpa using congrArg Array.size hEq
      _ = m := hRows t htOld

/-- Structural size invariant: vector split always produces exactly `k` digit rows. -/
theorem splitBalancedVec_size (z : Array F) (b k : Nat) :
  (splitBalancedVec z b k).size = k := by
  unfold splitBalancedVec
  by_cases hb : b < 2
  · simp [hb]
  · simp [hb]
    have hInner :
        ∀ (l : List Nat) (digits : Array (Array F)) (j : Nat),
          (List.foldl
              (fun b2 i => b2.setIfInBounds i (b2[i]!.setIfInBounds j (splitBalancedScalar z[j]! b k)[i]!))
              digits l).size =
            digits.size := by
      intro l
      induction l with
      | nil =>
          intro digits j
          simp
      | cons i is ih =>
          intro digits j
          simp [List.foldl_cons, ih, Array.size_setIfInBounds]
    have hOuter :
        ∀ (l : List Nat) (digits : Array (Array F)),
          (List.foldl
              (fun b1 j =>
                List.foldl
                  (fun b2 i => b2.setIfInBounds i (b2[i]!.setIfInBounds j (splitBalancedScalar z[j]! b k)[i]!))
                  b1
                  (List.range' 0 k))
              digits l).size =
            digits.size := by
      intro l
      induction l with
      | nil =>
          intro digits
          simp
      | cons j js ih =>
          intro digits
          simp [List.foldl_cons, hInner (List.range' 0 k), ih]
    simpa using hOuter (List.range' 0 z.size) (Array.replicate k (Array.replicate z.size 0))

/-- Structural row-width invariant: each split digit row has width `z.size`. -/
theorem splitBalancedVec_row_size
  {z : Array F} {b k i : Nat}
  (hi : i < (splitBalancedVec z b k).size) :
  ((splitBalancedVec z b k)[i]'hi).size = z.size := by
  unfold splitBalancedVec at hi ⊢
  by_cases hb : b < 2
  · simp [hb] at hi ⊢
  · simp [hb] at hi ⊢
    have hInner :
        ∀ (l : List Nat) (digits : Array (Array F)) (j : Nat),
          rowsHaveSize digits z.size →
          rowsHaveSize
            (List.foldl
              (fun b2 i =>
                b2.setIfInBounds i (b2[i]!.setIfInBounds j (splitBalancedScalar z[j]! b k)[i]!))
              digits
              l)
            z.size := by
      intro l
      induction l with
      | nil =>
          intro digits j hRows
          simpa using hRows
      | cons i is ih =>
          intro digits j hRows
          have hStep :
              rowsHaveSize
                (digits.setIfInBounds i
                  (digits[i]!.setIfInBounds j (splitBalancedScalar z[j]! b k)[i]!))
                z.size :=
            rowsHaveSize_setIfInBounds hRows
          simpa [List.foldl_cons] using ih _ _ hStep
    have hOuter :
        ∀ (l : List Nat) (digits : Array (Array F)),
          rowsHaveSize digits z.size →
          rowsHaveSize
            (List.foldl
              (fun b1 j =>
                List.foldl
                  (fun b2 i =>
                    b2.setIfInBounds i (b2[i]!.setIfInBounds j (splitBalancedScalar z[j]! b k)[i]!))
                  b1
                  (List.range' 0 k))
              digits
              l)
            z.size := by
      intro l
      induction l with
      | nil =>
          intro digits hRows
          simpa using hRows
      | cons j js ih =>
          intro digits hRows
          have hStep :
              rowsHaveSize
                (List.foldl
                  (fun b2 i =>
                    b2.setIfInBounds i (b2[i]!.setIfInBounds j (splitBalancedScalar z[j]! b k)[i]!))
                  digits
                  (List.range' 0 k))
                z.size :=
            hInner (List.range' 0 k) digits j hRows
          simpa [List.foldl_cons] using ih _ hStep
    have hRows :
        rowsHaveSize
          (List.foldl
            (fun b1 j =>
              List.foldl
                (fun b2 i =>
                  b2.setIfInBounds i (b2[i]!.setIfInBounds j (splitBalancedScalar z[j]! b k)[i]!))
                b1
                (List.range' 0 k))
            (Array.replicate k (Array.replicate z.size 0))
            (List.range' 0 z.size))
          z.size :=
      hOuter (List.range' 0 z.size) (Array.replicate k (Array.replicate z.size 0))
        (rowsHaveSize_replicate k z.size)
    exact hRows i hi

/-- Recompose z = Σ b^i z_i from split digits. -/
def recomposeSplitDigits (digits : Array (Array F)) (b : Nat) : Array F :=
  if digits.isEmpty then
    #[]
  else
    let m := (digits[0]!).size
    Id.run do
      let mut out := Array.replicate m (0 : F)
      let mut scale : F := 1
      let bF := F.ofNat b
      for i in [0:digits.size] do
        let row := digits[i]!
        for j in [0:m] do
          out := out.set! j (out[j]! + scale * row[j]!)
        scale := scale * bF
      return out

def digitsWithinBase (digits : Array (Array F)) (b : Nat) : Bool :=
  digits.all (fun row => row.all (fun x => normInfF x < b))

def digitsWithinBaseProp (digits : Array (Array F)) (b : Nat) : Prop :=
  ∀ i (hi : i < digits.size) j (hj : j < (digits[i]'hi).size),
    normInfF ((digits[i]'hi)[j]'hj) < b

theorem digitsWithinBase_sound
  {digits : Array (Array F)} {b : Nat}
  (hOk : digitsWithinBase digits b = true) :
  digitsWithinBaseProp digits b := by
  intro i hi j hj
  have hRow :
      (digits[i]'hi).all (fun x => normInfF x < b) = true :=
    (Array.all_eq_true.mp hOk) i hi
  exact decide_eq_true_eq.mp ((Array.all_eq_true.mp hRow) j hj)

theorem digitsWithinBase_complete
  {digits : Array (Array F)} {b : Nat}
  (hProp : digitsWithinBaseProp digits b) :
  digitsWithinBase digits b = true := by
  apply (Array.all_eq_true).2
  intro i hi
  apply (Array.all_eq_true).2
  intro j hj
  exact decide_eq_true (hProp i hi j hj)

theorem digitsWithinBase_iff_prop
  {digits : Array (Array F)} {b : Nat} :
  digitsWithinBase digits b = true ↔ digitsWithinBaseProp digits b := by
  constructor
  · exact digitsWithinBase_sound
  · exact digitsWithinBase_complete

theorem digitsWithinBaseProp_mono
  {digits : Array (Array F)} {b B : Nat}
  (hDigits : digitsWithinBaseProp digits b)
  (hMono : b ≤ B) :
  digitsWithinBaseProp digits B := by
  intro i hi j hj
  exact Nat.lt_of_lt_of_le (hDigits i hi j hj) hMono

theorem digitsWithinBase_mono
  {digits : Array (Array F)} {b B : Nat}
  (hDigits : digitsWithinBase digits b = true)
  (hMono : b ≤ B) :
  digitsWithinBase digits B = true := by
  exact digitsWithinBase_complete
    (digitsWithinBaseProp_mono (digitsWithinBase_sound hDigits) hMono)

def splitRoundTrip (z : Array F) (b k : Nat) : Bool :=
  if b < 2 then
    false
  else
    let digits := splitBalancedVec z b k
    decide (recomposeSplitDigits digits b = z) && digitsWithinBase digits b

theorem splitRoundTrip_digits_size
  {z : Array F} {b k : Nat}
  (_hOk : splitRoundTrip z b k = true) :
  (splitBalancedVec z b k).size = k := by
  exact splitBalancedVec_size z b k

theorem splitRoundTrip_sound
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  b ≥ 2 ∧
    let digits := splitBalancedVec z b k
    recomposeSplitDigits digits b = z ∧ digitsWithinBase digits b = true := by
  unfold splitRoundTrip at hOk
  by_cases hb : b < 2
  · simp [hb] at hOk
  · have hbGe : b ≥ 2 := Nat.le_of_not_lt hb
    simp [hb] at hOk
    have hAnd :
      decide (recomposeSplitDigits (splitBalancedVec z b k) b = z) = true ∧
        digitsWithinBase (splitBalancedVec z b k) b = true := by
      simpa [Bool.and_eq_true] using hOk
    refine ⟨hbGe, ?_⟩
    exact ⟨decide_eq_true_eq.mp hAnd.1, hAnd.2⟩

theorem splitRoundTrip_complete
  {z : Array F} {b k : Nat}
  (hProp : b ≥ 2 ∧
    let digits := splitBalancedVec z b k
    recomposeSplitDigits digits b = z ∧ digitsWithinBase digits b = true) :
  splitRoundTrip z b k = true := by
  rcases hProp with ⟨hbGe, hRest⟩
  have hbNotLt : ¬ b < 2 := Nat.not_lt.mpr hbGe
  rcases hRest with ⟨hRec, hDigits⟩
  unfold splitRoundTrip
  simp [hbNotLt, hRec, hDigits]

theorem splitRoundTrip_sound_prop
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  b ≥ 2 ∧
    let digits := splitBalancedVec z b k
    recomposeSplitDigits digits b = z ∧ digitsWithinBaseProp digits b := by
  rcases splitRoundTrip_sound hOk with ⟨hbGe, hRest⟩
  rcases hRest with ⟨hRec, hDigits⟩
  exact ⟨hbGe, hRec, digitsWithinBase_sound hDigits⟩

theorem splitRoundTrip_complete_prop
  {z : Array F} {b k : Nat}
  (hProp : b ≥ 2 ∧
    let digits := splitBalancedVec z b k
    recomposeSplitDigits digits b = z ∧ digitsWithinBaseProp digits b) :
  splitRoundTrip z b k = true := by
  rcases hProp with ⟨hbGe, hRest⟩
  rcases hRest with ⟨hRec, hDigits⟩
  exact splitRoundTrip_complete ⟨hbGe, hRec, digitsWithinBase_complete hDigits⟩

theorem splitRoundTrip_iff_prop
  {z : Array F} {b k : Nat} :
  splitRoundTrip z b k = true ↔
    (b ≥ 2 ∧
      let digits := splitBalancedVec z b k
      recomposeSplitDigits digits b = z ∧ digitsWithinBaseProp digits b) := by
  constructor
  · exact splitRoundTrip_sound_prop
  · exact splitRoundTrip_complete_prop

theorem splitRoundTrip_base_ge_two
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  b ≥ 2 := by
  exact (splitRoundTrip_sound_prop hOk).1

theorem splitRoundTrip_recompose_eq
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  recomposeSplitDigits (splitBalancedVec z b k) b = z := by
  exact (splitRoundTrip_sound_prop hOk).2.1

theorem splitRoundTrip_recompose_size_eq
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  (recomposeSplitDigits (splitBalancedVec z b k) b).size = z.size := by
  simpa [splitRoundTrip_recompose_eq hOk]

theorem splitRoundTrip_digitsWithinBaseProp
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  digitsWithinBaseProp (splitBalancedVec z b k) b := by
  exact (splitRoundTrip_sound_prop hOk).2.2

theorem splitRoundTrip_digit_bound
  {z : Array F} {b k i j : Nat}
  (hOk : splitRoundTrip z b k = true)
  (hi : i < (splitBalancedVec z b k).size)
  (hj : j < ((splitBalancedVec z b k)[i]'hi).size) :
  normInfF (((splitBalancedVec z b k)[i]'hi)[j]'hj) < b := by
  exact splitRoundTrip_digitsWithinBaseProp hOk i hi j hj

theorem splitRoundTrip_digit_row_size
  {z : Array F} {b k i : Nat}
  (_hOk : splitRoundTrip z b k = true)
  (hi : i < (splitBalancedVec z b k).size) :
  ((splitBalancedVec z b k)[i]'hi).size = z.size := by
  exact splitBalancedVec_row_size hi

/-- Tiny sanity check for Definition 3 + split_b style decomposition. -/
def decompSanity : Bool :=
  let z := #[F.ofNat 3, F.ofInt (-2), F.ofNat (q - 1)]
  splitRoundTrip z 2 8

end SuperNeo
