import SuperNeo.CoeffMaps

/-! Norm bounds and collapse assumptions for field and ring operations. -/


namespace SuperNeo

/-- Half-modulus threshold for centered representatives in F_q. -/
def halfQ : Nat := (q - 1) / 2

theorem halfQ_lt_q : halfQ < q := by
  unfold halfQ q
  decide

theorem q_sub_halfQ_succ_eq_halfQ : q - (halfQ + 1) = halfQ := by
  unfold q halfQ
  decide

/-- Centered absolute value for a canonical residue x mod q. -/
def centeredAbsNat (x : Nat) : Nat :=
  let xr := x % q
  if xr <= halfQ then
    xr
  else
    q - xr

theorem halfQ_add_halfQ_lt_q : halfQ + halfQ < q := by
  unfold halfQ q
  decide

theorem centeredAbsNat_mod (x : Nat) : centeredAbsNat (x % q) = centeredAbsNat x := by
  unfold centeredAbsNat
  simp

theorem centeredAbsNat_le_mod (x : Nat) : centeredAbsNat x ≤ x % q := by
  unfold centeredAbsNat
  by_cases h : x % q ≤ halfQ
  · simp [h]
  · have hx : halfQ < x % q := Nat.lt_of_not_ge h
    have hx_le : halfQ ≤ x % q := Nat.le_of_lt hx
    have hx1 : halfQ + 1 ≤ x % q := Nat.succ_le_of_lt hx
    have hsub : q - (x % q) ≤ q - (halfQ + 1) := Nat.sub_le_sub_left hx1 q
    have hqr : q - (x % q) ≤ halfQ := by
      simpa [q_sub_halfQ_succ_eq_halfQ] using hsub
    simpa [h] using (Nat.le_trans hqr hx_le)

theorem centeredAbsNat_le_self (x : Nat) : centeredAbsNat x ≤ x := by
  exact Nat.le_trans (centeredAbsNat_le_mod x) (Nat.mod_le _ _)

/--
Centered absolute value is invariant under modular negation.
This is the Nat-level `| -x | = | x |` symmetry used by later norm lemmas.
-/
theorem centeredAbsNat_neg_mod (x : Nat) :
  centeredAbsNat (q - (x % q)) = centeredAbsNat x := by
  let r : Nat := x % q
  have hxr : x % q = r := by
    simp [r]
  rw [hxr]
  have hxmod : centeredAbsNat x = centeredAbsNat r := by
    simpa [hxr] using (centeredAbsNat_mod x).symm
  rw [hxmod]
  have hr_lt : r < q := by
    simpa [r] using Nat.mod_lt x q_pos
  by_cases hr0 : r = 0
  · rw [hr0]
    unfold centeredAbsNat
    simp
  · have hrpos : 0 < r := Nat.pos_of_ne_zero hr0
    have hrle : r ≤ q := Nat.le_of_lt hr_lt
    have hqmr_lt : q - r < q := by
      have hEq : q - r + r = q := Nat.sub_add_cancel hrle
      have hlt : q - r < q - r + r := Nat.lt_add_of_pos_right hrpos
      simpa [hEq] using hlt
    have hmod : (q - r) % q = q - r := Nat.mod_eq_of_lt hqmr_lt
    by_cases hrH : r ≤ halfQ
    · have hnot : ¬ q - r ≤ halfQ := by
        intro hle
        have hEq : q - r + r = q := Nat.sub_add_cancel hrle
        have hSum : q - r + r ≤ halfQ + halfQ := Nat.add_le_add hle hrH
        have hqle : q ≤ halfQ + halfQ := by simpa [hEq] using hSum
        exact Nat.not_lt_of_ge hqle halfQ_add_halfQ_lt_q
      have hgt : halfQ < q - r := Nat.lt_of_not_ge hnot
      unfold centeredAbsNat
      have hrMod : r % q = r := Nat.mod_eq_of_lt hr_lt
      have hnot' : ¬ (q - r) % q ≤ halfQ := by
        simpa [hmod] using (Nat.not_le.mpr hgt)
      have hFalse : ¬ q ≤ halfQ + r := by
        intro hqle
        have h1 : q - r + r ≤ halfQ + r := by
          calc
            q - r + r = q := Nat.sub_add_cancel hrle
            _ ≤ halfQ + r := hqle
        have hqrLe : q - r ≤ halfQ := by
          exact (Nat.add_le_add_iff_right).1 h1
        exact hnot hqrLe
      have hSubSub : q - (q - r) = r := by
        apply (Nat.sub_eq_iff_eq_add (Nat.sub_le q r)).2
        have hEq : q - r + r = q := Nat.sub_add_cancel hrle
        simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using hEq.symm
      simp [hrMod, hmod, hrH, hFalse, hSubSub]
    · have hrgt : halfQ < r := Nat.lt_of_not_ge hrH
      have hr1 : halfQ + 1 ≤ r := Nat.succ_le_of_lt hrgt
      have hsub : q - r ≤ q - (halfQ + 1) := Nat.sub_le_sub_left hr1 q
      have hqmr_le : q - r ≤ halfQ := by
        simpa [q_sub_halfQ_succ_eq_halfQ] using hsub
      unfold centeredAbsNat
      have hrMod : r % q = r := Nat.mod_eq_of_lt hr_lt
      have hnot' : ¬ r % q ≤ halfQ := by simpa [hrMod] using hrH
      simp [hrMod, hmod, hqmr_le]
      intro hle
      exact (hrH hle).elim

/--
Triangle inequality for centered absolute values modulo `q`.
This is the Nat-level core needed for theorem-native additive blocker proofs.
-/
theorem centeredAbsNat_add_le (x y : Nat) :
  centeredAbsNat (x + y) ≤ centeredAbsNat x + centeredAbsNat y := by
  let a : Nat := x % q
  let b : Nat := y % q
  have ha_lt : a < q := by
    simpa [a] using Nat.mod_lt x q_pos
  have hb_lt : b < q := by
    simpa [b] using Nat.mod_lt y q_pos
  have hx : centeredAbsNat x = centeredAbsNat a := by
    simpa [a] using (centeredAbsNat_mod x).symm
  have hy : centeredAbsNat y = centeredAbsNat b := by
    simpa [b] using (centeredAbsNat_mod y).symm
  have hxy : centeredAbsNat (x + y) = centeredAbsNat (a + b) := by
    calc
      centeredAbsNat (x + y) = centeredAbsNat ((x + y) % q) := by
        simpa using (centeredAbsNat_mod (x + y)).symm
      _ = centeredAbsNat (((x % q) + (y % q)) % q) := by
        simp [Nat.add_mod]
      _ = centeredAbsNat ((a + b) % q) := by simp [a, b]
      _ = centeredAbsNat (a + b) := by
        simpa using centeredAbsNat_mod (a + b)

  have hMixed :
      ∀ u v : Nat,
        u < q → v < q →
        u ≤ halfQ → ¬ v ≤ halfQ →
        centeredAbsNat (u + v) ≤ centeredAbsNat u + centeredAbsNat v := by
    intro u v hu_lt hv_lt huH hvH
    let dv : Nat := q - v
    have huMod : u % q = u := Nat.mod_eq_of_lt hu_lt
    have hvMod : v % q = v := Nat.mod_eq_of_lt hv_lt
    have hU : centeredAbsNat u = u := by
      unfold centeredAbsNat
      simp [huMod, huH]
    have hV : centeredAbsNat v = dv := by
      unfold centeredAbsNat
      simp [hvMod, dv, hvH]
    have hv_le_q : v ≤ q := Nat.le_of_lt hv_lt
    have hq_sub_dv : q - dv = v := by
      simpa [dv] using (Nat.sub_sub_self hv_le_q)
    by_cases hcmp : dv ≤ u
    · have hsum : u + v = q + (u - dv) := by
        calc
          u + v = u + (q - dv) := by simpa [hq_sub_dv]
          _ = q + (u - dv) := by omega
      have hudv_lt_q : u - dv < q := by
        exact Nat.lt_of_le_of_lt (Nat.sub_le u dv) hu_lt
      have hmod : (u + v) % q = u - dv := by
        calc
          (u + v) % q = (q + (u - dv)) % q := by simpa [hsum]
          _ = (u - dv) % q := by
            simp [Nat.add_mod, Nat.mod_eq_of_lt q_pos]
          _ = u - dv := Nat.mod_eq_of_lt hudv_lt_q
      have hLHS : centeredAbsNat (u + v) = centeredAbsNat (u - dv) := by
        calc
          centeredAbsNat (u + v) = centeredAbsNat ((u + v) % q) := by
            simpa using (centeredAbsNat_mod (u + v)).symm
          _ = centeredAbsNat (u - dv) := by simpa [hmod]
      have hBound : centeredAbsNat (u - dv) ≤ u + dv := by
        exact Nat.le_trans
          (centeredAbsNat_le_self (u - dv))
          (by omega)
      simpa [hU, hV, hLHS, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hBound
    · have hlt : u < dv := Nat.lt_of_not_ge hcmp
      have hdv_le_q : dv ≤ q := Nat.sub_le _ _
      have hsum_lt_q : u + v < q := by
        calc
          u + v = u + (q - dv) := by simpa [hq_sub_dv]
          _ < dv + (q - dv) := Nat.add_lt_add_right hlt (q - dv)
          _ = q := by
            simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
              (Nat.sub_add_cancel hdv_le_q)
      have hsum : u + v = q - (dv - u) := by
        omega
      have hvgt : halfQ < v := Nat.lt_of_not_ge hvH
      have hvpos : 0 < v := Nat.lt_of_le_of_lt (Nat.zero_le halfQ) hvgt
      have hdv_lt_q : dv < q := by
        have hEq : q - v + v = q := Nat.sub_add_cancel (Nat.le_of_lt hv_lt)
        have hlt : q - v < q - v + v := Nat.lt_add_of_pos_right hvpos
        simpa [dv, hEq] using hlt
      have hdiff_lt_q : dv - u < q := by
        exact Nat.lt_of_le_of_lt (Nat.sub_le dv u) hdv_lt_q
      have hdiff_mod : (dv - u) % q = dv - u := Nat.mod_eq_of_lt hdiff_lt_q
      have hLHS : centeredAbsNat (u + v) = centeredAbsNat (dv - u) := by
        calc
          centeredAbsNat (u + v) = centeredAbsNat (q - (dv - u)) := by simpa [hsum]
          _ = centeredAbsNat (q - ((dv - u) % q)) := by simpa [hdiff_mod]
          _ = centeredAbsNat (dv - u) := by
            simpa using centeredAbsNat_neg_mod (dv - u)
      have hBound : centeredAbsNat (dv - u) ≤ u + dv := by
        exact Nat.le_trans
          (centeredAbsNat_le_self (dv - u))
          (by omega)
      simpa [hU, hV, hLHS, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hBound

  have hMain : centeredAbsNat (a + b) ≤ centeredAbsNat a + centeredAbsNat b := by
    by_cases haH : a ≤ halfQ
    · by_cases hbH : b ≤ halfQ
      · have hA : centeredAbsNat a = a := by
          unfold centeredAbsNat
          simp [Nat.mod_eq_of_lt ha_lt, haH]
        have hB : centeredAbsNat b = b := by
          unfold centeredAbsNat
          simp [Nat.mod_eq_of_lt hb_lt, hbH]
        have hL : centeredAbsNat (a + b) ≤ a + b := centeredAbsNat_le_self (a + b)
        simpa [hA, hB] using hL
      · exact hMixed a b ha_lt hb_lt haH hbH
    · by_cases hbH : b ≤ halfQ
      · have hSwap : centeredAbsNat (b + a) ≤ centeredAbsNat b + centeredAbsNat a :=
          hMixed b a hb_lt ha_lt hbH haH
        simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using hSwap
      · let da : Nat := q - a
        let db : Nat := q - b
        have hA : centeredAbsNat a = da := by
          unfold centeredAbsNat
          have hmod : a % q = a := Nat.mod_eq_of_lt ha_lt
          simp [hmod, da, haH]
        have hB : centeredAbsNat b = db := by
          unfold centeredAbsNat
          have hmod : b % q = b := Nat.mod_eq_of_lt hb_lt
          simp [hmod, db, hbH]
        have hda_le : da ≤ halfQ := by
          have hagt : halfQ < a := Nat.lt_of_not_ge haH
          have ha1 : halfQ + 1 ≤ a := Nat.succ_le_of_lt hagt
          have hsub : q - a ≤ q - (halfQ + 1) := Nat.sub_le_sub_left ha1 q
          simpa [da, q_sub_halfQ_succ_eq_halfQ] using hsub
        have hdb_le : db ≤ halfQ := by
          have hbgt : halfQ < b := Nat.lt_of_not_ge hbH
          have hb1 : halfQ + 1 ≤ b := Nat.succ_le_of_lt hbgt
          have hsub : q - b ≤ q - (halfQ + 1) := Nat.sub_le_sub_left hb1 q
          simpa [db, q_sub_halfQ_succ_eq_halfQ] using hsub
        have hsum_lt_q : da + db < q := by
          exact Nat.lt_of_le_of_lt (Nat.add_le_add hda_le hdb_le) halfQ_add_halfQ_lt_q
        have hsum : a + b = q + (q - (da + db)) := by
          omega
        have hmodsum : (a + b) % q = q - (da + db) := by
          calc
            (a + b) % q = (q + (q - (da + db))) % q := by simpa [hsum]
            _ = (q - (da + db)) % q := by
              simp [Nat.add_mod, Nat.mod_eq_of_lt q_pos]
            _ = q - (da + db) := Nat.mod_eq_of_lt (by omega)
        have hLHS : centeredAbsNat (a + b) = centeredAbsNat (da + db) := by
          calc
            centeredAbsNat (a + b) = centeredAbsNat ((a + b) % q) := by
              simpa using (centeredAbsNat_mod (a + b)).symm
            _ = centeredAbsNat (q - (da + db)) := by simpa [hmodsum]
            _ = centeredAbsNat (q - ((da + db) % q)) := by
              simp [Nat.mod_eq_of_lt hsum_lt_q]
            _ = centeredAbsNat (da + db) := by
              simpa using centeredAbsNat_neg_mod (da + db)
        have hBound : centeredAbsNat (da + db) ≤ da + db := centeredAbsNat_le_self (da + db)
        simpa [hA, hB, hLHS, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hBound

  calc
    centeredAbsNat (x + y) = centeredAbsNat (a + b) := hxy
    _ ≤ centeredAbsNat a + centeredAbsNat b := hMain
    _ = centeredAbsNat x + centeredAbsNat y := by
      simpa [hx, hy]

/-- Helper for Nat subtraction cancellation under a side-condition. -/
theorem sub_sub_cancel_of_le {a b : Nat} (h : b ≤ a) : a - (a - b) = b := by
  omega

/--
Nat-level modular identity used to eliminate one `q - _` factor in products:
`a * (q - b)` is `-(a*b)` modulo `q`.
-/
theorem mul_mod_q_sub (a b : Nat) (hb : b < q) :
  (a * (q - b)) % q = (q - ((a * b) % q)) % q := by
  let x : Nat := (a * (q - b)) % q
  let r : Nat := (a * b) % q
  have hx_lt : x < q := by
    simpa [x] using Nat.mod_lt (a * (q - b)) q_pos
  have hr_lt : r < q := by
    simpa [r] using Nat.mod_lt (a * b) q_pos
  have hsum : (a * (q - b) + a * b) % q = 0 := by
    have hbq : b ≤ q := Nat.le_of_lt hb
    calc
      (a * (q - b) + a * b) % q
          = (a * ((q - b) + b)) % q := by
              simpa [Nat.mul_add, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc]
      _ = (a * q) % q := by
            simp [Nat.sub_add_cancel hbq]
      _ = 0 := by simp
  have hsum0 : (x + r) % q = 0 := by
    calc
      (x + r) % q = (((a * (q - b)) % q) + ((a * b) % q)) % q := by
        simp [x, r]
      _ = (a * (q - b) + a * b) % q := by
        simp [Nat.add_mod]
      _ = 0 := hsum
  by_cases hr0 : r = 0
  · have hmod : x % q = 0 := by
      simpa [hr0, Nat.zero_add] using hsum0
    have hx0 : x = 0 := by
      simpa [Nat.mod_eq_of_lt hx_lt] using hmod
    simp [x, r, hr0, hx0]
  · have hrpos : 0 < r := Nat.pos_of_ne_zero hr0
    have hq_le_xr : q ≤ x + r := by
      by_cases h : q ≤ x + r
      · exact h
      · have hlt : x + r < q := Nat.lt_of_not_ge h
        have hmod : (x + r) % q = x + r := Nat.mod_eq_of_lt hlt
        have hxrz : x + r = 0 := by simpa [hmod] using hsum0
        have hr_le : r ≤ x + r := Nat.le_add_left r x
        have hr_zero : r = 0 := Nat.eq_zero_of_le_zero (by simpa [hxrz] using hr_le)
        exact False.elim (hr0 hr_zero)
    have hxr_lt_2q : x + r < q + q := Nat.add_lt_add hx_lt hr_lt
    have hdvd : q ∣ (x + r) := Nat.dvd_of_mod_eq_zero hsum0
    rcases hdvd with ⟨k, hk⟩
    have hk_ge1 : 1 ≤ k := by
      have hq_le_qk : q ≤ q * k := by simpa [hk] using hq_le_xr
      have hq1_le_qk : q * 1 ≤ q * k := by simpa [Nat.one_mul] using hq_le_qk
      exact Nat.le_of_mul_le_mul_left hq1_le_qk q_pos
    have hk_lt2 : k < 2 := by
      have hqk_lt_q2 : q * k < q * 2 := by
        simpa [hk, Nat.two_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hxr_lt_2q
      exact (Nat.mul_lt_mul_left q_pos).1 hqk_lt_q2
    have hk1 : k = 1 := by omega
    have hx_eq : x = q - r := by
      have hxr_eq : x + r = q := by simpa [hk, hk1]
      omega
    have hqmr_lt : q - r < q := by
      have hr_le_q : r ≤ q := Nat.le_of_lt hr_lt
      have hEq : q - r + r = q := Nat.sub_add_cancel hr_le_q
      have hlt : q - r < q - r + r := Nat.lt_add_of_pos_right hrpos
      simpa [hEq] using hlt
    have hqmr_mod : (q - r) % q = q - r := Nat.mod_eq_of_lt hqmr_lt
    simpa [x, r, hx_eq, hqmr_mod]

/-- Nat-level submultiplicativity of centered absolute values modulo `q`. -/
theorem centeredAbsNat_mul_le_mul (x y : Nat) :
  centeredAbsNat (x * y) ≤ centeredAbsNat x * centeredAbsNat y := by
  let a : Nat := x % q
  let b : Nat := y % q
  have ha_lt : a < q := by simpa [a] using Nat.mod_lt x q_pos
  have hb_lt : b < q := by simpa [b] using Nat.mod_lt y q_pos
  have hx : centeredAbsNat x = centeredAbsNat a := by
    simpa [a] using (centeredAbsNat_mod x).symm
  have hy : centeredAbsNat y = centeredAbsNat b := by
    simpa [b] using (centeredAbsNat_mod y).symm
  have hxy : centeredAbsNat (x * y) = centeredAbsNat (a * b) := by
    calc
      centeredAbsNat (x * y) = centeredAbsNat ((x * y) % q) := by
        simpa using (centeredAbsNat_mod (x * y)).symm
      _ = centeredAbsNat (((x % q) * (y % q)) % q) := by simp [Nat.mul_mod]
      _ = centeredAbsNat ((a * b) % q) := by simp [a, b]
      _ = centeredAbsNat (a * b) := by simpa using centeredAbsNat_mod (a * b)

  have hMain : centeredAbsNat (a * b) ≤ centeredAbsNat a * centeredAbsNat b := by
    by_cases haH : a ≤ halfQ
    · by_cases hbH : b ≤ halfQ
      · have hA : centeredAbsNat a = a := by
          unfold centeredAbsNat
          simp [Nat.mod_eq_of_lt ha_lt, haH]
        have hB : centeredAbsNat b = b := by
          unfold centeredAbsNat
          simp [Nat.mod_eq_of_lt hb_lt, hbH]
        have hL : centeredAbsNat (a * b) ≤ a * b := centeredAbsNat_le_self (a * b)
        simpa [hA, hB] using hL
      · let db : Nat := q - b
        have hA : centeredAbsNat a = a := by
          unfold centeredAbsNat
          simp [Nat.mod_eq_of_lt ha_lt, haH]
        have hB : centeredAbsNat b = db := by
          unfold centeredAbsNat
          simp [Nat.mod_eq_of_lt hb_lt, db, hbH]
        have hb_le_q : b ≤ q := Nat.le_of_lt hb_lt
        have hbEq : b = q - db := by
          simpa [db] using (sub_sub_cancel_of_le (a := q) (b := b) hb_le_q).symm
        have db_lt_q : db < q := by
          have hbgt : halfQ < b := Nat.lt_of_not_ge hbH
          have hbpos : 0 < b := Nat.lt_of_le_of_lt (Nat.zero_le halfQ) hbgt
          omega
        have hmodMul' : (a * b) % q = (q - ((a * db) % q)) % q := by
          simpa [hbEq] using (mul_mod_q_sub a db db_lt_q)
        have hLHS : centeredAbsNat (a * b) = centeredAbsNat (a * db) := by
          calc
            centeredAbsNat (a * b) = centeredAbsNat ((a * b) % q) := by
              simpa using (centeredAbsNat_mod (a * b)).symm
            _ = centeredAbsNat ((q - ((a * db) % q)) % q) := by simpa [hmodMul']
            _ = centeredAbsNat (q - ((a * db) % q)) := by
                  simpa using (centeredAbsNat_mod (q - ((a * db) % q)))
            _ = centeredAbsNat (a * db) := by
                  simpa using centeredAbsNat_neg_mod (a * db)
        have hBound : centeredAbsNat (a * db) ≤ a * db := centeredAbsNat_le_self (a * db)
        simpa [hA, hB, hLHS] using hBound
    · by_cases hbH : b ≤ halfQ
      · have hAgt : halfQ < a := Nat.lt_of_not_ge haH
        have hB : centeredAbsNat b = b := by
          unfold centeredAbsNat
          simp [Nat.mod_eq_of_lt hb_lt, hbH]
        let da : Nat := q - a
        have hA : centeredAbsNat a = da := by
          unfold centeredAbsNat
          simp [Nat.mod_eq_of_lt ha_lt, da, haH]
        have ha_le_q : a ≤ q := Nat.le_of_lt ha_lt
        have haEq : a = q - da := by
          simpa [da] using (sub_sub_cancel_of_le (a := q) (b := a) ha_le_q).symm
        have da_lt_q : da < q := by
          have hapos : 0 < a := Nat.lt_of_le_of_lt (Nat.zero_le halfQ) hAgt
          omega
        have hmodMul' : (b * a) % q = (q - ((b * da) % q)) % q := by
          simpa [haEq] using (mul_mod_q_sub b da da_lt_q)
        have hLHS : centeredAbsNat (a * b) = centeredAbsNat (b * da) := by
          calc
            centeredAbsNat (a * b) = centeredAbsNat (b * a) := by simp [Nat.mul_comm]
            _ = centeredAbsNat ((b * a) % q) := by
                  simpa using (centeredAbsNat_mod (b * a)).symm
            _ = centeredAbsNat ((q - ((b * da) % q)) % q) := by simpa [hmodMul']
            _ = centeredAbsNat (q - ((b * da) % q)) := by
                  simpa using (centeredAbsNat_mod (q - ((b * da) % q)))
            _ = centeredAbsNat (b * da) := by
                  simpa using centeredAbsNat_neg_mod (b * da)
        have hBound : centeredAbsNat (b * da) ≤ b * da := centeredAbsNat_le_self (b * da)
        simpa [hA, hB, Nat.mul_comm, hLHS] using hBound
      · have hAgt : halfQ < a := Nat.lt_of_not_ge haH
        have hBgt : halfQ < b := Nat.lt_of_not_ge hbH
        let da : Nat := q - a
        let db : Nat := q - b
        have hA : centeredAbsNat a = da := by
          unfold centeredAbsNat
          simp [Nat.mod_eq_of_lt ha_lt, da, haH]
        have hB : centeredAbsNat b = db := by
          unfold centeredAbsNat
          simp [Nat.mod_eq_of_lt hb_lt, db, hbH]
        have ha_le_q : a ≤ q := Nat.le_of_lt ha_lt
        have hb_le_q : b ≤ q := Nat.le_of_lt hb_lt
        have haEq : a = q - da := by
          simpa [da] using (sub_sub_cancel_of_le (a := q) (b := a) ha_le_q).symm
        have hbEq : b = q - db := by
          simpa [db] using (sub_sub_cancel_of_le (a := q) (b := b) hb_le_q).symm
        have hda_lt_q : da < q := by
          have hapos : 0 < a := Nat.lt_of_le_of_lt (Nat.zero_le halfQ) hAgt
          omega
        have hdb_lt_q : db < q := by
          have hbpos : 0 < b := Nat.lt_of_le_of_lt (Nat.zero_le halfQ) hBgt
          omega
        have h1' : (a * b) % q = (q - (((q - da) * db) % q)) % q := by
          simpa [haEq, hbEq] using (mul_mod_q_sub (q - da) db hdb_lt_q)
        have h2' : ((q - da) * db) % q = (q - ((da * db) % q)) % q := by
          have htmp : (db * (q - da)) % q = (q - ((db * da) % q)) % q :=
            mul_mod_q_sub db da hda_lt_q
          simpa [Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using htmp
        have hLHS : centeredAbsNat (a * b) = centeredAbsNat (da * db) := by
          calc
            centeredAbsNat (a * b) = centeredAbsNat ((a * b) % q) := by
              simpa using (centeredAbsNat_mod (a * b)).symm
            _ = centeredAbsNat ((q - ((q - ((da * db) % q)) % q)) % q) := by
                  simpa [h1', h2']
            _ = centeredAbsNat (q - ((q - ((da * db) % q)) % q)) := by
                  simpa using (centeredAbsNat_mod (q - ((q - ((da * db) % q)) % q)))
            _ = centeredAbsNat (q - ((da * db) % q)) := by
                  simpa using centeredAbsNat_neg_mod (q - ((da * db) % q))
            _ = centeredAbsNat ((da * db) % q) := by
                  simpa using centeredAbsNat_neg_mod ((da * db) % q)
            _ = centeredAbsNat (da * db) := by
                  simpa using centeredAbsNat_mod (da * db)
        have hBound : centeredAbsNat (da * db) ≤ da * db := centeredAbsNat_le_self (da * db)
        simpa [hA, hB, hLHS] using hBound

  calc
    centeredAbsNat (x * y) = centeredAbsNat (a * b) := hxy
    _ ≤ centeredAbsNat a * centeredAbsNat b := hMain
    _ = centeredAbsNat x * centeredAbsNat y := by
      simpa [hx, hy]

/-- Infinity norm of a field element in centered representation. -/
def normInfF (a : F) : Nat := centeredAbsNat a.val

theorem normInfF_neg (a : F) : normInfF (-a) = normInfF a := by
  cases a with
  | mk av =>
      unfold normInfF
      change centeredAbsNat (F.ofNat (q - (av % q))).val = centeredAbsNat av
      simpa [F.ofNat_val_mod, centeredAbsNat_mod] using centeredAbsNat_neg_mod av

theorem normInfF_add_le_add_theorem (x y : F) :
  normInfF (x + y) ≤ normInfF x + normInfF y := by
  unfold normInfF
  change centeredAbsNat ((x.val + y.val) % q) ≤ centeredAbsNat x.val + centeredAbsNat y.val
  have hAdd : centeredAbsNat (x.val + y.val) ≤ centeredAbsNat x.val + centeredAbsNat y.val :=
    centeredAbsNat_add_le x.val y.val
  simpa [centeredAbsNat_mod] using hAdd

theorem normInfF_mul_le_mul_theorem (x y : F) :
  normInfF (x * y) ≤ normInfF x * normInfF y := by
  unfold normInfF
  change centeredAbsNat ((x.val * y.val) % q) ≤ centeredAbsNat x.val * centeredAbsNat y.val
  have hMul : centeredAbsNat (x.val * y.val) ≤ centeredAbsNat x.val * centeredAbsNat y.val :=
    centeredAbsNat_mul_le_mul x.val y.val
  simpa [centeredAbsNat_mod] using hMul

/--
Signed centered representative of a Nat residue mod `q`.
Used as an Int-level bridge for proving norm inequalities.
-/
def centeredRepNat (x : Nat) : Int :=
  let xr := x % q
  if xr <= halfQ then
    Int.ofNat xr
  else
    -Int.ofNat (q - xr)

theorem centeredRepNat_natAbs_eq_centeredAbsNat (x : Nat) :
  Int.natAbs (centeredRepNat x) = centeredAbsNat x := by
  unfold centeredRepNat centeredAbsNat
  by_cases h : x % q <= halfQ
  · rw [if_pos h, if_pos h]
    simpa using (Int.natAbs_natCast (x % q))
  · rw [if_neg h, if_neg h]
    simp [Int.natAbs_natCast]

/-- Centered representative depends only on the residue class modulo `q`. -/
theorem centeredRepNat_mod (x : Nat) :
  centeredRepNat (x % q) = centeredRepNat x := by
  unfold centeredRepNat
  simp

/-- Signed centered representative of a field element. -/
def centeredRep (a : F) : Int := centeredRepNat a.val

theorem centeredRep_natAbs_eq_normInfF (a : F) :
  Int.natAbs (centeredRep a) = normInfF a := by
  unfold centeredRep normInfF
  exact centeredRepNat_natAbs_eq_centeredAbsNat a.val

theorem centeredRep_ofNat (x : Nat) :
  centeredRep (F.ofNat x) = centeredRepNat x := by
  unfold centeredRep
  simpa [F.ofNat_val_mod] using centeredRepNat_mod x

/--
Centered-representation additive triangle assumption.
This is the Int-level form of `schoolbookAddTriangleBound`.
-/
def centeredRepAddTriangleBound : Prop :=
  ∀ x y : F, Int.natAbs (centeredRep (x + y))
    ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)

/--
Centered-representation multiplicative submultiplicative assumption.
This is the Int-level form of `schoolbookMulUniversalBound`.
-/
def centeredRepMulUniversalBound : Prop :=
  ∀ x y : F, Int.natAbs (centeredRep (x * y))
    ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y)

/-- Common centered-representation blocker bundle used in non-coarse P5 paths. -/
def centeredRepMulAddBounds : Prop :=
  centeredRepMulUniversalBound ∧ centeredRepAddTriangleBound

theorem centeredRepMulAddBounds_mul
  (h : centeredRepMulAddBounds) :
  centeredRepMulUniversalBound := by
  exact h.1

theorem centeredRepMulAddBounds_add
  (h : centeredRepMulAddBounds) :
  centeredRepAddTriangleBound := by
  exact h.2

theorem centeredRepAddTriangleBound_theorem : centeredRepAddTriangleBound := by
  intro x y
  simpa [centeredRep_natAbs_eq_normInfF] using (normInfF_add_le_add_theorem x y)

theorem centeredRepMulUniversalBound_theorem : centeredRepMulUniversalBound := by
  intro x y
  simpa [centeredRep_natAbs_eq_normInfF] using (normInfF_mul_le_mul_theorem x y)

theorem centeredRepMulAddBounds_theorem : centeredRepMulAddBounds := by
  exact ⟨centeredRepMulUniversalBound_theorem, centeredRepAddTriangleBound_theorem⟩

theorem centeredRepMulAddBounds_of_mul
  (hMul : centeredRepMulUniversalBound) :
  centeredRepMulAddBounds := by
  exact ⟨hMul, centeredRepAddTriangleBound_theorem⟩

/-- Infinity norm of one ring element represented by its d coefficients. -/
def normInfCoeffs (a : Coeffs) : Nat :=
  a.foldl (fun m x => Nat.max m (normInfF x)) 0

/-- Infinity norm of a vector of ring elements (max over element norms). -/
def normInfCoeffMatrix (m : Array Coeffs) : Nat :=
  m.foldl (fun acc row => Nat.max acc (normInfCoeffs row)) 0

/--
Coefficientwise subtraction used by low-norm invertibility side conditions.
Returns `#[]` on mismatched sizes.
-/
def coeffSub (a b : Coeffs) : Coeffs :=
  if _h : a.size = b.size then
    Array.ofFn (fun i : Fin a.size => a[i.1]! - b[i.1]!)
  else
    #[]

theorem coeffSub_size_of_eq
  {a b : Coeffs}
  (hSize : a.size = b.size) :
  (coeffSub a b).size = a.size := by
  unfold coeffSub
  simp [hSize]

/-- Tiny sanity checks matching Definition 3 behavior. -/
def normSanity : Bool :=
  let x : F := F.ofNat 3
  let y : F := F.ofNat (q - 1)
  decide (normInfF x = 3 ∧ normInfF y = 1)

def normSanityProp : Prop :=
  let x : F := F.ofNat 3
  let y : F := F.ofNat (q - 1)
  normInfF x = 3 ∧ normInfF y = 1

theorem normSanity_sound (hOk : normSanity = true) : normSanityProp := by
  unfold normSanity at hOk
  simpa [normSanityProp] using (decide_eq_true_eq.mp hOk)

theorem normInfCoeffs_nonneg (a : Coeffs) : 0 <= normInfCoeffs a :=
  Nat.zero_le (normInfCoeffs a)

theorem normInfCoeffMatrix_nonneg (m : Array Coeffs) : 0 <= normInfCoeffMatrix m :=
  Nat.zero_le (normInfCoeffMatrix m)

theorem normInfCoeffs_le_of_entry_bound
  {a : Coeffs} {B : Nat}
  (hEntry : ∀ i (hi : i < a.size), normInfF (a[i]'hi) ≤ B) :
  normInfCoeffs a ≤ B := by
  unfold normInfCoeffs
  refine Array.foldl_induction (as := a) (motive := fun _ acc => acc ≤ B) ?_ ?_
  · exact Nat.zero_le B
  · intro i acc hAcc
    have hI : normInfF (a[i]) ≤ B := by
      simpa using hEntry i i.2
    exact (Nat.max_le).2 ⟨hAcc, hI⟩

theorem normInfCoeffMatrix_le_of_row_bound
  {m : Array Coeffs} {B : Nat}
  (hRow : ∀ i (hi : i < m.size), normInfCoeffs (m[i]'hi) ≤ B) :
  normInfCoeffMatrix m ≤ B := by
  unfold normInfCoeffMatrix
  refine Array.foldl_induction (as := m) (motive := fun _ acc => acc ≤ B) ?_ ?_
  · exact Nat.zero_le B
  · intro i acc hAcc
    have hI : normInfCoeffs (m[i]) ≤ B := by
      simpa using hRow i i.2
    exact (Nat.max_le).2 ⟨hAcc, hI⟩

theorem normInfCoeffs_le_of_hasRingDegreeShape_and_coeff_bound
  {a : Coeffs} {B : Nat}
  (hShape : hasRingDegreeShape a)
  (hCoeff : ∀ k, k < D → normInfF (a[k]!) ≤ B) :
  normInfCoeffs a ≤ B := by
  apply normInfCoeffs_le_of_entry_bound
  intro i hi
  have hSize : a.size = D := hShape
  have hiD : i < D := by
    simpa [hSize] using (show i < a.size from hi)
  have hIBang : normInfF (a[i]!) ≤ B := hCoeff i hiD
  simpa [hi] using hIBang

theorem normInfCoeffs_entry_le
  {a : Coeffs} {i : Nat}
  (hi : i < a.size) :
  normInfF (a[i]'hi) ≤ normInfCoeffs a := by
  unfold normInfCoeffs
  have hAll :
      ∀ t (ht : t < a.size), normInfF (a[t]!) ≤ a.foldl (fun m x => Nat.max m (normInfF x)) 0 := by
    exact Array.foldl_induction
      (as := a)
      (motive := fun j acc => ∀ t, t < j → normInfF (a[t]!) ≤ acc)
      (h0 := by
        intro t ht
        exact (Nat.not_lt_zero t ht).elim)
      (hf := by
        intro j acc hAcc t ht
        by_cases htj : t < j.1
        · exact Nat.le_trans (hAcc t htj) (Nat.le_max_left _ _)
        · have hle : t ≤ j.1 := Nat.le_of_lt_succ ht
          have hge : j.1 ≤ t := Nat.le_of_not_gt htj
          have hEq : t = j.1 := Nat.le_antisymm hle hge
          subst hEq
          simpa [j.2] using (Nat.le_max_right acc (normInfF (a[j])))
      )
  have hIBang : normInfF (a[i]!) ≤ a.foldl (fun m x => Nat.max m (normInfF x)) 0 := hAll i hi
  simpa [hi] using hIBang

theorem normInfF_getCoeff_le_of_normInfCoeffs
  {a : Coeffs} {B : Nat}
  (hA : normInfCoeffs a ≤ B)
  (i : Nat) :
  normInfF (getCoeff a i) ≤ B := by
  by_cases hi : i < a.size
  · have hEntry : normInfF (a[i]'hi) ≤ normInfCoeffs a :=
      normInfCoeffs_entry_le (a := a) hi
    exact Nat.le_trans (by simpa [getCoeff, hi] using hEntry) hA
  · have hGe : a.size ≤ i := Nat.le_of_not_gt hi
    have hZero : getCoeff a i = 0 := getCoeff_eq_zero_of_ge hGe
    have hZeroEq : normInfF (0 : F) = 0 := by
      native_decide
    have hZeroLe : normInfF (0 : F) ≤ B := by
      simpa [hZeroEq] using (Nat.zero_le B)
    simpa [hZero] using hZeroLe

theorem normInfF_getElemBang_le_of_normInfCoeffs
  {a : Coeffs} {B : Nat}
  (hA : normInfCoeffs a ≤ B)
  (i : Nat) :
  normInfF (a[i]!) ≤ B := by
  simpa [getElemBang_eq_getCoeff] using
    (normInfF_getCoeff_le_of_normInfCoeffs (a := a) (B := B) hA i)

theorem normInfF_ofNat_le_halfQ
  (x : Nat)
  (hx : x % q <= halfQ) :
  normInfF (F.ofNat x) = x % q := by
  unfold normInfF centeredAbsNat
  simp [F.ofNat, hx]

theorem normInfF_ofNat_gt_halfQ
  (x : Nat)
  (hx : halfQ < x % q) :
  normInfF (F.ofNat x) = q - (x % q) := by
  unfold normInfF centeredAbsNat
  have hNotLe : ¬x % q <= halfQ := Nat.not_le.mpr hx
  simp [F.ofNat, hNotLe]

theorem centeredAbsNat_le_q (x : Nat) : centeredAbsNat x ≤ q := by
  unfold centeredAbsNat
  by_cases h : x % q ≤ halfQ
  · have hmod : x % q < q := Nat.mod_lt _ q_pos
    have hle : x % q ≤ q := Nat.le_of_lt hmod
    exact by simpa [h] using hle
  · have hsub : q - (x % q) ≤ q := Nat.sub_le _ _
    simpa [h] using hsub

theorem centeredAbsNat_le_halfQ (x : Nat) : centeredAbsNat x ≤ halfQ := by
  unfold centeredAbsNat
  by_cases h : x % q ≤ halfQ
  · simpa [h] using h
  · have hx : halfQ < x % q := Nat.lt_of_not_ge h
    have hx' : halfQ + 1 ≤ x % q := Nat.succ_le_of_lt hx
    have hsub : q - (x % q) ≤ q - (halfQ + 1) := Nat.sub_le_sub_left hx' q
    simpa [h, q_sub_halfQ_succ_eq_halfQ] using hsub

theorem normInfF_le_q (a : F) : normInfF a ≤ q := by
  unfold normInfF
  exact centeredAbsNat_le_q a.val

theorem normInfF_le_halfQ (a : F) : normInfF a ≤ halfQ := by
  unfold normInfF
  exact centeredAbsNat_le_halfQ a.val

theorem centeredRepNat_natAbs_le_halfQ (x : Nat) :
  Int.natAbs (centeredRepNat x) ≤ halfQ := by
  rw [centeredRepNat_natAbs_eq_centeredAbsNat]
  exact centeredAbsNat_le_halfQ x

theorem centeredRep_natAbs_le_halfQ (a : F) :
  Int.natAbs (centeredRep a) ≤ halfQ := by
  rw [centeredRep_natAbs_eq_normInfF]
  exact normInfF_le_halfQ a

theorem normInfCoeffs_le_q (a : Coeffs) : normInfCoeffs a ≤ q := by
  exact normInfCoeffs_le_of_entry_bound (a := a) (B := q) (fun i hi => by
    simpa using normInfF_le_q (a[i]'hi))

theorem normInfCoeffs_le_halfQ (a : Coeffs) : normInfCoeffs a ≤ halfQ := by
  exact normInfCoeffs_le_of_entry_bound (a := a) (B := halfQ) (fun i hi => by
    simpa using normInfF_le_halfQ (a[i]'hi))

theorem normInfCoeffMatrix_le_q (m : Array Coeffs) : normInfCoeffMatrix m ≤ q := by
  exact normInfCoeffMatrix_le_of_row_bound (m := m) (B := q) (fun i hi => by
    simpa using normInfCoeffs_le_q (m[i]'hi))

theorem normInfCoeffMatrix_le_halfQ (m : Array Coeffs) : normInfCoeffMatrix m ≤ halfQ := by
  exact normInfCoeffMatrix_le_of_row_bound (m := m) (B := halfQ) (fun i hi => by
    simpa using normInfCoeffs_le_halfQ (m[i]'hi))

theorem normInfF_add_le_q (x y : F) : normInfF (x + y) ≤ q := by
  simpa using normInfF_le_q (x + y)

theorem normInfF_add_le_halfQ (x y : F) : normInfF (x + y) ≤ halfQ := by
  simpa using normInfF_le_halfQ (x + y)

theorem normInfF_sub_le_q (x y : F) : normInfF (x - y) ≤ q := by
  simpa using normInfF_le_q (x - y)

theorem normInfF_sub_le_halfQ (x y : F) : normInfF (x - y) ≤ halfQ := by
  simpa using normInfF_le_halfQ (x - y)

theorem normInfF_mul_le_q (x y : F) : normInfF (x * y) ≤ q := by
  simpa using normInfF_le_q (x * y)

theorem normInfF_mul_le_halfQ (x y : F) : normInfF (x * y) ≤ halfQ := by
  simpa using normInfF_le_halfQ (x * y)

theorem normInfCoeffs_vecAdd_le_q (a b : Coeffs) : normInfCoeffs (vecAdd a b) ≤ q := by
  simpa using normInfCoeffs_le_q (vecAdd a b)

theorem normInfCoeffs_vecAdd_le_halfQ (a b : Coeffs) : normInfCoeffs (vecAdd a b) ≤ halfQ := by
  simpa using normInfCoeffs_le_halfQ (vecAdd a b)

theorem normInfCoeffs_vecScale_le_q (s : F) (a : Coeffs) : normInfCoeffs (vecScale s a) ≤ q := by
  simpa using normInfCoeffs_le_q (vecScale s a)

theorem normInfCoeffs_vecScale_le_halfQ (s : F) (a : Coeffs) : normInfCoeffs (vecScale s a) ≤ halfQ := by
  simpa using normInfCoeffs_le_halfQ (vecScale s a)

theorem normInfCoeffs_mulRq_le_q (a b : Coeffs) : normInfCoeffs (mulRq a b) ≤ q := by
  simpa using normInfCoeffs_le_q (mulRq a b)

theorem normInfCoeffs_mulRq_le_halfQ (a b : Coeffs) : normInfCoeffs (mulRq a b) ≤ halfQ := by
  simpa using normInfCoeffs_le_halfQ (mulRq a b)

theorem normInfCoeffs_coeffSub_le_q (a b : Coeffs) : normInfCoeffs (coeffSub a b) ≤ q := by
  simpa using normInfCoeffs_le_q (coeffSub a b)

theorem normInfCoeffs_coeffSub_le_halfQ (a b : Coeffs) : normInfCoeffs (coeffSub a b) ≤ halfQ := by
  simpa using normInfCoeffs_le_halfQ (coeffSub a b)

theorem normInfCoeffs_superneoBarBlock_le_q
  (bar : Array (Array F)) (v : Coeffs) :
  normInfCoeffs (superneoBarBlock bar v) ≤ q := by
  simpa using normInfCoeffs_le_q (superneoBarBlock bar v)

theorem normInfCoeffs_superneoBarBlock_le_halfQ
  (bar : Array (Array F)) (v : Coeffs) :
  normInfCoeffs (superneoBarBlock bar v) ≤ halfQ := by
  simpa using normInfCoeffs_le_halfQ (superneoBarBlock bar v)

theorem normInfCoeffs_barLiftVec_le_q
  (bar : Array (Array F)) (v : Array F) :
  normInfCoeffs (barLiftVec bar v) ≤ q := by
  simpa using normInfCoeffs_le_q (barLiftVec bar v)

theorem normInfCoeffs_barLiftVec_le_halfQ
  (bar : Array (Array F)) (v : Array F) :
  normInfCoeffs (barLiftVec bar v) ≤ halfQ := by
  simpa using normInfCoeffs_le_halfQ (barLiftVec bar v)

theorem normInfCoeffMatrix_barLiftMatrix_le_q
  (bar : Array (Array F)) (m : Array (Array F)) :
  normInfCoeffMatrix (barLiftMatrix bar m) ≤ q := by
  simpa using normInfCoeffMatrix_le_q (barLiftMatrix bar m)

theorem normInfCoeffMatrix_barLiftMatrix_le_halfQ
  (bar : Array (Array F)) (m : Array (Array F)) :
  normInfCoeffMatrix (barLiftMatrix bar m) ≤ halfQ := by
  simpa using normInfCoeffMatrix_le_halfQ (barLiftMatrix bar m)

theorem normInfCoeffs_vecAdd_le_of_entry_bound
  {a b : Coeffs} {B : Nat}
  (hSize : a.size = b.size)
  (hEntry : ∀ i (hiA : i < a.size) (hiB : i < b.size),
      normInfF ((a[i]'hiA) + (b[i]'hiB)) ≤ B) :
  normInfCoeffs (vecAdd a b) ≤ B := by
  apply normInfCoeffs_le_of_entry_bound
  intro i hi
  have hiA : i < a.size := by
    simpa [vecAdd_size_of_eq hSize] using hi
  have hiB : i < b.size := by
    simpa [hSize] using hiA
  have hI : normInfF ((a[i]'hiA) + (b[i]'hiB)) ≤ B := hEntry i hiA hiB
  simpa [vecAdd, hSize, hiA, hiB] using hI

theorem normInfCoeffs_coeffSub_le_of_entry_bound
  {a b : Coeffs} {B : Nat}
  (hSize : a.size = b.size)
  (hEntry : ∀ i (hiA : i < a.size) (hiB : i < b.size),
      normInfF ((a[i]'hiA) - (b[i]'hiB)) ≤ B) :
  normInfCoeffs (coeffSub a b) ≤ B := by
  apply normInfCoeffs_le_of_entry_bound
  intro i hi
  have hiA : i < a.size := by
    simpa [coeffSub_size_of_eq hSize] using hi
  have hiB : i < b.size := by
    simpa [hSize] using hiA
  have hI : normInfF ((a[i]'hiA) - (b[i]'hiB)) ≤ B := hEntry i hiA hiB
  simpa [coeffSub, hSize, hiA, hiB] using hI

theorem normInfCoeffs_vecScale_le_of_entry_bound
  {s : F} {a : Coeffs} {B : Nat}
  (hEntry : ∀ i (hi : i < a.size), normInfF (s * (a[i]'hi)) ≤ B) :
  normInfCoeffs (vecScale s a) ≤ B := by
  apply normInfCoeffs_le_of_entry_bound
  intro i hi
  have hiA : i < a.size := by
    simpa [vecScale_size s a] using hi
  have hI : normInfF (s * (a[i]'hiA)) ≤ B := hEntry i hiA
  simpa [vecScale, hiA] using hI

theorem normInfCoeffs_mulRq_le_of_coeffSpec_bound
  {a b : Coeffs} {B : Nat}
  (hCoeff : ∀ k, k < D → normInfF (mulRqCoeffSpec a b k) ≤ B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  refine normInfCoeffs_le_of_hasRingDegreeShape_and_coeff_bound
    (hShape := hasRingDegreeShape_mulRq a b) ?_
  intro k hk
  have hK : (mulRq a b)[k]! = mulRqCoeffSpec a b k := mulRq_coeff_spec (a := a) (b := b) (k := k) hk
  simpa [hK] using hCoeff k hk

theorem normInfCoeffs_vecAdd_le_of_norm_bounds
  {a b : Coeffs} {BA BB B : Nat}
  (hSize : a.size = b.size)
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hAdd : ∀ x y, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x + y) ≤ B) :
  normInfCoeffs (vecAdd a b) ≤ B := by
  exact normInfCoeffs_vecAdd_le_of_entry_bound hSize (fun i hiA hiB => by
    exact hAdd (a[i]'hiA) (b[i]'hiB)
      (Nat.le_trans (normInfCoeffs_entry_le (a := a) hiA) hA)
      (Nat.le_trans (normInfCoeffs_entry_le (a := b) hiB) hB))

theorem normInfCoeffs_coeffSub_le_of_norm_bounds
  {a b : Coeffs} {BA BB B : Nat}
  (hSize : a.size = b.size)
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hSub : ∀ x y, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x - y) ≤ B) :
  normInfCoeffs (coeffSub a b) ≤ B := by
  exact normInfCoeffs_coeffSub_le_of_entry_bound hSize (fun i hiA hiB => by
    exact hSub (a[i]'hiA) (b[i]'hiB)
      (Nat.le_trans (normInfCoeffs_entry_le (a := a) hiA) hA)
      (Nat.le_trans (normInfCoeffs_entry_le (a := b) hiB) hB))

theorem normInfCoeffs_vecScale_le_of_norm_bounds
  {s : F} {a : Coeffs} {BS BA B : Nat}
  (hS : normInfF s ≤ BS)
  (hA : normInfCoeffs a ≤ BA)
  (hMul : ∀ x y, normInfF x ≤ BS → normInfF y ≤ BA → normInfF (x * y) ≤ B) :
  normInfCoeffs (vecScale s a) ≤ B := by
  exact normInfCoeffs_vecScale_le_of_entry_bound (s := s) (a := a) (B := B) (fun i hi => by
    exact hMul s (a[i]'hi) hS
      (Nat.le_trans (normInfCoeffs_entry_le (a := a) hi) hA))

/--
Assumption bundle: operand norm bounds imply a bound on `vecAdd`.
-/
def vecAddNormBoundFromOperands (BA BB B : Nat) : Prop :=
  ∀ a b : Coeffs,
    a.size = b.size →
    normInfCoeffs a ≤ BA →
    normInfCoeffs b ≤ BB →
    normInfCoeffs (vecAdd a b) ≤ B

/--
Assumption bundle: operand norm bounds imply a bound on `coeffSub`.
-/
def coeffSubNormBoundFromOperands (BA BB B : Nat) : Prop :=
  ∀ a b : Coeffs,
    a.size = b.size →
    normInfCoeffs a ≤ BA →
    normInfCoeffs b ≤ BB →
    normInfCoeffs (coeffSub a b) ≤ B

/--
Assumption bundle: scalar/operand norm bounds imply a bound on `vecScale`.
-/
def vecScaleNormBoundFromOperands (BS BA B : Nat) : Prop :=
  ∀ s : F, ∀ a : Coeffs,
    normInfF s ≤ BS →
    normInfCoeffs a ≤ BA →
    normInfCoeffs (vecScale s a) ≤ B

theorem vecAddNormBoundFromOperands_of_opBound
  {BA BB B : Nat}
  (hAdd : ∀ x y, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x + y) ≤ B) :
  vecAddNormBoundFromOperands BA BB B := by
  intro a b hSize hA hB
  exact normInfCoeffs_vecAdd_le_of_norm_bounds
    (a := a) (b := b) (BA := BA) (BB := BB) (B := B)
    hSize hA hB hAdd

theorem coeffSubNormBoundFromOperands_of_opBound
  {BA BB B : Nat}
  (hSub : ∀ x y, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x - y) ≤ B) :
  coeffSubNormBoundFromOperands BA BB B := by
  intro a b hSize hA hB
  exact normInfCoeffs_coeffSub_le_of_norm_bounds
    (a := a) (b := b) (BA := BA) (BB := BB) (B := B)
    hSize hA hB hSub

theorem vecScaleNormBoundFromOperands_of_opBound
  {BS BA B : Nat}
  (hMul : ∀ x y, normInfF x ≤ BS → normInfF y ≤ BA → normInfF (x * y) ≤ B) :
  vecScaleNormBoundFromOperands BS BA B := by
  intro s a hS hA
  exact normInfCoeffs_vecScale_le_of_norm_bounds
    (s := s) (a := a) (BS := BS) (BA := BA) (B := B)
    hS hA hMul

theorem vecAddNormBoundFromOperands_of_triangle
  {BA BB : Nat}
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y) :
  vecAddNormBoundFromOperands BA BB (BA + BB) := by
  exact vecAddNormBoundFromOperands_of_opBound (BA := BA) (BB := BB) (B := BA + BB) (fun x y hx hy => by
    exact Nat.le_trans (hAddTri x y) (Nat.add_le_add hx hy))

theorem coeffSubNormBoundFromOperands_of_triangle
  {BA BB : Nat}
  (hSubTri : ∀ x y : F, normInfF (x - y) ≤ normInfF x + normInfF y) :
  coeffSubNormBoundFromOperands BA BB (BA + BB) := by
  exact coeffSubNormBoundFromOperands_of_opBound (BA := BA) (BB := BB) (B := BA + BB) (fun x y hx hy => by
    exact Nat.le_trans (hSubTri x y) (Nat.add_le_add hx hy))

theorem vecScaleNormBoundFromOperands_of_universal
  {BS BA : Nat}
  (hMulUniv : ∀ x y : F, normInfF (x * y) ≤ normInfF x * normInfF y) :
  vecScaleNormBoundFromOperands BS BA (BS * BA) := by
  exact vecScaleNormBoundFromOperands_of_opBound (BS := BS) (BA := BA) (B := BS * BA) (fun x y hx hy => by
    exact Nat.le_trans (hMulUniv x y) (Nat.mul_le_mul hx hy))

theorem normInfCoeffs_mulRq_le_of_norm_bounds
  {a b : Coeffs} {BA BB B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hCoeffFromNorm :
    ∀ k, k < D →
      normInfCoeffs a ≤ BA →
      normInfCoeffs b ≤ BB →
      normInfF (mulRqCoeffSpec a b k) ≤ B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact normInfCoeffs_mulRq_le_of_coeffSpec_bound (a := a) (b := b) (B := B) (fun k hk => by
    exact hCoeffFromNorm k hk hA hB)

theorem normInfF_mulRqCoeffSpec_le_of_rawCoeffBound
  {a b : Coeffs} {k : Nat} {BRaw B : Nat}
  (hRaw : ∀ t, normInfF (mulRqRawCoeffSpec a b t) ≤ BRaw)
  (hAddSub : ∀ x y z, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF z ≤ BRaw → normInfF (x + y - z) ≤ B)
  (hSub : ∀ x y, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF (x - y) ≤ B) :
  normInfF (mulRqCoeffSpec a b k) ≤ B := by
  by_cases h25 : k ≤ 25
  · rw [mulRqCoeffSpec_of_le25_raw (a := a) (b := b) (k := k) h25]
    exact hAddSub
      (mulRqRawCoeffSpec a b k)
      (mulRqRawCoeffSpec a b (k + 81))
      (mulRqRawCoeffSpec a b (k + 54))
      (hRaw k)
      (hRaw (k + 81))
      (hRaw (k + 54))
  · by_cases h26 : k = 26
    · rw [mulRqCoeffSpec_of_eq26_raw (a := a) (b := b) (k := k) h26]
      exact hSub
        (mulRqRawCoeffSpec a b k)
        (mulRqRawCoeffSpec a b (k + 54))
        (hRaw k)
        (hRaw (k + 54))
    · have h26le : 26 ≤ k := Nat.succ_le_of_lt (Nat.lt_of_not_ge h25)
      have h26lt : 26 < k := Nat.lt_of_le_of_ne h26le (Ne.symm h26)
      have h27 : 27 ≤ k := Nat.succ_le_of_lt h26lt
      rw [mulRqCoeffSpec_of_ge27_raw (a := a) (b := b) (k := k) h27]
      exact hSub
        (mulRqRawCoeffSpec a b k)
        (mulRqRawCoeffSpec a b (k + 27))
        (hRaw k)
        (hRaw (k + 27))

theorem normInfF_mulRqRawCoeffSpec_le_of_inRangeBound
  {a b : Coeffs} {t BRaw : Nat}
  (hRawInRange : ∀ u, u < 2 * D - 1 → normInfF (mulRqRawCoeffSpec a b u) ≤ BRaw) :
  normInfF (mulRqRawCoeffSpec a b t) ≤ BRaw := by
  by_cases ht : t < 2 * D - 1
  · exact hRawInRange t ht
  · have hGe : 2 * D - 1 ≤ t := Nat.le_of_not_gt ht
    have hZero : mulRqRawCoeffSpec a b t = 0 := mulRqRawCoeffSpec_eq_zero_of_ge hGe
    have hZeroEq : normInfF (0 : F) = 0 := by
      native_decide
    have hZeroLe : normInfF (0 : F) ≤ BRaw := by
      simpa [hZeroEq] using (Nat.zero_le BRaw)
    simpa [hZero] using hZeroLe

theorem normInfF_mulRqRawCoeffSpec_le_of_rawCoeffsNorm
  {a b : Coeffs} {BRaw : Nat}
  (hRawCoeffs : normInfCoeffs (mulRqRawCoeffs a b) ≤ BRaw)
  (t : Nat) :
  normInfF (mulRqRawCoeffSpec a b t) ≤ BRaw := by
  have hGet : normInfF ((mulRqRawCoeffs a b)[t]!) ≤ BRaw :=
    normInfF_getElemBang_le_of_normInfCoeffs
      (a := mulRqRawCoeffs a b) (B := BRaw) hRawCoeffs t
  simpa [mulRqRawCoeffSpec_eq_rawCoeffs_getElemBang] using hGet

theorem normInfF_mulRqRawCoeffSpec_le_of_rawCoeffsNorm_inRange
  {a b : Coeffs} {BRaw : Nat}
  (hRawCoeffs : normInfCoeffs (mulRqRawCoeffs a b) ≤ BRaw) :
  ∀ t, t < 2 * D - 1 → normInfF (mulRqRawCoeffSpec a b t) ≤ BRaw := by
  intro t _ht
  exact normInfF_mulRqRawCoeffSpec_le_of_rawCoeffsNorm (a := a) (b := b) hRawCoeffs t

theorem normInfCoeffs_mulRqRawCoeffs_le_of_inRangeBound
  {a b : Coeffs} {BRaw : Nat}
  (hRawInRange : ∀ t, t < 2 * D - 1 → normInfF (mulRqRawCoeffSpec a b t) ≤ BRaw) :
  normInfCoeffs (mulRqRawCoeffs a b) ≤ BRaw := by
  apply normInfCoeffs_le_of_entry_bound
  intro i hi
  have hiRange : i < 2 * D - 1 := by
    simpa [mulRqRawCoeffs_size] using hi
  have hI : normInfF (mulRqRawCoeffSpec a b i) ≤ BRaw := hRawInRange i hiRange
  have hIBang : normInfF ((mulRqRawCoeffs a b)[i]!) ≤ BRaw := by
    simpa [mulRqRawCoeffSpec_eq_rawCoeffs_getElemBang] using hI
  simpa [hi] using hIBang

theorem normInf_mulRqRawCoeffSpec_inRange_iff_rawCoeffsNorm
  {a b : Coeffs} {BRaw : Nat} :
  (∀ t, t < 2 * D - 1 → normInfF (mulRqRawCoeffSpec a b t) ≤ BRaw) ↔
    normInfCoeffs (mulRqRawCoeffs a b) ≤ BRaw := by
  constructor
  · exact normInfCoeffs_mulRqRawCoeffs_le_of_inRangeBound
  · intro hRawCoeffs
    exact normInfF_mulRqRawCoeffSpec_le_of_rawCoeffsNorm_inRange hRawCoeffs

theorem normInfCoeffs_mulRq_le_of_rawCoeffBound
  {a b : Coeffs} {BRaw B : Nat}
  (hRaw : ∀ t, normInfF (mulRqRawCoeffSpec a b t) ≤ BRaw)
  (hAddSub : ∀ x y z, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF z ≤ BRaw → normInfF (x + y - z) ≤ B)
  (hSub : ∀ x y, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF (x - y) ≤ B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact normInfCoeffs_mulRq_le_of_coeffSpec_bound (a := a) (b := b) (B := B) (fun k hk => by
    exact normInfF_mulRqCoeffSpec_le_of_rawCoeffBound (a := a) (b := b) (k := k)
      hRaw hAddSub hSub)

theorem normInfCoeffs_mulRq_le_of_rawCoeffInRangeBound
  {a b : Coeffs} {BRaw B : Nat}
  (hRawInRange : ∀ t, t < 2 * D - 1 → normInfF (mulRqRawCoeffSpec a b t) ≤ BRaw)
  (hAddSub : ∀ x y z, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF z ≤ BRaw → normInfF (x + y - z) ≤ B)
  (hSub : ∀ x y, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF (x - y) ≤ B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact normInfCoeffs_mulRq_le_of_rawCoeffBound (a := a) (b := b) (BRaw := BRaw) (B := B)
    (hRaw := fun t =>
      normInfF_mulRqRawCoeffSpec_le_of_inRangeBound (a := a) (b := b) (t := t) hRawInRange)
    hAddSub hSub

theorem normInfCoeffs_mulRq_le_of_rawCoeffsNorm
  {a b : Coeffs} {BRaw B : Nat}
  (hRawCoeffs : normInfCoeffs (mulRqRawCoeffs a b) ≤ BRaw)
  (hAddSub : ∀ x y z, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF z ≤ BRaw → normInfF (x + y - z) ≤ B)
  (hSub : ∀ x y, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF (x - y) ≤ B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact normInfCoeffs_mulRq_le_of_rawCoeffBound (a := a) (b := b) (BRaw := BRaw) (B := B)
    (hRaw := fun t => normInfF_mulRqRawCoeffSpec_le_of_rawCoeffsNorm (a := a) (b := b) hRawCoeffs t)
    hAddSub hSub

theorem normInfCoeffs_mulRq_le_of_norm_bounds_via_rawCoeff
  {a b : Coeffs} {BA BB BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawFromNorm :
    ∀ t, normInfCoeffs a ≤ BA → normInfCoeffs b ≤ BB → normInfF (mulRqRawCoeffSpec a b t) ≤ BRaw)
  (hAddSub : ∀ x y z, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF z ≤ BRaw → normInfF (x + y - z) ≤ B)
  (hSub : ∀ x y, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF (x - y) ≤ B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact normInfCoeffs_mulRq_le_of_rawCoeffBound (a := a) (b := b) (BRaw := BRaw) (B := B)
    (hRaw := fun t => hRawFromNorm t hA hB)
    hAddSub hSub

theorem normInfCoeffs_mulRq_le_of_norm_bounds_via_rawCoeff_inRange
  {a b : Coeffs} {BA BB BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawFromNormInRange :
    ∀ t, t < 2 * D - 1 →
      normInfCoeffs a ≤ BA →
      normInfCoeffs b ≤ BB →
      normInfF (mulRqRawCoeffSpec a b t) ≤ BRaw)
  (hAddSub : ∀ x y z, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF z ≤ BRaw → normInfF (x + y - z) ≤ B)
  (hSub : ∀ x y, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF (x - y) ≤ B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact normInfCoeffs_mulRq_le_of_rawCoeffInRangeBound (a := a) (b := b) (BRaw := BRaw) (B := B)
    (hRawInRange := fun t ht => hRawFromNormInRange t ht hA hB)
    hAddSub hSub

theorem normInfCoeffs_mulRq_le_of_norm_bounds_via_rawCoeffsNorm
  {a b : Coeffs} {BA BB BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawCoeffsFromNorm :
    normInfCoeffs a ≤ BA →
    normInfCoeffs b ≤ BB →
    normInfCoeffs (mulRqRawCoeffs a b) ≤ BRaw)
  (hAddSub : ∀ x y z, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF z ≤ BRaw → normInfF (x + y - z) ≤ B)
  (hSub : ∀ x y, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF (x - y) ≤ B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact normInfCoeffs_mulRq_le_of_rawCoeffsNorm (a := a) (b := b) (BRaw := BRaw) (B := B)
    (hRawCoeffs := hRawCoeffsFromNorm hA hB)
    hAddSub hSub

private theorem normInfF_foldl_add_if_le_of_term_bound
  {α : Type}
  (xs : List α)
  (p : α → Prop)
  [DecidablePred p]
  (term : α → F)
  {init : F}
  {BAcc BTerm : Nat}
  (hInit : normInfF init ≤ BAcc)
  (hAdd : ∀ x y : F, normInfF x ≤ BAcc → normInfF y ≤ BTerm → normInfF (x + y) ≤ BAcc)
  (hTerm : ∀ x, x ∈ xs → p x → normInfF (term x) ≤ BTerm) :
  normInfF (List.foldl (fun acc x => if p x then acc + term x else acc) init xs) ≤ BAcc := by
  induction xs generalizing init with
  | nil =>
      simpa using hInit
  | cons x xs ih =>
      by_cases hp : p x
      · have hTermX : normInfF (term x) ≤ BTerm := hTerm x (by simp) hp
        have hInit' : normInfF (init + term x) ≤ BAcc := hAdd init (term x) hInit hTermX
        have hTail :
            normInfF (List.foldl (fun acc x => if p x then acc + term x else acc) (init + term x) xs) ≤ BAcc := by
          apply ih
          · exact hInit'
          · intro x' hx' hp'
            exact hTerm x' (by simp [hx']) hp'
        simpa [List.foldl_cons, hp] using hTail
      · have hTail :
            normInfF (List.foldl (fun acc x => if p x then acc + term x else acc) init xs) ≤ BAcc := by
          apply ih
          · exact hInit
          · intro x' hx' hp'
            exact hTerm x' (by simp [hx']) hp'
        simpa [List.foldl_cons, hp] using hTail

private theorem normInfF_schoolbook_row_scalar_fold_le_of_term_bound
  {aD bD : Coeffs}
  {i k : Nat}
  {acc : F}
  {BAcc BTerm : Nat}
  (hAcc : normInfF acc ≤ BAcc)
  (hAdd : ∀ x y : F, normInfF x ≤ BAcc → normInfF y ≤ BTerm → normInfF (x + y) ≤ BAcc)
  (hTerm : ∀ j, j < D → normInfF (aD[i]! * bD[j]!) ≤ BTerm) :
  normInfF
    (List.foldl
      (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
      acc
      (List.range' 0 D))
    ≤ BAcc := by
  apply normInfF_foldl_add_if_le_of_term_bound
    (xs := List.range' 0 D)
    (p := fun j => i + j = k)
    (term := fun j => aD[i]! * bD[j]!)
  · exact hAcc
  · exact hAdd
  · intro j hj _hp
    rcases (List.mem_range').1 hj with ⟨t, ht, hEq⟩
    have hEq' : j = t := by simpa using hEq
    exact hTerm j (by simpa [hEq'] using ht)

private theorem normInfF_schoolbook_scalar_fold_le_of_term_bound
  {aD bD : Coeffs}
  {k : Nat}
  {BAcc BTerm : Nat}
  (hInit : normInfF (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k) ≤ BAcc)
  (hAdd : ∀ x y : F, normInfF x ≤ BAcc → normInfF y ≤ BTerm → normInfF (x + y) ≤ BAcc)
  (hTerm : ∀ i j, i < D → j < D → normInfF (aD[i]! * bD[j]!) ≤ BTerm) :
  normInfF
    (List.foldl
      (fun acc i =>
        List.foldl
          (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
          acc
          (List.range' 0 D))
      (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
      (List.range' 0 D))
    ≤ BAcc := by
  let outerStep : F → Nat → F :=
    fun acc i =>
      List.foldl
        (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
        acc
        (List.range' 0 D)
  have hOuter :
      ∀ outer init,
        (∀ i, i ∈ outer → i < D) →
        normInfF init ≤ BAcc →
        normInfF (List.foldl outerStep init outer) ≤ BAcc := by
    intro outer
    induction outer with
    | nil =>
        intro init _hIn hInitAcc
        simpa using hInitAcc
    | cons i is ih =>
        intro init hIn hInitAcc
        have hiD : i < D := hIn i (by simp)
        have hRow :
            normInfF (outerStep init i) ≤ BAcc := by
          apply normInfF_schoolbook_row_scalar_fold_le_of_term_bound
            (aD := aD) (bD := bD) (i := i) (k := k) (acc := init)
          · exact hInitAcc
          · exact hAdd
          · intro j hj
            exact hTerm i j hiD hj
        have hTail :
            normInfF (List.foldl outerStep (outerStep init i) is) ≤ BAcc := by
          apply ih
          · intro i' hi'
            exact hIn i' (by simp [hi'])
          · exact hRow
        simpa [List.foldl_cons] using hTail
  have hRange : ∀ i, i ∈ List.range' 0 D → i < D := by
    intro i hi
    rcases (List.mem_range').1 hi with ⟨t, ht, hEq⟩
    have hEq' : i = t := by simpa using hEq
    simpa [hEq'] using ht
  simpa [outerStep] using
    hOuter
      (outer := List.range' 0 D)
      (init := getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
      hRange
      hInit

/--
Fold helper (sum-style): each selected term is bounded by `BTerm`, and addition
uses a triangle-style bound, so the final accumulator is bounded by
`init + (#selected steps upper bounded by xs.length) * BTerm`.
-/
private theorem normInfF_foldl_add_if_le_of_term_bound_sum
  {α : Type}
  (xs : List α)
  (p : α → Prop)
  [DecidablePred p]
  (term : α → F)
  {init : F}
  {BTerm : Nat}
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y)
  (hTerm : ∀ x, x ∈ xs → p x → normInfF (term x) ≤ BTerm) :
  normInfF (List.foldl (fun acc x => if p x then acc + term x else acc) init xs)
    ≤ normInfF init + xs.length * BTerm := by
  induction xs generalizing init with
  | nil =>
      simp
  | cons x xs ih =>
      by_cases hp : p x
      · have hTermX : normInfF (term x) ≤ BTerm := hTerm x (by simp) hp
        have hHead : normInfF (init + term x) ≤ normInfF init + BTerm := by
          exact Nat.le_trans
            (hAddTri init (term x))
            (Nat.add_le_add_left hTermX (normInfF init))
        have hTail :
            normInfF (List.foldl (fun acc x => if p x then acc + term x else acc) (init + term x) xs)
              ≤ normInfF (init + term x) + xs.length * BTerm := by
          apply ih
          intro x' hx' hp'
          exact hTerm x' (by simp [hx']) hp'
        have hTail' :
            normInfF (List.foldl (fun acc x => if p x then acc + term x else acc) (init + term x) xs)
              ≤ (normInfF init + BTerm) + xs.length * BTerm := by
          exact Nat.le_trans hTail (Nat.add_le_add_right hHead (xs.length * BTerm))
        simpa [List.foldl_cons, hp, Nat.succ_mul, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using hTail'
      · have hTail :
            normInfF (List.foldl (fun acc x => if p x then acc + term x else acc) init xs)
              ≤ normInfF init + xs.length * BTerm := by
          apply ih
          intro x' hx' hp'
          exact hTerm x' (by simp [hx']) hp'
        have hGrow :
            normInfF init + xs.length * BTerm
              ≤ normInfF init + (Nat.succ xs.length) * BTerm := by
          exact Nat.add_le_add_left
            (Nat.mul_le_mul_right BTerm (Nat.le_succ xs.length))
            (normInfF init)
        have hFinal :
            normInfF (List.foldl (fun acc x => if p x then acc + term x else acc) init (x :: xs))
              ≤ normInfF init + (x :: xs).length * BTerm := by
          exact Nat.le_trans (by simpa [List.foldl_cons, hp] using hTail) (by simpa using hGrow)
        exact hFinal

/--
Count-refined fold helper: growth is proportional to the number of selected
(`p = true`) terms, not to the full list length.
-/
private theorem normInfF_foldl_add_if_le_of_term_bound_count
  {α : Type}
  (xs : List α)
  (p : α → Bool)
  (term : α → F)
  {init : F}
  {BTerm : Nat}
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y)
  (hTerm : ∀ x, x ∈ xs → p x = true → normInfF (term x) ≤ BTerm) :
  normInfF (List.foldl (fun acc x => if p x then acc + term x else acc) init xs)
    ≤ normInfF init + (List.countP p xs) * BTerm := by
  induction xs generalizing init with
  | nil =>
      simp
  | cons x xs ih =>
      by_cases hp : p x = true
      · have hTermX : normInfF (term x) ≤ BTerm := hTerm x (by simp) hp
        have hHead : normInfF (init + term x) ≤ normInfF init + BTerm := by
          exact Nat.le_trans
            (hAddTri init (term x))
            (Nat.add_le_add_left hTermX (normInfF init))
        have hTail :
            normInfF (List.foldl (fun acc x => if p x then acc + term x else acc) (init + term x) xs)
              ≤ normInfF (init + term x) + (List.countP p xs) * BTerm := by
          apply ih
          intro x' hx' hp'
          exact hTerm x' (by simp [hx']) hp'
        have hTail' :
            normInfF (List.foldl (fun acc x => if p x then acc + term x else acc) (init + term x) xs)
              ≤ (normInfF init + BTerm) + (List.countP p xs) * BTerm := by
          exact Nat.le_trans hTail (Nat.add_le_add_right hHead ((List.countP p xs) * BTerm))
        calc
          normInfF (List.foldl (fun acc x => if p x then acc + term x else acc) init (x :: xs))
              = normInfF (List.foldl (fun acc x => if p x then acc + term x else acc) (init + term x) xs) := by
                  simp [List.foldl_cons, hp]
          _ ≤ (normInfF init + BTerm) + (List.countP p xs) * BTerm := hTail'
          _ = normInfF init + (List.countP p (x :: xs)) * BTerm := by
                rw [List.countP_cons, hp]
                simp [Nat.succ_mul, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
      · have hTail :
            normInfF (List.foldl (fun acc x => if p x then acc + term x else acc) init xs)
              ≤ normInfF init + (List.countP p xs) * BTerm := by
          apply ih
          intro x' hx' hp'
          exact hTerm x' (by simp [hx']) hp'
        calc
          normInfF (List.foldl (fun acc x => if p x then acc + term x else acc) init (x :: xs))
              = normInfF (List.foldl (fun acc x => if p x then acc + term x else acc) init xs) := by
                  simp [List.foldl_cons, hp]
          _ ≤ normInfF init + (List.countP p xs) * BTerm := hTail
          _ = normInfF init + (List.countP p (x :: xs)) * BTerm := by
                have hpFalse : p x = false := by
                  cases hpx : p x with
                  | false => rfl
                  | true => exact (hp hpx).elim
                rw [List.countP_cons, hpFalse]
                simp

/-- Fold helper for step functions that increase norm by at most a fixed increment. -/
private theorem normInfF_foldl_step_le_of_step_increment
  {α : Type}
  (xs : List α)
  (step : F → α → F)
  {init : F}
  {C : Nat}
  (hStep : ∀ acc x, x ∈ xs → normInfF (step acc x) ≤ normInfF acc + C) :
  normInfF (List.foldl step init xs) ≤ normInfF init + xs.length * C := by
  induction xs generalizing init with
  | nil =>
      simp
  | cons x xs ih =>
      have hHead : normInfF (step init x) ≤ normInfF init + C := hStep init x (by simp)
      have hTail :
          normInfF (List.foldl step (step init x) xs)
            ≤ normInfF (step init x) + xs.length * C := by
        apply ih
        intro acc x' hx'
        exact hStep acc x' (by simp [hx'])
      have hTail' :
          normInfF (List.foldl step (step init x) xs)
            ≤ (normInfF init + C) + xs.length * C := by
        exact Nat.le_trans hTail (Nat.add_le_add_right hHead (xs.length * C))
      simpa [List.foldl_cons, Nat.succ_mul, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using hTail'

private theorem schoolbook_row_hit_count_le_one (i k : Nat) :
  List.countP (fun j : Nat => decide (i + j = k)) (List.range' 0 D) ≤ 1 := by
  by_cases hik : i ≤ k
  · have hCountEq :
      List.countP (fun j : Nat => decide (i + j = k)) (List.range' 0 D)
        = List.count (k - i) (List.range' 0 D) := by
      have hCong :
          ∀ j, j ∈ List.range' 0 D →
            (decide (i + j = k) = true ↔ decide (j = k - i) = true) := by
        intro j _hj
        constructor
        · intro hDec
          have hEq : i + j = k := by simpa using hDec
          have hSub : j = k - i := by
            have hK : i + (k - i) = k := Nat.add_sub_of_le hik
            exact Nat.add_left_cancel (hEq.trans hK.symm)
          simpa [hSub]
        · intro hDec
          have hSub : j = k - i := by simpa using hDec
          have hEq : i + j = k := by
            calc
              i + j = i + (k - i) := by simpa [hSub]
              _ = k := Nat.add_sub_of_le hik
          simpa [hEq]
      have hCp :
          List.countP (fun j : Nat => decide (i + j = k)) (List.range' 0 D)
            = List.countP (fun j : Nat => decide (j = k - i)) (List.range' 0 D) :=
        List.countP_congr (l := List.range' 0 D) hCong
      have hCountP :
          List.countP (fun j : Nat => decide (j = k - i)) (List.range' 0 D)
            = List.count (k - i) (List.range' 0 D) := by
        simpa using (List.count_eq_countP (a := (k - i)) (l := List.range' 0 D)).symm
      exact hCp.trans hCountP
    have hLeCount : List.count (k - i) (List.range' 0 D) ≤ 1 := by
      rw [List.count_range_1' (a := k - i) (s := 0) (n := D)]
      split <;> simp
    simpa [hCountEq] using hLeCount
  · have hZero : List.countP (fun j : Nat => decide (i + j = k)) (List.range' 0 D) = 0 := by
      apply (List.countP_eq_zero).2
      intro j hj hDec
      have hkLt : k < i := Nat.lt_of_not_ge hik
      have hEq : i + j = k := by simpa using hDec
      have hLe : i ≤ k := by
        calc
          i ≤ i + j := Nat.le_add_right i j
          _ = k := hEq
      exact (Nat.not_lt_of_ge hLe) hkLt
    simpa [hZero]

private theorem normInfF_schoolbook_row_scalar_fold_le_of_term_bound_sum
  {aD bD : Coeffs}
  {i k : Nat}
  {acc : F}
  {BTerm : Nat}
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y)
  (hTerm : ∀ j, j < D → normInfF (aD[i]! * bD[j]!) ≤ BTerm) :
  normInfF
    (List.foldl
      (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
      acc
      (List.range' 0 D))
    ≤ normInfF acc + D * BTerm := by
  have hFold :
      normInfF
        (List.foldl
          (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
          acc
          (List.range' 0 D))
        ≤ normInfF acc + (List.range' 0 D).length * BTerm := by
    apply normInfF_foldl_add_if_le_of_term_bound_sum
      (xs := List.range' 0 D)
      (p := fun j => i + j = k)
      (term := fun j => aD[i]! * bD[j]!)
      (init := acc)
      (BTerm := BTerm)
      hAddTri
    intro j hj _hp
    rcases (List.mem_range').1 hj with ⟨t, ht, hEq⟩
    have hEq' : j = t := by simpa using hEq
    exact hTerm j (by simpa [hEq'] using ht)
  simpa using hFold

private theorem normInfF_schoolbook_row_scalar_fold_le_of_term_bound_sum_tight
  {aD bD : Coeffs}
  {i k : Nat}
  {acc : F}
  {BTerm : Nat}
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y)
  (hTerm : ∀ j, j < D → normInfF (aD[i]! * bD[j]!) ≤ BTerm) :
  normInfF
    (List.foldl
      (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
      acc
      (List.range' 0 D))
    ≤ normInfF acc + BTerm := by
  let p : Nat → Bool := fun j => decide (i + j = k)
  let step : F → Nat → F := fun acc' j => if p j then acc' + aD[i]! * bD[j]! else acc'
  have hFold :
      normInfF (List.foldl step acc (List.range' 0 D))
        ≤ normInfF acc + (List.countP p (List.range' 0 D)) * BTerm := by
    apply normInfF_foldl_add_if_le_of_term_bound_count
      (xs := List.range' 0 D)
      (p := p)
      (term := fun j => aD[i]! * bD[j]!)
      (init := acc)
      (BTerm := BTerm)
      hAddTri
    intro j hj hp
    rcases (List.mem_range').1 hj with ⟨t, ht, hEq⟩
    have hEq' : j = t := by simpa using hEq
    exact hTerm j (by simpa [hEq'] using ht)
  have hCountLe : List.countP p (List.range' 0 D) ≤ 1 := by
    simpa [p] using schoolbook_row_hit_count_le_one i k
  have hMulLe : (List.countP p (List.range' 0 D)) * BTerm ≤ 1 * BTerm :=
    Nat.mul_le_mul_right BTerm hCountLe
  have hBound :
      normInfF (List.foldl step acc (List.range' 0 D))
        ≤ normInfF acc + 1 * BTerm := by
    exact Nat.le_trans hFold (Nat.add_le_add_left hMulLe (normInfF acc))
  simpa [step, p, Nat.one_mul] using hBound

private theorem normInfF_schoolbook_scalar_fold_le_of_term_bound_sum
  {aD bD : Coeffs}
  {k : Nat}
  {BTerm : Nat}
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y)
  (hTerm : ∀ i j, i < D → j < D → normInfF (aD[i]! * bD[j]!) ≤ BTerm) :
  normInfF
    (List.foldl
      (fun acc i =>
        List.foldl
          (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
          acc
          (List.range' 0 D))
      (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
      (List.range' 0 D))
    ≤ (D * D) * BTerm := by
  let outerStep : F → Nat → F :=
    fun acc i =>
      List.foldl
        (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
        acc
        (List.range' 0 D)
  have hOuter :
      normInfF
        (List.foldl
          outerStep
          (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
          (List.range' 0 D))
      ≤ normInfF (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
          + (List.range' 0 D).length * (D * BTerm) := by
    apply normInfF_foldl_step_le_of_step_increment
      (xs := List.range' 0 D)
      (step := outerStep)
      (init := getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
      (C := D * BTerm)
    intro acc i hi
    rcases (List.mem_range').1 hi with ⟨t, ht, hEq⟩
    have hiD : i < D := by
      simpa [hEq] using ht
    have hRow :
        normInfF (outerStep acc i) ≤ normInfF acc + D * BTerm := by
      apply normInfF_schoolbook_row_scalar_fold_le_of_term_bound_sum
        (aD := aD) (bD := bD) (i := i) (k := k)
      · exact hAddTri
      · intro j hj
        exact hTerm i j hiD hj
    simpa [outerStep] using hRow
  have hInitZero : normInfF (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k) = 0 := by
    have hCoeffZero : getCoeff (Array.replicate (2 * D - 1) (0 : F)) k = 0 :=
      getCoeff_replicate_zero (2 * D - 1) k
    have hZero : normInfF (0 : F) = 0 := by
      native_decide
    simpa [hCoeffZero] using hZero
  calc
    normInfF
      (List.foldl
        (fun acc i =>
          List.foldl
            (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
            acc
            (List.range' 0 D))
        (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
        (List.range' 0 D))
        ≤ normInfF (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
            + (List.range' 0 D).length * (D * BTerm) := hOuter
    _ = (D * D) * BTerm := by
          simpa [hInitZero, Nat.mul_assoc]

private theorem normInfF_schoolbook_scalar_fold_le_of_term_bound_sum_tight
  {aD bD : Coeffs}
  {k : Nat}
  {BTerm : Nat}
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y)
  (hTerm : ∀ i j, i < D → j < D → normInfF (aD[i]! * bD[j]!) ≤ BTerm) :
  normInfF
    (List.foldl
      (fun acc i =>
        List.foldl
          (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
          acc
          (List.range' 0 D))
      (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
      (List.range' 0 D))
    ≤ D * BTerm := by
  let outerStep : F → Nat → F :=
    fun acc i =>
      List.foldl
        (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
        acc
        (List.range' 0 D)
  have hOuter :
      normInfF
        (List.foldl
          outerStep
          (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
          (List.range' 0 D))
      ≤ normInfF (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
          + (List.range' 0 D).length * BTerm := by
    apply normInfF_foldl_step_le_of_step_increment
      (xs := List.range' 0 D)
      (step := outerStep)
      (init := getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
      (C := BTerm)
    intro acc i hi
    rcases (List.mem_range').1 hi with ⟨t, ht, hEq⟩
    have hiD : i < D := by
      simpa [hEq] using ht
    have hRow :
        normInfF (outerStep acc i) ≤ normInfF acc + BTerm := by
      apply normInfF_schoolbook_row_scalar_fold_le_of_term_bound_sum_tight
        (aD := aD) (bD := bD) (i := i) (k := k)
      · exact hAddTri
      · intro j hj
        exact hTerm i j hiD hj
    simpa [outerStep] using hRow
  have hInitZero : normInfF (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k) = 0 := by
    have hCoeffZero : getCoeff (Array.replicate (2 * D - 1) (0 : F)) k = 0 :=
      getCoeff_replicate_zero (2 * D - 1) k
    have hZero : normInfF (0 : F) = 0 := by
      native_decide
    simpa [hCoeffZero] using hZero
  calc
    normInfF
      (List.foldl
        (fun acc i =>
          List.foldl
            (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
            acc
            (List.range' 0 D))
        (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
        (List.range' 0 D))
        ≤ normInfF (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
            + (List.range' 0 D).length * BTerm := hOuter
    _ = D * BTerm := by
          simpa [hInitZero]

/--
Schoolbook (raw) coefficient bound from operand coefficient bounds, via:
1) a per-term multiplication bound (`hMul`), and
2) a triangle-style add bound (`hAddTri`).

This avoids fixed-point accumulator closure and yields a concrete bound
`(D * D) * BTerm`.
-/
theorem normInfF_mulRqRawCoeffSpec_le_of_operand_norm_assumptions_via_schoolbook_sum
  {a b : Coeffs}
  {k BA BB BTerm : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y) :
  normInfF (mulRqRawCoeffSpec a b k) ≤ (D * D) * BTerm := by
  let aD : Coeffs := Array.ofFn (fun i : Fin D => a[i.1]!)
  let bD : Coeffs := Array.ofFn (fun i : Fin D => b[i.1]!)
  have hTerm :
      ∀ i j, i < D → j < D → normInfF (aD[i]! * bD[j]!) ≤ BTerm := by
    intro i j hi hj
    have hAiRaw : normInfF (a[i]!) ≤ BA :=
      normInfF_getElemBang_le_of_normInfCoeffs (a := a) (B := BA) hA i
    have hBjRaw : normInfF (b[j]!) ≤ BB :=
      normInfF_getElemBang_le_of_normInfCoeffs (a := b) (B := BB) hB j
    have hAi : normInfF (aD[i]!) ≤ BA := by
      simpa [aD, hi] using hAiRaw
    have hBj : normInfF (bD[j]!) ≤ BB := by
      simpa [bD, hj] using hBjRaw
    exact hMul (aD[i]!) (bD[j]!) hAi hBj
  have hFold :
      normInfF
        (List.foldl
          (fun acc i =>
            List.foldl
              (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
              acc
              (List.range' 0 D))
          (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
          (List.range' 0 D))
        ≤ (D * D) * BTerm := by
    exact normInfF_schoolbook_scalar_fold_le_of_term_bound_sum
      (aD := aD) (bD := bD) (k := k)
      (hAddTri := hAddTri)
      (hTerm := hTerm)
  have hEq :
      mulRqRawCoeffSpec a b k
        =
        List.foldl
          (fun acc i =>
            List.foldl
              (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
              acc
              (List.range' 0 D))
          (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
          (List.range' 0 D) := by
    simpa [aD, bD] using (mulRqRawCoeffSpec_eq_scalar_fold a b k)
  calc
    normInfF (mulRqRawCoeffSpec a b k)
        = normInfF
            (List.foldl
              (fun acc i =>
                List.foldl
                  (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
                  acc
                  (List.range' 0 D))
              (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
              (List.range' 0 D)) := by
                simpa [hEq]
    _ ≤ (D * D) * BTerm := hFold

theorem normInfF_mulRqRawCoeffSpec_le_of_operand_norm_assumptions_via_schoolbook_sum_tight
  {a b : Coeffs}
  {k BA BB BTerm : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y) :
  normInfF (mulRqRawCoeffSpec a b k) ≤ D * BTerm := by
  let aD : Coeffs := Array.ofFn (fun i : Fin D => a[i.1]!)
  let bD : Coeffs := Array.ofFn (fun i : Fin D => b[i.1]!)
  have hTerm :
      ∀ i j, i < D → j < D → normInfF (aD[i]! * bD[j]!) ≤ BTerm := by
    intro i j hi hj
    have hAiRaw : normInfF (a[i]!) ≤ BA :=
      normInfF_getElemBang_le_of_normInfCoeffs (a := a) (B := BA) hA i
    have hBjRaw : normInfF (b[j]!) ≤ BB :=
      normInfF_getElemBang_le_of_normInfCoeffs (a := b) (B := BB) hB j
    have hAi : normInfF (aD[i]!) ≤ BA := by
      simpa [aD, hi] using hAiRaw
    have hBj : normInfF (bD[j]!) ≤ BB := by
      simpa [bD, hj] using hBjRaw
    exact hMul (aD[i]!) (bD[j]!) hAi hBj
  have hFold :
      normInfF
        (List.foldl
          (fun acc i =>
            List.foldl
              (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
              acc
              (List.range' 0 D))
          (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
          (List.range' 0 D))
        ≤ D * BTerm := by
    exact normInfF_schoolbook_scalar_fold_le_of_term_bound_sum_tight
      (aD := aD) (bD := bD) (k := k)
      (hAddTri := hAddTri)
      (hTerm := hTerm)
  have hEq :
      mulRqRawCoeffSpec a b k
        =
        List.foldl
          (fun acc i =>
            List.foldl
              (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
              acc
              (List.range' 0 D))
          (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
          (List.range' 0 D) := by
    simpa [aD, bD] using (mulRqRawCoeffSpec_eq_scalar_fold a b k)
  calc
    normInfF (mulRqRawCoeffSpec a b k)
        = normInfF
            (List.foldl
              (fun acc i =>
                List.foldl
                  (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
                  acc
                  (List.range' 0 D))
              (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
              (List.range' 0 D)) := by
                simpa [hEq]
    _ ≤ D * BTerm := hFold

theorem mulRqRawCoeffBoundFromOperands_fn_of_schoolbook_term_assumptions_sum
  {BA BB BTerm : Nat}
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y) :
  ∀ a b : Coeffs, ∀ t : Nat,
    normInfCoeffs a ≤ BA →
    normInfCoeffs b ≤ BB →
    normInfF (mulRqRawCoeffSpec a b t) ≤ (D * D) * BTerm := by
  intro a b t hA hB
  exact normInfF_mulRqRawCoeffSpec_le_of_operand_norm_assumptions_via_schoolbook_sum
    (a := a) (b := b) (k := t)
    (hA := hA) (hB := hB)
    (hMul := hMul) (hAddTri := hAddTri)

theorem mulRqRawCoeffBoundFromOperands_fn_of_schoolbook_term_assumptions_sum_tight
  {BA BB BTerm : Nat}
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y) :
  ∀ a b : Coeffs, ∀ t : Nat,
    normInfCoeffs a ≤ BA →
    normInfCoeffs b ≤ BB →
    normInfF (mulRqRawCoeffSpec a b t) ≤ D * BTerm := by
  intro a b t hA hB
  exact normInfF_mulRqRawCoeffSpec_le_of_operand_norm_assumptions_via_schoolbook_sum_tight
    (a := a) (b := b) (k := t)
    (hA := hA) (hB := hB)
    (hMul := hMul) (hAddTri := hAddTri)

/--
Schoolbook (raw) coefficient bound from operand coefficient bounds, via:
1) a per-term multiplication bound (`hMul`), and
2) an accumulator-stable addition bound (`hAdd`).

This is a theorem-native bridge toward non-coarse P5 raw bounds.
-/
theorem normInfF_mulRqRawCoeffSpec_le_of_operand_norm_assumptions_via_schoolbook
  {a b : Coeffs}
  {k BA BB BTerm BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAdd : ∀ x y : F, normInfF x ≤ BRaw → normInfF y ≤ BTerm → normInfF (x + y) ≤ BRaw)
  (hZero : normInfF (0 : F) ≤ BRaw) :
  normInfF (mulRqRawCoeffSpec a b k) ≤ BRaw := by
  let aD : Coeffs := Array.ofFn (fun i : Fin D => a[i.1]!)
  let bD : Coeffs := Array.ofFn (fun i : Fin D => b[i.1]!)
  have hTerm :
      ∀ i j, i < D → j < D → normInfF (aD[i]! * bD[j]!) ≤ BTerm := by
    intro i j hi hj
    have hAiRaw : normInfF (a[i]!) ≤ BA :=
      normInfF_getElemBang_le_of_normInfCoeffs (a := a) (B := BA) hA i
    have hBjRaw : normInfF (b[j]!) ≤ BB :=
      normInfF_getElemBang_le_of_normInfCoeffs (a := b) (B := BB) hB j
    have hAi : normInfF (aD[i]!) ≤ BA := by
      simpa [aD, hi] using hAiRaw
    have hBj : normInfF (bD[j]!) ≤ BB := by
      simpa [bD, hj] using hBjRaw
    exact hMul (aD[i]!) (bD[j]!) hAi hBj
  have hInit : normInfF (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k) ≤ BRaw := by
    have hCoeffZero : getCoeff (Array.replicate (2 * D - 1) (0 : F)) k = 0 :=
      getCoeff_replicate_zero (2 * D - 1) k
    simpa [hCoeffZero] using hZero
  have hFold :
      normInfF
        (List.foldl
          (fun acc i =>
            List.foldl
              (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
              acc
              (List.range' 0 D))
          (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
          (List.range' 0 D))
        ≤ BRaw := by
    exact normInfF_schoolbook_scalar_fold_le_of_term_bound
      (aD := aD) (bD := bD) (k := k)
      (hInit := hInit)
      (hAdd := hAdd)
      (hTerm := hTerm)
  have hEq :
      mulRqRawCoeffSpec a b k
        =
        List.foldl
          (fun acc i =>
            List.foldl
              (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
              acc
              (List.range' 0 D))
          (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
          (List.range' 0 D) := by
    simpa [aD, bD] using (mulRqRawCoeffSpec_eq_scalar_fold a b k)
  calc
    normInfF (mulRqRawCoeffSpec a b k)
        = normInfF
            (List.foldl
              (fun acc i =>
                List.foldl
                  (fun acc' j => if i + j = k then acc' + aD[i]! * bD[j]! else acc')
                  acc
                  (List.range' 0 D))
              (getCoeff (Array.replicate (2 * D - 1) (0 : F)) k)
              (List.range' 0 D)) := by
                simpa [hEq]
    _ ≤ BRaw := hFold

theorem mulRqRawCoeffBoundFromOperands_fn_of_schoolbook_term_assumptions
  {BA BB BTerm BRaw : Nat}
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAdd : ∀ x y : F, normInfF x ≤ BRaw → normInfF y ≤ BTerm → normInfF (x + y) ≤ BRaw)
  (hZero : normInfF (0 : F) ≤ BRaw) :
  ∀ a b : Coeffs, ∀ t : Nat,
    normInfCoeffs a ≤ BA →
    normInfCoeffs b ≤ BB →
    normInfF (mulRqRawCoeffSpec a b t) ≤ BRaw := by
  intro a b t hA hB
  exact normInfF_mulRqRawCoeffSpec_le_of_operand_norm_assumptions_via_schoolbook
    (a := a) (b := b) (k := t)
    (hA := hA) (hB := hB)
    (hMul := hMul) (hAdd := hAdd) (hZero := hZero)

/--
Assumption bundle: operand norm bounds imply a bound on the raw schoolbook product norm.
-/
def mulRqRawNormBoundFromOperands (BA BB BRaw : Nat) : Prop :=
  ∀ a b : Coeffs,
    normInfCoeffs a ≤ BA →
    normInfCoeffs b ≤ BB →
    normInfCoeffs (mulRqRawCoeffs a b) ≤ BRaw

/--
Assumption bundle: operand norm bounds imply an in-range raw coefficient bound.
-/
def mulRqRawInRangeBoundFromOperands (BA BB BRaw : Nat) : Prop :=
  ∀ a b : Coeffs, ∀ t, t < 2 * D - 1 →
    normInfCoeffs a ≤ BA →
    normInfCoeffs b ≤ BB →
    normInfF (mulRqRawCoeffSpec a b t) ≤ BRaw

/--
Assumption bundle: operand norm bounds imply a bound on every raw schoolbook
coefficient accessor (all Nat indices, out-of-range included via default semantics).
-/
def mulRqRawCoeffBoundFromOperands (BA BB BRaw : Nat) : Prop :=
  ∀ a b : Coeffs, ∀ t : Nat,
    normInfCoeffs a ≤ BA →
    normInfCoeffs b ≤ BB →
    normInfF (mulRqRawCoeffSpec a b t) ≤ BRaw

/-- Assumption bundle for per-term schoolbook multiplication bounds. -/
def schoolbookMulTermBound (BA BB BTerm : Nat) : Prop :=
  ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm

/-- Assumption bundle for accumulator-stable addition in schoolbook folds. -/
def schoolbookAccAddBound (BRaw BTerm : Nat) : Prop :=
  ∀ x y : F, normInfF x ≤ BRaw → normInfF y ≤ BTerm → normInfF (x + y) ≤ BRaw

/-- Assumption bundle for triangle-style additive growth in schoolbook folds. -/
def schoolbookAddTriangleBound : Prop :=
  ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y

theorem schoolbookAddTriangleBound_theorem : schoolbookAddTriangleBound := by
  intro x y
  exact normInfF_add_le_add_theorem x y

/--
Universal multiplication blocker for a theorem-native non-coarse P5 path.
If this is proved, per-term schoolbook multiplication bounds can be derived
directly from operand norm bounds.
-/
def schoolbookMulUniversalBound : Prop :=
  ∀ x y : F, normInfF (x * y) ≤ normInfF x * normInfF y

/--
Universal subtraction-triangle blocker for a theorem-native non-coarse P5 path.
If this is proved, subtraction collapse assumptions can be derived directly.
-/
def schoolbookSubTriangleBound : Prop :=
  ∀ x y : F, normInfF (x - y) ≤ normInfF x + normInfF y

theorem schoolbookAddTriangleBound_iff_centeredRep :
  schoolbookAddTriangleBound ↔
    (∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) := by
  constructor
  · intro h x y
    simpa [centeredRep_natAbs_eq_normInfF] using h x y
  · intro h x y
    simpa [centeredRep_natAbs_eq_normInfF] using h x y

theorem schoolbookMulUniversalBound_iff_centeredRep :
  schoolbookMulUniversalBound ↔
    (∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y)) := by
  constructor
  · intro h x y
    simpa [centeredRep_natAbs_eq_normInfF] using h x y
  · intro h x y
    simpa [centeredRep_natAbs_eq_normInfF] using h x y

theorem schoolbookSubTriangleBound_iff_centeredRep :
  schoolbookSubTriangleBound ↔
    (∀ x y : F, Int.natAbs (centeredRep (x - y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) := by
  constructor
  · intro h x y
    simpa [centeredRep_natAbs_eq_normInfF] using h x y
  · intro h x y
    simpa [centeredRep_natAbs_eq_normInfF] using h x y

theorem schoolbookAddTriangleBound_of_centeredRep
  (h :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  schoolbookAddTriangleBound := by
  exact (schoolbookAddTriangleBound_iff_centeredRep).2 h

theorem schoolbookAddTriangleBound_of_centeredRepAddTriangleBound
  (h : centeredRepAddTriangleBound) :
  schoolbookAddTriangleBound := by
  exact schoolbookAddTriangleBound_of_centeredRep h

theorem schoolbookMulUniversalBound_of_centeredRep
  (h :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y)) :
  schoolbookMulUniversalBound := by
  exact (schoolbookMulUniversalBound_iff_centeredRep).2 h

theorem schoolbookMulUniversalBound_of_centeredRepMulUniversalBound
  (h : centeredRepMulUniversalBound) :
  schoolbookMulUniversalBound := by
  exact schoolbookMulUniversalBound_of_centeredRep h

theorem schoolbookSubTriangleBound_of_centeredRep
  (h :
    ∀ x y : F, Int.natAbs (centeredRep (x - y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  schoolbookSubTriangleBound := by
  exact (schoolbookSubTriangleBound_iff_centeredRep).2 h

/--
Convenience bundle for the two triangle blockers used throughout the non-coarse
P5 composition path.
-/
def schoolbookTriangleBounds : Prop :=
  schoolbookAddTriangleBound ∧ schoolbookSubTriangleBound

theorem schoolbookTriangleBounds_add
  (hTri : schoolbookTriangleBounds) :
  schoolbookAddTriangleBound := by
  exact hTri.1

theorem schoolbookTriangleBounds_sub
  (hTri : schoolbookTriangleBounds) :
  schoolbookSubTriangleBound := by
  exact hTri.2

theorem schoolbookSubTriangleBound_of_add
  (hAddTri : schoolbookAddTriangleBound) :
  schoolbookSubTriangleBound := by
  intro x y
  have hAdd : normInfF (x + (-y)) ≤ normInfF x + normInfF (-y) := hAddTri x (-y)
  simpa [F.sub_eq_add_neg, normInfF_neg] using hAdd

/-- Assumption-free subtraction triangle bound (derived from native add triangle). -/
theorem schoolbookSubTriangleBound_theorem : schoolbookSubTriangleBound := by
  exact schoolbookSubTriangleBound_of_add schoolbookAddTriangleBound_theorem

theorem schoolbookTriangleBounds_of_add
  (hAddTri : schoolbookAddTriangleBound) :
  schoolbookTriangleBounds := by
  exact ⟨hAddTri, schoolbookSubTriangleBound_of_add hAddTri⟩

theorem schoolbookTriangleBounds_theorem : schoolbookTriangleBounds := by
  exact schoolbookTriangleBounds_of_add schoolbookAddTriangleBound_theorem

theorem schoolbookTriangleBounds_of_centeredRep_add
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  schoolbookTriangleBounds := by
  exact schoolbookTriangleBounds_of_add
    (schoolbookAddTriangleBound_of_centeredRep hAddRep)

theorem schoolbookTriangleBounds_of_centeredRepAddTriangleBound
  (hAddRep : centeredRepAddTriangleBound) :
  schoolbookTriangleBounds := by
  exact schoolbookTriangleBounds_of_centeredRep_add hAddRep

theorem schoolbookMulUniversalAndTriangleBounds_of_centeredRepMulAddBounds
  (hRep : centeredRepMulAddBounds) :
  schoolbookMulUniversalBound ∧ schoolbookTriangleBounds := by
  exact ⟨
    schoolbookMulUniversalBound_of_centeredRepMulUniversalBound
      (centeredRepMulAddBounds_mul hRep),
    schoolbookTriangleBounds_of_centeredRepAddTriangleBound
      (centeredRepMulAddBounds_add hRep)
  ⟩

theorem schoolbookMulUniversalBound_theorem : schoolbookMulUniversalBound := by
  exact schoolbookMulUniversalBound_of_centeredRepMulUniversalBound
    centeredRepMulUniversalBound_theorem

/-- Assumption-free `vecAdd` endpoint using the native add-triangle theorem. -/
theorem vecAddNormBoundFromOperands_native
  (BA BB : Nat) :
  vecAddNormBoundFromOperands BA BB (BA + BB) := by
  exact vecAddNormBoundFromOperands_of_triangle (BA := BA) (BB := BB)
    schoolbookAddTriangleBound_theorem

/-- Assumption-free `coeffSub` endpoint using the native subtraction triangle theorem. -/
theorem coeffSubNormBoundFromOperands_native
  (BA BB : Nat) :
  coeffSubNormBoundFromOperands BA BB (BA + BB) := by
  exact coeffSubNormBoundFromOperands_of_triangle (BA := BA) (BB := BB)
    schoolbookSubTriangleBound_theorem

/-- Assumption-free `vecScale` endpoint using the native universal multiplication theorem. -/
theorem vecScaleNormBoundFromOperands_native
  (BS BA : Nat) :
  vecScaleNormBoundFromOperands BS BA (BS * BA) := by
  exact vecScaleNormBoundFromOperands_of_universal (BS := BS) (BA := BA)
    schoolbookMulUniversalBound_theorem

theorem schoolbookMulUniversalAndTriangleBounds_theorem :
  schoolbookMulUniversalBound ∧ schoolbookTriangleBounds := by
  exact schoolbookMulUniversalAndTriangleBounds_of_centeredRepMulAddBounds
    centeredRepMulAddBounds_theorem

theorem schoolbookMulTermBound_of_universal
  {BA BB : Nat}
  (hUniv : schoolbookMulUniversalBound) :
  schoolbookMulTermBound BA BB (BA * BB) := by
  have hUniv' : ∀ x y : F, normInfF (x * y) ≤ normInfF x * normInfF y := by
    simpa [schoolbookMulUniversalBound] using hUniv
  intro x y hx hy
  exact Nat.le_trans (hUniv' x y) (Nat.mul_le_mul hx hy)

theorem mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions
  {BA BB BTerm BRaw : Nat}
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAdd : ∀ x y : F, normInfF x ≤ BRaw → normInfF y ≤ BTerm → normInfF (x + y) ≤ BRaw)
  (hZero : normInfF (0 : F) ≤ BRaw) :
  mulRqRawCoeffBoundFromOperands BA BB BRaw := by
  intro a b t hA hB
  exact mulRqRawCoeffBoundFromOperands_fn_of_schoolbook_term_assumptions
    (hMul := hMul) (hAdd := hAdd) (hZero := hZero)
    a b t hA hB

/-- Zero element bound in `normInfF` for any target natural bound. -/
theorem normInfF_zero_le (B : Nat) : normInfF (0 : F) ≤ B := by
  have hZeroEq : normInfF (0 : F) = 0 := by
    native_decide
  simp [hZeroEq]

theorem mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions_of_term_le
  {BA BB BTerm BRaw : Nat}
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hTermLe : BTerm ≤ BRaw)
  (hAddCollapse : ∀ x y : F, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF (x + y) ≤ BRaw) :
  mulRqRawCoeffBoundFromOperands BA BB BRaw := by
  exact mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions
    (hMul := hMul)
    (hAdd := fun x y hx hy => hAddCollapse x y hx (Nat.le_trans hy hTermLe))
    (hZero := normInfF_zero_le BRaw)

theorem mulRqRawInRangeBoundFromOperands_of_schoolbook_term_assumptions_of_term_le
  {BA BB BTerm BRaw : Nat}
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hTermLe : BTerm ≤ BRaw)
  (hAddCollapse : ∀ x y : F, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF (x + y) ≤ BRaw) :
  mulRqRawInRangeBoundFromOperands BA BB BRaw := by
  intro a b t _ht hA hB
  exact
    (mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions_of_term_le
      (hMul := hMul) (hTermLe := hTermLe) (hAddCollapse := hAddCollapse))
      a b t hA hB

theorem mulRqRawNormBoundFromOperands_of_schoolbook_term_assumptions_of_term_le
  {BA BB BTerm BRaw : Nat}
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hTermLe : BTerm ≤ BRaw)
  (hAddCollapse : ∀ x y : F, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF (x + y) ≤ BRaw) :
  mulRqRawNormBoundFromOperands BA BB BRaw := by
  intro a b hA hB
  exact normInfCoeffs_mulRqRawCoeffs_le_of_inRangeBound
    (a := a) (b := b)
    (hRawInRange := fun t ht =>
      (mulRqRawInRangeBoundFromOperands_of_schoolbook_term_assumptions_of_term_le
        (hMul := hMul) (hTermLe := hTermLe) (hAddCollapse := hAddCollapse))
        a b t ht hA hB)

theorem mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions_sameBound
  {BA BB BRaw : Nat}
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BRaw)
  (hAddCollapse : ∀ x y : F, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF (x + y) ≤ BRaw) :
  mulRqRawCoeffBoundFromOperands BA BB BRaw := by
  exact mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions_of_term_le
    (hMul := hMul) (hTermLe := Nat.le_refl BRaw) (hAddCollapse := hAddCollapse)

theorem mulRqRawInRangeBoundFromOperands_of_schoolbook_term_assumptions_sameBound
  {BA BB BRaw : Nat}
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BRaw)
  (hAddCollapse : ∀ x y : F, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF (x + y) ≤ BRaw) :
  mulRqRawInRangeBoundFromOperands BA BB BRaw := by
  exact mulRqRawInRangeBoundFromOperands_of_schoolbook_term_assumptions_of_term_le
    (hMul := hMul) (hTermLe := Nat.le_refl BRaw) (hAddCollapse := hAddCollapse)

theorem mulRqRawNormBoundFromOperands_of_schoolbook_term_assumptions_sameBound
  {BA BB BRaw : Nat}
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BRaw)
  (hAddCollapse : ∀ x y : F, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF (x + y) ≤ BRaw) :
  mulRqRawNormBoundFromOperands BA BB BRaw := by
  exact mulRqRawNormBoundFromOperands_of_schoolbook_term_assumptions_of_term_le
    (hMul := hMul) (hTermLe := Nat.le_refl BRaw) (hAddCollapse := hAddCollapse)

/--
Assumption bundle for collapsing the `x + y - z` step in `mulRqCoeffSpec`.
-/
def rawAddSubCollapseBound (BRaw B : Nat) : Prop :=
  ∀ x y z : F,
    normInfF x ≤ BRaw →
    normInfF y ≤ BRaw →
    normInfF z ≤ BRaw →
    normInfF (x + y - z) ≤ B

/--
Assumption bundle for collapsing the `x - y` step in `mulRqCoeffSpec`.
-/
def rawSubCollapseBound (BRaw B : Nat) : Prop :=
  ∀ x y : F,
    normInfF x ≤ BRaw →
    normInfF y ≤ BRaw →
    normInfF (x - y) ≤ B

/--
Assumption bundle for collapsing the `x + y` step in `mulRqCoeffSpec`.
-/
def rawAddCollapseBound (BRaw B : Nat) : Prop :=
  ∀ x y : F,
    normInfF x ≤ BRaw →
    normInfF y ≤ BRaw →
    normInfF (x + y) ≤ B

/--
Convenience collapse surface: separate `x+y` and `x-y` bounds.
This matches the typical "field-op" assumption boundary where addition and subtraction
are handled directly, and then combined into the `x+y-z` step as needed.
-/
def rawFieldOpCollapseBound (BRaw B : Nat) : Prop :=
  rawAddCollapseBound BRaw B ∧ rawSubCollapseBound BRaw B

theorem rawAddCollapseBound_of_triangle
  {BRaw : Nat}
  (hAddTri : schoolbookAddTriangleBound) :
  rawAddCollapseBound BRaw (BRaw + BRaw) := by
  have hAddTri' : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y := by
    simpa [schoolbookAddTriangleBound] using hAddTri
  intro x y hx hy
  exact Nat.le_trans (hAddTri' x y) (Nat.add_le_add hx hy)

theorem rawSubCollapseBound_of_triangle
  {BRaw : Nat}
  (hSubTri : schoolbookSubTriangleBound) :
  rawSubCollapseBound BRaw (BRaw + BRaw) := by
  have hSubTri' : ∀ x y : F, normInfF (x - y) ≤ normInfF x + normInfF y := by
    simpa [schoolbookSubTriangleBound] using hSubTri
  intro x y hx hy
  exact Nat.le_trans (hSubTri' x y) (Nat.add_le_add hx hy)

theorem rawFieldOpCollapseBound_of_triangles
  {BRaw : Nat}
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound) :
  rawFieldOpCollapseBound BRaw (BRaw + BRaw) := by
  exact ⟨rawAddCollapseBound_of_triangle hAddTri, rawSubCollapseBound_of_triangle hSubTri⟩

theorem rawAddSubCollapseBound_of_triangles
  {BRaw : Nat}
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound) :
  rawAddSubCollapseBound BRaw (BRaw + BRaw + BRaw) := by
  have hAddTri' : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y := by
    simpa [schoolbookAddTriangleBound] using hAddTri
  have hSubTri' : ∀ x y : F, normInfF (x - y) ≤ normInfF x + normInfF y := by
    simpa [schoolbookSubTriangleBound] using hSubTri
  intro x y z hx hy hz
  have hSubStep : normInfF (x + y - z) ≤ normInfF (x + y) + normInfF z :=
    hSubTri' (x + y) z
  have hAddStep : normInfF (x + y) ≤ normInfF x + normInfF y :=
    hAddTri' x y
  have hLift :
      normInfF (x + y) + normInfF z
        ≤ (normInfF x + normInfF y) + normInfF z := by
    exact Nat.add_le_add hAddStep (Nat.le_refl _)
  have hBound :
      (normInfF x + normInfF y) + normInfF z
        ≤ (BRaw + BRaw) + BRaw := by
    exact Nat.add_le_add (Nat.add_le_add hx hy) hz
  exact Nat.le_trans hSubStep (Nat.le_trans hLift (by simpa [Nat.add_assoc] using hBound))

/-- Native `x+y` collapse: operands `≤ BRaw` imply `x+y ≤ BRaw + BRaw`. -/
theorem rawAddCollapseBound_native
  (BRaw : Nat) :
  rawAddCollapseBound BRaw (BRaw + BRaw) := by
  exact rawAddCollapseBound_of_triangle (BRaw := BRaw)
    schoolbookAddTriangleBound_theorem

/-- Native `x-y` collapse: operands `≤ BRaw` imply `x-y ≤ BRaw + BRaw`. -/
theorem rawSubCollapseBound_native
  (BRaw : Nat) :
  rawSubCollapseBound BRaw (BRaw + BRaw) := by
  exact rawSubCollapseBound_of_triangle (BRaw := BRaw)
    schoolbookSubTriangleBound_theorem

/-- Native `x+y-z` collapse: operands `≤ BRaw` imply `x+y-z ≤ BRaw + BRaw + BRaw`. -/
theorem rawAddSubCollapseBound_native
  (BRaw : Nat) :
  rawAddSubCollapseBound BRaw (BRaw + BRaw + BRaw) := by
  exact rawAddSubCollapseBound_of_triangles (BRaw := BRaw)
    schoolbookAddTriangleBound_theorem
    schoolbookSubTriangleBound_theorem

/-- Native field-op collapse bundle at `BRaw + BRaw`. -/
theorem rawFieldOpCollapseBound_native
  (BRaw : Nat) :
  rawFieldOpCollapseBound BRaw (BRaw + BRaw) := by
  exact rawFieldOpCollapseBound_of_triangles (BRaw := BRaw)
    schoolbookAddTriangleBound_theorem
    schoolbookSubTriangleBound_theorem

private theorem normInfF_zero_eq_zero_local : normInfF (0 : F) = 0 := by
  native_decide

theorem rawAddSubCollapseBound_of_add_and_sub
  {BRaw BAdd B : Nat}
  (hAdd : rawAddCollapseBound BRaw BAdd)
  (hSub : ∀ u z, normInfF u ≤ BAdd → normInfF z ≤ BRaw → normInfF (u - z) ≤ B) :
  rawAddSubCollapseBound BRaw B := by
  intro x y z hx hy hz
  exact hSub (x + y) z (hAdd x y hx hy) hz

theorem rawAddCollapseBound_of_addSub
  {BRaw B : Nat}
  (hAddSub : rawAddSubCollapseBound BRaw B) :
  rawAddCollapseBound BRaw B := by
  intro x y hx hy
  have hZero : normInfF (0 : F) ≤ BRaw := by
    have hEq : normInfF (0 : F) = 0 := normInfF_zero_eq_zero_local
    simp [hEq]
  have hAddSub0 : normInfF (x + y - (0 : F)) ≤ B := hAddSub x y (0 : F) hx hy hZero
  have hExpr : x + y - (0 : F) = x + y := by
    exact F.sub_zero_of_canonical (a := x + y) (F.canonical_add x y)
  exact hExpr ▸ hAddSub0

theorem rawAddSubCollapseBound_of_add_and_sub_same
  {BRaw : Nat}
  (hAdd : rawAddCollapseBound BRaw BRaw)
  (hSub : rawSubCollapseBound BRaw BRaw) :
  rawAddSubCollapseBound BRaw BRaw := by
  exact rawAddSubCollapseBound_of_add_and_sub
    (BRaw := BRaw) (BAdd := BRaw) (B := BRaw)
    hAdd (fun u z hu hz => hSub u z hu hz)

theorem rawFieldOpCollapseBound_of_addSub_and_sub
  {BRaw B : Nat}
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B) :
  rawFieldOpCollapseBound BRaw B := by
  exact ⟨rawAddCollapseBound_of_addSub hAddSub, hSub⟩

theorem rawAddSubCollapseBound_mono
  {BRaw B B' : Nat}
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hLe : B ≤ B') :
  rawAddSubCollapseBound BRaw B' := by
  intro x y z hx hy hz
  exact Nat.le_trans (hAddSub x y z hx hy hz) hLe

theorem rawSubCollapseBound_mono
  {BRaw B B' : Nat}
  (hSub : rawSubCollapseBound BRaw B)
  (hLe : B ≤ B') :
  rawSubCollapseBound BRaw B' := by
  intro x y hx hy
  exact Nat.le_trans (hSub x y hx hy) hLe

theorem rawAddCollapseBound_mono
  {BRaw B B' : Nat}
  (hAdd : rawAddCollapseBound BRaw B)
  (hLe : B ≤ B') :
  rawAddCollapseBound BRaw B' := by
  intro x y hx hy
  exact Nat.le_trans (hAdd x y hx hy) hLe

theorem rawFieldOpCollapseBound_mono
  {BRaw B B' : Nat}
  (hOps : rawFieldOpCollapseBound BRaw B)
  (hLe : B ≤ B') :
  rawFieldOpCollapseBound BRaw B' := by
  exact ⟨rawAddCollapseBound_mono hOps.1 hLe, rawSubCollapseBound_mono hOps.2 hLe⟩

theorem mulRqRawNormBoundFromOperands_halfQ
  (BA BB : Nat) :
  mulRqRawNormBoundFromOperands BA BB halfQ := by
  intro a b _hA _hB
  simpa using (normInfCoeffs_le_halfQ (mulRqRawCoeffs a b))

theorem mulRqRawInRangeBoundFromOperands_halfQ
  (BA BB : Nat) :
  mulRqRawInRangeBoundFromOperands BA BB halfQ := by
  intro a b t _ht _hA _hB
  simpa using (normInfF_le_halfQ (mulRqRawCoeffSpec a b t))

theorem mulRqRawNormBoundFromOperands_mono
  {BA BB BRaw BRaw' : Nat}
  (hRaw : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hLe : BRaw ≤ BRaw') :
  mulRqRawNormBoundFromOperands BA BB BRaw' := by
  intro a b hA hB
  exact Nat.le_trans (hRaw a b hA hB) hLe

theorem mulRqRawInRangeBoundFromOperands_mono
  {BA BB BRaw BRaw' : Nat}
  (hRawInRange : mulRqRawInRangeBoundFromOperands BA BB BRaw)
  (hLe : BRaw ≤ BRaw') :
  mulRqRawInRangeBoundFromOperands BA BB BRaw' := by
  intro a b t ht hA hB
  exact Nat.le_trans (hRawInRange a b t ht hA hB) hLe

theorem mulRqRawCoeffBoundFromOperands_mono
  {BA BB BRaw BRaw' : Nat}
  (hRawCoeff : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hLe : BRaw ≤ BRaw') :
  mulRqRawCoeffBoundFromOperands BA BB BRaw' := by
  intro a b t hA hB
  exact Nat.le_trans (hRawCoeff a b t hA hB) hLe

theorem mulRqRawNormBoundFromOperands_of_halfQ_le
  {BA BB BRaw : Nat}
  (hHalfQ : halfQ ≤ BRaw) :
  mulRqRawNormBoundFromOperands BA BB BRaw := by
  exact mulRqRawNormBoundFromOperands_mono
    (mulRqRawNormBoundFromOperands_halfQ BA BB)
    hHalfQ

theorem mulRqRawInRangeBoundFromOperands_of_halfQ_le
  {BA BB BRaw : Nat}
  (hHalfQ : halfQ ≤ BRaw) :
  mulRqRawInRangeBoundFromOperands BA BB BRaw := by
  exact mulRqRawInRangeBoundFromOperands_mono
    (mulRqRawInRangeBoundFromOperands_halfQ BA BB)
    hHalfQ

theorem vecAddNormBoundFromOperands_halfQ
  (BA BB : Nat) :
  vecAddNormBoundFromOperands BA BB halfQ := by
  intro a b hSize _hA _hB
  have : normInfCoeffs (vecAdd a b) ≤ halfQ := by
    simpa using normInfCoeffs_vecAdd_le_halfQ a b
  exact this

theorem coeffSubNormBoundFromOperands_halfQ
  (BA BB : Nat) :
  coeffSubNormBoundFromOperands BA BB halfQ := by
  intro a b hSize _hA _hB
  have : normInfCoeffs (coeffSub a b) ≤ halfQ := by
    simpa using normInfCoeffs_coeffSub_le_halfQ a b
  exact this

theorem vecScaleNormBoundFromOperands_halfQ
  (BS BA : Nat) :
  vecScaleNormBoundFromOperands BS BA halfQ := by
  intro s a _hS _hA
  simpa using normInfCoeffs_vecScale_le_halfQ s a

theorem vecAddNormBoundFromOperands_of_halfQ_le
  {BA BB B : Nat}
  (hHalfQ : halfQ ≤ B) :
  vecAddNormBoundFromOperands BA BB B := by
  intro a b hSize hA hB
  exact Nat.le_trans
    ((vecAddNormBoundFromOperands_halfQ BA BB) a b hSize hA hB)
    hHalfQ

theorem coeffSubNormBoundFromOperands_of_halfQ_le
  {BA BB B : Nat}
  (hHalfQ : halfQ ≤ B) :
  coeffSubNormBoundFromOperands BA BB B := by
  intro a b hSize hA hB
  exact Nat.le_trans
    ((coeffSubNormBoundFromOperands_halfQ BA BB) a b hSize hA hB)
    hHalfQ

theorem vecScaleNormBoundFromOperands_of_halfQ_le
  {BS BA B : Nat}
  (hHalfQ : halfQ ≤ B) :
  vecScaleNormBoundFromOperands BS BA B := by
  intro s a hS hA
  exact Nat.le_trans
    ((vecScaleNormBoundFromOperands_halfQ BS BA) s a hS hA)
    hHalfQ

theorem mulRqRawNormBoundFromOperands_of_inRange
  {BA BB BRaw : Nat}
  (hRawInRangeFromOperands : mulRqRawInRangeBoundFromOperands BA BB BRaw) :
  mulRqRawNormBoundFromOperands BA BB BRaw := by
  intro a b hA hB
  exact normInfCoeffs_mulRqRawCoeffs_le_of_inRangeBound
    (a := a) (b := b)
    (hRawInRange := fun t ht => hRawInRangeFromOperands a b t ht hA hB)

theorem mulRqRawInRangeBoundFromOperands_of_norm
  {BA BB BRaw : Nat}
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw) :
  mulRqRawInRangeBoundFromOperands BA BB BRaw := by
  intro a b t _ht hA hB
  exact normInfF_mulRqRawCoeffSpec_le_of_rawCoeffsNorm
    (a := a) (b := b)
    (hRawCoeffs := hRawFromOperands a b hA hB)
    t

theorem mulRqRawNormBoundFromOperands_iff_inRange
  {BA BB BRaw : Nat} :
  mulRqRawNormBoundFromOperands BA BB BRaw ↔
    mulRqRawInRangeBoundFromOperands BA BB BRaw := by
  constructor
  · exact mulRqRawInRangeBoundFromOperands_of_norm
  · exact mulRqRawNormBoundFromOperands_of_inRange

theorem mulRqRawCoeffBoundFromOperands_of_inRange
  {BA BB BRaw : Nat}
  (hRawInRangeFromOperands : mulRqRawInRangeBoundFromOperands BA BB BRaw) :
  mulRqRawCoeffBoundFromOperands BA BB BRaw := by
  intro a b t hA hB
  exact normInfF_mulRqRawCoeffSpec_le_of_inRangeBound
    (a := a) (b := b) (t := t)
    (hRawInRange := fun u hu => hRawInRangeFromOperands a b u hu hA hB)

theorem mulRqRawInRangeBoundFromOperands_of_rawCoeff
  {BA BB BRaw : Nat}
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw) :
  mulRqRawInRangeBoundFromOperands BA BB BRaw := by
  intro a b t _ht hA hB
  exact hRawCoeffFromOperands a b t hA hB

theorem mulRqRawNormBoundFromOperands_of_rawCoeff
  {BA BB BRaw : Nat}
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw) :
  mulRqRawNormBoundFromOperands BA BB BRaw := by
  exact mulRqRawNormBoundFromOperands_of_inRange
    (mulRqRawInRangeBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)

theorem mulRqRawInRangeBoundFromOperands_of_schoolbook_term_assumptions
  {BA BB BTerm BRaw : Nat}
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAdd : ∀ x y : F, normInfF x ≤ BRaw → normInfF y ≤ BTerm → normInfF (x + y) ≤ BRaw)
  (hZero : normInfF (0 : F) ≤ BRaw) :
  mulRqRawInRangeBoundFromOperands BA BB BRaw := by
  exact mulRqRawInRangeBoundFromOperands_of_rawCoeff
    (mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions
      (hMul := hMul) (hAdd := hAdd) (hZero := hZero))

theorem mulRqRawNormBoundFromOperands_of_schoolbook_term_assumptions
  {BA BB BTerm BRaw : Nat}
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAdd : ∀ x y : F, normInfF x ≤ BRaw → normInfF y ≤ BTerm → normInfF (x + y) ≤ BRaw)
  (hZero : normInfF (0 : F) ≤ BRaw) :
  mulRqRawNormBoundFromOperands BA BB BRaw := by
  exact mulRqRawNormBoundFromOperands_of_rawCoeff
    (mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions
      (hMul := hMul) (hAdd := hAdd) (hZero := hZero))

/--
Sum-style schoolbook wrapper: from per-term multiplication bounds and triangle
addition, derive a raw-coefficient bound at `((D * D) * BTerm)`.
-/
theorem mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions_sum
  {BA BB BTerm : Nat}
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y) :
  mulRqRawCoeffBoundFromOperands BA BB ((D * D) * BTerm) := by
  intro a b t hA hB
  exact mulRqRawCoeffBoundFromOperands_fn_of_schoolbook_term_assumptions_sum
    (hMul := hMul) (hAddTri := hAddTri)
    a b t hA hB

theorem mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions_sum_tight
  {BA BB BTerm : Nat}
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y) :
  mulRqRawCoeffBoundFromOperands BA BB (D * BTerm) := by
  intro a b t hA hB
  exact mulRqRawCoeffBoundFromOperands_fn_of_schoolbook_term_assumptions_sum_tight
    (hMul := hMul) (hAddTri := hAddTri)
    a b t hA hB

theorem mulRqRawInRangeBoundFromOperands_of_schoolbook_term_assumptions_sum
  {BA BB BTerm : Nat}
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y) :
  mulRqRawInRangeBoundFromOperands BA BB ((D * D) * BTerm) := by
  exact mulRqRawInRangeBoundFromOperands_of_rawCoeff
    (mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions_sum
      (hMul := hMul) (hAddTri := hAddTri))

theorem mulRqRawNormBoundFromOperands_of_schoolbook_term_assumptions_sum
  {BA BB BTerm : Nat}
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y) :
  mulRqRawNormBoundFromOperands BA BB ((D * D) * BTerm) := by
  exact mulRqRawNormBoundFromOperands_of_rawCoeff
    (mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions_sum
      (hMul := hMul) (hAddTri := hAddTri))

theorem mulRqRawInRangeBoundFromOperands_of_schoolbook_term_assumptions_sum_tight
  {BA BB BTerm : Nat}
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y) :
  mulRqRawInRangeBoundFromOperands BA BB (D * BTerm) := by
  exact mulRqRawInRangeBoundFromOperands_of_rawCoeff
    (mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions_sum_tight
      (hMul := hMul) (hAddTri := hAddTri))

theorem mulRqRawNormBoundFromOperands_of_schoolbook_term_assumptions_sum_tight
  {BA BB BTerm : Nat}
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y) :
  mulRqRawNormBoundFromOperands BA BB (D * BTerm) := by
  exact mulRqRawNormBoundFromOperands_of_rawCoeff
    (mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions_sum_tight
      (hMul := hMul) (hAddTri := hAddTri))

theorem schoolbookAccAddBound_of_rawAddCollapse
  {BRaw BTerm : Nat}
  (hAdd : rawAddCollapseBound BRaw BRaw)
  (hTermLe : BTerm ≤ BRaw) :
  schoolbookAccAddBound BRaw BTerm := by
  intro x y hx hy
  exact hAdd x y hx (Nat.le_trans hy hTermLe)

/--
Bundle-style wrapper for the non-sum schoolbook path.
This keeps the theorem-native interfaces explicit while avoiding repeated
function-typed argument plumbing at call sites.
-/
theorem mulRqRawCoeffBoundFromOperands_of_schoolbookAssumptionBundles
  {BA BB BTerm BRaw : Nat}
  (hMul : schoolbookMulTermBound BA BB BTerm)
  (hAdd : schoolbookAccAddBound BRaw BTerm) :
  mulRqRawCoeffBoundFromOperands BA BB BRaw := by
  exact mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions
    (hMul := hMul) (hAdd := hAdd) (hZero := normInfF_zero_le BRaw)

theorem mulRqRawInRangeBoundFromOperands_of_schoolbookAssumptionBundles
  {BA BB BTerm BRaw : Nat}
  (hMul : schoolbookMulTermBound BA BB BTerm)
  (hAdd : schoolbookAccAddBound BRaw BTerm) :
  mulRqRawInRangeBoundFromOperands BA BB BRaw := by
  exact mulRqRawInRangeBoundFromOperands_of_rawCoeff
    (mulRqRawCoeffBoundFromOperands_of_schoolbookAssumptionBundles
      (hMul := hMul) (hAdd := hAdd))

theorem mulRqRawNormBoundFromOperands_of_schoolbookAssumptionBundles
  {BA BB BTerm BRaw : Nat}
  (hMul : schoolbookMulTermBound BA BB BTerm)
  (hAdd : schoolbookAccAddBound BRaw BTerm) :
  mulRqRawNormBoundFromOperands BA BB BRaw := by
  exact mulRqRawNormBoundFromOperands_of_rawCoeff
    (mulRqRawCoeffBoundFromOperands_of_schoolbookAssumptionBundles
      (hMul := hMul) (hAdd := hAdd))

/-- Bundle-style wrapper for the sum schoolbook path (`(D*D)*BTerm` bound). -/
theorem mulRqRawCoeffBoundFromOperands_of_schoolbookAssumptionBundles_sum
  {BA BB BTerm : Nat}
  (hMul : schoolbookMulTermBound BA BB BTerm)
  (hAddTri : schoolbookAddTriangleBound) :
  mulRqRawCoeffBoundFromOperands BA BB ((D * D) * BTerm) := by
  exact mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions_sum
    (hMul := hMul) (hAddTri := hAddTri)

theorem mulRqRawInRangeBoundFromOperands_of_schoolbookAssumptionBundles_sum
  {BA BB BTerm : Nat}
  (hMul : schoolbookMulTermBound BA BB BTerm)
  (hAddTri : schoolbookAddTriangleBound) :
  mulRqRawInRangeBoundFromOperands BA BB ((D * D) * BTerm) := by
  exact mulRqRawInRangeBoundFromOperands_of_schoolbook_term_assumptions_sum
    (hMul := hMul) (hAddTri := hAddTri)

theorem mulRqRawNormBoundFromOperands_of_schoolbookAssumptionBundles_sum
  {BA BB BTerm : Nat}
  (hMul : schoolbookMulTermBound BA BB BTerm)
  (hAddTri : schoolbookAddTriangleBound) :
  mulRqRawNormBoundFromOperands BA BB ((D * D) * BTerm) := by
  exact mulRqRawNormBoundFromOperands_of_schoolbook_term_assumptions_sum
    (hMul := hMul) (hAddTri := hAddTri)

/-- Bundle-style wrapper for the tight sum schoolbook path (`D*BTerm` bound). -/
theorem mulRqRawCoeffBoundFromOperands_of_schoolbookAssumptionBundles_sum_tight
  {BA BB BTerm : Nat}
  (hMul : schoolbookMulTermBound BA BB BTerm)
  (hAddTri : schoolbookAddTriangleBound) :
  mulRqRawCoeffBoundFromOperands BA BB (D * BTerm) := by
  exact mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions_sum_tight
    (hMul := hMul) (hAddTri := hAddTri)

theorem mulRqRawInRangeBoundFromOperands_of_schoolbookAssumptionBundles_sum_tight
  {BA BB BTerm : Nat}
  (hMul : schoolbookMulTermBound BA BB BTerm)
  (hAddTri : schoolbookAddTriangleBound) :
  mulRqRawInRangeBoundFromOperands BA BB (D * BTerm) := by
  exact mulRqRawInRangeBoundFromOperands_of_schoolbook_term_assumptions_sum_tight
    (hMul := hMul) (hAddTri := hAddTri)

theorem mulRqRawNormBoundFromOperands_of_schoolbookAssumptionBundles_sum_tight
  {BA BB BTerm : Nat}
  (hMul : schoolbookMulTermBound BA BB BTerm)
  (hAddTri : schoolbookAddTriangleBound) :
  mulRqRawNormBoundFromOperands BA BB (D * BTerm) := by
  exact mulRqRawNormBoundFromOperands_of_schoolbook_term_assumptions_sum_tight
    (hMul := hMul) (hAddTri := hAddTri)

/--
Sum-path instantiation from universal multiplication and additive triangle
blockers: raw bound is `((D * D) * (BA * BB))`.
-/
theorem mulRqRawNormBoundFromOperands_of_universal_mul_and_add_sum
  {BA BB : Nat}
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound) :
  mulRqRawNormBoundFromOperands BA BB ((D * D) * (BA * BB)) := by
  exact mulRqRawNormBoundFromOperands_of_schoolbookAssumptionBundles_sum
    (hMul := schoolbookMulTermBound_of_universal (BA := BA) (BB := BB) hMulUniv)
    (hAddTri := hAddTri)

theorem mulRqRawInRangeBoundFromOperands_of_universal_mul_and_add_sum
  {BA BB : Nat}
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound) :
  mulRqRawInRangeBoundFromOperands BA BB ((D * D) * (BA * BB)) := by
  exact mulRqRawInRangeBoundFromOperands_of_norm
    (mulRqRawNormBoundFromOperands_of_universal_mul_and_add_sum
      (BA := BA) (BB := BB) hMulUniv hAddTri)

theorem mulRqRawCoeffBoundFromOperands_of_universal_mul_and_add_sum
  {BA BB : Nat}
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound) :
  mulRqRawCoeffBoundFromOperands BA BB ((D * D) * (BA * BB)) := by
  exact mulRqRawCoeffBoundFromOperands_of_inRange
    (mulRqRawInRangeBoundFromOperands_of_universal_mul_and_add_sum
      (BA := BA) (BB := BB) hMulUniv hAddTri)

theorem mulRqRawNormBoundFromOperands_of_universal_mul_and_triangles_sum
  {BA BB : Nat}
  (hMulUniv : schoolbookMulUniversalBound)
  (hTri : schoolbookTriangleBounds) :
  mulRqRawNormBoundFromOperands BA BB ((D * D) * (BA * BB)) := by
  exact mulRqRawNormBoundFromOperands_of_universal_mul_and_add_sum
    (BA := BA) (BB := BB) hMulUniv (schoolbookTriangleBounds_add hTri)

theorem mulRqRawInRangeBoundFromOperands_of_universal_mul_and_triangles_sum
  {BA BB : Nat}
  (hMulUniv : schoolbookMulUniversalBound)
  (hTri : schoolbookTriangleBounds) :
  mulRqRawInRangeBoundFromOperands BA BB ((D * D) * (BA * BB)) := by
  exact mulRqRawInRangeBoundFromOperands_of_norm
    (mulRqRawNormBoundFromOperands_of_universal_mul_and_triangles_sum
      (BA := BA) (BB := BB) hMulUniv hTri)

theorem mulRqRawCoeffBoundFromOperands_of_universal_mul_and_triangles_sum
  {BA BB : Nat}
  (hMulUniv : schoolbookMulUniversalBound)
  (hTri : schoolbookTriangleBounds) :
  mulRqRawCoeffBoundFromOperands BA BB ((D * D) * (BA * BB)) := by
  exact mulRqRawCoeffBoundFromOperands_of_inRange
    (mulRqRawInRangeBoundFromOperands_of_universal_mul_and_triangles_sum
      (BA := BA) (BB := BB) hMulUniv hTri)

theorem mulRqRawNormBoundFromOperands_of_centeredRep_mul_and_triangles_sum
  {BA BB : Nat}
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hTriRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  mulRqRawNormBoundFromOperands BA BB ((D * D) * (BA * BB)) := by
  exact mulRqRawNormBoundFromOperands_of_universal_mul_and_triangles_sum
    (BA := BA) (BB := BB)
    (schoolbookMulUniversalBound_of_centeredRep hMulRep)
    (schoolbookTriangleBounds_of_centeredRep_add hTriRep)

theorem mulRqRawInRangeBoundFromOperands_of_centeredRep_mul_and_triangles_sum
  {BA BB : Nat}
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hTriRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  mulRqRawInRangeBoundFromOperands BA BB ((D * D) * (BA * BB)) := by
  exact mulRqRawInRangeBoundFromOperands_of_norm
    (mulRqRawNormBoundFromOperands_of_centeredRep_mul_and_triangles_sum
      (BA := BA) (BB := BB) hMulRep hTriRep)

theorem mulRqRawCoeffBoundFromOperands_of_centeredRep_mul_and_triangles_sum
  {BA BB : Nat}
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hTriRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  mulRqRawCoeffBoundFromOperands BA BB ((D * D) * (BA * BB)) := by
  exact mulRqRawCoeffBoundFromOperands_of_inRange
    (mulRqRawInRangeBoundFromOperands_of_centeredRep_mul_and_triangles_sum
      (BA := BA) (BB := BB) hMulRep hTriRep)

theorem mulRqRawNormBoundFromOperands_of_centeredRep_mul_and_add_sum
  {BA BB : Nat}
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  mulRqRawNormBoundFromOperands BA BB ((D * D) * (BA * BB)) := by
  exact mulRqRawNormBoundFromOperands_of_universal_mul_and_add_sum
    (BA := BA) (BB := BB)
    (schoolbookMulUniversalBound_of_centeredRep hMulRep)
    (schoolbookAddTriangleBound_of_centeredRep hAddRep)

theorem mulRqRawInRangeBoundFromOperands_of_centeredRep_mul_and_add_sum
  {BA BB : Nat}
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  mulRqRawInRangeBoundFromOperands BA BB ((D * D) * (BA * BB)) := by
  exact mulRqRawInRangeBoundFromOperands_of_norm
    (mulRqRawNormBoundFromOperands_of_centeredRep_mul_and_add_sum
      (BA := BA) (BB := BB) hMulRep hAddRep)

theorem mulRqRawCoeffBoundFromOperands_of_centeredRep_mul_and_add_sum
  {BA BB : Nat}
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  mulRqRawCoeffBoundFromOperands BA BB ((D * D) * (BA * BB)) := by
  exact mulRqRawCoeffBoundFromOperands_of_inRange
    (mulRqRawInRangeBoundFromOperands_of_centeredRep_mul_and_add_sum
      (BA := BA) (BB := BB) hMulRep hAddRep)

/--
Tight sum-path instantiation from universal multiplication and additive triangle
blockers: raw bound becomes `D * (BA * BB)`.
-/
theorem mulRqRawNormBoundFromOperands_of_universal_mul_and_add_sum_tight
  {BA BB : Nat}
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound) :
  mulRqRawNormBoundFromOperands BA BB (D * (BA * BB)) := by
  exact mulRqRawNormBoundFromOperands_of_schoolbookAssumptionBundles_sum_tight
    (hMul := schoolbookMulTermBound_of_universal (BA := BA) (BB := BB) hMulUniv)
    (hAddTri := hAddTri)

theorem mulRqRawInRangeBoundFromOperands_of_universal_mul_and_add_sum_tight
  {BA BB : Nat}
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound) :
  mulRqRawInRangeBoundFromOperands BA BB (D * (BA * BB)) := by
  exact mulRqRawInRangeBoundFromOperands_of_norm
    (mulRqRawNormBoundFromOperands_of_universal_mul_and_add_sum_tight
      (BA := BA) (BB := BB) hMulUniv hAddTri)

theorem mulRqRawCoeffBoundFromOperands_of_universal_mul_and_add_sum_tight
  {BA BB : Nat}
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound) :
  mulRqRawCoeffBoundFromOperands BA BB (D * (BA * BB)) := by
  exact mulRqRawCoeffBoundFromOperands_of_inRange
    (mulRqRawInRangeBoundFromOperands_of_universal_mul_and_add_sum_tight
      (BA := BA) (BB := BB) hMulUniv hAddTri)

theorem mulRqRawNormBoundFromOperands_of_universal_mul_and_triangles_sum_tight
  {BA BB : Nat}
  (hMulUniv : schoolbookMulUniversalBound)
  (hTri : schoolbookTriangleBounds) :
  mulRqRawNormBoundFromOperands BA BB (D * (BA * BB)) := by
  exact mulRqRawNormBoundFromOperands_of_universal_mul_and_add_sum_tight
    (BA := BA) (BB := BB) hMulUniv (schoolbookTriangleBounds_add hTri)

theorem mulRqRawInRangeBoundFromOperands_of_universal_mul_and_triangles_sum_tight
  {BA BB : Nat}
  (hMulUniv : schoolbookMulUniversalBound)
  (hTri : schoolbookTriangleBounds) :
  mulRqRawInRangeBoundFromOperands BA BB (D * (BA * BB)) := by
  exact mulRqRawInRangeBoundFromOperands_of_norm
    (mulRqRawNormBoundFromOperands_of_universal_mul_and_triangles_sum_tight
      (BA := BA) (BB := BB) hMulUniv hTri)

theorem mulRqRawCoeffBoundFromOperands_of_universal_mul_and_triangles_sum_tight
  {BA BB : Nat}
  (hMulUniv : schoolbookMulUniversalBound)
  (hTri : schoolbookTriangleBounds) :
  mulRqRawCoeffBoundFromOperands BA BB (D * (BA * BB)) := by
  exact mulRqRawCoeffBoundFromOperands_of_inRange
    (mulRqRawInRangeBoundFromOperands_of_universal_mul_and_triangles_sum_tight
      (BA := BA) (BB := BB) hMulUniv hTri)

theorem mulRqRawNormBoundFromOperands_of_centeredRep_mul_and_add_sum_tight
  {BA BB : Nat}
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  mulRqRawNormBoundFromOperands BA BB (D * (BA * BB)) := by
  exact mulRqRawNormBoundFromOperands_of_universal_mul_and_add_sum_tight
    (BA := BA) (BB := BB)
    (schoolbookMulUniversalBound_of_centeredRep hMulRep)
    (schoolbookAddTriangleBound_of_centeredRep hAddRep)

theorem mulRqRawNormBoundFromOperands_of_centeredRepMulAddBounds_sum_tight
  {BA BB : Nat}
  (hRep : centeredRepMulAddBounds) :
  mulRqRawNormBoundFromOperands BA BB (D * (BA * BB)) := by
  exact mulRqRawNormBoundFromOperands_of_centeredRep_mul_and_add_sum_tight
    (BA := BA) (BB := BB)
    (centeredRepMulAddBounds_mul hRep)
    (centeredRepMulAddBounds_add hRep)

theorem mulRqRawNormBoundFromOperands_of_centeredRep_mul_sum_tight
  {BA BB : Nat}
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y)) :
  mulRqRawNormBoundFromOperands BA BB (D * (BA * BB)) := by
  exact mulRqRawNormBoundFromOperands_of_centeredRepMulAddBounds_sum_tight
    (BA := BA) (BB := BB) (centeredRepMulAddBounds_of_mul hMulRep)

theorem mulRqRawInRangeBoundFromOperands_of_centeredRep_mul_and_add_sum_tight
  {BA BB : Nat}
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  mulRqRawInRangeBoundFromOperands BA BB (D * (BA * BB)) := by
  exact mulRqRawInRangeBoundFromOperands_of_norm
    (mulRqRawNormBoundFromOperands_of_centeredRep_mul_and_add_sum_tight
      (BA := BA) (BB := BB) hMulRep hAddRep)

theorem mulRqRawInRangeBoundFromOperands_of_centeredRep_mul_sum_tight
  {BA BB : Nat}
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y)) :
  mulRqRawInRangeBoundFromOperands BA BB (D * (BA * BB)) := by
  exact mulRqRawInRangeBoundFromOperands_of_norm
    (mulRqRawNormBoundFromOperands_of_centeredRep_mul_sum_tight
      (BA := BA) (BB := BB) hMulRep)

theorem mulRqRawCoeffBoundFromOperands_of_centeredRep_mul_and_add_sum_tight
  {BA BB : Nat}
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  mulRqRawCoeffBoundFromOperands BA BB (D * (BA * BB)) := by
  exact mulRqRawCoeffBoundFromOperands_of_inRange
    (mulRqRawInRangeBoundFromOperands_of_centeredRep_mul_and_add_sum_tight
      (BA := BA) (BB := BB) hMulRep hAddRep)

theorem mulRqRawCoeffBoundFromOperands_of_centeredRep_mul_sum_tight
  {BA BB : Nat}
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y)) :
  mulRqRawCoeffBoundFromOperands BA BB (D * (BA * BB)) := by
  exact mulRqRawCoeffBoundFromOperands_of_inRange
    (mulRqRawInRangeBoundFromOperands_of_centeredRep_mul_sum_tight
      (BA := BA) (BB := BB) hMulRep)

theorem mulRqRawNormBoundFromOperands_of_centeredRep_mul_and_triangles_sum_tight
  {BA BB : Nat}
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hTriRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  mulRqRawNormBoundFromOperands BA BB (D * (BA * BB)) := by
  exact mulRqRawNormBoundFromOperands_of_universal_mul_and_triangles_sum_tight
    (BA := BA) (BB := BB)
    (schoolbookMulUniversalBound_of_centeredRep hMulRep)
    (schoolbookTriangleBounds_of_centeredRep_add hTriRep)

theorem mulRqRawInRangeBoundFromOperands_of_centeredRep_mul_and_triangles_sum_tight
  {BA BB : Nat}
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hTriRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  mulRqRawInRangeBoundFromOperands BA BB (D * (BA * BB)) := by
  exact mulRqRawInRangeBoundFromOperands_of_norm
    (mulRqRawNormBoundFromOperands_of_centeredRep_mul_and_triangles_sum_tight
      (BA := BA) (BB := BB) hMulRep hTriRep)

theorem mulRqRawCoeffBoundFromOperands_of_centeredRep_mul_and_triangles_sum_tight
  {BA BB : Nat}
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hTriRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  mulRqRawCoeffBoundFromOperands BA BB (D * (BA * BB)) := by
  exact mulRqRawCoeffBoundFromOperands_of_inRange
    (mulRqRawInRangeBoundFromOperands_of_centeredRep_mul_and_triangles_sum_tight
      (BA := BA) (BB := BB) hMulRep hTriRep)

/-- Assumption-free coarse sum-path raw norm constructor from proven blockers. -/
theorem mulRqRawNormBoundFromOperands_native_sum
  {BA BB : Nat} :
  mulRqRawNormBoundFromOperands BA BB ((D * D) * (BA * BB)) := by
  exact mulRqRawNormBoundFromOperands_of_universal_mul_and_triangles_sum
    (BA := BA) (BB := BB)
    schoolbookMulUniversalBound_theorem
    schoolbookTriangleBounds_theorem

/-- Assumption-free coarse sum-path in-range raw constructor from proven blockers. -/
theorem mulRqRawInRangeBoundFromOperands_native_sum
  {BA BB : Nat} :
  mulRqRawInRangeBoundFromOperands BA BB ((D * D) * (BA * BB)) := by
  exact mulRqRawInRangeBoundFromOperands_of_universal_mul_and_triangles_sum
    (BA := BA) (BB := BB)
    schoolbookMulUniversalBound_theorem
    schoolbookTriangleBounds_theorem

/-- Assumption-free coarse sum-path raw-coeff constructor from proven blockers. -/
theorem mulRqRawCoeffBoundFromOperands_native_sum
  {BA BB : Nat} :
  mulRqRawCoeffBoundFromOperands BA BB ((D * D) * (BA * BB)) := by
  exact mulRqRawCoeffBoundFromOperands_of_universal_mul_and_triangles_sum
    (BA := BA) (BB := BB)
    schoolbookMulUniversalBound_theorem
    schoolbookTriangleBounds_theorem

/-- Assumption-free tight sum-path raw norm constructor from proven blockers. -/
theorem mulRqRawNormBoundFromOperands_native_sum_tight
  {BA BB : Nat} :
  mulRqRawNormBoundFromOperands BA BB (D * (BA * BB)) := by
  exact mulRqRawNormBoundFromOperands_of_universal_mul_and_triangles_sum_tight
    (BA := BA) (BB := BB)
    schoolbookMulUniversalBound_theorem
    schoolbookTriangleBounds_theorem

/-- Assumption-free tight sum-path in-range raw constructor from proven blockers. -/
theorem mulRqRawInRangeBoundFromOperands_native_sum_tight
  {BA BB : Nat} :
  mulRqRawInRangeBoundFromOperands BA BB (D * (BA * BB)) := by
  exact mulRqRawInRangeBoundFromOperands_of_universal_mul_and_triangles_sum_tight
    (BA := BA) (BB := BB)
    schoolbookMulUniversalBound_theorem
    schoolbookTriangleBounds_theorem

/-- Assumption-free tight sum-path raw-coeff constructor from proven blockers. -/
theorem mulRqRawCoeffBoundFromOperands_native_sum_tight
  {BA BB : Nat} :
  mulRqRawCoeffBoundFromOperands BA BB (D * (BA * BB)) := by
  exact mulRqRawCoeffBoundFromOperands_of_universal_mul_and_triangles_sum_tight
    (BA := BA) (BB := BB)
    schoolbookMulUniversalBound_theorem
    schoolbookTriangleBounds_theorem

/-- Lift a raw-array norm bound assumption into the total raw-coeff accessor form. -/
theorem mulRqRawCoeffBoundFromOperands_of_norm
  {BA BB BRaw : Nat}
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw) :
  mulRqRawCoeffBoundFromOperands BA BB BRaw := by
  exact mulRqRawCoeffBoundFromOperands_of_inRange
    (mulRqRawInRangeBoundFromOperands_of_norm hRawFromOperands)

/-- Raw-coeff accessor bounds and raw-array norm bounds are equivalent assumptions. -/
theorem mulRqRawCoeffBoundFromOperands_iff_norm
  {BA BB BRaw : Nat} :
  mulRqRawCoeffBoundFromOperands BA BB BRaw ↔
    mulRqRawNormBoundFromOperands BA BB BRaw := by
  constructor
  · exact mulRqRawNormBoundFromOperands_of_rawCoeff
  · exact mulRqRawCoeffBoundFromOperands_of_norm

theorem mulRqRawCoeffBoundFromOperands_halfQ
  (BA BB : Nat) :
  mulRqRawCoeffBoundFromOperands BA BB halfQ := by
  intro a b t _hA _hB
  simpa using normInfF_le_halfQ (mulRqRawCoeffSpec a b t)

theorem mulRqRawCoeffBoundFromOperands_of_halfQ_le
  {BA BB BRaw : Nat}
  (hHalfQ : halfQ ≤ BRaw) :
  mulRqRawCoeffBoundFromOperands BA BB BRaw := by
  exact mulRqRawCoeffBoundFromOperands_mono
    (mulRqRawCoeffBoundFromOperands_halfQ BA BB)
    hHalfQ

theorem mulRqRawCoeffBoundFromOperands_iff_inRange
  {BA BB BRaw : Nat} :
  mulRqRawCoeffBoundFromOperands BA BB BRaw ↔
    mulRqRawInRangeBoundFromOperands BA BB BRaw := by
  constructor
  · exact mulRqRawInRangeBoundFromOperands_of_rawCoeff
  · exact mulRqRawCoeffBoundFromOperands_of_inRange

theorem normInfCoeffs_mulRq_le_of_operand_norm_assumptions
  {a b : Coeffs} {BA BB BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact normInfCoeffs_mulRq_le_of_rawCoeffsNorm
    (a := a) (b := b) (BRaw := BRaw) (B := B)
    (hRawCoeffs := hRawFromOperands a b hA hB)
    (hAddSub := hAddSub)
    (hSub := hSub)

theorem normInfCoeffs_mulRq_le_of_operand_norm_assumptions_inRange
  {a b : Coeffs} {BA BB BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawInRangeFromOperands : mulRqRawInRangeBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact normInfCoeffs_mulRq_le_of_operand_norm_assumptions
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw) (B := B)
    hA hB
    (mulRqRawNormBoundFromOperands_of_inRange hRawInRangeFromOperands)
    hAddSub hSub

/--
Field-op collapse variant: callers provide `x+y` and `x-y` bounds at the same `B`,
and we internally derive the needed `x+y-z` bound.
This is the common Goldilocks-style collapse surface (`BRaw = B`).
-/
theorem normInfCoeffs_mulRq_le_of_operand_norm_assumptions_fieldOp
  {a b : Coeffs} {BA BB B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB B)
  (hOps : rawFieldOpCollapseBound B B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  rcases hOps with ⟨hAdd, hSub⟩
  exact normInfCoeffs_mulRq_le_of_operand_norm_assumptions
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := B) (B := B)
    hA hB hRawFromOperands
    (rawAddSubCollapseBound_of_add_and_sub_same (BRaw := B) hAdd hSub)
    hSub

theorem normInfCoeffs_mulRq_le_of_operand_norm_assumptions_rawCoeff
  {a b : Coeffs} {BA BB BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact normInfCoeffs_mulRq_le_of_operand_norm_assumptions
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw) (B := B)
    hA hB
    (mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    hAddSub hSub

theorem normInfCoeffs_mulRq_le_of_operand_norm_assumptions_rawCoeff_fieldOp
  {a b : Coeffs} {BA BB B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB B)
  (hOps : rawFieldOpCollapseBound B B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact normInfCoeffs_mulRq_le_of_operand_norm_assumptions_fieldOp
    (a := a) (b := b) (BA := BA) (BB := BB) (B := B)
    hA hB
    (mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    hOps

/--
Convenience wrapper: derive the `mulRq` norm bound directly from theorem-native
schoolbook-term assumptions, then collapse through raw add/sub bounds.
-/
theorem normInfCoeffs_mulRq_le_of_operand_norm_assumptions_via_schoolbook
  {a b : Coeffs} {BA BB BTerm BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAdd : ∀ x y : F, normInfF x ≤ BRaw → normInfF y ≤ BTerm → normInfF (x + y) ≤ BRaw)
  (hZero : normInfF (0 : F) ≤ BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact normInfCoeffs_mulRq_le_of_operand_norm_assumptions_rawCoeff
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw) (B := B)
    hA hB
    (mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions
      (hMul := hMul) (hAdd := hAdd) (hZero := hZero))
    hAddSub hSub

/--
Sum-style schoolbook variant:
build raw bounds using only per-term multiplication bounds + triangle addition,
with derived raw bound `((D * D) * BTerm)`.
-/
theorem normInfCoeffs_mulRq_le_of_operand_norm_assumptions_via_schoolbook_sum
  {a b : Coeffs} {BA BB BTerm B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y)
  (hAddSub : rawAddSubCollapseBound ((D * D) * BTerm) B)
  (hSub : rawSubCollapseBound ((D * D) * BTerm) B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact normInfCoeffs_mulRq_le_of_operand_norm_assumptions_rawCoeff
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := (D * D) * BTerm) (B := B)
    hA hB
    (mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions_sum
      (hMul := hMul) (hAddTri := hAddTri))
    hAddSub hSub

theorem normInfCoeffs_mulRq_le_of_operand_norm_assumptions_via_schoolbookAssumptionBundles
  {a b : Coeffs} {BA BB BTerm BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMul : schoolbookMulTermBound BA BB BTerm)
  (hAdd : schoolbookAccAddBound BRaw BTerm)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact normInfCoeffs_mulRq_le_of_operand_norm_assumptions_rawCoeff
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw) (B := B)
    hA hB
    (mulRqRawCoeffBoundFromOperands_of_schoolbookAssumptionBundles
      (hMul := hMul) (hAdd := hAdd))
    hAddSub hSub

theorem normInfCoeffs_mulRq_le_of_operand_norm_assumptions_via_schoolbookAssumptionBundles_sum
  {a b : Coeffs} {BA BB BTerm B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMul : schoolbookMulTermBound BA BB BTerm)
  (hAddTri : schoolbookAddTriangleBound)
  (hAddSub : rawAddSubCollapseBound ((D * D) * BTerm) B)
  (hSub : rawSubCollapseBound ((D * D) * BTerm) B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact normInfCoeffs_mulRq_le_of_operand_norm_assumptions_rawCoeff
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := (D * D) * BTerm) (B := B)
    hA hB
    (mulRqRawCoeffBoundFromOperands_of_schoolbookAssumptionBundles_sum
      (hMul := hMul) (hAddTri := hAddTri))
    hAddSub hSub

/--
Theorem-native non-coarse bridge:
if universal multiplication and add/sub triangle blockers are available,
derive a complete `mulRq` norm bound through the schoolbook-sum path.
-/
theorem normInfCoeffs_mulRq_le_of_universal_blockers
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound) :
  normInfCoeffs (mulRq a b)
    ≤ ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) := by
  let BRaw : Nat := (D * D) * (BA * BB)
  have hAddSub : rawAddSubCollapseBound BRaw (BRaw + BRaw + BRaw) :=
    rawAddSubCollapseBound_of_triangles (BRaw := BRaw) hAddTri hSubTri
  have hSubBase : rawSubCollapseBound BRaw (BRaw + BRaw) :=
    rawSubCollapseBound_of_triangle (BRaw := BRaw) hSubTri
  have hSub : rawSubCollapseBound BRaw (BRaw + BRaw + BRaw) :=
    rawSubCollapseBound_mono hSubBase (Nat.le_add_right (BRaw + BRaw) BRaw)
  exact normInfCoeffs_mulRq_le_of_operand_norm_assumptions
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw) (B := BRaw + BRaw + BRaw)
    hA hB
    (mulRqRawNormBoundFromOperands_of_universal_mul_and_add_sum
      (BA := BA) (BB := BB) hMulUniv hAddTri)
    hAddSub hSub

/--
Triangle-bundle variant of `normInfCoeffs_mulRq_le_of_universal_blockers`.
-/
theorem normInfCoeffs_mulRq_le_of_universal_mul_and_triangles
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hTri : schoolbookTriangleBounds) :
  normInfCoeffs (mulRq a b)
    ≤ ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) := by
  exact normInfCoeffs_mulRq_le_of_universal_blockers
    (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB hMulUniv
    (schoolbookTriangleBounds_add hTri)
    (schoolbookTriangleBounds_sub hTri)

theorem normInfCoeffs_mulRq_le_of_universal_mul_and_add
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound) :
  normInfCoeffs (mulRq a b)
    ≤ ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) := by
  exact normInfCoeffs_mulRq_le_of_universal_mul_and_triangles
    (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB hMulUniv (schoolbookTriangleBounds_of_add hAddTri)

/--
Parameterized non-coarse bridge:
if a tighter raw schoolbook bound `BRaw` is available from operand norms,
combine it with universal add/sub triangle blockers to obtain a `mulRq` bound at
`BRaw + BRaw + BRaw`.
-/
theorem normInfCoeffs_mulRq_le_of_universal_blockers_and_raw
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound) :
  normInfCoeffs (mulRq a b) ≤ BRaw + BRaw + BRaw := by
  have hAddSub : rawAddSubCollapseBound BRaw (BRaw + BRaw + BRaw) :=
    rawAddSubCollapseBound_of_triangles (BRaw := BRaw) hAddTri hSubTri
  have hSubBase : rawSubCollapseBound BRaw (BRaw + BRaw) :=
    rawSubCollapseBound_of_triangle (BRaw := BRaw) hSubTri
  have hSub : rawSubCollapseBound BRaw (BRaw + BRaw + BRaw) :=
    rawSubCollapseBound_mono hSubBase (Nat.le_add_right (BRaw + BRaw) BRaw)
  exact normInfCoeffs_mulRq_le_of_operand_norm_assumptions
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw) (B := BRaw + BRaw + BRaw)
    hA hB hRawFromOperands hAddSub hSub

/--
Triangle-bundle variant of `normInfCoeffs_mulRq_le_of_universal_blockers_and_raw`.
-/
theorem normInfCoeffs_mulRq_le_of_universal_mul_and_triangles_and_raw
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hTri : schoolbookTriangleBounds) :
  normInfCoeffs (mulRq a b) ≤ BRaw + BRaw + BRaw := by
  exact normInfCoeffs_mulRq_le_of_universal_blockers_and_raw
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB hRawFromOperands
    (schoolbookTriangleBounds_add hTri)
    (schoolbookTriangleBounds_sub hTri)

theorem normInfCoeffs_mulRq_le_of_universal_mul_and_add_and_raw
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound) :
  normInfCoeffs (mulRq a b) ≤ BRaw + BRaw + BRaw := by
  exact normInfCoeffs_mulRq_le_of_universal_mul_and_triangles_and_raw
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB hRawFromOperands (schoolbookTriangleBounds_of_add hAddTri)

/--
Tighter universal-blocker bridge:
use the refined raw schoolbook coefficient bound `D * (BA * BB)` from the
sum-tight path, then collapse to `3 * (D * (BA * BB))`.
-/
theorem normInfCoeffs_mulRq_le_of_universal_blockers_tight
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound) :
  normInfCoeffs (mulRq a b)
    ≤ (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB)) := by
  let BRaw : Nat := D * (BA * BB)
  exact normInfCoeffs_mulRq_le_of_universal_blockers_and_raw
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB
    (mulRqRawNormBoundFromOperands_of_universal_mul_and_add_sum_tight
      (BA := BA) (BB := BB) hMulUniv hAddTri)
    hAddTri hSubTri

/--
Triangle-bundle variant of `normInfCoeffs_mulRq_le_of_universal_blockers_tight`.
-/
theorem normInfCoeffs_mulRq_le_of_universal_mul_and_triangles_tight
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hTri : schoolbookTriangleBounds) :
  normInfCoeffs (mulRq a b)
    ≤ (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB)) := by
  exact normInfCoeffs_mulRq_le_of_universal_blockers_tight
    (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB hMulUniv
    (schoolbookTriangleBounds_add hTri)
    (schoolbookTriangleBounds_sub hTri)

theorem normInfCoeffs_mulRq_le_of_universal_mul_and_add_tight
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound) :
  normInfCoeffs (mulRq a b)
    ≤ (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB)) := by
  exact normInfCoeffs_mulRq_le_of_universal_mul_and_triangles_tight
    (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB hMulUniv (schoolbookTriangleBounds_of_add hAddTri)

theorem normInfCoeffs_mulRq_le_of_centeredRep_mul_and_add_tight
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  normInfCoeffs (mulRq a b)
    ≤ (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB)) := by
  exact normInfCoeffs_mulRq_le_of_universal_mul_and_add_tight
    (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB
    (schoolbookMulUniversalBound_of_centeredRep hMulRep)
    (schoolbookAddTriangleBound_of_centeredRep hAddRep)

theorem normInfCoeffs_mulRq_le_of_centeredRepMulAddBounds_tight
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRep : centeredRepMulAddBounds) :
  normInfCoeffs (mulRq a b)
    ≤ (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB)) := by
  exact normInfCoeffs_mulRq_le_of_centeredRep_mul_and_add_tight
    (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB
    (centeredRepMulAddBounds_mul hRep)
    (centeredRepMulAddBounds_add hRep)

theorem normInfCoeffs_mulRq_le_of_centeredRep_mul_tight
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y)) :
  normInfCoeffs (mulRq a b)
    ≤ (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB)) := by
  exact normInfCoeffs_mulRq_le_of_centeredRepMulAddBounds_tight
    (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB (centeredRepMulAddBounds_of_mul hMulRep)

theorem normInfCoeffs_mulRq_le_of_centeredRep_mul_and_triangles_tight
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hTriRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  normInfCoeffs (mulRq a b)
    ≤ (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB)) := by
  exact normInfCoeffs_mulRq_le_of_universal_mul_and_triangles_tight
    (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB
    (schoolbookMulUniversalBound_of_centeredRep hMulRep)
    (schoolbookTriangleBounds_of_centeredRep_add hTriRep)

theorem normInfCoeffs_mulRq_le_of_centeredRep_mul_and_triangles
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hTriRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  normInfCoeffs (mulRq a b)
    ≤ ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) := by
  exact normInfCoeffs_mulRq_le_of_universal_mul_and_triangles
    (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB
    (schoolbookMulUniversalBound_of_centeredRep hMulRep)
    (schoolbookTriangleBounds_of_centeredRep_add hTriRep)

theorem normInfCoeffs_mulRq_le_of_centeredRep_mul_and_add
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  normInfCoeffs (mulRq a b)
    ≤ ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) := by
  exact normInfCoeffs_mulRq_le_of_centeredRep_mul_and_triangles
    (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB hMulRep hAddRep

theorem normInfCoeffs_mulRq_le_of_centeredRepMulAddBounds
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRep : centeredRepMulAddBounds) :
  normInfCoeffs (mulRq a b)
    ≤ ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) := by
  exact normInfCoeffs_mulRq_le_of_centeredRep_mul_and_add
    (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB
    (centeredRepMulAddBounds_mul hRep)
    (centeredRepMulAddBounds_add hRep)

theorem normInfCoeffs_mulRq_le_of_centeredRep_mul
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y)) :
  normInfCoeffs (mulRq a b)
    ≤ ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) := by
  exact normInfCoeffs_mulRq_le_of_centeredRepMulAddBounds
    (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB (centeredRepMulAddBounds_of_mul hMulRep)

/--
Assumption-free coarse non-coarse-P5 `mulRq` norm bound from proven universal
mul/add blockers.
-/
theorem normInfCoeffs_mulRq_le_native
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB) :
  normInfCoeffs (mulRq a b)
    ≤ ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) := by
  exact normInfCoeffs_mulRq_le_of_universal_mul_and_triangles
    (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB
    schoolbookMulUniversalBound_theorem
    schoolbookTriangleBounds_theorem

/--
Assumption-free tight non-coarse-P5 `mulRq` norm bound from proven universal
mul/add blockers.
-/
theorem normInfCoeffs_mulRq_le_native_tight
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB) :
  normInfCoeffs (mulRq a b)
    ≤ (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB)) := by
  exact normInfCoeffs_mulRq_le_of_universal_mul_and_triangles_tight
    (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB
    schoolbookMulUniversalBound_theorem
    schoolbookTriangleBounds_theorem

/--
Assumption-free collapse bridge for the parameterized raw-bound path:
if callers provide a raw schoolbook bound `BRaw`, we close `mulRq` at `3 * BRaw`
without extra triangle assumptions.
-/
theorem normInfCoeffs_mulRq_le_native_and_raw
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw) :
  normInfCoeffs (mulRq a b) ≤ BRaw + BRaw + BRaw := by
  exact normInfCoeffs_mulRq_le_of_universal_mul_and_triangles_and_raw
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB hRawFromOperands
    schoolbookTriangleBounds_theorem

/--
Raw-coeff form of `normInfCoeffs_mulRq_le_native_and_raw`.
-/
theorem normInfCoeffs_mulRq_le_native_and_rawCoeff
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw) :
  normInfCoeffs (mulRq a b) ≤ BRaw + BRaw + BRaw := by
  exact normInfCoeffs_mulRq_le_native_and_raw
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB (mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)

/--
Raw-coefficient variant of `normInfCoeffs_mulRq_le_of_universal_blockers_and_raw`.
Useful when the tight schoolbook result is first established at `mulRqRawCoeffSpec`.
-/
theorem normInfCoeffs_mulRq_le_of_universal_blockers_and_rawCoeff
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound) :
  normInfCoeffs (mulRq a b) ≤ BRaw + BRaw + BRaw := by
  exact normInfCoeffs_mulRq_le_of_universal_blockers_and_raw
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB
    (mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    hAddTri hSubTri

/--
Triangle-bundle variant of `normInfCoeffs_mulRq_le_of_universal_blockers_and_rawCoeff`.
-/
theorem normInfCoeffs_mulRq_le_of_universal_mul_and_triangles_and_rawCoeff
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hTri : schoolbookTriangleBounds) :
  normInfCoeffs (mulRq a b) ≤ BRaw + BRaw + BRaw := by
  exact normInfCoeffs_mulRq_le_of_universal_blockers_and_rawCoeff
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB hRawCoeffFromOperands
    (schoolbookTriangleBounds_add hTri)
    (schoolbookTriangleBounds_sub hTri)

theorem normInfCoeffs_mulRq_le_of_universal_mul_and_add_and_rawCoeff
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound) :
  normInfCoeffs (mulRq a b) ≤ BRaw + BRaw + BRaw := by
  exact normInfCoeffs_mulRq_le_of_universal_mul_and_triangles_and_rawCoeff
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB hRawCoeffFromOperands (schoolbookTriangleBounds_of_add hAddTri)

theorem normInfCoeffs_mulRq_le_of_centeredRep_mul_and_add_and_raw
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  normInfCoeffs (mulRq a b) ≤ BRaw + BRaw + BRaw := by
  exact normInfCoeffs_mulRq_le_of_universal_mul_and_add_and_raw
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB hRawFromOperands
    (schoolbookAddTriangleBound_of_centeredRep hAddRep)

theorem normInfCoeffs_mulRq_le_of_centeredRepMulAddBounds_and_raw
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hRep : centeredRepMulAddBounds) :
  normInfCoeffs (mulRq a b) ≤ BRaw + BRaw + BRaw := by
  exact normInfCoeffs_mulRq_le_of_centeredRep_mul_and_add_and_raw
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB hRawFromOperands
    (centeredRepMulAddBounds_add hRep)

theorem normInfCoeffs_mulRq_le_of_centeredRep_mul_and_triangles_and_raw
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hTriRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  normInfCoeffs (mulRq a b) ≤ BRaw + BRaw + BRaw := by
  exact normInfCoeffs_mulRq_le_of_universal_mul_and_triangles_and_raw
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB hRawFromOperands
    (schoolbookTriangleBounds_of_centeredRep_add hTriRep)

theorem normInfCoeffs_mulRq_le_of_centeredRep_mul_and_add_and_rawCoeff
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  normInfCoeffs (mulRq a b) ≤ BRaw + BRaw + BRaw := by
  exact normInfCoeffs_mulRq_le_of_centeredRep_mul_and_add_and_raw
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB
    (mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    hAddRep

theorem normInfCoeffs_mulRq_le_of_centeredRep_mul_and_triangles_and_rawCoeff
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hTriRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y)) :
  normInfCoeffs (mulRq a b) ≤ BRaw + BRaw + BRaw := by
  exact normInfCoeffs_mulRq_le_of_centeredRep_mul_and_triangles_and_raw
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB
    (mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    hTriRep

/--
Field-op-collapse variant of
`normInfCoeffs_mulRq_le_of_operand_norm_assumptions_via_schoolbook`.
-/
theorem normInfCoeffs_mulRq_le_of_operand_norm_assumptions_via_schoolbook_fieldOp
  {a b : Coeffs} {BA BB BTerm B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAdd : ∀ x y : F, normInfF x ≤ B → normInfF y ≤ BTerm → normInfF (x + y) ≤ B)
  (hZero : normInfF (0 : F) ≤ B)
  (hOps : rawFieldOpCollapseBound B B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact normInfCoeffs_mulRq_le_of_operand_norm_assumptions_rawCoeff_fieldOp
    (a := a) (b := b) (BA := BA) (BB := BB) (B := B)
    hA hB
    (mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions
      (hMul := hMul) (hAdd := hAdd) (hZero := hZero))
    hOps

theorem normInfCoeffs_mulRq_le_of_operand_norm_assumptions_via_schoolbook_sum_fieldOp
  {a b : Coeffs} {BA BB BTerm : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y)
  (hOps : rawFieldOpCollapseBound ((D * D) * BTerm) ((D * D) * BTerm)) :
  normInfCoeffs (mulRq a b) ≤ ((D * D) * BTerm) := by
  exact normInfCoeffs_mulRq_le_of_operand_norm_assumptions_rawCoeff_fieldOp
    (a := a) (b := b) (BA := BA) (BB := BB) (B := (D * D) * BTerm)
    hA hB
    (mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions_sum
      (hMul := hMul) (hAddTri := hAddTri))
    hOps

theorem normInfCoeffs_mulRq_le_of_operand_norm_assumptions_via_schoolbook_of_term_le
  {a b : Coeffs} {BA BB BTerm BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hTermLe : BTerm ≤ BRaw)
  (hAddCollapse : rawAddCollapseBound BRaw BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact normInfCoeffs_mulRq_le_of_operand_norm_assumptions_rawCoeff
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw) (B := B)
    hA hB
    (mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions_of_term_le
      (hMul := hMul) (hTermLe := hTermLe) (hAddCollapse := hAddCollapse))
    hAddSub hSub

theorem normInfCoeffs_mulRq_le_of_operand_norm_assumptions_via_schoolbook_sameBound
  {a b : Coeffs} {BA BB BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BRaw)
  (hAddCollapse : rawAddCollapseBound BRaw BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact normInfCoeffs_mulRq_le_of_operand_norm_assumptions_via_schoolbook_of_term_le
    (a := a) (b := b) (BA := BA) (BB := BB) (BTerm := BRaw) (BRaw := BRaw) (B := B)
    hA hB hMul (Nat.le_refl BRaw) hAddCollapse hAddSub hSub

theorem normInfCoeffs_mulRq_le_of_operand_norm_assumptions_via_schoolbook_sameBound_fieldOp
  {a b : Coeffs} {BA BB B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ B)
  (hOps : rawFieldOpCollapseBound B B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact normInfCoeffs_mulRq_le_of_operand_norm_assumptions_rawCoeff_fieldOp
    (a := a) (b := b) (BA := BA) (BB := BB) (B := B)
    hA hB
    (mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions_sameBound
      (hMul := hMul)
      (hAddCollapse := hOps.1))
    hOps

theorem normInfCoeffs_mulRq_le_of_operand_norm_assumptions_fieldOp_inRange
  {a b : Coeffs} {BA BB B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawInRangeFromOperands : mulRqRawInRangeBoundFromOperands BA BB B)
  (hOps : rawFieldOpCollapseBound B B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  rcases hOps with ⟨hAdd, hSub⟩
  exact normInfCoeffs_mulRq_le_of_operand_norm_assumptions_inRange
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := B) (B := B)
    hA hB hRawInRangeFromOperands
    (rawAddSubCollapseBound_of_add_and_sub_same (BRaw := B) hAdd hSub)
    hSub

theorem normInfF_add_sub_le_halfQ
  (x y z : F) :
  normInfF (x + y - z) ≤ halfQ := by
  exact normInfF_sub_le_halfQ (x + y) z

theorem rawAddSubCollapseBound_halfQ :
  rawAddSubCollapseBound halfQ halfQ := by
  intro x y z _hx _hy _hz
  exact normInfF_add_sub_le_halfQ x y z

theorem rawAddCollapseBound_halfQ :
  rawAddCollapseBound halfQ halfQ := by
  intro x y _hx _hy
  exact normInfF_add_le_halfQ x y

theorem rawSubCollapseBound_halfQ :
  rawSubCollapseBound halfQ halfQ := by
  intro x y _hx _hy
  exact normInfF_sub_le_halfQ x y

theorem rawFieldOpCollapseBound_halfQ :
  rawFieldOpCollapseBound halfQ halfQ := by
  exact ⟨rawAddCollapseBound_halfQ, rawSubCollapseBound_halfQ⟩

theorem rawAddSubCollapseBound_of_halfQ_le
  {B : Nat}
  (hHalfQ : halfQ ≤ B) :
  rawAddSubCollapseBound halfQ B := by
  exact rawAddSubCollapseBound_mono rawAddSubCollapseBound_halfQ hHalfQ

theorem rawSubCollapseBound_of_halfQ_le
  {B : Nat}
  (hHalfQ : halfQ ≤ B) :
  rawSubCollapseBound halfQ B := by
  exact rawSubCollapseBound_mono rawSubCollapseBound_halfQ hHalfQ

theorem rawAddCollapseBound_of_halfQ_le
  {B : Nat}
  (hHalfQ : halfQ ≤ B) :
  rawAddCollapseBound halfQ B := by
  exact rawAddCollapseBound_mono rawAddCollapseBound_halfQ hHalfQ

theorem rawFieldOpCollapseBound_of_halfQ_le
  {B : Nat}
  (hHalfQ : halfQ ≤ B) :
  rawFieldOpCollapseBound halfQ B := by
  exact rawFieldOpCollapseBound_mono rawFieldOpCollapseBound_halfQ hHalfQ

/--
Concrete raw-coefficient fallback: in Goldilocks, every field element is centered-bounded
by `halfQ`, so raw schoolbook coefficients satisfy this bound without extra assumptions.
-/
theorem normInfF_mulRqRawCoeffSpec_le_halfQ
  (a b : Coeffs) (t : Nat) :
  normInfF (mulRqRawCoeffSpec a b t) ≤ halfQ := by
  simpa using normInfF_le_halfQ (mulRqRawCoeffSpec a b t)

/--
Assumption-free raw-coefficient bridge for multiplication bounds (coarse concrete bound).
This discharges the `hRawFromNorm`/`hAddSub`/`hSub` obligations internally at `halfQ`.
-/
theorem normInfCoeffs_mulRq_le_of_norm_bounds_via_rawCoeff_halfQ
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB) :
  normInfCoeffs (mulRq a b) ≤ halfQ := by
  exact normInfCoeffs_mulRq_le_of_norm_bounds_via_rawCoeff
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := halfQ) (B := halfQ)
    hA hB
    (hRawFromNorm := fun t _ _ => normInfF_mulRqRawCoeffSpec_le_halfQ a b t)
    (hAddSub := fun x y z _ _ _ => normInfF_add_sub_le_halfQ x y z)
    (hSub := fun x y _ _ => normInfF_sub_le_halfQ x y)

theorem normInfCoeffs_mulRq_le_of_norm_bounds_via_rawCoeff_halfQ_inRange
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawFromNormInRange :
    ∀ t, t < 2 * D - 1 →
      normInfCoeffs a ≤ BA →
      normInfCoeffs b ≤ BB →
      normInfF (mulRqRawCoeffSpec a b t) ≤ halfQ) :
  normInfCoeffs (mulRq a b) ≤ halfQ := by
  exact normInfCoeffs_mulRq_le_of_norm_bounds_via_rawCoeff_inRange
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := halfQ) (B := halfQ)
    hA hB
    hRawFromNormInRange
    (hAddSub := fun x y z _ _ _ => normInfF_add_sub_le_halfQ x y z)
    (hSub := fun x y _ _ => normInfF_sub_le_halfQ x y)

theorem normInfCoeffs_mulRqRawCoeffs_le_halfQ
  (a b : Coeffs) :
  normInfCoeffs (mulRqRawCoeffs a b) ≤ halfQ := by
  simpa using normInfCoeffs_le_halfQ (mulRqRawCoeffs a b)

theorem normInfCoeffs_mulRq_le_of_operand_norm_assumptions_halfQ
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB halfQ) :
  normInfCoeffs (mulRq a b) ≤ halfQ := by
  exact normInfCoeffs_mulRq_le_of_operand_norm_assumptions
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := halfQ) (B := halfQ)
    hA hB hRawFromOperands rawAddSubCollapseBound_halfQ rawSubCollapseBound_halfQ

theorem normInfCoeffs_mulRq_le_of_operand_norm_assumptions_halfQ_inRange
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawInRangeFromOperands : mulRqRawInRangeBoundFromOperands BA BB halfQ) :
  normInfCoeffs (mulRq a b) ≤ halfQ := by
  exact normInfCoeffs_mulRq_le_of_operand_norm_assumptions_inRange
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := halfQ) (B := halfQ)
    hA hB hRawInRangeFromOperands rawAddSubCollapseBound_halfQ rawSubCollapseBound_halfQ

theorem normInfCoeffs_mulRq_le_of_rawCoeffInRangeBound_halfQ
  {a b : Coeffs}
  (hRawInRange : ∀ t, t < 2 * D - 1 → normInfF (mulRqRawCoeffSpec a b t) ≤ halfQ) :
  normInfCoeffs (mulRq a b) ≤ halfQ := by
  exact normInfCoeffs_mulRq_le_of_rawCoeffInRangeBound (a := a) (b := b) (BRaw := halfQ) (B := halfQ)
    hRawInRange
    (hAddSub := fun x y z _ _ _ => normInfF_add_sub_le_halfQ x y z)
    (hSub := fun x y _ _ => normInfF_sub_le_halfQ x y)

theorem normInfCoeffs_mulRq_le_of_rawCoeffsNorm_halfQ
  {a b : Coeffs}
  (hRawCoeffs : normInfCoeffs (mulRqRawCoeffs a b) ≤ halfQ) :
  normInfCoeffs (mulRq a b) ≤ halfQ := by
  exact normInfCoeffs_mulRq_le_of_rawCoeffsNorm (a := a) (b := b) (BRaw := halfQ) (B := halfQ)
    hRawCoeffs
    (hAddSub := fun x y z _ _ _ => normInfF_add_sub_le_halfQ x y z)
    (hSub := fun x y _ _ => normInfF_sub_le_halfQ x y)

theorem normInfCoeffs_mulRq_le_of_norm_bounds_via_rawCoeffsNorm_halfQ
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hRawCoeffsFromNorm :
    normInfCoeffs a ≤ BA →
    normInfCoeffs b ≤ BB →
    normInfCoeffs (mulRqRawCoeffs a b) ≤ halfQ) :
  normInfCoeffs (mulRq a b) ≤ halfQ := by
  exact normInfCoeffs_mulRq_le_of_rawCoeffsNorm_halfQ
    (hRawCoeffs := hRawCoeffsFromNorm hA hB)

theorem normInfCoeffs_mulRq_le_of_rawCoeffInRangeBound_halfQ_le
  {a b : Coeffs} {B : Nat}
  (hRawInRange : ∀ t, t < 2 * D - 1 → normInfF (mulRqRawCoeffSpec a b t) ≤ halfQ)
  (hHalfQ : halfQ ≤ B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact Nat.le_trans
    (normInfCoeffs_mulRq_le_of_rawCoeffInRangeBound_halfQ (a := a) (b := b) hRawInRange)
    hHalfQ

theorem normInfCoeffs_mulRq_le_of_rawCoeffsNorm_halfQ_le
  {a b : Coeffs} {B : Nat}
  (hRawCoeffs : normInfCoeffs (mulRqRawCoeffs a b) ≤ halfQ)
  (hHalfQ : halfQ ≤ B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact Nat.le_trans
    (normInfCoeffs_mulRq_le_of_rawCoeffsNorm_halfQ (a := a) (b := b) hRawCoeffs)
    hHalfQ

theorem normInfCoeffs_mulRq_le_of_norm_bounds_via_rawCoeff_halfQ_le
  {a b : Coeffs} {BA BB B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hHalfQ : halfQ ≤ B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact Nat.le_trans
    (normInfCoeffs_mulRq_le_of_norm_bounds_via_rawCoeff_halfQ hA hB)
    hHalfQ

/--
Challenge-coefficient predicate used by SuperNeo concrete parameters:
coefficients are centered in `{-2,-1,0,1,2}`.
-/
def IsChallengeCoeff (x : F) : Prop :=
  x = F.ofInt (-2) ∨ x = F.ofInt (-1) ∨ x = 0 ∨ x = 1 ∨ x = 2

instance isChallengeCoeff_decidable (x : F) : Decidable (IsChallengeCoeff x) := by
  unfold IsChallengeCoeff
  infer_instance

theorem normInfF_ofInt_neg_two : normInfF (F.ofInt (-2)) = 2 := by
  native_decide

theorem normInfF_ofInt_neg_one : normInfF (F.ofInt (-1)) = 1 := by
  native_decide

theorem normInfF_zero : normInfF (0 : F) = 0 := by
  native_decide

theorem normInfF_one : normInfF (1 : F) = 1 := by
  native_decide

theorem normInfF_two : normInfF (2 : F) = 2 := by
  native_decide

theorem normInfF_le_two_of_isChallengeCoeff
  {x : F}
  (hx : IsChallengeCoeff x) :
  normInfF x ≤ 2 := by
  rcases hx with hx | hx | hx | hx | hx
  · simpa [hx, normInfF_ofInt_neg_two]
  · simpa [hx, normInfF_ofInt_neg_one]
  · simpa [hx, normInfF_zero]
  · simpa [hx, normInfF_one]
  · simpa [hx, normInfF_two]

theorem normInfF_sub_le_four_of_isChallengeCoeff
  {x y : F}
  (hx : IsChallengeCoeff x)
  (hy : IsChallengeCoeff y) :
  normInfF (x - y) ≤ 4 := by
  rcases hx with hx | hx | hx | hx | hx <;>
  rcases hy with hy | hy | hy | hy | hy <;>
  subst x <;>
  subst y <;>
  native_decide

theorem normInfF_add_le_four_of_isChallengeCoeff
  {x y : F}
  (hx : IsChallengeCoeff x)
  (hy : IsChallengeCoeff y) :
  normInfF (x + y) ≤ 4 := by
  rcases hx with hx | hx | hx | hx | hx <;>
  rcases hy with hy | hy | hy | hy | hy <;>
  subst x <;>
  subst y <;>
  native_decide

theorem normInfF_mul_le_four_of_isChallengeCoeff
  {x y : F}
  (hx : IsChallengeCoeff x)
  (hy : IsChallengeCoeff y) :
  normInfF (x * y) ≤ 4 := by
  rcases hx with hx | hx | hx | hx | hx <;>
  rcases hy with hy | hy | hy | hy | hy <;>
  subst x <;>
  subst y <;>
  native_decide

def AllChallengeCoeffs (a : Coeffs) : Prop :=
  ∀ i (hi : i < a.size), IsChallengeCoeff (a[i]'hi)

instance allChallengeCoeffs_decidable (a : Coeffs) : Decidable (AllChallengeCoeffs a) := by
  unfold AllChallengeCoeffs
  infer_instance

theorem allChallengeCoeffs_of_all
  {a : Coeffs}
  (hAll : a.all (fun x => decide (IsChallengeCoeff x)) = true) :
  AllChallengeCoeffs a := by
  intro i hi
  exact decide_eq_true_eq.mp ((Array.all_eq_true.mp hAll) i hi)

theorem all_eq_true_of_allChallengeCoeffs
  {a : Coeffs}
  (hAll : AllChallengeCoeffs a) :
  a.all (fun x => decide (IsChallengeCoeff x)) = true := by
  apply (Array.all_eq_true).2
  intro i hi
  exact decide_eq_true (hAll i hi)

theorem normInfCoeffs_le_four_of_allChallenge
  {a : Coeffs}
  (hAll : AllChallengeCoeffs a) :
  normInfCoeffs a ≤ 4 := by
  exact normInfCoeffs_le_of_entry_bound (a := a) (B := 4) (fun i hi => by
    have hEntry : normInfF (a[i]'hi) ≤ 2 := normInfF_le_two_of_isChallengeCoeff (hAll i hi)
    exact Nat.le_trans hEntry (by decide))

theorem normInfCoeffMatrix_le_four_of_allChallenge
  {m : Array Coeffs}
  (hAll : ∀ i (hi : i < m.size), AllChallengeCoeffs (m[i]'hi)) :
  normInfCoeffMatrix m ≤ 4 := by
  exact normInfCoeffMatrix_le_of_row_bound (m := m) (B := 4) (fun i hi => by
    simpa using normInfCoeffs_le_four_of_allChallenge (hAll i hi))

theorem normInfCoeffs_le_four_of_allChallenge_sub
  {a b : Coeffs}
  (hSize : a.size = b.size)
  (hAllA : AllChallengeCoeffs a)
  (hAllB : AllChallengeCoeffs b) :
  normInfCoeffs (coeffSub a b) ≤ 4 := by
  apply normInfCoeffs_le_of_entry_bound
  intro i hi
  have hiA : i < a.size := by
    simpa [coeffSub_size_of_eq hSize] using hi
  have hiB : i < b.size := by
    simpa [hSize] using hiA
  have hA : IsChallengeCoeff (a[i]'hiA) := hAllA i hiA
  have hB : IsChallengeCoeff (b[i]'hiB) := hAllB i hiB
  have hSub : normInfF ((a[i]'hiA) - (b[i]'hiB)) ≤ 4 :=
    normInfF_sub_le_four_of_isChallengeCoeff hA hB
  simpa [coeffSub, hSize, hiA, hiB] using hSub

theorem normInfCoeffs_le_four_of_allChallenge_add
  {a b : Coeffs}
  (hSize : a.size = b.size)
  (hAllA : AllChallengeCoeffs a)
  (hAllB : AllChallengeCoeffs b) :
  normInfCoeffs (vecAdd a b) ≤ 4 := by
  apply normInfCoeffs_le_of_entry_bound
  intro i hi
  have hiA : i < a.size := by
    simpa [vecAdd_size_of_eq hSize] using hi
  have hiB : i < b.size := by
    simpa [hSize] using hiA
  have hA : IsChallengeCoeff (a[i]'hiA) := hAllA i hiA
  have hB : IsChallengeCoeff (b[i]'hiB) := hAllB i hiB
  have hAdd : normInfF ((a[i]'hiA) + (b[i]'hiB)) ≤ 4 :=
    normInfF_add_le_four_of_isChallengeCoeff hA hB
  simpa [vecAdd, hSize, hiA, hiB] using hAdd

theorem normInfCoeffs_le_four_of_allChallenge_scale
  {s : F} {a : Coeffs}
  (hS : IsChallengeCoeff s)
  (hAllA : AllChallengeCoeffs a) :
  normInfCoeffs (vecScale s a) ≤ 4 := by
  apply normInfCoeffs_le_of_entry_bound
  intro i hi
  have hiA : i < a.size := by
    simpa [vecScale_size s a] using hi
  have hA : IsChallengeCoeff (a[i]'hiA) := hAllA i hiA
  have hMul : normInfF (s * (a[i]'hiA)) ≤ 4 :=
    normInfF_mul_le_four_of_isChallengeCoeff hS hA
  simpa [vecScale, hiA] using hMul

theorem normInfF_le_of_halfQ_le
  {x : F} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfF x ≤ B := by
  exact Nat.le_trans (normInfF_le_halfQ x) hB

theorem normInfCoeffs_le_of_halfQ_le
  {a : Coeffs} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfCoeffs a ≤ B := by
  exact Nat.le_trans (normInfCoeffs_le_halfQ a) hB

theorem normInfCoeffMatrix_le_of_halfQ_le
  {m : Array Coeffs} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfCoeffMatrix m ≤ B := by
  exact Nat.le_trans (normInfCoeffMatrix_le_halfQ m) hB

theorem normInfF_add_le_of_halfQ_le
  {x y : F} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfF (x + y) ≤ B := by
  exact Nat.le_trans (normInfF_add_le_halfQ x y) hB

theorem normInfF_sub_le_of_halfQ_le
  {x y : F} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfF (x - y) ≤ B := by
  exact Nat.le_trans (normInfF_sub_le_halfQ x y) hB

theorem normInfF_mul_le_of_halfQ_le
  {x y : F} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfF (x * y) ≤ B := by
  exact Nat.le_trans (normInfF_mul_le_halfQ x y) hB

/-- Coarse instantiation of `schoolbookMulTermBound` from the global `halfQ` bound. -/
theorem schoolbookMulTermBound_of_halfQ_le
  {BA BB BTerm : Nat}
  (hHalfQ : halfQ ≤ BTerm) :
  schoolbookMulTermBound BA BB BTerm := by
  intro x y _hx _hy
  exact normInfF_mul_le_of_halfQ_le (x := x) (y := y) hHalfQ

/-- Coarse instantiation of `schoolbookAccAddBound` from the global `halfQ` bound. -/
theorem schoolbookAccAddBound_of_halfQ_le
  {BRaw BTerm : Nat}
  (hHalfQ : halfQ ≤ BRaw) :
  schoolbookAccAddBound BRaw BTerm := by
  intro x y _hx _hy
  exact normInfF_add_le_of_halfQ_le (x := x) (y := y) hHalfQ

theorem normInfCoeffs_vecAdd_le_of_halfQ_le
  {a b : Coeffs} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfCoeffs (vecAdd a b) ≤ B := by
  exact Nat.le_trans (normInfCoeffs_vecAdd_le_halfQ a b) hB

theorem normInfCoeffs_vecScale_le_of_halfQ_le
  {s : F} {a : Coeffs} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfCoeffs (vecScale s a) ≤ B := by
  exact Nat.le_trans (normInfCoeffs_vecScale_le_halfQ s a) hB

theorem normInfCoeffs_mulRq_le_of_halfQ_le
  {a b : Coeffs} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfCoeffs (mulRq a b) ≤ B := by
  exact Nat.le_trans (normInfCoeffs_mulRq_le_halfQ a b) hB

theorem normInfCoeffs_coeffSub_le_of_halfQ_le
  {a b : Coeffs} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfCoeffs (coeffSub a b) ≤ B := by
  exact Nat.le_trans (normInfCoeffs_coeffSub_le_halfQ a b) hB

theorem normInfCoeffs_superneoBarBlock_le_of_halfQ_le
  {bar : Array (Array F)} {v : Coeffs} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfCoeffs (superneoBarBlock bar v) ≤ B := by
  exact Nat.le_trans (normInfCoeffs_superneoBarBlock_le_halfQ bar v) hB

theorem normInfCoeffs_barLiftVec_le_of_halfQ_le
  {bar : Array (Array F)} {v : Array F} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfCoeffs (barLiftVec bar v) ≤ B := by
  exact Nat.le_trans (normInfCoeffs_barLiftVec_le_halfQ bar v) hB

theorem normInfCoeffMatrix_barLiftMatrix_le_of_halfQ_le
  {bar : Array (Array F)} {m : Array (Array F)} {B : Nat}
  (hB : halfQ ≤ B) :
  normInfCoeffMatrix (barLiftMatrix bar m) ≤ B := by
  exact Nat.le_trans (normInfCoeffMatrix_barLiftMatrix_le_halfQ bar m) hB

theorem normInfF_le_B_of_isChallengeCoeff
  {x : F} {B : Nat}
  (hB : 2 ≤ B)
  (hx : IsChallengeCoeff x) :
  normInfF x ≤ B := by
  exact Nat.le_trans (normInfF_le_two_of_isChallengeCoeff hx) hB

theorem normInfF_add_le_B_of_isChallengeCoeff
  {x y : F} {B : Nat}
  (hB : 4 ≤ B)
  (hx : IsChallengeCoeff x)
  (hy : IsChallengeCoeff y) :
  normInfF (x + y) ≤ B := by
  exact Nat.le_trans (normInfF_add_le_four_of_isChallengeCoeff hx hy) hB

theorem normInfF_sub_le_B_of_isChallengeCoeff
  {x y : F} {B : Nat}
  (hB : 4 ≤ B)
  (hx : IsChallengeCoeff x)
  (hy : IsChallengeCoeff y) :
  normInfF (x - y) ≤ B := by
  exact Nat.le_trans (normInfF_sub_le_four_of_isChallengeCoeff hx hy) hB

theorem normInfF_mul_le_B_of_isChallengeCoeff
  {x y : F} {B : Nat}
  (hB : 4 ≤ B)
  (hx : IsChallengeCoeff x)
  (hy : IsChallengeCoeff y) :
  normInfF (x * y) ≤ B := by
  exact Nat.le_trans (normInfF_mul_le_four_of_isChallengeCoeff hx hy) hB

theorem normInfCoeffs_le_B_of_allChallenge
  {a : Coeffs} {B : Nat}
  (hB : 4 ≤ B)
  (hAll : AllChallengeCoeffs a) :
  normInfCoeffs a ≤ B := by
  exact Nat.le_trans (normInfCoeffs_le_four_of_allChallenge hAll) hB

theorem normInfCoeffs_le_B_of_allChallenge_add
  {a b : Coeffs} {B : Nat}
  (hB : 4 ≤ B)
  (hSize : a.size = b.size)
  (hAllA : AllChallengeCoeffs a)
  (hAllB : AllChallengeCoeffs b) :
  normInfCoeffs (vecAdd a b) ≤ B := by
  exact Nat.le_trans (normInfCoeffs_le_four_of_allChallenge_add hSize hAllA hAllB) hB

theorem normInfCoeffs_le_B_of_allChallenge_sub
  {a b : Coeffs} {B : Nat}
  (hB : 4 ≤ B)
  (hSize : a.size = b.size)
  (hAllA : AllChallengeCoeffs a)
  (hAllB : AllChallengeCoeffs b) :
  normInfCoeffs (coeffSub a b) ≤ B := by
  exact Nat.le_trans (normInfCoeffs_le_four_of_allChallenge_sub hSize hAllA hAllB) hB

theorem normInfCoeffs_le_B_of_allChallenge_scale
  {s : F} {a : Coeffs} {B : Nat}
  (hB : 4 ≤ B)
  (hS : IsChallengeCoeff s)
  (hAllA : AllChallengeCoeffs a) :
  normInfCoeffs (vecScale s a) ≤ B := by
  exact Nat.le_trans (normInfCoeffs_le_four_of_allChallenge_scale hS hAllA) hB

end SuperNeo
