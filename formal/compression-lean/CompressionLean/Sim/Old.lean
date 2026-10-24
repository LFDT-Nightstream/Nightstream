import CompressionLean.Sim.Instr

/-!
Owns the old calculus while the simulation moves to `Calculus.lean`:
`OldEnsures` (a precondition `P` and the instruction shape in one
predicate), `Simulates` (`OldEnsures` of a result and of checks) and
`OldEnsuresAt`. `Simulates.of_ensures` and `OldEnsures.of_ensures` state a
new lemma at an old precondition, so a file that still uses the old calculus
reads the lemmas of a file that does not.
-/

namespace CompressionLean.Sim

open CompressionLean.Circuit
open CompressionLean.Lowering
open CompressionLean.Replay (Word Ext)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

/-- **A builder program, simulated for soundness.** `Simulates` in one direction:
when the run's checks hold, `b`'s result meets `Q`, and `Q` may use those
checks. `ProgramEncodes` needs only this direction. A result that means
something only once a check passes (a bit, a Merkle path) then feeds the
rest of the program, and a check may name the tape cells of its own run. -/
def OldEnsures {α : Type} (c : Ctx F) (n : ℕ) (P : Prop) (b : Build α) (Q : α → ℕ → Prop) : Prop :=
  ∀ (p : Builder) (E : Exec.State F), p.values = n → E.values.length = n →
    ∃ new : List Instr, (b.run p).2.instrs = p.instrs ++ new.toArray ∧
      (c.steps new E).values.length = (b.run p).2.values ∧
      (P → c.Follows (c.steps new E) → (c.steps new E).holds → Q (b.run p).1 (b.run p).2.values)

/-- **A builder program, simulated.** `OldEnsures` of the result and of the
checks `C`. -/
def Simulates {α : Type} (c : Ctx F) (n : ℕ) (P : Prop) (b : Build α) (Q : α → ℕ → Prop) (C : Prop) : Prop :=
  OldEnsures c n P b fun a m => Q a m ∧ C

namespace Simulates

variable {c : Ctx F} {α β : Type} {n : ℕ} {P : Prop} {Q : α → ℕ → Prop} {C : Prop}

theorem pure {a : α} (h : P → Q a n) : Simulates c n P (Pure.pure a) Q True := by
  intro p E hp hE
  refine ⟨[], by show p.instrs = p.instrs ++ #[]; simp, by show E.values.length = p.values; rw [hE, hp],
    fun hP _ _ => ⟨?_, trivial⟩⟩
  exact hp ▸ h hP

theorem bind {b : Build α} {f : α → Build β} {R : β → ℕ → Prop} {C' : Prop} (hb : Simulates c n P b Q C)
    (hf : ∀ a m, n ≤ m → Simulates c m (P ∧ Q a m) (f a) R C') : Simulates c n P (b >>= f) R (C ∧ C') := by
  intro p E hp hE
  obtain ⟨new₁, hi₁, hl₁, hs₁⟩ := hb p E hp hE
  have hle : n ≤ (b.run p).2.values := by
    obtain ⟨more, hm, -⟩ := c.steps_values new₁ E
    rw [← hl₁, hm, List.length_append, hE]
    omega
  obtain ⟨new₂, hi₂, hl₂, hs₂⟩ := hf (b.run p).1 _ hle (b.run p).2 (c.steps new₁ E) rfl hl₁
  refine ⟨new₁ ++ new₂, ?_, ?_, ?_⟩
  · show ((f (b.run p).1).run (b.run p).2).2.instrs = _
    rw [hi₂, hi₁]
    simp
  · rw [Ctx.steps_append]
    exact hl₂
  · intro hP hF hh
    rw [Ctx.steps_append] at hF hh
    obtain ⟨hq, hc⟩ := hs₁ hP (Ctx.follows_of_steps hF) (Ctx.holds_of_steps hh)
    obtain ⟨hr, hc'⟩ := hs₂ ⟨hP, hq⟩ hF hh
    exact ⟨hr, hc, hc'⟩

/-- `bind` when `b` checks nothing. -/
theorem bind₀ {b : Build α} {f : α → Build β} {R : β → ℕ → Prop} {C' : Prop} (hb : Simulates c n P b Q True)
    (hf : ∀ a m, n ≤ m → Simulates c m (P ∧ Q a m) (f a) R C') : Simulates c n P (b >>= f) R C' := by
  have h := hb.bind hf
  rwa [true_and] at h

/-- Weaken the result and change the checks, both under `P`. -/
theorem post {b : Build α} {Q' : α → ℕ → Prop} {C' : Prop} (h : Simulates c n P b Q C)
    (hq : P → ∀ a m, Q a m → Q' a m) (hc : P → (C ↔ C')) : Simulates c n P b Q' C' := by
  intro p E hp hE
  obtain ⟨new, hi, hl, hs⟩ := h p E hp hE
  exact ⟨new, hi, hl, fun hP hF hh => ⟨hq hP _ _ (hs hP hF hh).1, (hc hP).mp (hs hP hF hh).2⟩⟩

/-- Strengthen the precondition. -/
theorem pre {b : Build α} {P' : Prop} (h : Simulates c n P' b Q C) (hp : P → P') : Simulates c n P b Q C := by
  intro p E hp' hE
  obtain ⟨new, hi, hl, hs⟩ := h p E hp' hE
  exact ⟨new, hi, hl, fun hP hF hh => hs (hp hP) hF hh⟩

/-- One instruction. -/
theorem emit (i : Instr) {Q : Var → ℕ → Prop}
    (h : P → ∀ E : Exec.State F, E.values.length = n → c.Follows (c.step E i) →
      Q n (n + i.outputs) ∧ ((c.step E i).holds ↔ E.holds ∧ C)) :
    Simulates c n P (Build.emit i) Q C := by
  intro p E hp hE
  refine ⟨[i], by show p.instrs.push i = p.instrs ++ #[i]; simp, ?_, fun hP hF hh => ?_⟩
  · obtain ⟨more, hm, hl⟩ := c.step_values E i
    show (c.step E i).values.length = p.values + i.outputs
    rw [hm, List.length_append, hl, hE, hp]
  · show Q p.values (p.values + i.outputs) ∧ C
    rw [hp]
    exact ⟨(h hP E hE hF).1, ((h hP E hE hF).2.mp hh).2⟩

/-- `f <$> b`. -/
theorem map {b : Build α} {f : α → β} {Q' : β → ℕ → Prop} (h : Simulates c n P b Q C)
    (hq : ∀ a m, Q a m → Q' (f a) m) : Simulates c n P (f <$> b) Q' C := by
  intro p E hp hE
  obtain ⟨new, hi, hl, hs⟩ := h p E hp hE
  exact ⟨new, hi, hl, fun hP hF hh => ⟨hq _ _ (hs hP hF hh).1, (hs hP hF hh).2⟩⟩

/-- Fix the witness of an existential precondition. Any `x₀` gives the
instructions, which do not depend on it. -/
theorem elim {ι : Type} {b : Build α} {R : ι → Prop} (x₀ : ι) (h : ∀ x, Simulates c n (R x) b Q C)
    (hp : P → ∃ x, R x) : Simulates c n P b Q C := by
  intro p E hp' hE
  obtain ⟨new, hi, hl, -⟩ := h x₀ p E hp' hE
  refine ⟨new, hi, hl, fun hP hF hh => ?_⟩
  obtain ⟨x, hx⟩ := hp hP
  obtain ⟨new', hi', -, hs⟩ := h x p E hp' hE
  obtain rfl : new = new' := List.toArray_inj ((Array.append_right_inj _).mp (hi.symm.trans hi'))
  exact hs hx hF hh

end Simulates

section Instructions

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

namespace Simulates

omit [CharP F goldilocksModulus] in
theorem const (g : Word) : Simulates c n P (Build.const g) (fun h m => IsWord c m h g) True :=
  emit _ fun _ E hl hF => by
    refine ⟨⟨by simp [Instr.outputs], ?_⟩, Iff.rfl⟩
    rw [← hl, ← Nat.add_zero E.values.length, Ctx.val_next hF (by simp)]
    rfl

omit [CharP F goldilocksModulus] in
theorem priv : Simulates c n P Build.priv (fun h m => h < m ∧ c.val h = c.tape h) True :=
  emit _ fun _ E hl hF => by
    refine ⟨⟨by simp [Instr.outputs], ?_⟩, Iff.rfl⟩
    rw [← hl, ← Nat.add_zero E.values.length, Ctx.val_next hF (by simp)]
    rfl

theorem add {a b : Var} {ga gb : Word} (ha : P → IsWord c n a ga) (hb : P → IsWord c n b gb) :
    Simulates c n P (Build.add a b) (fun h m => IsWord c m h (ga + gb)) True :=
  emit _ fun hP E hl hF => by
    have hE := Ctx.follows_of_step hF
    refine ⟨⟨by simp [Instr.outputs], ?_⟩, Iff.rfl⟩
    rw [← hl, ← Nat.add_zero E.values.length, Ctx.val_next hF (by simp)]
    simp [(ha hP).read hE hl, (hb hP).read hE hl, word_add]

theorem sub {a b : Var} {ga gb : Word} (ha : P → IsWord c n a ga) (hb : P → IsWord c n b gb) :
    Simulates c n P (Build.sub a b) (fun h m => IsWord c m h (ga - gb)) True :=
  emit _ fun hP E hl hF => by
    have hE := Ctx.follows_of_step hF
    refine ⟨⟨by simp [Instr.outputs], ?_⟩, Iff.rfl⟩
    rw [← hl, ← Nat.add_zero E.values.length, Ctx.val_next hF (by simp)]
    simp [(ha hP).read hE hl, (hb hP).read hE hl, word_sub, word_neg]
    ring

theorem scale {a : Var} {ga : Word} (by_ : Word) (ha : P → IsWord c n a ga) :
    Simulates c n P (Build.scale a by_) (fun h m => IsWord c m h (ga * by_)) True :=
  emit _ fun hP E hl hF => by
    have hE := Ctx.follows_of_step hF
    refine ⟨⟨by simp [Instr.outputs], ?_⟩, Iff.rfl⟩
    rw [← hl, ← Nat.add_zero E.values.length, Ctx.val_next hF (by simp)]
    simp [(ha hP).read hE hl, word_mul]

theorem mul {a b : Var} {ga gb : Word} (ha : P → IsWord c n a ga) (hb : P → IsWord c n b gb) :
    Simulates c n P (Build.mul a b) (fun h m => IsWord c m h (ga * gb)) True :=
  emit _ fun hP E hl hF => by
    have hE := Ctx.follows_of_step hF
    refine ⟨⟨by simp [Instr.outputs], ?_⟩, Iff.rfl⟩
    rw [← hl, ← Nat.add_zero E.values.length, Ctx.val_next hF (by simp)]
    simp [(ha hP).read hE hl, (hb hP).read hE hl, word_mul]

theorem assertZero {a : Var} {ga : Word} (ha : P → IsWord c n a ga) :
    Simulates c n P (Build.assertZero a) (fun _ _ => True) (ga = 0) := by
  refine ((emit (Q := fun _ _ => True) (C := ga = 0) (.assertZero a) fun hP E hl hF => ⟨trivial, ?_⟩).bind
    fun _ _ _ => pure fun _ => trivial).post (fun _ _ _ h => h) (fun _ => Iff.of_eq (and_true _))
  show E.holds ∧ E.value a = 0 ↔ E.holds ∧ ga = 0
  rw [(ha hP).read (Ctx.follows_of_step hF) hl]
  exact and_congr_right fun _ => ⟨fun h => word_injective (h.trans word_zero.symm), fun h => h ▸ word_zero⟩

theorem assertEqual {a b : Var} {ga gb : Word} (ha : P → IsWord c n a ga) (hb : P → IsWord c n b gb) :
    Simulates c n P (Build.assertEqual a b) (fun _ _ => True) (ga = gb) :=
  ((sub ha hb).bind fun _ _ _ => assertZero fun h => h.2).post (fun _ _ _ h => h)
    (fun _ => by rw [true_and, gl_sub_eq_zero])

omit [CharP F goldilocksModulus] in
/-- `r1cs` (`record.rs:450-460`): the rows hold on the values of the column
handles. -/
theorem r1cs {rows : List (Shrink.R1csRow Word)} {columns : List Var} (hc : P → ∀ h ∈ columns, h < n) :
    Simulates c n P (Build.r1cs rows columns) (fun _ _ => True)
      (∀ r ∈ rows, R1csRow.Holds (fun j => (columns.map c.val).getD j 0) r) := by
  refine ((emit (Q := fun _ _ => True) (C := ∀ r ∈ rows, R1csRow.Holds (fun j => (columns.map c.val).getD j 0) r)
    (.r1cs rows columns) fun hP E hl hF => ⟨trivial, ?_⟩).bind fun _ _ _ => pure fun _ => trivial).post
    (fun _ _ _ h => h) (fun _ => Iff.of_eq (and_true _))
  show E.holds ∧ (∀ r ∈ rows, R1csRow.Holds (fun j => (columns.map E.value).getD j 0) r) ↔ _
  rw [List.map_congr_left fun h mem => Ctx.follows_of_step hF h (hl ▸ hc hP h mem)]

theorem extMul {a b : ExtVar} {ga gb : Ext} (ha : P → IsExt c n a ga) (hb : P → IsExt c n b gb) :
    Simulates c n P (Build.extMul a b) (fun r m => IsExt c m r (ga * gb)) True :=
  (emit (Q := fun h m => ∀ k : Fin 3, IsWord c m (h + k) ((ga * gb).coord k)) _ fun hP E hl hF => by
    have hE := Ctx.follows_of_step hF
    refine ⟨fun k => ⟨Nat.add_lt_add_left k.isLt n, ?_⟩, Iff.rfl⟩
    rw [← hl, Ctx.val_next hF (by simp)]
    have ra : (fun i => E.value (a i)) = fun i => Word.cast (ga.coord i) := funext fun i => (ha hP i).read hE hl
    have rb : (fun i => E.value (b i)) = fun i => Word.cast (gb.coord i) := funext fun i => (hb hP i).read hE hl
    simp only [List.getD_eq_getElem?_getD, List.getElem?_ofFn]
    simp [ra, rb, ← coord_mul]).map fun _ _ h k => h k

theorem extInv {a : ExtVar} {ga gi : Ext} (ha : P → IsExt c n a ga)
    (hi : P → ∀ k : Fin 3, c.tape (n + k) = Word.cast (gi.coord k)) :
    Simulates c n P (Build.extInv a) (fun r m => IsExt c m r gi) (ga * gi = 1) :=
  (emit (Q := fun h m => ∀ k : Fin 3, IsWord c m (h + k) (gi.coord k)) _ fun hP E hl hF => by
    have hE := Ctx.follows_of_step hF
    have ra : (fun i => E.value (a i)) = fun i => Word.cast (ga.coord i) := funext fun i => (ha hP i).read hE hl
    have ri : (fun k : Fin 3 => c.tape (E.values.length + k)) = fun k => Word.cast (gi.coord k) :=
      funext fun k => hl ▸ hi hP k
    refine ⟨fun k => ⟨Nat.add_lt_add_left k.isLt n, ?_⟩, ?_⟩
    · rw [← hl, Ctx.val_next hF (by simp)]
      simp only [List.getD_eq_getElem?_getD, List.getElem?_ofFn]
      rw [dif_pos k.isLt, Option.getD_some]
      exact congrFun ri k
    · show E.holds ∧ extProduct (fun i => E.value (a i)) (fun k => c.tape (E.values.length + k)) = ![1, 0, 0] ↔ _
      rw [ra, ri, ← coord_mul]
      have one : (fun k => (Word.cast ((1 : Ext).coord k) : F)) = ![1, 0, 0] := by
        funext k
        fin_cases k <;> simp
      rw [← one]
      exact and_congr_right fun _ => ⟨fun h => Ext.eq_of_coord (F := F) (congrFun h), fun h => by rw [h]⟩).map
    fun _ _ h k => h k

omit [CharP F goldilocksModulus] in
theorem bits {a : Var} {ga : Word} (count : ℕ) (ha : P → IsWord c n a ga) :
    Simulates c n P (Build.bits a count) (fun bs m => ∀ i, bs i = n + i ∧ bs i < m ∧ c.val (bs i) = c.tape (bs i))
      (Bits (Word.cast ga) count fun i => c.tape (n + i)) :=
  (emit (Q := fun h m => h = n ∧ ∀ i : Fin count, h + i < m ∧ c.val (h + i) = c.tape (h + i)) _ fun hP E hl hF => by
    refine ⟨⟨rfl, fun i => ⟨by simp [Instr.outputs], ?_⟩⟩, ?_⟩
    · rw [← hl, Ctx.val_next hF (by simp)]
      simp
    · show E.holds ∧ Bits (E.value a) count (fun i => c.tape (E.values.length + i)) ↔ _
      rw [(ha hP).read (Ctx.follows_of_step hF) hl, hl]).map fun _ _ h i => ⟨by rw [← h.1]; rfl, h.2 i⟩

omit [CharP F goldilocksModulus] in
theorem hint (count : ℕ) :
    Simulates c n P (Build.hint count) (fun hs m => ∀ i, hs i = n + i ∧ hs i < m ∧ c.val (hs i) = c.tape (hs i)) True :=
  (emit (Q := fun h m => h = n ∧ ∀ i : Fin count, h + i < m ∧ c.val (h + i) = c.tape (h + i)) _ fun _ E hl hF =>
    ⟨⟨rfl, fun i => ⟨by simp [Instr.outputs], by rw [← hl, Ctx.val_next hF (by simp)]; simp⟩⟩, Iff.rfl⟩).map
    fun _ _ h i => ⟨by rw [← h.1]; rfl, h.2 i⟩

omit [CharP F goldilocksModulus] in
theorem permute {state : Fin 16 → Var} {gs : Array Word} (hp : Permutes c.perm)
    (hs : P → ∀ k, IsWord c n (state k) gs[k]!) :
    Simulates c n P (Build.permute state) (fun r m => ∀ k, IsWord c m (r k) (Replay.permute gs)[k]!) True :=
  (emit (Q := fun h m => ∀ k : Fin 16, IsWord c m (h + k) (Replay.permute gs)[k]!) _ fun hP E hl hF => by
    have hE := Ctx.follows_of_step hF
    have rs : (fun k => E.value (state k)) = fun k : Fin 16 => Word.cast gs[k]! := funext fun k => (hs hP k).read hE hl
    refine ⟨fun k => ⟨Nat.add_lt_add_left k.isLt n, ?_⟩, Iff.rfl⟩
    rw [← hl, Ctx.val_next hF (by simp)]
    simp only [List.getD_eq_getElem?_getD, List.getElem?_ofFn]
    rw [dif_pos k.isLt, Option.getD_some, ← hp gs k, rs, List.getD_eq_getElem?_getD]).map fun _ _ h k => h k

end Simulates

end Instructions

/-! ## Loops and transcripts -/

namespace Simulates

variable {c : Ctx F} {α β γ δ : Type} {n : ℕ} {P : Prop} {Q : α → ℕ → Prop} {C : Prop}

/-- A loop from step `j` on. -/
theorem loop_from {xs : List δ} {body : δ → γ → Build (ForInStep γ)} (I : ℕ → γ → ℕ → Prop) (D : ℕ → Prop)
    (step : ∀ k (hk : k < xs.length) acc m, n ≤ m → Simulates c m (P ∧ I k acc m) (body xs[k] acc)
      (fun r m' => r = .yield r.value ∧ I (k + 1) r.value m') (D k)) :
    ∀ t j, xs.length - j = t → j ≤ xs.length → ∀ acc m, n ≤ m →
      Simulates c m (P ∧ I j acc m) (forIn (xs.drop j) acc body) (fun r m' => I xs.length r m')
        (∀ k, j ≤ k → k < xs.length → D k)
  | 0, j, ht, hj, acc, m, _ => by
      have e : j = xs.length := by omega
      subst e
      rw [List.drop_length, List.forIn_nil]
      exact (Simulates.pure fun h => h.2).post (fun _ _ _ h => h) fun _ => ⟨fun _ k lo hi => absurd hi (by omega),
        fun _ => trivial⟩
  | t + 1, j, ht, hj, acc, m, hm => by
      have hj' : j < xs.length := by omega
      rw [List.drop_eq_getElem_cons hj', List.forIn_cons]
      refine (Simulates.bind (C' := ∀ k, j + 1 ≤ k → k < xs.length → D k) (step j hj' acc m hm) fun r m' le => ?_).post (fun _ _ _ h => h) fun _ =>
        ⟨fun h k lo hi => (Nat.eq_or_lt_of_le lo).elim (fun e => e ▸ h.1) fun lt => h.2 k lt hi,
          fun h => ⟨h j le_rfl hj', fun k lo hi => h k (by omega) hi⟩⟩
      cases r with
      | done b =>
          exact (Simulates.pure (Q := fun _ _ => True) fun _ => trivial).post (fun h => absurd h.2.1 (by simp))
            fun h => absurd h.2.1 (by simp)
      | yield b =>
          exact (loop_from I D step t (j + 1) (by omega) hj' b m' (hm.trans le)).pre fun h => ⟨h.1.1, h.2.2⟩

/-- **A loop over a list.** `I k acc m`: after `k` steps the accumulator is
`acc` with `m` values. Step `k` checks `D k`. -/
theorem loop {xs : List δ} {init : γ} {body : δ → γ → Build (ForInStep γ)} (I : ℕ → γ → ℕ → Prop)
    (D : ℕ → Prop)
    (step : ∀ k (hk : k < xs.length) acc m, n ≤ m → Simulates c m (P ∧ I k acc m) (body xs[k] acc)
      (fun r m' => r = .yield r.value ∧ I (k + 1) r.value m') (D k)) :
    Simulates c n (P ∧ I 0 init n) (forIn xs init body) (fun r m => I xs.length r m) (∀ k < xs.length, D k) :=
  (loop_from I D step _ 0 rfl (Nat.zero_le _) init n le_rfl).post (fun _ _ _ h => h)
    fun _ => ⟨fun h k hk => h k (Nat.zero_le _) hk, fun h k _ hk => h k hk⟩

/-- `t >>= g` in a transcript. -/
theorem tbind {t : Transcript α} {g : α → Transcript β} {d : Duplex} {R : β × Duplex → ℕ → Prop} {C' : Prop}
    {Q : α × Duplex → ℕ → Prop} (ht : Simulates c n P (t.run d) Q C)
    (hg : ∀ a e m, n ≤ m → Simulates c m (P ∧ Q (a, e) m) ((g a).run e) R C') : Simulates c n P ((t >>= g).run d) R (C ∧ C') := by
  rw [StateT.run_bind]
  exact ht.bind fun r m le => hg r.1 r.2 m le

/-- `tbind` when `t` checks nothing. -/
theorem tbind₀ {t : Transcript α} {g : α → Transcript β} {d : Duplex} {R : β × Duplex → ℕ → Prop} {C' : Prop}
    {Q : α × Duplex → ℕ → Prop} (ht : Simulates c n P (t.run d) Q True)
    (hg : ∀ a e m, n ≤ m → Simulates c m (P ∧ Q (a, e) m) ((g a).run e) R C') : Simulates c n P ((t >>= g).run d) R C' := by
  have h := ht.tbind hg
  rwa [true_and] at h

/-- A builder call in a transcript. -/
theorem lift {b : Build α} {d : Duplex} (h : Simulates c n P b Q C) :
    Simulates c n P ((monadLift b : Transcript α).run d) (fun r m => Q r.1 m ∧ r.2 = d) C := by
  rw [StateT.run_monadLift]
  exact (h.bind fun a m _ => Simulates.pure fun h => ⟨h.2, rfl⟩).post (fun _ _ _ h => h) fun _ => Iff.of_eq (and_true _)

/-- A loop over a list in a transcript; `I k acc e m` also names the duplex. -/
theorem loopT_from {xs : List δ} {body : δ → γ → Transcript (ForInStep γ)} (I : ℕ → γ → Duplex → ℕ → Prop)
    (D : ℕ → Prop)
    (step : ∀ k (hk : k < xs.length) acc e m, n ≤ m → Simulates c m (P ∧ I k acc e m) ((body xs[k] acc).run e)
      (fun r m' => r.1 = .yield r.1.value ∧ I (k + 1) r.1.value r.2 m') (D k)) :
    ∀ t j, xs.length - j = t → j ≤ xs.length → ∀ acc e m, n ≤ m →
      Simulates c m (P ∧ I j acc e m) ((forIn (xs.drop j) acc body).run e) (fun r m' => I xs.length r.1 r.2 m')
        (∀ k, j ≤ k → k < xs.length → D k)
  | 0, j, ht, hj, acc, e, m, _ => by
      have h : j = xs.length := by omega
      subst h
      rw [List.drop_length, List.forIn_nil, StateT.run_pure]
      exact (Simulates.pure fun h => h.2).post (fun _ _ _ h => h) fun _ => ⟨fun _ k lo hi => absurd hi (by omega),
        fun _ => trivial⟩
  | t + 1, j, ht, hj, acc, e, m, hm => by
      have hj' : j < xs.length := by omega
      rw [List.drop_eq_getElem_cons hj', List.forIn_cons, StateT.run_bind]
      refine (Simulates.bind (C' := ∀ k, j + 1 ≤ k → k < xs.length → D k) (step j hj' acc e m hm) fun r m' le => ?_).post (fun _ _ _ h => h) fun _ =>
        ⟨fun h k lo hi => (Nat.eq_or_lt_of_le lo).elim (fun e => e ▸ h.1) fun lt => h.2 k lt hi,
          fun h => ⟨h j le_rfl hj', fun k lo hi => h k (by omega) hi⟩⟩
      obtain ⟨r, e'⟩ := r
      cases r with
      | done b =>
          show Simulates c m' _ (Pure.pure (b, e') : Build (γ × Duplex)) _ _
          exact (Simulates.pure (Q := fun _ _ => True) fun _ => trivial).post (fun h => absurd h.2.1 (by simp))
            fun h => absurd h.2.1 (by simp)
      | yield b =>
          exact (loopT_from I D step t (j + 1) (by omega) hj' b e' m' (hm.trans le)).pre fun h => ⟨h.1.1, h.2.2⟩

/-- **A loop over a list in a transcript.** -/
theorem loopT {xs : List δ} {init : γ} {body : δ → γ → Transcript (ForInStep γ)} {d : Duplex}
    (I : ℕ → γ → Duplex → ℕ → Prop) (D : ℕ → Prop)
    (step : ∀ k (hk : k < xs.length) acc e m, n ≤ m → Simulates c m (P ∧ I k acc e m) ((body xs[k] acc).run e)
      (fun r m' => r.1 = .yield r.1.value ∧ I (k + 1) r.1.value r.2 m') (D k)) :
    Simulates c n (P ∧ I 0 init d n) ((forIn xs init body).run d) (fun r m => I xs.length r.1 r.2 m)
      (∀ k < xs.length, D k) :=
  (loopT_from I D step _ 0 rfl (Nat.zero_le _) init d n le_rfl).post (fun _ _ _ h => h)
    fun _ => ⟨fun h k hk => h k (Nat.zero_le _) hk, fun h k _ hk => h k hk⟩

/-- `mapM` over a list: result `i` meets `R` at entry `i`, which survives
later values. -/
theorem listMapM {f : δ → Build α} (R : δ → α → ℕ → Prop) (mono : ∀ x a m m', R x a m → m ≤ m' → R x a m') :
    ∀ (xs : List δ) {n : ℕ} {P : Prop}, (∀ x ∈ xs, ∀ m, n ≤ m → Simulates c m P (f x) (R x) True) →
      Simulates c n P (xs.mapM f) (fun as m => List.Forall₂ (fun x a => R x a m) xs as) True
  | [], n, P, _ => by
      rw [List.mapM_nil]
      exact Simulates.pure fun _ => List.Forall₂.nil
  | x :: xs, n, P, h => by
      rw [List.mapM_cons]
      refine (h x List.mem_cons_self n le_rfl).bind₀ fun a m le => ?_
      refine (listMapM R mono xs (P := P ∧ R x a m) fun y hy m' le' =>
        (h y (List.mem_cons_of_mem _ hy) m' (le.trans le')).pre fun h => h.1).bind₀ fun as m' le' => ?_
      exact Simulates.pure fun hh => List.Forall₂.cons (mono _ _ _ _ hh.1.2 le') hh.2

/-- `mapM` over an array. -/
theorem mapM {n : ℕ} {P : Prop} {xs : Array δ} {f : δ → Build α} (R : δ → α → ℕ → Prop)
    (mono : ∀ x a m m', R x a m → m ≤ m' → R x a m') (h : ∀ x ∈ xs, ∀ m, n ≤ m → Simulates c m P (f x) (R x) True) :
    Simulates c n P (xs.mapM f) (fun as m => List.Forall₂ (fun x a => R x a m) xs.toList as.toList) True := by
  rw [Array.mapM_eq_mapM_toList]
  exact (listMapM R mono xs.toList fun x hx => h x (Array.mem_toList_iff.mp hx)).map fun as m h => by simpa using h

end Simulates

/-! ## Soundness only -/

variable {c : Ctx F} {α β γ δ : Type} {n : ℕ} {P : Prop} {Q : α → ℕ → Prop}

/-- A simulated program ensures its result and its checks. -/
theorem Simulates.ensures {b : Build α} {C : Prop} (h : Simulates c n P b Q C) : OldEnsures c n P b fun a m => Q a m ∧ C :=
  h

/-- A simulated program that checks nothing ensures its result. -/
theorem Simulates.ensures₀ {b : Build α} (h : Simulates c n P b Q True) : OldEnsures c n P b Q := by
  intro p E hp hE
  obtain ⟨new, hi, hl, hs⟩ := h p E hp hE
  exact ⟨new, hi, hl, fun hP hF hh => (hs hP hF hh).1⟩

namespace OldEnsures

theorem pure {a : α} (h : P → Q a n) : OldEnsures c n P (Pure.pure a) Q := by
  intro p E hp hE
  refine ⟨[], by show p.instrs = p.instrs ++ #[]; simp, by show E.values.length = p.values; rw [hE, hp],
    fun hP _ _ => ?_⟩
  exact hp ▸ h hP

theorem bind {b : Build α} {f : α → Build β} {R : β → ℕ → Prop} (hb : OldEnsures c n P b Q)
    (hf : ∀ a m, n ≤ m → OldEnsures c m (P ∧ Q a m) (f a) R) : OldEnsures c n P (b >>= f) R := by
  intro p E hp hE
  obtain ⟨new₁, hi₁, hl₁, hs₁⟩ := hb p E hp hE
  have hle : n ≤ (b.run p).2.values := by
    obtain ⟨more, hm, -⟩ := c.steps_values new₁ E
    rw [← hl₁, hm, List.length_append, hE]
    omega
  obtain ⟨new₂, hi₂, hl₂, hs₂⟩ := hf (b.run p).1 _ hle (b.run p).2 (c.steps new₁ E) rfl hl₁
  refine ⟨new₁ ++ new₂, ?_, ?_, ?_⟩
  · show ((f (b.run p).1).run (b.run p).2).2.instrs = _
    rw [hi₂, hi₁]
    simp
  · rw [Ctx.steps_append]
    exact hl₂
  · intro hP hF hh
    rw [Ctx.steps_append] at hF hh
    exact hs₂ ⟨hP, hs₁ hP (Ctx.follows_of_steps hF) (Ctx.holds_of_steps hh)⟩ hF hh

theorem post {b : Build α} {Q' : α → ℕ → Prop} (h : OldEnsures c n P b Q) (hq : P → ∀ a m, Q a m → Q' a m) :
    OldEnsures c n P b Q' := by
  intro p E hp hE
  obtain ⟨new, hi, hl, hs⟩ := h p E hp hE
  exact ⟨new, hi, hl, fun hP hF hh => hq hP _ _ (hs hP hF hh)⟩

theorem pre {b : Build α} {P' : Prop} (h : OldEnsures c n P' b Q) (hp : P → P') : OldEnsures c n P b Q := by
  intro p E hp' hE
  obtain ⟨new, hi, hl, hs⟩ := h p E hp' hE
  exact ⟨new, hi, hl, fun hP hF hh => hs (hp hP) hF hh⟩

theorem map {b : Build α} {f : α → β} {Q' : β → ℕ → Prop} (h : OldEnsures c n P b Q)
    (hq : ∀ a m, Q a m → Q' (f a) m) : OldEnsures c n P (f <$> b) Q' := by
  intro p E hp hE
  obtain ⟨new, hi, hl, hs⟩ := h p E hp hE
  exact ⟨new, hi, hl, fun hP hF hh => hq _ _ (hs hP hF hh)⟩

/-- Use a fact `A` that `P` gives. The instructions do not depend on the
precondition, so `h₀`, at any precondition, fixes them. -/
theorem under {P₀ A : Prop} {b : Build α} {Q₀ : α → ℕ → Prop} (h₀ : OldEnsures c n P₀ b Q₀) (hA : P → A)
    (h : A → OldEnsures c n P b Q) : OldEnsures c n P b Q := by
  intro p E hp hE
  obtain ⟨new, hi, hl, -⟩ := h₀ p E hp hE
  refine ⟨new, hi, hl, fun hP hF hh => ?_⟩
  obtain ⟨new', hi', -, hs⟩ := h (hA hP) p E hp hE
  obtain rfl : new = new' := List.toArray_inj ((Array.append_right_inj _).mp (hi.symm.trans hi'))
  exact hs hP hF hh

/-- Fix the witness of an existential precondition. Any `x₀` gives the
instructions. -/
theorem elim {ι : Type} {b : Build α} {R : ι → Prop} (x₀ : ι) (h : ∀ x, OldEnsures c n (R x) b Q)
    (hp : P → ∃ x, R x) : OldEnsures c n P b Q :=
  under (h x₀) hp fun ⟨x, hx⟩ => (h x).pre fun _ => hx

/-- Name the witness of an existential precondition. -/
theorem exists_and {ι : Type} {b : Build α} {X : ι → Prop} (i₀ : ι) (h : ∀ i, OldEnsures c n (P ∧ X i) b Q) :
    OldEnsures c n (P ∧ ∃ i, X i) b Q :=
  elim i₀ h fun ⟨hP, i, hx⟩ => ⟨i, hP, hx⟩

/-- Name the witness, using a fact `A` of each witness; `i₀` has it and fixes
the instructions. -/
theorem witness {ι : Type} {b : Build α} {X A : ι → Prop} (i₀ : ι) (h₀ : A i₀) (hA : ∀ i, X i → A i)
    (h : ∀ i, A i → OldEnsures c n (P ∧ X i) b Q) : OldEnsures c n (P ∧ ∃ i, X i) b Q :=
  under (A := ∃ i, X i ∧ A i) (h i₀ h₀) (fun H => let ⟨i, hx⟩ := H.2; ⟨i, hx, hA i hx⟩) fun ⟨i, hx, ha⟩ =>
    (h i ha).pre fun H => ⟨H.1, hx⟩

/-- `bind` after a step that checks nothing. -/
theorem bindSim {b : Build α} {f : α → Build β} {R : β → ℕ → Prop} (hb : Simulates c n P b Q True)
    (hf : ∀ a m, n ≤ m → OldEnsures c m (P ∧ Q a m) (f a) R) : OldEnsures c n P (b >>= f) R :=
  hb.ensures₀.bind hf

/-- `bind` after an assertion. -/
theorem bindCheck {b : Build α} {f : α → Build β} {C : Prop} {R : β → ℕ → Prop}
    (hb : Simulates c n P b (fun _ _ => True) C) (hf : ∀ a m, n ≤ m → OldEnsures c m (P ∧ C) (f a) R) :
    OldEnsures c n P (b >>= f) R :=
  (hb.ensures.post fun _ _ _ h => h.2).bind hf

/-- `t >>= g` in a transcript. -/
theorem tbind {t : Transcript α} {g : α → Transcript β} {d : Duplex} {R : β × Duplex → ℕ → Prop}
    {Q : α × Duplex → ℕ → Prop} (ht : OldEnsures c n P (t.run d) Q)
    (hg : ∀ a e m, n ≤ m → OldEnsures c m (P ∧ Q (a, e) m) ((g a).run e) R) : OldEnsures c n P ((t >>= g).run d) R := by
  rw [StateT.run_bind]
  exact ht.bind fun r m le => hg r.1 r.2 m le

/-- `tbind` after a step that checks nothing. -/
theorem tbindSim {t : Transcript α} {g : α → Transcript β} {d : Duplex} {R : β × Duplex → ℕ → Prop}
    {Q : α × Duplex → ℕ → Prop} (ht : Simulates c n P (t.run d) Q True)
    (hg : ∀ a e m, n ≤ m → OldEnsures c m (P ∧ Q (a, e) m) ((g a).run e) R) : OldEnsures c n P ((t >>= g).run d) R :=
  ht.ensures₀.tbind hg

/-- A builder call in a transcript. -/
theorem lift {b : Build α} {d : Duplex} (h : OldEnsures c n P b Q) :
    OldEnsures c n P ((monadLift b : Transcript α).run d) (fun r m => Q r.1 m ∧ r.2 = d) := by
  rw [StateT.run_monadLift]
  exact h.bind fun a m _ => OldEnsures.pure fun h => ⟨h.2, rfl⟩

/-- A loop from step `j` on. -/
theorem loop_from {xs : List δ} {body : δ → γ → Build (ForInStep γ)} (I : ℕ → γ → ℕ → Prop)
    (step : ∀ k (hk : k < xs.length) acc m, n ≤ m → OldEnsures c m (P ∧ I k acc m) (body xs[k] acc)
      (fun r m' => r = .yield r.value ∧ I (k + 1) r.value m')) :
    ∀ t j, xs.length - j = t → j ≤ xs.length → ∀ acc m, n ≤ m →
      OldEnsures c m (P ∧ I j acc m) (forIn (xs.drop j) acc body) (fun r m' => I xs.length r m')
  | 0, j, ht, hj, acc, m, _ => by
      have e : j = xs.length := by omega
      subst e
      rw [List.drop_length, List.forIn_nil]
      exact OldEnsures.pure fun h => h.2
  | t + 1, j, ht, hj, acc, m, hm => by
      have hj' : j < xs.length := by omega
      rw [List.drop_eq_getElem_cons hj', List.forIn_cons]
      refine (step j hj' acc m hm).bind fun r m' le => ?_
      cases r with
      | done b => exact (OldEnsures.pure (Q := fun _ _ => True) fun _ => trivial).post fun h => absurd h.2.1 (by simp)
      | yield b => exact (loop_from I step t (j + 1) (by omega) hj' b m' (hm.trans le)).pre fun h => ⟨h.1.1, h.2.2⟩

/-- **A loop over a list.** `I k acc m`: after `k` steps the accumulator is
`acc` with `m` values. -/
theorem loop {xs : List δ} {init : γ} {body : δ → γ → Build (ForInStep γ)} (I : ℕ → γ → ℕ → Prop)
    (step : ∀ k (hk : k < xs.length) acc m, n ≤ m → OldEnsures c m (P ∧ I k acc m) (body xs[k] acc)
      (fun r m' => r = .yield r.value ∧ I (k + 1) r.value m')) :
    OldEnsures c n (P ∧ I 0 init n) (forIn xs init body) (fun r m => I xs.length r m) :=
  loop_from I step _ 0 rfl (Nat.zero_le _) init n le_rfl

/-- A loop over a list in a transcript; `I k acc e m` also names the duplex. -/
theorem loopT_from {xs : List δ} {body : δ → γ → Transcript (ForInStep γ)} (I : ℕ → γ → Duplex → ℕ → Prop)
    (step : ∀ k (hk : k < xs.length) acc e m, n ≤ m → OldEnsures c m (P ∧ I k acc e m) ((body xs[k] acc).run e)
      (fun r m' => r.1 = .yield r.1.value ∧ I (k + 1) r.1.value r.2 m')) :
    ∀ t j, xs.length - j = t → j ≤ xs.length → ∀ acc e m, n ≤ m →
      OldEnsures c m (P ∧ I j acc e m) ((forIn (xs.drop j) acc body).run e) (fun r m' => I xs.length r.1 r.2 m')
  | 0, j, ht, hj, acc, e, m, _ => by
      have h : j = xs.length := by omega
      subst h
      rw [List.drop_length, List.forIn_nil, StateT.run_pure]
      exact OldEnsures.pure fun h => h.2
  | t + 1, j, ht, hj, acc, e, m, hm => by
      have hj' : j < xs.length := by omega
      rw [List.drop_eq_getElem_cons hj', List.forIn_cons, StateT.run_bind]
      refine (step j hj' acc e m hm).bind fun r m' le => ?_
      obtain ⟨r, e'⟩ := r
      cases r with
      | done b =>
          show OldEnsures c m' _ (Pure.pure (b, e') : Build (γ × Duplex)) _
          exact (OldEnsures.pure (Q := fun _ _ => True) fun _ => trivial).post fun h => absurd h.2.1 (by simp)
      | yield b =>
          exact (loopT_from I step t (j + 1) (by omega) hj' b e' m' (hm.trans le)).pre fun h => ⟨h.1.1, h.2.2⟩

/-- **A loop over a list in a transcript.** -/
theorem loopT {xs : List δ} {init : γ} {body : δ → γ → Transcript (ForInStep γ)} {d : Duplex}
    (I : ℕ → γ → Duplex → ℕ → Prop)
    (step : ∀ k (hk : k < xs.length) acc e m, n ≤ m → OldEnsures c m (P ∧ I k acc e m) ((body xs[k] acc).run e)
      (fun r m' => r.1 = .yield r.1.value ∧ I (k + 1) r.1.value r.2 m')) :
    OldEnsures c n (P ∧ I 0 init d n) ((forIn xs init body).run d) (fun r m => I xs.length r.1 r.2 m) :=
  loopT_from I step _ 0 rfl (Nat.zero_le _) init d n le_rfl

/-- **An accepting run meets what the program ensures.** -/
theorem exec {b : Build α} (h : OldEnsures c 0 P b Q) (hP : P)
    (hval : c.val = (Exec.run c.perm c.stmt c.tape b.program).value)
    (accepts : Exec c.perm b.program c.stmt c.tape) : Q (b.run {}).1 (b.run {}).2.values := by
  obtain ⟨new, hi, -, hs⟩ := h {} Exec.start rfl rfl
  have run : Exec.run c.perm c.stmt c.tape b.program = c.steps new Exec.start := by
    rw [Exec.run, Build.program, hi]
    simp only [Array.empty_append, List.toList_toArray]
    rfl
  exact hs hP (fun v _ => by rw [hval, run]) (by rw [Exec, run] at accepts; exact accepts)

end OldEnsures

/-- `OldEnsures` from one builder state `p`. The program's result is then a
known term, `(b.run p).1`, so the next program can name the handles that the
run writes. -/
def OldEnsuresAt (c : Ctx F) (p : Builder) (P : Prop) (b : Build α) (Q : α → ℕ → Prop) : Prop :=
  ∀ (E : Exec.State F), E.values.length = p.values →
    ∃ new : List Instr, (b.run p).2.instrs = p.instrs ++ new.toArray ∧
      (c.steps new E).values.length = (b.run p).2.values ∧
      (P → c.Follows (c.steps new E) → (c.steps new E).holds → Q (b.run p).1 (b.run p).2.values)

namespace OldEnsuresAt

variable {α β : Type} {P : Prop}

theorem of_ensures {p : Builder} {b : Build α} {Q : α → ℕ → Prop} (h : OldEnsures c p.values P b Q) :
    OldEnsuresAt c p P b Q := fun E hE => h p E rfl hE

/-- `bind` at the state that `b` leaves, with the result that `b` returns. -/
theorem bind {p : Builder} {b : Build α} {f : α → Build β} {Q : α → ℕ → Prop} {R : β → ℕ → Prop}
    (hb : OldEnsuresAt c p P b Q)
    (hf : p.values ≤ (b.run p).2.values →
      OldEnsuresAt c (b.run p).2 (P ∧ Q (b.run p).1 (b.run p).2.values) (f (b.run p).1) R) :
    OldEnsuresAt c p P (b >>= f) R := by
  intro E hE
  obtain ⟨new₁, hi₁, hl₁, hs₁⟩ := hb E hE
  have hle : p.values ≤ (b.run p).2.values := by
    obtain ⟨more, hm, -⟩ := c.steps_values new₁ E
    rw [← hl₁, hm, List.length_append, hE]
    omega
  obtain ⟨new₂, hi₂, hl₂, hs₂⟩ := hf hle (c.steps new₁ E) hl₁
  refine ⟨new₁ ++ new₂, ?_, ?_, ?_⟩
  · show ((f (b.run p).1).run (b.run p).2).2.instrs = _
    rw [hi₂, hi₁]
    simp
  · rw [Ctx.steps_append]
    exact hl₂
  · intro hP hF hh
    rw [Ctx.steps_append] at hF hh
    exact hs₂ ⟨hP, hs₁ hP (Ctx.follows_of_steps hF) (Ctx.holds_of_steps hh)⟩ hF hh

theorem pre {P' : Prop} {p : Builder} {b : Build α} {Q : α → ℕ → Prop} (h : OldEnsuresAt c p P b Q) (hp : P' → P) :
    OldEnsuresAt c p P' b Q := by
  intro E hE
  obtain ⟨new, hi, hl, hs⟩ := h E hE
  exact ⟨new, hi, hl, fun h' => hs (hp h')⟩

theorem post {p : Builder} {b : Build α} {Q Q' : α → ℕ → Prop} (h : OldEnsuresAt c p P b Q)
    (hq : P → Q (b.run p).1 (b.run p).2.values → Q' (b.run p).1 (b.run p).2.values) : OldEnsuresAt c p P b Q' := by
  intro E hE
  obtain ⟨new, hi, hl, hs⟩ := h E hE
  exact ⟨new, hi, hl, fun hP hF hh => hq hP (hs hP hF hh)⟩

/-- **An accepting run meets what the program ensures from the empty builder.** -/
theorem exec {b : Build α} {Q : α → ℕ → Prop} (h : OldEnsuresAt c {} P b Q) (hP : P)
    (hval : c.val = (Exec.run c.perm c.stmt c.tape b.program).value)
    (accepts : Exec c.perm b.program c.stmt c.tape) : Q (b.run {}).1 (b.run {}).2.values := by
  obtain ⟨new, hi, -, hs⟩ := h Exec.start rfl
  have run : Exec.run c.perm c.stmt c.tape b.program = c.steps new Exec.start := by
    rw [Exec.run, Build.program, hi]
    simp only [Array.empty_append, List.toList_toArray]
    rfl
  exact hs hP (fun v _ => by rw [hval, run]) (by rw [Exec, run] at accepts; exact accepts)

end OldEnsuresAt

section Bridges

variable {c : Ctx F} {α : Type} {n : ℕ} {P : Prop} {Q : α → ℕ → Prop}

/-- A new lemma at an old precondition. -/
theorem OldEnsures.of_ensures {b : Build α} (h : P → Ensures c n b Q) (hb : Appends b := by appends) :
    OldEnsures c n P b Q := by
  intro p E hp hE
  obtain ⟨new, hi, hv⟩ := hb.out p
  refine ⟨new, hi, ?_, fun hP hF hh => h hP p hp E new (hE.trans hp.symm) hi hF hh⟩
  obtain ⟨more, hm, hl⟩ := c.steps_values new E
  rw [hm, List.length_append, hl, hE, hv, hp]

/-- A new lemma at an old precondition, with the checks `C` in its result. -/
theorem Simulates.of_ensures {b : Build α} {C : Prop} (h : P → Ensures c n b fun a m => Q a m ∧ C)
    (hb : Appends b := by appends) : Simulates c n P b Q C :=
  OldEnsures.of_ensures h hb

/-- A new lemma that checks nothing, at an old precondition. -/
theorem Simulates.of_ensures₀ {b : Build α} (h : P → Ensures c n b Q) (hb : Appends b := by appends) :
    Simulates c n P b Q True :=
  OldEnsures.of_ensures (fun hP => (h hP).post fun _ _ q => ⟨q, trivial⟩) hb

end Bridges

end CompressionLean.Sim
