import CompressionLean.Sim.Calculus

/-!
Owns what each instruction of `Ir.lean` ensures (`X_ensures` for `Build.X`):
its outputs are the replay's values of its inputs, and an assertion's check
is the replay's equality. `bits` also gives its tape cells and its check,
which names those cells. `extInv_ensures` is in `Algebra.lean`, where the
coordinates of `Ext` are.

The lemmas that compute with words need a field of characteristic `p`; the
others do not.
-/

namespace CompressionLean.Sim

open CompressionLean.Circuit
open CompressionLean.Lowering
open CompressionLean.Replay (Word Ext)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F] {c : Ctx F} {n : ℕ}

theorem const_ensures (g : Word) : Ensures c n (Build.const g) fun h m => IsWord c m h g :=
  Ensures.emit _ fun E hl hF _ => by
    refine ⟨by simp [Instr.outputs], ?_⟩
    rw [← hl, ← Nat.add_zero E.values.length, Ctx.val_next hF (by simp)]
    rfl

/-- `private`: a proof word, a raw tape cell, the next handle. -/
theorem priv_ensures : Ensures c n Build.priv fun h m => h = n ∧ m = n + 1 ∧ c.val h = c.tape h :=
  Ensures.emit _ fun E hl hF _ => by
    refine ⟨rfl, rfl, ?_⟩
    rw [← hl, ← Nat.add_zero E.values.length, Ctx.val_next hF (by simp)]
    rfl

/-- `r1cs` (`record.rs::Recorder::r1cs`): the rows hold on the values of the column
handles. -/
theorem r1cs_ensures {rows : List (Shrink.R1csRow Word)} {columns : List Var} (hc : ∀ h ∈ columns, h < n) :
    Ensures c n (Build.r1cs rows columns) fun _ _ =>
      ∀ r ∈ rows, R1csRow.Holds (fun j => (columns.map c.val).getD j 0) r := by
  refine (Ensures.emit (Q := fun _ _ => ∀ r ∈ rows, R1csRow.Holds (fun j => (columns.map c.val).getD j 0) r)
    (.r1cs rows columns) fun E hl hF hh => ?_).bind fun _ _ _ h => Ensures.pure h
  have holds : E.holds ∧ ∀ r ∈ rows, R1csRow.Holds (fun j => (columns.map E.value).getD j 0) r := hh
  rw [List.map_congr_left fun h mem => Ctx.follows_of_step hF h (hl ▸ hc h mem)] at holds
  exact holds.2

/-- `bits`: `count` tape cells after the `n` values, which the check makes
the bits of `a`. -/
theorem bits_ensures {a : Var} {ga : Word} (count : ℕ) (ha : IsWord c n a ga) :
    Ensures c n (Build.bits a count) fun bs m => (∀ i, bs i = n + i ∧ bs i < m ∧ c.val (bs i) = c.tape (bs i)) ∧
      Bits (Word.cast ga) count fun i => c.tape (n + i) :=
  (Ensures.emit (Q := fun h m => (h = n ∧ ∀ i : Fin count, h + i < m ∧ c.val (h + i) = c.tape (h + i)) ∧
      Bits (Word.cast ga) count fun i => c.tape (n + i)) _ fun E hl hF hh => by
    refine ⟨⟨rfl, fun i => ⟨by simp [Instr.outputs], ?_⟩⟩, ?_⟩
    · rw [← hl, Ctx.val_next hF (by simp)]
      simp
    · have holds : E.holds ∧ Bits (E.value a) count fun i => c.tape (E.values.length + i) := hh
      rw [ha.read (Ctx.follows_of_step hF) hl, hl] at holds
      exact holds.2).map fun _ _ ⟨⟨first, cells⟩, check⟩ => ⟨fun i => ⟨by rw [← first]; rfl, cells i⟩, check⟩

/-- `hint`: `count` free proof words, raw tape cells. -/
theorem hint_ensures (count : ℕ) :
    Ensures c n (Build.hint count) fun hs m => ∀ i, hs i = n + i ∧ hs i < m ∧ c.val (hs i) = c.tape (hs i) :=
  (Ensures.emit (Q := fun h m => h = n ∧ ∀ i : Fin count, h + i < m ∧ c.val (h + i) = c.tape (h + i)) _
    fun E hl hF _ => ⟨rfl, fun i => ⟨by simp [Instr.outputs], by rw [← hl, Ctx.val_next hF (by simp)]; simp⟩⟩).map
    fun _ _ ⟨first, cells⟩ i => ⟨by rw [← first]; rfl, cells i⟩

/-- `permute`: one Poseidon2 block, the replay's permutation. -/
theorem permute_ensures {state : Fin 16 → Var} {gs : Array Word} (hp : Permutes c.perm)
    (hs : ∀ k, IsWord c n (state k) gs[k]!) :
    Ensures c n (Build.permute state) fun r m => ∀ k, IsWord c m (r k) (Replay.permute gs)[k]! :=
  (Ensures.emit (Q := fun h m => ∀ k : Fin 16, IsWord c m (h + k) (Replay.permute gs)[k]!) _ fun E hl hF _ => by
    have hE := Ctx.follows_of_step hF
    have rs : (fun k => E.value (state k)) = fun k : Fin 16 => Word.cast gs[k]! := funext fun k => (hs k).read hE hl
    refine fun k => ⟨Nat.add_lt_add_left k.isLt n, ?_⟩
    rw [← hl, Ctx.val_next hF (by simp)]
    simp only [List.getD_eq_getElem?_getD, List.getElem?_ofFn]
    rw [dif_pos k.isLt, Option.getD_some, ← hp gs k, rs, List.getD_eq_getElem?_getD]).map fun _ _ h k => h k

section Words

variable [CharP F goldilocksModulus]

theorem add_ensures {a b : Var} {ga gb : Word} (ha : IsWord c n a ga) (hb : IsWord c n b gb) :
    Ensures c n (Build.add a b) fun h m => IsWord c m h (ga + gb) :=
  Ensures.emit _ fun E hl hF _ => by
    have hE := Ctx.follows_of_step hF
    refine ⟨by simp [Instr.outputs], ?_⟩
    rw [← hl, ← Nat.add_zero E.values.length, Ctx.val_next hF (by simp)]
    simp [ha.read hE hl, hb.read hE hl, word_add]

theorem sub_ensures {a b : Var} {ga gb : Word} (ha : IsWord c n a ga) (hb : IsWord c n b gb) :
    Ensures c n (Build.sub a b) fun h m => IsWord c m h (ga - gb) :=
  Ensures.emit _ fun E hl hF _ => by
    have hE := Ctx.follows_of_step hF
    refine ⟨by simp [Instr.outputs], ?_⟩
    rw [← hl, ← Nat.add_zero E.values.length, Ctx.val_next hF (by simp)]
    simp [ha.read hE hl, hb.read hE hl, word_sub, word_neg]
    ring

theorem scale_ensures {a : Var} {ga : Word} (by_ : Word) (ha : IsWord c n a ga) :
    Ensures c n (Build.scale a by_) fun h m => IsWord c m h (ga * by_) :=
  Ensures.emit _ fun E hl hF _ => by
    have hE := Ctx.follows_of_step hF
    refine ⟨by simp [Instr.outputs], ?_⟩
    rw [← hl, ← Nat.add_zero E.values.length, Ctx.val_next hF (by simp)]
    simp [ha.read hE hl, word_mul]

theorem mul_ensures {a b : Var} {ga gb : Word} (ha : IsWord c n a ga) (hb : IsWord c n b gb) :
    Ensures c n (Build.mul a b) fun h m => IsWord c m h (ga * gb) :=
  Ensures.emit _ fun E hl hF _ => by
    have hE := Ctx.follows_of_step hF
    refine ⟨by simp [Instr.outputs], ?_⟩
    rw [← hl, ← Nat.add_zero E.values.length, Ctx.val_next hF (by simp)]
    simp [ha.read hE hl, hb.read hE hl, word_mul]

/-- `assert_zero`: the check holds iff the word is zero. -/
theorem assertZero_ensures {a : Var} {ga : Word} (ha : IsWord c n a ga) :
    Ensures c n (Build.assertZero a) fun _ _ => ga = 0 := by
  refine (Ensures.emit (Q := fun _ _ => ga = 0) (.assertZero a) fun E hl hF hh => ?_).bind
    fun _ _ _ h => Ensures.pure h
  have holds : E.holds ∧ E.value a = 0 := hh
  rw [ha.read (Ctx.follows_of_step hF) hl] at holds
  exact word_injective (holds.2.trans word_zero.symm)

theorem assertEqual_ensures {a b : Var} {ga gb : Word} (ha : IsWord c n a ga) (hb : IsWord c n b gb) :
    Ensures c n (Build.assertEqual a b) fun _ _ => ga = gb :=
  (sub_ensures ha hb).bind fun _ _ _ hd => (assertZero_ensures hd).post fun _ _ h => gl_sub_eq_zero.mp h

theorem extMul_ensures {a b : ExtVar} {ga gb : Ext} (ha : IsExt c n a ga) (hb : IsExt c n b gb) :
    Ensures c n (Build.extMul a b) fun r m => IsExt c m r (ga * gb) :=
  (Ensures.emit (Q := fun h m => ∀ k : Fin 3, IsWord c m (h + k) ((ga * gb).coord k)) _ fun E hl hF _ => by
    have hE := Ctx.follows_of_step hF
    refine fun k => ⟨Nat.add_lt_add_left k.isLt n, ?_⟩
    rw [← hl, Ctx.val_next hF (by simp)]
    have ra : (fun i => E.value (a i)) = fun i => Word.cast (ga.coord i) := funext fun i => (ha i).read hE hl
    have rb : (fun i => E.value (b i)) = fun i => Word.cast (gb.coord i) := funext fun i => (hb i).read hE hl
    simp only [List.getD_eq_getElem?_getD, List.getElem?_ofFn]
    simp [ra, rb, ← coord_mul]).map fun _ _ h k => h k

end Words

end CompressionLean.Sim
