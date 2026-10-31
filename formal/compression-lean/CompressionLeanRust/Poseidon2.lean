import NeoVerified.Funs
import CompressionLeanRust.Field
import CompressionLean.Ports.Replay.Transcript

/-! The translated permutation of `neo_verified::poseidon2` is F′'s `Poseidon2.permute`
(`permute_spec`), and so `Replay.permute` (`permute_sound`, `permute_complete`). Its constant tables
are F′'s rows by kernel computation (`initial_table`, `partial_table`, `terminal_table`,
`diagonal_table`).

`loop_upto` reads each translated loop as a step on `ℕ` indices, so the lane loops evaluate by
`simp` on a state of sixteen named lanes. The round loops go by induction (`upto_induct`): evaluating
them makes the kernel unfold every round when it checks the proof. `fOf` reads a translated word as
its element of `F`; every operation reduces its inputs, so each lemma holds for any word. -/

namespace CompressionLeanRust

open Aeneas Aeneas.Std Result
open neo_verified
open CompressionLean
open NightstreamFPrime.Spec (F goldilocksModulus)
open NightstreamFPrime.Export.NativePoseidon2 (add64_denote mul64_denote ofNat64_denote)
open Fin.CommRing

-- F′'s Poseidon2 names; opening its namespace would mix them with the translation's.
namespace P2
export NightstreamFPrime.Spec.Poseidon2 (ofNat sbox externalLayer internalLayer fullRound partialRound
  rounds permute initialConstants internalConstants terminalConstants internalDiagonal width halfFullRounds
  partialRounds mat4)
end P2

/-! ## Loops -/

/-- `step` at `k`, `k + s`, … while `k < n`, at most `fuel` times. -/
def upto {σ : Type} (n s : ℕ) (step : ℕ → σ → σ) : ℕ → ℕ → σ → σ
  | 0, _, a => a
  | fuel + 1, k, a => if k < n then upto n s step fuel (k + s) (step k a) else a

theorem wrapping_add_val (i j : Usize) (h : i.val + j.val < 4294967296) :
    (core.num.Usize.wrapping_add i j).val = i.val + j.val := by
  have : i.val + j.val < Usize.size := by scalar_tac
  simp [Nat.mod_eq_of_lt this]

/-- A translated `while i < n { …; i += s }` whose body is `step` on the rest `a` of its state. -/
theorem loop_upto {α σ β : Type} (body : α → Result (ControlFlow α β)) (pack : σ → Usize → α)
    (n s : ℕ) (S : Usize) (hS : S.val = s) (hs : 0 < s) (hn : n + s < 4294967296)
    (step : ℕ → σ → σ) (stop : σ → β)
    (hstep : ∀ a (i : Usize), i.val < n → i.val % s = 0 →
      body (pack a i) = ok (.cont (pack (step i.val a) (core.num.Usize.wrapping_add i S))))
    (hstop : ∀ a (i : Usize), n ≤ i.val → body (pack a i) = ok (.done (stop a))) :
    ∀ fuel a (i : Usize) k, i.val = k → k % s = 0 → n - k ≤ fuel →
      loop body (pack a i) = ok (stop (upto n s step fuel k a)) := by
  intro fuel
  induction fuel with
  | zero =>
    intro a i k hk _ h
    subst hk
    rw [loop, hstop a i (by omega)]
    simp only [bind_ok, upto]
  | succ fuel ih =>
    intro a i k hk hm h
    subst hk
    by_cases lt : i.val < n
    · rw [loop, hstep a i lt hm]
      simp only [bind_ok]
      have next : (core.num.Usize.wrapping_add i S).val = i.val + s := by
        rw [wrapping_add_val _ _ (by omega), hS]
      rw [ih _ _ _ next (by rw [Nat.add_mod_right]; exact hm) (by omega), upto, ite_eq_left lt]
    · rw [loop, hstop a i (by omega)]
      simp only [bind_ok, upto, ite_eq_right lt]

/-- `upto` keeps `P` when each step does. -/
theorem upto_induct {σ : Type} (n : ℕ) (step : ℕ → σ → σ) (P : ℕ → σ → Prop)
    (hstep : ∀ j b, j < n → P j b → P (j + 1) (step j b)) :
    ∀ fuel k a, n - k ≤ fuel → k ≤ n → P k a → P n (upto n 1 step fuel k a) := by
  intro fuel
  induction fuel with
  | zero =>
    intro k a h hk p
    obtain rfl : k = n := by omega
    exact p
  | succ fuel ih =>
    intro k a h hk p
    by_cases lt : k < n
    · rw [upto, ite_eq_left lt]
      exact ih (k + 1) (step k a) (by omega) (by omega) (hstep k a lt p)
    · obtain rfl : k = n := by omega
      rw [upto, ite_eq_right lt]
      exact p

/-! ## Arrays -/

/-- `a` with index `k` set to `x`. -/
def setN {α : Type} {n : Usize} (a : Aeneas.Std.Array α n) (k : ℕ) (x : α) : Aeneas.Std.Array α n :=
  Aeneas.Std.Array.from (a.val.set k x) (by rw [List.length_set]; exact a.property)

@[simp] theorem setN_val {α : Type} {n : Usize} (a : Aeneas.Std.Array α n) (k : ℕ) (x : α) :
    (setN a k x).val = a.val.set k x := Aeneas.Std.Array.from_val _ _

theorem index_eq {α : Type} [Inhabited α] {n : Usize} (a : Aeneas.Std.Array α n) (i : Usize)
    (h : i.val < n.val) : Aeneas.Std.Array.index_usize a i = ok a.val[i.val]! := by
  have hl : i.val < a.val.length := by rw [a.property]; exact h
  unfold Aeneas.Std.Array.index_usize
  rw [Aeneas.Std.Array.getElem?_Usize_eq, List.getElem?_eq_getElem hl, getElem!_pos a.val i.val hl]

theorem update_eq {α : Type} {n : Usize} (a : Aeneas.Std.Array α n) (i : Usize) (x : α)
    (h : i.val < n.val) : Aeneas.Std.Array.update a i x = ok (setN a i.val x) := by
  have hl : i.val < a.val.length := by rw [a.property]; exact h
  unfold Aeneas.Std.Array.update
  rw [Aeneas.Std.Array.getElem?_Usize_eq, List.getElem?_eq_getElem hl]
  rfl

theorem width_val : circuit.WIDTH.val = 16 := by
  unfold circuit.WIDTH
  rfl

theorem zero_val : (0#usize).val = 0 := by simp

theorem lt_width (i : Usize) : i < circuit.WIDTH ↔ i.val < 16 := by
  rw [UScalar.lt_equiv, width_val]

theorem list16 {α : Type} (l : List α) (h : l.length = 16) :
    ∃ s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 : α,
      l = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15] := by
  rcases l with _ | ⟨s0, _ | ⟨s1, _ | ⟨s2, _ | ⟨s3, _ | ⟨s4, _ | ⟨s5, _ | ⟨s6, _ | ⟨s7, _ | ⟨s8,
    _ | ⟨s9, _ | ⟨s10, _ | ⟨s11, _ | ⟨s12, _ | ⟨s13, _ | ⟨s14, _ | ⟨s15, _ | ⟨_, _⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩⟩ <;>
    simp at h
  exact ⟨s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, rfl⟩

/-! ## Words as elements of `F` -/

/-- A translated word as its element of `F`, canonical or not. -/
def fOf (g : field.Gl) : F := NightstreamFPrime.Spec.Poseidon2.ofNat g.val

/-- The lanes of a translated state as elements of `F`. -/
def lanes {n : Usize} (a : Aeneas.Std.Array field.Gl n) : List F := a.val.map fOf

theorem fOf_exact {c : U64} {x : Replay.Word} (h : Exact c x) : fOf c = x.word.denote := by
  unfold fOf UInt64.denote
  rw [h.val]
  rfl

theorem wordOf_denote (g : field.Gl) : (wordOf g).word.denote = fOf g :=
  ofNat64_denote g.val

theorem fOf_glAdd (a b : field.Gl) : fOf (glAdd a b) = fOf a + fOf b := by
  rw [fOf_exact (glAdd_exact a b), word_add, add64_denote _ _ (wordOf a).canonical (wordOf b).canonical,
    wordOf_denote, wordOf_denote]

theorem fOf_glMul (a b : field.Gl) : fOf (glMul a b) = fOf a * fOf b := by
  rw [fOf_exact (glMul_exact a b), word_mul, mul64_denote _ _ (wordOf a).canonical (wordOf b).canonical,
    wordOf_denote, wordOf_denote]

theorem fOf_redW (w : U64) : fOf (redW w) = fOf w := by
  rw [fOf_exact (show Exact (redW w) (wordOf w) from toU_red_word w), wordOf_denote]

theorem fOf_zero : fOf field.GL_ZERO = 0 := by
  unfold fOf field.GL_ZERO
  rfl

/-- F′ writes its numerals with `Fin`'s own `OfNat`, which `ring` reads as atoms. -/
theorem spec_two_mul (x : F) : 2 * x = x + x := by
  apply Fin.ext
  rw [Fin.val_mul, Fin.val_add]
  show 2 % goldilocksModulus * x.val % goldilocksModulus = _
  rw [show 2 % goldilocksModulus = 2 from rfl, Nat.two_mul]

theorem spec_three_mul (x : F) : 3 * x = x + x + x := by
  apply Fin.ext
  rw [Fin.val_mul, Fin.val_add, Fin.val_add]
  show 3 % goldilocksModulus * x.val % goldilocksModulus = _
  rw [show 3 % goldilocksModulus = 3 from rfl, Nat.mod_add_mod]
  congr 1
  ring

theorem glAdd_lt (a b : field.Gl) : (glAdd a b).val < goldilocksModulus :=
  (glAdd_exact a b).val ▸ (wordOf a + wordOf b).canonical

/-! ## The S-box -/

def sboxW (x : field.Gl) : field.Gl := glMul (glMul (glMul (glMul x x) (glMul x x)) (glMul x x)) x

theorem sbox_eq (x : field.Gl) : poseidon2.sbox x = ok (sboxW x) := by
  simp only [poseidon2.sbox, gl_mul_eq, bind_ok, sboxW]

theorem fOf_sboxW (x : field.Gl) : fOf (sboxW x) = NightstreamFPrime.Spec.Poseidon2.sbox (fOf x) := by
  simp only [sboxW, fOf_glMul, NightstreamFPrime.Spec.Poseidon2.sbox]

/-! ## The internal layer -/

def sumStep (l : List field.Gl) (k : ℕ) (sum : field.Gl) : field.Gl := glAdd sum l[k]!

def internalStep (l : List field.Gl) (sum : field.Gl) (k : ℕ) (out : Aeneas.Std.Array field.Gl 16#usize) :
    Aeneas.Std.Array field.Gl 16#usize :=
  setN out k (glAdd (glMul (redW poseidon2.DIAGONAL.val[k]!) l[k]!) sum)

theorem internal_sum_eq (state : Aeneas.Std.Array field.Gl 16#usize) :
    poseidon2.internal_layer_loop0 state field.GL_ZERO 0#usize =
      ok (upto 16 1 (sumStep state.val) 16 0 field.GL_ZERO) := by
  unfold poseidon2.internal_layer_loop0
  exact loop_upto _ (fun (a : field.Gl) (i : Usize) => (a, i)) 16 1 1#usize (by simp) (by omega) (by omega)
    (sumStep state.val) (fun a => a)
    (fun a i h _ => by
      simp only [poseidon2.internal_layer_loop0.body, (lt_width i).mpr h, ite_true,
        index_eq state i (by simpa using h), gl_add_eq, lift, bind_ok, sumStep])
    (fun a i h => by
      simp only [poseidon2.internal_layer_loop0.body, show ¬ i < circuit.WIDTH by rw [lt_width]; omega,
        ite_false])
    16 field.GL_ZERO 0#usize 0 (by simp) (by omega) (by omega)

theorem internal_out_eq (state : Aeneas.Std.Array field.Gl 16#usize) (sum : field.Gl) :
    poseidon2.internal_layer_loop1 state sum state 0#usize =
      ok (upto 16 1 (internalStep state.val sum) 16 0 state) := by
  unfold poseidon2.internal_layer_loop1
  exact loop_upto _ (fun (a : Aeneas.Std.Array field.Gl 16#usize) (i : Usize) => (a, i)) 16 1 1#usize
    (by simp) (by omega) (by omega) (internalStep state.val sum) (fun a => a)
    (fun a i h _ => by
      simp only [poseidon2.internal_layer_loop1.body, (lt_width i).mpr h, ite_true,
        index_eq state i (by simpa using h), index_eq poseidon2.DIAGONAL i (by simpa using h), reduce_eq,
        gl_add_eq, gl_mul_eq, update_eq a i _ (by simpa using h), lift, bind_ok, internalStep])
    (fun a i h => by
      simp only [poseidon2.internal_layer_loop1.body, show ¬ i < circuit.WIDTH by rw [lt_width]; omega,
        ite_false])
    16 state 0#usize 0 (by simp) (by omega) (by omega)

/-- What `internal_layer` returns. -/
def internalW (state : Aeneas.Std.Array field.Gl 16#usize) : Aeneas.Std.Array field.Gl 16#usize :=
  upto 16 1 (internalStep state.val (upto 16 1 (sumStep state.val) 16 0 field.GL_ZERO)) 16 0 state

theorem internal_layer_eq (state : Aeneas.Std.Array field.Gl 16#usize) :
    poseidon2.internal_layer state = ok (internalW state) := by
  simp only [poseidon2.internal_layer, internal_sum_eq, internal_out_eq, bind_ok, internalW]

/-! ## The constant tables are F′'s rows, by kernel computation -/

theorem initial_table :
    poseidon2.INITIAL.val.map (fun row => row.val.map UScalar.val) = P2.initialConstants := by
  unfold poseidon2.INITIAL
  decide

theorem partial_table : poseidon2.PARTIAL.val.map UScalar.val = P2.internalConstants := by
  unfold poseidon2.PARTIAL
  decide

theorem terminal_table :
    poseidon2.TERMINAL.val.map (fun row => row.val.map UScalar.val) = P2.terminalConstants := by
  unfold poseidon2.TERMINAL
  decide

theorem diagonal_table : poseidon2.DIAGONAL.val.map UScalar.val = P2.internalDiagonal := by
  unfold poseidon2.DIAGONAL
  decide

theorem fOf_u64 (w : U64) : fOf (redW w) = P2.ofNat w.val := by
  rw [fOf_redW]
  rfl

theorem getElem!_map_val {l : List U64} {k : ℕ} (h : k < l.length) :
    (l[k]!).val = (l.map UScalar.val).getD k 0 := by
  rw [getElem!_pos l k h, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_getElem h]
  rfl

theorem diagonal_at (k : ℕ) (h : k < 16) :
    fOf (redW poseidon2.DIAGONAL.val[k]!) = P2.ofNat (P2.internalDiagonal.getD k 0) := by
  rw [fOf_u64, getElem!_map_val (by simpa using h), diagonal_table]

theorem internalW_lanes (state : Aeneas.Std.Array field.Gl 16#usize) :
    lanes (internalW state) = P2.internalLayer (lanes state) := by
  obtain ⟨s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, hv⟩ :=
    list16 state.val (by simp)
  simp only [lanes, internalW, upto, internalStep, sumStep, Nat.reduceAdd, Nat.reduceLT, ↓reduceIte,
    setN_val, hv, List.set_cons_zero, List.set_cons_succ, List.getElem!_cons_zero, List.getElem!_cons_succ,
    List.map_cons, List.map_nil, fOf_glAdd, fOf_glMul, fOf_zero, diagonal_at]
  simp [P2.internalLayer, P2.width, List.range_succ]

/-! ## The external layer -/

abbrev Sums := Aeneas.Std.Array field.Gl 16#usize × field.Gl × field.Gl × field.Gl × field.Gl

/-- One block of `M_4`, as `apply_mat4` writes it, and its four outputs added to the sums. -/
def mixStep (l : List field.Gl) (k : ℕ) (a : Sums) : Sums :=
  let x0 := l[k]!
  let x1 := l[k + 1]!
  let x2 := l[k + 2]!
  let x3 := l[k + 3]!
  let t01 := glAdd x0 x1
  let t23 := glAdd x2 x3
  let t0123 := glAdd t01 t23
  let t01123 := glAdd t0123 x1
  let t01233 := glAdd t0123 x3
  let y0 := glAdd t01123 t01
  let y1 := glAdd t01123 (glAdd x2 x2)
  let y2 := glAdd t01233 t23
  let y3 := glAdd t01233 (glAdd x0 x0)
  (setN (setN (setN (setN a.1 k y0) (k + 1) y1) (k + 2) y2) (k + 3) y3,
    glAdd a.2.1 y0, glAdd a.2.2.1 y1, glAdd a.2.2.2.1 y2, glAdd a.2.2.2.2 y3)

/-- Each lane of one block plus the sum at its position. -/
def addStep (l : List field.Gl) (s0 s1 s2 s3 : field.Gl) (k : ℕ)
    (out : Aeneas.Std.Array field.Gl 16#usize) : Aeneas.Std.Array field.Gl 16#usize :=
  setN (setN (setN (setN out k (glAdd l[k]! s0)) (k + 1) (glAdd l[k + 1]! s1)) (k + 2)
    (glAdd l[k + 2]! s2)) (k + 3) (glAdd l[k + 3]! s3)

theorem offset_val (i : Usize) (j : ℕ) (J : Usize) (hJ : J.val = j) (hj : j ≤ 4) (h : i.val < 16) :
    (core.num.Usize.wrapping_add i J).val = i.val + j := by
  rw [wrapping_add_val i J (by omega), hJ]

theorem mix_eq (state : Aeneas.Std.Array field.Gl 16#usize) :
    poseidon2.external_layer_loop0 state state field.GL_ZERO field.GL_ZERO field.GL_ZERO field.GL_ZERO
        0#usize =
      let r := upto 16 4 (mixStep state.val) 16 0
        (state, field.GL_ZERO, field.GL_ZERO, field.GL_ZERO, field.GL_ZERO)
      ok (r.1, r.2.1, r.2.2.1, r.2.2.2.1, r.2.2.2.2) := by
  unfold poseidon2.external_layer_loop0
  exact loop_upto _ (fun (a : Sums) (i : Usize) => (a.1, a.2.1, a.2.2.1, a.2.2.2.1, a.2.2.2.2, i)) 16 4
    4#usize (by simp) (by omega) (by omega) (mixStep state.val)
    (fun a => (a.1, a.2.1, a.2.2.1, a.2.2.2.1, a.2.2.2.2))
    (fun a i h m => by
      have e1 := offset_val i 1 1#usize (by simp) (by omega) h
      have e2 := offset_val i 2 2#usize (by simp) (by omega) h
      have e3 := offset_val i 3 3#usize (by simp) (by omega) h
      have b0 : i.val < (16#usize).val := by simpa using h
      have b1 : (core.num.Usize.wrapping_add i 1#usize).val < (16#usize).val := by rw [e1]; scalar_tac
      have b2 : (core.num.Usize.wrapping_add i 2#usize).val < (16#usize).val := by rw [e2]; scalar_tac
      have b3 : (core.num.Usize.wrapping_add i 3#usize).val < (16#usize).val := by rw [e3]; scalar_tac
      simp only [poseidon2.external_layer_loop0.body, (lt_width i).mpr h, ite_true, index_eq state i b0,
        index_eq state _ b1, index_eq state _ b2, index_eq state _ b3, gl_add_eq, lift, bind_ok,
        update_eq _ i _ b0, update_eq _ _ _ b1, update_eq _ _ _ b2, update_eq _ _ _ b3, e1, e2, e3, mixStep])
    (fun a i h => by
      simp only [poseidon2.external_layer_loop0.body, show ¬ i < circuit.WIDTH by rw [lt_width]; omega,
        ite_false])
    16 (state, field.GL_ZERO, field.GL_ZERO, field.GL_ZERO, field.GL_ZERO) 0#usize 0 (by simp) (by omega)
    (by omega)

theorem add_sums_eq (mixed : Aeneas.Std.Array field.Gl 16#usize) (s0 s1 s2 s3 : field.Gl) :
    poseidon2.external_layer_loop1 mixed s0 s1 s2 s3 mixed 0#usize =
      ok (upto 16 4 (addStep mixed.val s0 s1 s2 s3) 16 0 mixed) := by
  unfold poseidon2.external_layer_loop1
  exact loop_upto _ (fun (a : Aeneas.Std.Array field.Gl 16#usize) (i : Usize) => (a, i)) 16 4 4#usize
    (by simp) (by omega) (by omega) (addStep mixed.val s0 s1 s2 s3) (fun a => a)
    (fun a i h m => by
      have e1 := offset_val i 1 1#usize (by simp) (by omega) h
      have e2 := offset_val i 2 2#usize (by simp) (by omega) h
      have e3 := offset_val i 3 3#usize (by simp) (by omega) h
      have b0 : i.val < (16#usize).val := by simpa using h
      have b1 : (core.num.Usize.wrapping_add i 1#usize).val < (16#usize).val := by rw [e1]; scalar_tac
      have b2 : (core.num.Usize.wrapping_add i 2#usize).val < (16#usize).val := by rw [e2]; scalar_tac
      have b3 : (core.num.Usize.wrapping_add i 3#usize).val < (16#usize).val := by rw [e3]; scalar_tac
      simp only [poseidon2.external_layer_loop1.body, (lt_width i).mpr h, ite_true, index_eq mixed i b0,
        index_eq mixed _ b1, index_eq mixed _ b2, index_eq mixed _ b3, gl_add_eq, lift, bind_ok,
        update_eq _ i _ b0, update_eq _ _ _ b1, update_eq _ _ _ b2, update_eq _ _ _ b3, e1, e2, e3, addStep])
    (fun a i h => by
      simp only [poseidon2.external_layer_loop1.body, show ¬ i < circuit.WIDTH by rw [lt_width]; omega,
        ite_false])
    16 mixed 0#usize 0 (by simp) (by omega) (by omega)

/-- What `external_layer` returns. -/
def externalW (state : Aeneas.Std.Array field.Gl 16#usize) : Aeneas.Std.Array field.Gl 16#usize :=
  let r := upto 16 4 (mixStep state.val) 16 0
    (state, field.GL_ZERO, field.GL_ZERO, field.GL_ZERO, field.GL_ZERO)
  upto 16 4 (addStep r.1.val r.2.1 r.2.2.1 r.2.2.2.1 r.2.2.2.2) 16 0 r.1

theorem external_layer_eq (state : Aeneas.Std.Array field.Gl 16#usize) :
    poseidon2.external_layer state = ok (externalW state) := by
  simp only [poseidon2.external_layer, mix_eq, add_sums_eq, bind_ok, externalW]
  rfl

theorem externalW_lanes (state : Aeneas.Std.Array field.Gl 16#usize) :
    lanes (externalW state) = P2.externalLayer (lanes state) := by
  obtain ⟨s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, hv⟩ :=
    list16 state.val (by simp)
  simp only [lanes, externalW, upto, mixStep, addStep, Nat.reduceAdd, Nat.reduceLT, ↓reduceIte,
    setN_val, hv, List.set_cons_zero, List.set_cons_succ, List.getElem!_cons_zero, List.getElem!_cons_succ,
    List.map_cons, List.map_nil, fOf_glAdd, fOf_zero]
  simp only [P2.externalLayer, P2.width, P2.mat4, List.range_succ, List.range_zero, List.nil_append,
    List.cons_append, List.flatMap_cons, List.flatMap_nil, List.drop, List.take, List.map_cons, List.map_nil,
    List.foldl_cons, List.foldl_nil, Nat.reduceMul, Nat.reduceAdd, Nat.reduceMod, List.getD_cons_zero,
    List.getD_cons_succ, List.cons.injEq, and_true, spec_two_mul, spec_three_mul]
  and_intros <;> ring

/-- Every lane of `externalW` is a sum, so its word is canonical. -/
theorem externalW_canonical (state : Aeneas.Std.Array field.Gl 16#usize) :
    ∀ g ∈ (externalW state).val, g.val < goldilocksModulus := by
  obtain ⟨s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, hv⟩ :=
    list16 state.val (by simp)
  simp only [externalW, upto, mixStep, addStep, Nat.reduceAdd, Nat.reduceLT, ↓reduceIte,
    setN_val, hv, List.set_cons_zero, List.set_cons_succ, List.getElem!_cons_zero, List.getElem!_cons_succ,
    List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  exact ⟨glAdd_lt _ _, glAdd_lt _ _, glAdd_lt _ _, glAdd_lt _ _, glAdd_lt _ _, glAdd_lt _ _, glAdd_lt _ _,
    glAdd_lt _ _, glAdd_lt _ _, glAdd_lt _ _, glAdd_lt _ _, glAdd_lt _ _, glAdd_lt _ _, glAdd_lt _ _,
    glAdd_lt _ _, glAdd_lt _ _⟩

/-! ## The rounds -/

/-- Each lane plus its constant, through the S-box. -/
def boxStep (l : List field.Gl) (c : List U64) (k : ℕ) (out : Aeneas.Std.Array field.Gl 16#usize) :
    Aeneas.Std.Array field.Gl 16#usize :=
  setN out k (sboxW (glAdd l[k]! (redW c[k]!)))

def fullW (state : Aeneas.Std.Array field.Gl 16#usize) (c : Aeneas.Std.Array U64 16#usize) :
    Aeneas.Std.Array field.Gl 16#usize :=
  externalW (upto 16 1 (boxStep state.val c.val) 16 0 state)

def partialW (state : Aeneas.Std.Array field.Gl 16#usize) (c : U64) : Aeneas.Std.Array field.Gl 16#usize :=
  internalW (setN state 0 (sboxW (glAdd state.val[0]! (redW c))))

theorem full_round_eq (state : Aeneas.Std.Array field.Gl 16#usize) (c : Aeneas.Std.Array U64 16#usize) :
    poseidon2.full_round state c = ok (fullW state c) := by
  have box : poseidon2.full_round_loop state c state 0#usize =
      ok (upto 16 1 (boxStep state.val c.val) 16 0 state) := by
    unfold poseidon2.full_round_loop
    exact loop_upto _ (fun (a : Aeneas.Std.Array field.Gl 16#usize) (i : Usize) => (a, i)) 16 1 1#usize
      (by simp) (by omega) (by omega) (boxStep state.val c.val) (fun a => a)
      (fun a i h _ => by
        simp only [poseidon2.full_round_loop.body, (lt_width i).mpr h, ite_true,
          index_eq state i (by simpa using h), index_eq c i (by simpa using h), reduce_eq, gl_add_eq, sbox_eq,
          update_eq a i _ (by simpa using h), lift, bind_ok, boxStep])
      (fun a i h => by
        simp only [poseidon2.full_round_loop.body, show ¬ i < circuit.WIDTH by rw [lt_width]; omega,
          ite_false])
      16 state 0#usize 0 (by simp) (by omega) (by omega)
  simp only [poseidon2.full_round, box, external_layer_eq, bind_ok, fullW]

theorem partial_round_eq (state : Aeneas.Std.Array field.Gl 16#usize) (c : U64) :
    poseidon2.partial_round state c = ok (partialW state c) := by
  simp only [poseidon2.partial_round, index_eq state 0#usize (by simp), reduce_eq, gl_add_eq, sbox_eq,
    update_eq state 0#usize _ (by simp), internal_layer_eq, bind_ok, partialW, zero_val]

/-- F′'s `fullRound` with its row of constants given. -/
def fullRoundOf (row : List ℕ) (l : List F) : List F :=
  P2.externalLayer ((List.range 16).map fun i => P2.sbox (l.getD i 0 + P2.ofNat (row.getD i 0)))

/-- F′'s `partialRound` with its constant given. -/
def partialRoundOf (c : ℕ) (l : List F) : List F :=
  P2.internalLayer ((List.range 16).map fun i =>
    if i = 0 then P2.sbox (l.getD 0 0 + P2.ofNat c) else l.getD i 0)

theorem fullW_lanes (state : Aeneas.Std.Array field.Gl 16#usize) (c : Aeneas.Std.Array U64 16#usize) :
    lanes (fullW state c) = fullRoundOf (c.val.map UScalar.val) (lanes state) := by
  rw [fullW, externalW_lanes, fullRoundOf]
  apply congrArg P2.externalLayer
  obtain ⟨s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, hv⟩ :=
    list16 state.val (by simp)
  obtain ⟨c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, hc⟩ :=
    list16 c.val (by simp)
  simp only [lanes, upto, boxStep, Nat.reduceAdd, Nat.reduceLT, ↓reduceIte, setN_val, hv, hc,
    List.set_cons_zero, List.set_cons_succ, List.getElem!_cons_zero, List.getElem!_cons_succ, List.map_cons,
    List.map_nil, fOf_sboxW, fOf_glAdd, fOf_u64]
  simp [List.range_succ]

theorem partialW_lanes (state : Aeneas.Std.Array field.Gl 16#usize) (c : U64) :
    lanes (partialW state c) = partialRoundOf c.val (lanes state) := by
  rw [partialW, internalW_lanes, partialRoundOf]
  apply congrArg P2.internalLayer
  obtain ⟨s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, hv⟩ :=
    list16 state.val (by simp)
  simp only [lanes, setN_val, hv, List.set_cons_zero, List.getElem!_cons_zero, List.map_cons, List.map_nil,
    fOf_sboxW, fOf_glAdd, fOf_u64]
  simp [List.range_succ]

/-! ## The permutation -/

theorem half_val : poseidon2.HALF_FULL_ROUNDS.val = 4 := by
  unfold poseidon2.HALF_FULL_ROUNDS
  simp

theorem partial_val : poseidon2.PARTIAL_ROUNDS.val = 22 := by
  unfold poseidon2.PARTIAL_ROUNDS
  simp

/-- The full rounds of one table, from round `0`. -/
def fullsW (rows : Aeneas.Std.Array (Aeneas.Std.Array U64 16#usize) 4#usize)
    (state : Aeneas.Std.Array field.Gl 16#usize) : Aeneas.Std.Array field.Gl 16#usize :=
  upto 4 1 (fun k s => fullW s rows.val[k]!) 4 0 state

def partialsW (state : Aeneas.Std.Array field.Gl 16#usize) : Aeneas.Std.Array field.Gl 16#usize :=
  upto 22 1 (fun k s => partialW s poseidon2.PARTIAL.val[k]!) 22 0 state

/-- What `permute` returns. -/
def permuteW (input : Aeneas.Std.Array field.Gl 16#usize) : Aeneas.Std.Array field.Gl 16#usize :=
  fullsW poseidon2.TERMINAL (partialsW (fullsW poseidon2.INITIAL (externalW input)))

theorem full_rounds_eq (rows : Aeneas.Std.Array (Aeneas.Std.Array U64 16#usize) 4#usize)
    (state : Aeneas.Std.Array field.Gl 16#usize) :
    poseidon2.full_rounds state rows = ok (fullsW rows state) := by
  have lt : ∀ i : Usize, i < poseidon2.HALF_FULL_ROUNDS ↔ i.val < 4 := fun i => by
    rw [UScalar.lt_equiv, half_val]
  unfold poseidon2.full_rounds poseidon2.full_rounds_loop
  exact loop_upto _ (fun (a : Aeneas.Std.Array field.Gl 16#usize) (i : Usize) => (a, i)) 4 1 1#usize
    (by simp) (by omega) (by omega) (fun k s => fullW s rows.val[k]!) (fun a => a)
    (fun a i h _ => by
      simp only [poseidon2.full_rounds_loop.body, (lt i).mpr h, ite_true, index_eq rows i (by simpa using h),
        full_round_eq, lift, bind_ok])
    (fun a i h => by
      simp only [poseidon2.full_rounds_loop.body, show ¬ i < poseidon2.HALF_FULL_ROUNDS by rw [lt]; omega,
        ite_false])
    4 state 0#usize 0 (by simp) (by omega) (by omega)

theorem partials_eq (state : Aeneas.Std.Array field.Gl 16#usize) :
    poseidon2.permute_loop state 0#usize = ok (partialsW state) := by
  have lt : ∀ i : Usize, i < poseidon2.PARTIAL_ROUNDS ↔ i.val < 22 := fun i => by
    rw [UScalar.lt_equiv, partial_val]
  unfold poseidon2.permute_loop
  exact loop_upto _ (fun (a : Aeneas.Std.Array field.Gl 16#usize) (i : Usize) => (a, i)) 22 1 1#usize
    (by simp) (by omega) (by omega) (fun k s => partialW s poseidon2.PARTIAL.val[k]!) (fun a => a)
    (fun a i h _ => by
      simp only [poseidon2.permute_loop.body, (lt i).mpr h, ite_true,
        index_eq poseidon2.PARTIAL i (by simpa using h), partial_round_eq, lift, bind_ok])
    (fun a i h => by
      simp only [poseidon2.permute_loop.body, show ¬ i < poseidon2.PARTIAL_ROUNDS by rw [lt]; omega,
        ite_false])
    22 state 0#usize 0 (by simp) (by omega) (by omega)

theorem permute_eq (input : Aeneas.Std.Array field.Gl 16#usize) :
    poseidon2.permute input = ok (permuteW input) := by
  simp only [poseidon2.permute, external_layer_eq, full_rounds_eq, partials_eq, bind_ok, permuteW]

/-! ## Each round loop is F′'s `rounds` -/

theorem fullRound_of (rows : List (List ℕ)) (round : ℕ) (l : List F) :
    P2.fullRound rows round l = fullRoundOf (rows.getD round []) l := rfl

theorem partialRound_of (round : ℕ) (l : List F) :
    P2.partialRound round l = partialRoundOf (P2.internalConstants.getD round 0) l := rfl

theorem row_at (rows : Aeneas.Std.Array (Aeneas.Std.Array U64 16#usize) 4#usize) (k : ℕ) (h : k < 4) :
    (rows.val[k]!).val.map UScalar.val = (rows.val.map fun row => row.val.map UScalar.val).getD k [] := by
  have hl : k < rows.val.length := by simpa using h
  rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_getElem hl, getElem!_pos rows.val k hl]
  rfl

theorem partial_at (k : ℕ) (h : k < 22) :
    (poseidon2.PARTIAL.val[k]!).val = P2.internalConstants.getD k 0 := by
  rw [getElem!_map_val (by simpa using h), partial_table]

theorem foldl_range_succ {σ : Type} (f : σ → ℕ → σ) (a : σ) (j : ℕ) :
    (List.range (j + 1)).foldl f a = f ((List.range j).foldl f a) j := by
  rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]

theorem fullsW_lanes (rows : Aeneas.Std.Array (Aeneas.Std.Array U64 16#usize) 4#usize)
    (table : List (List ℕ)) (h : rows.val.map (fun row => row.val.map UScalar.val) = table)
    (state : Aeneas.Std.Array field.Gl 16#usize) :
    lanes (fullsW rows state) = P2.rounds (P2.fullRound table) P2.halfFullRounds (lanes state) :=
  upto_induct 4 (fun k s => fullW s rows.val[k]!)
    (fun j b => lanes b = (List.range j).foldl (fun acc r => P2.fullRound table r acc) (lanes state))
    (fun j b hj p => by
      rw [fullW_lanes, foldl_range_succ, ← p, fullRound_of, row_at rows j hj, h])
    4 0 state (by omega) (by omega) rfl

theorem partialsW_lanes (state : Aeneas.Std.Array field.Gl 16#usize) :
    lanes (partialsW state) = P2.rounds P2.partialRound P2.partialRounds (lanes state) :=
  upto_induct 22 (fun k s => partialW s poseidon2.PARTIAL.val[k]!)
    (fun j b => lanes b = (List.range j).foldl (fun acc r => P2.partialRound r acc) (lanes state))
    (fun j b hj p => by
      rw [partialW_lanes, foldl_range_succ, ← p, partialRound_of, partial_at j hj])
    22 0 state (by omega) (by omega) rfl

theorem permuteW_lanes (input : Aeneas.Std.Array field.Gl 16#usize) :
    lanes (permuteW input) = P2.permute (lanes input) := by
  rw [permuteW, fullsW_lanes _ _ terminal_table, partialsW_lanes, fullsW_lanes _ _ initial_table,
    externalW_lanes]
  rfl

theorem permuteW_canonical (input : Aeneas.Std.Array field.Gl 16#usize) :
    ∀ g ∈ (permuteW input).val, g.val < goldilocksModulus :=
  upto_induct 4 (fun k s => fullW s poseidon2.TERMINAL.val[k]!)
    (fun j b => 0 < j → ∀ g ∈ b.val, g.val < goldilocksModulus)
    (fun _ b _ _ _ => externalW_canonical _)
    4 0 _ (by omega) (by omega) (fun h => absurd h (Nat.lt_irrefl 0)) (by omega)

/-- The translated permutation is F′'s `Poseidon2.permute` on the elements that its input words name,
and it returns canonical words. -/
theorem permute_spec (input output : Aeneas.Std.Array field.Gl 16#usize)
    (h : poseidon2.permute input = ok output) :
    lanes output = P2.permute (lanes input) ∧ ∀ g ∈ output.val, g.val < goldilocksModulus := by
  rw [permute_eq, Result.ok.injEq] at h
  subst h
  exact ⟨permuteW_lanes input, permuteW_canonical input⟩

/-! ## `Replay.permute` is F′'s `Poseidon2.permute` -/

theorem replay_permute (ws : List Replay.Word) (h : ws.length = 16) :
    (Replay.permute ws.toArray).toList.map (fun w => w.word.denote) =
      P2.permute (ws.map fun w => w.word.denote) := by
  obtain ⟨w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, hw⟩ := list16 ws h
  subst hw
  simp only [Replay.permute, List.getElem!_toArray, List.getElem!_cons_zero, List.getElem!_cons_succ,
    List.toList_toArray, List.map_cons, List.map_nil]
  exact NightstreamFPrime.Export.NativePoseidon2.State64.permute64_denote _

theorem fOf_val {g : field.Gl} (h : g.val < goldilocksModulus) : (fOf g).val = g.val := by
  unfold fOf P2.ofNat
  exact Nat.mod_eq_of_lt h

theorem denote_val (w : Replay.Word) : (w.word.denote).val = w.toNat :=
  Nat.mod_eq_of_lt w.canonical

theorem permute_sound : Sound poseidon2.permute glValues
    (fun x => (Replay.permute (x.val.map wordOf).toArray).toList.map Replay.Word.toNat) := by
  intro x y h
  obtain ⟨hl, hc⟩ := permute_spec x y h
  have hr := replay_permute (x.val.map wordOf) (by simp)
  rw [List.map_map,
    show (fun w : Replay.Word => w.word.denote) ∘ wordOf = fOf from funext wordOf_denote] at hr
  have same := congrArg (List.map Fin.val) (hl.trans ((show lanes x = x.val.map fOf from rfl) ▸ hr.symm))
  simp only [lanes, List.map_map] at same
  have left : List.map (Fin.val ∘ fOf) y.val = glValues y :=
    List.map_congr_left (fun g hg => fOf_val (hc g hg))
  have right : List.map (Fin.val ∘ fun w : Replay.Word => w.word.denote)
      (Replay.permute (x.val.map wordOf).toArray).toList =
        (Replay.permute (x.val.map wordOf).toArray).toList.map Replay.Word.toNat :=
    List.map_congr_left (fun w _ => denote_val w)
  exact left.symm.trans (same.trans right)

theorem permute_complete : Complete (fun _ => True) poseidon2.permute glValues
    (fun x => (Replay.permute (x.val.map wordOf).toArray).toList.map Replay.Word.toNat) :=
  complete_of permute_sound ⟨default, trivial⟩ (fun x _ => ⟨_, permute_eq x⟩)

end CompressionLeanRust
