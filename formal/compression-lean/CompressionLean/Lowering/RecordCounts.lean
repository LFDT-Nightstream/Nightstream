import CompressionLean.Lowering.Valid

/-!
Owns `lower_counts`: the counts of a lowered run do not depend on the
statement values.

A statement word enters the lowering as its wire and as an entry of
`publics`. No step reads that entry, and a constant fold sees only forms,
which hold no statement value. So every step but `pub` keeps `publics` and
ignores it (`Frame`), and two lowerings of one program differ only in the
statement values (`Same`).
-/

namespace CompressionLean.Lowering

open CompressionLean.Circuit
open CompressionLean.Shrink

section Counts

variable {F : Type}

/-- `m` neither reads nor changes the statement values. -/
def Frame {α : Type} (m : Lower F α) : Prop :=
  ∀ s p, go m { s with publics := p } = ((go m s).1, { (go m s).2 with publics := p })

namespace Frame

private theorem pure {α : Type} (a : α) : Frame (Pure.pure a : Lower F α) := fun _ _ => rfl

private theorem bind {α β : Type} {m : Lower F α} {k : α → Lower F β} (hm : Frame m) (hk : ∀ a, Frame (k a)) :
    Frame (m >>= k) := by
  intro s p
  rw [go_bind, go_bind, hm, hk]

private theorem publics {α : Type} {m : Lower F α} (hm : Frame m) (s : State F) : (go m s).2.publics = s.publics :=
  congrArg (fun r => r.2.publics) (hm s s.publics)

end Frame

private theorem frame_cells (count : ℕ) : Frame (cells count : Lower F ℕ) := fun _ _ => rfl

private theorem frame_row (products : List (List (Wire × F) × List (Wire × F))) (linear : List (Wire × F))
    (three : products.length ≤ 3) : Frame (row products linear three) := fun _ _ => rfl

private theorem frame_block : Frame (block : Lower F ℕ) := fun _ _ => rfl

private theorem frame_fromFn (m : Fin 3 → Lower F (Form F)) (hm : ∀ k, Frame (m k)) : Frame (fromFn m) :=
  Frame.bind (hm 0) fun _ => Frame.bind (hm 1) fun _ => Frame.bind (hm 2) fun _ => Frame.pure _

/-- Two lowering states that differ at most in the statement values. -/
structure Same (s t : State F) : Prop where
  values : s.values = t.values
  calls : s.calls = t.calls
  cells : s.cells = t.cells
  blocks : s.blocks = t.blocks
  size : s.publics.size = t.publics.size

variable [Field F]

private theorem frame_cell : Frame (cell : Lower F (Form F)) := fun _ _ => rfl

private theorem frame_booleanRows : ∀ cs : List ℕ, Frame (booleanRows cs : Lower F Unit)
  | [] => Frame.pure _
  | _ :: rest => Frame.bind (frame_row _ _ _) fun _ => frame_booleanRows rest

variable [DecidableEq F]

private theorem frame_form (terms : List (Wire × F)) (constant : F) : Frame (form terms constant) := by
  unfold form
  split
  · exact Frame.pure _
  · exact Frame.bind frame_cell fun _ => Frame.bind (frame_row _ _ _) fun _ => Frame.pure _

private theorem frame_linear (parts : List (Form F × F)) : Frame (linear parts) := frame_form _ _

private theorem frame_mul (a b : Form F) : Frame (mul a b) := by
  unfold mul
  split
  · exact frame_linear _
  split
  · exact frame_linear _
  · exact Frame.bind frame_cell fun _ => Frame.bind (frame_row _ _ _) fun _ => Frame.pure _

private theorem frame_extMul (a b : Fin 3 → Form F) : Frame (extMul a b) := by
  unfold extMul
  split
  · exact frame_fromFn _ fun _ => frame_linear _
  · exact frame_fromFn _ fun _ =>
      Frame.bind frame_cell fun _ => Frame.bind (frame_row _ _ _) fun _ => Frame.pure _

private theorem frame_extInv (a : Fin 3 → Form F) : Frame (extInv a) := by
  unfold extInv
  split
  · exact Frame.pure _
  · exact Frame.bind (frame_fromFn _ fun _ => frame_cell) fun _ =>
      Frame.bind (frame_row _ _ _) fun _ => Frame.bind (frame_row _ _ _) fun _ =>
        Frame.bind (frame_row _ _ _) fun _ => Frame.pure _

private theorem frame_assertZero (a : Form F) : Frame (assertZero a) := by
  unfold assertZero
  split
  · exact Frame.pure _
  · exact frame_row _ _ _

private theorem frame_chain : ∀ (high : Form F) (cs : List ℕ), Frame (chain high cs)
  | _, [] => Frame.pure _
  | _, _ :: rest => Frame.bind (frame_mul _ _) fun h => frame_chain h rest

private theorem frame_bits (a : Form F) (count : ℕ) : Frame (bits a count) := by
  unfold bits
  refine Frame.bind (frame_cells _) fun first => Frame.bind (frame_booleanRows _) fun _ =>
    Frame.bind (frame_row _ _ _) fun _ => ?_
  dsimp only
  split
  · exact Frame.bind (frame_chain _ _) fun _ => Frame.bind (frame_row _ _ _) fun _ => Frame.pure _
  · exact Frame.pure _

private theorem frame_tieRows (b : ℕ) (inputs : Fin 16 → Form F) :
    ∀ ks : List (Fin 16), Frame (tieRows b inputs ks)
  | [] => Frame.pure _
  | _ :: rest => Frame.bind (frame_row _ _ _) fun _ => frame_tieRows b inputs rest

private theorem frame_permute (inputs : Fin 16 → Form F) : Frame (permute inputs) :=
  Frame.bind frame_block fun _ => Frame.bind (frame_tieRows _ _ _) fun _ => Frame.pure _

private theorem frame_step (stmt : Array F) (i : Instr) (hi : i ≠ .pub) : Frame (step stmt i) := by
  have push (forms : List (Form F)) : Frame (push forms) := fun _ _ => rfl
  have frame_readValue (v : Var) : Frame (readValue v : Lower F (Form F)) := fun _ _ => rfl
  have frame_readValues (vs : List Var) : Frame (readValues vs : Lower F (List (Form F))) := fun _ _ => rfl
  have frame_readCoordinates {n : ℕ} (a : Fin n → Var) : Frame (readCoordinates a : Lower F (Fin n → Form F)) :=
    fun _ _ => rfl
  cases i with
  | const value => exact push _
  | priv => exact Frame.bind frame_cell fun _ => push _
  | pub => exact absurd rfl hi
  | linear parts => exact Frame.bind (frame_readValues _) fun _ => Frame.bind (frame_linear _) fun _ => push _
  | mul a b =>
      exact Frame.bind (frame_readValue _) fun _ => Frame.bind (frame_readValue _) fun _ =>
        Frame.bind (frame_mul _ _) fun _ => push _
  | extMul a b =>
      exact Frame.bind (frame_readCoordinates _) fun _ => Frame.bind (frame_readCoordinates _) fun _ =>
        Frame.bind (frame_extMul _ _) fun _ => push _
  | extInv a => exact Frame.bind (frame_readCoordinates _) fun _ => Frame.bind (frame_extInv _) fun _ => push _
  | assertZero a => exact Frame.bind (frame_readValue _) fun _ => frame_assertZero _
  | bits a count => exact Frame.bind (frame_readValue _) fun _ => Frame.bind (frame_bits _ _) fun _ => push _
  | permute state => exact Frame.bind (frame_readCoordinates _) fun _ => Frame.bind (frame_permute _) fun _ => push _
  | hint count => exact Frame.bind (frame_cells _) fun _ => push _
  | r1cs rows columns => exact Frame.bind (frame_readValues _) fun _ => fun _ _ => rfl

/-- One step from states that differ only in the statement values, on any
statement words. -/
theorem step_same (stmt stmt' : Array F) (i : Instr) {s t : State F} (h : Same s t) :
    Same (go (step stmt i) s).2 (go (step stmt' i) t).2 := by
  obtain ⟨hv, hc, hcl, hb, hs⟩ := h
  by_cases hi : i = .pub
  · subst hi
    refine ⟨?_, hc, hcl, hb, ?_⟩
    · show s.values ++ #[.wire (.pub (s.publics.size + 1))] = t.values ++ #[.wire (.pub (t.publics.size + 1))]
      rw [hv, hs]
    · show (s.publics.push _).size = (t.publics.push _).size
      simp [hs]
  · have hm := frame_step stmt i hi
    have e : step stmt' i = step stmt i := by
      cases i <;> first | rfl | exact absurd rfl hi
    have ht : t = { s with publics := t.publics } := by
      rcases s with ⟨_, _, _, _, _⟩
      rcases t with ⟨_, _, _, _, _⟩
      simp_all
    rw [e, ht, hm s t.publics]
    exact ⟨rfl, rfl, rfl, rfl, by simp [hm.publics, hs]⟩

/-- The steps from states that differ only in the statement values, on any
statement words. -/
theorem steps_same (stmt stmt' : Array F) :
    ∀ (L : List Instr) {s t : State F}, Same s t → Same (steps stmt L s) (steps stmt' L t)
  | [], _, _, h => h
  | i :: rest, _, _, h => steps_same stmt stmt' rest (step_same stmt stmt' i h)

/-- **The counts do not depend on the statement**: the lowerings of one
program on any two statement lists have the same counts. -/
theorem lower_counts (L : Array Instr) (stmt stmt' : List F) : (lower L stmt).counts = (lower L stmt').counts := by
  have h := steps_same stmt.toArray stmt'.toArray L.toList (s := {}) (t := {}) ⟨rfl, rfl, rfl, rfl, rfl⟩
  simp [lower, Run.counts, Run.rows, h.calls, h.cells, h.blocks, h.size]

end Counts

end CompressionLean.Lowering
