import CompressionLean.Circuit.Record

/-!
Owns `lower_counts`: the counts of a lowering do not depend on the
statement words. The words enter only `publics` (`statementWord`), and no
step reads `publics` except for its size.

`Frame m`: the lowering step `m` neither reads nor writes `publics`. Every
lowering of an instruction other than `pub` frames (`step_frame`).
-/

namespace CompressionLean.Circuit

open CompressionLean.Shrink

variable {F : Type}

/-- `m` neither reads nor writes `publics`. -/
structure Frame {α : Type} (m : Lower F α) : Prop where
  intro ::
  run : ∀ (s : State F) (q : Array F),
    m.run { s with publics := q } = ((m.run s).1, { (m.run s).2 with publics := q })

namespace Frame

variable {α β : Type}

theorem pure (a : α) : Frame (Pure.pure a : Lower F α) := ⟨fun _ _ => rfl⟩

theorem bind {m : Lower F α} {f : α → Lower F β} (hm : Frame m) (hf : ∀ a, Frame (f a)) : Frame (m >>= f) :=
  ⟨fun s q => by
    show (f (m.run { s with publics := q }).1).run (m.run { s with publics := q }).2 = _
    rw [hm.run, (hf _).run]
    rfl⟩

theorem modifyGet {f : State F → α × State F}
    (hf : ∀ s q, f { s with publics := q } = ((f s).1, { (f s).2 with publics := q })) :
    Frame (MonadState.modifyGet f : Lower F α) := ⟨fun s q => hf s q⟩

theorem modify {f : State F → State F} (hf : ∀ s q, f { s with publics := q } = { f s with publics := q }) :
    Frame (MonadState.modifyGet fun s => ((), f s) : Lower F Unit) := ⟨fun s q => by
  show ((), f { s with publics := q }) = _
  rw [hf]
  rfl⟩

/-- A step that reads the state but not `publics`. -/
theorem get_bind {f : State F → Lower F α} (reads : ∀ s q, f { s with publics := q } = f s)
    (hf : ∀ s, Frame (f s)) : Frame (MonadState.get >>= f) := ⟨fun s q => by
  show (f { s with publics := q }).run { s with publics := q } = _
  rw [reads, (hf s).run]
  rfl⟩

/-- A framed step keeps `publics`. -/
theorem keeps {m : Lower F α} (h : Frame m) (s : State F) : (m.run s).2.publics = s.publics :=
  congrArg (fun r => r.2.publics) (h.run s s.publics)

end Frame

theorem cells_frame (count : ℕ) : Frame (cells count : Lower F ℕ) := Frame.modifyGet fun _ _ => rfl

theorem row_frame (products : List (List (Wire × F) × List (Wire × F))) (linear : List (Wire × F))
    (three : products.length ≤ 3) : Frame (row products linear three) := Frame.modify fun _ _ => rfl

theorem push_frame (forms : List (Form F)) : Frame (push forms) := Frame.modify fun _ _ => rfl

theorem fromFn_frame {m : Fin 3 → Lower F (Form F)} (h : ∀ k, Frame (m k)) : Frame (fromFn m) :=
  Frame.bind (h 0) fun _ => Frame.bind (h 1) fun _ => Frame.bind (h 2) fun _ => .pure _

variable [Field F]

theorem cell_frame : Frame (cell : Lower F (Form F)) := Frame.modifyGet fun _ _ => rfl

theorem readValue_frame (v : Var) : Frame (readValue v : Lower F (Form F)) :=
  Frame.get_bind (fun _ _ => rfl) fun _ => .pure _

theorem readValues_frame (vs : List Var) : Frame (readValues vs : Lower F (List (Form F))) :=
  Frame.get_bind (fun _ _ => rfl) fun _ => .pure _

theorem readCoordinates_frame {n : ℕ} (a : Fin n → Var) : Frame (readCoordinates a : Lower F (Fin n → Form F)) :=
  Frame.get_bind (fun _ _ => rfl) fun _ => .pure _

/-- A new cell and its row. -/
theorem cellRow_frame (products : List (List (Wire × F) × List (Wire × F))) (linear : Form F → List (Wire × F))
    (three : products.length ≤ 3) :
    Frame (do let c ← cell; row products (linear c) three; pure c : Lower F (Form F)) :=
  Frame.bind cell_frame fun _ => Frame.bind (row_frame _ _ _) fun _ => .pure _

theorem booleanRows_frame : ∀ cs : List ℕ, Frame (booleanRows cs : Lower F Unit)
  | [] => .pure _
  | _ :: rest => Frame.bind (row_frame _ _ _) fun _ => booleanRows_frame rest

variable [DecidableEq F]

theorem form_frame (terms : List (Wire × F)) (constant : F) : Frame (form terms constant) := by
  unfold form
  split
  · exact .pure _
  · exact cellRow_frame _ _ _

theorem linear_frame (parts : List (Form F × F)) : Frame (linear parts) := form_frame _ _

theorem mul_frame (a b : Form F) : Frame (mul a b) := by
  unfold mul
  split
  · exact linear_frame _
  split
  · exact linear_frame _
  · exact cellRow_frame _ _ _

theorem extMul_frame (a b : Fin 3 → Form F) : Frame (extMul a b) := by
  unfold extMul
  split
  · exact fromFn_frame fun _ => linear_frame _
  · exact fromFn_frame fun _ => cellRow_frame _ _ _

theorem extInv_frame (a : Fin 3 → Form F) : Frame (extInv a) := by
  unfold extInv
  split
  · exact .pure _
  · exact Frame.bind (fromFn_frame fun _ => cell_frame) fun _ => Frame.bind (row_frame _ _ _) fun _ =>
      Frame.bind (row_frame _ _ _) fun _ => Frame.bind (row_frame _ _ _) fun _ => .pure _

theorem assertZero_frame (a : Form F) : Frame (assertZero a) := by
  unfold assertZero
  split
  · exact .pure _
  · exact row_frame _ _ _

theorem chain_frame : ∀ (high : Form F) (cs : List ℕ), Frame (chain high cs)
  | _, [] => .pure _
  | high, _ :: rest => Frame.bind (mul_frame high _) fun next => chain_frame next rest

theorem bits_frame (a : Form F) (count : ℕ) : Frame (bits a count) := by
  unfold bits
  refine Frame.bind (cells_frame _) fun _ => Frame.bind (booleanRows_frame _) fun _ =>
    Frame.bind (row_frame _ _ _) fun _ => ?_
  dsimp only
  split
  · exact Frame.bind (chain_frame _ _) fun _ => Frame.bind (row_frame _ _ _) fun _ => .pure _
  · exact .pure _

theorem tieRows_frame (b : ℕ) (inputs : Fin 16 → Form F) :
    ∀ ks : List (Fin 16), Frame (tieRows b inputs ks)
  | [] => .pure _
  | _ :: rest => Frame.bind (row_frame _ _ _) fun _ => tieRows_frame b inputs rest

theorem permute_frame (inputs : Fin 16 → Form F) : Frame (permute inputs) :=
  Frame.bind (Frame.modifyGet fun _ _ => rfl) fun _ => Frame.bind (tieRows_frame _ _ _) fun _ => .pure _

/-- Every instruction but `pub` lowers without the statement words. -/
theorem step_frame (stmt : Array F) (i : Instr) (notPub : i ≠ .pub) : Frame (step stmt i) := by
  cases i with
  | const _ => exact push_frame _
  | priv => exact Frame.bind cell_frame fun _ => push_frame _
  | pub => exact absurd rfl notPub
  | linear _ =>
    exact Frame.bind (readValues_frame _) fun _ => Frame.bind (linear_frame _) fun _ => push_frame _
  | mul _ _ =>
    exact Frame.bind (readValue_frame _) fun _ => Frame.bind (readValue_frame _) fun _ =>
      Frame.bind (mul_frame _ _) fun _ => push_frame _
  | extMul _ _ =>
    exact Frame.bind (readCoordinates_frame _) fun _ => Frame.bind (readCoordinates_frame _) fun _ =>
      Frame.bind (extMul_frame _ _) fun _ => push_frame _
  | extInv _ =>
    exact Frame.bind (readCoordinates_frame _) fun _ => Frame.bind (extInv_frame _) fun _ => push_frame _
  | assertZero _ => exact Frame.bind (readValue_frame _) fun _ => assertZero_frame _
  | bits _ _ =>
    exact Frame.bind (readValue_frame _) fun _ => Frame.bind (bits_frame _ _) fun _ => push_frame _
  | permute _ =>
    exact Frame.bind (readCoordinates_frame _) fun _ => Frame.bind (permute_frame _) fun _ => push_frame _
  | hint _ => exact Frame.bind (cells_frame _) fun _ => push_frame _
  | r1cs _ _ => exact Frame.bind (readValues_frame _) fun _ => Frame.modify fun _ _ => rfl

/-- The step of a `pub`: the next wire, and the next word in `publics`. -/
theorem step_pub (stmt : Array F) (s : State F) :
    ((step stmt .pub).run s).2 =
      { s with values := s.values.push (Form.wire (Wire.pub (s.publics.size + 1))),
               publics := s.publics.push (stmt.getD s.publics.size 0) } := rfl

/-- Two lowerings on different statement words differ only in the words of
`publics`, not in its size. -/
theorem steps_publics (stmt stmt' : Array F) :
    ∀ (L : List Instr) (s : State F) (q : Array F), q.size = s.publics.size →
      ∃ q' : Array F, q'.size = (steps stmt L s).publics.size ∧
        steps stmt' L { s with publics := q } = { steps stmt L s with publics := q' }
  | [], s, q, size => ⟨q, size, rfl⟩
  | i :: rest, s, q, size => by
      by_cases pub : i = .pub
      · subst pub
        have there : ((step stmt' .pub).run { s with publics := q }).2 =
            { ((step stmt .pub).run s).2 with publics := q.push (stmt'.getD q.size 0) } := by
          rw [step_pub, step_pub, size]
        obtain ⟨q', size', same⟩ := steps_publics stmt stmt' rest ((step stmt .pub).run s).2
          (q.push (stmt'.getD q.size 0)) (by rw [step_pub]; simp [size])
        exact ⟨q', size', by show steps stmt' rest ((step stmt' .pub).run _).2 = _; rw [there, same]; rfl⟩
      · have frame := step_frame stmt i pub
        obtain ⟨q', size', same⟩ := steps_publics stmt stmt' rest ((step stmt i).run s).2 q
          (by rw [frame.keeps, size])
        have same_step : step stmt' i = step stmt i := by cases i <;> first | rfl | exact absurd rfl pub
        exact ⟨q', size', by
          show steps stmt' rest ((step stmt' i).run _).2 = _
          rw [same_step, frame.run, same]
          rfl⟩

/-- **The counts do not depend on the statement words.** -/
theorem lower_counts (L : Array Instr) (stmt : List F) : (lower L stmt).counts = (lower L ([] : List F)).counts := by
  obtain ⟨q, size, same⟩ := steps_publics (#[] : Array F) stmt.toArray L.toList {} #[] rfl
  have same : steps stmt.toArray L.toList {} = { steps #[] L.toList {} with publics := q } := same
  simp only [lower, Run.counts, Run.rows, same, Array.length_toList, size]

end CompressionLean.Circuit
