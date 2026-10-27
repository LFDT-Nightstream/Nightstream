import CompressionLean.Circuit.Record

/-!
Owns `lower_counts`: the counts of a lowering do not depend on the
statement. The statement words enter only the recorder's `publics`
(`statementWord`); every other call of the lowering keeps `publics` and
never reads it (`Indep`). So the lowerings of one program on two statements
differ only in the values of their statement words (`SameShape`).
-/

namespace CompressionLean.Circuit

open CompressionLean.Shrink

section Lowering

variable {F : Type}

/-- `m` never reads or writes the statement words: running it on another
`publics` changes only `publics`. -/
def Indep {α : Type} (m : Lower F α) : Prop :=
  ∀ (s : State F) (ps : Array F), m.run { s with publics := ps } = ((m.run s).1, { (m.run s).2 with publics := ps })

namespace Indep

theorem pure {α : Type} (a : α) : Indep (Pure.pure a : Lower F α) := fun _ _ => rfl

theorem bind {α β : Type} {m : Lower F α} {f : α → Lower F β} (hm : Indep m) (hf : ∀ a, Indep (f a)) :
    Indep (m >>= f) := by
  intro s ps
  rw [StateT.run_bind, StateT.run_bind, hm s ps]
  exact hf _ _ _

theorem cells (k : ℕ) : Indep (cells k : Lower F ℕ) := fun _ _ => rfl
theorem row (p l three) : Indep (row p l three : Lower F Unit) := fun _ _ => rfl
theorem block : Indep (block : Lower F ℕ) := fun _ _ => rfl
theorem fromFn (m : Fin 3 → Lower F (Form F)) (h : ∀ k, Indep (m k)) : Indep (fromFn m) :=
  (h 0).bind fun _ => (h 1).bind fun _ => (h 2).bind fun _ => Indep.pure _

theorem push (forms : List (Form F)) : Indep (push forms) := fun _ _ => rfl
theorem r1cs (rows : List (R1csRow F)) (columns : List (Form F)) : Indep (r1cs rows columns) := fun _ _ => rfl

end Indep

variable [Field F] [DecidableEq F]

namespace Indep

omit [DecidableEq F] in
theorem cell : Indep (cell : Lower F (Form F)) := fun _ _ => rfl

theorem form (terms : List (Wire × F)) (k : F) : Indep (form terms k) := by
  unfold Circuit.form
  split
  · exact Indep.pure _
  · exact fun _ _ => rfl

theorem linear (parts : List (Form F × F)) : Indep (linear parts) := form _ _

theorem mul (a b : Form F) : Indep (mul a b) := by
  unfold Circuit.mul
  split
  · exact linear _
  split
  · exact linear _
  · exact fun _ _ => rfl

omit [DecidableEq F] in
theorem booleanRows : ∀ l : List ℕ, Indep (booleanRows l : Lower F Unit)
  | [] => Indep.pure _
  | _ :: rest => (row _ _ _).bind fun _ => booleanRows rest

theorem chain : ∀ (high : Form F) (l : List ℕ), Indep (chain high l)
  | _, [] => Indep.pure _
  | _, _ :: rest => (mul _ _).bind fun h => chain h rest

theorem tieRows (b : ℕ) (inputs : Fin 16 → Form F) : ∀ l, Indep (tieRows b inputs l)
  | [] => Indep.pure _
  | _ :: rest => (row _ _ _).bind fun _ => tieRows b inputs rest

theorem extMul (a b : Fin 3 → Form F) : Indep (extMul a b) := by
  unfold Circuit.extMul
  split
  · exact fromFn _ fun _ => linear _
  · exact fromFn _ fun k => cell.bind fun _ => (row _ _ _).bind fun _ => Indep.pure _

theorem extInv (a : Fin 3 → Form F) : Indep (extInv a) := by
  unfold Circuit.extInv
  split
  · exact Indep.pure _
  · exact (fromFn _ fun _ => cell).bind fun _ => (row _ _ _).bind fun _ => (row _ _ _).bind fun _ =>
      (row _ _ _).bind fun _ => Indep.pure _

theorem bits (a : Form F) (count : ℕ) : Indep (bits a count) := by
  unfold Circuit.bits
  refine (cells _).bind fun first => (booleanRows _).bind fun _ => (row _ _ _).bind fun _ => ?_
  split
  · exact (chain _ _).bind fun _ => (row _ _ _).bind fun _ => Indep.pure _
  · exact Indep.pure _

theorem assertZero (a : Form F) : Indep (assertZero a) := by
  unfold Circuit.assertZero
  split
  · exact Indep.pure _
  · exact row _ _ _

omit [DecidableEq F] in
theorem readValue (v : Var) : Indep (readValue v : Lower F (Form F)) := fun _ _ => rfl
omit [DecidableEq F] in
theorem readValues (vs : List Var) : Indep (readValues vs : Lower F (List (Form F))) := fun _ _ => rfl
omit [DecidableEq F] in
theorem readCoordinates {n : ℕ} (a : Fin n → Var) : Indep (readCoordinates a : Lower F (Fin n → Form F)) :=
  fun _ _ => rfl


/-- Every instruction but `pub`. -/
theorem step (stmt : Array F) (i : Instr) (h : i ≠ .pub) : Indep (step stmt i) := by
  cases i with
  | const => exact push _
  | priv => exact cell.bind fun _ => push _
  | pub => exact absurd rfl h
  | linear => exact (readValues _).bind fun _ => (linear _).bind fun _ => push _
  | mul => exact (readValue _).bind fun _ => (readValue _).bind fun _ => (mul _ _).bind fun _ => push _
  | extMul =>
      exact (readCoordinates _).bind fun _ => (readCoordinates _).bind fun _ => (extMul _ _).bind fun _ => push _
  | extInv => exact (readCoordinates _).bind fun _ => (extInv _).bind fun _ => push _
  | assertZero => exact (readValue _).bind fun _ => assertZero _
  | bits => exact (readValue _).bind fun _ => (bits _ _).bind fun _ => push _
  | permute =>
      exact (readCoordinates _).bind fun _ =>
        (block.bind fun _ => (tieRows _ _ _).bind fun _ => Indep.pure _).bind fun _ => push _
  | hint => exact (cells _).bind fun _ => push _
  | r1cs => exact (readValues _).bind fun _ => r1cs _ _

end Indep

/-- `s` is `t` but for the values of its statement words. -/
def SameShape (s t : State F) : Prop := { t with publics := s.publics } = s ∧ s.publics.size = t.publics.size

theorem SameShape.step (stmt stmt' : Array F) (i : Instr) {s t : State F} (h : SameShape s t) :
    SameShape ((Circuit.step stmt i).run s).2 ((Circuit.step stmt' i).run t).2 := by
  obtain ⟨e, size⟩ := h
  by_cases pub : i = .pub
  · subst pub
    rw [← e]
    refine ⟨?_, ?_⟩
    · simp [Circuit.step, statementWord, Circuit.push, size]
      rfl
    · show (s.publics.push _).size = (t.publics.push _).size
      simp [size]
  · have same : Circuit.step stmt i = Circuit.step stmt' i := by cases i <;> first | rfl | exact absurd rfl pub
    rw [same, ← e, Indep.step stmt' i pub t s.publics]
    have keep := Indep.step stmt' i pub t t.publics
    rw [show ({ t with publics := t.publics } : State F) = t from rfl] at keep
    refine ⟨rfl, ?_⟩
    have : ((Circuit.step stmt' i).run t).2.publics = t.publics := by
      have := congrArg (fun r => r.2.publics) keep
      exact this
    simp only [this, size]

theorem SameShape.steps (stmt stmt' : Array F) :
    ∀ (L : List Instr) {s t : State F}, SameShape s t → SameShape (steps stmt L s) (steps stmt' L t)
  | [], _, _, h => h
  | i :: L, _, _, h => SameShape.steps stmt stmt' L (h.step stmt stmt' i)

/-- **The counts do not read the statement.** The statement words enter only
the recorder's `publics`, so the lowering on any statement has the counts of
the lowering on none. -/
theorem lower_counts (L : Array Instr) (stmt : List F) : (lower L stmt).counts = (lower (F := F) L []).counts := by
  obtain ⟨e, size⟩ := SameShape.steps stmt.toArray (([] : List F).toArray) L.toList (s := {}) (t := {})
    ⟨rfl, rfl⟩
  unfold lower Run.counts Run.rows
  dsimp only
  rw [← e]
  simp only [Array.length_toList, size]

end Lowering

end CompressionLean.Circuit
