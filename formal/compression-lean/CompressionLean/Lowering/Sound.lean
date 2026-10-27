import CompressionLean.Lowering.Forms
import Mathlib.Data.List.GetD
import CompressionLean.Fields
import CompressionLean.Lib.Array
import CompressionLean.Lib.List

/-!
Owns `lower_sound`: a witness of a lowered run's CCS gives an accepting run
of the executor (`Exec`), on the tape read out of the witness (`tapeOf`).

- The wire values are `z` at each wire's column (`Run.wireValue`). There every
  glue row and every block holds (`Rows.lean`), so every lowering state of
  the run holds (`State.Holds`): each step only adds calls and blocks.
- Tape word `n` is the value of handle `n`'s form under the witness. The
  executor reads it only at the free handles: `priv`, `hint`, `bits` and
  `extInv`.
- The simulation (`Agrees`): at each step, the value of every handle's form
  is the executor's value, and the executor has read as many statement words
  as the lowering. The rows of a step give its check (`Forms.lean`).

The core (`lower_exec`) is generic in the field, the template and the
permutation. `lower_sound` takes Goldilocks, the Poseidon2 template and
Poseidon2.
-/

namespace CompressionLean.Lowering

open CompressionLean.Circuit
open CompressionLean.Lib
open CompressionLean.Shrink
open CompressionLean.Replay (Word)

section Simulation

variable {F : Type} [Field F] [DecidableEq F] {v : Wire → F} {perm : List F → List F}

/-- `t` extends `s`: more calls, glue cells and blocks, more values after
those of `s`, and at least as many statement words. -/
structure _root_.CompressionLean.Circuit.State.Extends (s t : State F) : Prop where
  calls : ∀ c ∈ s.calls, c ∈ t.calls
  blocks : s.blocks ≤ t.blocks
  values : ∃ more : List (Form F), t.values = s.values ++ more.toArray
  publics : s.publics ≤ t.publics

omit [Field F] [DecidableEq F] in
theorem _root_.CompressionLean.Circuit.State.Extends.refl (s : State F) : s.Extends s :=
  ⟨fun _ h => h, le_rfl, ⟨[], by simp⟩, le_rfl⟩

omit [Field F] [DecidableEq F] in
theorem _root_.CompressionLean.Circuit.State.Extends.trans {s t u : State F} (h : s.Extends t) (h' : t.Extends u) : s.Extends u := by
  obtain ⟨m, hm⟩ := h.values
  obtain ⟨m', hm'⟩ := h'.values
  exact ⟨fun c hc => h'.calls c (h.calls c hc), h.blocks.trans h'.blocks, ⟨m ++ m', by rw [hm', hm]; simp⟩,
    h.publics.trans h'.publics⟩

omit [Field F] [DecidableEq F] in
/-- A part that only adds, then a push of `forms`. -/
theorem _root_.CompressionLean.Circuit.State.Adds.extends_push {s t : State F} (h : s.Adds t) (forms : List (Form F)) :
    s.Extends { t with values := t.values ++ forms.toArray } :=
  ⟨h.calls, h.blocks, ⟨forms, by rw [h.values]⟩, h.publics.ge⟩

omit [Field F] [DecidableEq F] in
theorem _root_.CompressionLean.Circuit.State.Adds.extends {s t : State F} (h : s.Adds t) : s.Extends t :=
  ⟨h.calls, h.blocks, ⟨[], by simp [h.values]⟩, h.publics.ge⟩

omit [Field F] [DecidableEq F] in
theorem _root_.CompressionLean.Circuit.State.Extends.values_getD {s t : State F} (h : s.Extends t) {n : ℕ} (hn : n < s.values.size) (d : Form F) :
    t.values.getD n d = s.values.getD n d := by
  obtain ⟨more, hm⟩ := h.values
  rw [hm, array_getD_append, if_pos hn]

omit [Field F] [DecidableEq F] in
theorem _root_.CompressionLean.Circuit.State.Extends.values_size {s t : State F} (h : s.Extends t) : s.values.size ≤ t.values.size := by
  obtain ⟨more, hm⟩ := h.values
  rw [hm]
  simp

omit [DecidableEq F] in
/-- A push changes neither calls nor blocks. -/
theorem _root_.CompressionLean.Circuit.State.Holds.of_push {s : State F} {forms : Array (Form F)}
    (h : ({ s with values := s.values ++ forms } : State F).Holds v perm) : s.Holds v perm := h

omit [DecidableEq F] in
@[simp] theorem go_readValue (a : Var) (s : State F) : go (readValue a) s = (s.values.getD a (.const 0), s) := rfl

omit [DecidableEq F] in
@[simp] theorem go_readValues (vs : List Var) (s : State F) :
    go (readValues vs) s = (vs.map fun v => s.values.getD v (.const 0), s) := rfl

/-- The forms of `n` handles, as `readCoordinates` reads them, so that
`go_readCoordinates` is `rfl`. Rewriting them to `coordinates_apply`'s form
under `extInv` made the kernel check of `step_extends` take about a minute. -/
def coordinates {n : ℕ} (s : State F) (a : Fin n → Var) : Fin n → Form F :=
  fun k => (Array.ofFn fun k => s.values.getD (a k) (.const 0)).getD k (.const 0)

omit [DecidableEq F] in
theorem go_readCoordinates {n : ℕ} (a : Fin n → Var) (s : State F) :
    go (readCoordinates a) s = (coordinates s a, s) := rfl

omit [DecidableEq F] in
@[simp] theorem coordinates_apply {n : ℕ} (s : State F) (a : Fin n → Var) (k : Fin n) :
    coordinates s a k = s.values.getD (a k) (.const 0) := by
  simp [coordinates, Array.getD_eq_getD_getElem?]

omit [Field F] [DecidableEq F] in
@[simp] theorem go_push (forms : List (Form F)) (s : State F) :
    go (push forms) s = ((), { s with values := s.values ++ forms.toArray }) := rfl

omit [DecidableEq F] in
@[simp] theorem go_statementWord (s : State F) :
    go (statementWord : Lower F (Form F)) s = (.wire (.pub (s.publics + 1)), { s with publics := s.publics + 1 }) := rfl

/-- Each step extends its state. -/
theorem step_extends (one : v (.pub 0) = 1) (small : ∀ k : ℕ, k ≤ 32 → (k : F) = 0 → k = 0)
    (i : Instr) (s : State F) : s.Extends (go (step i) s).2 := by
  cases i with
  | const value => exact (State.Adds.refl s).extends_push _
  | priv =>
      simp only [step, go_bind, go_push]
      exact ((cell_sound (v := v) (perm := id) s).1).extends_push _
  | pub =>
      simp only [step, go_bind, go_statementWord, go_push]
      exact ⟨fun _ h => h, le_rfl, ⟨[_], rfl⟩, Nat.le_succ _⟩
  | linear parts =>
      simp only [step, go_bind, go_readValues, go_push]
      exact ((linear_sound (v := v) (perm := id) _ s).1).extends_push _
  | mul a b =>
      simp only [step, go_bind, go_readValue, go_push]
      exact ((mul_sound (v := v) (perm := id) one _ _ s).1).extends_push _
  | extMul a b =>
      simp only [step, go_bind, go_readCoordinates, go_push]
      exact ((extMul_sound (v := v) (perm := id) one _ _ s).1).extends_push _
  | extInv a =>
      simp only [step, go_bind, go_readCoordinates, go_push]
      exact ((extInv_sound (v := v) (perm := id) one _ s).1).extends_push _
  | assertZero a =>
      simp only [step, go_bind, go_readValue]
      exact ((assertZero_sound (v := v) (perm := id) _ s).1).extends
  | bits a count =>
      simp only [step, go_bind, go_readValue, go_push]
      exact ((bits_sound (v := v) (perm := id) one small _ _ s).1).extends_push _
  | permute state =>
      simp only [step, go_bind, go_readCoordinates, go_push]
      exact ((permute_sound (v := v) (perm := id) _ s).1).extends_push _
  | hint count =>
      simp only [step, go_bind, go_push]
      exact ((cells_sound (v := v) (perm := id) count s).1).extends_push _
  | r1cs rows columns =>
      simp only [step, go_bind, go_readValues]
      exact ((r1cs_sound (v := v) (perm := id) _ _ s).1).extends

theorem steps_extends (one : v (.pub 0) = 1) (small : ∀ k : ℕ, k ≤ 32 → (k : F) = 0 → k = 0) :
    ∀ (L : List Instr) (s : State F), s.Extends (steps L s)
  | [], s => State.Extends.refl s
  | i :: rest, s => (step_extends one small i s).trans (steps_extends one small rest _)

variable (v) in
/-- The simulation: every handle's form has the executor's value, the two have
read the same statement words, and every check so far holds. -/
structure Agrees (S : State F) (E : Exec.State F) : Prop where
  length : E.values.length = S.values.size
  value : ∀ n, (S.values.getD n (.const 0)).evaluate v = E.value n
  reads : E.reads = S.publics
  holds : E.holds

omit [DecidableEq F] in
/-- Appending forms and their values keeps the simulation. -/
theorem Agrees.append {S S' : State F} {E : Exec.State F} (agrees : Agrees v S E) {forms : List (Form F)}
    {outputs : List F} {reads : ℕ} {holds : Prop} (values : S'.values = S.values ++ forms.toArray)
    (evaluates : forms.map (·.evaluate v) = outputs) (hr : reads = S'.publics) (hh : holds) :
    Agrees v S' ⟨E.values ++ outputs, reads, holds⟩ := by
  refine ⟨?_, fun n => ?_, hr, hh⟩
  · simp [values, agrees.length, ← evaluates]
  · show _ = (E.values ++ outputs).getD n 0
    rw [values, array_getD_append, list_getD_append, agrees.length]
    by_cases h : n < S.values.size
    · rw [if_pos h, if_pos h]
      exact agrees.value n
    · rw [if_neg h, if_neg h]
      have read := List.getD_map (fun f : Form F => f.evaluate v) (l := forms) (n := n - S.values.size)
        (d := .const 0)
      rw [evaluate_const, zero_mul] at read
      rw [← evaluates, read]

omit [DecidableEq F] in
theorem Agrees.next {S S' : State F} {E : Exec.State F} (agrees : Agrees v S E) {forms : List (Form F)}
    {outputs : List F} {check : Prop} (values : S'.values = S.values ++ forms.toArray)
    (publics : S'.publics = S.publics) (evaluates : forms.map (·.evaluate v) = outputs) (hc : check) :
    Agrees v S' (E.next outputs check) :=
  agrees.append values evaluates (by rw [publics, agrees.reads]) ⟨agrees.holds, hc⟩

omit [DecidableEq F] in
/-- The tape at the new handles is the new forms' values. -/
theorem tape_forms {S S' : State F} {forms : List (Form F)} {tape : ℕ → F}
    (values : S'.values = S.values ++ forms.toArray)
    (free : ∀ n < S'.values.size, tape n = (S'.values.getD n (.const 0)).evaluate v) :
    forms.map (·.evaluate v) = (List.range forms.length).map fun i => tape (S.values.size + i) := by
  apply List.ext_getElem (by simp)
  intro i h1 _
  have hi : i < forms.length := by simpa using h1
  simp only [List.getElem_map, List.getElem_range]
  rw [free (S.values.size + i) (by rw [values]; simp; omega), values, array_getD_append, if_neg (by omega),
    show S.values.size + i - S.values.size = i by omega, List.getD_eq_getElem _ _ hi]

omit [DecidableEq F] in
/-- A decomposition is about its first `count` bits only. -/
theorem Bits.congr {a : F} {count : ℕ} {b b' : ℕ → F} (h : ∀ i < count, b i = b' i) (bits : Bits a count b) :
    Bits a count b' := by
  refine ⟨fun i hi => h i hi ▸ bits.boolean i hi, ?_, fun c64 high i hi => ?_⟩
  · rw [bits.sum]
    congr 1
    apply List.map_congr_left
    intro i hi
    rw [h i (List.mem_range.mp hi)]
  · rw [← h i (by omega)]
    exact bits.canonical c64 (fun j lo hj => by rw [h j (by omega)]; exact high j lo hj) i hi

/-- **One step keeps the simulation.** Its end state's rows and blocks hold,
the tape is its forms' values, and its statement wires are its words. -/
theorem step_agrees (one : v (.pub 0) = 1) (small : ∀ k : ℕ, k ≤ 32 → (k : F) = 0 → k = 0) (stmt : List F)
    (tape : ℕ → F) (i : Instr) {S : State F} {E : Exec.State F} (agrees : Agrees v S E)
    (holds : (go (step i) S).2.Holds v perm)
    (free : ∀ n < (go (step i) S).2.values.size, tape n = ((go (step i) S).2.values.getD n (.const 0)).evaluate v)
    (pubs : ∀ k < (go (step i) S).2.publics, v (.pub (k + 1)) = stmt.getD k 0) :
    Agrees v (go (step i) S).2 (Exec.step perm stmt tape E i) := by
  cases i with
  | const value =>
      exact agrees.next rfl rfl (by simp [one]) trivial
  | priv =>
      simp only [step, go_bind, go_push] at holds free ⊢
      obtain ⟨adds, -⟩ := cell_sound (v := v) (perm := perm) S
      have values :
          ({ (go cell S).2 with values := (go cell S).2.values ++ [(go cell S).1].toArray } : State F).values =
            S.values ++ [(go cell S).1].toArray := by
        rw [adds.values]
      refine agrees.next values (by rw [adds.publics]) ?_ trivial
      rw [tape_forms values free]
      simp [agrees.length]
  | pub =>
      simp only [step, go_bind, go_statementWord, go_push] at holds free pubs ⊢
      have word : v (.pub (S.publics + 1)) = stmt.getD S.publics 0 := pubs S.publics (by simp)
      refine agrees.append rfl ?_ (by simp [agrees.reads]) agrees.holds
      simp [word, agrees.reads]
  | linear parts =>
      simp only [step, go_bind, go_readValues, go_push] at holds free ⊢
      obtain ⟨adds, post⟩ := linear_sound (v := v) (perm := perm)
        (((parts.map (·.1)).map fun v => S.values.getD v (.const 0)).zip (parts.map fun p => Word.cast p.2)) S
      refine agrees.next (by rw [adds.values]) (by rw [adds.publics]) ?_ trivial
      rw [List.map_singleton, post (State.Holds.of_push holds), List.map_map, List.zip_map', List.map_map]
      congr 2
      apply List.map_congr_left
      intro p _
      simp only [Function.comp_apply, agrees.value]
  | mul a b =>
      simp only [step, go_bind, go_readValue, go_push] at holds free ⊢
      obtain ⟨adds, post⟩ := mul_sound (v := v) (perm := perm) one (S.values.getD a (.const 0))
        (S.values.getD b (.const 0)) S
      refine agrees.next (by rw [adds.values]) (by rw [adds.publics]) ?_ trivial
      rw [List.map_singleton, post (State.Holds.of_push holds), agrees.value, agrees.value]
  | extMul a b =>
      simp only [step, go_bind, go_readCoordinates, go_push] at holds free ⊢
      obtain ⟨adds, post⟩ := extMul_sound (v := v) (perm := perm) one (coordinates S a) (coordinates S b) S
      refine agrees.next (by rw [adds.values]) (by rw [adds.publics]) ?_ trivial
      have h := post (State.Holds.of_push holds)
      simp only [List.map_cons, List.map_nil, h, coordinates_apply, agrees.value, List.ofFn_succ, List.ofFn_zero]
      rfl
  | extInv a =>
      simp only [step, go_bind, go_readCoordinates, go_push] at holds free ⊢
      obtain ⟨adds, post⟩ := extInv_sound (v := v) (perm := perm) one (coordinates S a) S
      have values : ({ (go (extInv (coordinates S a)) S).2 with
          values := (go (extInv (coordinates S a)) S).2.values ++
            [(go (extInv (coordinates S a)) S).1 0, (go (extInv (coordinates S a)) S).1 1,
              (go (extInv (coordinates S a)) S).1 2].toArray } : State F).values =
          S.values ++ [(go (extInv (coordinates S a)) S).1 0, (go (extInv (coordinates S a)) S).1 1,
            (go (extInv (coordinates S a)) S).1 2].toArray := by rw [adds.values]
      have tapes := tape_forms values free
      simp only [List.map_cons, List.map_nil, List.length_cons, List.length_nil] at tapes
      simp only [List.range_succ, List.range_zero, List.nil_append, List.cons_append, List.map_cons, List.map_nil,
        List.cons.injEq, and_true, Nat.add_zero] at tapes
      obtain ⟨t0, t1, t2⟩ := tapes
      refine agrees.next values (by rw [adds.publics]) ?_ ?_
      · simp [t0, t1, t2, agrees.length, List.ofFn_succ]
      · have h := post (State.Holds.of_push holds)
        rw [← agrees.length] at t0 t1 t2
        have av : (fun i => (coordinates S a i).evaluate v) = fun i => E.value (a i) := by
          funext i
          rw [coordinates_apply, agrees.value]
        have iv : (fun k : Fin 3 => tape (E.values.length + k)) =
            fun k => ((go (extInv (coordinates S a)) S).1 k).evaluate v := by
          funext k
          fin_cases k
          exacts [t0.symm, t1.symm, t2.symm]
        rw [av] at h
        show extProduct (fun i => E.value (a i)) (fun k : Fin 3 => tape (E.values.length + k)) = ![1, 0, 0]
        rw [iv, h]
  | assertZero a =>
      simp only [step, go_bind, go_readValue] at holds free ⊢
      obtain ⟨adds, post⟩ := assertZero_sound (v := v) (perm := perm) (S.values.getD a (.const 0)) S
      refine agrees.next (forms := []) (by rw [adds.values]; simp) (by rw [adds.publics]) rfl ?_
      rw [← agrees.value a]
      exact post holds
  | bits a count =>
      simp only [step, go_bind, go_readValue, go_push] at holds free ⊢
      obtain ⟨adds, post⟩ := bits_sound (v := v) (perm := perm) one small (S.values.getD a (.const 0)) count S
      obtain ⟨first, forms, decomposition⟩ := post (State.Holds.of_push holds)
      have values : ({ (go (bits (S.values.getD a (.const 0)) count) S).2 with
          values := (go (bits (S.values.getD a (.const 0)) count) S).2.values ++
            (go (bits (S.values.getD a (.const 0)) count) S).1.toArray } : State F).values =
          S.values ++ (go (bits (S.values.getD a (.const 0)) count) S).1.toArray := by rw [adds.values]
      have tapes := tape_forms values free
      rw [forms, List.map_map, List.length_map, List.length_range'] at tapes
      have bit : ∀ i < count, tape (E.values.length + i) = v (.glue (first + i)) := by
        intro i hi
        have := congrArg (fun l => l.getD i 0) tapes
        simp only [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range', List.getElem?_range,
          hi, Option.map_some, Option.getD_some, Function.comp_apply, evaluate_wire, one_mul] at this
        rw [agrees.length, ← this]
      refine agrees.next values (by rw [adds.publics]) ?_ ?_
      · rw [forms, List.map_map]
        apply List.ext_getElem (by simp)
        intro i h1 _
        simp only [List.length_map, List.length_range'] at h1
        simp [bit i h1]
      · rw [← agrees.value a]
        exact decomposition.congr fun i hi => (bit i hi).symm
  | permute state =>
      simp only [step, go_bind, go_readCoordinates, go_push] at holds free ⊢
      obtain ⟨adds, post⟩ := permute_sound (v := v) (perm := perm) (coordinates S state) S
      have h := post (State.Holds.of_push holds)
      simp only [coordinates_apply, agrees.value] at h
      show Agrees v _ (E.next (List.ofFn fun k : Fin 16 => (perm (List.ofFn fun k => E.value (state k))).getD k 0) True)
      refine agrees.next (forms := List.ofFn (go (permute (coordinates S state)) S).1) (by rw [adds.values])
        (by rw [adds.publics]) ?_ trivial
      rw [List.map_ofFn, ← h]
      congr 1
      funext k
      rw [List.getD_eq_getElem?_getD, List.getElem?_ofFn]
      simp
  | hint count =>
      simp only [step, go_bind, go_push] at holds free ⊢
      obtain ⟨adds, -⟩ := cells_sound (v := v) (perm := perm) count S
      have values : ({ (go (cells count) S).2 with values := (go (cells count) S).2.values ++
            ((List.range' (go (cells count) S).1 count).map fun c => (Form.wire (.glue c) : Form F)).toArray } :
          State F).values =
          S.values ++
            ((List.range' (go (cells count) S).1 count).map fun c => (Form.wire (.glue c) : Form F)).toArray := by
        rw [adds.values]
      refine agrees.next values (by rw [adds.publics]) ?_ trivial
      rw [tape_forms values free, agrees.length]
      simp
  | r1cs rows columns =>
      simp only [step, go_bind, go_readValues] at holds free ⊢
      obtain ⟨adds, post⟩ := r1cs_sound (v := v) (perm := perm) (rows.map R1csRow.inField)
        (columns.map fun c => S.values.getD c (.const 0)) S
      refine agrees.next (forms := []) (by rw [adds.values]; simp) (by rw [adds.publics]) rfl ?_
      intro r hr
      have row := ofCall_holds v (by simp) (post holds (R1csRow.inField r) (List.mem_map.mpr ⟨r, hr, rfl⟩))
      have column : ∀ j, ((columns.map fun c => S.values.getD c (.const 0)).getD j ⟨[], 0⟩).evaluate v =
          (columns.map E.value).getD j 0 := by
        intro j
        simp only [List.getD_eq_getElem?_getD, List.getElem?_map]
        cases columns[j]? with
        | none => simp [Form.evaluate]
        | some c => simp only [Option.map_some, Option.getD_some]; exact agrees.value c
      simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, termsValue_expand,
        R1csRow.inField, List.map_map, Function.comp_def, column] at row
      unfold R1csRow.Holds
      exact row

/-- **The simulation of a program.** From a state that agrees with the
executor, the lowering of `L` agrees with the executor's run of `L`, when the
end state's rows and blocks hold, the tape is the end state's forms' values,
and the statement wires are its words. -/
theorem steps_agrees (one : v (.pub 0) = 1) (small : ∀ k : ℕ, k ≤ 32 → (k : F) = 0 → k = 0) (stmt : List F)
    (tape : ℕ → F) :
    ∀ (L : List Instr) (S : State F) (E : Exec.State F), Agrees v S E →
      (steps L S).Holds v perm →
      (∀ n < (steps L S).values.size, tape n = ((steps L S).values.getD n (.const 0)).evaluate v) →
      (∀ k < (steps L S).publics, v (.pub (k + 1)) = stmt.getD k 0) →
      Agrees v (steps L S) (L.foldl (Exec.step perm stmt tape) E)
  | [], _, _, agrees, _, _, _ => agrees
  | i :: rest, S, E, agrees, holds, free, pubs => by
      have ext := steps_extends one small rest (go (step i) S).2
      have next := step_agrees one small stmt tape i agrees (holds.mono ext.calls ext.blocks)
        (fun n hn => (free n (lt_of_lt_of_le hn ext.values_size)).trans
          (congrArg (Form.evaluate v) (ext.values_getD hn _)))
        (fun k hk => pubs k (lt_of_lt_of_le hk ext.publics))
      exact steps_agrees one small stmt tape rest _ _ next holds free pubs

end Simulation

section Lowering

variable {F : Type} [Field F] [DecidableEq F]

/-- **The tape of a witness.** Tape word `n` is the value of handle `n`'s form
at the wire values of `w`. The executor reads it only at the free handles,
whose forms are glue cells or, for a folded `extInv`, constants. -/
noncomputable def tapeOf (T : Template F) (L : Array Instr) (stmt : List F) (w : ℕ → F) (n : ℕ) : F :=
  ((steps L.toList {}).values.getD n (.const 0)).evaluate
    ((lower L stmt).wireValue T (lower L stmt).counts w)

/-- **Lowering soundness**, for any field, template and permutation: if the
template's rows force its output cells to be `perm` of its input cells, and no
count up to 32 is zero in `F`, then a witness of the lowered run's CCS gives an
accepting run of the executor on the witness's tape. -/
theorem lower_exec (T : Template F) (wf : T.WellFormed) (perm : List F → List F)
    (permutes : ∀ c : ℕ → F, T.Holds c →
      List.ofFn (fun k : Fin 16 => c (outputCell + k)) = perm (List.ofFn fun k : Fin 16 => c k))
    (small : ∀ k : ℕ, k ≤ 32 → (k : F) = 0 → k = 0) (L : Array Instr) (stmt : List F) (w : ℕ → F)
    (holds : ((lower L stmt).ccs T (lower L stmt).counts).Relation w) :
    Exec perm L stmt (tapeOf T L stmt w) := by
  have rows := Run.rows_hold wf rfl holds
  have blocks := Run.blocks_hold wf rfl holds
  have final : (steps L.toList {}).Holds ((lower L stmt).wireValue T (lower L stmt).counts w) perm :=
    ⟨fun c hc row hrow => rows row (List.mem_flatMap.mpr ⟨c, Array.mem_toList_iff.mpr hc, hrow⟩),
      fun b hb => permutes _ (blocks b hb)⟩
  have start : Agrees ((lower L stmt).wireValue T (lower L stmt).counts w) {} (Exec.start : Exec.State F) :=
    ⟨rfl, fun n => by simp [Exec.State.value, Exec.start, Array.getD_eq_getD_getElem?], rfl, trivial⟩
  have pubs : ∀ k < (steps (F := F) L.toList {}).publics,
      (lower L stmt).wireValue T (lower L stmt).counts w (.pub (k + 1)) = stmt.getD k 0 := by
    intro k hk
    rw [Run.wireValue_pub]
    simp [lower, List.getD_eq_getElem?_getD, hk]
  exact (steps_agrees (Run.wireValue_one _ _ _ _) small stmt (tapeOf T L stmt w) L.toList {} Exec.start start
    final (fun _ _ => rfl) pubs).holds

end Lowering

namespace Goldilocks

open NightstreamFPrime.Spec

/-- No count up to 32 is zero in Goldilocks. -/
theorem small (k : ℕ) (hk : k ≤ 32) (zero : (k : F) = 0) : k = 0 := by
  have : goldilocksModulus ∣ k := (ZMod.natCast_eq_zero_iff k goldilocksModulus).mp zero
  exact Nat.eq_zero_of_dvd_of_lt this (by unfold goldilocksModulus; omega)

end Goldilocks

open NightstreamFPrime in
/-- **Lowering soundness.** If `w` satisfies the CCS of the lowered run of
program `L` on the statement words `stmt`, with the Poseidon2 block template,
then the executor accepts `L` on `stmt` and the tape read out of `w`, with
Poseidon2 blocks. -/
theorem lower_sound (L : Array Instr) (stmt : List Spec.F) (w : ℕ → Spec.F)
    (holds : ((lower L stmt).ccs Block.poseidonTemplate (lower L stmt).counts).Relation w) :
    Exec Spec.Poseidon2.permute L stmt (tapeOf Block.poseidonTemplate L stmt w) :=
  lower_exec Block.poseidonTemplate Block.poseidonTemplate_wellFormed Spec.Poseidon2.permute
    (fun c h => Block.block_sound c h) Goldilocks.small L stmt w holds

end CompressionLean.Lowering
