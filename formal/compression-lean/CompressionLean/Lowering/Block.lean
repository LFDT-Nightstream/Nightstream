import CompressionLean.Ports.Circuit.Block
import CompressionLean.Shrink.Layout
import NightstreamFPrime.Gadgets.Poseidon2.Permutation
import CompressionLean.Lib.List

/-!
Owns the facts about the Poseidon2 block template (`Ports/Circuit/Block.lean`)
that the lowering reads.

- `Template.Holds`: every row of a template holds on the cells of one block
  (`Kind::failure`, `block.rs::Kind::failure`).
- `poseidonTemplate_wellFormed`: every entry of the template names a row and
  a cell of the block.
- `block_sound`: if every row holds, the output cells `166..182` are
  `Spec.Poseidon2.permute` of the input cells `0..16`. These are the cells
  that `Recorder::block` ties to its inputs and returns
  (`record.rs::Recorder::block`, `record.rs::Recorder::permute`).

The proof evaluates each step's forms to F′'s `Permutation.applyF`.
`runF_eq_reference` and `runReference_schedule` give the permutation.
-/

namespace CompressionLean.Lowering.Block

open CompressionLean.Lib
open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Gadgets.Poseidon2
open CompressionLean.Shrink CompressionLean.Circuit.Block
open Fin.CommRing

section Template

variable {R : Type} [CommRing R]


/-- An entry's term on the cells `c` of one block; `none` is the constant
one. -/
def _root_.CompressionLean.Shrink.TemplateEntry.value (c : ℕ → R) (e : TemplateEntry R) : R :=
  e.coef * (e.cell.map c).getD 1

/-- `(M_j·z)(r)` on the cells `c` of one block (`block.rs::Kind::failure`). -/
def _root_.CompressionLean.Shrink.Template.rowValue (T : Template R) (c : ℕ → R) (j : Fin 8) (r : ℕ) : R :=
  (T.entries.map fun e => if e.matrix = j ∧ e.row = r then e.value c else 0).sum

/-- Every row of `T` holds on the cells `c` of one block (`Kind::failure`,
`block.rs::Kind::failure`). -/
def _root_.CompressionLean.Shrink.Template.Holds (T : Template R) (c : ℕ → R) : Prop :=
  ∀ r < T.rows, constraint (fun j => T.rowValue c j r) = 0

end Template

/-! ## Forms -/

namespace Linear

open CompressionLean.Circuit.Block.Linear

/-- `Σ coefficient·c(cell)` over a term list. -/
private def termsValue (c : ℕ → F) (ts : List (ℕ × F)) : F := (ts.map fun t => t.2 * c t.1).sum

private theorem eval_eq (c : ℕ → F) (l : Linear) : l.eval c = l.constant + termsValue c l.terms := rfl

private theorem termsValue_cons (c : ℕ → F) (t : ℕ × F) (ts : List (ℕ × F)) :
    termsValue c (t :: ts) = t.2 * c t.1 + termsValue c ts := by
  simp [termsValue]

private theorem termsValue_addTerm (c : ℕ → F) :
    ∀ (ts : List (ℕ × F)) (t : ℕ × F), termsValue c (addTerm ts t) = termsValue c ts + t.2 * c t.1
  | [], t => by simp [addTerm, termsValue]
  | u :: rest, t => by
      by_cases same : u.1 = t.1
      · rw [addTerm, if_pos same, termsValue_cons, termsValue_cons]
        simp only [same]
        ring
      · rw [addTerm, if_neg same, termsValue_cons, termsValue_cons, termsValue_addTerm c rest t]
        ring

private theorem termsValue_foldl (c : ℕ → F) (w : F) :
    ∀ ts acc : List (ℕ × F),
      termsValue c (ts.foldl (fun terms t => addTerm terms (t.1, t.2 * w)) acc) =
        termsValue c acc + termsValue c ts * w
  | [], acc => by simp [termsValue]
  | t :: ts, acc => by
      rw [List.foldl_cons, termsValue_foldl c w ts, termsValue_addTerm, termsValue_cons]
      ring

private theorem mergePart_foldl (c : ℕ → F) :
    ∀ (parts : List (Linear × F)) (acc : List (ℕ × F) × F),
      (parts.foldl mergePart acc).2 + termsValue c (parts.foldl mergePart acc).1 =
        acc.2 + termsValue c acc.1 + (parts.map fun p => p.1.eval c * p.2).sum
  | [], acc => by simp
  | p :: parts, acc => by
      rw [List.foldl_cons, mergePart_foldl c parts, List.map_cons, List.sum_cons]
      simp only [mergePart, termsValue_foldl, eval_eq]
      ring

private theorem termsValue_filter (c : ℕ → F) :
    ∀ ts : List (ℕ × F), termsValue c (ts.filter fun t => t.2 ≠ 0) = termsValue c ts
  | [] => rfl
  | t :: ts => by
      by_cases zero : t.2 = 0
      · rw [List.filter_cons_of_neg (by simp [zero]), termsValue_filter c ts, termsValue_cons, zero,
          zero_mul, zero_add]
      · rw [List.filter_cons_of_pos (by simp [zero]), termsValue_cons, termsValue_filter c ts,
          termsValue_cons]

private theorem termsValue_mergeSort (c : ℕ → F) (ts : List (ℕ × F)) (le : ℕ × F → ℕ × F → Bool) :
    termsValue c (ts.mergeSort le) = termsValue c ts :=
  ((List.mergeSort_perm ts le).map _).sum_eq

private theorem eval_combine (c : ℕ → F) (parts : List (Linear × F)) :
    (combine parts).eval c = (parts.map fun p => p.1.eval c * p.2).sum := by
  have total := mergePart_foldl c parts ([], 0)
  dsimp only [combine]
  rw [eval_eq]
  dsimp only
  rw [termsValue_mergeSort, termsValue_filter, total]
  simp [termsValue]

@[simp] private theorem eval_cell (c : ℕ → F) (k : ℕ) : (cell k).eval c = c k := by
  simp [eval, cell]

@[simp] private theorem eval_zero (c : ℕ → F) : zero.eval c = 0 := by
  simp [eval, zero]

private theorem eval_shift (c : ℕ → F) (l : Linear) (a : F) :
    (⟨l.terms, l.constant + a⟩ : Linear).eval c = l.eval c + a := by
  simp only [eval]
  ring

end Linear

/-! ## Each step evaluates to F′'s step -/

/-- The lane values of a state on the cells `c`. -/
private def evalState (c : ℕ → F) (state : List Linear) : Layer.FState :=
  fun lane => (state.getD lane .zero).eval c

/-- S-box output cell `16 + k` holds the S-box of S-box input `k`. -/
private def SboxRows (c : ℕ → F) (inputs : List Linear) : Prop :=
  ∀ k < inputs.length, c (16 + k) = Spec.Poseidon2.sbox ((inputs.getD k .zero).eval c)

private theorem foldl_add (l : List F) (a : F) : l.foldl (· + ·) a = a + l.sum := by
  induction l generalizing a with
  | nil => simp
  | cons x l ih => rw [List.foldl_cons, ih, List.sum_cons]; ring

private theorem range_four : List.range 4 = [0, 1, 2, 3] := rfl

private theorem eval_mixLinear (c : ℕ → F) (state : List Linear) {k : ℕ} (hk : k < 16) :
    ((mixLinear state).getD k .zero).eval c = Layer.blockF (evalState c state) k := by
  have get : ∀ j < 4, Layer.getF (evalState c state) (4 * (k / 4) + j) =
      (state.getD (4 * (k / 4) + j) .zero).eval c := by
    intro j hj
    simp [Layer.getF, evalState, show 4 * (k / 4) + j < 16 by omega]
  have get0 := get 0 (by omega)
  have get1 := get 1 (by omega)
  have get2 := get 2 (by omega)
  have get3 := get 3 (by omega)
  simp only [Nat.add_zero] at get0
  rw [mixLinear, getD_range_map _ _ hk, Linear.eval_combine, List.map_map, range_four]
  simp only [Layer.blockF, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    Function.comp_apply, Nat.add_zero]
  have lane : k % 4 = 0 ∨ k % 4 = 1 ∨ k % 4 = 2 ∨ k % 4 = 3 := by omega
  rcases lane with h | h | h | h <;>
    simp only [h, m4, Layer.mat4F, get0, get1, get2, get3, List.getD_cons_zero,
      List.getD_cons_succ] <;> ring

private theorem evalState_externalLinear (c : ℕ → F) (state : List Linear) :
    evalState c (externalLinear state) = Layer.externalF (evalState c state) := by
  funext lane
  have mixed : ∀ k < 16, ((mixLinear state).getD k .zero).eval c =
      Layer.blockF (evalState c state) k := fun k hk => eval_mixLinear c state hk
  have chunks : ((List.range 4).map fun chunk =>
      ((mixLinear state).getD (4 * chunk + lane.val % 4) .zero).eval c * 1) =
        (List.range 4).map fun chunk => Layer.blockF (evalState c state) (4 * chunk + lane.val % 4) := by
    apply List.map_congr_left
    intro chunk member
    rw [List.mem_range] at member
    rw [mixed _ (by omega), mul_one]
  simp only [evalState, externalLinear]
  rw [getD_range_map _ _ lane.isLt, Linear.eval_combine, List.map_cons, List.sum_cons, List.map_map]
  simp only [Function.comp_def]
  rw [chunks, mixed _ lane.isLt, mul_one]
  simp only [Layer.externalF, Layer.columnF, foldl_add, zero_add]

private theorem evalState_cells (c : ℕ → F) (first : ℕ) :
    evalState c ((List.range 16).map fun lane => Linear.cell (first + lane)) =
      fun lane => c (first + lane.val) := by
  funext lane
  simp only [evalState]
  rw [getD_range_map _ _ lane.isLt, Linear.eval_cell]

private theorem fullRound_eval (c : ℕ → F) (rows : List (List ℕ)) (round : ℕ) (b : Build)
    (sbox : SboxRows c (fullRound rows round b).inputs) :
    evalState c (fullRound rows round b).state = Layer.fullF rows round (evalState c b.state) := by
  simp only [fullRound]
  rw [evalState_externalLinear, evalState_cells, Layer.fullF]
  congr 1
  funext lane
  have inside : b.inputs.length + lane.val < (fullRound rows round b).inputs.length := by
    simp [fullRound]
  have row := sbox _ inside
  simp only [fullRound, getD_append_length, getD_range_map _ _ lane.isLt, Linear.eval_shift] at row
  rw [show 16 + b.inputs.length + lane.val = 16 + (b.inputs.length + lane.val) by omega, row]
  rfl

/-- The lanes of `Layer.partialF` after its S-box. -/
private def sboxLanes (round : ℕ) (state : Layer.FState) : Layer.FState := fun lane =>
  if lane.val = 0 then
    Layer.sboxF (state lane + Spec.Poseidon2.ofNat (Spec.Poseidon2.internalConstants.getD round 0))
  else state lane

private theorem partialRound_eval (c : ℕ → F) (round : ℕ) (b : Build)
    (sbox : SboxRows c (partialRound round b).inputs) :
    evalState c (partialRound round b).state = Layer.partialF round (evalState c b.state) := by
  have first : c (16 + b.inputs.length) = Layer.sboxF (evalState c b.state 0 +
      Spec.Poseidon2.ofNat (Spec.Poseidon2.internalConstants.getD round 0)) := by
    have row := sbox b.inputs.length (by simp [partialRound])
    simpa [partialRound, getD_append_length, Linear.eval_shift, evalState, Layer.sboxF] using row
  have lanes : ∀ k < 16, ((partialLanes b).getD k .zero).eval c =
      Layer.getF (sboxLanes round (evalState c b.state)) k := by
    intro k hk
    rw [partialLanes, getD_range_map _ _ hk]
    simp only [Layer.getF, hk, dif_pos, sboxLanes]
    by_cases zero : k = 0
    · subst zero
      simpa using first
    · simp [zero, evalState]
  have sum : (Linear.combine ((partialLanes b).map fun form => (form, 1))).eval c =
      Layer.sumF (sboxLanes round (evalState c b.state)) := by
    rw [Linear.eval_combine, List.map_map, Layer.sumF, foldl_add, zero_add]
    have each : (partialLanes b).map ((fun p : Linear × F => p.1.eval c * p.2) ∘ fun form => (form, 1)) =
        (List.range 16).map (Layer.getF (sboxLanes round (evalState c b.state))) := by
      rw [partialLanes, List.map_map]
      apply List.map_congr_left
      intro k member
      rw [List.mem_range] at member
      have := lanes k member
      rw [partialLanes, getD_range_map _ _ member] at this
      simp only [Function.comp_apply, mul_one, this]
    rw [each]
  funext lane
  simp only [partialRound, evalState]
  rw [getD_range_map _ _ lane.isLt, Linear.eval_combine]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
  rw [sum, lanes _ lane.isLt]
  show _ = Layer.internalF (sboxLanes round (evalState c b.state)) lane
  simp only [Layer.internalF, Layer.getF, lane.isLt, dif_pos]
  ring

private theorem step_eval (c : ℕ → F) (b : Build) (s : Permutation.Step) (sbox : SboxRows c (step b s).inputs) :
    evalState c (step b s).state = Permutation.applyF s (evalState c b.state) := by
  cases s with
  | initialLayer => exact evalState_externalLinear c b.state
  | initialFullRound round => exact fullRound_eval c _ round b sbox
  | partialRound round => exact partialRound_eval c round b sbox
  | terminalFullRound round => exact fullRound_eval c _ round b sbox

private theorem step_inputs (b : Build) (s : Permutation.Step) : ∃ more, (step b s).inputs = b.inputs ++ more := by
  cases s with
  | initialLayer => exact ⟨[], (List.append_nil _).symm⟩
  | initialFullRound => exact ⟨_, rfl⟩
  | partialRound => exact ⟨_, rfl⟩
  | terminalFullRound => exact ⟨_, rfl⟩

private theorem foldl_inputs :
    ∀ (steps : List Permutation.Step) (b : Build), ∃ more, (steps.foldl step b).inputs = b.inputs ++ more
  | [], b => ⟨[], (List.append_nil _).symm⟩
  | s :: rest, b => by
      obtain ⟨first, here⟩ := step_inputs b s
      obtain ⟨later, there⟩ := foldl_inputs rest (step b s)
      exact ⟨first ++ later, by rw [List.foldl_cons, there, here, List.append_assoc]⟩

private theorem SboxRows.prefix {c : ℕ → F} {l more : List Linear} (rows : SboxRows c (l ++ more)) :
    SboxRows c l := by
  intro k hk
  have row := rows k (by simp; omega)
  rwa [List.getD_append _ _ _ _ hk] at row

private theorem run_eval (c : ℕ → F) :
    ∀ (steps : List Permutation.Step) (b : Build), SboxRows c (steps.foldl step b).inputs →
      evalState c (steps.foldl step b).state = Permutation.runF steps (evalState c b.state)
  | [], _, _ => rfl
  | s :: rest, b, rows => by
      rw [List.foldl_cons] at rows ⊢
      obtain ⟨more, grows⟩ := foldl_inputs rest (step b s)
      have here : SboxRows c (step b s).inputs := by
        rw [grows] at rows
        exact rows.prefix
      rw [run_eval c rest (step b s) rows, step_eval c b s here]
      rfl

/-! ## Sizes and cell bounds -/

/-- The S-boxes of one step. -/
private def sboxCount : Permutation.Step → ℕ
  | .initialLayer => 0
  | .initialFullRound _ => 16
  | .partialRound _ => 1
  | .terminalFullRound _ => 16

private theorem foldl_inputs_length :
    ∀ (steps : List Permutation.Step) (b : Build),
      (steps.foldl step b).inputs.length = b.inputs.length + (steps.map sboxCount).sum
  | [], b => by simp
  | s :: rest, b => by
      rw [List.foldl_cons, foldl_inputs_length rest, List.map_cons, List.sum_cons]
      cases s <;> simp [step, fullRound, partialRound, sboxCount] <;> omega

private theorem build_inputs_length : build.inputs.length = sboxes := by
  rw [build, foldl_inputs_length]
  rfl

private theorem foldl_state_length :
    ∀ (steps : List Permutation.Step) (b : Build), b.state.length = 16 →
      (steps.foldl step b).state.length = 16
  | [], _, h => h
  | s :: rest, b, _ => by
      rw [List.foldl_cons]
      apply foldl_state_length rest
      cases s <;> simp [step, fullRound, partialRound, externalLinear]

private theorem build_state_length : build.state.length = 16 :=
  foldl_state_length _ _ (by simp [start])

namespace Linear

open CompressionLean.Circuit.Block.Linear

/-- Every term of the form names a cell below `n`. -/
private def _root_.CompressionLean.Circuit.Block.Linear.Below (n : ℕ) (l : Linear) : Prop := ∀ t ∈ l.terms, t.1 < n

private theorem _root_.CompressionLean.Circuit.Block.Linear.Below.mono {n m : ℕ} {l : Linear} (h : l.Below n) (le : n ≤ m) : l.Below m :=
  fun t member => (h t member).trans_le le

private theorem below_addTerm {n : ℕ} :
    ∀ (ts : List (ℕ × F)) (t : ℕ × F), (∀ u ∈ ts, u.1 < n) → t.1 < n → ∀ u ∈ addTerm ts t, u.1 < n
  | [], t, _, ht, u, member => by
      simp only [addTerm, List.mem_singleton] at member
      exact member ▸ ht
  | v :: rest, t, hts, ht, u, member => by
      by_cases same : v.1 = t.1
      · rw [addTerm, if_pos same, List.mem_cons] at member
        rcases member with rfl | member
        · exact hts v List.mem_cons_self
        · exact hts u (List.mem_cons_of_mem _ member)
      · rw [addTerm, if_neg same, List.mem_cons] at member
        rcases member with rfl | member
        · exact hts u List.mem_cons_self
        · exact below_addTerm rest t (fun w hw => hts w (List.mem_cons_of_mem _ hw)) ht u member

private theorem below_foldl {n : ℕ} (w : F) :
    ∀ ts acc : List (ℕ × F), (∀ u ∈ acc, u.1 < n) → (∀ u ∈ ts, u.1 < n) →
      ∀ u ∈ ts.foldl (fun terms t => addTerm terms (t.1, t.2 * w)) acc, u.1 < n
  | [], _, hacc, _ => hacc
  | t :: ts, acc, hacc, hts => by
      rw [List.foldl_cons]
      exact below_foldl w ts _ (below_addTerm acc _ hacc (hts t List.mem_cons_self))
        fun u member => hts u (List.mem_cons_of_mem _ member)

private theorem below_mergePart {n : ℕ} :
    ∀ (parts : List (Linear × F)) (acc : List (ℕ × F) × F), (∀ u ∈ acc.1, u.1 < n) →
      (∀ p ∈ parts, p.1.Below n) → ∀ u ∈ (parts.foldl mergePart acc).1, u.1 < n
  | [], _, hacc, _ => hacc
  | p :: parts, acc, hacc, hparts => by
      rw [List.foldl_cons]
      exact below_mergePart parts _ (below_foldl p.2 p.1.terms acc.1 hacc (hparts p List.mem_cons_self))
        fun q member => hparts q (List.mem_cons_of_mem _ member)

private theorem below_combine {n : ℕ} {parts : List (Linear × F)} (h : ∀ p ∈ parts, p.1.Below n) :
    (combine parts).Below n := by
  intro t member
  dsimp only [combine] at member
  rw [List.mem_mergeSort, List.mem_filter] at member
  exact below_mergePart parts ([], 0) (by simp) h t member.1

private theorem below_cell {n k : ℕ} (h : k < n) : (cell k).Below n := by
  simp [Below, cell, h]

private theorem below_of_getD {n : ℕ} {L : List Linear} (h : ∀ k, (L.getD k .zero).Below n) :
    ∀ l ∈ L, l.Below n := by
  intro l member
  obtain ⟨k, hk, rfl⟩ := List.mem_iff_getElem.mp member
  have := h k
  rwa [List.getD_eq_getElem _ _ hk] at this

private theorem below_range_map {n : ℕ} {f : ℕ → Linear} (h : ∀ k < 16, (f k).Below n) :
    ∀ k, (((List.range 16).map f).getD k .zero).Below n := by
  intro k
  by_cases hk : k < 16
  · rw [getD_range_map _ _ hk]
    exact h k hk
  · rw [List.getD_eq_default _ _ (by simp; omega)]
    simp [Below, zero]

end Linear

/-- Every lane form and every S-box input names a cell below
`16 + inputs`. -/
private def _root_.CompressionLean.Circuit.Block.Build.Below (b : Build) : Prop :=
  (∀ k, (b.state.getD k .zero).Below (16 + b.inputs.length)) ∧
    ∀ l ∈ b.inputs, l.Below (16 + b.inputs.length)

private theorem below_externalLinear {n : ℕ} {state : List Linear} (h : ∀ k, (state.getD k .zero).Below n) :
    ∀ k, ((externalLinear state).getD k .zero).Below n := by
  have mixed : ∀ k, ((mixLinear state).getD k .zero).Below n := by
    apply Linear.below_range_map
    intro k _
    apply Linear.below_combine
    intro p member
    simp only [List.mem_map, List.mem_range] at member
    obtain ⟨j, _, rfl⟩ := member
    exact h _
  simp only [externalLinear]
  apply Linear.below_range_map
  intro lane _
  apply Linear.below_combine
  intro p member
  simp only [List.mem_cons, List.mem_map, List.mem_range] at member
  rcases member with rfl | ⟨chunk, _, rfl⟩
  · exact mixed _
  · exact mixed _

private theorem fullRound_below (rows : List (List ℕ)) (round : ℕ) (b : Build) (h : b.Below) :
    (fullRound rows round b).Below := by
  obtain ⟨hstate, hinputs⟩ := h
  have length : (fullRound rows round b).inputs.length = b.inputs.length + 16 := by
    simp [fullRound]
  rw [Build.Below, length]
  refine ⟨?_, ?_⟩
  · apply below_externalLinear
    exact Linear.below_range_map fun lane hlane => Linear.below_cell (by omega)
  · intro l member
    simp only [fullRound, List.mem_append, List.mem_map, List.mem_range] at member
    rcases member with member | ⟨lane, _, rfl⟩
    · exact (hinputs l member).mono (by omega)
    · exact (hstate lane).mono (by omega)

private theorem partialRound_below (round : ℕ) (b : Build) (h : b.Below) : (partialRound round b).Below := by
  obtain ⟨hstate, hinputs⟩ := h
  have length : (partialRound round b).inputs.length = b.inputs.length + 1 := by
    simp [partialRound]
  have lanes : ∀ k, ((partialLanes b).getD k .zero).Below (16 + (b.inputs.length + 1)) := by
    apply Linear.below_range_map
    intro k _
    split_ifs
    · exact Linear.below_cell (by omega)
    · exact (hstate k).mono (by omega)
  rw [Build.Below, length]
  refine ⟨?_, ?_⟩
  · simp only [partialRound]
    apply Linear.below_range_map
    intro lane _
    apply Linear.below_combine
    intro p member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · apply Linear.below_combine
      intro q member
      simp only [List.mem_map] at member
      obtain ⟨form, member, rfl⟩ := member
      exact Linear.below_of_getD lanes form member
    · exact lanes _
  · intro l member
    simp only [partialRound, List.mem_append, List.mem_singleton] at member
    rcases member with member | rfl
    · exact (hinputs l member).mono (by omega)
    · exact (hstate 0).mono (by omega)

private theorem foldl_below : ∀ (steps : List Permutation.Step) (b : Build), b.Below → (steps.foldl step b).Below
  | [], _, h => h
  | s :: rest, b, h => by
      rw [List.foldl_cons]
      apply foldl_below rest
      cases s with
      | initialLayer => exact ⟨below_externalLinear h.1, h.2⟩
      | initialFullRound round => exact fullRound_below _ round b h
      | partialRound round => exact partialRound_below round b h
      | terminalFullRound round => exact fullRound_below _ round b h

private theorem build_below : build.Below :=
  foldl_below _ _ ⟨Linear.below_range_map fun k hk => Linear.below_cell (by simp [start]; omega),
    by simp [start]⟩

/-! ## Well-formedness -/

private theorem mem_push {m : Fin 8} {row : ℕ} {form : Linear} {sign : F} {e : TemplateEntry F}
    (member : e ∈ push m row form sign) :
    e.matrix = m ∧ e.row = row ∧ ∀ k, e.cell = some k → ∃ t ∈ form.terms, t.1 = k := by
  simp only [push, List.mem_append, List.mem_map] at member
  rcases member with ⟨t, ht, rfl⟩ | member
  · exact ⟨rfl, rfl, fun k hk => ⟨t, ht, by simpa using hk⟩⟩
  · split_ifs at member
    · rw [List.mem_singleton] at member
      subst member
      exact ⟨rfl, rfl, fun k hk => by simp at hk⟩
    · simp at member

private theorem rowCell_eq (row : ℕ) : rowCell row = 16 + row := by
  unfold rowCell output
  split_ifs <;> omega

/-- Every entry names a template row and a template cell. -/
theorem poseidonTemplate_wellFormed : poseidonTemplate.WellFormed := by
  obtain ⟨hstate, hinputs⟩ := build_below
  rw [build_inputs_length] at hstate hinputs
  have outputs := Linear.below_of_getD hstate
  have inputs_length := build_inputs_length
  have state_length := build_state_length
  constructor
  · intro e member
    simp only [poseidonTemplate, entries, List.mem_append, List.mem_flatMap, List.mem_map,
      List.mem_range] at member
    rcases member with (⟨⟨form, i⟩, hp, he⟩ | ⟨⟨form, i⟩, hp, he⟩) | ⟨row, hrow, rfl⟩
    · have := (List.mem_zipIdx hp).2.1
      rw [(mem_push he).2.1]
      show i < sboxes + 16
      omega
    · have := (List.mem_zipIdx hp).2.1
      rw [(mem_push he).2.1]
      show sboxes + i < sboxes + 16
      omega
    · exact hrow
  · intro e member k hk
    simp only [poseidonTemplate, entries, List.mem_append, List.mem_flatMap, List.mem_map,
      List.mem_range] at member
    show k < output + 16
    rcases member with (⟨⟨form, i⟩, hp, he⟩ | ⟨⟨form, i⟩, hp, he⟩) | ⟨row, hrow, rfl⟩
    · obtain ⟨t, ht, rfl⟩ := (mem_push he).2.2 k hk
      have := hinputs form (List.fst_mem_of_mem_zipIdx hp) t ht
      simp only [output]
      omega
    · obtain ⟨t, ht, rfl⟩ := (mem_push he).2.2 k hk
      have := outputs form (List.fst_mem_of_mem_zipIdx hp) t ht
      simp only [output]
      omega
    · simp only [Option.some.injEq] at hk
      subst hk
      rw [rowCell_eq]
      simp only [output]
      omega

/-! ## Rows -/

private theorem push_rowSum (c : ℕ → F) (m : Fin 8) (row : ℕ) (form : Linear) (sign : F) (j : Fin 8) (r : ℕ) :
    ((push m row form sign).map fun e => if e.matrix = j ∧ e.row = r then e.value c else 0).sum =
      if m = j ∧ row = r then sign * form.eval c else 0 := by
  by_cases here : m = j ∧ row = r
  · have terms : ((form.terms.map fun t => (⟨m, row, some t.1, sign * t.2⟩ : TemplateEntry F)).map
        fun e => if e.matrix = j ∧ e.row = r then e.value c else 0).sum =
          sign * Linear.termsValue c form.terms := by
      rw [List.map_map, Linear.termsValue, ← List.sum_map_mul_left]
      congr 1
      apply List.map_congr_left
      intro t _
      simp [here, TemplateEntry.value, mul_assoc]
    have constant : ((if form.constant ≠ 0 then [(⟨m, row, none, sign * form.constant⟩ : TemplateEntry F)]
        else []).map fun e => if e.matrix = j ∧ e.row = r then e.value c else 0).sum =
          sign * form.constant := by
      by_cases zero : form.constant = 0
      · simp [zero]
      · simp [zero, here, TemplateEntry.value]
    rw [push, List.map_append, List.sum_append, terms, constant, if_pos here, Linear.eval_eq]
    ring
  · rw [if_neg here]
    apply List.sum_eq_zero
    intro x member
    rw [List.mem_map] at member
    obtain ⟨e, he, rfl⟩ := member
    obtain ⟨hm, hrow, -⟩ := mem_push he
    rw [if_neg (by rw [hm, hrow]; exact here)]

private theorem zipIdx_rowSum (c : ℕ → F) (m : Fin 8) (base : ℕ) (sign : F) (j : Fin 8) (r : ℕ) :
    ∀ (L : List Linear) (first : ℕ),
      (((L.zipIdx first).flatMap fun p => push m (base + p.2) p.1 sign).map
          fun e => if e.matrix = j ∧ e.row = r then e.value c else 0).sum =
        if m = j ∧ base + first ≤ r ∧ r < base + first + L.length then
          sign * (L.getD (r - (base + first)) .zero).eval c
        else 0
  | [], first => by simp
  | l :: L, first => by
      rw [List.zipIdx_cons, List.flatMap_cons, List.map_append, List.sum_append, push_rowSum,
        zipIdx_rowSum c m base sign j r L (first + 1)]
      by_cases hm : m = j
      · by_cases here : base + first = r
        · subst here
          simp [hm]
        · rw [if_neg (fun h => here h.2), zero_add, List.length_cons]
          by_cases inside : base + (first + 1) ≤ r ∧ r < base + (first + 1) + L.length
          · rw [if_pos ⟨hm, inside⟩, if_pos ⟨hm, by omega, by omega⟩]
            obtain ⟨k, hk⟩ : ∃ k, r - (base + first) = k + 1 := ⟨r - (base + first) - 1, by omega⟩
            rw [hk, List.getD_cons_succ, show r - (base + (first + 1)) = k by omega]
          · rw [if_neg (fun h => inside h.2), if_neg (fun h => inside ⟨by omega, by omega⟩)]
      · simp [hm]

private theorem ident_rowSum (c : ℕ → F) (j : Fin 8) (r : ℕ) :
    ∀ n, (((List.range n).map fun row => (⟨7, row, some (rowCell row), 1⟩ : TemplateEntry F)).map
        fun e => if e.matrix = j ∧ e.row = r then e.value c else 0).sum =
      if j = 7 ∧ r < n then c (16 + r) else 0
  | 0 => by simp
  | n + 1 => by
      rw [List.range_succ, List.map_append, List.map_append, List.sum_append, ident_rowSum c j r n]
      simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero,
        TemplateEntry.value, rowCell_eq, Option.map_some, Option.getD_some, one_mul]
      by_cases hj : j = 7
      · subst hj
        by_cases hr : r < n
        · rw [if_pos ⟨rfl, hr⟩, if_neg (by omega), if_pos ⟨rfl, by omega⟩, add_zero]
        · by_cases hn : n = r
          · subst hn
            simp
          · rw [if_neg (by omega), if_neg (by omega), if_neg (by omega), add_zero]
      · simp [hj, Ne.symm hj]

private theorem rowValue_eq (c : ℕ → F) (j : Fin 8) (r : ℕ) :
    poseidonTemplate.rowValue c j r =
      (if 0 = j ∧ r < sboxes then (build.inputs.getD r .zero).eval c else 0) +
        (if 7 = j ∧ sboxes ≤ r ∧ r < sboxes + 16 then -(build.state.getD (r - sboxes) .zero).eval c
          else 0) +
        (if j = 7 ∧ r < sboxes + 16 then c (16 + r) else 0) := by
  have partA := zipIdx_rowSum c 0 0 1 j r build.inputs 0
  have partB := zipIdx_rowSum c 7 sboxes (-1) j r build.state 0
  have partC := ident_rowSum c j r (sboxes + 16)
  simp only [zero_add, add_zero, Nat.zero_le, true_and, one_mul, neg_one_mul, Nat.sub_zero,
    build_inputs_length, build_state_length] at partA partB
  simp only [Template.rowValue, poseidonTemplate, entries, List.map_append, List.sum_append]
  rw [partA, partB, partC]

private theorem sbox_eq (x : F) : Spec.Poseidon2.sbox x = x * x * x * x * x * x * x := by
  unfold Spec.Poseidon2.sbox
  ring

private theorem sbox_row (c : ℕ → F) (holds : poseidonTemplate.Holds c) {r : ℕ} (hr : r < sboxes) :
    c (16 + r) = Spec.Poseidon2.sbox ((build.inputs.getD r .zero).eval c) := by
  have row := holds r (show r < sboxes + 16 by omega)
  simp only [constraint, rowValue_eq] at row
  simp [hr, show ¬sboxes ≤ r by omega, show r < sboxes + 16 by omega] at row
  rw [sbox_eq, List.getD_eq_getElem?_getD, ← sub_eq_zero.mp row]
  ring

private theorem output_row (c : ℕ → F) (holds : poseidonTemplate.Holds c) {i : ℕ} (hi : i < 16) :
    c (output + i) = (build.state.getD i .zero).eval c := by
  have row := holds (sboxes + i) (show sboxes + i < sboxes + 16 by omega)
  simp only [constraint, rowValue_eq] at row
  simp [hi, show ¬sboxes + i < sboxes by omega] at row
  have cell : output + i = 16 + (sboxes + i) := by
    unfold output
    omega
  rw [cell, List.getD_eq_getElem?_getD]
  exact neg_add_eq_zero.mp row

/-! ## Soundness -/

/-- **Block soundness.** If every row of the Poseidon2 template holds on the
cells `c` of one block, the output cells `166..182` are the Poseidon2
permutation of the input cells `0..16`. -/
theorem block_sound (c : ℕ → F) (holds : poseidonTemplate.Holds c) :
    List.ofFn (fun i : Fin 16 => c (output + i)) =
      Spec.Poseidon2.permute (List.ofFn fun i : Fin 16 => c i) := by
  have rows : SboxRows c build.inputs := by
    intro k hk
    rw [build_inputs_length] at hk
    exact sbox_row c holds hk
  have initial : evalState c start.state = fun i : Fin 16 => c i := by
    funext lane
    simp only [evalState, start]
    rw [getD_range_map _ _ lane.isLt, Linear.eval_cell]
  have run : evalState c build.state = Permutation.runF Permutation.schedule fun i : Fin 16 => c i := by
    have := run_eval c Permutation.schedule start rows
    rwa [initial] at this
  calc List.ofFn (fun i : Fin 16 => c (output + i))
      = List.ofFn (evalState c build.state) := by
        congr 1
        funext i
        exact output_row c holds i.isLt
    _ = List.ofFn (Permutation.runF Permutation.schedule fun i : Fin 16 => c i) := by rw [run]
    _ = Permutation.runReference Permutation.schedule (List.ofFn fun i : Fin 16 => c i) :=
        Permutation.runF_eq_reference _ _
    _ = Spec.Poseidon2.permute (List.ofFn fun i : Fin 16 => c i) :=
        Permutation.runReference_schedule _

end CompressionLean.Lowering.Block
