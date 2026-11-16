import NightstreamFPrime.Lifecycle.BoundedDepth.State
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.StatementAbsorption

/-! Slot words in a step circuit (CW-2.5d, d4a). A `SlotExpr` holds the expressions of one slot's
fields. Its words are the circuit form of `serializeSlot`, so a hash gadget over them binds the
slot's value (`words_eval`), and equal words give equal slots (`value_eq_of_words`).
`selectRows` is one row per word, `out = left + c · (right - left)`: it copies `left` when `c` is
0 and `right` when `c` is 1 (`selectRows_eval`). -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth.SlotWords

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_2

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}

/-- The expressions of one slot: its claim's fields and its depth word. -/
structure SlotExpr (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth)
    where
  commitment : Fin productionProfile.commitmentWidth → Fin ringDegree → Expr
  evaluation : StatementAbsorption.EvaluationExpr
  point : Fin productionShape.cubeVariables → KExpr
  publicInput : Fin (FullShape logicalWidth publicFits).publicWidth → Expr
  depth : Expr

namespace SlotExpr

/-- The slot's words in the order of `serializeSlot`. -/
def words (slot : SlotExpr logicalWidth publicFits) : List Expr :=
  StatementAbsorption.serializeCommitmentExpr slot.commitment ++
    (StatementAbsorption.serializeEvalKExpr slot.evaluation ++
      StatementAbsorption.serializeEvalAExpr slot.evaluation) ++
    StatementAbsorption.serializePointExpr slot.point ++
    StatementAbsorption.serializePublicInputExpr slot.publicInput ++ [slot.depth]

/-- The slot that the expressions carry; the depth is the value of the depth word. -/
def value (slot : SlotExpr logicalWidth publicFits) (env : Env) :
    SlotData logicalWidth publicFits :=
  (⟨fun row lane => (slot.commitment row lane).eval env,
      fun column => (slot.publicInput column).eval env,
      StatementAbsorption.evalPoint slot.point env,
      StatementAbsorption.evalEvaluation slot.evaluation env⟩,
    (slot.depth.eval env).val)

theorem natWord_val (word : F) : natWord word.val = word :=
  Fin.ext (by simp [natWord, Poseidon2.ofNat, Nat.mod_eq_of_lt word.isLt])

private theorem publicInput_eval
    (input : Fin (FullShape logicalWidth publicFits).publicWidth → Expr) (env : Env) :
    Hash.evalList env (StatementAbsorption.serializePublicInputExpr input) =
      serializePublicInput (publicFits := publicFits) (fun column => (input column).eval env) := by
  unfold Hash.evalList StatementAbsorption.serializePublicInputExpr serializePublicInput
  simp [List.map_map, Function.comp_def]

/-- The words evaluate to the serialization of the slot they carry. -/
theorem words_eval (slot : SlotExpr logicalWidth publicFits) (env : Env) :
    Hash.evalList env slot.words = serializeSlot (slot.value env) := by
  unfold words serializeSlot serializeEvaluations
  simp only [Hash.evalList, List.map_append, List.map_cons, List.map_nil]
  rw [← Hash.evalList, ← Hash.evalList, ← Hash.evalList, ← Hash.evalList, ← Hash.evalList,
    StatementAbsorption.serializeCommitmentExpr_eval, StatementAbsorption.serializeEvalKExpr_eval,
    StatementAbsorption.serializeEvalAExpr_eval, StatementAbsorption.serializePointExpr_eval,
    publicInput_eval]
  simp only [value, natWord_val]

theorem words_length (slot : SlotExpr logicalWidth publicFits) : slot.words.length = 2169 := by
  have length := congrArg List.length (words_eval slot (fun _ => 0))
  simp only [Hash.evalList, List.length_map] at length
  rw [length, serializeSlot_length_eq]

/-- Words that serialize a slot of depth below the modulus carry that slot. -/
theorem value_eq_of_words (slot : SlotExpr logicalWidth publicFits) (env : Env)
    (data : SlotData logicalWidth publicFits) (depth : data.2 < fieldModulus)
    (same : Hash.evalList env slot.words = serializeSlot data) : slot.value env = data :=
  serializeSlot_injective (slot.depth.eval env).isLt depth ((words_eval slot env).symm.trans same)

end SlotExpr

/-- One row per word: `out = left + c · (right - left)`. -/
def selectRows (condition : Expr) (left right out : List Expr) : List Expr :=
  ((left.zip right).zip out).map fun words => words.2 - words.1.1 - condition * (words.1.2 - words.1.1)

/-- Under a boolean condition, the select rows copy `left` at 0 and `right` at 1. -/
theorem selectRows_eval (env : Env) (condition : Expr) :
    ∀ (left right out : List Expr), left.length = out.length → right.length = out.length →
      ConstraintsHold env (selectRows condition left right out) →
      (condition.eval env = 0 → Hash.evalList env out = Hash.evalList env left) ∧
        (condition.eval env = 1 → Hash.evalList env out = Hash.evalList env right)
  | [], [], [], _, _, _ => ⟨fun _ => rfl, fun _ => rfl⟩
  | leftWord :: left, rightWord :: right, outWord :: out, leftLength, rightLength, rows => by
      have head := rows _ (List.mem_cons_self ..)
      have tail := selectRows_eval env condition left right out
        (Nat.succ.inj leftLength) (Nat.succ.inj rightLength)
        (fun row member => rows row (List.mem_cons_of_mem _ member))
      simp only [Expr.eval_sub, Expr.eval_hmul] at head
      constructor
      · intro zero
        rw [zero, zero_mul, sub_zero, sub_eq_zero] at head
        simp only [Hash.evalList, List.map_cons] at tail ⊢
        rw [head, tail.1 zero]
      · intro one
        rw [one, one_mul, sub_sub_sub_cancel_right, sub_eq_zero] at head
        simp only [Hash.evalList, List.map_cons] at tail ⊢
        rw [head, tail.2 one]
  | [], _ :: _, [], _, rightLength, _ => by simp at rightLength
  | _ :: _, _, [], leftLength, _, _ => by simp at leftLength
  | [], _, _ :: _, leftLength, _, _ => by simp at leftLength
  | _ :: _, [], _ :: _, _, rightLength, _ => by simp at rightLength

/-- The converse of `selectRows_eval`: words that copy `left` under a 0 condition, or `right`
under a 1 condition, satisfy the select rows. -/
theorem selectRows_holds (env : Env) (condition : Expr) :
    ∀ (left right out : List Expr), left.length = out.length → right.length = out.length →
      (condition.eval env = 0 ∧ Hash.evalList env out = Hash.evalList env left) ∨
        (condition.eval env = 1 ∧ Hash.evalList env out = Hash.evalList env right) →
      ConstraintsHold env (selectRows condition left right out)
  | [], [], [], _, _, _ => fun _ member => by simp [selectRows] at member
  | leftWord :: left, rightWord :: right, outWord :: out, leftLength, rightLength, copy => by
      simp only [Hash.evalList, List.map_cons, List.cons.injEq] at copy
      have tail := selectRows_holds env condition left right out
        (Nat.succ.inj leftLength) (Nat.succ.inj rightLength) (by
          rcases copy with ⟨zero, _, words⟩ | ⟨one, _, words⟩
          · exact Or.inl ⟨zero, words⟩
          · exact Or.inr ⟨one, words⟩)
      intro row member
      simp only [selectRows, List.zip_cons_cons, List.map_cons, List.mem_cons] at member
      rcases member with rfl | member
      · simp only [Expr.eval_sub, Expr.eval_hmul]
        rcases copy with ⟨zero, head, _⟩ | ⟨one, head, _⟩
        · rw [zero, head, zero_mul, sub_zero, sub_self]
        · rw [one, head, one_mul, sub_self]
      · exact tail row member
  | [], _ :: _, [], _, rightLength, _ => by simp at rightLength
  | _ :: _, _, [], leftLength, _, _ => by simp at leftLength
  | [], _, _ :: _, leftLength, _, _ => by simp at leftLength
  | _ :: _, [], _ :: _, _, rightLength, _ => by simp at rightLength

/-- The select rows read only the condition and the three word lists, so they hold again when
those keep their values. -/
theorem selectRows_congr (left right : Env) (condition : Expr)
    (conditionEq : condition.eval right = condition.eval left) :
    ∀ (first second out : List Expr),
      Hash.evalList right first = Hash.evalList left first →
      Hash.evalList right second = Hash.evalList left second →
      Hash.evalList right out = Hash.evalList left out →
      ConstraintsHold left (selectRows condition first second out) →
      ConstraintsHold right (selectRows condition first second out)
  | firstWord :: first, secondWord :: second, outWord :: out, firstEq, secondEq, outEq, holds => by
      simp only [Hash.evalList, List.map_cons, List.cons.injEq] at firstEq secondEq outEq
      have tail := selectRows_congr left right condition conditionEq first second out
        firstEq.2 secondEq.2 outEq.2
        (fun row member => holds row (by
          simp only [selectRows, List.zip_cons_cons, List.map_cons, List.mem_cons]
          exact Or.inr member))
      intro row member
      simp only [selectRows, List.zip_cons_cons, List.map_cons, List.mem_cons] at member
      rcases member with rfl | member
      · have head := holds _ (List.mem_cons_self ..)
        simp only [Expr.eval_sub, Expr.eval_hmul] at head ⊢
        rw [conditionEq, firstEq.1, secondEq.1, outEq.1]
        exact head
      · exact tail row member
  | [], _, _, _, _, _, _ => fun _ member => by simp [selectRows] at member
  | _ :: _, [], _, _, _, _, _ => fun _ member => by simp [selectRows] at member
  | _ :: _, _ :: _, [], _, _, _, _ => fun _ member => by simp [selectRows] at member

/-- The select rows read the condition and the words. -/
theorem selectRows_below {bound : Nat} {condition : Expr}
    (conditionBelow : condition.VarsBelow bound)
    {left right out : List Expr} (leftBelow : ∀ word ∈ left, word.VarsBelow bound)
    (rightBelow : ∀ word ∈ right, word.VarsBelow bound)
    (outBelow : ∀ word ∈ out, word.VarsBelow bound) :
    ∀ row ∈ selectRows condition left right out, row.VarsBelow bound := by
  intro row member
  simp only [selectRows, List.mem_map] at member
  obtain ⟨⟨⟨leftWord, rightWord⟩, outWord⟩, inside, rfl⟩ := member
  obtain ⟨pairs, outMember⟩ := List.of_mem_zip inside
  obtain ⟨leftMember, rightMember⟩ := List.of_mem_zip pairs
  exact Expr.VarsBelow.sub _ _ _
    (Expr.VarsBelow.sub _ _ _ (outBelow _ outMember) (leftBelow _ leftMember))
    (Expr.VarsBelow.mul _ _ _ conditionBelow
      (Expr.VarsBelow.sub _ _ _ (rightBelow _ rightMember) (leftBelow _ leftMember)))

theorem ofFn_below {bound : Nat} {words : Fin 4 → Expr}
    (below : ∀ lane, (words lane).VarsBelow bound) :
    ∀ word ∈ List.ofFn words, word.VarsBelow bound := by
  intro word member
  obtain ⟨lane, rfl⟩ := List.mem_ofFn.mp member
  exact below lane

/-- The select rows read only `allowed` columns when the condition and the words do. -/
theorem selectRows_satisfy {allowed : Nat → Prop} {condition : Expr}
    (conditionReads : condition.VarsSatisfy allowed)
    {left right out : List Expr} (leftReads : ∀ word ∈ left, word.VarsSatisfy allowed)
    (rightReads : ∀ word ∈ right, word.VarsSatisfy allowed)
    (outReads : ∀ word ∈ out, word.VarsSatisfy allowed) :
    ∀ row ∈ selectRows condition left right out, row.VarsSatisfy allowed := by
  intro row member
  simp only [selectRows, List.mem_map] at member
  obtain ⟨⟨⟨leftWord, rightWord⟩, outWord⟩, inside, rfl⟩ := member
  obtain ⟨pairs, outMember⟩ := List.of_mem_zip inside
  obtain ⟨leftMember, rightMember⟩ := List.of_mem_zip pairs
  exact Expr.VarsSatisfy.sub _ _ _
    (Expr.VarsSatisfy.sub _ _ _ (outReads _ outMember) (leftReads _ leftMember))
    (Expr.VarsSatisfy.mul _ _ _ conditionReads
      (Expr.VarsSatisfy.sub _ _ _ (rightReads _ rightMember) (leftReads _ leftMember)))

theorem ofFn_satisfy {allowed : Nat → Prop} {words : Fin 4 → Expr}
    (reads : ∀ lane, (words lane).VarsSatisfy allowed) :
    ∀ word ∈ List.ofFn words, word.VarsSatisfy allowed := by
  intro word member
  obtain ⟨lane, rfl⟩ := List.mem_ofFn.mp member
  exact reads lane

end NightstreamFPrime.Lifecycle.BoundedDepth.SlotWords
