import NightstreamFPrime.Lifecycle.OptionB.State
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.StatementAbsorption

/-! Slot words in a step circuit (CW-2.5d, d4a). A `SlotExpr` holds the expressions of one slot's
fields. Its words are the circuit form of `serializeSlot`, so a hash gadget over them binds the
slot's value (`words_eval`), and equal words give equal slots (`value_eq_of_words`).
`selectRows` is one row per word, `out = left + c · (right - left)`: it copies `left` when `c` is
0 and `right` when `c` is 1 (`selectRows_eval`). -/

namespace NightstreamFPrime.Lifecycle.OptionB.SlotWords

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_2

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The expressions of one slot: its claim's fields and its depth word. -/
structure SlotExpr (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth)
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

theorem words_length (slot : SlotExpr logicalWidth publicFits) : slot.words.length = 2055 := by
  have length := congrArg List.length (words_eval slot (fun _ => 0))
  simp only [Hash.evalList, List.length_map] at length
  rw [length, serializeSlot_length_eq]

/-- Words that serialize a slot of depth below the modulus carry that slot. -/
theorem value_eq_of_words (slot : SlotExpr logicalWidth publicFits) (env : Env)
    (data : SlotData logicalWidth publicFits) (depth : data.2 < goldilocksModulus)
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

end NightstreamFPrime.Lifecycle.OptionB.SlotWords
