import NightstreamFPrime.Circuit.Sequence
import NightstreamFPrime.Gadgets.Poseidon2.RawFormal
import NightstreamFPrime.Gadgets.Range.CanonicalPublicU64
import NightstreamFPrime.Lifecycle.XOut

/-!
Owns the low-norm prior-state hash circuit: one raw Poseidon2 hash, four
canonical word-to-public-bit children, one marker row, and thirteen zero-tail
rows. The public cells are caller-owned; all digest words are internal wires.
-/

namespace NightstreamFPrime.Lifecycle.PriorStateHash

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Range
open NightstreamFPrime.Lifecycle.PaperAlgebra

def publicWidth : Nat := ringDegree * publicRingColumns

theorem publicWidth_eq : publicWidth = 270 := by
  rfl

def markerIndex : Fin publicWidth := ⟨0, by rw [publicWidth_eq]; omega⟩

def digestBitIndexNat (word : Fin 4) (bit : Nat) : Fin publicWidth :=
  ⟨1 + word.val * 64 + bit % 64, by
    have wordBound := word.isLt
    have bitBound := Nat.mod_lt bit (by decide : 0 < 64)
    rw [publicWidth_eq]
    omega⟩

def digestBitIndex (word : Fin 4) (bit : Fin 64) : Fin publicWidth :=
  digestBitIndexNat word bit.val

def tailIndex (lane : Fin 13) : Fin publicWidth :=
  ⟨257 + lane.val, by
    have laneBound := lane.isLt
    rw [publicWidth_eq]
    omega⟩

def encodedHash (digest : Digest) : Fin publicWidth → F :=
  NightstreamFPrime.Lifecycle.encodedHashCells digest

@[simp] theorem encodedHash_marker (digest : Digest) :
    encodedHash digest markerIndex = 1 := by
  rfl

@[simp] theorem encodedHash_digestBitNat (digest : Digest) (word : Fin 4)
    (bit : Nat) (bounded : bit < 64) :
    encodedHash digest (digestBitIndexNat word bit) =
      NightstreamFPrime.Lifecycle.digestBitWord
        (digest.getD word.val 0) bit := by
  have indexEq : digestBitIndexNat word bit =
      NightstreamFPrime.Lifecycle.digestBitIndexNat
        (logicalWidth := 0) word bit := by
    apply Fin.ext
    simp [digestBitIndexNat,
      NightstreamFPrime.Lifecycle.digestBitIndexNat]
  rw [indexEq]
  exact NightstreamFPrime.Lifecycle.encodedHashCells_digestBitNat
    digest word bit bounded

@[simp] theorem encodedHash_digestBit (digest : Digest)
    (word : Fin 4) (bit : Fin 64) :
    encodedHash digest (digestBitIndex word bit) =
      NightstreamFPrime.Lifecycle.digestBitWord
        (digest.getD word.val 0) bit.val := by
  exact encodedHash_digestBitNat digest word bit.val bit.isLt

@[simp] theorem encodedHash_tail (digest : Digest) (lane : Fin 13) :
    encodedHash digest (tailIndex lane) = 0 := by
  apply NightstreamFPrime.Lifecycle.encodedHashCells_tail
  simp [tailIndex]

theorem column_cases (column : Fin publicWidth) :
    column = markerIndex ∨
      (∃ word : Fin 4, ∃ bit : Fin 64,
        column = digestBitIndex word bit) ∨
      ∃ lane : Fin 13, column = tailIndex lane := by
  by_cases marker : column.val = 0
  · exact Or.inl (Fin.ext marker)
  · by_cases encoded : column.val < 257
    · let position := column.val - 1
      let word : Fin 4 := ⟨position / 64, by
        have columnPositive : 0 < column.val := Nat.pos_of_ne_zero marker
        have positionBound : position < 256 := by
          dsimp [position]
          omega
        omega⟩
      let bit : Fin 64 := ⟨position % 64, Nat.mod_lt _ (by decide)⟩
      refine Or.inr (Or.inl ⟨word, bit, ?_⟩)
      apply Fin.ext
      simp [digestBitIndex, digestBitIndexNat, word, bit, position]
      have columnPositive : 0 < column.val := Nat.pos_of_ne_zero marker
      omega
    · let lane : Fin 13 := ⟨column.val - 257, by
        have columnBound := column.isLt
        change column.val < 270 at columnBound
        omega⟩
      refine Or.inr (Or.inr ⟨lane, ?_⟩)
      apply Fin.ext
      simp [tailIndex, lane]
      omega

structure Interface where
  preimage : Nat → List Expr
  publicInput : Nat → Fin publicWidth → Expr

def Assumptions (interface : Interface) (offset : Nat) (_env : Env) : Prop :=
  (∀ expression ∈ interface.preimage offset, expression.VarsBelow offset) ∧
    ∀ column, (interface.publicInput offset column).VarsBelow offset

def SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  (fun column => (interface.publicInput offset column).eval env) =
    encodedHash (Poseidon2.hash
      (Hash.evalList env (interface.preimage offset)))

def hashInterface (interface : Interface) : RawFormal.Interface where
  input := interface.preimage

@[simp] theorem hashInterface_input (interface : Interface) (offset : Nat) :
    (hashInterface interface).input offset = interface.preimage offset := by
  rfl

def hashCircuit (interface : Interface) : FormalCircuit :=
  RawFormal.circuit (hashInterface interface)

def compiledHashLength (interface : Interface) (offset : Nat) : Nat :=
  (RawFormal.program (hashInterface interface) offset).recipes.length

def hashLength (interface : Interface) (offset : Nat) : Nat :=
  (Hash.inputChunks (interface.preimage offset)).length * 592 + 592

theorem compiledHashLength_eq_hashLength (interface : Interface) (offset : Nat) :
    compiledHashLength interface offset = hashLength interface offset := by
  unfold compiledHashLength hashLength RawFormal.program hashInterface
  exact Hash.compile_recipes_length offset (interface.preimage offset)

theorem hashLength_eq (interface : Interface) (offset : Nat) :
    hashLength interface offset =
      (Hash.inputChunks (interface.preimage offset)).length * 592 + 592 := by
  rfl

def hashEnd (interface : Interface) (offset : Nat) : Nat :=
  offset + hashLength interface offset

def wordOffset (interface : Interface) (offset : Nat) (word : Fin 4) : Nat :=
  hashEnd interface offset + word.val * CanonicalPublicU64.privateCount

def finalOffset (interface : Interface) (offset : Nat) : Nat :=
  hashEnd interface offset + 4 * CanonicalPublicU64.privateCount

def wordInterface (interface : Interface) (parentOffset : Nat)
    (word : Fin 4) : CanonicalPublicU64.Interface where
  source := fun _ => RawFormal.digest (hashInterface interface) parentOffset word
  bit := fun _ bit => interface.publicInput parentOffset
    (digestBitIndexNat word bit)

@[simp] theorem wordInterface_source (interface : Interface)
    (parentOffset childOffset : Nat) (word : Fin 4) :
    (wordInterface interface parentOffset word).source childOffset =
      RawFormal.digest (hashInterface interface) parentOffset word := by
  rfl

@[simp] theorem wordInterface_bit (interface : Interface)
    (parentOffset childOffset : Nat) (word : Fin 4) (bit : Nat) :
    (wordInterface interface parentOffset word).bit childOffset bit =
      interface.publicInput parentOffset (digestBitIndexNat word bit) := by
  rfl

def wordCircuit (interface : Interface) (parentOffset : Nat)
    (word : Fin 4) : FormalCircuit :=
  CanonicalPublicU64.circuit (wordInterface interface parentOffset word)

def hashName : String := "prior_state_hash.poseidon2"

def wordName (word : Fin 4) : String :=
  "prior_state_hash.word." ++ toString word.val

def hashOp (interface : Interface) (offset : Nat) : Op :=
  Sequence.childOp hashName (hashCircuit interface) offset

theorem hashOp_flatConstraints_eq (interface : Interface) (offset : Nat) :
    flatConstraints [hashOp interface offset] =
      recipeConstraints offset
        (Hash.compile offset (interface.preimage offset)).recipes := by
  unfold hashOp Sequence.childOp
  rw [flatConstraints_singleton]
  simp only [Op.flatConstraints, FormalCircuit.asSubcircuit_constraints]
  exact RawFormal.flatConstraints_eq (hashInterface interface) offset

private theorem hashCircuit_localLength_compiled
    (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (hashCircuit interface).main offset) =
      compiledHashLength interface offset := by
  exact RawFormal.localLength_eq (hashInterface interface) offset

def wordOp (interface : Interface) (offset : Nat) (word : Fin 4) : Op :=
  Sequence.childOp (wordName word) (wordCircuit interface offset word)
    (wordOffset interface offset word)

def wordOps (interface : Interface) (offset : Nat) : List Op :=
  List.ofFn (wordOp interface offset)

def bindingAssertions (interface : Interface) (offset : Nat) : List Op :=
  Op.assertZero (interface.publicInput offset markerIndex - 1) ::
    List.ofFn (fun lane : Fin 13 =>
      Op.assertZero (interface.publicInput offset (tailIndex lane)))

def opsAt (interface : Interface) (offset : Nat) : List Op :=
  hashOp interface offset ::
    (wordOps interface offset ++ bindingAssertions interface offset)

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), finalOffset interface offset, opsAt interface offset)

@[simp] theorem main_ops (interface : Interface) (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := by
  rfl

def logicalPrivateCount (interface : Interface) (offset : Nat) : Nat :=
  hashLength interface offset + 264

def logicalRowCount (interface : Interface) (offset : Nat) : Nat :=
  hashLength interface offset + 538

private theorem hashOp_localLength (interface : Interface) (offset : Nat) :
    (hashOp interface offset).localLength = hashLength interface offset := by
  rw [hashOp, Sequence.childOp_localLength]
  calc
    localLength (Circuit.ops (hashCircuit interface).main offset) =
        compiledHashLength interface offset :=
      RawFormal.localLength_eq _ _
    _ = hashLength interface offset :=
      compiledHashLength_eq_hashLength interface offset

private theorem wordCircuit_localLength_eq
    (interface : Interface) (offset : Nat) (word : Fin 4) :
    localLength (Circuit.ops (wordCircuit interface offset word).main
      (wordOffset interface offset word)) =
      CanonicalPublicU64.privateCount := by
  exact CanonicalPublicU64.localLength_eq
    (wordInterface interface offset word) (wordOffset interface offset word)

private theorem wordOp_localLength (interface : Interface) (offset : Nat)
    (word : Fin 4) :
    (wordOp interface offset word).localLength =
      CanonicalPublicU64.privateCount := by
  rw [wordOp, Sequence.childOp_localLength]
  exact wordCircuit_localLength_eq interface offset word

private theorem wordOps_localLength (interface : Interface) (offset : Nat) :
    localLength (wordOps interface offset) = 264 := by
  change (List.ofFn (fun word : Fin 4 =>
    (wordOp interface offset word).localLength)).sum = 264
  simp only [wordOp_localLength]
  decide

private theorem bindingAssertions_localLength
    (interface : Interface) (offset : Nat) :
    localLength (bindingAssertions interface offset) = 0 := by
  change (0 :: List.ofFn (fun _ : Fin 13 => 0)).sum = 0
  simp

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (main interface) offset) =
      logicalPrivateCount interface offset := by
  rw [main_ops]
  change (hashOp interface offset).localLength +
    localLength (wordOps interface offset ++
      bindingAssertions interface offset) = _
  rw [Sequence.localLength_append, hashOp_localLength,
    wordOps_localLength, bindingAssertions_localLength]
  rfl

private theorem hashOp_rowCount (interface : Interface) (offset : Nat) :
    (hashOp interface offset).rowCount = hashLength interface offset := by
  change compiledHashLength interface offset = hashLength interface offset
  exact compiledHashLength_eq_hashLength interface offset

private theorem wordOp_rowCount (interface : Interface) (offset : Nat)
    (word : Fin 4) :
    (wordOp interface offset word).rowCount =
      CanonicalPublicU64.rowCount := by
  rfl

private theorem wordOps_rowCount (interface : Interface) (offset : Nat) :
    rowCount (wordOps interface offset) = 524 := by
  change (List.ofFn (fun word : Fin 4 =>
    (wordOp interface offset word).rowCount)).sum = 524
  simp only [wordOp_rowCount]
  decide

private theorem bindingAssertions_rowCount
    (interface : Interface) (offset : Nat) :
    rowCount (bindingAssertions interface offset) = 14 := by
  change (1 :: List.ofFn (fun _ : Fin 13 => 1)).sum = 14
  decide

theorem flatConstraints_length_eq (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (main interface) offset)).length =
      logicalRowCount interface offset := by
  rw [flatConstraints_length_eq_rowCount]
  change (hashOp interface offset).rowCount +
    rowCount (wordOps interface offset ++
      bindingAssertions interface offset) = _
  rw [rowCount_append, hashOp_rowCount, wordOps_rowCount,
    bindingAssertions_rowCount]
  rfl

private theorem rawAssumptions (interface : Interface) (offset : Nat)
    {env : Env} (assumptions : Assumptions interface offset env) :
    RawFormal.Assumptions (hashInterface interface) offset env :=
  assumptions.1

private theorem wordAssumptions (interface : Interface) (offset : Nat)
    {env : Env} (assumptions : Assumptions interface offset env)
    (word : Fin 4) :
    CanonicalPublicU64.Assumptions (wordInterface interface offset word)
      (wordOffset interface offset word) env := by
  have digestBelow := RawFormal.digest_varsBelow (hashInterface interface)
    offset (rawAssumptions interface offset assumptions) word
  rw [show (RawFormal.program (hashInterface interface) offset).recipes.length =
      hashLength interface offset by
        exact compiledHashLength_eq_hashLength interface offset] at digestBelow
  constructor
  · exact Expr.VarsBelow.mono _
      digestBelow (by simp [wordOffset, hashEnd])
  · intro bit bounded
    exact Expr.VarsBelow.mono _
      (assumptions.2 (digestBitIndexNat word bit)) (by
        simp [wordOffset, hashEnd]
        omega)

private theorem hashCall_sound (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env)
    (rows : holds env (opsAt interface offset)) :
    RawFormal.SpecHolds (hashInterface interface) offset env := by
  have call := rows (hashOp interface offset) (by simp [opsAt])
  change RawFormal.Assumptions (hashInterface interface) offset env →
    RawFormal.SpecHolds (hashInterface interface) offset env at call
  exact call (rawAssumptions interface offset assumptions)

private theorem wordCall_sound (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env)
    (rows : holds env (opsAt interface offset)) (word : Fin 4) :
    CanonicalPublicU64.SpecHolds (wordInterface interface offset word)
      (wordOffset interface offset word) env := by
  have member : wordOp interface offset word ∈ opsAt interface offset := by
    apply List.mem_cons_of_mem
    apply List.mem_append_left
    rw [wordOps, List.mem_ofFn']
    exact Set.mem_range_self word
  have call := rows (wordOp interface offset word) member
  change CanonicalPublicU64.Assumptions (wordInterface interface offset word)
      (wordOffset interface offset word) env →
    CanonicalPublicU64.SpecHolds (wordInterface interface offset word)
      (wordOffset interface offset word) env at call
  exact call (wordAssumptions interface offset assumptions word)

private theorem markerAssertion_mem (interface : Interface) (offset : Nat) :
    Op.assertZero (interface.publicInput offset markerIndex - 1) ∈
      opsAt interface offset := by
  apply List.mem_cons_of_mem
  apply List.mem_append_right (wordOps interface offset)
  simp [bindingAssertions]

private theorem tailAssertion_mem (interface : Interface) (offset : Nat)
    (lane : Fin 13) :
    Op.assertZero (interface.publicInput offset (tailIndex lane)) ∈
      opsAt interface offset := by
  apply List.mem_cons_of_mem
  apply List.mem_append_right (wordOps interface offset)
  simp only [bindingAssertions, List.mem_cons]
  apply Or.inr
  rw [List.mem_ofFn']
  exact Set.mem_range_self lane

private theorem ofFn_getD {Alpha : Type} {count : Nat}
    (values : Fin count → Alpha) (lane : Fin count) (fallback : Alpha) :
    (List.ofFn values).getD lane.val fallback = values lane := by
  rw [List.getD_eq_get (List.ofFn values) fallback
    ⟨lane.val, by simp⟩]
  simp

private theorem hashDigestValue (interface : Interface) (offset : Nat)
    (env : Env)
    (hashSpec : RawFormal.SpecHolds (hashInterface interface) offset env)
    (word : Fin 4) :
    (RawFormal.digest (hashInterface interface) offset word).eval env =
      (Poseidon2.hash (Hash.evalList env (interface.preimage offset))).getD
        word.val 0 := by
  have selected := congrArg (fun values : List F => values.getD word.val 0)
    hashSpec
  calc
    (RawFormal.digest (hashInterface interface) offset word).eval env =
        (List.ofFn (fun lane =>
          (RawFormal.digest (hashInterface interface) offset lane).eval env)).getD
            word.val 0 := (ofFn_getD (fun lane : Fin 4 =>
              (RawFormal.digest (hashInterface interface) offset lane).eval env)
              word 0).symm
    _ = (Poseidon2.hash
        (Hash.evalList env (interface.preimage offset))).getD word.val 0 :=
      selected

private theorem spec_of_parts (interface : Interface) (offset : Nat)
    (env : Env)
    (hashSpec : RawFormal.SpecHolds (hashInterface interface) offset env)
    (words : ∀ word, CanonicalPublicU64.SpecHolds
      (wordInterface interface offset word)
      (wordOffset interface offset word) env)
    (marker : (interface.publicInput offset markerIndex).eval env = 1)
    (tail : ∀ lane,
      (interface.publicInput offset (tailIndex lane)).eval env = 0) :
    SpecHolds interface offset env := by
  unfold SpecHolds
  funext column
  rcases column_cases column with rfl | bitOrTail
  · simpa using marker
  · rcases bitOrTail with ⟨word, bit, rfl⟩ | ⟨lane, rfl⟩
    · apply Fin.eq_of_val_eq
      have wordValue := hashDigestValue interface offset env hashSpec word
      calc
        ((interface.publicInput offset (digestBitIndex word bit)).eval env).val =
            ((RawFormal.digest (hashInterface interface) offset word).eval env).val /
              2 ^ bit.val % 2 :=
          words word bit.val bit.isLt
        _ = ((Poseidon2.hash
              (Hash.evalList env (interface.preimage offset))).getD word.val 0).val /
              2 ^ bit.val % 2 := by rw [wordValue]
        _ = (encodedHash (Poseidon2.hash
              (Hash.evalList env (interface.preimage offset)))
              (digestBitIndex word bit)).val := by
          rw [encodedHash_digestBit]
          unfold NightstreamFPrime.Lifecycle.digestBitWord Poseidon2.ofNat
          have binary :
              ((Poseidon2.hash
                (Hash.evalList env (interface.preimage offset))).getD word.val 0).val /
                  2 ^ bit.val % 2 < goldilocksModulus := by
            exact lt_trans (Nat.mod_lt _ (by decide)) (by
              norm_num [goldilocksModulus])
          exact (Nat.mod_eq_of_lt binary).symm
    · simpa using tail lane

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    SpecHolds interface offset env := by
  change holds env (opsAt interface offset) at rows
  have hashSpec := hashCall_sound interface offset env assumptions rows
  have wordSpecs := wordCall_sound interface offset env assumptions rows
  have markerRow := rows
    (.assertZero (interface.publicInput offset markerIndex - 1))
    (markerAssertion_mem interface offset)
  have marker : (interface.publicInput offset markerIndex).eval env = 1 := by
    exact sub_eq_zero.mp (by simpa [Expr.eval_sub] using markerRow)
  have tail (lane : Fin 13) :
      (interface.publicInput offset (tailIndex lane)).eval env = 0 :=
    rows (.assertZero (interface.publicInput offset (tailIndex lane)))
      (tailAssertion_mem interface offset lane)
  exact spec_of_parts interface offset env hashSpec wordSpecs marker tail

private theorem spec_preserved_below (interface : Interface) (offset : Nat)
    (before after : Env) (assumptions : Assumptions interface offset before)
    (agrees : ∀ index, index < offset → after index = before index)
    (specification : SpecHolds interface offset before) :
    SpecHolds interface offset after := by
  have publicEqual :
      (fun column => (interface.publicInput offset column).eval after) =
        fun column => (interface.publicInput offset column).eval before := by
    funext column
    exact (interface.publicInput offset column).eval_eq_of_agree_below
      offset after before (assumptions.2 column) agrees
  have preimageEqual :
      Hash.evalList after (interface.preimage offset) =
        Hash.evalList before (interface.preimage offset) := by
    unfold Hash.evalList
    apply List.map_congr_left
    intro expression member
    exact expression.eval_eq_of_agree_below offset after before
      (assumptions.1 expression member) agrees
  unfold SpecHolds at specification ⊢
  rw [publicEqual, preimageEqual]
  exact specification

private theorem wordSpec_of_spec (interface : Interface) (offset : Nat)
    (env : Env) (specification : SpecHolds interface offset env)
    (hashSpec : RawFormal.SpecHolds (hashInterface interface) offset env)
    (word : Fin 4) :
    CanonicalPublicU64.SpecHolds (wordInterface interface offset word)
      (wordOffset interface offset word) env := by
  intro bit bounded
  have publicValue := congrFun specification (digestBitIndexNat word bit)
  have digestValue := hashDigestValue interface offset env hashSpec word
  calc
    ((interface.publicInput offset (digestBitIndexNat word bit)).eval env).val =
        (encodedHash (Poseidon2.hash
          (Hash.evalList env (interface.preimage offset)))
          (digestBitIndexNat word bit)).val := congrArg Fin.val publicValue
    _ = (NightstreamFPrime.Lifecycle.digestBitWord
        ((Poseidon2.hash
          (Hash.evalList env (interface.preimage offset))).getD word.val 0)
        bit).val := congrArg Fin.val
          (encodedHash_digestBitNat _ word bit bounded)
    _ = ((Poseidon2.hash
          (Hash.evalList env (interface.preimage offset))).getD word.val 0).val /
        2 ^ bit % 2 := by
      unfold NightstreamFPrime.Lifecycle.digestBitWord Poseidon2.ofNat
      apply Nat.mod_eq_of_lt
      exact lt_trans (Nat.mod_lt _ (by decide)) (by
        norm_num [goldilocksModulus])
    _ = ((RawFormal.digest (hashInterface interface) offset word).eval env).val /
        2 ^ bit % 2 := by rw [digestValue]

private theorem hashScope (interface : Interface) (offset : Nat) {env : Env}
    (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (hashCircuit interface).main offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (hashCircuit interface).main offset)) :=
  RawFormal.flatConstraints_varsBelow (hashInterface interface) offset
    (rawAssumptions interface offset assumptions)

private theorem wordScope (interface : Interface) (offset : Nat) {env : Env}
    (assumptions : Assumptions interface offset env) (word : Fin 4) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (wordCircuit interface offset word).main
        (wordOffset interface offset word)),
      expression.VarsBelow
        (wordOffset interface offset word +
          localLength (Circuit.ops (wordCircuit interface offset word).main
            (wordOffset interface offset word))) :=
  CanonicalPublicU64.flatConstraints_varsBelow
    (wordInterface interface offset word) (wordOffset interface offset word)
    (wordAssumptions interface offset assumptions word)

theorem flatConstraints_varsBelow
    (interface : Interface) (offset : Nat) {env : Env}
    (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (main interface) offset),
      expression.VarsBelow
        (offset + localLength (Circuit.ops (main interface) offset)) := by
  rw [localLength_eq]
  change ∀ expression ∈ flatConstraints (opsAt interface offset),
    expression.VarsBelow (offset + logicalPrivateCount interface offset)
  intro expression member
  simp only [flatConstraints, List.mem_flatMap] at member
  rcases member with ⟨operation, operationMember, constraintMember⟩
  simp only [opsAt, List.mem_cons] at operationMember
  rcases operationMember with rfl | operationMember
  · change expression ∈ flatConstraints
      (Circuit.ops (hashCircuit interface).main offset) at constraintMember
    have below := hashScope interface offset assumptions expression
      constraintMember
    apply Expr.VarsBelow.mono expression below
    rw [hashCircuit_localLength_compiled,
      compiledHashLength_eq_hashLength]
    unfold logicalPrivateCount
    omega
  · rcases List.mem_append.mp operationMember with wordMember | bindingMember
    · rw [wordOps, List.mem_ofFn'] at wordMember
      rcases wordMember with ⟨word, rfl⟩
      change expression ∈ flatConstraints
        (Circuit.ops (wordCircuit interface offset word).main
          (wordOffset interface offset word)) at constraintMember
      have below := wordScope interface offset assumptions word expression
        constraintMember
      apply Expr.VarsBelow.mono expression below
      rw [wordCircuit_localLength_eq]
      have wordBound := word.isLt
      unfold wordOffset hashEnd logicalPrivateCount
      simp only [CanonicalPublicU64.privateCount,
        CanonicalU64.auxiliaryCount]
      omega
    · simp only [bindingAssertions, List.mem_cons] at bindingMember
      rcases bindingMember with rfl | bindingMember
      · simp only [Op.flatConstraints, List.mem_singleton] at constraintMember
        subst expression
        apply Expr.VarsBelow.sub
        · exact Expr.VarsBelow.mono _ (assumptions.2 markerIndex) (by omega)
        · exact trivial
      · rw [List.mem_ofFn'] at bindingMember
        rcases bindingMember with ⟨lane, rfl⟩
        simp only [Op.flatConstraints, List.mem_singleton] at constraintMember
        subst expression
        exact Expr.VarsBelow.mono _ (assumptions.2 (tailIndex lane)) (by omega)

private theorem prefixHashSpec
    (interface : Interface) (offset : Nat) (initial : Env)
    (completedPrefix : Sequence.Prefix initial offset)
    (assumptions : Assumptions interface offset completedPrefix.current)
    (member : hashOp interface offset ∈ completedPrefix.operations) :
    RawFormal.SpecHolds (hashInterface interface) offset
      completedPrefix.current := by
  have logical := holdsFlat_implies_holds completedPrefix.current
    completedPrefix.operations completedPrefix.rows
  have call := logical (hashOp interface offset) member
  change RawFormal.Assumptions (hashInterface interface) offset
      completedPrefix.current →
    RawFormal.SpecHolds (hashInterface interface) offset
      completedPrefix.current at call
  exact call (rawAssumptions interface offset assumptions)

private theorem appendWord
    (interface : Interface) (offset : Nat) (initial : Env)
    (initialAssumptions : Assumptions interface offset initial)
    (initialSpecification : SpecHolds interface offset initial)
    (before : Sequence.Prefix initial offset) (word : Fin 4)
    (startEq : offset + localLength before.operations =
      wordOffset interface offset word)
    (hashMember : hashOp interface offset ∈ before.operations) :
    ∃ after : Sequence.Prefix initial offset,
      after.operations = before.operations ++ [wordOp interface offset word] ∧
      offset + localLength after.operations =
        wordOffset interface offset word + CanonicalPublicU64.privateCount := by
  have agreesBelow : ∀ index, index < offset →
      before.current index = initial index := by
    intro index below
    exact before.agrees index (Or.inl below)
  have currentAssumptions : Assumptions interface offset before.current :=
    initialAssumptions
  have currentSpecification := spec_preserved_below interface offset
    initial before.current initialAssumptions agreesBelow initialSpecification
  have currentHash := prefixHashSpec interface offset initial before
    currentAssumptions hashMember
  have currentWord := wordSpec_of_spec interface offset before.current
    currentSpecification currentHash word
  have childAssumptions := wordAssumptions interface offset
    currentAssumptions word
  rcases CanonicalPublicU64.complete (wordInterface interface offset word)
      before.current (wordOffset interface offset word) childAssumptions
      currentWord with
    ⟨wordEnv, wordAgrees, wordRows⟩
  rcases Sequence.appendBuiltAt before (wordName word)
      (wordCircuit interface offset word) (wordOffset interface offset word)
      startEq (wordScope interface offset currentAssumptions word)
      wordEnv wordAgrees wordRows with
    ⟨after, operationsEq, endEq, _, _⟩
  refine ⟨after, operationsEq, ?_⟩
  have childLength : localLength
      (Circuit.ops (wordCircuit interface offset word).main
        (wordOffset interface offset word)) =
        CanonicalPublicU64.privateCount := by
    exact wordCircuit_localLength_eq interface offset word
  rw [childLength] at endEq
  exact endEq

private theorem coreOps_localLength (interface : Interface) (offset : Nat) :
    localLength (hashOp interface offset :: wordOps interface offset) =
      logicalPrivateCount interface offset := by
  change (hashOp interface offset).localLength +
    localLength (wordOps interface offset) = _
  rw [hashOp_localLength, wordOps_localLength]
  rfl

private theorem bindingRowsAt
    (interface : Interface) (offset : Nat) (env : Env)
    (specification : SpecHolds interface offset env) :
    ConstraintsHold env (flatConstraints (bindingAssertions interface offset)) := by
  intro expression member
  simp only [flatConstraints, List.mem_flatMap] at member
  rcases member with ⟨operation, operationMember, constraintMember⟩
  simp only [bindingAssertions, List.mem_cons] at operationMember
  rcases operationMember with rfl | operationMember
  · simp only [Op.flatConstraints, List.mem_singleton] at constraintMember
    subst expression
    have marker := congrFun specification markerIndex
    change (interface.publicInput offset markerIndex - 1).eval env = 0
    rw [Expr.eval_sub]
    apply sub_eq_zero.mpr
    simpa using marker
  · rw [List.mem_ofFn'] at operationMember
    rcases operationMember with ⟨lane, rfl⟩
    simp only [Op.flatConstraints, List.mem_singleton] at constraintMember
    subst expression
    have tail := congrFun specification (tailIndex lane)
    simpa using tail

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  let empty := Sequence.empty env offset
  rcases RawFormal.complete (hashInterface interface) env offset
      (rawAssumptions interface offset assumptions) with
    ⟨hashEnv, hashAgrees, hashRows⟩
  rcases Sequence.appendBuiltAt empty hashName (hashCircuit interface) offset
      (by rfl) (hashScope interface offset assumptions)
      hashEnv hashAgrees hashRows with
    ⟨afterHash, hashOps, hashEndEq, _, _⟩
  have hashPrefix : afterHash.operations = [hashOp interface offset] := by
    simpa [empty, hashOp] using hashOps
  have word0Start : offset + localLength afterHash.operations =
      wordOffset interface offset 0 := by
    calc
      offset + localLength afterHash.operations =
          offset + localLength (Circuit.ops (hashCircuit interface).main offset) :=
        hashEndEq
      _ = offset + compiledHashLength interface offset := by
        exact congrArg (fun length => offset + length)
          (hashCircuit_localLength_compiled interface offset)
      _ = wordOffset interface offset 0 := by
        rw [compiledHashLength_eq_hashLength]
        simp [wordOffset, hashEnd]
  rcases appendWord interface offset env assumptions specification afterHash 0
      word0Start (by simp [hashPrefix]) with
    ⟨after0, word0Ops, word0End⟩
  have word1Start : offset + localLength after0.operations =
      wordOffset interface offset 1 := by
    rw [word0End]
    simp [wordOffset, hashEnd, CanonicalPublicU64.privateCount]
  rcases appendWord interface offset env assumptions specification after0 1
      word1Start (by simp [word0Ops, hashPrefix]) with
    ⟨after1, word1Ops, word1End⟩
  have word2Start : offset + localLength after1.operations =
      wordOffset interface offset 2 := by
    rw [word1End]
    simp [wordOffset, hashEnd, CanonicalPublicU64.privateCount]
    omega
  rcases appendWord interface offset env assumptions specification after1 2
      word2Start (by simp [word1Ops, word0Ops, hashPrefix]) with
    ⟨after2, word2Ops, word2End⟩
  have word3Start : offset + localLength after2.operations =
      wordOffset interface offset 3 := by
    rw [word2End]
    simp [wordOffset, hashEnd, CanonicalPublicU64.privateCount]
    omega
  rcases appendWord interface offset env assumptions specification after2 3
      word3Start (by simp [word2Ops, word1Ops, word0Ops, hashPrefix]) with
    ⟨after3, word3Ops, word3End⟩
  have completedOps : after3.operations =
      hashOp interface offset :: wordOps interface offset := by
    rw [word3Ops, word2Ops, word1Ops, word0Ops, hashPrefix]
    simp [wordOps, List.ofFn_succ]
  have agreesBelow : ∀ index, index < offset → after3.current index = env index := by
    intro index below
    exact after3.agrees index (Or.inl below)
  have completedSpecification := spec_preserved_below interface offset env
    after3.current assumptions agreesBelow specification
  refine ⟨after3.current, ?_, ?_⟩
  · have fullLength : localLength (Circuit.ops (main interface) offset) =
        localLength after3.operations := by
      calc
        localLength (Circuit.ops (main interface) offset) =
            logicalPrivateCount interface offset :=
          localLength_eq interface offset
        _ = localLength (hashOp interface offset :: wordOps interface offset) :=
          (coreOps_localLength interface offset).symm
        _ = localLength after3.operations :=
          (congrArg localLength completedOps).symm
    rw [fullLength]
    exact after3.agrees
  · have coreRows : holdsFlat after3.current
        (hashOp interface offset :: wordOps interface offset) := by
      rw [← completedOps]
      exact after3.rows
    unfold holdsFlat at coreRows ⊢
    change ConstraintsHold after3.current
      (flatConstraints ((hashOp interface offset :: wordOps interface offset) ++
        bindingAssertions interface offset))
    rw [flatConstraints_append, constraintsHold_append]
    constructor
    · exact coreRows
    · exact bindingRowsAt interface offset after3.current
        completedSpecification

/-- The production logical builder for the low-norm prior-state hash. -/
def circuit (interface : Interface) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  privateCount := logicalPrivateCount interface
  rowCount := logicalRowCount interface
  privateCount_eq := localLength_eq interface
  rowCount_eq := flatConstraints_length_eq interface
  soundness := soundness interface
  completeness := completeness interface

theorem circuit_localLength (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) =
      logicalPrivateCount interface offset :=
  localLength_eq interface offset

end NightstreamFPrime.Lifecycle.PriorStateHash
