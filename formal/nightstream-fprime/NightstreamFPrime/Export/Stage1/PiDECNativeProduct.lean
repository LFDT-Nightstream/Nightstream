import NightstreamFPrime.Export.NativePoseidon2RoundCore
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
import Mathlib.Tactic.SplitIfs

/-!
Native-word execution of the existing PiDEC ring product and fixed-width
accumulation. Field arithmetic is owned by NativePoseidon2RoundCore.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECNativeProduct

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open NightstreamFPrime.Export.NativePoseidon2

@[inline] private def toWord (value : F) : UInt64 := UInt64.ofNat value.val

private theorem toWord_toNat (value : F) : (toWord value).toNat = value.val := by
  exact UInt64.toNat_ofNat_of_lt'
    (Nat.lt_trans value.isLt (by decide : goldilocksModulus < 2 ^ 64))

private theorem toWord_denote (value : F) : (toWord value).denote = value := by
  apply Fin.ext
  change (toWord value).toNat % goldilocksModulus = value.val
  rw [toWord_toNat, Nat.mod_eq_of_lt value.isLt]

private def toWords (value : StoredRing) : Vector UInt64 ringDegree :=
  Vector.ofFn fun lane => toWord (value.get lane)

private theorem toWords_get (value : StoredRing) (lane : Fin ringDegree) :
    (toWords value).get lane = toWord (value.get lane) := by
  change (Vector.ofFn (fun index => toWord (value.get index)))[lane.val] = _
  rw [Vector.getElem_ofFn]

private theorem toWords_canonical (value : StoredRing) (lane : Fin ringDegree) :
    ((toWords value).get lane).toNat < goldilocksModulus := by
  rw [toWords_get, toWord_toNat]
  exact (value.get lane).isLt

private theorem toWords_denote (value : StoredRing) (lane : Fin ringDegree) :
    ((toWords value).get lane).denote = value.get lane := by
  rw [toWords_get, toWord_denote]

private def negativeOne : UInt64 := UInt64.ofNat (goldilocksModulus - 1)

private theorem zero_denote : (0 : UInt64).denote = (0 : F) := by decide
private theorem one_denote : (1 : UInt64).denote = (1 : F) := by decide
private theorem negativeOne_denote : negativeOne.denote = (-1 : F) := by decide

@[inline] private def accumulate64 (acc key digit : UInt64) : UInt64 :=
  if digit = 0 then acc
  else if digit = 1 then add64 acc key
  else if digit = negativeOne then sub64 acc key
  else add64 acc (mul64 key digit)

private theorem accumulate64_canonical (acc key digit : UInt64)
    (ha : acc.toNat < goldilocksModulus) (hk : key.toNat < goldilocksModulus) :
    (accumulate64 acc key digit).toNat < goldilocksModulus := by
  unfold accumulate64
  split_ifs
  · exact ha
  · exact add64_canonical _ _ ha hk
  · exact sub64_canonical _ _ ha hk
  · exact add64_canonical _ _ ha (mul64_canonical _ _)

private theorem accumulate64_denote (acc key digit : UInt64)
    (ha : acc.toNat < goldilocksModulus)
    (hk : key.toNat < goldilocksModulus)
    (hd : digit.toNat < goldilocksModulus) :
    (accumulate64 acc key digit).denote = acc.denote + key.denote * digit.denote := by
  unfold accumulate64
  split_ifs with zero one negative
  · rw [zero, zero_denote]
    have product : key.denote * (0 : F) = 0 := ConcreteCarrier.baseLaws.mul_zero _
    have addition : acc.denote + (0 : F) = acc.denote := ConcreteCarrier.baseLaws.add_zero _
    rw [product, addition]
  · rw [add64_denote _ _ ha hk, one, one_denote]
    have product : key.denote * (1 : F) = key.denote := ConcreteCarrier.baseLaws.mul_one _
    rw [product]
  · rw [sub64_denote _ _ ha hk, negative, negativeOne_denote]
    have product : key.denote * (-1 : F) = -key.denote := by
      calc
        key.denote * (-1) = (-1) * key.denote := ConcreteCarrier.baseLaws.mul_comm _ _
        _ = -(1 * key.denote) := ConcreteCarrier.baseLaws.neg_mul 1 _
        _ = -key.denote := congrArg (fun value : F => -value) (ConcreteCarrier.baseLaws.one_mul _)
    rw [product, Fin.sub_eq_add_neg]
  · rw [add64_denote _ _ ha (mul64_canonical _ _), mul64_denote _ _ hk hd]

private def read64 (values : Vector UInt64 ringDegree) (index : Nat) : UInt64 :=
  if live : index < ringDegree then values.get ⟨index, live⟩ else 0

private theorem read64_canonical (values : Vector UInt64 ringDegree)
    (canonical : ∀ lane, (values.get lane).toNat < goldilocksModulus) (index : Nat) :
    (read64 values index).toNat < goldilocksModulus := by
  unfold read64
  split_ifs with live
  · exact canonical ⟨index, live⟩
  · decide

private theorem read64_denote (values : StoredRing) (index : Nat) :
    (read64 (toWords values) index).denote = ringFCoeff values.get index := by
  unfold read64 ringFCoeff
  split_ifs with live
  · exact toWords_denote values ⟨index, live⟩
  · exact zero_denote

private def rawStep64 (key digit : Vector UInt64 ringDegree)
    (degree : Nat) (acc : UInt64) (input : Nat) : UInt64 :=
  if input ≤ degree ∧ degree - input < ringDegree then
    accumulate64 acc (read64 key input) (read64 digit (degree - input))
  else acc

private theorem rawStep64_canonical (key digit : Vector UInt64 ringDegree)
    (degree : Nat) (acc : UInt64) (input : Nat)
    (keyCanonical : ∀ lane, (key.get lane).toNat < goldilocksModulus)
    (ha : acc.toNat < goldilocksModulus) :
    (rawStep64 key digit degree acc input).toNat < goldilocksModulus := by
  unfold rawStep64
  split_ifs
  · exact accumulate64_canonical _ _ _ ha (read64_canonical key keyCanonical input)
  · exact ha

private theorem rawStep64_denote (key digit : StoredRing)
    (degree : Nat) (acc : UInt64) (input : Nat)
    (ha : acc.toNat < goldilocksModulus) :
    (rawStep64 (toWords key) (toWords digit) degree acc input).denote =
      if input ≤ degree ∧ degree - input < ringDegree then
        acc.denote + ringFCoeff key.get input * ringFCoeff digit.get (degree - input)
      else acc.denote := by
  unfold rawStep64
  split_ifs
  · rw [accumulate64_denote _ _ _ ha
      (read64_canonical _ (toWords_canonical key) input)
      (read64_canonical _ (toWords_canonical digit) (degree - input)),
      read64_denote, read64_denote]
  · rfl

private theorem fold64_canonical (key digit : Vector UInt64 ringDegree) (degree : Nat)
    (keyCanonical : ∀ lane, (key.get lane).toNat < goldilocksModulus)
    (inputs : List Nat) :
    ∀ acc : UInt64, acc.toNat < goldilocksModulus →
      (inputs.foldl (rawStep64 key digit degree) acc).toNat < goldilocksModulus := by
  induction inputs with
  | nil => intro acc ha; exact ha
  | cons input inputs ih =>
      intro acc ha
      rw [List.foldl_cons]
      exact ih _ (rawStep64_canonical key digit degree acc input keyCanonical ha)

private theorem fold64_denote (key digit : StoredRing) (degree : Nat) (inputs : List Nat) :
    ∀ acc : UInt64, acc.toNat < goldilocksModulus →
      (inputs.foldl (rawStep64 (toWords key) (toWords digit) degree) acc).denote =
        inputs.foldl (fun total input =>
          if input ≤ degree ∧ degree - input < ringDegree then
            total + ringFCoeff key.get input * ringFCoeff digit.get (degree - input)
          else total) acc.denote := by
  induction inputs with
  | nil => intro acc ha; rfl
  | cons input inputs ih =>
      intro acc ha
      rw [List.foldl_cons, List.foldl_cons,
        ih _ (rawStep64_canonical _ _ degree acc input (toWords_canonical key) ha),
        rawStep64_denote key digit degree acc input ha]

private def rawPairs (degree : Nat) : List (Nat × Nat) :=
  (List.range ringDegree).filterMap fun input =>
    if input ≤ degree ∧ degree - input < ringDegree then
      some (input, degree - input)
    else none

private def rawPairTable (_ : Unit) : Vector (List (Nat × Nat)) (2 * ringDegree - 1) :=
  Vector.ofFn fun degree => rawPairs degree.val

private def rawCoefficient64 (key digit : Vector UInt64 ringDegree) (degree : Nat) : UInt64 :=
  let pairs := if live : degree < 2 * ringDegree - 1 then
    (rawPairTable ()).get ⟨degree, live⟩
    else rawPairs degree
  pairs.foldl (fun acc pair =>
    accumulate64 acc (read64 key pair.1) (read64 digit pair.2)) 0

private theorem rawCoefficient64_eq_fold (key digit : Vector UInt64 ringDegree)
    (degree : Nat) :
    rawCoefficient64 key digit degree =
      (List.range ringDegree).foldl (rawStep64 key digit degree) 0 := by
  have pairs :
      (if live : degree < 2 * ringDegree - 1 then
        (rawPairTable ()).get ⟨degree, live⟩
      else rawPairs degree) = rawPairs degree := by
    split_ifs with live
    · change (Vector.ofFn (fun index : Fin (2 * ringDegree - 1) =>
        rawPairs index.val))[degree] = _
      rw [Vector.getElem_ofFn]
    · rfl
  unfold rawCoefficient64
  rw [pairs, rawPairs, List.foldl_filterMap]
  apply congrArg (fun step : UInt64 → Nat → UInt64 =>
    (List.range ringDegree).foldl step 0)
  funext acc input
  unfold rawStep64
  split_ifs <;> rfl

private theorem rawCoefficient64_canonical (key digit : Vector UInt64 ringDegree)
    (degree : Nat) (keyCanonical : ∀ lane, (key.get lane).toNat < goldilocksModulus) :
    (rawCoefficient64 key digit degree).toNat < goldilocksModulus := by
  rw [rawCoefficient64_eq_fold]
  exact fold64_canonical key digit degree keyCanonical (List.range ringDegree) 0 (by decide)

private theorem rawCoefficient64_denote (key digit : StoredRing) (degree : Nat) :
    (rawCoefficient64 (toWords key) (toWords digit) degree).denote =
      rawMulCoeffF key.get digit.get degree := by
  rw [rawCoefficient64_eq_fold]
  unfold rawMulCoeffF
  rw [fold64_denote key digit degree (List.range ringDegree) 0 (by decide), zero_denote]

private def foldedCoefficients64 (key digit : Vector UInt64 ringDegree) :
    Vector UInt64 ringMiddleDegree :=
  Vector.ofFn fun index => rawCoefficient64 key digit (index.val + ringDegree)

private theorem foldedCoefficients64_get (key digit : Vector UInt64 ringDegree)
    (index : Fin ringMiddleDegree) :
    (foldedCoefficients64 key digit).get index =
      rawCoefficient64 key digit (index.val + ringDegree) := by
  change (Vector.ofFn (fun lane : Fin ringMiddleDegree =>
    rawCoefficient64 key digit (lane.val + ringDegree)))[index.val] = _
  rw [Vector.getElem_ofFn]

private def folded64 (coefficients : Vector UInt64 ringMiddleDegree)
    (output : Fin ringDegree) : UInt64 :=
  if below : output.val < ringMiddleDegree then
    coefficients.get ⟨output.val, below⟩
  else coefficients.get ⟨output.val - ringMiddleDegree, by
    have live := output.isLt
    change output.val < 54 at live
    change output.val - 27 < 27
    omega⟩

private theorem folded64_value (key digit : Vector UInt64 ringDegree)
    (output : Fin ringDegree) :
    folded64 (foldedCoefficients64 key digit) output =
      if output.val < ringMiddleDegree then
        rawCoefficient64 key digit (output.val + ringDegree)
      else rawCoefficient64 key digit (output.val + ringMiddleDegree) := by
  unfold folded64
  split_ifs with below
  · exact foldedCoefficients64_get key digit ⟨output.val, below⟩
  · rw [foldedCoefficients64_get]
    congr 1
    change output.val - 27 + 54 = output.val + 27
    change ¬ output.val < 27 at below
    omega

private theorem folded64_canonical (key digit : Vector UInt64 ringDegree)
    (output : Fin ringDegree)
    (keyCanonical : ∀ lane, (key.get lane).toNat < goldilocksModulus) :
    (folded64 (foldedCoefficients64 key digit) output).toNat < goldilocksModulus := by
  rw [folded64_value]
  split_ifs <;> exact rawCoefficient64_canonical key digit _ keyCanonical

private theorem folded64_denote (key digit : StoredRing) (output : Fin ringDegree) :
    (folded64 (foldedCoefficients64 (toWords key) (toWords digit)) output).denote =
      if output.val < ringMiddleDegree then
        rawMulCoeffF key.get digit.get (output.val + ringDegree)
      else rawMulCoeffF key.get digit.get (output.val + ringMiddleDegree) := by
  rw [folded64_value]
  split_ifs <;> exact rawCoefficient64_denote key digit _

private def twice64 (key digit : Vector UInt64 ringDegree) (output : Fin ringDegree) : UInt64 :=
  if output.val + 81 ≤ 106 then rawCoefficient64 key digit (output.val + 81) else 0

private theorem twice64_canonical (key digit : Vector UInt64 ringDegree)
    (output : Fin ringDegree)
    (keyCanonical : ∀ lane, (key.get lane).toNat < goldilocksModulus) :
    (twice64 key digit output).toNat < goldilocksModulus := by
  unfold twice64
  split_ifs
  · exact rawCoefficient64_canonical key digit _ keyCanonical
  · decide

private theorem twice64_denote (key digit : StoredRing) (output : Fin ringDegree) :
    (twice64 (toWords key) (toWords digit) output).denote =
      if output.val + 81 ≤ 106 then
        rawMulCoeffF key.get digit.get (output.val + 81)
      else 0 := by
  unfold twice64
  split_ifs
  · exact rawCoefficient64_denote key digit _
  · exact zero_denote

private def coefficient64 (key digit : Vector UInt64 ringDegree)
    (folded : Vector UInt64 ringMiddleDegree) (output : Fin ringDegree) : UInt64 :=
  add64 (sub64 (rawCoefficient64 key digit output.val) (folded64 folded output))
    (twice64 key digit output)

private theorem coefficient64_canonical (key digit : Vector UInt64 ringDegree)
    (output : Fin ringDegree)
    (keyCanonical : ∀ lane, (key.get lane).toNat < goldilocksModulus) :
    (coefficient64 key digit (foldedCoefficients64 key digit) output).toNat < goldilocksModulus :=
  add64_canonical _ _
    (sub64_canonical _ _ (rawCoefficient64_canonical key digit output.val keyCanonical)
      (folded64_canonical key digit output keyCanonical))
    (twice64_canonical key digit output keyCanonical)

private theorem coefficient64_denote (key digit : StoredRing) (output : Fin ringDegree) :
    (coefficient64 (toWords key) (toWords digit)
      (foldedCoefficients64 (toWords key) (toWords digit)) output).denote =
      ringFMul key.get digit.get output := by
  unfold coefficient64
  rw [add64_denote _ _
      (sub64_canonical _ _
        (rawCoefficient64_canonical _ _ output.val (toWords_canonical key))
        (folded64_canonical _ _ output (toWords_canonical key)))
      (twice64_canonical _ _ output (toWords_canonical key)),
    sub64_denote _ _
      (rawCoefficient64_canonical _ _ output.val (toWords_canonical key))
      (folded64_canonical _ _ output (toWords_canonical key)),
    rawCoefficient64_denote, folded64_denote, twice64_denote]
  rfl

private def fromWord (word : UInt64) (canonical : word.toNat < goldilocksModulus) : F :=
  ⟨word.toNat, canonical⟩

private theorem fromWord_denote (word : UInt64) (canonical : word.toNat < goldilocksModulus) :
    fromWord word canonical = word.denote := by
  apply Fin.ext
  change word.toNat = word.toNat % goldilocksModulus
  exact (Nat.mod_eq_of_lt canonical).symm

/-- Convert each input once and share raw degrees 54–80 across both output
halves. Return the same stored base-field ring; no caller supplies canonicality. -/
def multiply (key digit : StoredRing) : StoredRing :=
  let keyWords := toWords key
  let digitWords := toWords digit
  let folded := foldedCoefficients64 keyWords digitWords
  Vector.ofFn fun output =>
    fromWord (coefficient64 keyWords digitWords folded output)
      (coefficient64_canonical keyWords digitWords output (toWords_canonical key))

/-- The complete native-word product is the existing Phi81 multiplication. -/
theorem multiply_value (key digit : StoredRing) :
    (multiply key digit).get = ringFMul key.get digit.get := by
  funext output
  change (Vector.ofFn (fun lane : Fin ringDegree =>
    fromWord (coefficient64 (toWords key) (toWords digit)
      (foldedCoefficients64 (toWords key) (toWords digit)) lane)
      (coefficient64_canonical _ _ lane (toWords_canonical key))))[output.val] = _
  rw [Vector.getElem_ofFn, fromWord_denote]
  exact coefficient64_denote key digit output

/-- A key ring converted once for reuse by the children at one row/block. -/
structure PreparedKey where
  private mk ::
  private words : Vector UInt64 ringDegree
  private canonical : ∀ lane, (words.get lane).toNat < goldilocksModulus

/-- Preserve the complete key ring while preparing its native words. -/
def prepareKey (key : StoredRing) : PreparedKey where
  words := toWords key
  canonical := toWords_canonical key

/-- One child ring prepared once for all key rows. None records the exact
zero decision; only nonzero children allocate canonical native words. -/
structure PreparedDigit where
  private mk ::
  private words : Option (Vector UInt64 ringDegree)

private theorem allZero_iff (digit : StoredRing) :
    digit.all (fun value => value == 0) = true ↔
      ∀ lane : Fin ringDegree, digit.get lane = 0 := by
  simp only [Vector.all_eq_true, beq_iff_eq]
  constructor
  · intro zero lane
    exact zero lane.val lane.isLt
  · intro zero index bound
    exact zero ⟨index, bound⟩

def prepareDigit (digit : StoredRing) : PreparedDigit :=
  if digit.all (fun value => value == 0) then ⟨none⟩
  else ⟨some (toWords digit)⟩

/-- One canonical native-word ring accumulator. Its field view is materialized
only when the caller finishes a partial sum. -/
structure Accumulator where
  private mk ::
  private words : Vector UInt64 ringDegree
  private canonical : ∀ lane, (words.get lane).toNat < goldilocksModulus

namespace Accumulator

/-- Convert the completed native sum to the existing stored field boundary. -/
def finish (value : Accumulator) : StoredRing :=
  Vector.ofFn fun lane => fromWord (value.words.get lane) (value.canonical lane)

private theorem finish_get (value : Accumulator) (lane : Fin ringDegree) :
    value.finish.get lane = (value.words.get lane).denote := by
  change (Vector.ofFn (fun index : Fin ringDegree =>
    fromWord (value.words.get index) (value.canonical index)))[lane.val] = _
  rw [Vector.getElem_ofFn, fromWord_denote]

/-- Start an empty native ring sum. -/
def zero : Accumulator where
  words := Vector.replicate ringDegree 0
  canonical := by
    intro lane
    change ((Vector.replicate ringDegree (0 : UInt64))[lane.val]).toNat < goldilocksModulus
    rw [Vector.getElem_replicate]
    decide

theorem zero_value : zero.finish.get = ringFZero := by
  funext lane
  rw [finish_get]
  change ((Vector.replicate ringDegree (0 : UInt64))[lane.val]).denote = (0 : F)
  rw [Vector.getElem_replicate, zero_denote]

/-- Merge native partial sums without constructing intermediate field values. -/
def add (left right : Accumulator) : Accumulator where
  words := Vector.ofFn fun lane => add64 (left.words.get lane) (right.words.get lane)
  canonical := by
    intro lane
    change ((Vector.ofFn (fun index : Fin ringDegree =>
      add64 (left.words.get index) (right.words.get index)))[lane.val]).toNat < goldilocksModulus
    rw [Vector.getElem_ofFn]
    exact add64_canonical _ _ (left.canonical lane) (right.canonical lane)

theorem add_value (left right : Accumulator) :
    (add left right).finish.get = ringFAdd left.finish.get right.finish.get := by
  funext lane
  change (add left right).finish.get lane = left.finish.get lane + right.finish.get lane
  simp only [finish_get]
  change ((Vector.ofFn (fun index : Fin ringDegree =>
    add64 (left.words.get index) (right.words.get index)))[lane.val]).denote = _
  rw [Vector.getElem_ofFn, add64_denote _ _ (left.canonical lane) (right.canonical lane)]

@[inline] private def addWordProduct (initial : Accumulator) (key : PreparedKey)
    (digitWords : Vector UInt64 ringDegree) : Accumulator :=
  let keyWords := key.words
  let folded := foldedCoefficients64 keyWords digitWords
  { words := Vector.ofFn fun output =>
      add64 (initial.words.get output) (coefficient64 keyWords digitWords folded output)
    canonical := by
      intro output
      change ((Vector.ofFn (fun lane : Fin ringDegree =>
        add64 (initial.words.get lane)
          (coefficient64 keyWords digitWords folded lane)))[output.val]).toNat < goldilocksModulus
      rw [Vector.getElem_ofFn]
      exact add64_canonical _ _ (initial.canonical output)
        (coefficient64_canonical keyWords digitWords output key.canonical) }

/-- Add the existing complete ring product directly to a native sum. All
field-valued key and digit inputs retain the generic multiplication path. -/
def addProduct (initial : Accumulator) (key : PreparedKey) (digit : StoredRing) : Accumulator :=
  addWordProduct initial key (toWords digit)

/-- Reuse the child's zero decision and native words across key rows. -/
def addPreparedProduct (initial : Accumulator) (key : PreparedKey)
    (digit : PreparedDigit) : Accumulator :=
  match digit.words with
  | none => initial
  | some words => addWordProduct initial key words

theorem addPreparedProduct_eq (initial : Accumulator) (key : PreparedKey)
    (digit : StoredRing) :
    addPreparedProduct initial key (prepareDigit digit) =
      if ∀ lane : Fin ringDegree, digit.get lane = 0 then initial
      else addProduct initial key digit := by
  unfold prepareDigit addPreparedProduct
  simp only [allZero_iff]
  split_ifs <;> rfl

theorem addProduct_value (initial : Accumulator) (key digit : StoredRing) :
    (addProduct initial (prepareKey key) digit).finish.get =
      ringFAdd initial.finish.get (ringFMul key.get digit.get) := by
  funext output
  change (addProduct initial (prepareKey key) digit).finish.get output =
    initial.finish.get output + ringFMul key.get digit.get output
  simp only [finish_get]
  change ((Vector.ofFn (fun lane : Fin ringDegree =>
    add64 (initial.words.get lane)
      (coefficient64 (toWords key) (toWords digit)
        (foldedCoefficients64 (toWords key) (toWords digit)) lane)))[output.val]).denote = _
  rw [Vector.getElem_ofFn,
    add64_denote _ _ (initial.canonical output)
      (coefficient64_canonical _ _ output (toWords_canonical key)),
    coefficient64_denote]

end Accumulator

end NightstreamFPrime.Export.Stage1.PiDECNativeProduct
