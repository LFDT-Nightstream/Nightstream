import NightstreamFPrime.Export.NativePoseidon2RoundCore
import NightstreamFPrime.Export.Stage1.PiDECCyclicCoefficient
import NightstreamFPrime.Export.Stage1.PiDECSignedDigits
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
import Mathlib.Tactic.SplitIfs

/-!
Native-word execution of the existing PiDEC ring product and fixed-width
accumulation. Field arithmetic is owned by NativePoseidon2RoundCore.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECNativeProduct

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
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

private def cycleWord (key : Vector UInt64 ringDegree) (index : Nat) : UInt64 :=
  if index < 27 then read64 key index
  else if index < 54 then sub64 (read64 key index) (read64 key (index - 27))
  else sub64 0 (read64 key (index - 27))

private theorem cycleWord_canonical (key : Vector UInt64 ringDegree)
    (canonical : ∀ lane, (key.get lane).toNat < goldilocksModulus) (index : Nat) :
    (cycleWord key index).toNat < goldilocksModulus := by
  unfold cycleWord
  split_ifs
  · exact read64_canonical key canonical index
  · exact sub64_canonical _ _ (read64_canonical key canonical index)
      (read64_canonical key canonical (index - 27))
  · exact sub64_canonical _ _ (by decide)
      (read64_canonical key canonical (index - 27))

private theorem cycleWord_denote (key : StoredRing) (index : Nat) :
    (cycleWord (toWords key) index).denote =
      PiDECCyclicCoefficient.cycle key.get index := by
  unfold cycleWord PiDECCyclicCoefficient.cycle
  split_ifs
  · exact read64_denote key index
  · rw [sub64_denote _ _ (read64_canonical _ (toWords_canonical key) index)
        (read64_canonical _ (toWords_canonical key) (index - 27)),
      read64_denote, read64_denote]
  · rw [sub64_denote _ _ (by decide)
        (read64_canonical _ (toWords_canonical key) (index - 27)),
      zero_denote, read64_denote, Fin.sub_eq_add_neg, Fin.zero_add]

/-- Duplicate one period so a signed gather needs no modular index operation. -/
private def cycleWords (key : Vector UInt64 ringDegree) : Vector UInt64 162 :=
  let period : Vector UInt64 81 := Vector.ofFn fun index => cycleWord key index.val
  period ++ period

private theorem cycleWords_get (key : Vector UInt64 ringDegree) (index : Fin 162) :
    (cycleWords key).get index = cycleWord key (index.val % 81) := by
  change ((Vector.ofFn (fun lane : Fin 81 => cycleWord key lane.val)) ++
    (Vector.ofFn (fun lane : Fin 81 => cycleWord key lane.val)))[index.val] = _
  by_cases below : index.val < 81
  · rw [Vector.getElem_append_left below, Vector.getElem_ofFn,
      Nat.mod_eq_of_lt below]
  · have residue : index.val % 81 = index.val - 81 := by
      have bound := index.isLt
      omega
    rw [Vector.getElem_append_right index.isLt (by omega),
      Vector.getElem_ofFn, residue]

private theorem cycleWords_canonical (key : Vector UInt64 ringDegree)
    (canonical : ∀ lane, (key.get lane).toNat < goldilocksModulus) (index : Fin 162) :
    ((cycleWords key).get index).toNat < goldilocksModulus := by
  rw [cycleWords_get]
  exact cycleWord_canonical key canonical _

private theorem cycleWords_denote (key : StoredRing) (index : Fin 162) :
    ((cycleWords (toWords key)).get index).denote =
      PiDECCyclicCoefficient.cycle key.get (index.val % 81) := by
  rw [cycleWords_get, cycleWord_denote]

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
  private cycle : Vector UInt64 162
  private cycle_eq : cycle = cycleWords words

/-- Preserve the complete key ring while preparing its native words. -/
def prepareKey (key : StoredRing) : PreparedKey where
  words := toWords key
  canonical := toWords_canonical key
  cycle := cycleWords (toWords key)
  cycle_eq := rfl

private def keyView (key : PreparedKey) : StoredRing :=
  Vector.ofFn fun lane => fromWord (key.words.get lane) (key.canonical lane)

private theorem keyView_get (key : PreparedKey) (lane : Fin ringDegree) :
    (keyView key).get lane = fromWord (key.words.get lane) (key.canonical lane) := by
  change (Vector.ofFn (fun index : Fin ringDegree =>
    fromWord (key.words.get index) (key.canonical index)))[lane.val] = _
  rw [Vector.getElem_ofFn]

private theorem keyView_words (key : PreparedKey) :
    toWords (keyView key) = key.words := by
  apply Vector.ext
  intro index bound
  change (toWords (keyView key)).get ⟨index, bound⟩ = key.words.get ⟨index, bound⟩
  rw [toWords_get, keyView_get]
  apply UInt64.toNat_inj.1
  rw [toWord_toNat]
  rfl

@[inline] private def gather64 (key : PreparedKey) (base byte : UInt8) : UInt64 :=
  let index := base + byte
  if live : index < (162 : UInt8) then
    key.cycle.uget index.toUSize (by
      simpa only [UInt8.toNat_toUSize] using UInt8.lt_iff_toNat_lt.mp live)
  else 0

private theorem gather64_canonical (key : PreparedKey) (base byte : UInt8) :
    (gather64 key base byte).toNat < goldilocksModulus := by
  unfold gather64
  dsimp only
  split_ifs with live
  · have bound : (base + byte).toUSize.toNat < 162 := by
      simpa only [UInt8.toNat_toUSize] using UInt8.lt_iff_toNat_lt.mp live
    change (key.cycle.get ⟨(base + byte).toUSize.toNat, bound⟩).toNat < _
    rw [key.cycle_eq]
    exact cycleWords_canonical key.words key.canonical ⟨_, bound⟩
  · decide

private theorem gather64_offset (key : PreparedKey) (base : UInt8)
    (baseBound : base.toNat < 81) (input : Fin ringDegree) :
    (gather64 key base (PiDECSignedDigits.offset input)).denote =
      PiDECCyclicCoefficient.cycle (keyView key).get
        ((base.toNat + 81 - input.val) % 81) := by
  have inputBound : input.val < 54 := by simpa only [ringDegree] using input.isLt
  have totalBound : base.toNat + (81 - input.val) < 256 := by omega
  have indexValue : (base + PiDECSignedDigits.offset input).toUSize.toNat =
      base.toNat + 81 - input.val := by
    simp only [UInt8.toNat_toUSize, UInt8.toNat_add, PiDECSignedDigits.offset_toNat]
    rw [Nat.mod_eq_of_lt totalBound]
    omega
  have live : (base + PiDECSignedDigits.offset input).toUSize.toNat < 162 := by
    rw [indexValue]
    omega
  have byteLive : base + PiDECSignedDigits.offset input < (162 : UInt8) :=
    UInt8.lt_iff_toNat_lt.mpr (by simpa only [UInt8.toNat_toUSize] using live)
  unfold gather64
  rw [dif_pos byteLive]
  change (key.cycle.get
    ⟨(base + PiDECSignedDigits.offset input).toUSize.toNat, live⟩).denote = _
  rw [key.cycle_eq, ← keyView_words key, cycleWords_denote]
  simp only [indexValue]

@[inline] private def gatherBase (output : Fin ringDegree) : UInt8 :=
  UInt8.ofNat (if output.val < 27 then output.val else output.val + 27)

private theorem gatherBase_toNat (output : Fin ringDegree) :
    (gatherBase output).toNat =
      if output.val < 27 then output.val else output.val + 27 := by
  apply UInt8.toNat_ofNat_of_lt'
  change (if output.val < 27 then output.val else output.val + 27) < 256
  have bound := output.isLt
  change output.val < 54 at bound
  split_ifs <;> omega

private theorem gatherBase_bound (output : Fin ringDegree) :
    (gatherBase output).toNat < 81 := by
  rw [gatherBase_toNat]
  have bound := output.isLt
  change output.val < 54 at bound
  split_ifs <;> omega

private theorem gather64_coefficient (key : PreparedKey)
    (output input : Fin ringDegree) :
    (if output.val < 27 then
      (gather64 key (gatherBase output) (PiDECSignedDigits.offset input)).denote
    else
      -(gather64 key (gatherBase output) (PiDECSignedDigits.offset input)).denote) =
      CarrierAction.rightCoefficient (keyView key).get output input := by
  simp only [gather64_offset key (gatherBase output) (gatherBase_bound output) input,
    gatherBase_toNat]
  rw [← PiDECCyclicCoefficient.coefficient_eq_rightCoefficient]
  unfold PiDECCyclicCoefficient.coefficient
  split_ifs <;> rfl

private theorem neg_sumRange (count : Nat) (term : Nat → F) :
    -sumRange ConcreteCarrier.baseOps count term =
      sumRange ConcreteCarrier.baseOps count (fun index => -term index) := by
  induction count with
  | zero => exact Lean.Grind.AddCommGroup.neg_zero
  | succ count ih =>
      change -(sumRange ConcreteCarrier.baseOps count term + term count) =
        sumRange ConcreteCarrier.baseOps count (fun index => -term index) + -term count
      rw [Lean.Grind.AddCommGroup.neg_add, ih]

private theorem gatherLinear_eq (key : PreparedKey) (digit : StoredRing)
    (output : Fin ringDegree) :
    (if output.val < 27 then
      PiDECSignedDigits.linearCombination digit
        (fun byte => (gather64 key (gatherBase output) byte).denote)
    else
      -PiDECSignedDigits.linearCombination digit
        (fun byte => (gather64 key (gatherBase output) byte).denote)) =
      ringFMul (keyView key).get digit.get output := by
  rw [CarrierAction.ringFMul_apply_eq_rightLinear]
  unfold PiDECSignedDigits.linearCombination
  by_cases low : output.val < 27
  · rw [if_pos low]
    apply sumRange_congr
    intro index live
    simp only [dif_pos live]
    have kernel := gather64_coefficient key output ⟨index, live⟩
    rw [if_pos low] at kernel
    rw [kernel]
  · rw [if_neg low, neg_sumRange]
    apply sumRange_congr
    intro index live
    simp only [dif_pos live]
    have kernel := gather64_coefficient key output ⟨index, live⟩
    rw [if_neg low] at kernel
    rw [← Lean.Grind.Fin.neg_mul, kernel]

@[inline] private def addSignedCoefficient64 (key : PreparedKey)
    (digit : PiDECSignedDigits.Prepared) (output : Fin ringDegree)
    (initial : UInt64) : UInt64 :=
  let total := PiDECSignedDigits.fold64 digit (gather64 key (gatherBase output)) 0
  if output.val < 27 then add64 initial total else sub64 initial total

private theorem addSignedCoefficient64_canonical (key : PreparedKey)
    (digit : PiDECSignedDigits.Prepared) (output : Fin ringDegree)
    (initial : UInt64) (initialBound : initial.toNat < goldilocksModulus) :
    (addSignedCoefficient64 key digit output initial).toNat < goldilocksModulus := by
  have totalBound := (PiDECSignedDigits.fold64_correct digit
    (gather64 key (gatherBase output)) (gather64_canonical key (gatherBase output))
    0 (by decide)).1
  unfold addSignedCoefficient64
  split_ifs
  · exact add64_canonical _ _ initialBound totalBound
  · exact sub64_canonical _ _ initialBound totalBound

private theorem addSignedCoefficient64_denote (key : PreparedKey)
    (value : PiDECSignedDigits.Prepared) (digit : StoredRing)
    (success : PiDECSignedDigits.prepare digit = some value)
    (output : Fin ringDegree) (initial : UInt64)
    (initialBound : initial.toNat < goldilocksModulus) :
    (addSignedCoefficient64 key value output initial).denote =
      initial.denote + ringFMul (keyView key).get digit.get output := by
  have totalBound := (PiDECSignedDigits.fold64_correct value
    (gather64 key (gatherBase output)) (gather64_canonical key (gatherBase output))
    0 (by decide)).1
  have totalValue := PiDECSignedDigits.prepare_fold64 digit value success
    (gather64 key (gatherBase output)) (gather64_canonical key (gatherBase output))
    0 (by decide)
  simp only [zero_denote, Fin.zero_add] at totalValue
  have linear := gatherLinear_eq key digit output
  unfold addSignedCoefficient64
  split_ifs with low
  · rw [add64_denote _ _ initialBound totalBound, totalValue]
    rw [if_pos low] at linear
    rw [linear]
  · rw [sub64_denote _ _ initialBound totalBound, totalValue, Fin.sub_eq_add_neg]
    rw [if_neg low] at linear
    rw [linear]

/-- Prepare one child for all key rows. Keep zero, signed support, or
canonical native words for the general field path. -/
structure PreparedDigit where
  private mk ::
  private words : Option (Sum PiDECSignedDigits.Prepared (Vector UInt64 ringDegree))

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
  else match PiDECSignedDigits.prepare digit with
    | some value => ⟨some (.inl value)⟩
    | none => ⟨some (.inr (toWords digit))⟩

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

private theorem canonicalWord_eq (left right : UInt64)
    (leftBound : left.toNat < goldilocksModulus)
    (rightBound : right.toNat < goldilocksModulus)
    (equal : left.denote = right.denote) : left = right := by
  apply UInt64.toNat_inj.1
  have values := congrArg Fin.val equal
  change left.toNat % goldilocksModulus = right.toNat % goldilocksModulus at values
  simpa only [Nat.mod_eq_of_lt leftBound, Nat.mod_eq_of_lt rightBound] using values

private theorem eq_of_finish_get (left right : Accumulator)
    (equal : left.finish.get = right.finish.get) : left = right := by
  have wordsEqual : left.words = right.words := by
    apply Vector.ext
    intro index bound
    apply canonicalWord_eq _ _ (left.canonical ⟨index, bound⟩)
      (right.canonical ⟨index, bound⟩)
    have laneEqual := congrFun equal ⟨index, bound⟩
    simpa only [finish_get] using laneEqual
  cases left
  cases right
  cases wordsEqual
  rfl

@[inline] private def addSignedProduct (initial : Accumulator) (key : PreparedKey)
    (digit : PiDECSignedDigits.Prepared) : Accumulator where
  words := Vector.ofFn fun output =>
    addSignedCoefficient64 key digit output (initial.words.get output)
  canonical := by
    intro output
    change ((Vector.ofFn (fun lane : Fin ringDegree =>
      addSignedCoefficient64 key digit lane (initial.words.get lane)))[output.val]).toNat < _
    rw [Vector.getElem_ofFn]
    exact addSignedCoefficient64_canonical key digit output _ (initial.canonical output)

private theorem addSignedProduct_eq (initial : Accumulator) (key : PreparedKey)
    (digit : StoredRing) (value : PiDECSignedDigits.Prepared)
    (success : PiDECSignedDigits.prepare digit = some value) :
    addSignedProduct initial key value = addProduct initial key digit := by
  apply eq_of_finish_get
  funext output
  simp only [finish_get]
  change ((Vector.ofFn (fun lane : Fin ringDegree =>
    addSignedCoefficient64 key value lane (initial.words.get lane)))[output.val]).denote =
      ((Vector.ofFn (fun lane : Fin ringDegree =>
        add64 (initial.words.get lane)
          (coefficient64 key.words (toWords digit)
            (foldedCoefficients64 key.words (toWords digit)) lane)))[output.val]).denote
  rw [Vector.getElem_ofFn, Vector.getElem_ofFn,
    addSignedCoefficient64_denote key value digit success output _ (initial.canonical output),
    add64_denote _ _ (initial.canonical output)
      (coefficient64_canonical key.words (toWords digit) output key.canonical)]
  have product := coefficient64_denote (keyView key) digit output
  simp only [keyView_words] at product
  rw [product]

/-- Reuse the child's zero decision and native words across key rows. -/
def addPreparedProduct (initial : Accumulator) (key : PreparedKey)
    (digit : PreparedDigit) : Accumulator :=
  match digit.words with
  | none => initial
  | some (.inl value) => addSignedProduct initial key value
  | some (.inr words) => addWordProduct initial key words

theorem addPreparedProduct_eq (initial : Accumulator) (key : PreparedKey)
    (digit : StoredRing) :
    addPreparedProduct initial key (prepareDigit digit) =
      if ∀ lane : Fin ringDegree, digit.get lane = 0 then initial
      else addProduct initial key digit := by
  unfold prepareDigit addPreparedProduct
  simp only [allZero_iff]
  split_ifs with zero
  · rfl
  · cases success : PiDECSignedDigits.prepare digit with
    | none => rfl
    | some value => exact addSignedProduct_eq initial key digit value success

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
