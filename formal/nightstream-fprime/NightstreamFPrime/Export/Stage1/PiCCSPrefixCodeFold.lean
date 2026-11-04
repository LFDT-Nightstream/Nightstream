import Batteries.Data.Fin.Coding
import NightstreamFPrime.Export.Stage1.PiCCSPrefixNorm

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSPrefixCodeFold

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

def decode {count : Nat} (values : Vector K count)
    (codes : Array (Fin count)) : Array K :=
  codes.map values.get

def pairedTable {count : Nat} (values : Vector K count) (challenge : K) :
    Vector K (count * count) :=
  Vector.ofFn fun code =>
    let pair : Fin count × Fin count := Fin.decodeProd code
    PrefixFold.interpolate extensionOps challenge
      (values.get pair.1) (values.get pair.2)

def pairedCodes {count : Nat} (zeroCode : Fin count)
    (codes : Array (Fin count)) : Array (Fin (count * count)) :=
  Array.ofFn fun pair : Fin ((codes.size + 1) / 2) =>
    Fin.encodeProd
      (codes.getD (2 * pair.val) zeroCode,
       codes.getD (2 * pair.val + 1) zeroCode)

theorem pairedTable_pair {count : Nat} (values : Vector K count)
    (challenge : K) (low high : Fin count) :
    (pairedTable values challenge).get (Fin.encodeProd (low, high)) =
      PrefixFold.interpolate extensionOps challenge (values.get low) (values.get high) := by
  change (Vector.ofFn _)[(Fin.encodeProd (low, high)).val] = _
  rw [Vector.getElem_ofFn, Fin.decodeProd_encodeProd]

private theorem decoded_getD {count : Nat} (values : Vector K count)
    (zeroCode : Fin count) (zeroValue : values.get zeroCode = K.zero)
    (codes : Array (Fin count)) (index : Nat) :
    (decode values codes).getD index K.zero = values.get (codes.getD index zeroCode) := by
  simp only [decode, Array.getD_eq_getD_getElem?, Array.getElem?_map]
  cases loaded : codes[index]? with
  | none =>
      simpa only [loaded, Option.map_none, Option.getD_none] using zeroValue.symm
  | some code =>
      simp only [Option.map_some, Option.getD_some]

theorem decode_pairedCodes {count : Nat} (values : Vector K count)
    (zeroCode : Fin count) (zeroValue : values.get zeroCode = K.zero)
    (codes : Array (Fin count)) (challenge : K) :
    decode (pairedTable values challenge) (pairedCodes zeroCode codes) =
      PrefixFold.foldOne extensionOps (decode values codes) challenge := by
  unfold decode pairedCodes PrefixFold.foldOne
  apply Array.ext
  · simp only [Array.size_map, Array.size_ofFn]
  · intro index leftBound rightBound
    simp only [Array.getElem_map, Array.getElem_ofFn]
    rw [pairedTable_pair]
    change _ = PrefixFold.interpolate extensionOps challenge
      ((decode values codes).getD (2 * index) K.zero)
      ((decode values codes).getD (2 * index + 1) K.zero)
    rw [decoded_getD values zeroCode zeroValue codes (2 * index),
      decoded_getD values zeroCode zeroValue codes (2 * index + 1)]

private theorem interpolate_zero (challenge : K) :
    PrefixFold.interpolate extensionOps challenge K.zero K.zero = K.zero := by
  change PrefixFold.interpolate extensionOps challenge extensionOps.zero extensionOps.zero =
    extensionOps.zero
  unfold PrefixFold.interpolate InterpolationOps.sub
  rw [extensionLaws.add_neg, extensionLaws.mul_zero, extensionLaws.add_zero]

theorem pairedTable_zero {count : Nat} (values : Vector K count)
    (zeroCode : Fin count) (zeroValue : values.get zeroCode = K.zero) (challenge : K) :
    (pairedTable values challenge).get (Fin.encodeProd (zeroCode, zeroCode)) = K.zero := by
  rw [pairedTable_pair, zeroValue]
  exact interpolate_zero challenge

theorem firstTable_zero (challenge : K) :
    (PiCCSPrefixNorm.values challenge).get
      (PiCCSPrefixNorm.pairCode ⟨1, by decide⟩ ⟨1, by decide⟩) = K.zero := by
  rw [PiCCSPrefixNorm.values_pairCode]
  change PrefixFold.interpolate extensionOps challenge K.zero K.zero = K.zero
  exact interpolate_zero challenge

theorem decode_secondFold (firstChallenge secondChallenge : K) (codes : Array (Fin 9)) :
    decode (pairedTable (PiCCSPrefixNorm.values firstChallenge) secondChallenge)
        (pairedCodes (PiCCSPrefixNorm.pairCode ⟨1, by decide⟩ ⟨1, by decide⟩) codes) =
      PrefixFold.foldOne extensionOps (decode (PiCCSPrefixNorm.values firstChallenge) codes)
        secondChallenge := by
  exact decode_pairedCodes (PiCCSPrefixNorm.values firstChallenge)
    (PiCCSPrefixNorm.pairCode ⟨1, by decide⟩ ⟨1, by decide⟩)
    (firstTable_zero firstChallenge) codes secondChallenge

theorem secondTable_zero (firstChallenge secondChallenge : K) :
    (pairedTable (PiCCSPrefixNorm.values firstChallenge) secondChallenge).get
      ⟨40, by decide⟩ = K.zero := by
  exact pairedTable_zero (PiCCSPrefixNorm.values firstChallenge)
    (PiCCSPrefixNorm.pairCode ⟨1, by decide⟩ ⟨1, by decide⟩)
    (firstTable_zero firstChallenge) secondChallenge

def quadCode (codes : Nat → Fin 3) (index : Nat) : Fin 81 :=
  Fin.encodeProd
    (PiCCSPrefixNorm.pairCode (codes (4 * index)) (codes (4 * index + 1)),
     PiCCSPrefixNorm.pairCode (codes (4 * index + 2)) (codes (4 * index + 3)))

theorem quadCode_val (codes : Nat → Fin 3) (index : Nat) :
    (quadCode codes index).val =
      (PiCCSPrefixNorm.pairCode (codes (4 * index)) (codes (4 * index + 1))).val * 9 +
      (PiCCSPrefixNorm.pairCode (codes (4 * index + 2)) (codes (4 * index + 3))).val := by
  change 9 * _ + _ = _ * 9 + _
  omega

theorem code_byte (code : Fin 81) : code.val.toUInt8.toNat = code.val := by
  change (UInt8.ofNat code.val).toNat = code.val
  exact UInt8.toNat_ofNat_of_lt' (Nat.lt_trans code.isLt (by decide))

def quadCodes (codes : Nat → Fin 3) (start count : Nat) : Array (Fin 81) :=
  Array.ofFn fun index : Fin count => quadCode codes (start + index.val)

theorem quadCodes_size (codes : Nat → Fin 3) (start count : Nat) :
    (quadCodes codes start count).size = count := by
  simp only [quadCodes, Array.size_ofFn]

theorem quadCode_value (firstChallenge secondChallenge : K)
    (codes : Nat → Fin 3) (index : Nat) :
    (pairedTable (PiCCSPrefixNorm.values firstChallenge) secondChallenge).get
        (quadCode codes index) =
      PrefixFold.interpolate extensionOps secondChallenge
        (PrefixFold.interpolate extensionOps firstChallenge
          (K.embed (PiCCSNormCache.signedValue (codes (4 * index))))
          (K.embed (PiCCSNormCache.signedValue (codes (4 * index + 1)))))
        (PrefixFold.interpolate extensionOps firstChallenge
          (K.embed (PiCCSNormCache.signedValue (codes (4 * index + 2))))
          (K.embed (PiCCSNormCache.signedValue (codes (4 * index + 3))))) := by
  rw [quadCode, pairedTable_pair, PiCCSPrefixNorm.values_pairCode,
    PiCCSPrefixNorm.values_pairCode]

theorem decode_quadCodes_twoFolds (firstChallenge secondChallenge : K)
    (codes : Nat → Fin 3) (start count : Nat) :
    decode (pairedTable (PiCCSPrefixNorm.values firstChallenge) secondChallenge)
        (quadCodes codes start count) =
      PrefixFold.foldOne extensionOps
        (PrefixFold.foldOne extensionOps
          (Array.ofFn fun index : Fin (4 * count) =>
            K.embed (PiCCSNormCache.signedValue (codes (4 * start + index.val))))
          firstChallenge)
        secondChallenge := by
  apply Array.ext
  · simp only [decode, quadCodes, Array.size_map, PrefixFold.foldOne_size,
      Array.size_ofFn]
    omega
  · intro index leftBound rightBound
    have inside : index < count := by
      simpa only [decode, quadCodes, Array.size_map, Array.size_ofFn] using leftBound
    have lowLow : 2 * (2 * index) < 4 * count := by omega
    have lowHigh : 2 * (2 * index) + 1 < 4 * count := by omega
    have highLow : 2 * (2 * index + 1) < 4 * count := by omega
    have highHigh : 2 * (2 * index + 1) + 1 < 4 * count := by omega
    rw [Array.getElem_eq_getD extensionOps.zero, Array.getElem_eq_getD extensionOps.zero,
      PrefixFold.foldOne_getD extensionOps extensionLaws,
      PrefixFold.foldOne_getD extensionOps extensionLaws,
      PrefixFold.foldOne_getD extensionOps extensionLaws]
    simp only [decode, quadCodes, Array.getD_eq_getD_getElem?, Array.getElem?_map,
      Array.getElem?_ofFn, dite_eq_left inside, dite_eq_left lowLow, dite_eq_left lowHigh,
      dite_eq_left highLow, dite_eq_left highHigh, Option.map_some, Option.getD_some]
    rw [quadCode_value]
    congr 5 <;> omega

end NightstreamFPrime.Export.Stage1.PiCCSPrefixCodeFold
