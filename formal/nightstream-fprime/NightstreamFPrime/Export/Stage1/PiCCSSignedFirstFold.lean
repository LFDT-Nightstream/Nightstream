import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PrefixFold
import NightstreamFPrime.Export.Stage1.PiCCSNormCache

/-! First prefix fold from the original signed-unit code reader. Nine calls
to the existing interpolation formula are prepared once; the output retains
the existing Array K type and every pair of the explicit input prefix. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSSignedFirstFold

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-- The existing negative/zero/positive code order gives exactly nine pairs.
The challenge and original PrefixFold interpolation are the only arithmetic. -/
def prepare (challenge : K) : Vector (Vector K 3) 3 :=
  Vector.ofFn fun low => Vector.ofFn fun high =>
    PrefixFold.interpolate extensionOps challenge
      (K.embed (PiCCSNormCache.signedValue low))
      (K.embed (PiCCSNormCache.signedValue high))

private theorem get_ofFn {Alpha : Type} {count : Nat}
    (values : Fin count → Alpha) (index : Fin count) :
    (Vector.ofFn values).get index = values index := by
  change (Vector.ofFn values)[index.val] = values index
  rw [Vector.getElem_ofFn]

/-- Every table entry is the original interpolation of the decoded codes. -/
theorem prepare_value (challenge : K) (low high : Fin 3) :
    ((prepare challenge).get low).get high =
      PrefixFold.interpolate extensionOps challenge
        (K.embed (PiCCSNormCache.signedValue low))
        (K.embed (PiCCSNormCache.signedValue high)) := by
  simp only [prepare, get_ofFn]

/-- Fill the existing output array by table lookup. A missing high endpoint
uses the existing zero code 1. No reader call is made for that absent endpoint. -/
def foldOne (table : Vector (Vector K 3) 3) (count : Nat)
    (codes : Nat → Fin 3) : Array K :=
  Array.ofFn fun pair : Fin ((count + 1) / 2) =>
    let low := codes (2 * pair.val)
    let high := if 2 * pair.val + 1 < count then codes (2 * pair.val + 1) else ⟨1, by decide⟩
    (table.get low).get high

private theorem decoded_getD (count : Nat) (codes : Nat → Fin 3) (index : Nat) :
    (Array.ofFn (fun entry : Fin count => K.embed (PiCCSNormCache.signedValue (codes entry.val)))).getD
        index extensionOps.zero =
      K.embed (PiCCSNormCache.signedValue
        (if index < count then codes index else ⟨1, by decide⟩)) := by
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_ofFn]
  by_cases inside : index < count
  · simp only [dif_pos inside, if_pos inside, Option.getD_some]
  · simp only [dif_neg inside, if_neg inside, Option.getD_none]
    rfl

/-- Exact equality of complete arrays, including empty input, an odd final
pair and every supplied tail coordinate. No source-value premise is needed:
the reference array is the direct decoding of the same original code reader. -/
theorem foldOne_prepare (challenge : K) (count : Nat) (codes : Nat → Fin 3) :
    foldOne (prepare challenge) count codes =
      PrefixFold.foldOne extensionOps
        (Array.ofFn (fun entry : Fin count => K.embed (PiCCSNormCache.signedValue (codes entry.val))))
        challenge := by
  unfold foldOne PrefixFold.foldOne
  apply Array.ext
  · simp only [Array.size_ofFn]
  · intro index lowBound highBound
    simp only [Array.size_ofFn] at lowBound
    simp only [Array.getElem_ofFn]
    rw [prepare_value, decoded_getD, decoded_getD]
    have lowInside : 2 * index < count := by omega
    rw [if_pos lowInside]

end NightstreamFPrime.Export.Stage1.PiCCSSignedFirstFold
