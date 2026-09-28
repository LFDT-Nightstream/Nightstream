import NightstreamFPrime.Export.Stage1.PiCCSFirstRoundPair
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
import NightstreamFPrime.Lifecycle.Types

/-! Cache the existing per-source norm cubics for signed-unit endpoints.
Other extension-field endpoints use the original constructor. Preparation and
lookup preserve every source and every coefficient in canonical source order. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSNormCache

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Lifecycle (productionShape)

abbrev PairTable := Vector (Vector (FixedPolynomial K 3) 3) 3
abbrev WeightedTable := Vector PairTable productionShape.sourceCount

private theorem get_ofFn {Alpha : Type} {count : Nat}
    (values : Fin count → Alpha) (index : Fin count) :
    (Vector.ofFn values).get index = values index := by
  change (Vector.ofFn values)[index.val] = values index
  rw [Vector.getElem_ofFn]

/-- Code order is negative, zero, positive. -/
def signedValue (code : Fin 3) : F :=
  if code.val = 0 then -1 else if code.val = 1 then 0 else 1

/-- Nine calls to the existing polynomial constructor; no new cubic formula. -/
def pairTable (_ : Unit) : PairTable :=
  Vector.ofFn fun low => Vector.ofFn fun high =>
    PiCCSFirstRoundPair.normPair extensionOps
      (K.embed (signedValue low)) (K.embed (signedValue high))

theorem pairTable_value (low high : Fin 3) :
    ((pairTable ()).get low).get high =
      PiCCSFirstRoundPair.normPair extensionOps
        (K.embed (signedValue low)) (K.embed (signedValue high)) := by
  simp only [pairTable, get_ofFn]

/-- Prepare once for the public gamma power callback. Every source has its
own scale; no nonlinear expression is applied to an aggregate of sources. -/
def prepare (powers : Nat → K) : WeightedTable :=
  let pairs := pairTable ()
  Vector.ofFn fun source => Vector.ofFn fun low => Vector.ofFn fun high =>
    FixedPolynomial.scale extensionOps.toOps (powers source.val)
      ((pairs.get low).get high)

/-- Public lookup also serves the existing individual-pair runner. -/
def weightedLookup (table : WeightedTable) (source : Fin productionShape.sourceCount)
    (low high : Fin 3) : FixedPolynomial K 3 :=
  (((table.get source).get low).get high)

theorem weightedLookup_prepare (powers : Nat → K) (source : Fin productionShape.sourceCount)
    (low high : Fin 3) :
    weightedLookup (prepare powers) source low high =
      FixedPolynomial.scale extensionOps.toOps (powers source.val)
        (PiCCSFirstRoundPair.normPair extensionOps
          (K.embed (signedValue low)) (K.embed (signedValue high))) := by
  simp only [weightedLookup, prepare, get_ofFn, pairTable_value]

/-- Recognize only exact embedded signed-unit values. Other K values are
left for the original constructor. Zero keeps its existing table code. -/
def signedCode? (value : K) : Option (Fin 3) :=
  if value = K.embed (signedValue ⟨1, by decide⟩) then some ⟨1, by decide⟩
  else if value = K.embed (signedValue ⟨2, by decide⟩) then some ⟨2, by decide⟩
  else if value = K.embed (signedValue ⟨0, by decide⟩) then some ⟨0, by decide⟩
  else none

theorem signedCode?_some (value : K) (code : Fin 3)
    (found : signedCode? value = some code) :
    K.embed (signedValue code) = value := by
  unfold signedCode? at found
  split at found
  next matched =>
    rw [← Option.some.inj found]
    exact matched.symm
  next _ =>
    split at found
    next matched =>
      rw [← Option.some.inj found]
      exact matched.symm
    next _ =>
      split at found
      next matched =>
        rw [← Option.some.inj found]
        exact matched.symm
      next _ => cases found

/-- Total lookup: the fallback is the same original scaled cubic. -/
def cachedLookup (table : WeightedTable) (powers : Nat → K)
    (source : Fin productionShape.sourceCount) (low high : K) : FixedPolynomial K 3 :=
  match signedCode? low, signedCode? high with
  | some lowCode, some highCode => weightedLookup table source lowCode highCode
  | _, _ =>
      FixedPolynomial.scale extensionOps.toOps (powers source.val)
        (PiCCSFirstRoundPair.normPair extensionOps low high)

/-- This equality holds for arbitrary K endpoints, with no signedness premise. -/
theorem cachedLookup_prepare (powers : Nat → K) (source : Fin productionShape.sourceCount)
    (low high : K) :
    cachedLookup (prepare powers) powers source low high =
      FixedPolynomial.scale extensionOps.toOps (powers source.val)
        (PiCCSFirstRoundPair.normPair extensionOps low high) := by
  cases lowFound : signedCode? low with
  | none => simp only [cachedLookup, lowFound]
  | some lowCode =>
      cases highFound : signedCode? high with
      | none => simp only [cachedLookup, lowFound, highFound]
      | some highCode =>
          simp only [cachedLookup, lowFound, highFound]
          rw [weightedLookup_prepare,
            signedCode?_some low lowCode lowFound,
            signedCode?_some high highCode highFound]

/-- Preserve every source cubic and the existing canonical source order.
The already computed message fields are sufficient for this total kernel. -/
def sourceNorm (table : WeightedTable) (powers : Nat → K)
    (low high : ProtocolPolynomial.OutputMessage K productionShape) :
    FixedPolynomial K 3 :=
  FixedPolynomial.sum extensionOps.toOps
    (canonicalFinIndices productionShape.sourceCount) fun source =>
      cachedLookup table powers source
        (low.sourceAssignment source) (high.sourceAssignment source)

/-- Exact coefficient equality to the current source norm constructor. -/
theorem sourceNorm_eq (powers : Nat → K)
    (low high : ProtocolPolynomial.OutputMessage K productionShape) :
    sourceNorm (prepare powers) powers low high =
      PiCCSFirstRoundPair.normPolynomialWithPowers extensionOps powers low high := by
  change FixedPolynomial.sum extensionOps.toOps
      (canonicalFinIndices productionShape.sourceCount)
      (fun source => cachedLookup (prepare powers) powers source
        (low.sourceAssignment source) (high.sourceAssignment source)) =
    FixedPolynomial.sum extensionOps.toOps
      (canonicalFinIndices productionShape.sourceCount)
      (fun source => FixedPolynomial.scale extensionOps.toOps (powers source.val)
        (PiCCSFirstRoundPair.normPair extensionOps
          (low.sourceAssignment source) (high.sourceAssignment source)))
  simp only [cachedLookup_prepare]

end NightstreamFPrime.Export.Stage1.PiCCSNormCache
