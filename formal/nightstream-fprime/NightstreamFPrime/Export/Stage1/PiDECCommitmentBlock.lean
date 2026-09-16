import NightstreamFPrime.Spec.AjtaiSetupV1
import NightstreamFPrime.Export.NativeAjtaiChaCha
import NightstreamFPrime.Export.Stage1.PiDECNativeProduct
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.CarrierAction
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic

/-!
One fixed-key Ajtai block contribution for all sixteen PiDEC children.
The key block is materialized once and shared by the sixteen stored ring
products. These products can be materialized or added directly to native row
accumulators. No message evaluation or IO is defined.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECCommitmentBlock

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)

/-- A zero digit contributes the exact zero ring without multiplication. -/
def multiplyChild (key child : StoredRing) : StoredRing :=
  if ∀ lane : Fin ringDegree, child.get lane = 0 then Vector.replicate ringDegree 0
  else PiDECNativeProduct.multiply key child

/-- Zero omission preserves the complete semantic key product. -/
theorem multiplyChild_value (key child : StoredRing) :
    (multiplyChild key child).get = ringFMul key.get child.get := by
  unfold multiplyChild
  split_ifs with zero
  · have childZero : child.get = ringFZero := funext zero
    rw [childZero, CarrierAction.ringFMul_zero_right]
    funext lane
    change (Vector.replicate ringDegree (0 : F))[lane.val] = 0
    rw [Vector.getElem_replicate]
  · exact PiDECNativeProduct.multiply_value key child

/-- Materialize the exact lazy key at one row and block. -/
def keyBlock {verifierRows messageColumns : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns) : StoredRing :=
  NightstreamFPrime.Export.NativeAjtaiChaCha.keyBlock setup row block

/-- Stored access is the existing semantic key coordinate. -/
theorem keyBlock_value {verifierRows messageColumns : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns) :
    (keyBlock setup row block).get = setup.verifierKey row block :=
  NightstreamFPrime.Export.NativeAjtaiChaCha.keyBlock_value setup row block

/-- Share one materialized key across the stored block products. -/
def products {count : Nat} (key : StoredRing) (children : Vector StoredRing count) :
    Vector StoredRing count :=
  Vector.ofFn fun child => multiplyChild key (children.get child)

/-- Each stored product has the same key and child as the ring specification. -/
theorem products_value {count : Nat} (key : StoredRing)
    (children : Vector StoredRing count)
    (child : Fin count) :
    ((products key children).get child).get = ringFMul key.get (children.get child).get := by
  change ((Vector.ofFn (fun selected : Fin count =>
    multiplyChild key (children.get selected)))[child.val]).get = _
  rw [Vector.getElem_ofFn, multiplyChild_value]

/-- Compute all sixteen contributions at one exact key row and block.
The existing typed setup fixes the key; no key coefficients are supplied. -/
def contributions {verifierRows messageColumns : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns)
    (children : Vector StoredRing productionGlobalParams.k) :
    Vector StoredRing productionGlobalParams.k :=
  products (keyBlock setup row block) children

/-- Each returned ring is the exact semantic key-block product for that
same child. No norm, opening, expected output, or runtime premise is used. -/
theorem contributions_value {verifierRows messageColumns : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns)
    (children : Vector StoredRing productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) :
    ((contributions setup row block children).get child).get =
      ringFMul (setup.verifierKey row block) (children.get child).get := by
  simp only [contributions, products_value, keyBlock_value]

/-- Add one exact key row/block into all native child accumulators. Zero
children keep their initial sum; no field-valued product is materialized. -/
def accumulateContributions {verifierRows messageColumns : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns)
    (children : Vector StoredRing productionGlobalParams.k)
    (initial : Vector PiDECNativeProduct.Accumulator productionGlobalParams.k) :
    Vector PiDECNativeProduct.Accumulator productionGlobalParams.k :=
  let key := PiDECNativeProduct.prepareKey (keyBlock setup row block)
  Vector.ofFn fun child =>
    if ∀ lane : Fin ringDegree, (children.get child).get lane = 0 then initial.get child
    else (initial.get child).addProduct key (children.get child)

theorem accumulateContributions_value {verifierRows messageColumns : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns)
    (children : Vector StoredRing productionGlobalParams.k)
    (initial : Vector PiDECNativeProduct.Accumulator productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) :
    ((accumulateContributions setup row block children initial).get child).finish.get =
      ringFAdd (initial.get child).finish.get
        (ringFMul (setup.verifierKey row block) (children.get child).get) := by
  change ((Vector.ofFn (fun selected : Fin productionGlobalParams.k =>
    if ∀ lane : Fin ringDegree, (children.get selected).get lane = 0 then initial.get selected
    else (initial.get selected).addProduct (PiDECNativeProduct.prepareKey (keyBlock setup row block))
      (children.get selected)))[child.val]).finish.get = _
  rw [Vector.getElem_ofFn]
  split_ifs with zero
  · have childZero : (children.get child).get = ringFZero := funext zero
    rw [childZero, CarrierAction.ringFMul_zero_right]
    funext lane
    exact (NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseLaws.add_zero _).symm
  · rw [PiDECNativeProduct.Accumulator.addProduct_value, keyBlock_value]

/-- Reuse all child preparations for one exact key row and block. -/
def accumulatePreparedContributions {verifierRows messageColumns : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns)
    (children : Vector PiDECNativeProduct.PreparedDigit productionGlobalParams.k)
    (initial : Vector PiDECNativeProduct.Accumulator productionGlobalParams.k) :
    Vector PiDECNativeProduct.Accumulator productionGlobalParams.k :=
  let key := PiDECNativeProduct.prepareKey (keyBlock setup row block)
  Vector.ofFn fun child => (initial.get child).addPreparedProduct key (children.get child)

/-- Preparing children before the row scan preserves the complete native
result, including the unchanged initial accumulator for every zero child. -/
theorem accumulatePreparedContributions_eq {verifierRows messageColumns : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns)
    (children : Vector StoredRing productionGlobalParams.k)
    (initial : Vector PiDECNativeProduct.Accumulator productionGlobalParams.k) :
    accumulatePreparedContributions setup row block
        (children.map PiDECNativeProduct.prepareDigit) initial =
      accumulateContributions setup row block children initial := by
  apply Vector.ext
  intro index bound
  simp only [accumulatePreparedContributions, accumulateContributions, Vector.getElem_ofFn]
  have prepared : (children.map PiDECNativeProduct.prepareDigit).get ⟨index, bound⟩ =
      PiDECNativeProduct.prepareDigit (children.get ⟨index, bound⟩) := by
    change (children.map PiDECNativeProduct.prepareDigit)[index] = _
    rw [Vector.getElem_map]
    rfl
  rw [prepared, PiDECNativeProduct.Accumulator.addPreparedProduct_eq]

end NightstreamFPrime.Export.Stage1.PiDECCommitmentBlock
