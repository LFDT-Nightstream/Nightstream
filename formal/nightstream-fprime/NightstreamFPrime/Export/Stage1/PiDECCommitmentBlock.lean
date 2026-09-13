import NightstreamFPrime.Spec.AjtaiSetupV1
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.CarrierAction
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic

/-!
One fixed-key Ajtai block contribution for all sixteen PiDEC children.
The key block is materialized once and shared by the sixteen stored ring
products. No commitment accumulator, message evaluation, or IO is defined.
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
  else Vector.ofFn (ringFMul key.get child.get)

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
  · funext lane
    change (Vector.ofFn (ringFMul key.get child.get))[lane.val] = _
    rw [Vector.getElem_ofFn]

/-- Materialize the exact lazy key at one row and block. -/
def keyBlock {verifierRows messageColumns : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns) : StoredRing :=
  Vector.ofFn (setup.verifierKey row block)

/-- Stored access is the existing semantic key coordinate. -/
theorem keyBlock_value {verifierRows messageColumns : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns) :
    (keyBlock setup row block).get = setup.verifierKey row block := by
  funext lane
  change (Vector.ofFn (setup.verifierKey row block))[lane.val] = _
  rw [Vector.getElem_ofFn]

/-- Share one materialized key across the sixteen child products. -/
def products (key : StoredRing) (children : Vector StoredRing productionGlobalParams.k) :
    Vector StoredRing productionGlobalParams.k :=
  Vector.ofFn fun child => multiplyChild key (children.get child)

/-- Each stored product has the same key and child as the ring specification. -/
theorem products_value (key : StoredRing)
    (children : Vector StoredRing productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) :
    ((products key children).get child).get = ringFMul key.get (children.get child).get := by
  change ((Vector.ofFn (fun selected : Fin productionGlobalParams.k =>
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

end NightstreamFPrime.Export.Stage1.PiDECCommitmentBlock
