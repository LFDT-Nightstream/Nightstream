import NightstreamFPrime.Export.Stage1.PiDECCommitmentBlock

/-!
One fresh-assignment block contribution to a native commitment accumulator.
The supplied setup owns the exact key; the existing product kernel owns
arithmetic and zero-digit handling. Range coverage and IO belong to callers.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.FreshCommitmentBlock

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)

/-- Add one prepared message block at the unchanged setup row and block. -/
def accumulatePrepared {verifierRows messageColumns : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns)
    (digit : PiDECNativeProduct.PreparedDigit)
    (initial : PiDECNativeProduct.Accumulator) : PiDECNativeProduct.Accumulator :=
  let key := PiDECNativeProduct.prepareKey (PiDECCommitmentBlock.keyBlock setup row block)
  initial.addPreparedProduct key digit

/-- Every finished coefficient is the exact prior sum plus the semantic
key-block product. No signed-unit or nonzero premise is required. -/
theorem accumulatePrepared_value {verifierRows messageColumns : Nat}
    (setup : AjtaiSetupV1.Setup verifierRows messageColumns)
    (row : Fin verifierRows) (block : Fin messageColumns)
    (digit : StoredRing) (initial : PiDECNativeProduct.Accumulator) :
    (accumulatePrepared setup row block (PiDECNativeProduct.prepareDigit digit)
      initial).finish.get =
      ringFAdd initial.finish.get (ringFMul (setup.verifierKey row block) digit.get) := by
  unfold accumulatePrepared
  rw [PiDECNativeProduct.Accumulator.addPreparedProduct_eq]
  split_ifs with zero
  · have digitZero : digit.get = ringFZero := funext zero
    rw [digitZero, CarrierAction.ringFMul_zero_right]
    funext lane
    exact (NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseLaws.add_zero _).symm
  · rw [PiDECNativeProduct.Accumulator.addProduct_value, PiDECCommitmentBlock.keyBlock_value]

end NightstreamFPrime.Export.Stage1.FreshCommitmentBlock
