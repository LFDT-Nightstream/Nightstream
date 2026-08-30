import NightstreamFPrime.Gadgets.Polynomial.SparseSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal

/-!
Owns variable-support propagation for the production PiCCS CCS terminal.
It changes no polynomial, circuit, or row.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-- Exact support propagation through the production two-row CCS terminal. -/
theorem flatConstraints_varsSatisfy
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface) (offset : Nat) (allowed : Nat → Prop)
    (freshMatrixSupport : ∀ matrix,
      Horner.KSupported (interface.freshMatrix offset matrix) allowed)
    (localSupport : ∀ index,
      offset ≤ index →
      index < offset + localLength
        (Circuit.ops (circuit relation interface).main offset) →
      allowed index) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (circuit relation interface).main offset),
      expression.VarsSatisfy allowed := by
  have supported := Sparse.Owned.flatConstraints_varsSatisfy
    (polynomial relation) (sparseInterface interface) offset allowed
    (by intro matrix; simpa [sparseInterface] using freshMatrixSupport matrix)
    (by
      intro index lower upper
      apply localSupport index lower
      simpa [circuit] using upper)
  simpa [circuit] using supported

/-- The two-cell CCS residual output preserves the exact row support. -/
theorem output_varsSatisfy
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface) (offset : Nat) (allowed : Nat → Prop)
    (localSupport : ∀ index,
      offset ≤ index →
      index < offset + localLength
        (Circuit.ops (circuit relation interface).main offset) →
      allowed index) :
    Horner.KSupported (output relation interface offset) allowed := by
  have supported := Sparse.Owned.output_varsSatisfy
    (polynomial relation) (sparseInterface interface) offset allowed
    (by
      intro index lower upper
      apply localSupport index lower
      simpa [circuit] using upper)
  simpa [output, circuit] using supported

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal
