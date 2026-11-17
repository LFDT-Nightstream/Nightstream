import NightstreamFPrime.Gadgets.Polynomial.SparseSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.CcsTerminal

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_2.CcsTerminal

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

theorem flatConstraints_varsSatisfy
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
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

theorem output_varsSatisfy
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth}
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

/-! ### The exporter's blocks (SPEC.md section 4.4) -/

section Exporter

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}

theorem marks_nil (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface) (offset : Nat) :
    marks (Circuit.ops (circuit relation interface).main offset) = [] := by
  show marks (Sparse.Owned.opsAt (polynomial relation) (sparseInterface interface) offset) = []
  simp [Sparse.Owned.opsAt, marks, Op.marks]

/-- The rows read the fresh matrix evaluations and the gadget's own locals. -/
theorem free (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface) (offset : Nat) {live : Nat → Prop}
    (freshMatrix : ∀ matrix, Horner.KSupported (interface.freshMatrix offset matrix) live)
    (locals : ∀ column, offset ≤ column →
      column < offset + localLength (Circuit.ops (circuit relation interface).main offset) →
        live column) :
    ∀ row ∈ flatConstraints (Circuit.ops (circuit relation interface).main offset),
      row.VarsSatisfy live :=
  flatConstraints_varsSatisfy relation interface offset live freshMatrix locals

/-- The output reads only the gadget's own locals. -/
theorem output_live (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface) (offset : Nat) {live : Nat → Prop}
    (locals : ∀ column, offset ≤ column →
      column < offset + localLength (Circuit.ops (circuit relation interface).main offset) →
        live column) :
    Horner.KSupported (output relation interface offset) live :=
  output_varsSatisfy relation interface offset live locals

end Exporter

end NightstreamFPrime.Lifecycle.PiCCS.v1_2.CcsTerminal
