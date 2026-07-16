import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-!
Concrete Phi81 coefficient kernel for the paper matrix embedding.

Protocol: SuperNeo coefficient embedding (Section 5, Theorems 3 and 4).
Phase: concrete cyclotomic instantiation of the single-matrix source.
Constraint family: transformed matrix / assignment ring-product coefficients.

Owns: the closed-form Phi81 bar transform on coefficient bases; the exact
coefficient kernel obtained by multiplying that transformed basis by an
assignment basis with the independently defined Phi81 ring multiplication;
and the constant-term Kronecker law required by Theorem 3.

Does not own: proof that Rust's runtime Gram-matrix inversion returns this
closed form, production matrix construction or padding, R1CS lowering, row
removal, or constraint counts.

Emits constraints: no.

Authority boundary: the theorem is a finite, kernel-checked algebraic fact
about `Concrete.ringFMul`; it is not imported from a Rust trace or the old
circuit. Production assurance still requires a separate refinement from the
Rust bar matrix and transformed matrix cache to these definitions.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| coefficient embedding | bar transform | Phi81 basis map | `nativeBarEntry` is the closed-form linear transform |
| coefficient embedding | ring action | transformed basis / assignment basis | `phi81Kernel.weight` is an actual `ringFMul` coefficient |
| coefficient embedding | constant term | basis kernel | `phi81ConstantTermLaw` is the Kronecker identity |
| `Pi_CCS` | matrix source | all coefficient lanes | `phi81Kernel` instantiates the derived single-`M` model |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel

open Nightstream.SuperNeo.Concrete
open MatrixCoefficientSource

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

/-- Closed-form entry of the native Phi81 bar matrix. Rows are output
coefficients and columns are input coefficients. -/
def nativeBarEntry (output input : Fin ringDegree) : F :=
  if output.val = 0 then
    if input.val = 0 then 1 else 0
  else if output.val < ringMiddleDegree then
    if input.val = ringMiddleDegree - output.val ∨
        input.val = ringDegree - output.val then
      -1
    else
      0
  else if input.val = ringDegree - output.val then
    -1
  else
    0

/-- Image of one coefficient basis under the closed-form bar transform. -/
def barBasis (input : Fin ringDegree) : RingF :=
  fun output => nativeBarEntry output input

/-- Canonical constant coefficient of Phi81. -/
def constant : Fin ringDegree :=
  ⟨0, by decide⟩

/-- Exact coefficient kernel of `bar(e_row) * e_assignment` modulo Phi81. -/
def phi81Kernel : CoefficientKernel F ringDegree where
  constant := constant
  weight := fun output row assignment =>
    ringFMul (barBasis row) (ringFMonomial assignment.val 1) output

/-- Finite basis form of the paper Theorem-3 inner-product transform. -/
theorem basisConstantTerm :
    forall row assignment : Fin ringDegree,
      phi81Kernel.weight phi81Kernel.constant row assignment =
        if row = assignment then 1 else 0 := by
  decide

/-- Concrete Phi81 discharge of the constant-term kernel obligation used by
the connected single-matrix source. -/
theorem phi81ConstantTermLaw :
    ConstantTermLaw ConcreteCarrier.baseOps phi81Kernel := by
  constructor
  exact basisConstantTerm

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel
