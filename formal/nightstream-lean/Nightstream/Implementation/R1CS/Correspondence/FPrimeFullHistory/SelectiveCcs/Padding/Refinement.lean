import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial.Semantics
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Padding.Semantics

/-!
Contract: model-level specialization of the exact 13-port selective
polynomial to a public-padding zero row.

Owns: the matrix-image point with `GENERAL_SELECTOR = z[0]`, `C = z[pad]`,
and every other port zero; exact sparse-polynomial evaluation at that point;
and equivalence with the typed normalized padding obligation.

Does not own: proof that any concrete Rust row has this matrix-image point,
the source row index, matrix triplets, multiplicity, or a generated artifact.
Consequently these theorems close the mathematical specialization but are not
yet Rust-conformant row-removal authority.

Emits constraints: no.

| Stage path | Mathematical obligation | Lean result | Assurance tier |
|---|---|---|---|
| `f_prime.selective_ccs.padding.port_point` | only ports 1 and 4 are populated | `paddingPortPoint` | model-level |
| `f_prime.selective_ccs.padding.residual` | `P(point) = -(z0*zpad)` | `evaluate_paddingPortPoint` | model-level |
| `f_prime.selective_ccs.padding.zero_set` | residual vanishes iff normalized zero pin holds | `evaluate_paddingRowPoint_eq_zero_iff` | model-level |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Refinement

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Semantics

/-- Matrix-image vector of the intended public-padding specialization. -/
def paddingPortPoint (constantValue paddingValue : F) : Fin 13 -> F :=
  fun port =>
    if port = Role.generalSelector.index then constantValue
    else if port = Role.c.index then paddingValue
    else 0

@[simp] theorem paddingPortPoint_generalSelector
    (constantValue paddingValue : F) :
    paddingPortPoint constantValue paddingValue Role.generalSelector.index =
      constantValue := by
  simp [paddingPortPoint]

@[simp] theorem paddingPortPoint_c
    (constantValue paddingValue : F) :
    paddingPortPoint constantValue paddingValue Role.c.index = paddingValue := by
  simp [paddingPortPoint]

theorem paddingPortPoint_other
    (constantValue paddingValue : F) (role : Role)
    (notGeneral : role ≠ .generalSelector) (notC : role ≠ .c) :
    paddingPortPoint constantValue paddingValue role.index = 0 := by
  cases role <;> simp_all [paddingPortPoint, Role.index]

/-- Exact evaluation of all 66 sparse terms on the padding specialization. -/
theorem evaluate_paddingPortPoint (constantValue paddingValue : F) :
    evaluate (paddingPortPoint constantValue paddingValue) =
      -(constantValue * paddingValue) := by
  simp [evaluate, polynomial, terms, baseTerms, canonicalTerms,
    monomial, exponentVector,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.evaluatePolynomial,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.evaluateMonomial,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.pow,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.canonicalFinIndices,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps,
    paddingPortPoint, Role.index, Fin.zero_mul, Fin.mul_zero, Fin.one_mul,
    Fin.mul_one, Fin.zero_add, Fin.add_zero, Lean.Grind.Fin.neg_mul]

/-- Exact 13-port point induced by one typed carrier candidate and padding
coordinate. -/
def paddingRowPoint (dimensions : Dimensions)
    (candidate : Assignment dimensions.shape)
    (offset : Fin fixedPaddingWidth) : Fin 13 -> F :=
  paddingPortPoint
    (candidate (constantColumn dimensions))
    (candidate (paddingCarrierColumn dimensions offset))

theorem evaluate_paddingRowPoint (dimensions : Dimensions)
    (candidate : Assignment dimensions.shape)
    (offset : Fin fixedPaddingWidth) :
    evaluate (paddingRowPoint dimensions candidate offset) =
      -(zeroPinProduct dimensions candidate offset) := by
  exact evaluate_paddingPortPoint _ _

theorem evaluate_paddingRowPoint_eq_zero_iff (dimensions : Dimensions)
    (candidate : Assignment dimensions.shape)
    (offset : Fin fixedPaddingWidth) :
    evaluate (paddingRowPoint dimensions candidate offset) = 0 <->
      ZeroPinHolds dimensions candidate offset := by
  rw [evaluate_paddingRowPoint]
  constructor
  · intro residual
    have negated := congrArg (fun value : F => -value) residual
    simpa only [Lean.Grind.AddCommGroup.neg_neg,
      Lean.Grind.AddCommGroup.neg_zero] using negated
  · intro zeroPin
    rw [show zeroPinProduct dimensions candidate offset = 0 from zeroPin]
    rfl

/-- Canonical 270-carrier construction satisfies the exact sparse-polynomial
specialization, not merely a separately stated normalized predicate. -/
theorem canonicalAssignment_sparse_complete (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions) (offset : Fin fixedPaddingWidth) :
    evaluate (paddingRowPoint dimensions (assignment dimensions legacy) offset) = 0 := by
  exact (evaluate_paddingRowPoint_eq_zero_iff dimensions _ offset).2
    (canonicalAssignment_complete dimensions legacy offset)

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Refinement
