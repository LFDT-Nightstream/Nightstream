import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanDomain
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormRange

/-!
Concrete strict-norm residual tables for the paper-level joint `Pi_CCS` model.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: construction of the uncompressed norm block `NC(X, C)`.
Constraint family: one strict `b = 2` norm obligation per source and Boolean
assignment coordinate.

Owns: typed Boolean-domain assignments, canonical base-field cubic residual
tables, their pointwise equivalence with centered strict norm, and batch
composition for all `K + k` sources.

Does not own: a kernel proof of Goldilocks primality/no-zero-divisors,
external numeric row or bit order, embedding the base residual into the
extension-field joint polynomial, CCS, carried evaluation, SumCheck,
Fiat--Shamir, Rust, R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: the strict centered norm predicate determines acceptance.
The cubic is derived from that predicate in `NormRange`; this file only places
the resulting residual on the one shared `BooleanDomain`. `BaseZeroAgreement`
states merely that the interpolation interface uses the concrete base zero; it
does not supply a semantic iff or evaluator.

| Code owner | Paper object | Mathematical obligation | Proven result |
|---|---|---|---|
| `BooleanAssignment` | one source `z_i` on `{0,1}^ell` | typed coordinate values | shared low/high order |
| `residualTable` | one `NC` source factor | `(z+1)z(z-1)` at every vertex | exact canonical leaves |
| `residualTable_allEntriesZero_iff_strictNormBounded` | Lemma 7 Item 2 | every centered magnitude is `< 2` | conditional only on no zero divisors |
| `strictNormBounded_iff_orderedValues_normBounded` | concrete semantic norm | canonical finite assignment list | exact predicate equivalence |
| `SourceBatch.allResidualTablesZero_iff_allStrictNormBounded` | all `K+k` sources | every source has fresh norm | typed batch composition |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormResidualTable

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- One source assignment indexed by the shared semantic Boolean domain. -/
abbrev BooleanAssignment (variables : Nat) := BooleanVertex variables -> F

/-- The only interface agreement needed to compare table-zero with concrete
base-field zero. Arithmetic inside the residual itself is not caller-owned. -/
structure BaseZeroAgreement (ops : InterpolationOps F) : Prop where
  zero_eq : ops.zero = (0 : F)

/-- The authoritative pointwise strict `b = 2` centered norm. -/
def StrictNormBounded
    {variables : Nat}
    (assignment : BooleanAssignment variables) : Prop :=
  ∀ vertex, centeredMagnitude (assignment vertex) < 2

/-- Canonical finite assignment serialization internal to the paper model. -/
def orderedValues
    {variables : Nat}
    (assignment : BooleanAssignment variables) : List F :=
  (BooleanVertex.all variables).map assignment

/-- The typed pointwise predicate is exactly the concrete semantic list norm
for the canonical paper-model order. This does not identify that order with a
production integer/bit serialization. -/
theorem strictNormBounded_iff_orderedValues_normBounded
    {variables : Nat}
    (assignment : BooleanAssignment variables) :
    StrictNormBounded assignment ↔ normBounded 2 (orderedValues assignment) := by
  constructor
  · intro bounded value member
    rw [orderedValues] at member
    rcases List.mem_map.mp member with ⟨vertex, _, rfl⟩
    exact bounded vertex
  · intro bounded vertex
    exact bounded (assignment vertex) (by
      rw [orderedValues]
      exact List.mem_map.mpr
        ⟨vertex, BooleanVertex.mem_all vertex, rfl⟩)

/-- Canonical cubic residual table for one source. -/
def residualTable
    {variables : Nat}
    (assignment : BooleanAssignment variables) : BooleanTable F variables :=
  BooleanTable.tabulate fun vertex =>
    NormRange.cubicResidual (assignment vertex)

/-- The exact leaf formula and order for one norm table. -/
theorem residualTable_entries_eq
    {variables : Nat}
    (assignment : BooleanAssignment variables) :
    (residualTable assignment).entries =
      (BooleanVertex.all variables).map fun vertex =>
        NormRange.cubicResidual (assignment vertex) := by
  exact BooleanTable.entries_tabulate _

/-- The constructed norm table is zero exactly when the independently defined
strict centered norm holds at every typed Boolean coordinate. -/
theorem residualTable_allEntriesZero_iff_strictNormBounded
    (ops : InterpolationOps F)
    (zeroAgreement : BaseZeroAgreement ops)
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {variables : Nat}
    (assignment : BooleanAssignment variables) :
    (residualTable assignment).AllEntriesZero ops ↔
      StrictNormBounded assignment := by
  unfold residualTable
  rw [BooleanTable.tabulate_allEntriesZero_iff]
  constructor
  · intro allZero vertex
    apply (NormRange.cubicResidual_eq_zero_iff_strictNormTwo
      noZeroDivisors (assignment vertex)).mp
    rw [← zeroAgreement.zero_eq]
    exact allZero vertex
  · intro bounded vertex
    rw [zeroAgreement.zero_eq]
    exact (NormRange.cubicResidual_eq_zero_iff_strictNormTwo
      noZeroDivisors (assignment vertex)).mpr (bounded vertex)

/-- Equivalent list-level statement using the concrete semantic
`normBounded 2` predicate on the canonical paper-model serialization. -/
theorem residualTable_allEntriesZero_iff_normBounded
    (ops : InterpolationOps F)
    (zeroAgreement : BaseZeroAgreement ops)
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {variables : Nat}
    (assignment : BooleanAssignment variables) :
    (residualTable assignment).AllEntriesZero ops ↔
      normBounded 2 (orderedValues assignment) := by
  exact Iff.trans
    (residualTable_allEntriesZero_iff_strictNormBounded
      ops zeroAgreement noZeroDivisors assignment)
    (strictNormBounded_iff_orderedValues_normBounded assignment)

/-- The canonical alpha polynomial of the concrete norm table is
coefficient-zero exactly when the strict centered norm holds. This remains a
base-field theorem; extension-field placement is separate. -/
theorem residualPolynomial_coefficientZero_iff_strictNormBounded
    (ops : InterpolationOps F)
    (zeroLaws : InterpolationZeroLaws ops)
    (zeroAgreement : BaseZeroAgreement ops)
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : Shape}
    (assignment : BooleanAssignment shape.cubeVariables) :
    AlphaPolynomial.CoefficientZero ops.toOps
        ((residualTable assignment).toAlphaPolynomial ops) ↔
      StrictNormBounded assignment := by
  exact Iff.trans
    (BooleanTable.toAlphaPolynomial_coefficientZero_iff_allEntriesZero
      ops zeroLaws (residualTable assignment))
    (residualTable_allEntriesZero_iff_strictNormBounded
      ops zeroAgreement noZeroDivisors assignment)

/-- Every source assignment in the joint `K + k` norm family. -/
structure SourceBatch (shape : Shape) where
  assignments : Fin shape.sourceCount ->
    BooleanAssignment shape.cubeVariables

namespace SourceBatch

/-- Exact norm table family expected by `TableResidualData.norm`. -/
def residualTables
    {shape : Shape}
    (batch : SourceBatch shape) :
    Fin shape.sourceCount -> BooleanTable F shape.cubeVariables :=
  fun source => residualTable (batch.assignments source)

/-- Independent strict-norm truth for every source. -/
def AllStrictNormBounded
    {shape : Shape}
    (batch : SourceBatch shape) : Prop :=
  ∀ source, StrictNormBounded (batch.assignments source)

/-- All norm residual tables are zero iff every typed source satisfies the
authoritative strict norm. -/
theorem allResidualTablesZero_iff_allStrictNormBounded
    (ops : InterpolationOps F)
    (zeroAgreement : BaseZeroAgreement ops)
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : Shape}
    (batch : SourceBatch shape) :
    (∀ source, (batch.residualTables source).AllEntriesZero ops) ↔
      batch.AllStrictNormBounded := by
  constructor
  · intro allZero source
    exact (residualTable_allEntriesZero_iff_strictNormBounded
      ops zeroAgreement noZeroDivisors (batch.assignments source)).mp
        (allZero source)
  · intro allBounded source
    exact (residualTable_allEntriesZero_iff_strictNormBounded
      ops zeroAgreement noZeroDivisors (batch.assignments source)).mpr
        (allBounded source)

end SourceBatch

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormResidualTable
