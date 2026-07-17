import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row.Decoder
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row.Boolean
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Padding.Refinement

/-!
Coefficient-based classification of one selective-CCS public-padding row.

Owns: the exact decoded sparse-row shape with unit terms only at
`GENERAL_SELECTOR = z[constant]` and `C = z[padding]`; its matrix action; and
the resulting `-(z[constant] * z[padding])` residual.

Does not own: any concrete generated row, Rust extraction, row indices,
multiplicity, full relation satisfaction, constraint necessity, or row
removal.

Emits constraints: no.

| Stage path | Mathematical obligation | Evidence consumed | Lean result |
|---|---|---|---|
| `f_prime.selective_ccs.padding.artifact.shape` | ports 1/4 are the two exact unit terms; every other port is empty | decoded sparse coefficients | `ValidatedPaddingRow` |
| `f_prime.selective_ccs.padding.artifact.action` | matrix action is the independent padding point | validated shape | `rowPoint_eq_paddingPortPoint` |
| `f_prime.selective_ccs.padding.artifact.residual` | full 27-term polynomial is `-(z0*zpad)` | independent polynomial theorem | `residual_eq_neg_product` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.ArtifactRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Decoder
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Refinement

/-- Exact coefficient shape of one public-padding row. Family metadata is not
part of this predicate. -/
def IsPaddingAt (row : DecodedRow)
    (constantColumn paddingColumn : Fin row.columns) : Prop :=
  (row.port Role.generalSelector.index).terms =
      [{ column := constantColumn, coefficient := 1 }] ∧
    (row.port Role.c.index).terms =
      [{ column := paddingColumn, coefficient := 1 }] ∧
    ∀ port : Fin 13,
      port ≠ Role.generalSelector.index →
      port ≠ Role.c.index →
      (row.port port).terms = []

instance (row : DecodedRow)
    (constantColumn paddingColumn : Fin row.columns) :
    Decidable (IsPaddingAt row constantColumn paddingColumn) := by
  unfold IsPaddingAt
  infer_instance

/-- Proof-carrying padding classification derived only from decoded
coefficients. -/
structure ValidatedPaddingRow (row : DecodedRow) where
  constantColumn : Fin row.columns
  paddingColumn : Fin row.columns
  shape : IsPaddingAt row constantColumn paddingColumn

def validatePaddingAt (row : DecodedRow)
    (constantColumn paddingColumn : Fin row.columns) :
    Option (ValidatedPaddingRow row) :=
  if shape : IsPaddingAt row constantColumn paddingColumn then
    some ⟨constantColumn, paddingColumn, shape⟩
  else
    none

theorem rowPoint_eq_paddingPortPoint
    (row : DecodedRow) (validated : ValidatedPaddingRow row)
    (assignment : Fin row.columns → F) :
    rowPoint row assignment =
      paddingPortPoint
        (assignment validated.constantColumn)
        (assignment validated.paddingColumn) := by
  funext port
  by_cases generalPort : port = Role.generalSelector.index
  · subst port
    simp only [rowPoint, action]
    rw [validated.shape.1]
    simp [paddingPortPoint, Role.index, Fin.one_mul, Fin.zero_add]
  · by_cases cPort : port = Role.c.index
    · subst port
      simp only [rowPoint, action]
      rw [validated.shape.2.1]
      simp [paddingPortPoint, Role.index, Fin.one_mul, Fin.zero_add]
    · simp only [rowPoint, action]
      rw [validated.shape.2.2 port generalPort cPort]
      have generalPortValue : port ≠ (1 : Fin 13) := by
        intro equal
        apply generalPort
        simpa [Role.index] using equal
      have cPortValue : port ≠ (4 : Fin 13) := by
        intro equal
        apply cPort
        simpa [Role.index] using equal
      simp [paddingPortPoint, generalPortValue, cPortValue]

/-- The exact decoded row realizes the independently proved public-padding
specialization of the complete selective polynomial. -/
theorem residual_eq_neg_product
    (row : DecodedRow) (validated : ValidatedPaddingRow row)
    (assignment : Fin row.columns → F) :
    residual row assignment =
      -(assignment validated.constantColumn *
        assignment validated.paddingColumn) := by
  rw [residual,
    rowPoint_eq_paddingPortPoint row validated assignment,
    evaluate_paddingPortPoint]

/-- Under the separately owned constant-one invariant, the decoded residual
vanishes exactly when its named padding coordinate is zero. -/
theorem residual_eq_zero_iff
    (row : DecodedRow) (validated : ValidatedPaddingRow row)
    (assignment : Fin row.columns → F)
    (constantOne : assignment validated.constantColumn = 1) :
    residual row assignment = 0 ↔
      assignment validated.paddingColumn = 0 := by
  rw [residual_eq_neg_product row validated assignment, constantOne,
    Fin.one_mul]
  constructor
  · intro negated
    have := congrArg (fun value : F => -value) negated
    simpa only [Lean.Grind.AddCommGroup.neg_neg,
      Lean.Grind.AddCommGroup.neg_zero] using this
  · intro paddingZero
    rw [paddingZero]
    rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.ArtifactRefinement
