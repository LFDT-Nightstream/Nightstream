import Mathlib.Algebra.MvPolynomial.Eval
import Mathlib.Data.ZMod.Basic
import Nightstream.Implementation.R1CS.Core.Semantics
import Nightstream.SuperNeo.CheckPlan

/-!
Artifact-bound checking for recursive-verifier constraint classifications.

Assurance tier: model-level for the generic polynomial theorems. A concrete
result is artifact-checked only after its complete `Artifact` value and
certificate pass the checkers in this module. Rust-conformant and
security-reduced claims require separate composition theorems.

Owns: exact scalar polynomial-combination checking, family-level redundancy
transport, exact artifact equality, and executable removal counterexamples.

Does not own: cvc5 trust, Rust export conformance, protocol soundness,
recursive fixed-point costs, or a global minimum claim.

Emits constraints: no.
-/

namespace Nightstream.Assurance.ConstraintMinimization

open MvPolynomial
open Nightstream.SuperNeo.CheckPlan

namespace Numeric

abbrev Row := Nightstream.Implementation.R1CS.Row
def modulus := Nightstream.Implementation.R1CS.goldilocksP

end Numeric

local instance : NeZero Numeric.modulus := ⟨by decide⟩

abbrev Field := ZMod Numeric.modulus
abbrev Polynomial := MvPolynomial Nat Field

/-- One source row with its stable source index and semantic family. -/
structure IndexedRow where
  sourceIndex : Nat
  family : String
  row : Numeric.Row
deriving DecidableEq, Repr

/-- Complete value checked at the minimization boundary. The diagnostic digest
is metadata. Equality of this structure, including every row, is authority. -/
structure Artifact where
  schema : String
  profile : String
  diagnosticDigest : String
  totalRows : Nat
  columnCount : Nat
  constantOneColumn : Nat
  publicInputCount : Nat
  rows : List IndexedRow
deriving DecidableEq, Repr

namespace Artifact

def ExactValidation (authoritative carried : Artifact) : Bool :=
  decide (carried = authoritative)

theorem exactValidation_eq_true_iff
    {authoritative carried : Artifact} :
    ExactValidation authoritative carried = true ↔
      carried = authoritative := by
  simp [ExactValidation]

theorem accepted_eq_authoritative
    {authoritative carried : Artifact}
    (accepted : ExactValidation authoritative carried = true) :
    carried = authoritative :=
  exactValidation_eq_true_iff.mp accepted

end Artifact

namespace Algebraic

/-- Sparse linear combination as a formal polynomial. -/
noncomputable def linearPolynomial : List (Nat × Nat) → Polynomial
  | [] => 0
  | term :: tail =>
      C (term.2 : Field) * X term.1 + linearPolynomial tail

/-- Formal residual `(A z) * (B z) - C z`. -/
noncomputable def residual (row : Numeric.Row) : Polynomial :=
  linearPolynomial row.a * linearPolynomial row.b -
    linearPolynomial row.c

def linearEval (assignment : Nat → Field) : List (Nat × Nat) → Field
  | [] => 0
  | term :: tail =>
      (term.2 : Field) * assignment term.1 + linearEval assignment tail

def Holds (assignment : Nat → Field) (row : Numeric.Row) : Prop :=
  linearEval assignment row.a * linearEval assignment row.b =
    linearEval assignment row.c

instance (assignment : Nat → Field) (row : Numeric.Row) :
    Decidable (Holds assignment row) := by
  unfold Holds
  infer_instance

theorem eval_linearPolynomial
    (assignment : Nat → Field) (terms : List (Nat × Nat)) :
    eval assignment (linearPolynomial terms) =
      linearEval assignment terms := by
  induction terms with
  | nil => simp [linearPolynomial, linearEval]
  | cons head tail inductionHypothesis =>
      simp [linearPolynomial, linearEval, inductionHypothesis]

theorem eval_residual
    (assignment : Nat → Field) (row : Numeric.Row) :
    eval assignment (residual row) =
      linearEval assignment row.a * linearEval assignment row.b -
        linearEval assignment row.c := by
  simp [residual, eval_linearPolynomial]

end Algebraic

/-- One retained residual and its scalar coefficient. -/
structure ScalarSupport where
  source : IndexedRow
  coefficient : Field
deriving DecidableEq, Repr

noncomputable def scalarCombination : List ScalarSupport → Polynomial
  | [] => 0
  | support :: tail =>
      C support.coefficient * Algebraic.residual support.source.row +
        scalarCombination tail

/-- A small, proof-producing certificate grammar. It deliberately covers only
constant scalar combinations of retained residual polynomials. -/
structure ScalarCertificate where
  candidate : IndexedRow
  support : List ScalarSupport
deriving DecidableEq, Repr

namespace ScalarCertificate

def Valid (certificate : ScalarCertificate) : Prop :=
  Algebraic.residual certificate.candidate.row =
    scalarCombination certificate.support

theorem candidate_holds_of_valid
    (certificate : ScalarCertificate)
    (valid : certificate.Valid)
    (assignment : Nat → Field)
    (supportHolds : ∀ support ∈ certificate.support,
      Algebraic.Holds assignment support.source.row) :
    Algebraic.Holds assignment certificate.candidate.row := by
  have combinationZero : ∀ supports : List ScalarSupport,
      (∀ support ∈ supports,
        Algebraic.Holds assignment support.source.row) →
      eval assignment (scalarCombination supports) = 0 := by
    intro supports
    induction supports with
    | nil =>
        intro _
        simp [scalarCombination]
    | cons head tail inductionHypothesis =>
        intro holds
        have headHolds := holds head (by simp)
        have headResidual :
            eval assignment (Algebraic.residual head.source.row) = 0 := by
          rw [Algebraic.eval_residual]
          exact sub_eq_zero.mpr headHolds
        have tailHolds : ∀ support ∈ tail,
            Algebraic.Holds assignment support.source.row := by
          intro support member
          exact holds support (by simp [member])
        simp [scalarCombination, headResidual,
          inductionHypothesis tailHolds]
  have candidateResidual :
      eval assignment (Algebraic.residual certificate.candidate.row) = 0 := by
    rw [valid]
    exact combinationZero certificate.support supportHolds
  rw [Algebraic.eval_residual] at candidateResidual
  exact sub_eq_zero.mp candidateResidual

end ScalarCertificate

def candidateRows (artifact : Artifact) (family : String) :
    List IndexedRow :=
  artifact.rows.filter fun row => decide (row.family = family)

def FamilyHolds (artifact : Artifact) (family : String)
    (assignment : Nat → Field) : Prop :=
  ∀ row ∈ artifact.rows, row.family = family →
    Algebraic.Holds assignment row.row

instance (artifact : Artifact) (family : String)
    (assignment : Nat → Field) :
    Decidable (FamilyHolds artifact family assignment) := by
  unfold FamilyHolds
  infer_instance

def Target (artifact : Artifact) (assignment : Nat → Field) : Prop :=
  assignment artifact.constantOneColumn = 1 ∧
    ∀ row ∈ artifact.rows, Algebraic.Holds assignment row.row

instance (artifact : Artifact) (assignment : Nat → Field) :
    Decidable (Target artifact assignment) := by
  unfold Target
  infer_instance

/-- Exact family coverage plus artifact-bound support rows. Every support must
remain in the plan and must have a different family from the removed family. -/
structure FamilyCertificate where
  family : String
  certificates : List ScalarCertificate
deriving DecidableEq, Repr

namespace FamilyCertificate

def Valid (certificate : FamilyCertificate)
    (artifact : Artifact) (plan : List String) : Prop :=
  certificate.certificates.map (fun scalar => scalar.candidate) =
      candidateRows artifact certificate.family ∧
    ∀ scalar ∈ certificate.certificates,
      scalar.Valid ∧
        ∀ support ∈ scalar.support,
          support.source ∈ artifact.rows ∧
            support.source.family ∈ plan ∧
              support.source.family ≠ certificate.family

/-- An artifact-checked family certificate becomes a semantic redundancy
proof. cvc5 does not occur in this theorem or its assumptions. -/
theorem redundant_of_valid
    (certificate : FamilyCertificate)
    (artifact : Artifact) (plan : List String)
    (valid : certificate.Valid artifact plan) :
    Redundant (FamilyHolds artifact) plan certificate.family := by
  intro assignment accepted row rowMember rowFamily
  have candidateMember : row ∈ candidateRows artifact certificate.family := by
    simp [candidateRows, rowMember, rowFamily]
  have mappedMember :
      row ∈ certificate.certificates.map (fun scalar => scalar.candidate) := by
    rw [valid.1]
    exact candidateMember
  rcases List.mem_map.mp mappedMember with
    ⟨scalar, scalarMember, scalarCandidate⟩
  have scalarFacts := valid.2 scalar scalarMember
  have candidateHolds := ScalarCertificate.candidate_holds_of_valid
    scalar scalarFacts.1 assignment (by
      intro support supportMember
      have supportFacts := scalarFacts.2 support supportMember
      have familyAccepted := accepted support.source.family
        (mem_without_iff.mpr ⟨supportFacts.2.1, supportFacts.2.2⟩)
      exact familyAccepted support.source supportFacts.1 rfl)
  simpa [scalarCandidate] using candidateHolds

end FamilyCertificate

/-- Finite model record used for a checked removal counterexample. -/
structure RemovalCounterexample where
  removedFamily : String
  values : List Field
deriving DecidableEq, Repr

namespace RemovalCounterexample

def assignment (counterexample : RemovalCounterexample) : Nat → Field :=
  fun column => counterexample.values.getD column 0

def Valid (counterexample : RemovalCounterexample)
    (artifact : Artifact) (plan : List String) : Prop :=
  counterexample.values.length = artifact.columnCount ∧
    counterexample.assignment artifact.constantOneColumn = 1 ∧
      Accepts (FamilyHolds artifact)
        (without plan counterexample.removedFamily)
        counterexample.assignment ∧
        ¬ Target artifact counterexample.assignment

instance (counterexample : RemovalCounterexample)
    (artifact : Artifact) (plan : List String) :
    Decidable (counterexample.Valid artifact plan) := by
  unfold Valid Accepts
  infer_instance

def check (counterexample : RemovalCounterexample)
    (artifact : Artifact) (plan : List String) : Bool :=
  decide (counterexample.Valid artifact plan)

theorem valid_of_check
    {counterexample : RemovalCounterexample}
    {artifact : Artifact} {plan : List String}
    (checked : counterexample.check artifact plan = true) :
    counterexample.Valid artifact plan := by
  simpa [check] using of_decide_eq_true checked

theorem necessary_of_valid
    (counterexample : RemovalCounterexample)
    (artifact : Artifact) (plan : List String)
    (valid : counterexample.Valid artifact plan) :
    NecessaryForSoundness (FamilyHolds artifact) (Target artifact)
      plan counterexample.removedFamily :=
  ⟨counterexample.assignment, valid.2.2.1, valid.2.2.2⟩

end RemovalCounterexample

end Nightstream.Assurance.ConstraintMinimization
