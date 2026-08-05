import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialFixedWidth
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanReproduction

/-!
Model-level fixed-width padding obstruction for the paper-joint `Pi_CCS`
SumCheck.

Protocol: SuperNeo Definition 6, Section 7.3, and Appendix D.4.
Phase: one message-before-challenge SumCheck round.
Constraint family: paper protocol polynomial and fixed-width coefficient
transport only; this file emits no rows.

Owns: an artifact-independent paper-polynomial instance whose syntax-derived
round degree is four, a verifier-visible fixed polynomial of declared degree
six with a nonzero highest coefficient, and an exact
`ProtocolPolynomial.FixedWidth.SumCheckCollision` at each of six distinct
challenge values.

Does not own: probability, verifier-coin generation, conditioning,
Schwartz--Zippel, alpha/gamma mixing, binding, Fiat--Shamir, Rust, R1CS,
generated dimensions, or a claim that the paper permits unchecked nonzero
coefficients above its degree bound.

Emits constraints: no.

| Property | Kernel-checked owner |
|---|---|
| positive fresh-source instance | `shape_freshCount_positive` |
| syntax degree four | `syntaxDegree_eq_four` |
| exact paper polynomial is zero | `protocolPolynomial_eq_zero` |
| degree-six stored message | `rootPolynomial_messageDegree_eq_six` |
| nonzero position above both relevant ceilings | `rootPolynomial_highCoefficient_eq_one` |
| violated paper-ceiling padding discipline | `rootPolynomial_not_zeroAbovePaperDegree_four` |
| exact repository collision | `collision_at` |

Authority boundary: the paper's SumCheck error uses the degree of the
univariate round polynomial. The current fixed-width checker instead accepts
any coefficient vector whose declared degree is the selected width, while
`StrongExecution.Context` requires only `syntaxDegree <= width`. This module
isolates the resulting missing padding discipline; it does not alter the
checker or frozen protocol semantics.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckFixedWidthPadding

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier

abbrev Extension := K

/-- One SumCheck variable, one positive fresh source, and no carried source.
The selected source tables are zero, so the exact paper protocol polynomial is
zero while retaining the syntax-owned strict-norm degree ceiling of four. -/
def shape : Shape where
  cubeVariables := 1
  freshCount := 1
  runningCount := 0
  matrixCount := 1
  coefficientCount := 1

/-- The public prior point of the zero protocol datum. -/
def zeroPoint : CubePoint Extension shape.cubeVariables where
  coordinates := [extensionOps.zero]
  dimension := rfl

/-- The empty sparse CCS polynomial used by the zero protocol datum. -/
def zeroConstraint :
    CCSResidualTable.ConstraintPolynomial Extension shape.matrixCount where
  degreeBound := 0
  terms := []
  termsBelowDegree := by simp

/-- The identically-zero Boolean source table. -/
def zeroTable : BooleanTable Extension shape.cubeVariables :=
  BooleanTable.tabulate (fun _ => extensionOps.zero)

/-- A closed paper-owned protocol-polynomial datum at the positive-fresh-source
obstruction shape. Every hidden source table is identically zero. -/
def zeroProtocolData : ProtocolPolynomial.Data Extension shape where
  constraintPolynomial := zeroConstraint
  freshMatrixImages := fun _ _ => zeroTable
  sourceAssignments := fun _ => zeroTable
  priorPoint := zeroPoint
  carriedImages := fun coordinate => Fin.elim0 coordinate.running
  claimedCoefficient := fun coordinate => Fin.elim0 coordinate.running

/-- The shape has a genuine positive fresh source; the obstruction does not use
an empty-batch degeneration. -/
theorem shape_freshCount_positive : 0 < shape.freshCount := by
  decide

/-- The exact syntax-derived degree is four; no measured or generated width is
used. -/
theorem syntaxDegree_eq_four :
    zeroProtocolData.toVerifierInput.sumcheckDegreeBound = 4 := by
  rfl

private theorem zero_mul (value : Extension) :
    extensionOps.mul extensionOps.zero value = extensionOps.zero := by
  rw [extensionLaws.mul_comm, extensionLaws.mul_zero]

/-- At this positive-fresh-source shape the selected exact paper protocol
polynomial is identically zero: the sparse CCS polynomial is empty, the source
assignment table is zero, and the strict norm residual therefore vanishes. -/
theorem protocolPolynomial_eq_zero
    (alpha : CubePoint Extension shape.cubeVariables)
    (gamma : Extension)
    (coordinates : List Extension) :
    ProtocolPolynomial.polynomial extensionOps zeroProtocolData alpha gamma coordinates =
      extensionOps.zero := by
  unfold ProtocolPolynomial.polynomial
  split
  case isTrue dimension =>
    let point : CubePoint Extension shape.cubeVariables :=
      { coordinates := coordinates, dimension := dimension }
    change ProtocolPolynomial.qAtPoint extensionOps zeroProtocolData alpha gamma point =
      extensionOps.zero
    unfold ProtocolPolynomial.qAtPoint ProtocolPolynomial.terminalFromMessage
      ProtocolPolynomial.ccsAtMessage ProtocolPolynomial.normAtMessage
      ProtocolPolynomial.carriedAtMessage ProtocolPolynomial.messageAt
      ProtocolPolynomial.strictNormResidual
    simp only [zeroProtocolData, zeroTable]
    have evalZero :
        BooleanTable.evaluate extensionOps
            (BooleanTable.tabulate (fun _ : BooleanVertex shape.cubeVariables =>
              extensionOps.zero)) point = extensionOps.zero :=
      BooleanReproduction.evaluate_tabulate_constant extensionOps extensionLaws
        extensionOps.zero point
    repeat rw [evalZero]
    simp [zeroConstraint, ProtocolPolynomial.Data.toVerifierInput, shape,
      Shape.sourceCount, Shape.carriedEvaluationOffset,
      canonicalFinIndices, canonicalCarriedCoordinates,
      BooleanTable.finiteSum, SignedJointIdentity.sumMap,
      SignedJointIdentity.gammaTerm, CCSResidualTable.evaluatePolynomial,
      extensionLaws.mul_zero, zero_mul, extensionLaws.zero_add]
  case isFalse => rfl

private def linearRoot (value : Extension) :
    SumCheck.Finite.FixedPolynomial Extension 1 :=
  SumCheck.Finite.FixedPolynomial.affine (extensionOps.neg value)
    extensionOps.one

/-- The verifier-visible degree-six polynomial
`X (X-1) (X-2) (X-3) (X-4) (X-5)`. Its storage has exactly seven
coefficients; no canonical trimming is assumed. -/
def rootPolynomial :
    SumCheck.Finite.FixedPolynomial Extension 6 :=
  SumCheck.Finite.FixedPolynomial.mul extensionOps.toOps
    (SumCheck.Finite.FixedPolynomial.mul extensionOps.toOps
      (SumCheck.Finite.FixedPolynomial.mul extensionOps.toOps
        (SumCheck.Finite.FixedPolynomial.mul extensionOps.toOps
          (SumCheck.Finite.FixedPolynomial.mul extensionOps.toOps
            (linearRoot (K.embed 0)) (linearRoot (K.embed 1)))
          (linearRoot (K.embed 2)))
        (linearRoot (K.embed 3)))
      (linearRoot (K.embed 4)))
    (linearRoot (K.embed 5))

/-- Raw fixed-width transport derives degree six from the seven stored
coefficients. -/
theorem rootPolynomial_messageDegree_eq_six :
    rootPolynomial.toMessage.degreeUpperBound = 6 := by
  exact SumCheck.Finite.FixedPolynomial.toMessage_degreeUpperBound rootPolynomial

/-- The highest position is above both the syntax degree four and corrected
Appendix D.4 degree ceiling four, and is genuinely nonzero. -/
theorem rootPolynomial_highCoefficient_eq_one :
    rootPolynomial.coefficients[6]? = some extensionOps.one := by
  decide

/-- The minimal fixed-width discipline missing from the current context: every
stored coefficient position strictly above the supplied degree ceiling is zero.
This predicate does not require or perform canonical trimming. -/
def ZeroAboveDegree
    (syntaxDegree : Nat)
    {width : Nat}
    (polynomial : SumCheck.Finite.FixedPolynomial Extension width) : Prop :=
  forall index : Fin (width + 1),
    syntaxDegree < index.val ->
      polynomial.coefficients[index.val]? = some extensionOps.zero

/-- The admitted degree-six message violates the exact zero-padding discipline
at position six. -/
theorem rootPolynomial_not_zeroAboveSyntaxDegree_four :
    ¬ ZeroAboveDegree 4 rootPolynomial := by
  intro padded
  have highZero := padded (show Fin (6 + 1) from ⟨6, by decide⟩) (by decide)
  rw [rootPolynomial_highCoefficient_eq_one] at highZero
  have one_eq_zero : extensionOps.one = extensionOps.zero :=
    Option.some.inj highZero
  exact (by decide : extensionOps.one ≠ extensionOps.zero) one_eq_zero

/-- The same high coefficient violates Appendix D.4's corrected permitted
per-variable degree ceiling four for this zero-CCS, `b = 2` instance. -/
theorem rootPolynomial_not_zeroAbovePaperDegree_four :
    ¬ ZeroAboveDegree 4 rootPolynomial :=
  rootPolynomial_not_zeroAboveSyntaxDegree_four

private theorem linearRoot_evaluate_self (value : Extension) :
    (linearRoot value).evaluate extensionOps.toOps value = extensionOps.zero := by
  change
    (SumCheck.Finite.FixedPolynomial.affine (extensionOps.neg value)
      extensionOps.one).evaluate extensionOps.toOps value = extensionOps.zero
  rw [SumCheck.Finite.FixedPolynomial.evaluate_affine extensionOps.toOps
    (ProtocolPolynomialDegree.Support.polynomialLaws extensionLaws)]
  rw [extensionLaws.mul_one, extensionLaws.add_comm]
  exact extensionLaws.add_neg value

/-- The malicious polynomial vanishes at challenge zero. -/
theorem rootPolynomial_zero_at_zero :
    rootPolynomial.evaluate extensionOps.toOps (K.embed 0) = extensionOps.zero := by
  simp [rootPolynomial, SumCheck.Finite.FixedPolynomial.evaluate_mul,
    ProtocolPolynomialDegree.Support.polynomialLaws extensionLaws,
    linearRoot_evaluate_self, zero_mul]

/-- The malicious polynomial vanishes at challenge one. -/
theorem rootPolynomial_zero_at_one :
    rootPolynomial.evaluate extensionOps.toOps (K.embed 1) = extensionOps.zero := by
  simp [rootPolynomial, SumCheck.Finite.FixedPolynomial.evaluate_mul,
    ProtocolPolynomialDegree.Support.polynomialLaws extensionLaws,
    linearRoot_evaluate_self, extensionLaws.mul_zero, zero_mul]

/-- The malicious polynomial vanishes at challenge two. -/
theorem rootPolynomial_zero_at_two :
    rootPolynomial.evaluate extensionOps.toOps (K.embed 2) = extensionOps.zero := by
  simp [rootPolynomial, SumCheck.Finite.FixedPolynomial.evaluate_mul,
    ProtocolPolynomialDegree.Support.polynomialLaws extensionLaws,
    linearRoot_evaluate_self, extensionLaws.mul_zero, zero_mul]

/-- The malicious polynomial vanishes at challenge three. -/
theorem rootPolynomial_zero_at_three :
    rootPolynomial.evaluate extensionOps.toOps (K.embed 3) = extensionOps.zero := by
  simp [rootPolynomial, SumCheck.Finite.FixedPolynomial.evaluate_mul,
    ProtocolPolynomialDegree.Support.polynomialLaws extensionLaws,
    linearRoot_evaluate_self, extensionLaws.mul_zero, zero_mul]

/-- The malicious polynomial vanishes at challenge four. -/
theorem rootPolynomial_zero_at_four :
    rootPolynomial.evaluate extensionOps.toOps (K.embed 4) = extensionOps.zero := by
  simp [rootPolynomial, SumCheck.Finite.FixedPolynomial.evaluate_mul,
    ProtocolPolynomialDegree.Support.polynomialLaws extensionLaws,
    linearRoot_evaluate_self, extensionLaws.mul_zero, zero_mul]

/-- The malicious polynomial vanishes at challenge five. -/
theorem rootPolynomial_zero_at_five :
    rootPolynomial.evaluate extensionOps.toOps (K.embed 5) = extensionOps.zero := by
  simp [rootPolynomial, SumCheck.Finite.FixedPolynomial.evaluate_mul,
    ProtocolPolynomialDegree.Support.polynomialLaws extensionLaws,
    linearRoot_evaluate_self, extensionLaws.mul_zero]

/-- The claimed degree-six polynomial is not the correct zero round
polynomial. Evaluation at six witnesses the functional inequality without
coefficient trimming. -/
theorem rootPolynomial_nonzero_function :
    rootPolynomial.evaluate extensionOps.toOps ≠
      fun _ => extensionOps.zero := by
  intro equal
  have atSix :
      rootPolynomial.evaluate extensionOps.toOps (K.embed 6) =
        extensionOps.zero := congrFun equal (K.embed 6)
  have nonzero :
      rootPolynomial.evaluate extensionOps.toOps (K.embed 6) ≠
        extensionOps.zero := by
    decide
  exact nonzero atSix

/-- The exact fixed-width certificate transported by the causal strategy. -/
def certificate : SumCheck.Finite.FixedPhase.Certificate Extension 6 where
  rounds := [rootPolynomial]

/-- A one-coordinate verifier point. -/
def point (challenge : Extension) : CubePoint Extension shape.cubeVariables where
  coordinates := [challenge]
  dimension := rfl

/-- Exact deterministic alignment with the repository event: whenever the
stored degree-six polynomial vanishes, the decoded certificate produces the
actual paper-joint `ProtocolPolynomial.FixedWidth.SumCheckCollision`. The
correct expected round is independently recomputed from the exact protocol
polynomial and is zero. -/
theorem collision_at
    (alpha : CubePoint Extension shape.cubeVariables)
    (gamma challenge : Extension)
    (rootH : rootPolynomial.evaluate extensionOps.toOps challenge =
      extensionOps.zero) :
    ProtocolPolynomial.FixedWidth.SumCheckCollision extensionOps zeroProtocolData alpha
      gamma 6 6 (point challenge) certificate := by
  let q := ProtocolPolynomial.polynomial extensionOps zeroProtocolData alpha gamma
  let expected : Extension -> Extension := fun value =>
    SumCheck.Finite.HypercubeTruth.sumCompletions extensionOps.toOps q [value] 0
  let round : SumCheck.Round Extension Extension := {
    claimed := rootPolynomial.evaluate extensionOps.toOps
    expected := expected
    challenge := challenge
    degree := 6
  }
  refine ⟨round, ?_⟩
  constructor
  · unfold SumCheck.Finite.FixedPhase.AlgebraicBadChallenge SumCheck.BadChallenge
    refine ⟨?_, Nat.le_refl _, ?_, ?_⟩
    · change round ∈ [{
          claimed := rootPolynomial.evaluate extensionOps.toOps
          expected := fun value =>
            SumCheck.Finite.HypercubeTruth.sumCompletions extensionOps.toOps
              q ([] ++ [value]) [].length
          challenge := challenge
          degree := 6
        }]
      simp [round, expected]
    · intro equal
      apply rootPolynomial_nonzero_function
      funext value
      have expectedZero : expected value = extensionOps.zero := by
        simp [expected, q, SumCheck.Finite.HypercubeTruth.sumCompletions,
          protocolPolynomial_eq_zero]
      simpa [round, expectedZero] using congrFun equal value
    · have expectedZero : expected challenge = extensionOps.zero := by
        simp [expected, q, SumCheck.Finite.HypercubeTruth.sumCompletions,
          protocolPolynomial_eq_zero]
      simpa [round, expectedZero] using rootH
  · refine ⟨rootPolynomial,
      SumCheck.Finite.FixedPolynomial.zero extensionOps.toOps 6, ?_, ?_⟩
    · intro value
      rfl
    · intro value
      have expectedZero : expected value = extensionOps.zero := by
        simp [expected, q, SumCheck.Finite.HypercubeTruth.sumCompletions,
          protocolPolynomial_eq_zero]
      simpa [round, expectedZero] using
        (SumCheck.Finite.FixedPolynomial.evaluate_zero extensionOps.toOps
          (ProtocolPolynomialDegree.Support.polynomialLaws extensionLaws)
          6 value)

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckFixedWidthPadding
