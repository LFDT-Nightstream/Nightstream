import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract

/-!
Focused executable regressions for the model-level fixed-width SumCheck
soundness obstruction.

These tests cover the adversarial boundary cases that are independent of a
future positive root-counting development. The headline counterexample itself
uses the exact causal repository event and actual verifier product support.
-/

set_option autoImplicit false

namespace tests.PiCcsPaperJointSumCheckSoundnessObstruction

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckFixedWidthPadding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract

#check syntaxDegree_eq_four
#check shape_freshCount_positive
#check protocolPolynomial_eq_zero
#check rootPolynomial_messageDegree_eq_six
#check rootPolynomial_highCoefficient_eq_one
#check rootPolynomial_not_zeroAboveSyntaxDegree_four
#check rootPolynomial_not_zeroAbovePaperDegree_five
#check rootPolynomial_nonzero_function
#check collision_at
#check paperRoundDegreeCeiling_eq_five
#check paperSumCheckBudget_eq_five_six
#check context_challengeSetSize_eq_alphabet_cardinality
#check context_not_paperDegreeWidthExact
#check strategy_roundMessage_eq
#check sourceProtocolData_eq_zero
#check sumCheckFailure_execute
#check sumCheckFailure_probability_eq_one
#check not_sumCheckSoundnessContract_of_lt_one
#check not_sumCheckSoundnessContract_at_paper_budget

/-- Edge case 1: an empty challenge support is rejected by the `Support`
constructor's nonempty proof. -/
example (support : Support Empty) : False := by
  obtain ⟨value, _⟩ := List.exists_mem_of_ne_nil support.values support.nonempty
  exact nomatch value

/-- Edge case 2: cardinality-one supports are representable and must therefore
be handled explicitly by any quotient bound. -/
def singletonChallengeSupport : Support Bool where
  values := [false]
  nodup := by decide
  nonempty := by decide

example : singletonChallengeSupport.cardinality = 1 := by
  rfl

/-- Edge case 3: degree zero is represented by exactly one coefficient and a
nonzero constant has no root at any challenge. -/
def degreeZeroPolynomial : SumCheck.Finite.FixedPolynomial Extension 0 :=
  SumCheck.Finite.FixedPolynomial.constant extensionOps.one

example : degreeZeroPolynomial.toMessage.degreeUpperBound = 0 := by
  exact SumCheck.Finite.FixedPolynomial.toMessage_degreeUpperBound
    degreeZeroPolynomial

example (challenge : Extension) :
    degreeZeroPolynomial.evaluate extensionOps.toOps challenge =
      extensionOps.one := by
  exact SumCheck.Finite.FixedPolynomial.evaluate_constant extensionOps.toOps
    (ProtocolPolynomialDegree.Support.polynomialLaws extensionLaws)
    extensionOps.one challenge

/-- Edge case 4: here the actual represented degree equals the challenge-set
cardinality, and all challenges are roots. -/
example : rootPolynomial.toMessage.degreeUpperBound = alphabet.cardinality := by
  rw [rootPolynomial_messageDegree_eq_six, alphabet_cardinality_eq_six]

/-- Edge case 5: the zero difference cannot satisfy the strict polynomial
inequality conjunct of `BadChallenge`. -/
example (polynomial : Extension -> Extension) :
    ¬ polynomial ≠ polynomial := by
  simp

/-- Edge case 6: two fixed polynomials with identical padded coefficient
storage are definitionally the same polynomial; a distinct pair cannot hide
behind identical storage. -/
example {left right : SumCheck.Finite.FixedPolynomial Extension 6}
    (sameStorage : left.coefficients = right.coefficients) :
    left = right := by
  cases left
  cases right
  cases sameStorage
  rfl

/-- Edge case 7: a coefficient position above both syntax degree four and
Appendix D.4 degree five is nonzero. -/
example : rootPolynomial.coefficients[6]? = some extensionOps.one :=
  rootPolynomial_highCoefficient_eq_one

example : ¬ ZeroAboveDegree 4 rootPolynomial :=
  rootPolynomial_not_zeroAboveSyntaxDegree_four

example : ¬ ZeroAboveDegree 5 rootPolynomial :=
  rootPolynomial_not_zeroAbovePaperDegree_five

/-- Edge case 8: malformed raw messages fail closed at exact-width decoding. -/
def malformedCertificate : SumCheck.Finite.Certificate Extension where
  rounds := [{ coefficients := [extensionOps.zero] }]

example :
    SumCheck.Finite.FixedPhase.RawCertificate.decode 6 malformedCertificate =
      none := by
  decide

/-- Edge case 9: the verifier-owned syntax metadata is smaller than the raw
message's represented degree in the admitted context. -/
example :
    zeroProtocolData.toVerifierInput.sumcheckDegreeBound <
      rootPolynomial.toMessage.degreeUpperBound := by
  rw [syntaxDegree_eq_four, rootPolynomial_messageDegree_eq_six]
  decide

example : paperRoundDegreeCeiling < rootPolynomial.toMessage.degreeUpperBound := by
  rw [paperRoundDegreeCeiling_eq_five, rootPolynomial_messageDegree_eq_six]
  decide

example : paperSumCheckBudget = ratio 5 6 :=
  paperSumCheckBudget_eq_five_six

/-- Edge case 10: a generic finite support can correlate two coordinates even
when it is nonempty and duplicate-free. A positive multi-round proof must use
the actual product challenge law rather than cardinality alone. -/
def correlatedPairSupport : Support (Bool × Bool) where
  values := [(false, false), (true, true)]
  nodup := by decide
  nonempty := by decide

example : forall pair, pair ∈ correlatedPairSupport.values -> pair.1 = pair.2 := by
  decide

/- Edge case 11 is rejected by the strategy interface: the current challenge
is absent from `roundMessage`; the constant-message theorem checks the actual
strategy used by the counterexample. -/
#check StrongExecution.Strategy.roundMessage

/- Edge case 12 is rejected at the contract boundary: the witness quantifier
is outside the fresh experiment probability. -/
#check SumCheckSoundnessContract

/- Edge cases 13 and 14 remain separate named events. The exact bridge targets
SumCheck failure; terminal output inconsistency and alpha/gamma mixing each have
their own predicates and are not used to manufacture a root collision. -/
#check SumCheckFailure
#check ProtocolPolynomial.OutputMismatch
#check SignedCoefficientObject.MixingRoot

/- Edge case 15 is guarded at the existing security-composition theorem; the
SumCheck budget is added before the separately adjusted mismatch term. -/
#check extraction_after_first_success_of_securityContracts

/-- Edge case 16: no trimming premise is assumed. The nonzero high coefficient
survives raw conversion and determines degree six. -/
example :
    rootPolynomial.toMessage.coefficients[6]? = some extensionOps.one := by
  exact rootPolynomial_highCoefficient_eq_one

end tests.PiCcsPaperJointSumCheckSoundnessObstruction
