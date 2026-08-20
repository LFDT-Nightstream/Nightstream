import Mathlib.FieldTheory.Finite.Basic
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentitySoundness
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrong
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.RootCountingSecurity

/-!
Contract: finite interactive security boundary for the selected
`PaddedRowIdentity` relation.

Owns: closure of the production `U^2 - 7` no-zero-divisor premise from the
kernel-checked Goldilocks generator certificate, and selected-profile
specialization of the generic one-joint root-counting theorem.

Does not own: Poseidon2 or Fiat--Shamir security, Ajtai binding, Rust or R1CS
conformance, or release security.

Emits constraints: no.

| Code owner | Protocol object | Mathematical obligation | Proven result |
|---|---|---|---|
| `sevenProjectiveNonresidue` | quadratic challenge carrier `K` | `U^2 - 7` must be irreducible over Goldilocks | derived from the checked generator order |
| `extensionNoZeroDivisors` | finite root counting over `K` | nonzero products must not vanish | exact concrete law |
| `fullChallengeSupport` | ideal public-coin alphabet | contain each element of `K` exactly once | cardinality `q^2` |
| `selectedContext` | one-joint interactive protocol | use the selected relation, degree, and finite challenge support | exact typed context |
| `fixedFirstBadBound_of_rootCounting` | algebraic soundness error | bound both bad events without assuming either probability contract | exact finite root-count bound |
| `algebraicNumerator_exact` | selected one-fold loss | include mixing and all 24 degree-nine SumCheck rounds | `12329` field units |
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySecurity

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalExperiment
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.RootCountingSecurity
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySoundness

universe uCommitment uPublicInput uProverSeed uTargetSeed uProverTape

private local instance : Fact (Nat.Prime goldilocksP) :=
  ⟨GoldilocksField.goldilocks_natPrime⟩

/-- The checked primitive-generator certificate implies that seven is not a
square in the Goldilocks field. Therefore `U^2 - 7` has no projective root. -/
theorem sevenProjectiveNonresidue : SevenProjectiveNonresidue := by
  intro real imaginary equation
  let equiv : F ≃+* ZMod goldilocksP := ZMod.finEquiv goldilocksP
  by_cases imaginaryZero : imaginary = 0
  · subst imaginary
    have realSquareZero : real * real = 0 := by
      simpa using equation
    rcases
        (NormRange.baseFieldNoZeroDivisors_of_modulusEuclid
          GoldilocksField.goldilocks_euclidPrime)
          real real realSquareZero with realZero | realZero
    · exact ⟨realZero, rfl⟩
    · exact ⟨realZero, rfl⟩
  · have mappedEquation := congrArg equiv equation
    have mappedEquation' :
        equiv real * equiv real -
            (7 : ZMod goldilocksP) * equiv imaginary * equiv imaginary = 0 := by
      simpa only [map_sub, map_mul, map_ofNat, map_zero] using mappedEquation
    have squareEquation :
        (equiv real) ^ 2 = (7 : ZMod goldilocksP) * (equiv imaginary) ^ 2 := by
      simpa only [pow_two, mul_assoc] using sub_eq_zero.mp mappedEquation'
    have imaginaryMappedNonzero : equiv imaginary ≠ 0 := by
      intro equal
      apply imaginaryZero
      exact equiv.injective (by simpa using equal)
    have realMappedNonzero : equiv real ≠ 0 := by
      intro realMappedZero
      have rightZero :
          (7 : ZMod goldilocksP) * (equiv imaginary) ^ 2 = 0 := by
        rw [← squareEquation, realMappedZero]
        simp
      have sevenNonzero : (7 : ZMod goldilocksP) ≠ 0 := by
        intro equal
        exact (by decide : ¬ goldilocksP ∣ 7)
          ((ZMod.natCast_eq_zero_iff 7 goldilocksP).mp equal)
      have imaginarySquareZero : (equiv imaginary) ^ 2 = 0 :=
        (mul_eq_zero.mp rightZero).resolve_left sevenNonzero
      exact imaginaryMappedNonzero
        (eq_zero_of_pow_eq_zero imaginarySquareZero)
    let exponent := (goldilocksP - 1) / 2
    have exponentTwice : 2 * exponent = goldilocksP - 1 := by
      decide
    have raised :
        (equiv real) ^ (goldilocksP - 1) =
          (7 : ZMod goldilocksP) ^ exponent *
            (equiv imaginary) ^ (goldilocksP - 1) := by
      calc
        (equiv real) ^ (goldilocksP - 1) =
            (equiv real) ^ (2 * exponent) := by rw [exponentTwice]
        _ = ((equiv real) ^ 2) ^ exponent := by rw [pow_mul]
        _ = ((7 : ZMod goldilocksP) * (equiv imaginary) ^ 2) ^ exponent := by
          rw [squareEquation]
        _ = (7 : ZMod goldilocksP) ^ exponent *
            ((equiv imaginary) ^ 2) ^ exponent := by rw [mul_pow]
        _ = (7 : ZMod goldilocksP) ^ exponent *
            (equiv imaginary) ^ (2 * exponent) := by rw [pow_mul]
        _ = (7 : ZMod goldilocksP) ^ exponent *
            (equiv imaginary) ^ (goldilocksP - 1) := by rw [exponentTwice]
    have sevenPowerOne :
        (7 : ZMod goldilocksP) ^ exponent = 1 := by
      rw [ZMod.pow_card_sub_one_eq_one realMappedNonzero,
        ZMod.pow_card_sub_one_eq_one imaginaryMappedNonzero] at raised
      simpa using raised.symm
    exact False.elim
      (GoldilocksField.order_not_halved_zmod (by
        simpa [exponent] using sevenPowerOne))

/-- The concrete quadratic extension used for joint challenges has no zero
divisors. No irreducibility premise remains. -/
theorem extensionNoZeroDivisors :
    FiniteRootCounting.NoZeroDivisors extensionOps :=
  extensionNoZeroDivisors_of_base_and_seven
    (NormRange.baseFieldNoZeroDivisors_of_modulusEuclid
      GoldilocksField.goldilocks_euclidPrime)
    sevenProjectiveNonresidue

/-- Coefficient representation of the production quadratic extension. -/
def coefficientPairEquiv : (F × F) ≃ K where
  toFun pair := ⟨pair.1, pair.2⟩
  invFun value := (value.c0, value.c1)
  left_inv := fun _ => rfl
  right_inv := fun _ => rfl

private local instance : Fintype K :=
  Fintype.ofEquiv (F × F) coefficientPairEquiv

/-- Ideal interactive challenge alphabet: every quadratic-extension element
appears exactly once. This is not a Fiat--Shamir or Poseidon2 claim. -/
noncomputable def fullChallengeSupport : Support K where
  values := Finset.univ.toList
  nodup := Finset.nodup_toList Finset.univ
  nonempty := by
    intro empty
    have member : K.zero ∈ (Finset.univ : Finset K).toList := by
      simp
    rw [empty] at member
    simp at member

/-- The ideal interactive alphabet has exactly `q^2` challenges. -/
theorem fullChallengeSupport_cardinality :
    fullChallengeSupport.cardinality = goldilocksP * goldilocksP := by
  calc
    fullChallengeSupport.cardinality = Fintype.card K := by
      simp [fullChallengeSupport, Support.cardinality]
    _ = Fintype.card (F × F) :=
      (Fintype.card_congr coefficientPairEquiv).symm
    _ = goldilocksModulus * goldilocksModulus := by
      simp [F]
    _ = goldilocksP * goldilocksP := by
      rfl

/-- The exact selected interactive context. Its challenge-set size is the
cardinality of the explicit verifier support, so the probability theorem does
not use a caller-supplied size that can disagree with sampling. -/
noncomputable def selectedContext
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    (openingMaps : OpeningMaps Commitment PublicInput assignmentColumns)
    (matrices : ApplicationMatrices)
    (commitments : Fin shape.sourceCount -> Commitment)
    (publicInputs : Fin shape.sourceCount -> PublicInput)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K)
    (alphabet : Support K) :
    Context K Commitment PublicInput shape assignmentColumns
      (Phi81ColumnLayout.blockCount assignmentColumns) where
  baseOps := baseOps
  baseLaws := baseLaws
  baseZero := baseZeroAgreement
  noZeroDivisors :=
    NormRange.baseFieldNoZeroDivisors_of_modulusEuclid
      GoldilocksField.goldilocks_euclidPrime
  extensionOps := extensionOps
  extensionLaws := extensionLaws
  extensionZeroLaws := extensionZeroLaws
  lift := K.embed
  liftLaws := protocolLift
  openingMaps := openingMaps
  params := productionGlobalParams
  freshBound := rfl
  statement := statement matrices commitments publicInputs priorPoint
    claimedCoefficient
  ambientDecision := fun _ _ => Classical.propDecidable _
  constantLaw := Phi81CoefficientKernel.phi81ConstantTermLaw
  sumcheckWidth := 9
  sumcheckDegreeBound_le := by
    rw [statement_sumcheckDegree_exact]
  challengeSetSize := alphabet.cardinality

/-- The selected transport width is the exact verifier-computed degree, not a
larger storage allowance. -/
theorem selectedContext_degreeWidthExact
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    (openingMaps : OpeningMaps Commitment PublicInput assignmentColumns)
    (matrices : ApplicationMatrices)
    (commitments : Fin shape.sourceCount -> Commitment)
    (publicInputs : Fin shape.sourceCount -> PublicInput)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K)
    (alphabet : Support K) :
    PaperDegreeWidthExact
      (selectedContext openingMaps matrices commitments publicInputs
        priorPoint claimedCoefficient alphabet) := by
  unfold PaperDegreeWidthExact selectedContext
  exact statement_sumcheckDegree_exact matrices commitments publicInputs
    priorPoint claimedCoefficient

/-- Root-count numerator for joint alpha/gamma mixing. This is the corrected
paper maximum of the alpha-dependent table branch and the alpha-free carried
branch. -/
def mixingNumerator : Nat :=
  MixingSoundness.paperMixingNumerator shape

/-- Root-count numerator for all selected SumCheck rounds. -/
def sumCheckNumerator : Nat := shape.cubeVariables * 9

/-- Total one-fold algebraic union-bound numerator. -/
def algebraicNumerator : Nat := mixingNumerator + sumCheckNumerator

theorem mixingNumerator_exact : mixingNumerator = 12113 := by
  decide

theorem sumCheckNumerator_exact : sumCheckNumerator = 216 := by
  decide

theorem algebraicNumerator_exact : algebraicNumerator = 12329 := by
  decide

/-- The exact numerator still gives at least 114 classical bits for one fold
when alpha, gamma, and round challenges are uniform over all `K`. -/
theorem oneFoldAlgebraicBits_at_least_114 :
    algebraicNumerator * 2 ^ 114 <= goldilocksP * goldilocksP := by
  decide

/-- The one-fold algebraic floor is not 115 bits. -/
theorem oneFoldAlgebraicBits_not_115 :
    ¬ algebraicNumerator * 2 ^ 115 <= goldilocksP * goldilocksP := by
  decide

/-- A union bound over 64 folds still gives at least 108 algebraic bits. -/
theorem sixtyFourFoldAlgebraicBits_at_least_108 :
    algebraicNumerator * 64 * 2 ^ 108 <=
      goldilocksP * goldilocksP := by
  decide

/-- The 64-fold algebraic floor is not 109 bits. -/
theorem sixtyFourFoldAlgebraicBits_not_109 :
    ¬ algebraicNumerator * 64 * 2 ^ 109 <=
      goldilocksP * goldilocksP := by
  decide

/-- Selected-profile finite interactive soundness for the algebraic part of
PiCCS. This theorem constructs both probability contracts by root counting;
neither is a premise. -/
theorem fixedFirstBadBound_of_rootCounting
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {ProverSeed : Type uProverSeed}
    {TargetSeed : Type uTargetSeed}
    {ProverTape : Type uProverTape}
    (openingMaps : OpeningMaps Commitment PublicInput assignmentColumns)
    (matrices : ApplicationMatrices)
    (commitments : Fin shape.sourceCount -> Commitment)
    (publicInputs : Fin shape.sourceCount -> PublicInput)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K)
    (alphabet : Support K)
    (adversary : Adversary
      (selectedContext openingMaps matrices commitments publicInputs
        priorPoint claimedCoefficient alphabet)
      ProverSeed TargetSeed ProverTape) :
    FixedFirstBadBound
      (selectedContext openingMaps matrices commitments publicInputs
        priorPoint claimedCoefficient alphabet)
      alphabet adversary
      (ratio mixingNumerator alphabet.cardinality +
        ratio sumCheckNumerator alphabet.cardinality) := by
  simpa [mixingNumerator, sumCheckNumerator] using
    RootCountingSecurity.fixedFirstBadBound_of_rootCounting
      (selectedContext openingMaps matrices commitments publicInputs
        priorPoint claimedCoefficient alphabet)
      (selectedContext_degreeWidthExact openingMaps matrices commitments
        publicInputs priorPoint claimedCoefficient alphabet)
      extensionNoZeroDivisors alphabet rfl adversary

/-- Both selected `Pi_CCS` probability contracts are consequences of finite
root counting. The caller supplies only its executable-time predicate; it
does not supply either algebraic soundness claim. -/
theorem selectedNamedSecurityContracts
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {ProverSeed : Type uProverSeed}
    {TargetSeed : Type uTargetSeed}
    {ProverTape : Type uProverTape}
    (openingMaps : OpeningMaps Commitment PublicInput assignmentColumns)
    (matrices : ApplicationMatrices)
    (commitments : Fin shape.sourceCount -> Commitment)
    (publicInputs : Fin shape.sourceCount -> PublicInput)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K)
    (alphabet : Support K)
    (adversaryExpectedPolynomialTime :
      Adversary
        (selectedContext openingMaps matrices commitments publicInputs
          priorPoint claimedCoefficient alphabet)
        ProverSeed TargetSeed ProverTape -> Prop) :
    FinitePaperStrong.NamedSecurityContracts
      (selectedContext openingMaps matrices commitments publicInputs
        priorPoint claimedCoefficient alphabet)
      alphabet adversaryExpectedPolynomialTime
      (ratio mixingNumerator alphabet.cardinality)
      (ratio sumCheckNumerator alphabet.cardinality) := by
  constructor
  · intro adversary _expected
    simpa [mixingNumerator] using
      MixingSoundness.mixingRootProbabilityContract_of_rootCounting
        (selectedContext openingMaps matrices commitments publicInputs
          priorPoint claimedCoefficient alphabet)
        extensionNoZeroDivisors alphabet adversary
  · intro adversary _expected
    simpa [sumCheckNumerator] using
      SumCheckSoundness.sumCheckSoundnessContract_of_rootCounting
        (selectedContext openingMaps matrices commitments publicInputs
          priorPoint claimedCoefficient alphabet)
        (selectedContext_degreeWidthExact openingMaps matrices commitments
          publicInputs priorPoint claimedCoefficient alphabet)
        extensionNoZeroDivisors alphabet rfl adversary

/-- Fully specialized ideal-interactive algebraic error bound. The denominator
is the complete production quadratic-extension challenge space. -/
theorem fullChallengeFixedFirstBadBound
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {ProverSeed : Type uProverSeed}
    {TargetSeed : Type uTargetSeed}
    {ProverTape : Type uProverTape}
    (openingMaps : OpeningMaps Commitment PublicInput assignmentColumns)
    (matrices : ApplicationMatrices)
    (commitments : Fin shape.sourceCount -> Commitment)
    (publicInputs : Fin shape.sourceCount -> PublicInput)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K)
    (adversary : Adversary
      (selectedContext openingMaps matrices commitments publicInputs
        priorPoint claimedCoefficient fullChallengeSupport)
      ProverSeed TargetSeed ProverTape) :
    FixedFirstBadBound
      (selectedContext openingMaps matrices commitments publicInputs
        priorPoint claimedCoefficient fullChallengeSupport)
      fullChallengeSupport adversary
      (ratio mixingNumerator (goldilocksP * goldilocksP) +
        ratio sumCheckNumerator (goldilocksP * goldilocksP)) := by
  simpa [fullChallengeSupport_cardinality] using
    fixedFirstBadBound_of_rootCounting openingMaps matrices commitments
      publicInputs priorPoint claimedCoefficient fullChallengeSupport adversary

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySecurity
