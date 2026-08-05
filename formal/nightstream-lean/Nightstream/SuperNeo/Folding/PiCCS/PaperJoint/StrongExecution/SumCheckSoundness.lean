import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CausalSumCheckBound
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Reindex

/-!
Concrete finite SumCheck soundness for the causal paper `Pi_CCS` experiment.

Assurance tier: model-level.

Owns: construction of the existing `SumCheckSoundnessContract` from the
repaired exact degree-width invariant, finite root counting, causal
message-before-challenge replay, successive-coordinate independence, exact
transport to `sumCheckBadChallengeEvent`, and the explicit round union bound.

Does not own: Fiat--Shamir, alpha/gamma Schwartz--Zippel, rejection sampling,
Rust, R1CS, artifacts, minimality, or costs.

Emits constraints: no.

| Owned object | Exact equation or bound |
|---|---|
| event transport | `sumCheckBadChallengeEvent = true -> detects = true` |
| concrete probability | `Pr[sumCheckBadChallengeEvent] <= sumCheckBudget` |
| contract construction | `SumCheckSoundnessContract` from root counting |
| corrected extraction loss | `(mixing + sumcheck) + rootMismatch` |

The only extra algebraic input is the paper field law that multiplication has
no zero divisors.  `SumCheckSoundnessContract` is a conclusion of the concrete
theorem below and is never retained as one of its premises.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SumCheckSoundness

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.SumCheck.Finite

universe uExtension uCommitment uPublicInput uProverSeed uTargetSeed uProverTape

private theorem mixture_probabilityBool_le_of_components
    {Prefix : Type uProverSeed}
    {Outcome : Type uTargetSeed}
    (mixture : Mixture Prefix Outcome)
    (event : Outcome -> Bool)
    (bound : Rat)
    (componentBound : forall outer,
      outer ∈ mixture.prefixes.values ->
        (mixture.component outer).probabilityBool event <= bound) :
    mixture.probabilityBool event <= bound := by
  rw [← mixture.probability_bool_event]
  apply Mixture.probability_le_of_components
  intro outer member
  rw [(mixture.component outer).probability_bool_event]
  exact componentBound outer member

private def wordPoint
    {Extension : Type uExtension}
    {variables : Nat}
    (word : Fin variables -> Extension) :
    CubePoint Extension variables where
  coordinates := List.ofFn word
  dimension := by simp

private theorem sourceDegree_eq_width
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (exact : PaperDegreeWidthExact context)
    (witness : OutputWitness shape columns) :
    ProtocolPolynomial.VerifierInput.sumcheckDegreeBound
        (context.statement.sourceProtocolData context.lift witness).toVerifierInput =
      context.sumcheckWidth := by
  rw [context.statement.sourceProtocolData_toVerifierInput
    context.lift witness]
  exact exact

/-- Total list-to-word projection used only to make the process state a
round-independent type. Reachable states have the exact length, so the
fallback is unreachable in every transported protocol event. -/
private def prefixWord
    {Extension : Type uExtension}
    (zero : Extension)
    (round : Nat)
    (priorChallenges : List Extension) :
    Fin round -> Extension :=
  fun index => (priorChallenges[index.val]?).getD zero

private theorem ofFn_prefixWord
    {Extension : Type uExtension}
    (zero : Extension)
    (round : Nat)
    (priorChallenges : List Extension)
    (length : priorChallenges.length = round) :
    List.ofFn (prefixWord zero round priorChallenges) = priorChallenges := by
  apply List.ext_get
  · simp [length]
  · intro index leftLt rightLt
    simp only [List.get_eq_getElem, List.getElem_ofFn, prefixWord]
    rw [List.getElem?_eq_getElem (by omega)]
    rfl

private theorem wordPrefix_eq_prefixWord
    {Extension : Type uExtension}
    (zero : Extension)
    {rounds : Nat}
    (word : Fin rounds -> Extension)
    (before : List Extension)
    (challenge : Extension)
    (after : List Extension)
    (wordList :
      List.ofFn word = before ++ challenge :: after) :
    (fun index : Fin before.length =>
      word (Fin.castLT index (by
        have wordLength := List.length_ofFn (f := word)
        rw [wordList] at wordLength
        simp only [List.length_append, List.length_cons]
          at wordLength
        omega))) =
      prefixWord zero before.length before := by
  funext index
  let fullIndex : Fin (List.ofFn word).length :=
    ⟨index.val, by
      rw [wordList]
      simp only [List.length_append, List.length_cons]
      omega⟩
  have transferred := List.get_of_eq wordList fullIndex
  have beforeValue :
      (before ++ challenge :: after)[index.val] =
        before[index.val] := by
    rw [List.getElem_append_left]
  change word _ = (before[index.val]?).getD zero
  rw [List.getElem?_eq_getElem index.isLt]
  calc
    word _ = (List.ofFn word).get fullIndex := by
      simp only [List.get_eq_getElem, List.getElem_ofFn]
      congr 1
    _ = (before ++ challenge :: after).get
        ⟨fullIndex.val, wordList ▸ fullIndex.isLt⟩ := transferred
    _ = before[index.val] := by
      simp [fullIndex, List.get_eq_getElem] at beforeValue ⊢

private noncomputable def expectedPolynomial
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (exact : PaperDegreeWidthExact context)
    (witness : OutputWitness shape columns)
    (alpha : CubePoint Extension shape.cubeVariables)
    (gamma : Extension)
    (round : Nat)
    (within : round < shape.cubeVariables)
    (priorChallenges : List Extension) :
    FixedPolynomial Extension context.sumcheckWidth := by
  let data := context.statement.sourceProtocolData context.lift witness
  let word := prefixWord context.extensionOps.zero round priorChallenges
  have represented :=
    ProtocolPolynomialDegree.sequentialRoundRepresentable
      context.extensionOps context.extensionLaws data alpha gamma
      (List.ofFn word)
      (shape.cubeVariables - (round + 1)) (by
        simp only [List.length_ofFn]
        omega)
  rw [sourceDegree_eq_width context exact witness] at represented
  exact Classical.choose represented

private theorem expectedPolynomial_represents
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (exact : PaperDegreeWidthExact context)
    (witness : OutputWitness shape columns)
    (alpha : CubePoint Extension shape.cubeVariables)
    (gamma : Extension)
    (round : Nat)
    (within : round < shape.cubeVariables)
    (priorChallenges : List Extension) :
    SumCheck.Finite.FixedPhase.Represents context.extensionOps.toOps
      (expectedPolynomial context exact witness alpha gamma round within
        priorChallenges)
      (fun point =>
        SumCheck.Finite.HypercubeTruth.sumCompletions
          context.extensionOps.toOps
          (ProtocolPolynomial.polynomial context.extensionOps
            (context.statement.sourceProtocolData context.lift witness)
            alpha gamma)
          (List.ofFn
            (prefixWord context.extensionOps.zero round priorChallenges) ++
              [point])
          (shape.cubeVariables - (round + 1))) := by
  let data := context.statement.sourceProtocolData context.lift witness
  let word := prefixWord context.extensionOps.zero round priorChallenges
  have represented :=
    ProtocolPolynomialDegree.sequentialRoundRepresentable
      context.extensionOps context.extensionLaws data alpha gamma
      (List.ofFn word)
      (shape.cubeVariables - (round + 1)) (by
        simp only [List.length_ofFn]
        omega)
  rw [sourceDegree_eq_width context exact witness] at represented
  exact Classical.choose_spec represented

/-- Exact causal process for one fixed prover tape, alpha/gamma prefix, and
post-protocol witness. Its state is only the prior challenge list. -/
private noncomputable def process
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (exact : PaperDegreeWidthExact context)
    (strategy : Strategy Extension shape ProverTape)
    (proverTape : ProverTape)
    (witness : OutputWitness shape columns)
    (alpha : CubePoint Extension shape.cubeVariables)
    (gamma : Extension) :
    CausalSumCheckBound.Process Extension context.sumcheckWidth
      shape.cubeVariables where
  State := fun _ => List Extension
  initial := []
  polynomials := fun round within priorChallenges =>
    let word := prefixWord context.extensionOps.zero round priorChallenges
    let history := replayPrefix strategy proverTape alpha gamma round
      (Nat.le_of_lt within) word
    let message :=
      strategy.roundMessage ⟨round, within⟩ proverTape alpha gamma history
    match SumCheck.Finite.FixedPhase.RawCertificate.decodeMessage
        context.sumcheckWidth message with
    | none => none
    | some claimed =>
        some (claimed,
          expectedPolynomial context exact witness alpha gamma round within
            priorChallenges)
  advance := fun _ _ priorChallenges challenge =>
    priorChallenges ++ [challenge]

private noncomputable def verifierDetects
    {Extension : Type uExtension}
    [DecidableEq Extension]
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (exact : PaperDegreeWidthExact context)
    (strategy : Strategy Extension shape ProverTape)
    (proverTape : ProverTape)
    (witness : OutputWitness shape columns)
    (seed : VerifierCoins.Seed Extension shape.cubeVariables) : Bool :=
  CausalSumCheckBound.detects context.extensionOps
    (process context exact strategy proverTape witness
      (wordPoint (VerifierCoins.alphaWord seed))
      (VerifierCoins.gamma seed))
    (VerifierCoins.roundWord seed)

/-- Root counting and recursive Cartesian enumeration bound the causal
monitor after averaging over the independent alpha and gamma coordinates.
Only the SumCheck round word is charged; the outer averages do not multiply
the loss. -/
theorem verifierDetects_probability_le
    {Extension : Type uExtension}
    [DecidableEq Extension]
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (exact : PaperDegreeWidthExact context)
    (noZeroDivisors :
      FiniteRootCounting.NoZeroDivisors context.extensionOps)
    (strategy : Strategy Extension shape ProverTape)
    (proverTape : ProverTape)
    (witness : OutputWitness shape columns)
    (alphabet : Support Extension) :
    ((VerifierCoins.support alphabet shape.cubeVariables).uniform
      ).probabilityBool
        (verifierDetects context exact strategy proverTape witness) <=
      ratio (shape.cubeVariables * context.sumcheckWidth)
        alphabet.cardinality := by
  let words :=
    Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords.Support.challengeVectors
      alphabet shape.cubeVariables
  let bound :=
    ratio (shape.cubeVariables * context.sumcheckWidth)
      alphabet.cardinality
  let bad :=
    verifierDetects context exact strategy proverTape witness
  let gammaMixture :
      (VerifierCoins.Word Extension shape.cubeVariables) ->
        Mixture Extension
          (VerifierCoins.Seed Extension shape.cubeVariables) :=
    fun alphaWord => {
      prefixes := alphabet
      component := fun gamma => {
        Seed := VerifierCoins.Word Extension shape.cubeVariables
        support := words
        outcome := fun roundWord => (alphaWord, (gamma, roundWord))
      }
    }
  have gammaBound :
      forall alphaWord,
        (gammaMixture alphaWord).probabilityBool bad <= bound := by
    intro alphaWord
    apply mixture_probabilityBool_le_of_components
    intro gamma _member
    change
      words.uniform.probabilityBool
          (fun roundWord => bad (alphaWord, (gamma, roundWord))) <=
        bound
    simpa only [bad, bound, verifierDetects, VerifierCoins.alphaWord,
      VerifierCoins.gamma, VerifierCoins.roundWord] using
      CausalSumCheckBound.probability_detects_le_ratio
        context.extensionOps context.extensionLaws noZeroDivisors
        (process context exact strategy proverTape witness
          (wordPoint alphaWord) gamma)
        alphabet
  let alphaMixture :
      Mixture
        (VerifierCoins.Word Extension shape.cubeVariables)
        (VerifierCoins.Seed Extension shape.cubeVariables) := {
    prefixes := words
    component := fun alphaWord => {
      Seed :=
        Extension × VerifierCoins.Word Extension shape.cubeVariables
      support := alphabet.product words
      outcome := fun gammaAndRounds => (alphaWord, gammaAndRounds)
    }
  }
  have alphaComponentBound :
      forall alphaWord, alphaWord ∈ alphaMixture.prefixes.values ->
        (alphaMixture.component alphaWord).probabilityBool bad <= bound := by
    intro alphaWord _member
    have productEquality :=
      Mixture.sharedSupport_probabilityBool_eq_product
        alphabet words
        (fun gamma roundWord => (alphaWord, (gamma, roundWord)))
        bad
    change
      ((alphabet.product words).uniform).probabilityBool
          (fun gammaAndRounds =>
            bad (alphaWord, gammaAndRounds)) <= bound
    calc
      _ = (gammaMixture alphaWord).probabilityBool bad := by
        simpa only [gammaMixture] using productEquality.symm
      _ <= bound := gammaBound alphaWord
  have alphaBound :
      alphaMixture.probabilityBool bad <= bound :=
    mixture_probabilityBool_le_of_components alphaMixture bad bound
      alphaComponentBound
  have verifierProductEquality :=
    Mixture.sharedSupport_probabilityBool_eq_product
      words (alphabet.product words)
      (fun alphaWord gammaAndRounds => (alphaWord, gammaAndRounds))
      bad
  change
    ((VerifierCoins.support alphabet shape.cubeVariables).uniform
      ).probabilityBool bad <= bound
  calc
    _ = alphaMixture.probabilityBool bad := by
      simpa only [VerifierCoins.support, alphaMixture, words] using
        verifierProductEquality.symm
    _ <= bound := alphaBound

private theorem detectsListFrom_true_of_current
    {Extension : Type uExtension}
    [DecidableEq Extension]
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (exact : PaperDegreeWidthExact context)
    (strategy : Strategy Extension shape ProverTape)
    (proverTape : ProverTape)
    (witness : OutputWitness shape columns)
    (alpha : CubePoint Extension shape.cubeVariables)
    (gamma : Extension)
    (priorState before : List Extension)
    (challenge : Extension)
    (after : List Extension)
    (round : Nat)
    (stateLength : priorState.length = round)
    (total :
      round + (before ++ challenge :: after).length =
        shape.cubeVariables)
    (bad :
      CausalSumCheckBound.currentBad context.extensionOps
        (process context exact strategy proverTape witness alpha gamma)
        (round + before.length) (by
          simp only [List.length_append, List.length_cons]
            at total
          omega)
        (priorState ++ before) challenge = true) :
    CausalSumCheckBound.detectsListFrom context.extensionOps
      (process context exact strategy proverTape witness alpha gamma)
      round (before ++ challenge :: after) total priorState = true := by
  induction before generalizing round priorState with
  | nil =>
      simp only [List.nil_append] at bad ⊢
      unfold CausalSumCheckBound.detectsListFrom
      rw [Bool.or_eq_true]
      exact Or.inl (by simpa using bad)
  | cons head before inductionHypothesis =>
      have tailTotal :
          (round + 1) + (before ++ challenge :: after).length =
            shape.cubeVariables := by
        simp only [List.length_append, List.length_cons]
          at total ⊢
        omega
      have nextLength :
          (priorState ++ [head]).length = round + 1 := by
        simp [stateLength]
      have tailBad :
          CausalSumCheckBound.currentBad context.extensionOps
            (process context exact strategy proverTape witness alpha gamma)
            ((round + 1) + before.length) (by
              simp only [List.length_append, List.length_cons]
                at tailTotal
              omega)
            ((priorState ++ [head]) ++ before) challenge = true := by
        have roundEqual :
            round + (before.length + 1) =
              (round + 1) + before.length := by
          omega
        simp only [List.length_cons] at bad
        have stateEqual :
            priorState ++ head :: before =
              (priorState ++ [head]) ++ before := by
          simp [List.append_assoc]
        have eventEqual :=
          CausalSumCheckBound.currentBad_congr context.extensionOps
            (process context exact strategy proverTape witness alpha gamma)
            roundEqual (by
              simp only [List.length_append, List.length_cons]
                at total
              omega) (by
              simp only [List.length_append, List.length_cons]
                at tailTotal
              omega)
            (priorState ++ head :: before)
            ((priorState ++ [head]) ++ before)
            (by
              rw [stateEqual]
              exact HEq.rfl) challenge
        rw [← eventEqual]
        exact bad
      have tailResult :=
        inductionHypothesis (round := round + 1)
          (priorState := priorState ++ [head]) nextLength tailTotal tailBad
      let within : round < shape.cubeVariables := by
        simp only [List.length_append, List.length_cons]
          at total
        omega
      have step :
          CausalSumCheckBound.detectsListFrom context.extensionOps
              (process context exact strategy proverTape witness alpha gamma)
              round (head :: before ++ challenge :: after) total
              priorState =
            (CausalSumCheckBound.currentBad context.extensionOps
                (process context exact strategy proverTape witness alpha gamma)
                round within priorState head ||
              CausalSumCheckBound.detectsListFrom context.extensionOps
                (process context exact strategy proverTape witness alpha gamma)
                (round + 1) (before ++ challenge :: after) tailTotal
                ((process context exact strategy proverTape witness alpha gamma
                  ).advance round within priorState head)) := by
        rfl
      rw [step, Bool.or_eq_true]
      exact Or.inr (by simpa [process] using tailResult)

/-- Exact event transport from the repository's submitted raw SumCheck
certificate to the causal root-counting monitor. The selected round message
is decoded from the same execution transcript; no replacement certificate or
post-challenge prover choice is introduced. -/
theorem sumCheckFailure_implies_detects
    {Extension : Type uExtension}
    [DecidableEq Extension]
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (exact : PaperDegreeWidthExact context)
    (strategy : Strategy Extension shape ProverTape)
    (proverTape : ProverTape)
    (witness : OutputWitness shape columns)
    (coinSeed : VerifierCoins.Seed Extension shape.cubeVariables)
    (failure :
      SumCheckFailure context
        (execute strategy proverTape (VerifierCoins.toPublicCoins coinSeed))
        witness) :
    CausalSumCheckBound.detects context.extensionOps
      (process context exact strategy proverTape witness
        (VerifierCoins.toPublicCoins coinSeed).alpha
        (VerifierCoins.toPublicCoins coinSeed).gamma)
      (VerifierCoins.roundWord coinSeed) = true := by
  classical
  unfold SumCheckFailure StrongReduction.FixedWidthSumCheckFailure at failure
  rcases failure with ⟨certificate, decoded, collision⟩
  obtain ⟨beforeChallenges, challenge, afterChallenges,
      beforePolynomials, claimedPolynomial, afterPolynomials,
      challengesEqual, polynomialsEqual, prefixLengths,
      functionsDifferent, valuesEqual⟩ :=
    SumCheck.Finite.FixedPhase.badChallenge_implies_causal_decomposition
      context.extensionOps.toOps
      (ProtocolPolynomial.polynomial context.extensionOps
        (context.statement.sourceProtocolData context.lift witness)
        (VerifierCoins.toPublicCoins coinSeed).alpha
        (VerifierCoins.toPublicCoins coinSeed).gamma)
      context.challengeSetSize
      ((context.statement.sourceProtocolData context.lift witness
        ).toVerifierInput.initial context.extensionOps
          (VerifierCoins.toPublicCoins coinSeed).gamma)
      (VerifierCoins.toPublicCoins coinSeed).roundPoint.coordinates
      certificate collision
  have wordList :
      List.ofFn (VerifierCoins.roundWord coinSeed) =
        beforeChallenges ++ challenge :: afterChallenges := by
    simpa only [VerifierCoins.toPublicCoins_round_coordinates] using
      challengesEqual
  have beforeWithin :
      beforeChallenges.length < shape.cubeVariables := by
    have wordLength :=
      List.length_ofFn (f := VerifierCoins.roundWord coinSeed)
    rw [wordList] at wordLength
    simp only [List.length_append, List.length_cons]
      at wordLength
    omega
  have totalLength :
      (beforeChallenges ++ challenge :: afterChallenges).length =
        shape.cubeVariables := by
    have wordLength :=
      List.length_ofFn (f := VerifierCoins.roundWord coinSeed)
    rw [wordList] at wordLength
    exact wordLength
  have afterLength :
      shape.cubeVariables - (beforeChallenges.length + 1) =
        afterChallenges.length := by
    simp only [List.length_append, List.length_cons]
      at totalLength
    omega
  let causalRun :=
    execute strategy proverTape (VerifierCoins.toPublicCoins coinSeed)
  have rawRoundsLength :
      causalRun.probe.response.rounds.rounds.length =
        shape.cubeVariables := by
    rw [execute_rounds_eq_history]
    exact causalRun.history.messages_length
  let rawIndex : Fin causalRun.probe.response.rounds.rounds.length :=
    ⟨beforeChallenges.length, rawRoundsLength ▸ beforeWithin⟩
  have decodedRounds :=
    SumCheck.Finite.FixedPhase.RawCertificate.decode_eq_some_implies_rounds
      context.sumcheckWidth causalRun.probe.response.rounds certificate
      decoded
  have selectedDecoded :=
    SumCheck.Finite.FixedPhase.RawCertificate.decodeRounds_eq_some_get
      context.sumcheckWidth causalRun.probe.response.rounds.rounds
      certificate.rounds decodedRounds rawIndex
  have certificateAtIndex :
      certificate.rounds.get
          (Fin.cast
            (SumCheck.Finite.FixedPhase.RawCertificate.decodeRounds_eq_some_implies_length
                context.sumcheckWidth causalRun.probe.response.rounds.rounds
                certificate.rounds decodedRounds)
            rawIndex) =
        claimedPolynomial := by
    let fixedIndex : Fin certificate.rounds.length :=
      Fin.cast
        (SumCheck.Finite.FixedPhase.RawCertificate.decodeRounds_eq_some_implies_length
            context.sumcheckWidth causalRun.probe.response.rounds.rounds
            certificate.rounds decodedRounds)
        rawIndex
    calc
      certificate.rounds.get fixedIndex =
          (beforePolynomials ++ claimedPolynomial :: afterPolynomials).get
            ⟨fixedIndex.val, polynomialsEqual ▸ fixedIndex.isLt⟩ :=
        List.get_of_eq polynomialsEqual fixedIndex
      _ = claimedPolynomial := by
        simp only [fixedIndex, rawIndex, Fin.cast, prefixLengths,
          List.get_eq_getElem]
        rw [List.getElem_append_right (by omega)]
        simp
  have priorWordEqual :=
    wordPrefix_eq_prefixWord context.extensionOps.zero
      (VerifierCoins.roundWord coinSeed) beforeChallenges challenge
      afterChallenges wordList
  have selectedRawMessage :
      causalRun.probe.response.rounds.rounds.get rawIndex =
        strategy.roundMessage
          ⟨beforeChallenges.length, beforeWithin⟩ proverTape
          (VerifierCoins.toPublicCoins coinSeed).alpha
          (VerifierCoins.toPublicCoins coinSeed).gamma
          (replayPrefix strategy proverTape
            (VerifierCoins.toPublicCoins coinSeed).alpha
            (VerifierCoins.toPublicCoins coinSeed).gamma
            beforeChallenges.length (Nat.le_of_lt beforeWithin)
            (prefixWord context.extensionOps.zero beforeChallenges.length
              beforeChallenges)) := by
    have historyEqual :=
      execute_history_eq_replayPrefix strategy proverTape
        (VerifierCoins.toPublicCoins coinSeed)
        (VerifierCoins.roundWord coinSeed)
        (VerifierCoins.toPublicCoins_round_coordinates coinSeed)
    simp only [rawIndex, List.get_eq_getElem]
    unfold causalRun
    simp only [execute_rounds_eq_history]
    simp only [historyEqual, replayPrefix_messages, List.getElem_ofFn]
    rw [priorWordEqual]
    congr 1
  have selectedMessageDecoded :
      SumCheck.Finite.FixedPhase.RawCertificate.decodeMessage
          context.sumcheckWidth
          (strategy.roundMessage
            ⟨beforeChallenges.length, beforeWithin⟩ proverTape
            (VerifierCoins.toPublicCoins coinSeed).alpha
            (VerifierCoins.toPublicCoins coinSeed).gamma
            (replayPrefix strategy proverTape
              (VerifierCoins.toPublicCoins coinSeed).alpha
              (VerifierCoins.toPublicCoins coinSeed).gamma
              beforeChallenges.length (Nat.le_of_lt beforeWithin)
              (prefixWord context.extensionOps.zero
                beforeChallenges.length beforeChallenges))) =
        some claimedPolynomial := by
    rw [← selectedRawMessage, selectedDecoded, certificateAtIndex]
  have processPolynomials :
      (process context exact strategy proverTape witness
        (VerifierCoins.toPublicCoins coinSeed).alpha
        (VerifierCoins.toPublicCoins coinSeed).gamma).polynomials
          beforeChallenges.length beforeWithin beforeChallenges =
        some
          (claimedPolynomial,
            expectedPolynomial context exact witness
              (VerifierCoins.toPublicCoins coinSeed).alpha
              (VerifierCoins.toPublicCoins coinSeed).gamma
              beforeChallenges.length beforeWithin beforeChallenges) := by
    change
      (match
          SumCheck.Finite.FixedPhase.RawCertificate.decodeMessage
            context.sumcheckWidth
            (strategy.roundMessage
              ⟨beforeChallenges.length, beforeWithin⟩ proverTape
              (VerifierCoins.toPublicCoins coinSeed).alpha
              (VerifierCoins.toPublicCoins coinSeed).gamma
              (replayPrefix strategy proverTape
                (VerifierCoins.toPublicCoins coinSeed).alpha
                (VerifierCoins.toPublicCoins coinSeed).gamma
                beforeChallenges.length (Nat.le_of_lt beforeWithin)
                (prefixWord context.extensionOps.zero
                  beforeChallenges.length beforeChallenges))) with
        | none => none
        | some claimed =>
            some
              (claimed,
                expectedPolynomial context exact witness
                  (VerifierCoins.toPublicCoins coinSeed).alpha
                  (VerifierCoins.toPublicCoins coinSeed).gamma
                  beforeChallenges.length beforeWithin beforeChallenges)) =
        _
    rw [selectedMessageDecoded]
  have semanticFunction :
      (fun point =>
        (expectedPolynomial context exact witness
          (VerifierCoins.toPublicCoins coinSeed).alpha
          (VerifierCoins.toPublicCoins coinSeed).gamma
          beforeChallenges.length beforeWithin beforeChallenges).evaluate
            context.extensionOps.toOps point) =
        (fun point =>
          SumCheck.Finite.HypercubeTruth.sumCompletions
            context.extensionOps.toOps
            (ProtocolPolynomial.polynomial context.extensionOps
              (context.statement.sourceProtocolData context.lift witness)
              (VerifierCoins.toPublicCoins coinSeed).alpha
              (VerifierCoins.toPublicCoins coinSeed).gamma)
            (beforeChallenges ++ [point]) afterChallenges.length) := by
    funext point
    rw [expectedPolynomial_represents context exact witness
      (VerifierCoins.toPublicCoins coinSeed).alpha
      (VerifierCoins.toPublicCoins coinSeed).gamma
      beforeChallenges.length beforeWithin beforeChallenges point]
    rw [ofFn_prefixWord context.extensionOps.zero beforeChallenges.length
      beforeChallenges rfl, afterLength]
  have selectedDifferent :
      (fun point => claimedPolynomial.evaluate context.extensionOps.toOps point) ≠
        (fun point =>
          (expectedPolynomial context exact witness
            (VerifierCoins.toPublicCoins coinSeed).alpha
            (VerifierCoins.toPublicCoins coinSeed).gamma
            beforeChallenges.length beforeWithin beforeChallenges).evaluate
              context.extensionOps.toOps point) := by
    intro equal
    exact functionsDifferent (equal.trans semanticFunction)
  have selectedCollision :
      claimedPolynomial.evaluate context.extensionOps.toOps challenge =
        (expectedPolynomial context exact witness
          (VerifierCoins.toPublicCoins coinSeed).alpha
          (VerifierCoins.toPublicCoins coinSeed).gamma
          beforeChallenges.length beforeWithin beforeChallenges).evaluate
            context.extensionOps.toOps challenge := by
    calc
      _ =
          SumCheck.Finite.HypercubeTruth.sumCompletions
            context.extensionOps.toOps
            (ProtocolPolynomial.polynomial context.extensionOps
              (context.statement.sourceProtocolData context.lift witness)
              (VerifierCoins.toPublicCoins coinSeed).alpha
              (VerifierCoins.toPublicCoins coinSeed).gamma)
            (beforeChallenges ++ [challenge]) afterChallenges.length :=
        valuesEqual
      _ = _ := (congrFun semanticFunction challenge).symm
  have currentIsBad :
      CausalSumCheckBound.currentBad context.extensionOps
        (process context exact strategy proverTape witness
          (VerifierCoins.toPublicCoins coinSeed).alpha
          (VerifierCoins.toPublicCoins coinSeed).gamma)
        beforeChallenges.length beforeWithin beforeChallenges challenge =
          true := by
    unfold CausalSumCheckBound.currentBad
    rw [processPolynomials]
    simp only
    rw [if_pos selectedDifferent]
    simpa only [decide_eq_true_eq] using selectedCollision
  have listDetection :=
    detectsListFrom_true_of_current context exact strategy proverTape witness
      (VerifierCoins.toPublicCoins coinSeed).alpha
      (VerifierCoins.toPublicCoins coinSeed).gamma
      [] beforeChallenges challenge afterChallenges 0 rfl (by
        simpa only [Nat.zero_add] using totalLength)
      (by
        have eventEqual :=
          CausalSumCheckBound.currentBad_congr context.extensionOps
            (process context exact strategy proverTape witness
              (VerifierCoins.toPublicCoins coinSeed).alpha
              (VerifierCoins.toPublicCoins coinSeed).gamma)
            (leftRound := beforeChallenges.length)
            (rightRound := 0 + beforeChallenges.length)
            (by omega) beforeWithin (by omega)
            beforeChallenges ([] ++ beforeChallenges) (by
              exact HEq.rfl) challenge
        exact eventEqual.symm.trans currentIsBad)
  rw [CausalSumCheckBound.detects_eq_detectsListFrom]
  simpa [wordList, process] using listDetection

/-- Seed-level causal monitor for the exact operational experiment. The
post-prefix target seed is deliberately absent. -/
private noncomputable def runDetects
    {Extension : Type uExtension}
    [DecidableEq Extension]
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {TargetSeed : Type uTargetSeed}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (exact : PaperDegreeWidthExact context)
    (adversary :
      OperationalExperiment.Adversary context ProverSeed TargetSeed ProverTape)
    (witness : OutputWitness shape columns)
    (seed :
      OperationalExperiment.RunSeed Extension shape ProverSeed TargetSeed) :
    Bool :=
  verifierDetects context exact adversary.strategy
    (adversary.proverTape seed.1) witness seed.2.2

/-- The named repository SumCheck event is bounded in the literal
prover-by-target-by-verifier operational experiment. The challenge-size
receipt aligns the context metadata with the actual finite support; the
probability denominator is computed from that support. -/
theorem sumCheckBadChallenge_probability_le
    {Extension : Type uExtension}
    [DecidableEq Extension]
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {TargetSeed : Type uTargetSeed}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (exact : PaperDegreeWidthExact context)
    (noZeroDivisors :
      FiniteRootCounting.NoZeroDivisors context.extensionOps)
    (alphabet : Support Extension)
    (challengeSetSize_eq :
      context.challengeSetSize = alphabet.cardinality)
    (adversary :
      OperationalExperiment.Adversary context ProverSeed TargetSeed ProverTape)
    (witness : OutputWitness shape columns) :
    (OperationalExperiment.experiment context alphabet adversary
      ).probabilityBool
        (SecurityContracts.sumCheckBadChallengeEvent context witness) <=
      ratio (shape.cubeVariables * context.sumcheckWidth)
        alphabet.cardinality := by
  let verifierSupport :=
    VerifierCoins.support alphabet shape.cubeVariables
  let bound :=
    ratio (shape.cubeVariables * context.sumcheckWidth)
      alphabet.cardinality
  let monitor := runDetects context exact adversary witness
  let proverMixture :
      Mixture ProverSeed
        (OperationalExperiment.RunSeed Extension shape
          ProverSeed TargetSeed) := {
    prefixes := adversary.proverSupport
    component := fun proverSeed => {
      Seed :=
        TargetSeed ×
          VerifierCoins.Seed Extension shape.cubeVariables
      support := adversary.targetSupport.product verifierSupport
      outcome := fun targetAndVerifier =>
        (proverSeed, targetAndVerifier)
    }
  }
  have proverComponentBound :
      forall proverSeed,
        proverSeed ∈ proverMixture.prefixes.values ->
          (proverMixture.component proverSeed).probabilityBool monitor <=
            bound := by
    intro proverSeed _member
    change
      ((adversary.targetSupport.product verifierSupport).uniform
        ).probabilityBool
          (fun targetAndVerifier =>
            monitor (proverSeed, targetAndVerifier)) <= bound
    calc
      _ =
          verifierSupport.uniform.probabilityBool
            (fun verifierSeed =>
              verifierDetects context exact adversary.strategy
                (adversary.proverTape proverSeed) witness verifierSeed) := by
        simpa only [monitor, runDetects] using
          Support.product_uniform_probabilityBool_second
            adversary.targetSupport verifierSupport
            (fun verifierSeed =>
              verifierDetects context exact adversary.strategy
                (adversary.proverTape proverSeed) witness verifierSeed)
      _ <= bound := by
        simpa only [verifierSupport, bound] using
          verifierDetects_probability_le context exact noZeroDivisors
            adversary.strategy (adversary.proverTape proverSeed) witness
            alphabet
  have proverBound :
      proverMixture.probabilityBool monitor <= bound :=
    mixture_probabilityBool_le_of_components proverMixture monitor bound
      proverComponentBound
  have runProductEquality :=
    Mixture.sharedSupport_probabilityBool_eq_product
      adversary.proverSupport
      (adversary.targetSupport.product verifierSupport)
      (fun proverSeed targetAndVerifier =>
        (proverSeed, targetAndVerifier))
      monitor
  have monitorBound :
      ((OperationalExperiment.runSupport context alphabet adversary).uniform
        ).probabilityBool monitor <= bound := by
    calc
      _ = proverMixture.probabilityBool monitor := by
        simpa only [OperationalExperiment.runSupport, verifierSupport,
          proverMixture] using runProductEquality.symm
      _ <= bound := proverBound
  change
    ((OperationalExperiment.runSupport context alphabet adversary).uniform
      ).probabilityBool
        (fun seed =>
          SecurityContracts.sumCheckBadChallengeEvent context witness
            (OperationalExperiment.run context adversary seed)) <= bound
  calc
    _ <=
        ((OperationalExperiment.runSupport context alphabet adversary).uniform
          ).probabilityBool monitor := by
      apply Experiment.probabilityBool_mono
      intro seed eventTrue
      have failure :
          SumCheckFailure context
            (execute adversary.strategy (adversary.proverTape seed.1)
              (VerifierCoins.toPublicCoins seed.2.2)) witness := by
        have namedFailure :=
          (SecurityContracts.sumCheckBadChallengeEvent_eq_true_iff
            context witness
            (OperationalExperiment.run context adversary seed)).mp eventTrue
        simpa only [OperationalExperiment.run_causalRun] using namedFailure
      have alignedFailure :
          StrongReduction.FixedWidthSumCheckFailure context.extensionOps
            context.lift context.statement context.sumcheckWidth
            alphabet.cardinality
            (execute adversary.strategy (adversary.proverTape seed.1)
              (VerifierCoins.toPublicCoins seed.2.2)).probe witness := by
        simpa only [SumCheckFailure, challengeSetSize_eq] using failure
      have failureAgain :
          SumCheckFailure context
            (execute adversary.strategy (adversary.proverTape seed.1)
              (VerifierCoins.toPublicCoins seed.2.2)) witness := by
        simpa only [SumCheckFailure, challengeSetSize_eq] using alignedFailure
      have detected :=
        sumCheckFailure_implies_detects context exact adversary.strategy
          (adversary.proverTape seed.1) witness seed.2.2 failureAgain
      simpa only [monitor, runDetects, verifierDetects, wordPoint,
        VerifierCoins.toPublicCoins, VerifierCoins.alphaWord,
        VerifierCoins.gamma, VerifierCoins.roundWord] using detected
    _ <= bound := monitorBound

/-- Concrete construction of the existing repository contract from the
paper-owned degree invariant and finite field/sampling laws. In particular,
`SumCheckSoundnessContract` is not an assumption. -/
theorem sumCheckSoundnessContract_of_rootCounting
    {Extension : Type uExtension}
    [DecidableEq Extension]
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {TargetSeed : Type uTargetSeed}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (exact : PaperDegreeWidthExact context)
    (noZeroDivisors :
      FiniteRootCounting.NoZeroDivisors context.extensionOps)
    (alphabet : Support Extension)
    (challengeSetSize_eq :
      context.challengeSetSize = alphabet.cardinality)
    (adversary :
      OperationalExperiment.Adversary context ProverSeed TargetSeed ProverTape) :
    SecurityContracts.SumCheckSoundnessContract context alphabet adversary
      (ratio (shape.cubeVariables * context.sumcheckWidth)
        alphabet.cardinality) := by
  intro witness
  exact sumCheckBadChallenge_probability_le context exact noZeroDivisors
    alphabet challengeSetSize_eq adversary witness

/-- Legacy floor-based extraction with the concrete SumCheck loss inserted.
The only remaining probabilistic premise is the separate alpha/gamma mixing
contract. This is not the corrected paper-facing extractor. -/
theorem extraction_after_first_success_of_rootCounting
    {Extension : Type uExtension}
    [DecidableEq Extension]
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {TargetSeed : Type uTargetSeed}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (exact : PaperDegreeWidthExact context)
    (noZeroDivisors :
      FiniteRootCounting.NoZeroDivisors context.extensionOps)
    (alphabet : Support Extension)
    (challengeSetSize_eq :
      context.challengeSetSize = alphabet.cardinality)
    (adversary :
      OperationalExperiment.Adversary context ProverSeed TargetSeed ProverTape)
    (successFloor rawMismatchBudget mixingBudget : Rat)
    (floorPos : 0 < successFloor)
    (floorBound : successFloor <=
      (OperationalExperiment.experiment context alphabet adversary
        ).probabilityBool (OperationalExperiment.success context))
    (rawMismatchBound :
      (OperationalExperiment.experiment context alphabet adversary
        ).iidPair.probabilityBool
          (OperationalEvents.witnessDisagreement context) <=
        rawMismatchBudget)
    (mixingBound :
      SecurityContracts.MixingRootProbabilityContract context alphabet
        adversary mixingBudget) :
    let sumCheckBudget :=
      ratio (shape.cubeVariables * context.sumcheckWidth)
        alphabet.cardinality
    let base :=
      OperationalExperiment.experiment context alphabet adversary
    let nonempty :
        base.support.values.filter
          (fun seed =>
            OperationalExperiment.success context (base.outcome seed)) ≠ [] :=
      OperationalExperiment.successfulSupport_nonempty_of_floor
        context alphabet adversary successFloor floorPos floorBound
    base.probabilityBool (OperationalExperiment.success context) -
          ((mixingBudget + sumCheckBudget) +
            rawMismatchBudget / successFloor) <=
      (base.firstConditionedFreshSecond
        (OperationalExperiment.success context) nonempty).probabilityBool
          (OperationalEvents.sourceExtracted context) := by
  exact
    SecurityContracts.extraction_after_first_success_of_securityContracts
      context alphabet adversary successFloor rawMismatchBudget
      mixingBudget
      (ratio (shape.cubeVariables * context.sumcheckWidth)
        alphabet.cardinality)
      floorPos floorBound rawMismatchBound mixingBound
      (sumCheckSoundnessContract_of_rootCounting context exact
        noZeroDivisors alphabet challengeSetSize_eq adversary)

/-- Corrected Appendix D.4 success-gated extraction with the concrete
SumCheck loss inserted. The raw two-run disagreement budget is charged through
a nonnegative root envelope, and no pointwise success floor is required. The
only remaining probabilistic premise is the separate alpha/gamma mixing
contract; Fiat--Shamir is not involved. -/
theorem extraction_after_success_gate_of_rootCounting
    {Extension : Type uExtension}
    [DecidableEq Extension]
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {TargetSeed : Type uTargetSeed}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (exact : PaperDegreeWidthExact context)
    (noZeroDivisors :
      FiniteRootCounting.NoZeroDivisors context.extensionOps)
    (alphabet : Support Extension)
    (challengeSetSize_eq :
      context.challengeSetSize = alphabet.cardinality)
    (adversary :
      OperationalExperiment.Adversary context ProverSeed TargetSeed ProverTape)
    (rawMismatchBudget rootMismatchBudget mixingBudget : Rat)
    (rootNonnegative : 0 <= rootMismatchBudget)
    (rawBudget_le_rootSquare :
      rawMismatchBudget <= rootMismatchBudget * rootMismatchBudget)
    (rawMismatchBound :
      (OperationalExperiment.experiment context alphabet adversary
        ).iidPair.probabilityBool
          (OperationalEvents.witnessDisagreement context) <=
        rawMismatchBudget)
    (mixingBound :
      SecurityContracts.MixingRootProbabilityContract context alphabet
        adversary mixingBudget)
    (nonempty :
      (OperationalExperiment.experiment context alphabet adversary
        ).support.values.filter
          (fun seed => OperationalExperiment.success context
            ((OperationalExperiment.experiment context alphabet adversary
              ).outcome seed)) ≠ []) :
    let sumCheckBudget :=
      ratio (shape.cubeVariables * context.sumcheckWidth)
        alphabet.cardinality
    let base :=
      OperationalExperiment.experiment context alphabet adversary
    base.probabilityBool (OperationalExperiment.success context) -
          ((mixingBudget + sumCheckBudget) + rootMismatchBudget) <=
      (base.firstConditionedFreshSecond
        (OperationalExperiment.success context) nonempty).probabilityBool
          (OperationalEvents.successGatedSourceExtracted context) := by
  exact
    SecurityContracts.extraction_after_success_gate_of_securityContracts
      context alphabet adversary rawMismatchBudget rootMismatchBudget
      mixingBudget
      (ratio (shape.cubeVariables * context.sumcheckWidth)
        alphabet.cardinality)
      rootNonnegative rawBudget_le_rootSquare rawMismatchBound mixingBound
      (sumCheckSoundnessContract_of_rootCounting context exact
        noZeroDivisors alphabet challengeSetSize_eq adversary)
      nonempty

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SumCheckSoundness
