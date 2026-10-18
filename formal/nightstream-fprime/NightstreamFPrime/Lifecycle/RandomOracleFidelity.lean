import NightstreamFPrime.Lifecycle.RandomOracleExtraction

/-!
Owns the fidelity of the oracle verifier: at an oracle that answers every
challenge of one execution as the Poseidon2 sponge does, the oracle verifier
`RandomOracleExtraction.Accepts` is the production NIFS verifier
`PaperNonInteractive.verify` plus the openings of the returned children.

Inputs: the production key, one execution (running and fresh statements and a
proof) and an oracle.

Outputs:
- `Deployed`: the oracle's value at each challenge point is the sponge read
  at that challenge's address, as `TranscriptCoverage.challenge_seal` states
  for the key;
- `deployedOracle_deployed`: the oracle that answers every point with the
  lanes of the sponge state after its calls is such an oracle for every
  execution;
- `attempt_eq`: the key's `Π_DEC` attempt is the oracle attempt;
- `accepts_iff_verify`: `Accepts` holds iff `verify` returns a running output
  and the witnesses open the attempt's children.

Does not own: the probability model, or the oracle's answers at points that
are not challenges of the execution.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.RandomOracleFidelity

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.TranscriptCoverage
open NightstreamFPrime.Lifecycle.RandomOracleTest
open NightstreamFPrime.Lifecycle.RandomOracleExtraction
open StrongReduction ConcreteCarrier

attribute [local instance low] Classical.propDecidable

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

noncomputable def oracleValue {degree : Nat} (oracle : Point logicalWidth publicFits degree → Answer)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) : (challenge : Challenge) → challenge.Value
  | .alpha coordinate => decodeK (oracle (point fresh proof (.alpha coordinate)))
  | .gamma => decodeK (oracle (point fresh proof .gamma))
  | .round index => decodeK (oracle (point fresh proof (.round index)))
  | .rho index => RandomOracleExtraction.readRho (oracle (point fresh proof (.rho index)))

def Deployed {degree : Nat} (oracle : Point logicalWidth publicFits degree → Answer)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) : Prop :=
  ∀ challenge, oracleValue oracle fresh proof challenge =
    challenge.read (TranscriptCoverage.run Transcript.initialState
      (challengeCalls fresh proof challenge))

private def stateAnswer (slot : Slot) (state : Transcript.State) : Answer :=
  fun word => state.getD (2 * slot.val + word.val) 0

noncomputable def deployedOracle {degree : Nat} :
    Point logicalWidth publicFits degree → Answer := fun target =>
  stateAnswer target.2 (TranscriptCoverage.run Transcript.initialState target.1.val)

theorem deployedOracle_deployed {degree : Nat}
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) :
    Deployed deployedOracle fresh proof := by
  intro challenge
  cases challenge with
  | rho index =>
      show RandomOracleExtraction.readRho (stateAnswer (slotOf 0) (TranscriptCoverage.run
          Transcript.initialState (challengeCalls fresh proof (.rho index)))) =
        TranscriptCoverage.readRho _
      have block : stateAnswer (slotOf 0) (TranscriptCoverage.run Transcript.initialState
          (challengeCalls fresh proof (.rho index))) =
          Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.block
            (TranscriptCoverage.run Transcript.initialState
              (challengeCalls fresh proof (.rho index))) := by
        funext word
        simp [stateAnswer, slotOf, Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.block]
      rw [block]
      rfl
  | _ => rfl

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  {fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
  {proof : Proof (ProductionKey.degreeBound relation)}
  {oracle : Point logicalWidth publicFits (ProductionKey.degreeBound relation) → Answer}

theorem coins_eq (deployed : Deployed oracle fresh proof) :
    coins oracle fresh proof =
      ((ProductionKey.key relation ajtai).piCcsProbe running fresh proof).coins := by
  rw [piCcsProbe_coins relation ajtai running fresh proof]
  have alpha (coordinate : Fin productionShape.cubeVariables) :
      read oracle (challengeAddress fresh proof (.alpha coordinate)) =
        readK (slotOf coordinate.val) (TranscriptCoverage.run Transcript.initialState
          (challengeCalls fresh proof (.alpha coordinate))) :=
    (read_challengeAddress oracle fresh proof _).trans (deployed (.alpha coordinate))
  have gamma : read oracle (challengeAddress fresh proof .gamma) =
      readK (slotOf productionShape.cubeVariables) (TranscriptCoverage.run
        Transcript.initialState (challengeCalls fresh proof .gamma)) :=
    (read_challengeAddress oracle fresh proof _).trans (deployed .gamma)
  have round (index : Fin productionShape.cubeVariables) :
      read oracle (challengeAddress fresh proof (.round index)) =
        readK (slotOf 0) (TranscriptCoverage.run Transcript.initialState
          (challengeCalls fresh proof (.round index))) :=
    (read_challengeAddress oracle fresh proof _).trans (deployed (.round index))
  simp only [coins, coinsFrom, alpha, gamma, round]
  rfl

theorem oracleProbe_eq (deployed : Deployed oracle fresh proof) :
    oracleProbe relation ajtai running oracle fresh proof =
      (ProductionKey.key relation ajtai).piCcsProbe running fresh proof := by
  unfold oracleProbe
  rw [coins_eq relation ajtai running deployed]

theorem piRlcChallenges_eq (deployed : Deployed oracle fresh proof) :
    (ProductionKey.key relation ajtai).piRlcChallenges running fresh proof =
      some (rho oracle fresh proof) := by
  rw [rho_seal relation ajtai running fresh proof]
  exact congrArg some (funext fun index => (deployed (.rho index)).symm)

theorem attempt_eq (deployed : Deployed oracle fresh proof) :
    (ProductionKey.key relation ajtai).piDecAttempt running fresh proof =
      some (attempt relation ajtai oracle running fresh proof) := by
  unfold Nifs.PaperNonInteractive.Key.piDecAttempt Nifs.PaperNonInteractive.Key.parent
  rw [piRlcChallenges_eq relation ajtai running deployed]
  unfold attempt batch
  rw [oracleProbe_eq relation ajtai running deployed]
  rfl

private theorem piCcsCheck_iff :
    Nifs.PaperNonInteractive.piCcsCheck (ProductionKey.key relation ajtai) running fresh proof = true ↔
      ((ProductionKey.key relation ajtai).piCcsProbe running fresh proof).FixedWidthAccepted
        extensionOps K.embed ((ProductionKey.key relation ajtai).statement running fresh)
        (ProductionKey.degreeBound relation) :=
  Nifs.PaperNonInteractive.piCcsCheck_eq_true_iff_fixedWidthAccepted _ _ _ _

private theorem piDecCheck_iff (deployed : Deployed oracle fresh proof) :
    Nifs.PaperNonInteractive.piDecCheck (ProductionKey.key relation ajtai) running fresh proof = true ↔
      PiDEC.PaperVerifier.Accepted (ProductionKey.key relation ajtai).piDecAlgebra
        (ProductionKey.key relation ajtai).piDecPublicInputSplit
        (ProductionKey.key relation ajtai).piDecEvaluationArity
        (attempt relation ajtai oracle running fresh proof) := by
  rw [Nifs.PaperNonInteractive.piDecCheck_eq_true_iff, attempt_eq relation ajtai running deployed]
  constructor
  · rintro ⟨_, same, accepted⟩
    cases Option.some.inj same
    exact accepted
  · exact fun accepted => ⟨_, rfl, accepted⟩

private theorem output_isSome (deployed : Deployed oracle fresh proof) :
    ((ProductionKey.key relation ajtai).output running fresh proof).isSome =
      ((ProductionKey.key relation ajtai).piDecPublicInputSplit.checked
        (attempt relation ajtai oracle running fresh proof).parent.publicInput).isSome := by
  have attemptEq := attempt_eq relation ajtai running deployed
  by_cases bounded : (ProductionKey.key relation ajtai).piDecPublicInputSplit.parentBounded
      (attempt relation ajtai oracle running fresh proof).parent.publicInput
  · rw [Nifs.PaperNonInteractive.Key.output_eq_some_of_parentBounded _ running fresh proof _
        attemptEq bounded,
      (ProductionKey.key relation ajtai).piDecPublicInputSplit.checked_eq_some _ bounded]
    rfl
  · rw [Nifs.PaperNonInteractive.Key.output_eq_none_of_parentUnbounded _ running fresh proof _
        attemptEq bounded,
      (ProductionKey.key relation ajtai).piDecPublicInputSplit.checked_eq_none _ bounded]
    rfl

theorem accepts_iff_verify
    (children : Fin productionShape.runningCount →
      PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
    (deployed : Deployed oracle fresh proof) :
    Accepts relation ajtai oracle running fresh proof children ↔
      (∃ next, Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
          running fresh proof = some next) ∧
        ∀ child, CE.Holds (ProductionKey.key relation ajtai).piRlcSemantics
          (ProductionKey.key relation ajtai).params
          (PiDEC.PaperVerifier.children (ProductionKey.key relation ajtai).piDecPublicInputSplit
            (attempt relation ajtai oracle running fresh proof) child)
          (children (Fin.cast (ProductionKey.key relation ajtai).outputCount_eq child)) := by
  have verified : (∃ next, Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
        running fresh proof = some next) ↔
      Nifs.PaperNonInteractive.piCcsCheck (ProductionKey.key relation ajtai) running fresh proof = true ∧
        Nifs.PaperNonInteractive.piDecCheck (ProductionKey.key relation ajtai) running fresh proof = true ∧
          ((ProductionKey.key relation ajtai).output running fresh proof).isSome := by
    simp only [Nifs.PaperNonInteractive.verify_eq_some_iff, Option.isSome_iff_exists]
    exact ⟨fun ⟨next, ccs, dec, out⟩ => ⟨ccs, dec, next, out⟩,
      fun ⟨ccs, dec, next, out⟩ => ⟨next, ccs, dec, out⟩⟩
  rw [verified, piCcsCheck_iff, piDecCheck_iff relation ajtai running deployed,
    output_isSome relation ajtai running deployed, ← oracleProbe_eq relation ajtai running deployed]
  unfold Accepts
  exact ⟨fun ⟨ccs, dec, split, opens⟩ => ⟨⟨ccs, dec, split⟩, opens⟩,
    fun ⟨⟨ccs, dec, split⟩, opens⟩ => ⟨ccs, dec, split, opens⟩⟩

end NightstreamFPrime.Lifecycle.RandomOracleFidelity
