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
  after that challenge's calls, as `TranscriptCoverage.challenge_seal` states
  for the key;
- `deployedOracle_deployed`: every execution has such an oracle, because
  distinct challenges have distinct points (`challengeCalls_injective`);
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

/-- The oracle's value of one challenge: the decoded extension element, or the
sampled `Π_RLC` scalar. -/
noncomputable def oracleValue {degree : Nat} (oracle : Point logicalWidth publicFits degree → Answer)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) : (challenge : Challenge) → challenge.Value
  | .alpha coordinate => decodeK (oracle (point fresh proof (.alpha coordinate)))
  | .gamma => decodeK (oracle (point fresh proof .gamma))
  | .round index => decodeK (oracle (point fresh proof (.round index)))
  | .rho index => RandomOracleExtraction.readRho (oracle (point fresh proof (.rho index)))

/-- The oracle answers every challenge of this execution as the sponge does. -/
def Deployed {degree : Nat} (oracle : Point logicalWidth publicFits degree → Answer)
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) : Prop :=
  ∀ challenge, oracleValue oracle fresh proof challenge =
    challenge.read (TranscriptCoverage.run Transcript.initialState
      (challengeCalls fresh proof challenge))

/-! ## Every execution has a deployed oracle -/

/-- The answer that a challenge reads from a sponge state. -/
private def stateAnswer : Challenge → Transcript.State → Answer
  | .rho _, state => Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.block state
  | _, state => fun lane =>
      if lane.val = 0 then state.getD 0 0 else (Poseidon2.permute state).getD 0 0

/-- The sponge's answer at each challenge point of one execution, and zero
elsewhere. -/
noncomputable def deployedOracle {degree : Nat}
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) : Point logicalWidth publicFits degree → Answer := fun target =>
  if found : ∃ challenge, point fresh proof challenge = target then
    stateAnswer found.choose
      (TranscriptCoverage.run Transcript.initialState (challengeCalls fresh proof found.choose))
  else fun _ => 0

private theorem deployedOracle_point {degree : Nat}
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) (challenge : Challenge) :
    deployedOracle fresh proof (point fresh proof challenge) =
      stateAnswer challenge
        (TranscriptCoverage.run Transcript.initialState (challengeCalls fresh proof challenge)) := by
  have found : ∃ other, point fresh proof other = point fresh proof challenge := ⟨challenge, rfl⟩
  have chosen : found.choose = challenge :=
    challengeCalls_injective fresh proof (congrArg Subtype.val found.choose_spec)
  unfold deployedOracle
  rw [dif_pos found, chosen]

theorem deployedOracle_deployed {degree : Nat}
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) :
    Deployed (deployedOracle fresh proof) fresh proof := by
  intro challenge
  cases challenge <;> simp only [oracleValue, deployedOracle_point] <;> rfl

/-! ## The oracle verifier at a deployed oracle -/

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  {fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
  {proof : Proof (ProductionKey.degreeBound relation)}
  {oracle : Point logicalWidth publicFits (ProductionKey.degreeBound relation) → Answer}

/-- At a deployed oracle, the oracle's `Π_CCS` coins are the key's coins. -/
theorem coins_eq (deployed : Deployed oracle fresh proof) :
    coins oracle fresh proof =
      ((ProductionKey.key relation ajtai).piCcsProbe running fresh proof).coins := by
  rw [piCcsProbe_coins relation ajtai running fresh proof]
  have alpha (coordinate : Fin productionShape.cubeVariables) :
      read oracle (challengeCalls fresh proof (.alpha coordinate)) =
        readK (TranscriptCoverage.run Transcript.initialState
          (challengeCalls fresh proof (.alpha coordinate))) :=
    (read_challengeCalls oracle fresh proof _).trans (deployed (.alpha coordinate))
  have gamma : read oracle (challengeCalls fresh proof .gamma) =
      readK (TranscriptCoverage.run Transcript.initialState (challengeCalls fresh proof .gamma)) :=
    (read_challengeCalls oracle fresh proof _).trans (deployed .gamma)
  have round (index : Fin productionShape.cubeVariables) :
      read oracle (challengeCalls fresh proof (.round index)) =
        readK (TranscriptCoverage.run Transcript.initialState
          (challengeCalls fresh proof (.round index))) :=
    (read_challengeCalls oracle fresh proof _).trans (deployed (.round index))
  simp only [coins, coinsFrom, alpha, gamma, round]

theorem oracleProbe_eq (deployed : Deployed oracle fresh proof) :
    oracleProbe relation ajtai running oracle fresh proof =
      (ProductionKey.key relation ajtai).piCcsProbe running fresh proof := by
  unfold oracleProbe
  rw [coins_eq relation ajtai running deployed]

/-- At a deployed oracle, the key's `Π_RLC` challenges are the oracle's. -/
theorem piRlcChallenges_eq (deployed : Deployed oracle fresh proof) :
    (ProductionKey.key relation ajtai).piRlcChallenges running fresh proof =
      some (rho oracle fresh proof) := by
  rw [rho_seal relation ajtai running fresh proof]
  exact congrArg some (funext fun index => (deployed (.rho index)).symm)

/-- At a deployed oracle, the key's `Π_DEC` attempt is the oracle attempt. -/
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

/-- Verifier fidelity: at a deployed oracle, the oracle verifier accepts
exactly when the production NIFS verifier returns a running output and the
witnesses open the children of its `Π_DEC` attempt. -/
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
