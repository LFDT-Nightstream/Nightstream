import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal

/-!
Owns the cross-phase verifier view of one accepted PiCCS execution.

The phase assembler already proves the transcript child specifications. This
module exposes the key-owned round point needed by the zero-copy PiRLC input
bridge without unfolding any child operation or restating PiCCS acceptance.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.VerifierView

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

theorem roundPoint_eq_key
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (offset : Nat) (env : Env)
    (template : Proof (ProductionKey.degreeBound relation))
    (specification : Formal.SpecHolds relation interface offset env) :
    RoundTranscript.evalRoundPoint
        (Formal.roundTranscriptInterface (Formal.atOffset interface offset))
        (Formal.roundTranscriptOffset interface offset) env =
      ((ProductionKey.key relation ajtai).piCcsExecution
        (Formal.evalRunning interface offset env)
        (Formal.evalFresh interface offset env)
        (Formal.evalProof relation interface offset env template)
      ).coins.roundPoint := by
  let shared := Formal.atOffset interface offset
  let running := Formal.evalRunning interface offset env
  let fresh := Formal.evalFresh interface offset env
  let proof := Formal.evalProof relation interface offset env template
  let context := ChallengeDerivation.productionContext
    relation ajtai running fresh
  have statementState := StatementAbsorption.spec_implies_keyInitialState
    relation ajtai (Formal.statementAbsorptionInterface shared)
      (Formal.statementAbsorptionOffset interface offset) env
      specification.statementAbsorption
  dsimp only at statementState
  rw [ProductionKey.key_oracle_eq relation ajtai] at statementState
  have challengeCoverage :=
    ChallengeDerivation.spec_implies_derivePreSumcheck
      (Formal.challengeInterface shared offset)
      (Formal.challengeOffset interface offset) env context (by
        simpa [shared, running, fresh, context, Formal.challengeInterface,
          Formal.statementAbsorptionInterface, Formal.atOffset,
          Formal.evalRunning, Formal.evalFresh] using statementState)
      specification.challenge
  have roundCoverage := RoundTranscript.spec_implies_keyExecution_rounds
    relation ajtai running fresh proof (Formal.roundTranscriptInterface shared)
      (Formal.roundTranscriptOffset interface offset) env (by
        simpa [shared, context, Formal.challengeInterface,
          Formal.roundTranscriptInterface, Formal.atOffset] using
            challengeCoverage.2.2)
      (by intro roundIndex; rfl) specification.roundTranscript
  exact roundCoverage.1

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.VerifierView
