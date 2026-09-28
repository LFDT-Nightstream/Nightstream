import NightstreamFPrime.Export.Stage1.PiRLCParent
import NightstreamFPrime.Export.Stage1.HyperNovaStepData
import NightstreamFPrime.Export.Stage1.HyperNovaInput

/-! Compose the actual checked C/R/D fields into the selected NIFS return.
The full C output is absorbed before R sampling. D's checked public fields
identify the returned running claim. Child openings and the next fresh
assignment remain separate consumers of this exact result. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.CheckedReplayNifs

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Layout.Stage1
open Poseidon2HashChainV1Setup (productionAjtaiKey)
open Nifs.PaperNonInteractive

private noncomputable abbrev selectedKey :=
  ProductionKey.key PiDECInputCheck.relation productionAjtaiKey

/-- Use exactly the supplied C polynomials/full output and D message fields.
The returned point and public inputs remain verifier-derived. -/
def proof (input : PiCCSInputCheck.Input) (messages : PiDECInputCheck.Messages) :
    Lifecycle.Proof 9 where
  piCcsRounds := PiCCSProofInputs.roundPolynomial (PiCCSInputCheck.proofValues input)
  piCcsOutput := PiCCSProofInputs.output (PiCCSInputCheck.proofValues input)
  piDecCommitments := (PiCCSInputCheck.runningFromInput messages).commitments
  piDecEvaluations := (PiCCSInputCheck.runningFromInput messages).evaluations

private abbrev pre (input : PiCCSInputCheck.Input) :=
  PiCCS.Transcript.deriveFromState Transcript.piCcsOracle.transcript
    (ProductionKey.absorbPublicInput
      (Transcript.absorb Transcript.initialState Transcript.piCcsDigestDomainTag)
      (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input))

private theorem point_ext {arity : Nat} (left right : CubePoint K arity)
    (same : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  cases same
  rfl

private theorem execution_point (input : PiCCSInputCheck.Input)
    (messages : PiDECInputCheck.Messages) :
    (selectedKey.piCcsExecution (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input)
      (proof input messages)).coins.roundPoint = (PiCCSInputCheck.execute input).point := by
  rw [Key.piCcsExecution_coins_eq_derive]
  apply point_ext
  change (FiatShamir.deriveRoundsFrom Transcript.piCcsOracle.transcript
      (fun round => (PiCCSProofInputs.roundPolynomial (PiCCSInputCheck.proofValues input) round).toMessage)
      (pre input).state (canonicalFinIndices productionShape.cubeVariables)).1 =
    (PiCCSInputCheck.traceFrom input (pre input).state
      (canonicalFinIndices productionShape.cubeVariables)).challenges
  exact (congrArg Prod.fst (PiCCSInputCheck.traceFrom_eq_derive input
    (pre input).state (canonicalFinIndices productionShape.cubeVariables))).symm

private theorem execution_finalState
    {Extension Commitment PublicInput Scalar State : Type}
    {shape : Shape} {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape columns blockCount degreeBound)
    (running : Nifs.PaperNonInteractive.Running Extension Commitment PublicInput shape)
    (fresh : Nifs.PaperNonInteractive.Fresh Commitment PublicInput shape)
    (localProof : Nifs.PaperNonInteractive.Proof Extension Commitment shape degreeBound) :
    (key.piCcsExecution running fresh localProof).coins.finalState =
      (FiatShamir.deriveRoundsFrom key.oracle.transcript
        (fun round => (localProof.piCcsRounds round).toMessage)
        (PiCCS.Transcript.deriveFromState key.oracle.transcript
          (key.publicInputState running fresh)).state
        (canonicalFinIndices shape.cubeVariables)).2 := by
  rw [Key.piCcsExecution_coins_eq_derive]
  have result := (PiCCS.Transcript.derive_rounds_holds key.oracle.transcript
    ({ priorState := key.publicInputState running fresh
       input := (key.statement running fresh).verifierInput key.lift } :
      PiCCS.TranscriptReplay.Statement Extension State shape)
    ({ rounds := fun round => (localProof.piCcsRounds round).toMessage } :
      FiatShamir.Certificate Extension shape)).finalState_eq
  rw [← PiCCS.Transcript.deriveFromState_initialState,
    key.oracle.initialState_is_prior] at result
  exact result

private theorem attempt_of_sample
    {Extension Commitment PublicInput Scalar State : Type}
    {shape : Shape} {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape columns blockCount degreeBound)
    (running : Nifs.PaperNonInteractive.Running Extension Commitment PublicInput shape)
    (fresh : Nifs.PaperNonInteractive.Fresh Commitment PublicInput shape)
    (localProof : Nifs.PaperNonInteractive.Proof Extension Commitment shape degreeBound)
    (rho : Fin key.arity.total → Scalar)
    (sampled : key.piRlcChallenges running fresh localProof = some rho) :
    key.piDecAttempt running fresh localProof =
      some (key.piDecAttemptForParent localProof
        (key.parentForChallenges running fresh localProof rho)) := by
  simp only [Key.piDecAttempt, Key.parent, sampled, Option.map]

private theorem production_oracle
    (relation : ProductionKey.LogicalRelation PiDECInputCheck.logicalWidth PiDECInputCheck.publicFits)
    (ajtai : AjtaiKey (logicalWidth := PiDECInputCheck.logicalWidth)
      (publicFits := PiDECInputCheck.publicFits)) :
    (ProductionKey.key relation ajtai).oracle = Transcript.piCcsOracle := by
  rfl

private theorem execute_outgoing (input : PiCCSInputCheck.Input) :
    (PiCCSInputCheck.execute input).outgoing =
      ProductionKey.absorbFullOutput
        (PiCCSInputCheck.traceFrom input (pre input).state
          (canonicalFinIndices productionShape.cubeVariables)).state
        (PiCCSProofInputs.output (PiCCSInputCheck.proofValues input)) := by
  rfl

private theorem execution_outgoing (input : PiCCSInputCheck.Input)
    (messages : PiDECInputCheck.Messages) :
    (selectedKey.piCcsExecution (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input)
      (proof input messages)).outgoingState = (PiCCSInputCheck.execute input).outgoing := by
  rw [Key.piCcsExecution_outgoingState_eq_absorbPiCcsOutput,
    ProductionKey.key_absorbPiCcsOutput, execute_outgoing]
  apply congrArg (fun state => ProductionKey.absorbFullOutput state
    (PiCCSProofInputs.output (PiCCSInputCheck.proofValues input)))
  rw [execution_finalState, production_oracle, ProductionKey.key_publicInputState_eq]
  simpa only [proof, pre, ProductionKey.absorbPublicInput] using!
    (congrArg Prod.snd (PiCCSInputCheck.traceFrom_eq_derive input
      (pre input).state (canonicalFinIndices productionShape.cubeVariables))).symm

private theorem probe_eq (input : PiCCSInputCheck.Input)
    (messages : PiDECInputCheck.Messages) :
    selectedKey.piCcsProbe (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input)
      (proof input messages) = PiCCSInputCheck.probe input := by
  have raw :
      (selectedKey.piCcsCertificate (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input)
        (proof input messages)).toFinite =
      SumCheck.Finite.FixedPhase.RawCertificate.encode {
        rounds := List.ofFn (PiCCSProofInputs.roundPolynomial (PiCCSInputCheck.proofValues input)) } := by
    simp only [Key.piCcsCertificate, PiCCS.TranscriptReplay.Certificate.toFinite,
      PiCCS.TranscriptReplay.Certificate.toTranscript, FiatShamir.Certificate.toFinite,
      SumCheck.Finite.FixedPhase.RawCertificate.encode, List.map_ofFn, Function.comp_def, proof]
    rfl
  simp only [Key.piCcsProbe, raw, execution_point]
  rfl

private theorem parent_eq (input : PiCCSInputCheck.Input)
    (batch : PiRLCParent.Batch) (parent : PiRLCParent.Values)
    (messages : PiDECInputCheck.Messages)
    (returned : PiRLCParent.computedParent input batch = some parent) :
    selectedKey.parentForChallenges (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input)
        (proof input messages) batch.challenges = PiDECInputCheck.parent parent := by
  have combined := PiRLCParent.computedParent_eq_combined input batch parent returned
  rw [PiRLCParent.inputBatch_eq_probe] at combined
  unfold Key.parentForChallenges Key.piCcsOutputs
  rw [execution_point, probe_eq]
  exact combined.symm

private theorem attempt_eq (input : PiCCSInputCheck.Input)
    (batch : PiRLCParent.Batch) (parent : PiRLCParent.Values)
    (messages : PiDECInputCheck.Messages)
    (sampled : PiRLCInputCheck.sampled input = some batch)
    (returned : PiRLCParent.computedParent input batch = some parent) :
    selectedKey.piDecAttempt (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input)
      (proof input messages) = some (PiDECInputCheck.attempt parent messages) := by
  have sampledKey :
      selectedKey.piRlcChallenges (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input)
        (proof input messages) = some batch.challenges := by
    rw [Key.piRlcChallenges, execution_outgoing]
    exact (PiRLCInputCheck.sampled_response input batch sampled).2
  have sampledAttempt := attempt_of_sample selectedKey
    (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input)
    (proof input messages) batch.challenges sampledKey
  have sameParent := parent_eq input batch parent messages returned
  have sameAttempt :
      selectedKey.piDecAttemptForParent (proof input messages) (PiDECInputCheck.parent parent) =
        PiDECInputCheck.attempt parent messages := by
    rfl
  exact sampledAttempt.trans (congrArg some
    ((congrArg (selectedKey.piDecAttemptForParent (proof input messages)) sameParent).trans sameAttempt))

private theorem running_ext
    (left right : Running (logicalWidth := PiDECInputCheck.logicalWidth)
      (publicFits := PiDECInputCheck.publicFits))
    (point : left.point = right.point) (commitments : left.commitments = right.commitments)
    (publicInputs : left.publicInputs = right.publicInputs)
    (evaluations : left.evaluations = right.evaluations) : left = right := by
  cases left
  cases right
  cases point
  cases commitments
  cases publicInputs
  cases evaluations
  rfl

private theorem output_eq (input : PiCCSInputCheck.Input)
    (parent : PiRLCParent.Values) (messages : PiDECInputCheck.Messages)
    (checked : PiDECInputCheck.accepted parent messages = true) :
    selectedKey.outputForAttempt (proof input messages) (PiDECInputCheck.attempt parent messages)
        (selectedKey.piDecPublicInputSplit.split (PiDECInputCheck.parent parent).publicInput) =
      PiCCSInputCheck.runningFromInput messages := by
  have matchResult := PiDECInputCheck.accepted_implies_outputMatches parent messages checked
  simp only [PiDECInputCheck.outputMatches, Bool.and_eq_true] at matchResult
  have pointWords : messages.point.toList = parent.point.coordinates := of_decide_eq_true matchResult.1
  have publicWords := List.all_eq_true.mp matchResult.2
  apply running_ext
  · apply point_ext
    exact pointWords.symm
  · rfl
  · funext child coordinate
    have checkedChild := publicWords child (by simp)
    have words : (messages.publicInputs.get child).toList =
        List.ofFn (fun column : Fin 270 =>
          (PiDECInputCheck.children parent messages child).publicInput column) :=
      of_decide_eq_true checkedChild
    have vector : messages.publicInputs.get child =
        Vector.ofFn (fun column : Fin 270 =>
          (PiDECInputCheck.children parent messages child).publicInput column) :=
      Vector.toList_inj.mp (by simpa only [Vector.toList_ofFn] using words)
    change _ = (messages.publicInputs.get child).get coordinate
    rw [vector]
    change _ = (Vector.ofFn _)[coordinate.val]
    rw [Vector.getElem_ofFn]
    rfl
  · rfl

/-- The actual C/R/D checker results imply the exact selected NIFS return.
Sampling and D acceptance are checked computation outcomes. No wanted running
claim, source-value equality, witness validity, or expected artifact is supplied. -/
theorem checked_verifies (input : PiCCSInputCheck.Input)
    (batch : PiRLCParent.Batch) (parent : PiRLCParent.Values)
    (messages : PiDECInputCheck.Messages)
    (sampled : PiRLCInputCheck.sampled input = some batch)
    (returned : PiRLCParent.computedParent input batch = some parent)
    (checked : PiDECInputCheck.accepted parent messages = true) :
    Nifs.PaperNonInteractive.verify selectedKey
      (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input) (proof input messages) =
      some (PiCCSInputCheck.runningFromInput messages) := by
  have attempt := attempt_eq input batch parent messages sampled returned
  have accepted := PiDECInputCheck.accepted_implies_paper parent messages checked
  apply (Nifs.PaperNonInteractive.verify_eq_some_iff selectedKey
    (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input)
    (proof input messages) (PiCCSInputCheck.runningFromInput messages)).mpr
  refine ⟨?_, ?_, ?_⟩
  · apply (Nifs.PaperNonInteractive.piCcsCheck_eq_true_iff_fixedWidthAccepted
      selectedKey _ _ _).mpr
    rw [probe_eq]
    exact PiRLCInputCheck.sampled_fixedWidthAccepted PiDECInputCheck.relation
      productionAjtaiKey input batch sampled
  · exact (Nifs.PaperNonInteractive.piDecCheck_eq_true_iff selectedKey _ _ _).mpr
      ⟨PiDECInputCheck.attempt parent messages, attempt, accepted⟩
  · exact (Key.output_eq_some_of_parentBounded selectedKey
      (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input)
      (proof input messages) (PiDECInputCheck.attempt parent messages) attempt accepted.parentBounded).trans
        (congrArg some (output_eq input parent messages checked))

/-- Transport to the unchanged relation used by HyperNovaStepData. -/
theorem checked_verifies_selected (input : PiCCSInputCheck.Input)
    (batch : PiRLCParent.Batch) (parent : PiRLCParent.Values)
    (messages : PiDECInputCheck.Messages)
    (sampled : PiRLCInputCheck.sampled input = some batch)
    (returned : PiRLCParent.computedParent input batch = some parent)
    (checked : PiDECInputCheck.accepted parent messages = true) :
    Nifs.PaperNonInteractive.verify
      (ProductionKey.key
        (PerApplicationFixedPoint.relation Poseidon2HashChainV1Package.application
          Poseidon2HashChainV1Package.fits) productionAjtaiKey)
      (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input) (proof input messages) =
      some (PiCCSInputCheck.runningFromInput messages) := by
  have selected := checked_verifies input batch parent messages sampled returned checked
  change Nifs.PaperNonInteractive.verify (ProductionKey.key PiDECInputCheck.relation productionAjtaiKey)
    _ _ _ = _ at selected
  rw [PiDECInputCheck.relation_eq_selected] at selected
  exact selected

end NightstreamFPrime.Export.Stage1.CheckedReplayNifs
