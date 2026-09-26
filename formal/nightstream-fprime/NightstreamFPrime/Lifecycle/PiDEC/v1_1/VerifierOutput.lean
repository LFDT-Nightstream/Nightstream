import NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics

/-! Exact PiDEC message, child and verifier-output facts independent of physical layout. -/

namespace NightstreamFPrime.Lifecycle.PiDEC.v1_1.VerifierOutput

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

theorem evaluation_ext (left right : PaperAlgebra.Evaluation)
    (pad : left.pad = right.pad) (matrix : left.matrix = right.matrix) : left = right := by
  cases left
  cases right
  simp_all

theorem message_ext (left right : PiDEC.PaperVerifier.ChildMessage
    PaperAlgebra.Evaluation PaperAlgebra.Commitment)
    (commitment : left.commitment = right.commitment)
    (evaluations : left.evaluations = right.evaluations) : left = right := by
  cases left
  cases right
  simp_all

theorem attempt_ext
    (left right : PiDEC.v1_1.InputBinding.Attempt logicalWidth publicFits)
    (parent : left.parent = right.parent) (messages : left.messages = right.messages) : left = right := by
  cases left
  cases right
  simp_all

theorem instance_ext
    (left right : PiDEC.v1_1.OutputBinding.Output logicalWidth publicFits)
    (system : left.constraintSystem = right.constraintSystem)
    (commitment : left.commitment = right.commitment)
    (publicInput : left.publicInput = right.publicInput)
    (point : left.point = right.point)
    (evaluations : left.evaluations = right.evaluations)
    (stage : left.stage = right.stage) : left = right := by
  cases left
  cases right
  simp_all

theorem children_outputAccepted
    (attempt : PiDEC.v1_1.InputBinding.Attempt logicalWidth publicFits)
    (checks : PiDEC.PaperVerifier.Accepted (PaperAlgebra.piDecAlgebra ajtai)
      (PaperAlgebra.publicInputSplit ajtai) (PaperAlgebra.evaluationArity ajtai) attempt) :
    PiDEC.PaperVerifier.OutputAccepted (PaperAlgebra.piDecAlgebra ajtai)
      (PaperAlgebra.publicInputSplit ajtai) (PaperAlgebra.evaluationArity ajtai)
      attempt.parent (PiDEC.PaperVerifier.children (PaperAlgebra.publicInputSplit ajtai) attempt) := by
  have recovered : PiDEC.PaperVerifier.attemptForOutput attempt.parent
      (PiDEC.PaperVerifier.children (PaperAlgebra.publicInputSplit ajtai) attempt) = attempt := by
    cases attempt
    rfl
  exact ⟨by rw [recovered], by rw [recovered]; exact checks⟩

theorem computed_output_eq
    (key : ProductionKey.KeyType relation)
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation))
    (output : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (challenges : Fin key.arity.total → RingF)
    (sampled : key.piRlcChallenges running fresh proof = some challenges)
    (checks : PiDEC.PaperVerifier.Accepted key.piDecAlgebra key.piDecPublicInputSplit key.piDecEvaluationArity
      (key.piDecAttemptForParent proof (key.parentForChallenges running fresh proof challenges)))
    (accepted : Nifs.PaperNonInteractive.verify key running fresh proof = some output) :
    key.outputForAttempt proof
      (key.piDecAttemptForParent proof (key.parentForChallenges running fresh proof challenges))
      (key.piDecPublicInputSplit.split (key.parentForChallenges running fresh proof challenges).publicInput) =
      output := by
  have acceptedOutput := (Nifs.PaperNonInteractive.verify_eq_some_iff key running fresh proof output).mp accepted |>.2.2
  have computed : key.output running fresh proof = some
      (key.outputForAttempt proof
        (key.piDecAttemptForParent proof (key.parentForChallenges running fresh proof challenges))
        (key.piDecPublicInputSplit.split (key.parentForChallenges running fresh proof challenges).publicInput)) := by
    simp only [Nifs.PaperNonInteractive.Key.output, Nifs.PaperNonInteractive.Key.piDecAttempt,
      Nifs.PaperNonInteractive.Key.parent, sampled, Option.map_some, Option.bind_some]
    rw [PiDEC.PaperVerifier.PublicInputSplit.checked_eq_some _ _ checks.parentBounded]
    rfl
  exact Option.some.inj (computed.symm.trans acceptedOutput)

end NightstreamFPrime.Lifecycle.PiDEC.v1_1.VerifierOutput
