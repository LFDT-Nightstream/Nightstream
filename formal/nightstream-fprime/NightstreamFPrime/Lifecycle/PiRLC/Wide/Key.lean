import NightstreamFPrime.Lifecycle.PiRLC.Wide.Semantics

/-! Candidate NIFS key using the proved one-window PiRLC transcript.
This definition does not select a production package or assert a concrete
Fiat–Shamir security bound. All other verifier fields are unchanged. -/

namespace NightstreamFPrime.Lifecycle.PiRLC.Wide.Key

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open Spec.Folding Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PaperAlgebra

def response (state : Transcript.State) : Fin Nifs.PaperProfile.arity.total → RingF :=
  fun source => Nifs.NonInteractive.PiRlcWideSampler.Transcript.challengeAt state source.val

def piRlcResponse (state : Transcript.State) : Option (Fin Nifs.PaperProfile.arity.total → RingF) :=
  some (response state)

theorem response_valid (state : Transcript.State) (source : Fin Nifs.PaperProfile.arity.total) :
    Phi81Relation.PiRLCAlgebra.Challenge.challengeValid (response state source) :=
  Nifs.NonInteractive.PiRlcWideSampler.Transcript.challengeAt_member state source.val

theorem piRlcResponse_valid (state : Transcript.State)
    (values : Fin Nifs.PaperProfile.arity.total → RingF)
    (returned : piRlcResponse state = some values) (source : Fin Nifs.PaperProfile.arity.total) :
    Phi81Relation.PiRLCAlgebra.Challenge.challengeValid (values source) := by
  have same := Option.some.inj returned
  rw [← same]
  exact response_valid state source

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

noncomputable def key (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) : ProductionKey.KeyType relation :=
  { ProductionKey.key relation ajtai with
    piRlcResponse := piRlcResponse
    piRlcResponseValid := piRlcResponse_valid }

private theorem ccs_with_response (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (before : ProductionKey.KeyType relation)
    (response : Transcript.State → Option (Fin before.arity.total → RingF))
    (valid : ∀ state values, response state = some values → ∀ index, before.piRlcAlgebra.challengeValid (values index))
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation)) :
    let after := { before with piRlcResponse := response, piRlcResponseValid := valid }
    after.piCcsExecution running fresh proof = before.piCcsExecution running fresh proof ∧
      after.piCcsOutputs running fresh proof = before.piCcsOutputs running fresh proof ∧
      Nifs.PaperNonInteractive.piCcsCheck after running fresh proof =
        Nifs.PaperNonInteractive.piCcsCheck before running fresh proof := by
  intro after
  constructor
  · unfold Nifs.PaperNonInteractive.Key.piCcsExecution
    rfl
  constructor
  · unfold Nifs.PaperNonInteractive.Key.piCcsOutputs
    rfl
  · unfold Nifs.PaperNonInteractive.piCcsCheck
    rfl

theorem piCcsOutputs_unchanged (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation)) :
    (key relation ajtai).piCcsOutputs running fresh proof =
      (ProductionKey.key relation ajtai).piCcsOutputs running fresh proof := by
  exact (ccs_with_response relation (ProductionKey.key relation ajtai) piRlcResponse piRlcResponse_valid running fresh proof).2.1

theorem piCcsExecution_unchanged (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation)) :
    (key relation ajtai).piCcsExecution running fresh proof =
      (ProductionKey.key relation ajtai).piCcsExecution running fresh proof :=
  (ccs_with_response relation (ProductionKey.key relation ajtai) piRlcResponse piRlcResponse_valid running fresh proof).1

theorem piCcsCheck_unchanged (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation)) :
    Nifs.PaperNonInteractive.piCcsCheck (key relation ajtai) running fresh proof =
      Nifs.PaperNonInteractive.piCcsCheck (ProductionKey.key relation ajtai) running fresh proof :=
  (ccs_with_response relation (ProductionKey.key relation ajtai) piRlcResponse piRlcResponse_valid running fresh proof).2.2

theorem key_response (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) (state : Transcript.State) :
    (key relation ajtai).piRlcResponse state = some (response state) := rfl

theorem phase_response (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits) (offset : Nat) (env : Env)
    (phase : Semantics.PhaseHolds relation ajtai interface offset env) :
    (key relation ajtai).piRlcResponse (Scalar.evalState env (interface.initialState offset)) =
      some (Semantics.evalChallenges interface offset env) := by
  rw [key_response]
  exact congrArg some phase.response

theorem profile_unchanged (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (key relation ajtai).params = productionGlobalParams := rfl

theorem challengeSet_unchanged (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (key relation ajtai).challengeSetSize =
      Nat.card {value : RingF // Phi81StrongSet.ProductionMember value} :=
  ProductionKey.key_challengeSetSize_cardinality relation ajtai

end NightstreamFPrime.Lifecycle.PiRLC.Wide.Key
