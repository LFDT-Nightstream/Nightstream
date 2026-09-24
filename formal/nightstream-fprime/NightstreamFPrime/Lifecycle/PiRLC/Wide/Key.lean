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
