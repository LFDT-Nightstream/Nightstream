import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.Acceptance
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.Interface
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.BlockLane

/-!
Claims-level terminal refinement for the production block×lane combined NC
SumCheck.

Assurance tier: model-level until the Rust verifier and generated terminal
rows refine `AcceptedFromMessage`.

Owns: the verifier-computable delayed terminal built from the current
`Pi_CCS` output message; its equality with the independent raw-assignment
terminal under the exact weak-relation output binding; and the resulting
claims-acceptance-to-raw-acceptance partition.

Does not own: derivation of output binding from extraction or terminal
openings, transcript replay, commitment security, Rust, generated rows,
costs, or row removal.

Emits constraints: none; claims-level verifier/refinement semantics only.

Authority boundary: `Claims.yZcol` is used only to compute the public
terminal. It becomes semantic authority only through
`PackedYZcolBoundAtBlock`, which states equality to every corresponding raw
source assignment. Failure remains explicit; no digest or terminal scalar is
accepted as a substitute.

| Stage path | Mathematical obligation | Authority class |
|---|---|---|
| `pi_ccs.nc.delayed.message.running` | radix-weight the running output evaluations in canonical source order | computed |
| `pi_ccs.nc.delayed.message.terminal` | reuse the ordinary public terminal plus the delayed public running value | computed |
| `pi_ccs.nc.delayed.message.refine` | a genuine output opening makes that terminal equal the raw combined polynomial | derived/security boundary |
| `pi_ccs.nc.delayed.message.accept` | public claimed-chain acceptance yields raw acceptance or an exact output-binding failure | derived/security partition |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.MessageTerminal

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.SumCheck.Finite

private abbrev ops := ConcreteCarrier.extensionOps

universe uState

/-- Verifier-computable interpolation of the weighted running suffix. The
message values are transport only; the refinement theorem below supplies
their raw-assignment authority. -/
def runningValueFromMessage
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (message : Claims shape)
    (weights : RunningWeights shape)
    (lanePrime : CubePoint K domain.laneVariables) : K :=
  FiniteSumAlgebra.sumMap ops
    (canonicalFinIndices shape.runningCount) fun running =>
      K.mul (K.embed (weights running))
        (Terminal.valueAt message (Data.runningIndex running) lanePrime)

/-- Full output binding identifies the verifier-computable running suffix
with the independent raw running-assignment polynomial at the same typed
block×lane point. -/
theorem runningValueFromMessage_eq_authoritative_of_bound
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (weights : RunningWeights shape)
    (point : Point domain)
    (message : Claims shape)
    (bound : Terminal.PackedYZcolBoundAtBlock covers data point.block message) :
    runningValueFromMessage message weights point.lane =
      authoritativeRunningValueAt covers data weights point := by
  unfold runningValueFromMessage authoritativeRunningValueAt
  apply FiniteSumAlgebra.sumMap_congr
  intro running _
  apply congrArg (K.mul (K.embed (weights running)))
  exact Terminal.valueAt_eq_sourceValueAt_of_bound
    covers data point message bound (Data.runningIndex running)

/-- Delayed part of the verifier terminal, computed only from the public
output message and verifier-owned challenges. -/
def delayedFromMessage
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (message : Claims shape)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (point : Point domain) : K :=
  K.mul batchWeight
    (K.mul
      (SumCheckTruthPath.pointEquality ops point.block oldBlock)
      (K.mul (betaPowerSelector producerBeta point.lane)
        (runningValueFromMessage message weights point.lane)))

/-- Exact terminal a claims-only production verifier can compute. -/
def verifierTerminal
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (message : Claims shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (point : Point domain) : K :=
  K.add (Terminal.terminalFromMessage message coins point)
    (delayedFromMessage message weights producerBeta batchWeight oldBlock
      point)

/-- The weak-relation output opening refines the public terminal to the
independent combined polynomial at the verifier-derived point. -/
theorem verifierTerminal_eq_combinedAtPoint_of_bound
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (point : Point domain)
    (message : Claims shape)
    (bound : Terminal.PackedYZcolBoundAtBlock covers data point.block message) :
    verifierTerminal message coins weights producerBeta batchWeight oldBlock
        point =
      combinedAtPoint covers data coins weights producerBeta batchWeight
        oldBlock point := by
  unfold verifierTerminal delayedFromMessage combinedAtPoint delayedAtPoint
  rw [Terminal.terminal_eq_qAtPoint_of_bound
    covers data coins point message bound]
  rw [runningValueFromMessage_eq_authoritative_of_bound
    covers data weights point message bound]

/-- Public terminal equality with the exact serialized independent
polynomial. The typed point supplies the complete 21-block-plus-6-lane
arity. -/
theorem verifierTerminal_eq_sumcheckPolynomial_of_bound
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (point : Point domain)
    (message : Claims shape)
    (bound : Terminal.PackedYZcolBoundAtBlock covers data point.block message) :
    verifierTerminal message coins weights producerBeta batchWeight oldBlock
        point =
      sumcheckPolynomial covers data coins weights producerBeta batchWeight
        oldBlock point.coordinates := by
  rw [verifierTerminal_eq_combinedAtPoint_of_bound
    covers data coins weights producerBeta batchWeight oldBlock point
    message bound]
  exact
    (Acceptance.sumcheckPolynomial_coordinates_eq_combinedAtPoint
      covers data coins weights producerBeta batchWeight oldBlock point).symm

/-- Exact claims-level combined-NC acceptance. This is the public verifier
surface: the certificate contains round messages, while the verifier computes
the terminal from the output message. -/
def AcceptedFromMessage
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (message : Claims shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight initial : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (point : Point domain)
    (certificate : SumCheck.Nc.Certificate) : Prop :=
  SumCheck.Nc.Accepted initial point.coordinates
    (verifierTerminal message coins weights producerBeta batchWeight oldBlock
      point)
    certificate

/-- A claims-only accepted chain either refines to the independent raw
combined-NC acceptance or exposes exactly the missing weak-relation output
opening. This is the extraction seam used by sequence-level composition. -/
theorem acceptedFromMessage_implies_rawAccepted_or_outputBindingFailure
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight initial : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (point : Point domain)
    (message : Claims shape)
    (certificate : SumCheck.Nc.Certificate)
    (accepted : AcceptedFromMessage message coins weights producerBeta
      batchWeight initial oldBlock point certificate) :
    FixedPhase.Accepted ops.toOps
        (sumcheckPolynomial covers data coins weights producerBeta batchWeight
          oldBlock)
        initial point.coordinates certificate ∨
      ¬ Terminal.PackedYZcolBoundAtBlock covers data point.block message := by
  by_cases bound :
      Terminal.PackedYZcolBoundAtBlock covers data point.block message
  · apply Or.inl
    unfold FixedPhase.Accepted
    unfold AcceptedFromMessage SumCheck.Nc.Accepted at accepted
    rw [← verifierTerminal_eq_sumcheckPolynomial_of_bound
      covers data coins weights producerBeta batchWeight oldBlock point
      message bound]
    exact accepted
  · exact Or.inr bound

/-- Transcript-bound form of `AcceptedFromMessage`. The FE-to-NC handoff
state is an explicit verifier input here; the production specialization
supplies the exact FE successor state. -/
def TranscriptAcceptedFromMessage
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {State : Type uState}
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (message : Claims shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight initial : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (certificate : Transcript.Nc.BlockLane.Certificate domain) : Prop :=
  let point :=
    (Transcript.Nc.BlockLane.derive machine initialState certificate
      ).challengePoint
  Transcript.Nc.BlockLane.Accepted machine initialState initial
    (verifierTerminal message coins weights producerBeta batchWeight oldBlock
      point)
    certificate

/-- Exact transcript replay plus the public combined terminal yields the raw
combined-polynomial acceptance or the same explicit output-opening failure.
No caller supplies a challenge point or phase transition. -/
theorem transcriptAcceptedFromMessage_implies_rawAccepted_or_outputBindingFailure
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {State : Type uState}
    (covers : domain.Covers shape)
    (data : Data shape)
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (message : Claims shape)
    (coins : Mixing.Coins domain)
    (weights : RunningWeights shape)
    (producerBeta batchWeight initial : K)
    (oldBlock : CubePoint K domain.blockVariables)
    (certificate : Transcript.Nc.BlockLane.Certificate domain)
    (accepted : TranscriptAcceptedFromMessage machine initialState message
      coins weights producerBeta batchWeight initial oldBlock certificate) :
    let point :=
      (Transcript.Nc.BlockLane.derive machine initialState certificate
        ).challengePoint
    FixedPhase.Accepted ops.toOps
        (sumcheckPolynomial covers data coins weights producerBeta batchWeight
          oldBlock)
        initial point.coordinates certificate.toSumCheck ∨
      ¬ Terminal.PackedYZcolBoundAtBlock covers data point.block message := by
  dsimp only
  apply acceptedFromMessage_implies_rawAccepted_or_outputBindingFailure
    covers data coins weights producerBeta batchWeight initial oldBlock
      (Transcript.Nc.BlockLane.derive machine initialState certificate
        ).challengePoint
      message certificate.toSumCheck
  simpa [TranscriptAcceptedFromMessage,
    Transcript.Nc.BlockLane.Accepted] using accepted

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.MessageTerminal
