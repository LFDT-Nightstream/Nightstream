import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe.HonestProver
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane.HonestProver
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane

/-!
Sequential honest-prover composition for canonical FE plus block×lane NC.

Assurance tier: model-level.

Owns: FE certificate construction, the exact FE-to-NC state handoff, canonical
block×lane NC certificate construction, source-derived `yRing` and packed
`yZcol` output construction at transcript-derived points, and honest protocol
acceptance.

Does not own: concrete Poseidon2 encoding, Fiat--Shamir probability, CE
materialization, Π_RLC, Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: every output coordinate is computed from independent
source data at verifier-derived points. The certificate contains neither
points nor challenges. The full output is absorbed only by the protocol's
post-NC handoff.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.prover.block_lane.fe` | construct FE messages before their challenges | derived | `Fe.HonestProver.exists_honest_certificate` |
| `nifs.pi_ccs.prover.block_lane.handoff` | initialize NC from FE's exact successor | direct dataflow | `complete_of_paperObligations` |
| `nifs.pi_ccs.prover.block_lane.nc` | construct exact block-then-lane messages before challenges | derived | `Nc.BlockLane.HonestProver.complete_of_truth` |
| `nifs.pi_ccs.prover.block_lane.output` | compute `yRing` and packed `yZcol` from sources at derived points | computed | `canonicalOutput` |
| `nifs.pi_ccs.prover.block_lane.output_authority` | the canonical output satisfies the exact protocol predicate | derived | `canonicalOutput_bound` |
| `nifs.pi_ccs.prover.block_lane.completeness` | paper obligations yield one accepted sequential certificate | derived | `complete_of_paperObligations` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.HonestProver

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

universe uVerifierKey uInput uState

/-- The complete honest-prover result at one explicit polynomial input.
Keeping the input as an index makes the only dependent transport visible and
separate from certificate construction. -/
private def ResultAt
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    (covers : domains.nc.Covers shape)
    (schedule : Schedule VerifierKey Input shape domains State)
    (priorState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (statement : Statement VerifierKey Input)
    (data : Data shape)
    (input : PublicInput shape)
    (certificate : Certificate input domains) : Prop :=
  let project := fun (_ : Input) => input
  Accepted project schedule priorState profile statement certificate ∧
    OutputBound covers data
      (derive project schedule priorState profile statement certificate)
      certificate.output

/-- Reindex an already-proved result across the explicit public-input
equality. The certificate payload is unchanged. -/
private theorem resultAt_transport
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    (covers : domains.nc.Covers shape)
    (schedule : Schedule VerifierKey Input shape domains State)
    (priorState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (statement : Statement VerifierKey Input)
    (data : Data shape)
    {input : PublicInput shape}
    (bound : input = PublicInput.ofSources data)
    (certificate : Certificate (PublicInput.ofSources data) domains)
    (holds : ResultAt covers schedule priorState profile statement data
      (PublicInput.ofSources data) certificate) :
    ∃ transported : Certificate input domains,
      ResultAt covers schedule priorState profile statement data input
        transported := by
  subst input
  exact ⟨certificate, holds⟩

/-- `ResultAt` at the statement's projection is definitionally the public
protocol result; no function extensionality or extra authority is used. -/
private theorem resultAt_projected_iff
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    (covers : domains.nc.Covers shape)
    (projectInput : Input -> PublicInput shape)
    (schedule : Schedule VerifierKey Input shape domains State)
    (priorState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (statement : Statement VerifierKey Input)
    (data : Data shape)
    (certificate : Certificate (projectInput statement.input) domains) :
    ResultAt covers schedule priorState profile statement data
        (projectInput statement.input) certificate ↔
      Accepted projectInput schedule priorState profile statement certificate ∧
        OutputBound covers data
          (derive projectInput schedule priorState profile statement certificate)
          certificate.output := by
  rfl

/-- The unique source-derived raw output at one FE row point and one canonical
NC block point. -/
def canonicalOutput
    {shape : SemanticShape}
    {domains : Domains}
    (covers : domains.nc.Covers shape)
    (data : Data shape)
    (row : CubePoint K shape.rowVariables)
    (block : CubePoint K domains.nc.blockVariables) :
    OutputMessage shape where
  yRing := Polynomial.Fe.sourceYRingAt data row
  yZcol := fun source =>
    PackedBlockAction.packedYZcol covers (data.assignment source) block

/-- Canonical output construction discharges both independent output-binding
children by computation. -/
theorem canonicalOutput_bound
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    (covers : domains.nc.Covers shape)
    (data : Data shape)
    (execution : Execution shape domains State) :
    OutputBound covers data execution
      (canonicalOutput covers data execution.fePoint.row
        execution.ncPoint.block) := by
  constructor
  · rfl
  · intro source lane
    rfl

/-- The packed terminal computed from canonical output is the independent NC
polynomial at the same transcript-derived point. -/
private theorem canonicalNcTerminal
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    (covers : domains.nc.Covers shape)
    (data : Data shape)
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Polynomial.Nc.BlockLane.Mixing.Coins domains.nc)
    (certificate : Transcript.Nc.BlockLane.Certificate domains.nc)
    (row : CubePoint K shape.rowVariables) :
    let point := (Nc.BlockLane.derive machine initialState certificate).point
    let output := canonicalOutput covers data row point.block
    Polynomial.Nc.BlockLane.Terminal.terminalFromMessage output coins point =
      Polynomial.Nc.BlockLane.InitialSum.sumcheckPolynomial
        covers data coins point.coordinates := by
  let point := (Nc.BlockLane.derive machine initialState certificate).point
  let output := canonicalOutput covers data row point.block
  have bound :
      Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock
        covers data point.block output := by
    intro source lane
    rfl
  calc
    Polynomial.Nc.BlockLane.Terminal.terminalFromMessage output coins point =
        Polynomial.Nc.BlockLane.Mixing.qAtPoint covers data coins point :=
      Polynomial.Nc.BlockLane.Terminal.terminal_eq_qAtPoint_of_bound
        covers data coins point output bound
    _ = Polynomial.Nc.BlockLane.InitialSum.sumcheckPolynomial
          covers data coins point.coordinates :=
      (Polynomial.Nc.BlockLane.InitialSum.sumcheckPolynomial_coordinates_eq_qAtPoint
        covers data coins point).symm

/-- Every independent paper-valid source family has one accepted canonical
FE→block×lane-NC certificate under the statement-derived challenge record.
The explicit projection equality is the sole bridge from the enriched public
statement to source semantics. -/
theorem complete_of_paperObligations
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    (covers : domains.nc.Covers shape)
    (projectInput : Input -> PublicInput shape)
    (schedule : Schedule VerifierKey Input shape domains State)
    (priorState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (statement : Statement VerifierKey Input)
    (data : Data shape)
    (inputBound : projectInput statement.input = PublicInput.ofSources data)
    (obligations : Semantics.Paper.Holds data) :
    ∃ certificate : Certificate (projectInput statement.input) domains,
      Accepted projectInput schedule priorState profile statement certificate ∧
        OutputBound covers data
          (derive projectInput schedule priorState profile statement certificate)
          certificate.output := by
  let pre := derivePreSumcheck schedule priorState statement
  let initialClaim := Polynomial.Fe.initial profile
    (PublicInput.ofSources data) pre.challenges.feCoins
  let feTranscript := feMachine schedule initialClaim
  have feTruth : Semantics.Fe.Truth data :=
    ⟨obligations.1, obligations.2.2⟩
  have ncTruth : Semantics.Nc.Truth data := obligations.2.1
  rcases Fe.HonestProver.exists_honest_certificate
      profile data feTranscript pre.state pre.challenges.feCoins with
    ⟨feCertificate, feHonest⟩
  let feExecution :=
    Transcript.Fe.derive feTranscript pre.state feCertificate
  rcases Nc.BlockLane.HonestProver.complete_of_truth
      covers data (ncMachine schedule) feExecution.finalState
      pre.challenges.ncCoins ncTruth with
    ⟨ncCertificate, ncSemanticAccepted⟩
  let ncExecution :=
    Nc.BlockLane.derive (ncMachine schedule) feExecution.finalState
      ncCertificate
  let output := canonicalOutput covers data feExecution.challengePoint.row
    ncExecution.point.block
  let sourceCertificate :
      Certificate (PublicInput.ofSources data) domains := {
    fe := feCertificate
    nc := ncCertificate
    output := output
  }
  have feMessageBound :
      output.yRing =
        Polynomial.Fe.sourceYRingAt data feExecution.challengePoint.row := by
    rfl
  have feAccepted :
      Fe.Accepted feTranscript pre.state profile (PublicInput.ofSources data)
        pre.challenges.feCoins output feCertificate :=
    Fe.accepted_of_truth_and_honestAt feTranscript pre.state profile data
      pre.challenges.feCoins output feCertificate feTruth feMessageBound
      feHonest
  have ncTerminal :
      Polynomial.Nc.BlockLane.Terminal.terminalFromMessage output
          pre.challenges.ncCoins ncExecution.point =
        Polynomial.Nc.BlockLane.InitialSum.sumcheckPolynomial covers data
          pre.challenges.ncCoins ncExecution.point.coordinates := by
    simpa [output, ncExecution, feExecution] using
      canonicalNcTerminal covers data (ncMachine schedule)
        feExecution.finalState pre.challenges.ncCoins ncCertificate
        feExecution.challengePoint.row
  have ncAccepted :
      Nc.BlockLane.Accepted (ncMachine schedule) feExecution.finalState
        pre.challenges.ncCoins output ncCertificate := by
    unfold Nc.BlockLane.Accepted
    change SumCheck.Nc.Accepted
      Polynomial.Nc.BlockLane.InitialSum.claimedInitial
      ncExecution.point.coordinates
      (Polynomial.Nc.BlockLane.Terminal.terminalFromMessage output
        pre.challenges.ncCoins ncExecution.point)
      ncCertificate.toSumCheck
    rw [ncTerminal]
    exact ncSemanticAccepted
  have sourceResult :
      ResultAt covers schedule priorState profile statement data
        (PublicInput.ofSources data) sourceCertificate := by
    constructor
    · change
        Fe.Accepted feTranscript pre.state profile (PublicInput.ofSources data)
            pre.challenges.feCoins sourceCertificate.output
            sourceCertificate.fe ∧
          Nc.BlockLane.Accepted (ncMachine schedule)
            (Transcript.Fe.derive feTranscript pre.state
              sourceCertificate.fe).finalState
            pre.challenges.ncCoins sourceCertificate.output sourceCertificate.nc
      exact ⟨feAccepted, ncAccepted⟩
    · change
        sourceCertificate.output.yRing =
            Polynomial.Fe.sourceYRingAt data feExecution.challengePoint.row ∧
          Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock covers data
            ncExecution.point.block sourceCertificate.output
      constructor
      · exact feMessageBound
      · intro source lane
        rfl
  rcases resultAt_transport covers schedule priorState profile statement data
      inputBound sourceCertificate sourceResult with
    ⟨certificate, result⟩
  exact ⟨certificate,
    (resultAt_projected_iff covers projectInput schedule priorState profile
      statement data certificate).mp result⟩

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.HonestProver
