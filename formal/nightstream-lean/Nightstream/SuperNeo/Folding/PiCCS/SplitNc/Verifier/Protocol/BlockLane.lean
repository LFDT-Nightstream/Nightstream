import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

/-!
Canonical FE-to-block×lane-NC Split-NC `Pi_CCS` verifier composition.

Assurance tier: model-level.

Owns: one message-only two-phase certificate; explicit projection from the
complete transcript-bound statement to the polynomial input consumed by FE
and NC; derivation of the shared pre-SumCheck challenge record;
verifier-computed FE entry; exact FE-to-NC state handoff; post-NC output
absorption; executable/logical parity; and deterministic composition of the
independent FE and NC soundness theorems.

Does not own: concrete Poseidon2, transcript collision bounds, an honest
Fiat--Shamir prover, public-input/source refinement, construction of packed
`yZcol` authority, `Pi_RLC`, Rust, R1CS, costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: the verifier derives the pre-SumCheck record from the
complete typed statement. That one record supplies both phase coin views. FE
starts from its verifier-computed initial claim; NC starts from FE's exact
successor state; the output message is absorbed only after NC. `OutputBound`
keeps raw `yRing` and every active packed `yZcol` lane tied to independent
sources at the two transcript-derived points.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.verify.block_lane.certificate` | FE messages, exact nine-round NC messages, and raw output only | checked by type | `Certificate` |
| `nifs.pi_ccs.verify.block_lane.input_projection` | derive the polynomial input from the complete transcript-bound statement | direct dataflow | `projectInput` argument to `derive`, `Accepted`, and `check` |
| `nifs.pi_ccs.verify.block_lane.prepare` | bind statement, derive one challenge record, compute FE entry | computed | `Accepted`, `check`, `derive` |
| `nifs.pi_ccs.verify.block_lane.fe` | FE uses the shared coin projection and computed entry | computed/checked | `Accepted`, `check` |
| `nifs.pi_ccs.verify.block_lane.handoff` | NC begins at FE's exact outgoing state | direct dataflow | `Accepted`, `check`, `derive_ncPoint` |
| `nifs.pi_ccs.verify.block_lane.nc` | canonical block×lane NC uses the same record | computed/checked | `Accepted`, `check` |
| `nifs.pi_ccs.verify.block_lane.output` | absorb complete raw output after NC | computed | `derive_finalState` |
| `nifs.pi_ccs.verify.block_lane.output_binding` | bind `yRing` and all active packed `yZcol` lanes at derived points | explicit proof boundary | `OutputBound` |
| `nifs.pi_ccs.verify.block_lane.soundness` | acceptance yields paper truth, missing binding, or a named phase event | derived | `accepted_implies_paperObligations_or_unbound_or_badEvent` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

universe uVerifierKey uInput uState

/-- Complete prover-visible payload. Challenge records, challenge points,
transcript states, terminals, and semantic witnesses are absent. -/
structure Certificate
    {shape : SemanticShape}
    (input : PublicInput shape)
    (domains : Domains) where
  fe : SumCheck.Fe.Certificate input domains.fe
  nc : Transcript.Nc.BlockLane.Certificate domains.nc
  output : OutputMessage shape

/-- Verifier-derived two-phase points and post-output transcript state. -/
structure Execution
    (shape : SemanticShape)
    (domains : Domains)
    (State : Type uState) where
  fePoint : Polynomial.Fe.Point shape domains.fe
  ncPoint : Polynomial.Nc.BlockLane.Point domains.nc
  finalState : State

/-- Replay canonical FE and NC from one statement-derived preparation, then
absorb the raw output message only after NC completes. -/
def derive
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    (projectInput : Input -> PublicInput shape)
    (schedule : Schedule VerifierKey Input shape domains State)
    (priorState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (statement : Statement VerifierKey Input)
    (certificate : Certificate (projectInput statement.input) domains) :
    Execution shape domains State :=
  let pre := derivePreSumcheck schedule priorState statement
  let polynomialInput := projectInput statement.input
  let initialClaim := Polynomial.Fe.initial profile polynomialInput
    pre.challenges.feCoins
  let feTranscript := feMachine schedule initialClaim
  let feExecution := Transcript.Fe.derive feTranscript pre.state certificate.fe
  let ncExecution := Nc.BlockLane.derive (ncMachine schedule)
    feExecution.finalState certificate.nc
  {
    fePoint := feExecution.challengePoint
    ncPoint := ncExecution.point
    finalState := schedule.absorbOutput ncExecution.finalState certificate.output
  }

/-- The canonical NC point comes from a replay whose incoming state is FE's
exact outgoing state. -/
theorem derive_ncPoint
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    (projectInput : Input -> PublicInput shape)
    (schedule : Schedule VerifierKey Input shape domains State)
    (priorState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (statement : Statement VerifierKey Input)
    (certificate : Certificate (projectInput statement.input) domains) :
    let pre := derivePreSumcheck schedule priorState statement
    let polynomialInput := projectInput statement.input
    let initialClaim := Polynomial.Fe.initial profile polynomialInput
      pre.challenges.feCoins
    let feTranscript := feMachine schedule initialClaim
    let feExecution := Transcript.Fe.derive feTranscript pre.state certificate.fe
    (derive projectInput schedule priorState profile statement certificate).ncPoint =
      (Nc.BlockLane.derive (ncMachine schedule) feExecution.finalState
        certificate.nc).point := by
  rfl

/-- The public output is absorbed exactly once, after canonical NC returns
its successor state. -/
theorem derive_finalState
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    (projectInput : Input -> PublicInput shape)
    (schedule : Schedule VerifierKey Input shape domains State)
    (priorState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (statement : Statement VerifierKey Input)
    (certificate : Certificate (projectInput statement.input) domains) :
    let pre := derivePreSumcheck schedule priorState statement
    let polynomialInput := projectInput statement.input
    let initialClaim := Polynomial.Fe.initial profile polynomialInput
      pre.challenges.feCoins
    let feTranscript := feMachine schedule initialClaim
    let feExecution := Transcript.Fe.derive feTranscript pre.state certificate.fe
    let ncExecution := Nc.BlockLane.derive (ncMachine schedule)
      feExecution.finalState certificate.nc
    (derive projectInput schedule priorState profile statement certificate).finalState =
      schedule.absorbOutput ncExecution.finalState certificate.output := by
  rfl

/-- Logical acceptance with no caller-supplied coins, points, phase boundary,
terminal scalar, or outgoing state. -/
def Accepted
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    (projectInput : Input -> PublicInput shape)
    (schedule : Schedule VerifierKey Input shape domains State)
    (priorState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (statement : Statement VerifierKey Input)
    (certificate : Certificate (projectInput statement.input) domains) : Prop :=
  let pre := derivePreSumcheck schedule priorState statement
  let polynomialInput := projectInput statement.input
  let initialClaim := Polynomial.Fe.initial profile polynomialInput
    pre.challenges.feCoins
  let feTranscript := feMachine schedule initialClaim
  let feExecution := Transcript.Fe.derive feTranscript pre.state certificate.fe
  Fe.Accepted feTranscript pre.state profile polynomialInput
      pre.challenges.feCoins certificate.output certificate.fe ∧
    Nc.BlockLane.Accepted (ncMachine schedule) feExecution.finalState
      pre.challenges.ncCoins certificate.output certificate.nc

/-- Executable checker with the same statement-derived preparation and exact
phase handoff. -/
def check
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    (projectInput : Input -> PublicInput shape)
    (schedule : Schedule VerifierKey Input shape domains State)
    (priorState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (statement : Statement VerifierKey Input)
    (certificate : Certificate (projectInput statement.input) domains) : Bool :=
  let pre := derivePreSumcheck schedule priorState statement
  let polynomialInput := projectInput statement.input
  let initialClaim := Polynomial.Fe.initial profile polynomialInput
    pre.challenges.feCoins
  let feTranscript := feMachine schedule initialClaim
  let feExecution := Transcript.Fe.derive feTranscript pre.state certificate.fe
  Fe.check feTranscript pre.state profile polynomialInput
      pre.challenges.feCoins certificate.output certificate.fe &&
    Nc.BlockLane.check (ncMachine schedule) feExecution.finalState
      pre.challenges.ncCoins certificate.output certificate.nc

/-- Executable and logical acceptance coincide exactly. -/
theorem check_eq_true_iff_accepted
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    (projectInput : Input -> PublicInput shape)
    (schedule : Schedule VerifierKey Input shape domains State)
    (priorState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (statement : Statement VerifierKey Input)
    (certificate : Certificate (projectInput statement.input) domains) :
    check projectInput schedule priorState profile statement certificate = true ↔
      Accepted projectInput schedule priorState profile statement certificate := by
  simp only [check, Accepted, Bool.and_eq_true]
  rw [Fe.check_eq_true_iff_accepted, Nc.BlockLane.check_eq_true_iff_accepted]

/-- Both raw output families are tied to the independent source data at the
two verifier-derived points. -/
def OutputBound
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    (covers : domains.nc.Covers shape)
    (data : Data shape)
    (execution : Execution shape domains State)
    (message : OutputMessage shape) : Prop :=
  message.yRing = Polynomial.Fe.sourceYRingAt data execution.fePoint.row ∧
    Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock covers data
      execution.ncPoint.block message

/-- Algebraic bad events remain owned by the phase that exposes them. -/
inductive BadEvent
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (covers : domains.nc.Covers shape)
    (data : Data shape)
    (challenges : Challenges shape domains)
    (execution : Execution shape domains State)
    (certificate : Certificate (PublicInput.ofSources data) domains)
    (challengeSetSize : Nat) : Prop where
  | fe
      (bad : SumCheck.Fe.BadEvent profile data challenges.feCoins
        execution.fePoint certificate.fe challengeSetSize) :
      BadEvent profile covers data challenges execution certificate
        challengeSetSize
  | nc
      (bad : SumCheck.Nc.BlockLane.BadEvent covers data challenges.ncCoins
        execution.ncPoint certificate.nc.toSumCheck challengeSetSize) :
      BadEvent profile covers data challenges execution certificate
        challengeSetSize

/-- Reindex one physical certificate only after an explicit equality identifies
its polynomial input with the independent source projection. The certificate
payload is unchanged. -/
def certificateAtSources
    {shape : SemanticShape}
    {domains : Domains}
    {input : PublicInput shape}
    (data : Data shape)
    (certificate : Certificate input domains)
    (bound : input = PublicInput.ofSources data) :
    Certificate (PublicInput.ofSources data) domains :=
  Eq.mp
    (congrArg (fun projected => Certificate projected domains) bound)
    certificate

/-- Reindexing the FE certificate across the source-projection equality
preserves FE acceptance. -/
private theorem feAccepted_atSources
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    {input : PublicInput shape}
    (data : Data shape)
    (certificate : Certificate input domains)
    (bound : input = PublicInput.ofSources data)
    (machine : Transcript.Fe.Machine State)
    (initialState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (coins : Polynomial.Fe.Coins shape domains.fe)
    (message : OutputMessage shape)
    (accepted : Fe.Accepted machine initialState profile input coins message
      certificate.fe) :
    Fe.Accepted machine initialState profile (PublicInput.ofSources data) coins
      message (certificateAtSources data certificate bound).fe := by
  subst input
  exact accepted

/-- The FE execution is unchanged when only its dependent input index is
transported to the equal source projection. -/
private theorem feDerive_atSources_eq
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    {input : PublicInput shape}
    (data : Data shape)
    (certificate : Certificate input domains)
    (bound : input = PublicInput.ofSources data)
    (machine : Transcript.Fe.Machine State)
    (initialState : State) :
    Transcript.Fe.derive machine initialState
        (certificateAtSources data certificate bound).fe =
      Transcript.Fe.derive machine initialState certificate.fe := by
  subst input
  rfl

/-- NC certificate data is independent of the polynomial-input index and is
unchanged by the source-projection transport. -/
private theorem certificateAtSources_nc
    {shape : SemanticShape}
    {domains : Domains}
    {input : PublicInput shape}
    (data : Data shape)
    (certificate : Certificate input domains)
    (bound : input = PublicInput.ofSources data) :
    (certificateAtSources data certificate bound).nc = certificate.nc := by
  subst input
  rfl

/-- Deterministic soundness of the canonical two-phase model.

The complete statement is bound before challenge derivation. Only its explicit
`projectInput` image enters FE/NC, and the theorem crosses from that physical
projection to independent source semantics through `inputBound`. -/
theorem accepted_implies_paperObligations_or_unbound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
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
    (certificate : Certificate (projectInput statement.input) domains)
    (challengeSetSize : Nat)
    (accepted : Accepted projectInput schedule priorState profile statement
      certificate) :
    let pre := derivePreSumcheck schedule priorState statement
    let execution :=
      derive projectInput schedule priorState profile statement certificate
    Semantics.Paper.Holds data ∨
      ¬ OutputBound covers data execution certificate.output ∨
      BadEvent profile covers data pre.challenges execution
        (certificateAtSources data certificate inputBound)
        challengeSetSize := by
  let polynomialInput := projectInput statement.input
  let pre := derivePreSumcheck schedule priorState statement
  let initialClaim := Polynomial.Fe.initial profile polynomialInput
    pre.challenges.feCoins
  let feTranscript := feMachine schedule initialClaim
  let feExecution := Transcript.Fe.derive feTranscript pre.state certificate.fe
  let sourceCertificate := certificateAtSources data certificate inputBound
  let sourceFeExecution :=
    Transcript.Fe.derive feTranscript pre.state sourceCertificate.fe
  let execution :=
    derive projectInput schedule priorState profile statement certificate
  have sourceFeExecution_eq : sourceFeExecution = feExecution := by
    exact feDerive_atSources_eq data certificate inputBound feTranscript
      pre.state
  change Semantics.Paper.Holds data ∨
    ¬ OutputBound covers data execution certificate.output ∨
    BadEvent profile covers data pre.challenges execution
      sourceCertificate challengeSetSize
  change
    Fe.Accepted feTranscript pre.state profile polynomialInput
        pre.challenges.feCoins certificate.output certificate.fe ∧
      Nc.BlockLane.Accepted (ncMachine schedule) feExecution.finalState
        pre.challenges.ncCoins certificate.output certificate.nc at accepted
  rcases accepted with ⟨feAccepted, ncAccepted⟩
  have sourceFeAccepted :
      Fe.Accepted feTranscript pre.state profile (PublicInput.ofSources data)
        pre.challenges.feCoins certificate.output sourceCertificate.fe := by
    exact feAccepted_atSources data certificate inputBound feTranscript
      pre.state profile pre.challenges.feCoins certificate.output feAccepted
  have feSoundness := Fe.accepted_implies_truth_or_mismatch_or_badEvent
      feTranscript pre.state profile data pre.challenges.feCoins
      certificate.output sourceCertificate.fe challengeSetSize sourceFeAccepted
  change
    Semantics.Fe.Truth data ∨
      Polynomial.Fe.OutputMismatch data sourceFeExecution.challengePoint
        certificate.output ∨
      SumCheck.Fe.BadEvent profile data pre.challenges.feCoins
        sourceFeExecution.challengePoint sourceCertificate.fe challengeSetSize
      at feSoundness
  rw [sourceFeExecution_eq] at feSoundness
  rcases feSoundness with
    feTruth | feMismatch | feBad
  · rcases Nc.BlockLane.accepted_implies_truth_or_unbound_or_badEvent
        noZeroDivisors covers data (ncMachine schedule)
        feExecution.finalState pre.challenges.ncCoins certificate.output
        certificate.nc challengeSetSize ncAccepted with
      ncTruth | ncUnbound | ncBad
    · exact Or.inl <| (Semantics.truth_iff_paperHolds data).mp
        ⟨feTruth, ncTruth⟩
    · apply Or.inr
      apply Or.inl
      intro bound
      exact ncUnbound bound.2
    · apply Or.inr
      apply Or.inr
      apply BadEvent.nc
      simpa [sourceCertificate, certificateAtSources_nc] using ncBad
  · apply Or.inr
    apply Or.inl
    intro bound
    exact feMismatch bound.1
  · exact Or.inr (Or.inr (.fe feBad))

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane
