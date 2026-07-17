import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe.HonestProver
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.HonestProver

/-!
Sequential honest-prover composition for the Split-NC `Pi_CCS` verifier.

Owns: FE certificate construction, exact FE-to-NC transcript-state handoff,
NC certificate construction, canonical source-derived output claims at the
two verifier-derived points, and honest protocol acceptance.

Does not own: derivation of the FE/NC coin records, concrete Poseidon2 event
encoding, Fiat--Shamir probability, the paper single-`Q` transcript,
PiRLC handoff, Rust, R1CS, rows, costs, or row removal.

Emits constraints: no.

Authority boundary: the certificate output is `canonicalClaims` evaluated at
the FE row point and NC column point derived from the same sequential replay.
Neither claim family, either point, nor the FE-to-NC state boundary is
caller-supplied. The coin records remain explicit verifier inputs owned by a
later complete transcript-schedule theorem.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.prover.fe` | construct physical FE messages before their challenges | derived | `Fe.HonestProver.exists_honest_certificate` |
| `nifs.pi_ccs.prover.handoff` | initialize NC from FE's exact final state | direct dataflow | `complete_of_paperObligations` |
| `nifs.pi_ccs.prover.nc` | construct physical NC messages before their challenges | derived | `Nc.HonestProver.complete_of_truth` |
| `nifs.pi_ccs.prover.output` | compute both raw output branches from sources at derived points | computed | `canonicalClaims` |
| `nifs.pi_ccs.prover.output_authority` | canonical output satisfies both source bindings | derived | `canonicalClaims_boundToSources` |
| `nifs.pi_ccs.prover.completeness` | paper obligations yield accepted sequential protocol certificate | derived | `complete_of_paperObligations` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.HonestProver

open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

/-- The canonical NC output terminal is the independent NC polynomial at the
transcript-derived point. -/
private theorem canonicalNcTerminal
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (covers : domain.Covers shape)
    (data : Data shape)
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Polynomial.Nc.Mixing.Coins domain)
    (certificate : Transcript.Nc.Certificate domain)
    (rPrime : CubePoint Nightstream.SuperNeo.Concrete.K shape.rowVariables) :
    let point := (Nc.derive machine initialState certificate).point
    let points : VerifierPoints shape domain := {
      rPrime := rPrime
      sPrime := point.column
    }
    Polynomial.Nc.Terminal.terminalFromMessage .paperNc
        (canonicalClaims covers data points) coins point =
      Polynomial.Nc.InitialSum.sumcheckPolynomial .paperNc
        covers data coins point.coordinates := by
  let point := (Nc.derive machine initialState certificate).point
  let points : VerifierPoints shape domain := {
    rPrime := rPrime
    sPrime := point.column
  }
  let ncPoints : VerifierPoints shape domain := {
    rPrime := data.priorPoint
    sPrime := point.column
  }
  have boundAtOutput :
      YZcolBoundToSources covers data points
        (canonicalClaims covers data points) :=
    canonicalClaims_yZcolBoundToSources covers data points
  have sameColumn : ncPoints.sPrime = points.sPrime := by
    rfl
  have boundAtNc :
      YZcolBoundToSources covers data ncPoints
        (canonicalClaims covers data points) :=
    (yZcolBoundToSources_iff_of_sPrime_eq covers data ncPoints points
      (canonicalClaims covers data points) sameColumn).mpr boundAtOutput
  calc
    Polynomial.Nc.Terminal.terminalFromMessage .paperNc
        (canonicalClaims covers data points) coins point =
        Polynomial.Nc.Mixing.qAtPoint .paperNc covers data coins point :=
      Polynomial.Nc.Terminal.terminal_eq_qAtPoint_of_yZcolBoundToSources
        .paperNc covers data coins point
        (canonicalClaims covers data points) boundAtNc
    _ = Polynomial.Nc.InitialSum.sumcheckPolynomial .paperNc
          covers data coins point.coordinates :=
      (Polynomial.Nc.InitialSum.sumcheckPolynomial_coordinates_eq_qAtPoint
        .paperNc covers data coins point).symm

/-- For explicit verifier-owned FE/NC coins, every source satisfying the
paper obligation set has one accepted sequential Split-NC certificate whose
raw output is source-bound at the same replay-derived points.

The theorem closes honest message construction and phase handoff. It does not
yet claim that the explicit coin records equal a concrete Poseidon2 transcript
schedule; that remains a separate refinement obligation. -/
theorem complete_of_paperObligations
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (covers : domain.Covers shape)
    (feMachine : Transcript.Fe.Machine State)
    (ncMachine : Transcript.Nc.Machine State)
    (initialState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape)
    (feCoins : Polynomial.Fe.Coins shape domain)
    (ncCoins : Polynomial.Nc.Mixing.Coins domain)
    (obligations : Semantics.Paper.Holds data) :
    ∃ certificate :
        Protocol.Certificate (PublicInput.ofSources data) domain,
      Protocol.Accepted feMachine ncMachine initialState profile
          (PublicInput.ofSources data) feCoins ncCoins certificate ∧
        BoundToSources covers data
          (Protocol.derive feMachine ncMachine initialState certificate).outputPoints
          certificate.output := by
  have feTruth : Semantics.Fe.Truth data :=
    ⟨obligations.1, obligations.2.2⟩
  have ncTruth : Semantics.Nc.Truth data :=
    obligations.2.1
  rcases Fe.HonestProver.exists_honest_certificate
      profile data feMachine initialState feCoins with
    ⟨feCertificate, feHonest⟩
  let feExecution :=
    Transcript.Fe.derive feMachine initialState feCertificate
  rcases Nc.HonestProver.complete_of_truth .paperNc covers data ncMachine
      feExecution.finalState ncCoins ncTruth with
    ⟨ncCertificate, ncSemanticAccepted⟩
  let ncExecution :=
    Nc.derive ncMachine feExecution.finalState ncCertificate
  let points : VerifierPoints shape domain := {
    rPrime := feExecution.challengePoint.row
    sPrime := ncExecution.point.column
  }
  let output := canonicalClaims covers data points
  let certificate :
      Protocol.Certificate (PublicInput.ofSources data) domain := {
    fe := feCertificate
    nc := ncCertificate
    output := output
  }
  have feMessageBound :
      output.yRing =
        Polynomial.Fe.sourceYRingAt data feExecution.challengePoint.row := by
    rfl
  have feAccepted :
      Fe.Accepted feMachine initialState profile
        (PublicInput.ofSources data) feCoins output feCertificate :=
    Fe.accepted_of_truth_and_honestAt feMachine initialState profile data
      feCoins output feCertificate feTruth feMessageBound feHonest
  have ncTerminal :
      Polynomial.Nc.Terminal.terminalFromMessage .paperNc output ncCoins
          ncExecution.point =
        Polynomial.Nc.InitialSum.sumcheckPolynomial .paperNc
          covers data ncCoins ncExecution.point.coordinates := by
    simpa [output, points, ncExecution, feExecution] using
      canonicalNcTerminal covers data ncMachine feExecution.finalState
        ncCoins ncCertificate feExecution.challengePoint.row
  have ncAccepted :
      Nc.Accepted ncMachine feExecution.finalState ncCoins output
        ncCertificate := by
    unfold Nc.Accepted
    change SumCheck.Nc.Accepted Polynomial.Nc.InitialSum.claimedInitial
      ncExecution.point.coordinates
      (Polynomial.Nc.Terminal.terminalFromMessage .paperNc output ncCoins
        ncExecution.point)
      ncCertificate.toSumCheck
    rw [ncTerminal]
    unfold Transcript.Nc.Accepted at ncSemanticAccepted
    rw [Nc.derive_point_coordinates]
    exact ncSemanticAccepted
  refine ⟨certificate, ⟨?_, ?_⟩⟩
  · exact ⟨feAccepted, ncAccepted⟩
  · change BoundToSources covers data points output
    exact canonicalClaims_boundToSources covers data points

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.HonestProver
