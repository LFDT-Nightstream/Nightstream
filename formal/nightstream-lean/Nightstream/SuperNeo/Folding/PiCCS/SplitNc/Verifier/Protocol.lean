import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.Semantics
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc

/-!
Protocol-level composition of the production-shaped Split-NC `Pi_CCS`
verifier.

Owns: sequential FE-to-NC transcript state flow; the shared raw output
message; verifier-derived row/column output points; and deterministic
soundness of the composed FE and NC phase evaluators against the independent
Section 7.3 statement.

Does not own: commitment/public-input forwarding, derivation of the FE/NC
coin records, an honest Fiat--Shamir prover fixed point, probability bounds,
equivalence with the paper's single displayed `Q`, PiRLC handoff, Poseidon2
refinement, Rust, R1CS, rows, costs, or row removal.

Emits constraints: no.

Authority boundary: a protocol certificate contains only physical FE and NC
round messages plus raw `yRing`/`yZcol` values. FE derives its point first;
NC starts from FE's exact outgoing transcript state. The combined output
claim is authoritative only when both branches are bound to the sole source
family at those two derived points.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.paper.statement` | fresh CCS, all-source strict norm, and carried evaluations | independent specification | `Semantics.Paper.Holds` |
| `nifs.pi_ccs.semantic.split_exact` | Split-NC truth is exactly the generalized paper relation-obligation set; verifier-flow equivalence is not claimed | derived | `Semantics.truth_iff_paperHolds` |
| `nifs.pi_ccs.verify.certificate` | one FE certificate, one NC certificate, one raw output product | checked by type | `Certificate` |
| `nifs.pi_ccs.verify.transcript_handoff` | NC starts from FE's exact outgoing state | direct dataflow | `Accepted`, `check`, `derive` |
| `nifs.pi_ccs.verify.output_points` | FE row point and NC column point form the output authority pair | computed | `Execution.outputPoints` |
| `nifs.pi_ccs.verify.output_authority` | both output branches bind to the same source family | security boundary | `OutputClaims.BoundToSources` |
| `nifs.pi_ccs.verify.soundness` | acceptance implies paper obligations, failed output authority, or a named FE/NC bad event | derived | `accepted_implies_paperObligations_or_unbound_or_badEvent` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol

open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

/-- Complete prover-visible payload for the two-phase verifier. Coins,
challenge points, transcript states, and semantic witnesses are absent. -/
structure Certificate
    {shape : SemanticShape}
    (input : PublicInput shape)
    (domain : FlatNcDomain) where
  fe : SumCheck.Fe.Certificate input domain
  nc : Transcript.Nc.Certificate domain
  output : OutputMessage shape

/-- Verifier-derived phase conclusions and final transcript state. -/
structure Execution
    (shape : SemanticShape)
    (domain : FlatNcDomain)
    (State : Type uState) where
  fePoint : Polynomial.Fe.Point shape domain
  ncPoint : Polynomial.Nc.Point domain
  finalState : State

namespace Execution

/-- The only output points: FE owns the row point and NC owns the column
point. Their internal lane points remain phase-local. -/
def outputPoints
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (execution : Execution shape domain State) :
    VerifierPoints shape domain where
  rPrime := execution.fePoint.row
  sPrime := execution.ncPoint.column

end Execution

/-- Replay FE first, then start NC from FE's exact outgoing state. -/
def derive
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {input : PublicInput shape}
    (feMachine : Transcript.Fe.Machine State)
    (ncMachine : Transcript.Nc.Machine State)
    (initialState : State)
    (certificate : Certificate input domain) :
    Execution shape domain State :=
  let feExecution := Transcript.Fe.derive feMachine initialState certificate.fe
  let ncExecution := Nc.derive ncMachine feExecution.finalState certificate.nc
  {
    fePoint := feExecution.challengePoint
    ncPoint := ncExecution.point
    finalState := ncExecution.finalState
  }

/-- Logical acceptance of both sequential phases over one raw output product. -/
def Accepted
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (feMachine : Transcript.Fe.Machine State)
    (ncMachine : Transcript.Nc.Machine State)
    (initialState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (input : PublicInput shape)
    (feCoins : Polynomial.Fe.Coins shape domain)
    (ncCoins : Polynomial.Nc.Mixing.Coins domain)
    (certificate : Certificate input domain) : Prop :=
  Fe.Accepted feMachine initialState profile input feCoins certificate.output
      certificate.fe ∧
    Nc.Accepted ncMachine
      (Transcript.Fe.derive feMachine initialState certificate.fe).finalState
      ncCoins certificate.output certificate.nc

/-- Executable two-phase checker with the same exact FE-to-NC state handoff. -/
def check
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (feMachine : Transcript.Fe.Machine State)
    (ncMachine : Transcript.Nc.Machine State)
    (initialState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (input : PublicInput shape)
    (feCoins : Polynomial.Fe.Coins shape domain)
    (ncCoins : Polynomial.Nc.Mixing.Coins domain)
    (certificate : Certificate input domain) : Bool :=
  Fe.check feMachine initialState profile input feCoins certificate.output
      certificate.fe &&
    Nc.check ncMachine
      (Transcript.Fe.derive feMachine initialState certificate.fe).finalState
      ncCoins certificate.output certificate.nc

/-- Executable and logical protocol acceptance coincide exactly. -/
theorem check_eq_true_iff_accepted
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (feMachine : Transcript.Fe.Machine State)
    (ncMachine : Transcript.Nc.Machine State)
    (initialState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (input : PublicInput shape)
    (feCoins : Polynomial.Fe.Coins shape domain)
    (ncCoins : Polynomial.Nc.Mixing.Coins domain)
    (certificate : Certificate input domain) :
    check feMachine ncMachine initialState profile input feCoins ncCoins
        certificate = true ↔
      Accepted feMachine ncMachine initialState profile input feCoins ncCoins
        certificate := by
  simp only [check, Accepted, Bool.and_eq_true]
  rw [Fe.check_eq_true_iff_accepted, Nc.check_eq_true_iff_accepted]

/-- The FE source-binding predicate is exactly whole-function equality with
the source-derived `yRing` value at the execution's row point. -/
theorem yRingBoundToSources_iff_yRing_eq
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (data : Data shape)
    (execution : Execution shape domain State)
    (message : OutputMessage shape) :
    YRingBoundToSources data execution.outputPoints message ↔
      message.yRing =
        Polynomial.Fe.sourceYRingAt data execution.fePoint.row := by
  constructor
  · intro bound
    funext source matrix lane
    simpa [Execution.outputPoints, canonicalYRing,
      Polynomial.Fe.sourceYRingAt] using bound source matrix lane
  · intro equal source matrix lane
    have coordinate :=
      congrFun (congrFun (congrFun equal source) matrix) lane
    simpa [Execution.outputPoints, canonicalYRing,
      Polynomial.Fe.sourceYRingAt] using coordinate

/-- Algebraic bad events remain owned by the phase that exposes them. -/
inductive BadEvent
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (covers : domain.Covers shape)
    (data : Data shape)
    (feCoins : Polynomial.Fe.Coins shape domain)
    (ncCoins : Polynomial.Nc.Mixing.Coins domain)
    (execution : Execution shape domain State)
    (certificate : Certificate (PublicInput.ofSources data) domain)
    (challengeSetSize : Nat) : Prop where
  | fe
      (bad :
        SumCheck.Fe.BadEvent profile data feCoins execution.fePoint
          certificate.fe challengeSetSize) :
      BadEvent profile covers data feCoins ncCoins execution certificate
        challengeSetSize
  | nc
      (bad :
        SumCheck.Nc.BadEvent covers
          data ncCoins execution.ncPoint.coordinates certificate.nc.toSumCheck
          challengeSetSize) :
      BadEvent profile covers data feCoins ncCoins execution certificate
        challengeSetSize

/-- Deterministic soundness of the complete two-phase semantic verifier.

The theorem does not assign authority to a digest or to a self-consistent raw
output. If either output branch is not source-bound at the two points derived
by this same execution, that failure remains an explicit outcome. -/
theorem accepted_implies_paperObligations_or_unbound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
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
    (certificate :
      Certificate (PublicInput.ofSources data) domain)
    (challengeSetSize : Nat)
    (accepted :
      Accepted feMachine ncMachine initialState profile
        (PublicInput.ofSources data) feCoins ncCoins certificate) :
    let execution :=
      derive feMachine ncMachine initialState certificate
    Semantics.Paper.Holds data ∨
      ¬ BoundToSources covers data execution.outputPoints certificate.output ∨
      BadEvent profile covers data feCoins ncCoins execution certificate
        challengeSetSize := by
  let execution := derive feMachine ncMachine initialState certificate
  change Semantics.Paper.Holds data ∨
    ¬ BoundToSources covers data execution.outputPoints certificate.output ∨
    BadEvent profile covers data feCoins ncCoins execution certificate
      challengeSetSize
  change
    Fe.Accepted feMachine initialState profile (PublicInput.ofSources data)
        feCoins certificate.output certificate.fe ∧
      Nc.Accepted ncMachine
        (Transcript.Fe.derive feMachine initialState certificate.fe).finalState
        ncCoins certificate.output certificate.nc at accepted
  rcases accepted with ⟨feAccepted, ncAccepted⟩
  rcases Fe.accepted_implies_truth_or_mismatch_or_badEvent
      feMachine initialState profile data feCoins certificate.output
      certificate.fe challengeSetSize feAccepted with
    feTruth | feMismatch | feBad
  · rcases Nc.accepted_implies_truth_or_unbound_or_badEvent
        noZeroDivisors covers data ncMachine
        (Transcript.Fe.derive feMachine initialState certificate.fe).finalState
        ncCoins certificate.output certificate.nc challengeSetSize ncAccepted with
      ncTruth | ncUnbound | ncBad
    · exact Or.inl <|
        (Semantics.truth_iff_paperHolds data).mp ⟨feTruth, ncTruth⟩
    · apply Or.inr
      apply Or.inl
      intro bound
      apply ncUnbound
      let ncPoints : VerifierPoints shape domain := {
        rPrime := data.priorPoint
        sPrime := execution.ncPoint.column
      }
      have sameColumn :
          ncPoints.sPrime = execution.outputPoints.sPrime := by
        rfl
      exact (yZcolBoundToSources_iff_of_sPrime_eq covers data
        ncPoints execution.outputPoints certificate.output sameColumn).mpr
          bound.yZcol
    · exact Or.inr (Or.inr (.nc ncBad))
  · apply Or.inr
    apply Or.inl
    intro bound
    apply feMismatch
    exact (yRingBoundToSources_iff_yRing_eq data execution
      certificate.output).mp bound.yRing
  · exact Or.inr (Or.inr (.fe feBad))

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol
