import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.RunningAuthority
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiRlc
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DerivedPiDec
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.OutputRefinement

/-!
Independent concrete Phi81 NIFS transition over the exact Split-NC
`Pi_CCS → Pi_RLC → Pi_DEC` dataflow.

Protocol: SuperNeo NIFS.
Phase: complete three-phase semantic verifier.
Constraint family: logical acceptance only; this file emits no rows.

Owns: a strict separation between executable verifier acceptance and semantic
source authority; checked incoming accumulator authority; sampler-derived
`Pi_RLC` challenge authority; the independently proved concrete `Pi_RLC`
equations; the three retained outgoing `Pi_DEC` recomposition equations over
canonical children; an independent semantic transition predicate; and
deterministic soundness with output mismatch and FE/NC bad events explicit.

Does not own: Poseidon2 instantiation of the abstract sampler machine,
extraction, Ajtai/MSIS binding security, F-prime selection/lifecycle, Rust,
R1CS, rows, costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: `Accepted` reads public context and raw certificate only.
It contains no private source data, opening witness, paper truth, or output
truth. Bootstrap requires an absent incoming parent; active mode validates the
complete transcript-bound parent against the exact running children.
`SemanticInput` separately binds a rich independent source family to the
public polynomial input and source product. `OutputBound` remains a separate
semantic outcome: physical acceptance alone does not assume it. A failed
output binding or SumCheck mixing claim is returned explicitly rather than
silently promoted to verifier authority.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.concrete.input.projection` | public polynomial input is exactly the source projection | semantic bridge | `PublicInputBound` |
| `nifs.concrete.input.sources` | every public source field opens the aligned semantic source | semantic bridge | `InputBound` |
| `nifs.concrete.running_authority` | bootstrap has no parent; active parent strictly recomposes to the `k` running children | checked | `RunningAuthority.Accepted` |
| `nifs.concrete.pi_ccs` | exact physical FE→NC transcript accepts | checked | `PiCcsAccepted` |
| `nifs.concrete.pi_ccs.output` | complete `yRing`/`yZcol` binds to the independent sources | explicit semantic outcome | `OutputBound` |
| `nifs.concrete.pi_rlc.sampler` | replay binds every challenge and derives production-set membership | checked/derived | `Accepted.sampler`, `TailAccepted.piRlcAccepted` |
| `nifs.concrete.pi_rlc.source_structure` | every materialized source uses the verifier-selected structure | checked | `TailAccepted.sourceStructures` |
| `nifs.concrete.pi_rlc.derived_equations` | stage, point, commitment, public input, and evaluations of the parent are canonical | computed | `TailAccepted.piRlcEquations` |
| `nifs.concrete.pi_dec.recomposition` | canonical child payloads recompose to the one derived parent | checked | `TailAccepted.piDecRecomposition` |
| `nifs.concrete.pi_dec.inherited` | child structure, point, and fresh stage are inherited from the parent | computed | `TailAccepted.piDec` |
| `nifs.concrete.semantic` | paper source truth plus explicit semantic bridges and physical tail checks | independent specification | `Holds` |
| `nifs.concrete.soundness` | physical acceptance plus semantic input authority implies transition, output mismatch, or named FE/NC bad event | derived | `accepted_implies_transition_or_outputUnbound_or_badEvent` |
| `nifs.concrete.completeness` | honest paper sources and valid challenges construct all three phases and valid children | derived | `complete_of_paperObligations` |
| `nifs.concrete.completeness.outcome` | honest paper sources produce a transition or one exact bounded-sampler shortfall | exhaustive model outcome | `complete_or_samplerShortfall` |
| `nifs.concrete.public_relation` | hide the raw certificate while fixing exact input context and child output | existential projection | `Transition` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

variable
  {shape : SemanticShape}
  {domain : FlatNcDomain}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits :
    ringDegree * publicRingColumns <= shape.carrierWidth}
  {arity : BatchArity productionGlobalParams}

/-- The polynomial verifier input is exactly the public projection of one
independent semantic source family. This is not an executable verifier check;
it is the refinement bridge used to interpret physical acceptance. -/
def PublicInputBound
    (context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity)
    (data : Data shape) : Prop :=
  context.piCcsInput = PublicInput.ofSources data

/-- Exact source-product authority used by the semantic refinement. This is
not a field of physical verifier acceptance. -/
def InputBound
    (context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity)
    (data : Data shape) : Prop :=
  InputAuthority.BoundToSources publicRingColumns publicFits
    (commit context.key) data context.alignment context.input

/-- Complete semantic input authority. The two bridges are deliberately
separate because the polynomial verifier input and public source product are
distinct protocol surfaces. -/
structure SemanticInput
    (context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity)
    (data : Data shape) : Prop where
  publicInput : PublicInputBound context data
  sources : InputBound context data

/-- Exact physical Split-NC phase acceptance over the public polynomial input.
No source witness or semantic output binding is read here. -/
def PiCcsAccepted
    (context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity)
    (certificate :
      Certificate (domain := domain) (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) : Prop :=
  Protocol.Accepted
    context.feMachine context.ncMachine context.initialState context.profile
    context.piCcsInput context.feCoins context.ncCoins
    certificate.piCcs

/-- Complete semantic output binding at the two points derived by the same
physical transcript. This remains an explicit soundness outcome, not a field
of `Accepted`. -/
def OutputBound
    (context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity)
    (data : Data shape)
    (certificate :
      Certificate (domain := domain) (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) : Prop :=
  OutputClaims.BoundToSources context.covers data
    (derive context certificate).piCcs.outputPoints certificate.piCcs.output

/-- Tail acceptance over the unique derived `Pi_CCS` product and `Pi_RLC`
parent. -/
structure TailAccepted
    (context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity)
    (certificate :
      Certificate (domain := domain) (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) : Prop where
  sourceStructures : DerivedPiRlc.SourceStructuresBound context
  piDecRecomposition :
    DerivedPiDec.RecompositionEquations context certificate

namespace TailAccepted

/-- The canonical dataflow derives all public `Pi_RLC` equations from the one
retained source-structure family. This is an eliminated check family, not an
additional field of `TailAccepted`. -/
def piRlcEquations
    {context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity}
    {certificate :
      Certificate (domain := domain) (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (tail : TailAccepted context certificate) :
    PiRLC.Equations (rlcAlgebra context.key)
      ((derive context certificate).piRlcAttempt certificate) :=
  DerivedPiRlc.equations_of_sourceStructures tail.sourceStructures

/-- Exact sampler replay supplies the challenge-validity theorem needed to
assemble complete model-level Π_RLC acceptance. There is no independent
challenge-membership check in the concrete transition. -/
def piRlcAccepted
    {context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity}
    {certificate :
      Certificate (domain := domain) (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (tail : TailAccepted context certificate)
    (sampler : Sampler.CertificateAccepted context certificate) :
    PiRLC.Accepted (rlcAlgebra context.key)
      ((derive context certificate).piRlcAttempt certificate) :=
  tail.piRlcEquations.withChallengesValid
    (Sampler.certificateAccepted_challengesValid sampler)

/-- The retained recomposition equations assemble complete model-level
`PiDEC.Accepted`; inherited child fields are construction facts rather than
independent checks. -/
def piDec
    {context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity}
    {certificate :
      Certificate (domain := domain) (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (tail : TailAccepted context certificate) :
    PiDEC.Accepted (decAlgebra context.key)
      ((derive context certificate).piDecAttempt certificate) :=
  DerivedPiDec.accepted_of_recomposition tail.piDecRecomposition

end TailAccepted

/-- Complete physical verifier acceptance. -/
structure Accepted
    (context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity)
    (certificate :
      Certificate (domain := domain) (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) : Prop where
  running : RunningAuthority.Accepted context
  piCcs : PiCcsAccepted context certificate
  sampler : Sampler.CertificateAccepted context certificate
  tail : TailAccepted context certificate

/-- Independent semantic transition for the same raw certificate and source
family. The public/private bridges are explicit and remain outside physical
verifier acceptance. -/
structure Holds
    (context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity)
    (data : Data shape)
    (certificate :
      Certificate (domain := domain) (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) : Prop where
  paper : Semantics.Paper.Holds data
  input : SemanticInput context data
  running : RunningAuthority.Accepted context
  output : OutputBound context data certificate
  sampler : Sampler.CertificateAccepted context certificate
  tail : TailAccepted context certificate

/-- Transport a physical protocol certificate across an explicit semantic
input-projection bridge. The input is an ordinary variable rather than a
projection from a dependent context, so the refinement is kernel-transparent. -/
def semanticProtocolCertificate
    (input : PublicInput shape)
    (data : Data shape)
    (certificate : Protocol.Certificate input domain)
    (bound : input = PublicInput.ofSources data) :
    Protocol.Certificate (PublicInput.ofSources data) domain :=
  Eq.mp
    (congrArg
      (fun input => Protocol.Certificate input domain)
      bound)
    certificate

/-- A phase bad event interpreted through an explicit source projection,
without assigning semantic authority to the physical public input itself. -/
def ProtocolBadEventAtSources
    (covers : domain.Covers shape)
    (feMachine : Transcript.Fe.Machine State)
    (ncMachine : Transcript.Nc.Machine State)
    (initialState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (feCoins : Polynomial.Fe.Coins shape domain)
    (ncCoins : Polynomial.Nc.Mixing.Coins domain)
    (challengeSetSize : Nat)
    (input : PublicInput shape)
    (data : Data shape)
    (certificate : Protocol.Certificate input domain) : Prop :=
  ∃ bound : input = PublicInput.ofSources data,
    let semanticCertificate :=
      semanticProtocolCertificate input data certificate bound
    Protocol.BadEvent
      profile covers data feCoins ncCoins
      (Protocol.derive feMachine ncMachine initialState semanticCertificate)
      semanticCertificate challengeSetSize

/-- Generic physical-to-semantic Split-NC soundness at an explicit public
input variable. This is the only place where the projection equality is
eliminated. -/
private theorem protocolAccepted_implies_paper_or_outputUnbound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (covers : domain.Covers shape)
    (feMachine : Transcript.Fe.Machine State)
    (ncMachine : Transcript.Nc.Machine State)
    (initialState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (feCoins : Polynomial.Fe.Coins shape domain)
    (ncCoins : Polynomial.Nc.Mixing.Coins domain)
    (challengeSetSize : Nat)
    (input : PublicInput shape)
    (data : Data shape)
    (certificate : Protocol.Certificate input domain)
    (inputBound : input = PublicInput.ofSources data)
    (accepted :
      Protocol.Accepted feMachine ncMachine initialState profile input feCoins
        ncCoins certificate) :
    Semantics.Paper.Holds data ∨
      ¬ OutputClaims.BoundToSources covers data
          (Protocol.derive feMachine ncMachine initialState
            certificate).outputPoints
          certificate.output ∨
      ProtocolBadEventAtSources covers feMachine ncMachine initialState
        profile feCoins ncCoins challengeSetSize input data certificate := by
  cases inputBound
  have phaseSoundness :=
    Protocol.accepted_implies_paperObligations_or_unbound_or_badEvent
      noZeroDivisors covers feMachine ncMachine initialState profile data
      feCoins ncCoins certificate challengeSetSize accepted
  rcases phaseSoundness with paper | unbound | bad
  · exact Or.inl paper
  · exact Or.inr (Or.inl unbound)
  · apply Or.inr
    apply Or.inr
    refine ⟨rfl, ?_⟩
    simpa [semanticProtocolCertificate] using bad

/-- Generic honest Split-NC construction at an explicitly bound public input. -/
private theorem exists_piCcsCertificate_of_paper
    (covers : domain.Covers shape)
    (feMachine : Transcript.Fe.Machine State)
    (ncMachine : Transcript.Nc.Machine State)
    (initialState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (feCoins : Polynomial.Fe.Coins shape domain)
    (ncCoins : Polynomial.Nc.Mixing.Coins domain)
    (input : PublicInput shape)
    (data : Data shape)
    (inputBound : input = PublicInput.ofSources data)
    (paper : Semantics.Paper.Holds data) :
    ∃ certificate : Protocol.Certificate input domain,
      Protocol.Accepted feMachine ncMachine initialState profile input feCoins
          ncCoins certificate ∧
        OutputClaims.BoundToSources covers data
          (Protocol.derive feMachine ncMachine initialState
            certificate).outputPoints
          certificate.output := by
  cases inputBound
  exact Protocol.HonestProver.complete_of_paperObligations
    covers feMachine ncMachine initialState profile data feCoins ncCoins paper

/-- Named FE/NC algebraic failure, interpreted only through an explicit
public-input projection bridge. -/
def PiCcsBadEvent
    (context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity)
    (data : Data shape)
    (certificate :
      Certificate (domain := domain) (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) : Prop :=
  ProtocolBadEventAtSources context.covers context.feMachine context.ncMachine
    context.initialState context.profile context.feCoins context.ncCoins
    context.challengeSetSize context.piCcsInput data certificate.piCcs

/-- External concrete NIFS relation: public context and child output remain
visible, while the independent semantic source family and raw certificate are
existential witnesses. -/
def Transition
    (context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity)
    (output : Fin productionGlobalParams.k ->
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) : Prop :=
  ∃ data : Data shape,
    ∃ certificate :
        Certificate (domain := domain) (arity := arity)
          publicRingColumns publicFits verifierRows context.piCcsInput,
      outputChildren context certificate = output ∧
        Holds context data certificate

/-- Physical Split-NC acceptance implies the independent paper obligations,
an explicit output-binding failure, or one named FE/NC bad event once the
public polynomial input is known to be the source projection. -/
theorem accepted_implies_paper_or_outputUnbound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity}
    {data : Data shape}
    {certificate :
      Certificate (domain := domain) (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (publicInputBound : PublicInputBound context data)
    (accepted : Accepted context certificate) :
    Semantics.Paper.Holds data ∨
      ¬ OutputBound context data certificate ∨
      PiCcsBadEvent context data certificate := by
  exact protocolAccepted_implies_paper_or_outputUnbound_or_badEvent
    noZeroDivisors context.covers context.feMachine context.ncMachine
    context.initialState context.profile context.feCoins context.ncCoins
    context.challengeSetSize context.piCcsInput data certificate.piCcs
    publicInputBound accepted.piCcs

/-- Exact Split-NC soundness lifts through both shared tail phases without
smuggling semantic output truth into physical acceptance. -/
theorem accepted_implies_holds_or_outputUnbound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity}
    {data : Data shape}
    {certificate :
      Certificate (domain := domain) (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (input : SemanticInput context data)
    (accepted : Accepted context certificate) :
    Holds context data certificate ∨
      ¬ OutputBound context data certificate ∨
      PiCcsBadEvent context data certificate := by
  rcases accepted_implies_paper_or_outputUnbound_or_badEvent
      noZeroDivisors input.publicInput accepted with
    paper | outputUnbound | bad
  · by_cases output : OutputBound context data certificate
    · exact Or.inl {
        paper := paper
        input := input
        running := accepted.running
        output := output
        sampler := accepted.sampler
        tail := accepted.tail
      }
    · exact Or.inr (Or.inl output)
  · exact Or.inr (Or.inl outputUnbound)
  · exact Or.inr (Or.inr bad)

/-- Public projection of physical acceptance is the independent concrete
transition, an explicit output-binding failure, or one named FE/NC bad event. -/
theorem accepted_implies_transition_or_outputUnbound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity}
    {data : Data shape}
    {certificate :
      Certificate (domain := domain) (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (input : SemanticInput context data)
    (accepted : Accepted context certificate) :
    Transition context (outputChildren context certificate) ∨
      ¬ OutputBound context data certificate ∨
      PiCcsBadEvent context data certificate := by
  rcases accepted_implies_holds_or_outputUnbound_or_badEvent
      noZeroDivisors input accepted with
    holds | outputUnbound | bad
  · exact Or.inl ⟨data, certificate, rfl, holds⟩
  · exact Or.inr (Or.inl outputUnbound)
  · exact Or.inr (Or.inr bad)

/-- The exact honest-completeness failure that remains possible after the
canonical `Pi_CCS` prefix has accepted and its semantic output has been bound.
The failure names one challenge coordinate whose fixed 64-candidate prefix
contains fewer than the required 54 accepted coefficients. -/
def HonestSamplerShortfall
    (context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity)
    (data : Data shape) : Prop :=
  ∃ piCcsCertificate : Protocol.Certificate context.piCcsInput domain,
    Protocol.Accepted context.feMachine context.ncMachine
        context.initialState context.profile context.piCcsInput
        context.feCoins context.ncCoins piCcsCertificate ∧
      OutputClaims.BoundToSources context.covers data
        (Protocol.derive context.feMachine context.ncMachine
          context.initialState piCcsCertificate).outputPoints
        piCcsCertificate.output ∧
      Exists fun coordinate : Fin arity.total =>
        Nifs.NonInteractive.PiRlcSampler.ShortfallAt
          (Sampler.Specification context.piRlcMachine)
          Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.candidateBound
          (context.piCcsOutputHandoff
            (Protocol.derive context.feMachine context.ncMachine
              context.initialState piCcsCertificate).finalState
            piCcsCertificate.output)
          coordinate.val

/-- Finish the concrete NIFS honest construction from the one canonical
honest `Pi_CCS` prefix and one successful bounded sampler batch. Keeping this
step separate prevents the public completeness theorem from requiring one
challenge vector to work for every accepted `Pi_CCS` certificate. -/
private theorem complete_of_honestPiCcsAndSampler
    (context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity)
    (data : Data shape)
    (paper : Semantics.Paper.Holds data)
    (input : SemanticInput context data)
    (running : RunningAuthority.Accepted context)
    (piCcsCertificate : Protocol.Certificate context.piCcsInput domain)
    (piCcsTranscript :
      Protocol.Accepted context.feMachine context.ncMachine
        context.initialState context.profile context.piCcsInput
        context.feCoins context.ncCoins piCcsCertificate)
    (outputAuthority :
      OutputClaims.BoundToSources context.covers data
        (Protocol.derive context.feMachine context.ncMachine
          context.initialState piCcsCertificate).outputPoints
        piCcsCertificate.output)
    (challenges : Fin arity.total -> RingF)
    (samplerBound :
      Sampler.Bound context.piRlcMachine
        (context.piCcsOutputHandoff
          (Protocol.derive context.feMachine context.ncMachine
            context.initialState piCcsCertificate).finalState
          piCcsCertificate.output)
        challenges) :
    ∃ certificate :
        Certificate (domain := domain) (arity := arity)
          publicRingColumns publicFits verifierRows context.piCcsInput,
      Accepted context certificate ∧
        Holds context data certificate ∧
        ∀ child,
          CE.Holds (semantics context.key) productionGlobalParams
            (outputChildren context certificate child)
            ((decAlgebra context.key).splitAssignment
              (PiRLC.combinedWitness (rlcAlgebra context.key) challenges
                (InputAuthority.productAssignments
                  data context.alignment))
              child) := by
  rcases input with ⟨publicInputBound, inputBound⟩
  let piCcsExecution :=
    Protocol.derive context.feMachine context.ncMachine context.initialState
      piCcsCertificate
  let piRlcInitialState :=
    context.piCcsOutputHandoff piCcsExecution.finalState
      piCcsCertificate.output
  let piCcsOutputs :=
    OutputProduct.materialize publicRingColumns publicFits context.alignment
      context.input piCcsExecution.outputPoints.rPrime
      piCcsCertificate.output
  let assignments :=
    InputAuthority.productAssignments data context.alignment
  have challengesValid :
      ∀ source, (rlcAlgebra context.key).challengeValid
        (challenges source) := by
    exact samplerBound.challengeValid
  have outputsHold :
      ProductHolds publicRingColumns publicFits (commit context.key)
        piCcsOutputs assignments :=
    Protocol.OutputRefinement.materializedOutputsHold_of_yRingBound
      publicRingColumns publicFits (commit context.key)
      data context.alignment context.input piCcsExecution.outputPoints
      piCcsCertificate.output production_norm_stages.1 paper inputBound
      outputAuthority.yRing
  have outputsValid :
      ∀ source,
        CE.Holds (semantics context.key) productionGlobalParams
          (piCcsOutputs source) (assignments source) := by
    exact outputsHold
  let system := context.system
  have systemBound :
      system =
        Phi81Relation.Structure.ofSourceData
          publicRingColumns publicFits data := by
    simpa [system, Context.system] using
      (inputBound.fresh ⟨0, arity.freshPositive⟩).constraintSystem
  let point := piCcsExecution.outputPoints.rPrime
  have rlcComplete :=
    PiRLC.complete (semantics context.key) productionGlobalParams
      (rlcAlgebra context.key) arity system point piCcsOutputs challenges
      assignments (fun _ => rfl)
      (fun source => by
        calc
          (piCcsOutputs source).constraintSystem =
              (context.input.source source).constraintSystem := by
            rfl
          _ = Phi81Relation.Structure.ofSourceData
                publicRingColumns publicFits data :=
            InputAuthority.BoundToSources.sourceStructure
              publicRingColumns publicFits (commit context.key) data
              context.alignment context.input inputBound source
          _ = system := systemBound.symm)
      (fun _ => rfl)
      challengesValid outputsValid
      (Phi81Relation.evaluationPointValid_holds system point)
  let rlcOutput :=
    PiRLC.combinedOutput (rlcAlgebra context.key) system point piCcsOutputs
      challenges
  let combinedAssignment :=
    PiRLC.combinedWitness (rlcAlgebra context.key) challenges assignments
  have decComplete :=
    PiDEC.complete (semantics context.key) productionGlobalParams
      (decAlgebra context.key) rlcOutput combinedAssignment rfl rlcComplete.2
  let children :=
    PiDEC.childrenOf (decAlgebra context.key) rlcOutput combinedAssignment
  let certificate :
      Certificate (domain := domain) (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput := {
    piCcs := piCcsCertificate
    piRlcChallenges := challenges
    piDecPayloads := fun child =>
      PiDecChildPayload.ofStatement (children child)
  }
  have tail : TailAccepted context certificate := by
    refine ⟨?_, ?_⟩
    · intro source
      calc
        (context.input.source source).constraintSystem =
            Phi81Relation.Structure.ofSourceData
              publicRingColumns publicFits data :=
          InputAuthority.BoundToSources.sourceStructure
            publicRingColumns publicFits (commit context.key) data
            context.alignment context.input inputBound source
        _ = system := systemBound.symm
        _ = context.system := rfl
    · apply DerivedPiDec.recomposition_of_accepted
      simpa [certificate, derive, piCcsExecution, piCcsOutputs, system, point,
        rlcOutput, children, Execution.piDecAttempt,
        Execution.piDecChildren, PiDecChildPayload.ofStatement,
        PiDecChildPayload.materialize] using decComplete.1
  have sampler : Sampler.CertificateAccepted context certificate := by
    refine ⟨?_⟩
    simpa [Sampler.CertificateBound, certificate, derive, piCcsExecution,
      piRlcInitialState] using samplerBound
  have physical : Accepted context certificate := {
    running := running
    piCcs := piCcsTranscript
    sampler := sampler
    tail := tail
  }
  have semantic : Holds context data certificate := {
    paper := paper
    input := ⟨publicInputBound, inputBound⟩
    running := running
    output := by
      simpa [certificate, derive, piCcsExecution] using outputAuthority
    sampler := sampler
    tail := tail
  }
  refine ⟨certificate, physical, semantic, ?_⟩
  intro child
  simpa [certificate, children, combinedAssignment, assignments] using
    decComplete.2 child

/-- Honest completeness of the exact concrete composition, conditional only
on bounded sampler availability for the honest Split-NC prefix. Membership in
the production challenge set is derived from that replay witness rather than
accepted as a separate premise.

The remaining implementation refinement must instantiate
`context.piRlcMachine` with the exact Poseidon2 schedule and prove that native
and R1CS execution provide this same bounded batch. -/
theorem complete_of_paperObligations
    (context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity)
    (data : Data shape)
    (paper : Semantics.Paper.Holds data)
    (input : SemanticInput context data)
    (running : RunningAuthority.Accepted context)
    (challenges : Fin arity.total -> RingF)
    (samplerAvailable :
      ∀ piCcsCertificate :
          Protocol.Certificate context.piCcsInput domain,
        Protocol.Accepted context.feMachine context.ncMachine
            context.initialState context.profile
            context.piCcsInput context.feCoins context.ncCoins
            piCcsCertificate →
          Sampler.Bound context.piRlcMachine
            (context.piCcsOutputHandoff
              (Protocol.derive context.feMachine context.ncMachine
                context.initialState piCcsCertificate).finalState
              piCcsCertificate.output)
            challenges) :
    ∃ certificate :
        Certificate (domain := domain) (arity := arity)
          publicRingColumns publicFits verifierRows context.piCcsInput,
      Accepted context certificate ∧
        Holds context data certificate ∧
        ∀ child,
          CE.Holds (semantics context.key) productionGlobalParams
            (outputChildren context certificate child)
            ((decAlgebra context.key).splitAssignment
              (PiRLC.combinedWitness (rlcAlgebra context.key) challenges
                (InputAuthority.productAssignments
                  data context.alignment))
              child) := by
  rcases exists_piCcsCertificate_of_paper
      context.covers context.feMachine context.ncMachine context.initialState
      context.profile context.feCoins context.ncCoins context.piCcsInput data
      input.publicInput paper with
    ⟨piCcsCertificate, piCcsTranscript, outputAuthority⟩
  have samplerBound :
      Sampler.Bound context.piRlcMachine
        (context.piCcsOutputHandoff
          (Protocol.derive context.feMachine context.ncMachine
            context.initialState piCcsCertificate).finalState
          piCcsCertificate.output)
        challenges := by
    exact samplerAvailable piCcsCertificate piCcsTranscript
  exact complete_of_honestPiCcsAndSampler context data paper input running
    piCcsCertificate piCcsTranscript outputAuthority challenges samplerBound

/-- Honest concrete NIFS completeness without a hidden total-sampler
assumption. The independently constructed `Pi_CCS` prefix either extends
through one complete transcript-bound challenge batch and both tail phases,
or the result names the exact bounded-sampler coordinate that shortfalls. -/
theorem complete_or_samplerShortfall
    (context :
      Context shape domain State publicRingColumns publicFits verifierRows
        arity)
    (data : Data shape)
    (paper : Semantics.Paper.Holds data)
    (input : SemanticInput context data)
    (running : RunningAuthority.Accepted context) :
    (∃ challenges : Fin arity.total -> RingF,
      ∃ certificate :
          Certificate (domain := domain) (arity := arity)
            publicRingColumns publicFits verifierRows context.piCcsInput,
        Accepted context certificate ∧
          Holds context data certificate ∧
          ∀ child,
            CE.Holds (semantics context.key) productionGlobalParams
              (outputChildren context certificate child)
              ((decAlgebra context.key).splitAssignment
                (PiRLC.combinedWitness (rlcAlgebra context.key) challenges
                  (InputAuthority.productAssignments
                    data context.alignment))
                child)) \/
      HonestSamplerShortfall context data := by
  rcases exists_piCcsCertificate_of_paper
      context.covers context.feMachine context.ncMachine context.initialState
      context.profile context.feCoins context.ncCoins context.piCcsInput data
      input.publicInput paper with
    ⟨piCcsCertificate, piCcsTranscript, outputAuthority⟩
  let piRlcInitialState :=
    context.piCcsOutputHandoff
      (Protocol.derive context.feMachine context.ncMachine
        context.initialState piCcsCertificate).finalState
      piCcsCertificate.output
  rcases Sampler.exists_bound_or_exists_shortfall context.piRlcMachine
      arity.total piRlcInitialState with bound | shortfall
  · rcases bound with ⟨challenges, ⟨samplerBound⟩⟩
    apply Or.inl
    refine ⟨challenges, ?_⟩
    apply complete_of_honestPiCcsAndSampler context data paper input running
      piCcsCertificate piCcsTranscript outputAuthority challenges
    simpa [piRlcInitialState] using samplerBound
  · apply Or.inr
    refine ⟨piCcsCertificate, piCcsTranscript, outputAuthority, ?_⟩
    simpa [piRlcInitialState] using shortfall

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
