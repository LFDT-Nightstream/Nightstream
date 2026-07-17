import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra
import Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputProduct
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane

/-!
Typed carrier and deterministic dataflow for the concrete Phi81 NIFS
composition.

Protocol: SuperNeo NIFS.
Phase: exact Split-NC `Pi_CCS` output → `Pi_RLC` parent → `Pi_DEC` children.
Constraint family: carrier and derived views only; this file emits no rows.

Owns: the exact typed Ajtai key/commitment aliases; the concrete relation and
`Pi_RLC`/`Pi_DEC` algebra instances; one complete public `Pi_CCS` statement;
one statement-bound transcript schedule; the raw verifier certificate; and
one deterministic execution whose internal phase boundaries are shared by
construction.

Does not own: phase acceptance, paper truth, transcript soundness, sampler
refinement, binding security, F-prime composition, Rust, R1CS, rows, costs, or
row removal.

Emits constraints: no.

Authority boundary: the transcript binds both the complete public source
product, the checked incoming accumulator parent, and the polynomial verifier
input before deriving any coin. FE and NC coins are projections of one
schedule-derived challenge record; they cannot be supplied or diverge
independently. The certificate carries only prover-visible `Pi_CCS` messages,
the complete `Pi_RLC` challenge vector, and the non-inherited public payload
of each `Pi_DEC` child. Child structure, point, and fresh stage are inherited
from the computed parent. The `Pi_CCS` CE product and outgoing `Pi_RLC` parent
are derived exactly once from public data. No digest or duplicated boundary
copy can override them.

| Stage path | Owner | Mathematical obligation | Authority class |
|---|---|---|---|
| `nifs.concrete.setup.key` | `VerifierKey`, `commit` | exact typed Ajtai key and opening map | verifier-owned setup |
| `nifs.concrete.setup.algebra` | `rlcAlgebra`, `decAlgebra` | independently proved concrete Phi81 phase algebras | computed |
| `nifs.concrete.context` | `Context` | key, source alignment, incoming parent, both public input surfaces, prior state, the fixed production domain, one transcript schedule, and supported profile | verifier-owned context |
| `nifs.concrete.pi_ccs.statement` | `Context.piCcsStatement` | bind the verifier key, source product, checked-parent carrier, and polynomial public input together | computed |
| `nifs.concrete.pi_ccs.coins` | `Context.piCcsPreSumcheck` | derive one shared FE/NC challenge record after statement binding | computed |
| `nifs.concrete.pi_ccs.fe` | `Context.feMachine`, `Context.initialState` | compute the FE initial claim and sole phase-entry state | computed |
| `nifs.concrete.pi_ccs.nc` | `Context.ncMachine` | continue through the same configured transcript schedule | computed |
| `nifs.concrete.certificate.pi_ccs` | `Certificate.piCcs` | raw FE/NC messages and output claims | prover message |
| `nifs.concrete.certificate.pi_rlc` | `Certificate.piRlcChallenges` | one scalar per canonical source | verifier-derived carrier |
| `nifs.concrete.certificate.pi_dec.payload` | `Certificate.piDecPayloads` | exactly `k` commitment/public-input/evaluation payloads | prover message |
| `nifs.concrete.derive.pi_ccs_output` | `derive` | canonical source-ordered CE product | computed |
| `nifs.concrete.derive.pi_ccs_handoff` | `Execution.piRlcInitialState` | reuse the canonical post-output Π_CCS state as the sole Π_RLC sampler root | direct dataflow |
| `nifs.concrete.pi_rlc.sampler.machine` | `Context.piRlcMachine` | deterministic four-block challenge source | verifier-owned context |
| `nifs.concrete.derive.pi_rlc_parent` | `derive` | one combined CE parent from the shared product | computed |
| `nifs.concrete.derive.pi_dec.children` | `Execution.piDecChildren` | inherit structure, point, and fresh stage from the computed parent | computed |
| `nifs.concrete.derive.pi_dec_view` | `Execution.piDecAttempt` | computed parent plus canonical children | direct dataflow |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState uCommitment

/-- Exact verifier-owned Ajtai key for the batch-invariant concrete relation. -/
abbrev VerifierKey
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) :=
  PiRLCAlgebra.Commitment.Key
    (RelationShape shape publicRingColumns publicFits) verifierRows

/-- Exact public commitment carrier. -/
abbrev CommitmentValue (verifierRows : Nat) :=
  PiRLCAlgebra.Commitment.Value verifierRows

/-- The sole commitment map used by input authority and both tail phases. -/
def commit
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (assignment : SourceAssignment shape) :
    CommitmentValue verifierRows :=
  PiRLCAlgebra.Commitment.commit key assignment

/-- Concrete Phi81 relation semantics at the exact verifier key. -/
def semantics
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : VerifierKey shape publicRingColumns publicFits verifierRows) :=
  productSemantics publicRingColumns publicFits (commit key)

/-- Complete independently proved concrete `Pi_RLC` algebra. -/
def rlcAlgebra
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : VerifierKey shape publicRingColumns publicFits verifierRows) :=
  PiRLCAlgebra.Algebra.concrete key

/-- Complete independently proved concrete `Pi_DEC` algebra. -/
def decAlgebra
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : VerifierKey shape publicRingColumns publicFits verifierRows) :=
  PiDECAlgebra.Algebra.concrete key

/-- Complete public payload bound by the `Pi_CCS` transcript. The source
product and polynomial verifier input are distinct protocol surfaces and
neither may be omitted from statement binding. -/
structure StatementInput
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat)
    (arity : BatchArity productionGlobalParams) where
  sources :
    SourceProduct shape publicRingColumns publicFits
      (CommitmentValue verifierRows) productionGlobalParams arity
  runningParent :
    Option
      (Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows))
  polynomial : PiCCS.SplitNc.Verifier.PublicInput shape

/-- Verifier-owned context for one concrete transition. Every field either
fixes public setup or deterministically interprets the raw certificate. -/
structure Context
    (shape : SemanticShape)
    (State : Type uState)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat)
    (arity : BatchArity productionGlobalParams) where
  covers : PiCcsDomains.production.nc.Covers shape
  key : VerifierKey shape publicRingColumns publicFits verifierRows
  alignment : SourceAlignment shape productionGlobalParams arity
  input :
    SourceProduct shape publicRingColumns publicFits
      (CommitmentValue verifierRows) productionGlobalParams arity
  runningParent :
    Option
      (Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows))
  piCcsInput : PiCCS.SplitNc.Verifier.PublicInput shape
  priorState : State
  piCcsSchedule :
    PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane.Schedule
      (VerifierKey shape publicRingColumns publicFits verifierRows)
      (StatementInput shape publicRingColumns publicFits verifierRows arity)
      shape PiCcsDomains.production State
  piRlcMachine :
    Nifs.NonInteractive.PiRlcSampler.ProductionSchedule.Machine State
  profile : PiCCS.SplitNc.Verifier.Polynomial.Fe.SupportedProfile shape
    PiCcsDomains.production.fe
  challengeSetSize : Nat

namespace Context

/-- Exact public statement bound before any `Pi_CCS` challenge is derived. -/
def piCcsStatement
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity) :
    PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane.Statement
      (VerifierKey shape publicRingColumns publicFits verifierRows)
      (StatementInput shape publicRingColumns publicFits verifierRows
        arity) where
  verifierKey := context.key
  input := {
    sources := context.input
    runningParent := context.runningParent
    polynomial := context.piCcsInput
  }

/-- Statement binding includes the complete public source product exactly. -/
@[simp] theorem piCcsStatement_sources
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity) :
    context.piCcsStatement.input.sources = context.input := by
  rfl

/-- Statement binding includes the complete polynomial verifier input
exactly. -/
@[simp] theorem piCcsStatement_polynomial
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity) :
    context.piCcsStatement.input.polynomial = context.piCcsInput := by
  rfl

/-- Statement binding includes the complete incoming checked-parent carrier
exactly. The separate running-authority verifier decides whether it is valid;
the transcript never receives only an unverified digest. -/
@[simp] theorem piCcsStatement_runningParent
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity) :
    context.piCcsStatement.input.runningParent = context.runningParent := by
  rfl

/-- The canonical public relation structure is the first fresh source's
structure. Production arity proves that source exists, and `Pi_RLC` equations
later check every materialized source against this same value. -/
def system
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity) :
    Phi81Relation.Structure
      (RelationShape shape publicRingColumns publicFits) :=
  (context.input.fresh ⟨0, arity.freshPositive⟩).constraintSystem

/-- The combined parent system is definitionally the first public fresh
source's system; there is no hidden semantic-data lookup. -/
@[simp] theorem system_eq_firstFresh
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity) :
    context.system =
      (context.input.fresh ⟨0, arity.freshPositive⟩).constraintSystem := by
  rfl

/-- One statement-bound pre-SumCheck execution owns every FE/NC coin and the
sole state entering FE. -/
def piCcsPreSumcheck
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity) :
    PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane.PreSumcheck
      shape PiCcsDomains.production State :=
  PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane.derivePreSumcheck
    context.piCcsSchedule context.priorState context.piCcsStatement

/-- FE coins are a projection of the unique pre-SumCheck challenge record. -/
def feCoins
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity) :
    PiCCS.SplitNc.Verifier.Polynomial.Fe.Coins shape
      PiCcsDomains.production.fe :=
  context.piCcsPreSumcheck.challenges.feCoins

/-- NC coins are a projection of the same challenge record. -/
def ncCoins
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity) :
    PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Mixing.Coins
      PiCcsDomains.production.nc :=
  context.piCcsPreSumcheck.challenges.ncCoins

/-- The FE initial claim is verifier-computed from the bound public input and
the schedule-derived FE coins. -/
def feInitial
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity) : K :=
  PiCCS.SplitNc.Verifier.Polynomial.Fe.initial context.profile
    context.piCcsInput context.feCoins

/-- FE transcript machine parameterized only by the computed initial claim. -/
def feMachine
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity) :
    PiCCS.SplitNc.Verifier.Transcript.Fe.Machine State :=
  PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane.feMachine
    context.piCcsSchedule context.feInitial

/-- NC transcript machine projected from the same schedule. -/
def ncMachine
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity) :
    PiCCS.SplitNc.Verifier.Transcript.Nc.Machine State :=
  PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane.ncMachine
    context.piCcsSchedule

/-- Sole state entering FE after statement binding and pre-SumCheck sampling. -/
def initialState
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity) : State :=
  context.piCcsPreSumcheck.state

/-- The shared challenge carrier makes FE/NC lane authority definitionally
identical. -/
@[simp] theorem ncCoins_betaA_eq_feCoins_betaA
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity) :
    context.ncCoins.betaA = context.feCoins.betaA := by
  exact
    PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane.Challenges.ncCoins_betaA_eq_feCoins_betaA
      context.piCcsPreSumcheck.challenges

/-- The shared challenge carrier makes FE/NC mixing authority definitionally
identical. -/
@[simp] theorem ncCoins_gamma_eq_feCoins_gamma
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity) :
    context.ncCoins.gamma = context.feCoins.gamma := by
  exact
    PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane.Challenges.ncCoins_gamma_eq_feCoins_gamma
      context.piCcsPreSumcheck.challenges

end Context

/-- The irreducible prover-supplied public payload of one `Pi_DEC` child.
Relation structure, evaluation point, and fresh stage are not duplicated here:
the verifier inherits them from the one computed parent. -/
structure PiDecChildPayload
    (shape : Phi81Relation.Shape)
    (Commitment : Type uCommitment) where
  commitment : Commitment
  publicInput : Phi81Relation.PublicInput shape
  evaluations : Array Phi81Relation.Evaluation

namespace PiDecChildPayload

/-- Construct the unique child statement determined by a computed parent and
one prover-supplied payload. -/
def materialize
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    (parent : Phi81Relation.CEStatement shape Commitment)
    (payload : PiDecChildPayload shape Commitment) :
    Phi81Relation.CEStatement shape Commitment := {
  constraintSystem := parent.constraintSystem
  commitment := payload.commitment
  publicInput := payload.publicInput
  point := parent.point
  evaluations := payload.evaluations
  stage := .fresh
}

/-- Forget exactly the three fields that canonical child materialization
inherits from its parent. -/
def ofStatement
    {shape : Phi81Relation.Shape}
    {Commitment : Type uCommitment}
    (statement : Phi81Relation.CEStatement shape Commitment) :
    PiDecChildPayload shape Commitment := {
  commitment := statement.commitment
  publicInput := statement.publicInput
  evaluations := statement.evaluations
}

end PiDecChildPayload

/-- Raw verifier-visible certificate for the complete three-phase transition.
The two internal derived carriers are intentionally absent. -/
structure Certificate
    {shape : SemanticShape}
    {arity : BatchArity productionGlobalParams}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat)
    (piCcsInput : PiCCS.SplitNc.Verifier.PublicInput shape) where
  piCcs :
    PiCCS.SplitNc.Verifier.Protocol.BlockLane.Certificate
      piCcsInput PiCcsDomains.production
  piRlcChallenges : Fin arity.total -> RingF
  piDecPayloads : Fin productionGlobalParams.k ->
    PiDecChildPayload
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows)

/-- Deterministically derived phase carriers. `piCcsOutputs` is the sole
`Pi_RLC` input vector and `piRlcOutput` is the sole `Pi_DEC` parent. -/
structure Execution
    (shape : SemanticShape)
    (State : Type uState)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat)
    (arity : BatchArity productionGlobalParams) where
  piCcs :
    PiCCS.SplitNc.Verifier.Protocol.BlockLane.Execution shape
      PiCcsDomains.production State
  piCcsOutputs :
    Product shape publicRingColumns publicFits
      (CommitmentValue verifierRows) productionGlobalParams arity
  piRlcOutput :
    Phi81Relation.CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows)

namespace Execution

/-- Sole Π_RLC sampler root. Canonical BlockLane replay has already absorbed
the complete Π_CCS output exactly once. -/
def piRlcInitialState
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (execution :
      Execution shape State publicRingColumns publicFits verifierRows arity) :
    State :=
  execution.piCcs.finalState

/-- The exact `Pi_RLC` view over the one derived `Pi_CCS` product. -/
def piRlcAttempt
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    {piCcsInput : PiCCS.SplitNc.Verifier.PublicInput shape}
    (execution :
      Execution shape State publicRingColumns publicFits verifierRows
        arity)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows piCcsInput) :
    PiRLC.Attempt
      (Phi81Relation.Structure
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.PublicInput
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.Point
        (RelationShape shape publicRingColumns publicFits))
      Phi81Relation.Evaluation (CommitmentValue verifierRows) RingF
      productionGlobalParams arity := {
  inputs := execution.piCcsOutputs
  challenges := certificate.piRlcChallenges
  output := execution.piRlcOutput
}

/-- The canonical child family. All inherited fields are constructed from the
computed parent rather than checked against prover-supplied duplicates. -/
def piDecChildren
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    {piCcsInput : PiCCS.SplitNc.Verifier.PublicInput shape}
    (execution :
      Execution shape State publicRingColumns publicFits verifierRows
        arity)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows piCcsInput) :
    Fin productionGlobalParams.k ->
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows) :=
  fun child =>
    PiDecChildPayload.materialize execution.piRlcOutput
      (certificate.piDecPayloads child)

/-- The exact `Pi_DEC` view over the one derived `Pi_RLC` parent. -/
def piDecAttempt
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    {piCcsInput : PiCCS.SplitNc.Verifier.PublicInput shape}
    (execution :
      Execution shape State publicRingColumns publicFits verifierRows
        arity)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows piCcsInput) :
    PiDEC.Attempt
      (Phi81Relation.Structure
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.PublicInput
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.Point
        (RelationShape shape publicRingColumns publicFits))
      Phi81Relation.Evaluation (CommitmentValue verifierRows)
      productionGlobalParams := {
  parent := execution.piRlcOutput
  children := execution.piDecChildren certificate
}

end Execution

/-- Deterministic three-phase dataflow. -/
def derive
    {shape : SemanticShape}
    {State : Type uState}
    {arity : BatchArity productionGlobalParams}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {verifierRows : Nat}
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) :
    Execution shape State publicRingColumns publicFits verifierRows
      arity :=
  let piCcsExecution :=
    PiCCS.SplitNc.Verifier.Protocol.BlockLane.derive
      StatementInput.polynomial context.piCcsSchedule context.priorState
      context.profile context.piCcsStatement certificate.piCcs
  let piCcsOutputs :=
    OutputProduct.materialize publicRingColumns publicFits context.alignment
      context.input piCcsExecution.fePoint.row
      certificate.piCcs.output
  let piRlcOutput :=
    PiRLC.combinedOutput (rlcAlgebra context.key)
      context.system
      piCcsExecution.fePoint.row piCcsOutputs
      certificate.piRlcChallenges
  {
    piCcs := piCcsExecution
    piCcsOutputs := piCcsOutputs
    piRlcOutput := piRlcOutput
  }

/-- Canonical public accumulator output of the complete three-phase dataflow.
This is the only child-statement surface exported by a raw certificate. -/
def outputChildren
    {shape : SemanticShape}
    {State : Type uState}
    {arity : BatchArity productionGlobalParams}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {verifierRows : Nat}
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) :
    Fin productionGlobalParams.k ->
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows) :=
  (derive context certificate).piDecChildren certificate

/-- The Π_RLC sampler starts at the canonical post-output Π_CCS state. -/
@[simp] theorem derive_piRlcInitialState
    {shape : SemanticShape}
    {State : Type uState}
    {arity : BatchArity productionGlobalParams}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {verifierRows : Nat}
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) :
    (derive context certificate).piRlcInitialState =
      (derive context certificate).piCcs.finalState := by
  rfl

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
