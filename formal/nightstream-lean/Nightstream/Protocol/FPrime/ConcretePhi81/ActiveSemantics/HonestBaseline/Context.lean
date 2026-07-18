import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.RunningAuthority
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Sources
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestNifs
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.BaseLinear
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm.Centered
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains

/-!
Degenerate model-level fixed-active context for the independent
270-coordinate source baseline.

Owns: the concrete verifier-owned context values needed to place
`HonestBaseline.Sources` inside the fixed-active NIFS carrier; exact public
source binding; checked incoming-parent authority; and the resulting honest
semantic-premise package.

Does not own: Rust or artifact conformance, Poseidon2, transcript security,
sampler probability, R1CS lowering, costs, minimality, or row removal.

Emits constraints: no.

Authority boundary: this is deliberately a model fixture. Its zero-row Ajtai
key, `Unit` transcript, and constant-zero sampler are explicit typed values,
not conformance evidence. The fresh statement is computed from `Sources.data`;
the combined parent and all fourteen children are computed from one complete
zero opening; and checked running authority follows from `PiDEC.complete`,
never from a digest or copied acceptance bit.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.active.honest_baseline.context.profile` | exact 270-coordinate shape, five public rings, canonical production dimensions, and supported FE profile | model setup | `covers`, `profile` |
| `fprime.active.honest_baseline.context.alignment` | one fresh and fourteen running sources preserve their partition | computed | `alignment` |
| `fprime.active.honest_baseline.context.key` | canonical empty verifier-row Ajtai key | model setup | `key` |
| `fprime.active.honest_baseline.context.transcript` | deterministic typed `Unit` schedule | model setup, not Poseidon2 refinement | `piCcsSchedule` |
| `fprime.active.honest_baseline.context.sampler.chunk` | candidate value two is accepted and decodes to the centered-zero coefficient; literal chunk zero would decode to minus two | computed | `centeredZeroChunk_accepted`, `centeredZeroChunk_symbol` |
| `fprime.active.honest_baseline.context.sampler.execution` | the constant centered-zero stream constructs the canonical bounded first-accepted execution at every coordinate | computed | `exists_centeredZeroExecution` |
| `fprime.active.honest_baseline.context.sampler.batch` | the coordinate executions form one exact transcript-threaded batch whose assembled RingF challenges are all zero | computed | `centeredZeroBatch`, `centeredZeroBatch_challenge` |
| `fprime.active.honest_baseline.context.sampler.bound` | the all-zero RingF vector has an actual production-sized bounded sampler witness from every `Unit` state | computed | `samplerBound` |
| `fprime.active.honest_baseline.context.radix_zero` | every canonical PiDEC digit of the complete zero opening is zero | derived | `splitZero` |
| `fprime.active.honest_baseline.context.sources` | physical fresh/children statements bind exactly to `Sources.data` | derived | `sourceBound`, `semanticInput` |
| `fprime.active.honest_baseline.context.parent_opening` | the context-owned zero parent has its context-owned zero opening | derived | `parentHolds` |
| `fprime.active.honest_baseline.context.running` | the installed combined parent strictly recomposes from the installed children | derived | `runningAccepted` |
| `fprime.active.honest_baseline.context.premises` | independent paper, source, incoming-parent authority, and the explicit sampler batch form honest NIFS premises | derived | `semanticPremises`, `honestPremises` |
| `fprime.active.honest_baseline.context.transition` | the degenerate model fixture has one physically accepted certificate and independent fixed-active result transition | derived | `exists_resultTransition` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Context

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

/-! ## Exact model setup -/

def publicRingColumns : Nat := 5

def verifierRows : Nat := 0

def slotCount : Nat := 1

theorem publicFits :
    ringDegree * publicRingColumns <= Sources.shape.carrierWidth := by
  decide

theorem covers : PiCcsDomains.production.nc.Covers Sources.shape := by
  simpa [PiCcsDomains.production_nc, Sources.shape, Sources.dimensions,
    PiCcsSources.semanticShape, PiCcsDomain.plainShape] using
      (PiCcsDomain.blockDomain_covers 1 1 14 3)

theorem profile :
    Polynomial.Fe.SupportedProfile Sources.shape PiCcsDomains.production.fe where
  row_nonempty := by decide
  fresh_nonempty := by decide
  lane_variables := rfl

def alignment :
    SourceAlignment Sources.shape productionGlobalParams FixedActive.arity where
  freshCount_eq := rfl
  runningCount_eq := rfl

def selected : Fin slotCount := ⟨0, by decide⟩

/-- There are no verifier rows in this model fixture, so the canonical key has
no observable row value. This is not key-conformance evidence. -/
def key :
    ConcretePhi81.VerifierKey Sources.shape publicRingColumns publicFits
      verifierRows :=
  fun row => Fin.elim0 row

def system :
    RelationStructure Sources.shape publicRingColumns publicFits :=
  Phi81Relation.Structure.ofSourceData publicRingColumns publicFits Sources.data

def point : RelationPoint Sources.shape publicRingColumns publicFits :=
  Sources.priorPoint

def zeroAssignment :
    Phi81Relation.Assignment
      (RelationShape Sources.shape publicRingColumns publicFits) :=
  fun _ => 0

/-! ## Canonical fresh and running statements -/

def freshStatement :
    Phi81Relation.CCSStatement
      (RelationShape Sources.shape publicRingColumns publicFits)
      (CommitmentValue verifierRows) :=
  Phi81Relation.canonicalCCSStatement (ConcretePhi81.commit key) system .fresh
    (Sources.data.freshAssignment selected)

def parent :
    Phi81Relation.CEStatement
      (RelationShape Sources.shape publicRingColumns publicFits)
      (CommitmentValue verifierRows) :=
  Phi81Relation.canonicalCEStatement (ConcretePhi81.commit key) system
    .combined point zeroAssignment

def children : Fin productionGlobalParams.k ->
    Phi81Relation.CEStatement
      (RelationShape Sources.shape publicRingColumns publicFits)
      (CommitmentValue verifierRows) :=
  PiDEC.childrenOf (ConcretePhi81.decAlgebra key) parent zeroAssignment

def runningSlot :
    Slot Sources.shape publicRingColumns publicFits verifierRows where
  parent := parent
  children := children

/-! ## Deterministic model transcript and sampler -/

def zeroPoint (dimension : Nat) : CubePoint K dimension where
  coordinates := List.replicate dimension K.zero
  dimension := by simp

/-- Typed deterministic schedule used only to inhabit the independent model
context. It makes no cryptographic transcript claim. -/
def piCcsSchedule :
    Protocol.TranscriptAuthority.BlockLane.Schedule
      (ConcretePhi81.VerifierKey
        Sources.shape publicRingColumns publicFits verifierRows)
      (ConcretePhi81.StatementInput
        Sources.shape publicRingColumns publicFits verifierRows
        FixedActive.arity)
      Sources.shape PiCcsDomains.production Unit where
  bindStatement := fun _ _ => ()
  derivePreSumcheck := fun _ => {
    challenges := {
      alpha := zeroPoint PiCcsDomains.production.laneVariables
      betaA := zeroPoint PiCcsDomains.production.laneVariables
      betaR := zeroPoint Sources.shape.rowVariables
      gamma := K.zero
      betaBlock := zeroPoint PiCcsDomains.production.blockVariables
    }
    state := ()
  }
  enterFe := fun _ _ => ()
  absorbFeRound := fun _ _ => ()
  squeezeFeChallenge := fun _ => (K.zero, ())
  enterNc := fun _ => ()
  absorbNcRound := fun _ _ => ()
  squeezeNcChallenge := fun _ => (K.zero, ())
  absorbOutput := fun _ _ => ()

/-! Candidate value two, rather than literal candidate zero, decodes to the
centered coefficient zero. Literal candidate zero decodes to centered minus
two and therefore would not assemble to `ringFZero`. -/
def centeredZeroChunk :
    Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.Chunk :=
  ⟨2, by decide⟩

theorem centeredZeroChunk_accepted :
    Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.verifier.accepts
        centeredZeroChunk = true := by
  decide

def centeredZeroCoefficient :
    Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.Coefficient :=
  ⟨2, by decide⟩

@[simp] theorem centeredZeroChunk_symbol :
    Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.verifier.symbol
        centeredZeroChunk = centeredZeroCoefficient := by
  rfl

def centeredZeroScalar :
    Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet.Scalar :=
  fun _ => centeredZeroCoefficient

theorem centeredZeroScalar_embed_eq_zero :
    Phi81StrongSet.embedScalar centeredZeroScalar = ringFZero := by
  funext position
  rfl

def piRlcMachine :
    Nifs.NonInteractive.PiRlcSampler.ProductionSchedule.Machine Unit where
  enterScalar := fun _ _ => ()
  digestBlock := fun _ _ => ((), fun _ => centeredZeroChunk)

/-- Every candidate emitted by the model digest-block machine is accepted and
decodes to the centered-zero coefficient. This is an alphabet fact only, not
a Poseidon2 or probability theorem. -/
theorem piRlcMachine_digestChunk_accepted
    (state : Unit) (seed : Nat)
    (lane : Fin
      Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.chunksPerDigest) :
    Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.verifier.accepts
        ((piRlcMachine.digestBlock state seed).2 lane) = true := by
  exact centeredZeroChunk_accepted

/-! ## Exact bounded sampler construction -/

def zeroChallenges : Fin FixedActive.arity.total -> RingF :=
  fun _ => ringFZero

@[simp] theorem piRlcCandidate_eq_centeredZero
    (initial : Unit) (coordinate position : Nat) :
    (Nifs.NonInteractive.PiRlcSampler.sourceAt
      (ConcretePhi81.Sampler.Specification piRlcMachine)
      initial coordinate).stream position = centeredZeroChunk := by
  rfl

theorem piRlcCandidatePrefix_eq_replicate
    (initial : Unit) (coordinate count : Nat) :
    Nightstream.SuperNeo.Sampling.FirstAccepted.streamPrefix
        (Nifs.NonInteractive.PiRlcSampler.sourceAt
          (ConcretePhi81.Sampler.Specification piRlcMachine)
          initial coordinate).stream count =
      List.replicate count centeredZeroChunk := by
  unfold Nightstream.SuperNeo.Sampling.FirstAccepted.streamPrefix
  have streamEq :
      (Nifs.NonInteractive.PiRlcSampler.sourceAt
        (ConcretePhi81.Sampler.Specification piRlcMachine)
        initial coordinate).stream =
          (fun _ : Nat => centeredZeroChunk) := by
    funext position
    exact piRlcCandidate_eq_centeredZero initial coordinate position
  rw [streamEq]
  rw [List.map_const', List.length_range]

theorem piRlcBoundedSample_eq_centeredZero
    (initial : Unit) (coordinate : Nat) :
    Nightstream.SuperNeo.Sampling.FirstAccepted.boundedSample
        Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.verifier
        Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.coefficientCount
        (Nightstream.SuperNeo.Sampling.FirstAccepted.streamPrefix
          (Nifs.NonInteractive.PiRlcSampler.sourceAt
            (ConcretePhi81.Sampler.Specification piRlcMachine)
            initial coordinate).stream
          Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.candidateBound) =
      some (List.replicate
        Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.coefficientCount
        centeredZeroCoefficient) := by
  rw [piRlcCandidatePrefix_eq_replicate]
  decide

theorem exists_centeredZeroExecution
    (initial : Unit) (coordinate : Nat) :
    exists execution :
        Nifs.NonInteractive.PiRlcSampler.CoefficientExecution
          (ConcretePhi81.Sampler.Specification piRlcMachine)
          Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.candidateBound
          initial coordinate,
      execution.output =
        List.replicate
          Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.coefficientCount
          centeredZeroCoefficient := by
  exact
    Nightstream.SuperNeo.Sampling.FirstAccepted.BoundedExecution.exists_of_bounded_success
      (piRlcBoundedSample_eq_centeredZero initial coordinate)

noncomputable def centeredZeroBatch (initial : Unit) :
    Nifs.NonInteractive.PiRlcSampler.BatchExecution
      (ConcretePhi81.Sampler.Specification piRlcMachine)
      FixedActive.arity.total
      Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.candidateBound
      initial where
  execution coordinate :=
    Classical.choose (exists_centeredZeroExecution initial coordinate.val)

theorem centeredZeroBatch_output
    (initial : Unit) (coordinate : Fin FixedActive.arity.total) :
    ((centeredZeroBatch initial).execution coordinate).output =
      List.replicate
        Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.coefficientCount
        centeredZeroCoefficient := by
  exact Classical.choose_spec
    (exists_centeredZeroExecution initial coordinate.val)

theorem centeredZeroBatch_coefficient
    (initial : Unit) (coordinate : Fin FixedActive.arity.total)
    (position : Fin
      Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.coefficientCount) :
    Nifs.NonInteractive.PiRlcSampler.coefficient
        ((centeredZeroBatch initial).execution coordinate) position =
      centeredZeroCoefficient := by
  unfold Nifs.NonInteractive.PiRlcSampler.coefficient
  rw [List.get_of_eq (centeredZeroBatch_output initial coordinate)]
  simpa only [List.get_eq_getElem] using
    (List.getElem_replicate
      (a := centeredZeroCoefficient)
      (n :=
        Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.coefficientCount)
      (i := position.val) (by simpa using position.isLt))

theorem centeredZeroBatch_challenge
    (initial : Unit) (coordinate : Fin FixedActive.arity.total) :
    Nifs.NonInteractive.PiRlcSampler.challenge
        (centeredZeroBatch initial) coordinate = ringFZero := by
  change
    Phi81StrongSet.embedScalar
        (fun position =>
          Nifs.NonInteractive.PiRlcSampler.coefficient
            ((centeredZeroBatch initial).execution coordinate) position) =
      ringFZero
  calc
    Phi81StrongSet.embedScalar
        (fun position =>
          Nifs.NonInteractive.PiRlcSampler.coefficient
            ((centeredZeroBatch initial).execution coordinate) position) =
        Phi81StrongSet.embedScalar centeredZeroScalar := by
      apply congrArg Phi81StrongSet.embedScalar
      funext position
      exact centeredZeroBatch_coefficient initial coordinate position
    _ = ringFZero := centeredZeroScalar_embed_eq_zero

/-- Exact bounded sampler witness for the canonical all-zero RingF vector.
The proof constructs each first-accepted execution from bounded success and
then forms the batch; it does not assume abstract sampler availability. -/
noncomputable def samplerBound (initial : Unit) :
    ConcretePhi81.Sampler.Bound piRlcMachine initial zeroChallenges where
  batch := centeredZeroBatch initial
  challenges_eq coordinate := by
    exact (centeredZeroBatch_challenge initial coordinate).symm

def template :
    Nightstream.Protocol.FPrime.ConcretePhi81.Context.Template
      Sources.shape Unit publicRingColumns publicFits verifierRows where
  covers := covers
  key := key
  alignment := alignment
  piCcsSchedule := piCcsSchedule
  piRlcMachine := piRlcMachine
  profile := profile
  challengeSetSize := 1

def setup :
    Setup Unit Unit Unit Unit Sources.shape publicRingColumns publicFits
      verifierRows slotCount where
  template := fun _ _ => template
  expectedStructure := fun _ _ => system
  piCcsInput := fun _ _ => PublicInput.ofSources Sources.data
  priorTranscriptState := fun _ _ => ()

def input :
    Input Unit Unit Unit Sources.shape publicRingColumns publicFits verifierRows
      slotCount where
  verifierKey := ()
  iteration := 1
  z0 := ()
  zi := ()
  running := fun _ => runningSlot
  fresh := freshStatement
  priorPc := 1
  witness := ()

/-- Sole fixed-active context selected by the one-slot model setup. -/
abbrev context := contextAt setup input selected

@[simp] theorem context_input_fresh
    (source : Fin FixedActive.arity.freshCount) :
    context.input.fresh source = freshStatement := by
  rfl

@[simp] theorem context_input_running
    (source : Fin (FixedActive.arity.mode.count productionGlobalParams)) :
    context.input.running source = children source := by
  rfl

/-! ## Exact source and running authority -/

/-- The concrete signed-radix decomposition of the complete zero assignment
is pointwise zero in every one of the fourteen child positions. -/
@[simp] theorem splitZero (child : Fin productionGlobalParams.k) :
    (ConcretePhi81.decAlgebra key).splitAssignment zeroAssignment child =
      zeroAssignment := by
  change
    Phi81Relation.PiDECAlgebra.Radix.splitAssignment zeroAssignment child =
      zeroAssignment
  funext column
  change
    Phi81Relation.PiDECAlgebra.Radix.splitScalar 0 child = 0
  simp [Phi81Relation.PiDECAlgebra.Radix.splitScalar,
    Phi81Relation.PiDECAlgebra.Radix.combinedBound,
    Phi81Relation.PiDECAlgebra.Radix.boundedDigit,
    Phi81Relation.PiDECAlgebra.Radix.isNonnegative,
    Phi81Relation.PiDECAlgebra.Radix.magnitudeDigit,
    Phi81Relation.PiDECAlgebra.Radix.natBit,
    Phi81Relation.PiDECAlgebra.Radix.fieldOfNat,
    Phi81Relation.PiRLCAlgebra.Norm.Centered.centeredMagnitude_zero,
    productionGlobalParams, GlobalParams.bigB]

@[simp] theorem sourceRunningAssignment_zero
    (source : Fin Sources.shape.runningCount) :
    Sources.data.runningAssignments source = zeroAssignment := by
  funext column
  rfl

theorem sourceBound :
    InputAuthority.BoundToSources publicRingColumns publicFits
      (ConcretePhi81.commit key) Sources.data alignment context.input := by
  refine { fresh := ?_, running := ?_ }
  · intro source
    refine {
      constraintSystem := ?_
      commitment := ?_
      publicInput := ?_
      stage := ?_
    }
    · rw [context_input_fresh]
      rfl
    · rw [context_input_fresh]
      rfl
    · rw [context_input_fresh]
      rfl
    · rw [context_input_fresh]
      rfl
  · intro source
    refine {
      constraintSystem := ?_
      commitment := ?_
      publicInput := ?_
      point := ?_
      evaluations := ?_
      stage := ?_
    }
    · rw [context_input_running]
      rfl
    · rw [context_input_running, sourceRunningAssignment_zero]
      change
        ConcretePhi81.commit key zeroAssignment =
          (children source).commitment
      unfold children PiDEC.childrenOf
      rw [splitZero]
      rfl
    · rw [context_input_running, sourceRunningAssignment_zero]
      change
        Phi81Relation.projectPublicInput zeroAssignment =
          (children source).publicInput
      unfold children PiDEC.childrenOf
      rw [splitZero]
      rfl
    · rw [context_input_running]
      rfl
    · rw [context_input_running]
      unfold children PiDEC.childrenOf
      rw [splitZero]
      have carried :=
        InputAuthority.relationEvaluations_eq_priorEvaluations_of_carriedTruth
          publicRingColumns publicFits Sources.data source
          Sources.carriedEvaluationsHold
      rw [sourceRunningAssignment_zero] at carried
      simpa only [parent, Phi81Relation.canonicalCEStatement, system, point]
        using carried
    · rw [context_input_running]
      rfl

theorem semanticInput : ConcretePhi81.SemanticInput context Sources.data where
  publicInput := rfl
  sources := sourceBound

/-- The context-owned parent opens at the context-owned complete-zero
assignment. This is a fact about the degenerate model fixture only. -/
theorem parentHolds :
    CE.Holds (ConcretePhi81.semantics key) productionGlobalParams parent
      zeroAssignment := by
  apply Phi81Relation.canonicalCE_holds
  intro column
  simp [zeroAssignment, productionGlobalParams, NormStage.bound,
    GlobalParams.bigB,
    Phi81Relation.PiRLCAlgebra.Norm.Centered.centeredMagnitude_zero]

theorem runningAccepted :
    ConcretePhi81.RunningAuthority.Accepted context := by
  apply
    Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.RunningAuthority.accepted_of_combinedOpening
      context zeroAssignment parent
  · rfl
  · exact {
      parentCombined := rfl
      parentValid := parentHolds
      childrenEq := rfl
    }

/-- Closed honest semantic authority for the degenerate 270-coordinate model
context. Sampler success remains an explicit later outcome of honest
completeness. -/
def semanticPremises :
    HonestNifs.SemanticPremises setup input selected where
  data := Sources.data
  paper := Sources.paperHolds
  semanticInput := semanticInput
  running := runningAccepted

/-- Closed honest NIFS premises for the degenerate model fixture. Sampler
success is established by the explicit centered-zero batch above, not by a
totality or probability assumption. -/
noncomputable def honestPremises :
    HonestNifs.Premises setup input selected where
  data := Sources.data
  paper := Sources.paperHolds
  semanticInput := semanticInput
  running := runningAccepted
  challenges := zeroChallenges
  samplerAvailable := by
    intro piCcsCertificate piCcsAccepted
    exact samplerBound
      (Protocol.BlockLane.derive StatementInput.polynomial
        context.piCcsSchedule context.priorState context.profile
        context.piCcsStatement piCcsCertificate).finalState

/-- The degenerate model fixture has an actual accepted fixed-active NIFS
certificate whose canonical result satisfies the independent semantic
transition. This theorem is model-level only; it makes no Poseidon2, Rust,
artifact, or R1CS conformance claim. -/
theorem exists_resultTransition :
    exists certificate : FixedActive.Certificate context,
      ConcretePhi81.Accepted context certificate /\
        FixedActive.ResultTransition context
          (FixedActive.resultOf context certificate) := by
  exact HonestNifs.Premises.exists_resultTransition setup input selected
    honestPremises

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Context
