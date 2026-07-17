import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CarrierEquality
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler

/-!
Executable fixed-bound checker for the concrete Phi81 `Pi_RLC` sampler.

Protocol: SuperNeo NIFS.
Phase: post-`Pi_CCS` state to the complete carried `Pi_RLC` challenge vector.
Constraint family: canonical 54-of-64 sampling and challenge equality; this
file emits no rows.

Owns: direct execution of the verifier-owned finite candidate prefix, exact
assembly of every successful coefficient vector, a finite Boolean batch
check, and its equivalence to the proof-carrying sampler boundary.

Does not own: Poseidon2 implementation, transcript-machine refinement,
probability of shortfall, `Pi_RLC` algebra, Rust/R1CS rows, costs, necessity,
or row removal.

Emits constraints: no.

Authority boundary: `sampleChallenge?` reads candidates only from
`ProductionSchedule.sourceAt` at the verifier-computed initial state. It
returns `none` on a 54-of-64 shortfall and never accepts a caller-provided
coefficient vector. `check` compares the carried challenge with this computed
result at every coordinate.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.concrete.pi_rlc.sampler.prefix` | inspect exactly the first 64 candidates of one transcript-owned source | computed | `candidatePrefix` |
| `nifs.concrete.pi_rlc.sampler.select` | return exactly the first 54 accepted coefficients or fail | computed | `sampleChallenge?` |
| `nifs.concrete.pi_rlc.sampler.assemble` | embed the exact selected vector into one `RingF` challenge | computed | `sampleChallenge?` |
| `nifs.concrete.pi_rlc.sampler.binding` | computed challenge equals the carried challenge at every coordinate | checked | `check` |
| `nifs.concrete.pi_rlc.sampler.exact` | Boolean batch check iff the proof-carrying bound exists | exact model theorem | `check_eq_true_iff_bound` |
| `nifs.concrete.pi_rlc.sampler.certificate` | specialize exact checking to the actual derived post-`Pi_CCS` state | exact model theorem | `certificateCheck_eq_true_iff_accepted` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81StrongSet
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Sampling
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CarrierEquality

universe uState

/-- Exact finite candidate prefix inspected for one scalar coordinate. -/
def candidatePrefix
    {State : Type uState}
    (machine : ProductionSchedule.Machine State)
    (initialState : State)
    (coordinate : Nat) : List Chunk :=
  FirstAccepted.streamPrefix
    (sourceAt (Specification machine) initialState coordinate).stream
    candidateBound

/-- Harmless total fallback used only to make list indexing executable.
Successful sampling proves every one of the 54 reads is in bounds, so this
value is unreachable on an accepted path. -/
def defaultCoefficient : Coefficient := ⟨0, by decide⟩

/-- Interpret the first 54 list positions as the fixed scalar carrier.
The sampler success theorem later proves all accepted-path reads are in
bounds. -/
def scalarOfList (output : List Coefficient) : Scalar :=
  fun position => output.getD position.val defaultCoefficient

/-- Execute the canonical bounded sampler and assemble its output. -/
def sampleChallenge?
    {State : Type uState}
    (machine : ProductionSchedule.Machine State)
    (initialState : State)
    (coordinate : Nat) : Option RingF :=
  (FirstAccepted.boundedSample verifier coefficientCount
      (candidatePrefix machine initialState coordinate)).map fun output =>
    embedScalar (scalarOfList output)

/-- A proof-carrying bounded execution evaluates to exactly the executable
challenge returned by `sampleChallenge?`. -/
theorem sampleChallenge?_eq_some_of_execution
    {State : Type uState}
    {machine : ProductionSchedule.Machine State}
    {initialState : State}
    {coordinate : Nat}
    (execution :
      CoefficientExecution (Specification machine) candidateBound
        initialState coordinate) :
    sampleChallenge? machine initialState coordinate =
      some (embedScalar fun position => coefficient execution position) := by
  have success :
      FirstAccepted.boundedSample verifier coefficientCount
          (candidatePrefix machine initialState coordinate) =
        some execution.output := by
    exact
      FirstAccepted.boundedSample_eq_some_iff_boundedExecution.mpr
        ⟨execution, rfl⟩
  rw [sampleChallenge?, success]
  simp only [Option.map_some, Option.some.injEq]
  congr 1
  funext position
  have within : position.val < execution.output.length := by
    rw [execution.output_length]
    exact position.isLt
  unfold scalarOfList coefficient
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem within]
  rfl

/-- Executable success exposes the unique proof-carrying bounded execution
and the exact assembled challenge. -/
theorem exists_execution_of_sampleChallenge?_eq_some
    {State : Type uState}
    {machine : ProductionSchedule.Machine State}
    {initialState : State}
    {coordinate : Nat}
    {value : RingF}
    (sampled :
      sampleChallenge? machine initialState coordinate = some value) :
    ∃ execution :
        CoefficientExecution (Specification machine) candidateBound
          initialState coordinate,
      value = embedScalar fun position => coefficient execution position := by
  unfold sampleChallenge? at sampled
  cases success :
      FirstAccepted.boundedSample verifier coefficientCount
        (candidatePrefix machine initialState coordinate) with
  | none => simp [success] at sampled
  | some output =>
      rcases
          FirstAccepted.BoundedExecution.exists_of_bounded_success success with
        ⟨execution, _outputEq⟩
      refine ⟨execution, ?_⟩
      apply Option.some.inj
      exact sampled.symm.trans
        (sampleChallenge?_eq_some_of_execution execution)

/-- Fail-closed comparison between an optional computed challenge and one
carried challenge. -/
def challengeMatches (computed : Option RingF) (carried : RingF) : Bool :=
  match computed with
  | none => false
  | some value => ringFEqual value carried

/-- Optional challenge matching succeeds exactly when bounded sampling
returned the carried challenge. -/
theorem challengeMatches_eq_true_iff
    (computed : Option RingF) (carried : RingF) :
    challengeMatches computed carried = true ↔ computed = some carried := by
  cases computed with
  | none => simp [challengeMatches]
  | some value =>
      simp [challengeMatches, ringFEqual_eq_true_iff]

/-- Finite executable equality check for a complete challenge batch. -/
def check
    {State : Type uState}
    {challengeCount : Nat}
    (machine : ProductionSchedule.Machine State)
    (initialState : State)
    (challenges : Fin challengeCount -> RingF) : Bool :=
  (List.finRange challengeCount).all fun coordinate =>
    challengeMatches
      (sampleChallenge? machine initialState coordinate.val)
      (challenges coordinate)

/-- The finite Boolean batch check is exactly the proof-carrying sampler
boundary. The forward theorem constructs proof data from actual finite
execution; the checker itself contains no `Classical.choose`. -/
theorem check_eq_true_iff_bound
    {State : Type uState}
    {challengeCount : Nat}
    (machine : ProductionSchedule.Machine State)
    (initialState : State)
    (challenges : Fin challengeCount -> RingF) :
    check machine initialState challenges = true ↔
      Nonempty (Bound machine initialState challenges) := by
  constructor
  · intro checked
    have sampled : ∀ coordinate : Fin challengeCount,
        sampleChallenge? machine initialState coordinate.val =
          some (challenges coordinate) := by
      intro coordinate
      exact (challengeMatches_eq_true_iff _ _).mp
        ((List.all_eq_true.mp checked) coordinate (by simp))
    classical
    have executionExists : ∀ coordinate : Fin challengeCount,
        ∃ execution :
            CoefficientExecution (Specification machine) candidateBound
              initialState coordinate.val,
          challenges coordinate =
            embedScalar fun position => coefficient execution position := by
      intro coordinate
      exact exists_execution_of_sampleChallenge?_eq_some (sampled coordinate)
    let executions := fun coordinate =>
      Classical.choose (executionExists coordinate)
    have challengeEqual : ∀ coordinate,
        challenges coordinate =
          embedScalar fun position =>
            coefficient (executions coordinate) position := by
      intro coordinate
      exact Classical.choose_spec (executionExists coordinate)
    let batch :
        BatchExecution (Specification machine) challengeCount candidateBound
          initialState := {
      execution := executions
    }
    refine ⟨{
      batch := batch
      challenges_eq := ?_
    }⟩
    intro coordinate
    change challenges coordinate =
      embedScalar fun position => coefficient (executions coordinate) position
    exact challengeEqual coordinate
  · rintro ⟨bound⟩
    apply List.all_eq_true.mpr
    intro coordinate _member
    apply (challengeMatches_eq_true_iff _ _).mpr
    calc
      sampleChallenge? machine initialState coordinate.val =
          some (embedScalar fun position =>
            coefficient (bound.batch.execution coordinate) position) :=
        sampleChallenge?_eq_some_of_execution
          (bound.batch.execution coordinate)
      _ = some (challenges coordinate) := by
        rw [bound.challenges_eq coordinate]
        rfl

/-- Concrete checker specialized to the actual post-`Pi_CCS` state and raw
challenge vector in one NIFS certificate. -/
def certificateCheck
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) : Bool :=
  check context.piRlcMachine
    (derive context certificate).piRlcInitialState
    certificate.piRlcChallenges

/-- The certificate checker exactly decides the existing independent sampler
acceptance predicate. -/
theorem certificateCheck_eq_true_iff_accepted
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) :
    certificateCheck context certificate = true ↔
      CertificateAccepted context certificate := by
  exact
    check_eq_true_iff_bound context.piRlcMachine
      (derive context certificate).piRlcInitialState
      certificate.piRlcChallenges

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker
