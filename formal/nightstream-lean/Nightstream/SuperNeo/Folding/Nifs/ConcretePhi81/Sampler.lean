import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Types
import Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Challenge

/-!
Exact bounded production sampler boundary for the concrete Phi81 NIFS
transition.

Protocol: SuperNeo NIFS.
Phase: post-`Pi_CCS` transcript state → complete `Pi_RLC` challenge vector.
Constraint family: four-block 54-of-64 rejection sampling and RingF assembly;
this file emits no rows.

Owns: specialization of the independent block schedule to exact Phi81 RingF
assembly; binding of every carried transition challenge to one
transcript-chained bounded batch starting at the actual derived `Pi_CCS`
outgoing state; derivation of unary production-set membership; and fail-closed
shortfall exclusion.

Does not own: a concrete Poseidon2 implementation of the abstract block
machine, Rust/R1CS refinement, pairwise low-norm invertibility, extraction,
rows, costs, or row removal.

Emits constraints: no.

Authority boundary: a carried RingF vector is valid only when it is exactly
the output of the bounded batch. The initial state is
`(derive context certificate).piRlcInitialState`; it is the verifier-owned
post-`Pi_CCS` output-handoff state and is never supplied independently. A
digest or coefficient-range claim alone is insufficient.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.concrete.pi_rlc.sampler.machine` | four-block source and successor-state threading | verifier-owned computation | `Specification` |
| `nifs.concrete.pi_rlc.sampler.batch` | one successful bounded execution for every source coordinate | checked witness | `Bound.batch` |
| `nifs.concrete.pi_rlc.sampler.binding` | carried RingF challenge equals exact batch assembly | checked | `Bound.challenges_eq` |
| `nifs.concrete.pi_rlc.challenge.membership` | bounded batch output is in the 54-coordinate production set | derived | `Bound.challengeValid` |
| `nifs.concrete.pi_rlc.sampler.shortfall` | accepted batch excludes fixed-prefix shortfall at every coordinate | derived/fail-closed | `Bound.excludesShortfall` |
| `nifs.concrete.pi_rlc.sampler.outcome` | a finite challenge batch exists or one exact coordinate shortfalls | exhaustive model outcome | `exists_bound_or_exists_shortfall` |
| `nifs.concrete.pi_rlc.sampler.handoff` | batch starts after the actual derived `Pi_CCS` output-message handoff | direct dataflow | `CertificateBound` |
| `nifs.concrete.pi_rlc.sampler.acceptance` | a complete context-bound batch exists | checked proposition | `CertificateAccepted` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81StrongSet
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources

universe uState

/-- Exact production sampler specialized to direct Phi81 RingF assembly. -/
def Specification
    {State : Type uState}
    (machine : ProductionSchedule.Machine State) :=
  ProductionSchedule.specification machine embedScalar

/-- One complete bounded challenge batch and exact equality to the carried
RingF vector. -/
structure Bound
    {State : Type uState}
    {challengeCount : Nat}
    (machine : ProductionSchedule.Machine State)
    (initialState : State)
    (challenges : Fin challengeCount -> RingF) where
  batch :
    BatchExecution (Specification machine) challengeCount candidateBound
      initialState
  challenges_eq :
    ∀ coordinate, challenges coordinate = challenge batch coordinate

namespace Bound

/-- Every carried challenge is an exact unary member of the production
Phi81 sampling set. -/
theorem challengeValid
    {State : Type uState}
    {challengeCount : Nat}
    {machine : ProductionSchedule.Machine State}
    {initialState : State}
    {challenges : Fin challengeCount -> RingF}
    (bound : Bound machine initialState challenges)
    (coordinate : Fin challengeCount) :
    PiRLCAlgebra.Challenge.challengeValid (challenges coordinate) := by
  rw [bound.challenges_eq coordinate]
  change PiRLCAlgebra.Challenge.challengeValid
    (embedScalar (fun position =>
      coefficient (bound.batch.execution coordinate) position))
  exact PiRLCAlgebra.Challenge.embedScalar_valid _

/-- Successful bounded sampling fail-closes every per-coordinate shortfall. -/
theorem excludesShortfall
    {State : Type uState}
    {challengeCount : Nat}
    {machine : ProductionSchedule.Machine State}
    {initialState : State}
    {challenges : Fin challengeCount -> RingF}
    (bound : Bound machine initialState challenges)
    (coordinate : Fin challengeCount) :
    ¬ ShortfallAt (Specification machine) candidateBound initialState
        coordinate.val := by
  apply available_excludes_shortfall
  exact ⟨bound.batch, True.intro⟩

end Bound

/-- At a fixed transcript state, the production-shaped sampler has one of two
exhaustive outcomes: a complete challenge vector bound to a successful batch,
or an exact coordinate whose 64-candidate prefix contains fewer than 54
accepted coefficients. No probability or hash-success assumption is hidden in
this split. -/
theorem exists_bound_or_exists_shortfall
    {State : Type uState}
    (machine : ProductionSchedule.Machine State)
    (challengeCount : Nat)
    (initialState : State) :
    (Exists fun challenges : Fin challengeCount -> RingF =>
      Nonempty (Bound machine initialState challenges)) \/
      Exists fun coordinate : Fin challengeCount =>
        ShortfallAt (Specification machine) candidateBound initialState
          coordinate.val := by
  rcases
      available_or_exists_shortfall (Specification machine) challengeCount
        candidateBound initialState with available | shortfall
  · rcases available with ⟨batch, _⟩
    let challenges : Fin challengeCount -> RingF := fun coordinate =>
      challenge batch coordinate
    apply Or.inl
    refine ⟨challenges, ⟨{ batch := batch, challenges_eq := ?_ }⟩⟩
    intro coordinate
    rfl
  · exact Or.inr shortfall

/-- Bind a carried transition challenge vector to the exact bounded sampler
starting from the actual derived `Pi_CCS` outgoing state. -/
abbrev CertificateBound
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
        publicRingColumns publicFits verifierRows context.piCcsInput) :=
  Bound context.piRlcMachine (derive context certificate).piRlcInitialState
    certificate.piRlcChallenges

/-- Propositional acceptance boundary consumed by the verifier. The concrete
batch remains typed data, while the verifier only asserts that such a batch
exists. -/
def CertificateAccepted
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
        publicRingColumns publicFits verifierRows context.piCcsInput) : Prop :=
  Nonempty (CertificateBound context certificate)

/-- The context-bound batch derives exactly the challenge-validity field used
by the concrete `Pi_RLC` algebra. -/
theorem certificateBound_challengesValid
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    {certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (bound : CertificateBound context certificate) :
    ∀ coordinate,
      (rlcAlgebra context.key).challengeValid
        (certificate.piRlcChallenges coordinate) := by
  intro coordinate
  exact bound.challengeValid coordinate

/-- Propositional sampler acceptance derives the complete challenge-validity
family without a second verifier check. -/
theorem certificateAccepted_challengesValid
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {arity : BatchArity productionGlobalParams}
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    {certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (accepted : CertificateAccepted context certificate) :
    ∀ coordinate,
      (rlcAlgebra context.key).challengeValid
        (certificate.piRlcChallenges coordinate) := by
  rcases accepted with ⟨bound⟩
  exact certificateBound_challengesValid bound

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler
