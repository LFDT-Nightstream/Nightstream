import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilySequence
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyXOutContinuity

/-!
Contract: full-`x_out` continuity for the exact 110-arm physical PiRLC run.

Owns one stateful outer state before and after every accepted physical family
arm, exact binding of each local family digest into the outer semantic-state
lane, adjacent complete-`x_out` equality, and construction of the exact
model-level family run modulo the named two-layer continuity failure.

Does not own lifecycle rows that derive the opaque outer-envelope fields, the
physical-to-typed outer-state bridge, Rust assignment conformance, outer-state
transition semantics, start or finish circuits, collision resistance,
Module-SIS hardness, or the recursive lifecycle.

Emits constraints: no.

Assurance tier: security-reduced. Phase-side hash and structural rows are
artifact-checked elsewhere. Missing lifecycle authority remains an explicit
premise. No carried digest is used as independent authority.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlc
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySequence
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyContinuity
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPhysicalState
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutContinuity
open Nightstream.Protocol.FPrime

universe uParams uStructure uHeader uRunning uFresh uNebulaDigest

abbrev PhysicalFamily :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlc.Family
abbrev PhysicalSource :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlc.Source
abbrev PhysicalInputRings :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlc.InputRings
abbrev PhysicalRing := Nightstream.SuperNeo.Concrete.RingF

abbrev OuterState
    (Running : Type uRunning) (Fresh : Type uFresh) (Nebula : Type) :=
  Nightstream.HyperNova.Construction2.State FamilyDigest Running Fresh Nebula

/-- One accepted physical family arm plus the two complete recursive states
whose semantic lanes recompute the arm's local before and after digests. -/
structure AcceptedFullStateArm
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (semantics :
      XOut.Semantics Params StructureDigest Header FamilyDigest Nebula
        NebulaDigest)
    (context : XOut.Context Params StructureDigest Header FamilyDigest)
    (setup : InputBindingSetup) (family : PhysicalFamily) where
  physical : AcceptedArm setup family
  beforeOuter : OuterState Running Fresh Nebula
  afterOuter : OuterState Running Fresh Nebula
  beforePinned :
    XOut.StatePinned semantics .stateful context beforeOuter
  afterPinned :
    XOut.StatePinned semantics .stateful context afterOuter
  beforeSemantic :
    beforeOuter.semanticState = familyStateDigest physical.beforeState
  afterSemantic :
    afterOuter.semanticState = familyStateDigest physical.afterState

/-- One exact accepted arm per verifier-owned family ordinal. Adjacent arms
share the complete recursive-state output through equality of their
independently recomputed `x_out` values. -/
structure AcceptedFullStateRun
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (semantics :
      XOut.Semantics Params StructureDigest Header FamilyDigest Nebula
        NebulaDigest)
    (context : XOut.Context Params StructureDigest Header FamilyDigest)
    (setup : InputBindingSetup) where
  arm : ∀ ordinal : Fin exactFamilyCount,
    AcceptedFullStateArm Running Fresh semantics context setup
      (familyAtOrdinal ordinal)
  continuous : ∀ (ordinal : Fin exactFamilyCount)
      (hasNext : ordinal.val + 1 < exactFamilyCount),
    XOut.compute semantics .stateful context (arm ordinal).afterOuter =
      XOut.compute semantics .stateful context
        (arm ⟨ordinal.val + 1, hasNext⟩).beforeOuter

namespace AcceptedFullStateRun

/-- The last verifier-owned family ordinal. -/
def lastOrdinal : Fin exactFamilyCount := ⟨109, by decide⟩

/-- The exact family-major input array decoded from all accepted bodies. -/
def inputRings
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {semantics :
      XOut.Semantics Params StructureDigest Header FamilyDigest Nebula
        NebulaDigest}
    {context : XOut.Context Params StructureDigest Header FamilyDigest}
    {setup : InputBindingSetup}
    (run : AcceptedFullStateRun Running Fresh semantics context setup) :
    PhysicalInputRings :=
  fun source family =>
    (run.arm (familyIndex family)).physical.phaseInputs source

/-- The exact output array decoded from all accepted bodies. -/
def outputs
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {semantics :
      XOut.Semantics Params StructureDigest Header FamilyDigest Nebula
        NebulaDigest}
    {context : XOut.Context Params StructureDigest Header FamilyDigest}
    {setup : InputBindingSetup}
    (run : AcceptedFullStateRun Running Fresh semantics context setup) :
    PhysicalFamily → PhysicalRing :=
  fun family =>
    (run.arm (familyIndex family)).physical.phaseOutput

/-- Total local family state at each sequence boundary. -/
def boundaryState
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {semantics :
      XOut.Semantics Params StructureDigest Header FamilyDigest Nebula
        NebulaDigest}
    {context : XOut.Context Params StructureDigest Header FamilyDigest}
    {setup : InputBindingSetup}
    (run : AcceptedFullStateRun Running Fresh semantics context setup)
    (index : Nat) : FamilyState :=
  if bound : index < exactFamilyCount then
    (run.arm ⟨index, bound⟩).physical.beforeState
  else
    (run.arm lastOrdinal).physical.afterState

@[simp] theorem boundaryState_before
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {semantics :
      XOut.Semantics Params StructureDigest Header FamilyDigest Nebula
        NebulaDigest}
    {context : XOut.Context Params StructureDigest Header FamilyDigest}
    {setup : InputBindingSetup}
    (run : AcceptedFullStateRun Running Fresh semantics context setup)
    (ordinal : Fin exactFamilyCount) :
    run.boundaryState ordinal.val =
      (run.arm ordinal).physical.beforeState := by
  simp [boundaryState, ordinal.isLt]

/-- Equal adjacent complete outputs recover the exact local family state, or
expose the named outer or inner binding failure. -/
theorem adjacent_state_or_failure
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {semantics :
      XOut.Semantics Params StructureDigest Header FamilyDigest Nebula
        NebulaDigest}
    {context : XOut.Context Params StructureDigest Header FamilyDigest}
    {setup : InputBindingSetup}
    (run : AcceptedFullStateRun Running Fresh semantics context setup)
    (ordinal : Fin exactFamilyCount)
    (hasNext : ordinal.val + 1 < exactFamilyCount) :
    (run.arm ordinal).physical.afterState =
        (run.arm ⟨ordinal.val + 1, hasNext⟩).physical.beforeState ∨
      ContinuityFailure semantics := by
  let left := run.arm ordinal
  let right := run.arm ⟨ordinal.val + 1, hasNext⟩
  exact familyState_eq_or_continuity_failure semantics .stateful .stateful
    context context left.afterOuter right.beforeOuter
    left.physical.afterState right.physical.beforeState
    left.afterPinned right.beforePinned left.afterSemantic
    right.beforeSemantic
    (accepted_after_state_fields_canonical left.physical)
    (accepted_before_state_fields_canonical right.physical)
    (run.continuous ordinal hasNext)

private theorem adjacent_state_eq
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {semantics :
      XOut.Semantics Params StructureDigest Header FamilyDigest Nebula
        NebulaDigest}
    {context : XOut.Context Params StructureDigest Header FamilyDigest}
    {setup : InputBindingSetup}
    (run : AcceptedFullStateRun Running Fresh semantics context setup)
    (noFailure : ¬ ContinuityFailure semantics)
    (ordinal : Fin exactFamilyCount)
    (hasNext : ordinal.val + 1 < exactFamilyCount) :
    (run.arm ordinal).physical.afterState =
      (run.arm ⟨ordinal.val + 1, hasNext⟩).physical.beforeState :=
  (run.adjacent_state_or_failure ordinal hasNext).resolve_right noFailure

theorem ordinal_eq_last_of_no_next
    (ordinal : Fin exactFamilyCount)
    (noNext : ¬ ordinal.val + 1 < exactFamilyCount) :
    ordinal = lastOrdinal := by
  apply Fin.ext
  have bound : ordinal.val < 110 := by
    exact ordinal.isLt
  have noNext' : ¬ ordinal.val + 1 < 110 := by
    simpa [exactFamilyCount,
      ProductionStreamingPiRlcInputBinding.familyCount] using noNext
  have value : ordinal.val = 109 := by omega
  simpa [lastOrdinal] using value

/-- With no binding failure, the boundary after one arm is its decoded after
state. -/
theorem boundaryState_after
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {semantics :
      XOut.Semantics Params StructureDigest Header FamilyDigest Nebula
        NebulaDigest}
    {context : XOut.Context Params StructureDigest Header FamilyDigest}
    {setup : InputBindingSetup}
    (run : AcceptedFullStateRun Running Fresh semantics context setup)
    (noFailure : ¬ ContinuityFailure semantics)
    (ordinal : Fin exactFamilyCount) :
    run.boundaryState (ordinal.val + 1) =
      (run.arm ordinal).physical.afterState := by
  by_cases hasNext : ordinal.val + 1 < exactFamilyCount
  · rw [boundaryState, dif_pos hasNext]
    exact (run.adjacent_state_eq noFailure ordinal hasNext).symm
  · rw [boundaryState, dif_neg hasNext]
    rw [ordinal_eq_last_of_no_next ordinal hasNext]

/-- In the no-failure branch, the complete full-state chain constructs the
exact model-level family sequence. -/
def semanticRun
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {semantics :
      XOut.Semantics Params StructureDigest Header FamilyDigest Nebula
        NebulaDigest}
    {context : XOut.Context Params StructureDigest Header FamilyDigest}
    {setup : InputBindingSetup}
    (run : AcceptedFullStateRun Running Fresh semantics context setup)
    (noFailure : ¬ ContinuityFailure semantics) :
    AcceptedRun setup run.inputRings where
  state := run.boundaryState
  output := run.outputs
  phase := by
    intro ordinal
    rw [run.boundaryState_before ordinal,
      run.boundaryState_after noFailure ordinal]
    change FamilyPhaseRelation setup (run.arm ordinal).physical.beforeState
      (run.arm ordinal).physical.afterState (familyAtOrdinal ordinal)
      (fun source =>
        (run.arm (familyIndex (familyAtOrdinal ordinal))).physical.phaseInputs
          source)
      (run.arm (familyIndex (familyAtOrdinal ordinal))).physical.phaseOutput
    rw [familyIndex_familyAtOrdinal]
    exact (run.arm ordinal).physical.phase

/-- The complete physical family chain refines the exact semantic run, or
exposes one named two-layer continuity failure. -/
theorem semanticRun_or_failure
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {semantics :
      XOut.Semantics Params StructureDigest Header FamilyDigest Nebula
        NebulaDigest}
    {context : XOut.Context Params StructureDigest Header FamilyDigest}
    {setup : InputBindingSetup}
    (run : AcceptedFullStateRun Running Fresh semantics context setup) :
    Nonempty (AcceptedRun setup run.inputRings) ∨
      ContinuityFailure semantics := by
  classical
  by_cases failure : ContinuityFailure semantics
  · exact Or.inr failure
  · exact Or.inl ⟨run.semanticRun failure⟩

/-- Exact start and finish authority recover every body-supplied input, or
expose the concrete Module-SIS failure or a two-layer continuity failure. -/
theorem start_finish_recovers_inputs_or_failure
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {semantics :
      XOut.Semantics Params StructureDigest Header FamilyDigest Nebula
        NebulaDigest}
    {context : XOut.Context Params StructureDigest Header FamilyDigest}
    {setup : InputBindingSetup}
    {authoritative : PhysicalInputRings}
    {authoritativeChallenges : PhysicalSource → PhysicalRing}
    (run : AcceptedFullStateRun Running Fresh semantics context setup)
    (start : FamilyStartRelation (run.boundaryState 0)
      authoritativeChallenges (concreteBinding setup authoritative))
    (finish : FamilyFinishRelation
      (run.boundaryState exactFamilyCount)) :
    run.inputRings = authoritative ∨
      ConcreteBindingFailure setup ∨ ContinuityFailure semantics := by
  classical
  by_cases failure : ContinuityFailure semantics
  · exact Or.inr (Or.inr failure)
  · let semantic := run.semanticRun failure
    have result := semantic.start_finish_recovers_inputs_or_failure
      (by simpa [semantic, semanticRun] using start)
      (by simpa [semantic, semanticRun] using finish)
    rcases result with exact | bindingFailure
    · exact Or.inl (by simpa [semantic, semanticRun] using exact)
    · exact Or.inr (Or.inl bindingFailure)

/-- In the non-failure branch, every physical output is the exact PiRLC
combination of the authoritative PiCCS inputs and challenge array. -/
theorem outputs_exact_or_failure
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    {semantics :
      XOut.Semantics Params StructureDigest Header FamilyDigest Nebula
        NebulaDigest}
    {context : XOut.Context Params StructureDigest Header FamilyDigest}
    {setup : InputBindingSetup}
    {authoritative : PhysicalInputRings}
    {authoritativeChallenges : PhysicalSource → PhysicalRing}
    (run : AcceptedFullStateRun Running Fresh semantics context setup)
    (start : FamilyStartRelation (run.boundaryState 0)
      authoritativeChallenges (concreteBinding setup authoritative))
    (finish : FamilyFinishRelation
      (run.boundaryState exactFamilyCount)) :
    (∀ family,
        run.outputs family =
          familyOutput authoritativeChallenges authoritative family) ∨
      ConcreteBindingFailure setup ∨ ContinuityFailure semantics := by
  classical
  by_cases failure : ContinuityFailure semantics
  · exact Or.inr (Or.inr failure)
  · let semantic := run.semanticRun failure
    have result := semantic.outputs_exact_or_failure
      (by simpa [semantic, semanticRun] using start)
      (by simpa [semantic, semanticRun] using finish)
    rcases result with exact | bindingFailure
    · exact Or.inl (by simpa [semantic, semanticRun] using exact)
    · exact Or.inr (Or.inl bindingFailure)

end AcceptedFullStateRun

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence
