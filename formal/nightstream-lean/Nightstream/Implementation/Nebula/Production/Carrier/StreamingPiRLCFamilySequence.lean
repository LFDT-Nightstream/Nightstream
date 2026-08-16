import Mathlib.Algebra.BigOperators.Fin
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCAuthority

/-!
Contract: exact ordered composition of the 110 production PiRLC family
transitions.

Assurance tier: model-level exact refinement and Module-SIS reduction
boundary.

Owns one continuous family-state chain, exact ordinal family selection,
cross-family challenge carry, telescoping of all 110 concrete residual
updates, the terminal cursor value, and recovery of the authoritative PiCCS
inputs from a zero terminal residual or one named Module-SIS failure.

Does not own generated start or finish rows, normalized artifact decoding,
Rust assignment conformance, Poseidon2 collision resistance, the outer
streaming schedule, or Module-SIS hardness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 262144

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySequence

open scoped BigOperators
open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlc
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup

abbrev Family := ProductionStreamingPiRlcInputBinding.Family
abbrev Source := ProductionStreamingPiRlcInputBinding.Source
abbrev InputRings := ProductionStreamingPiRlcInputBinding.InputRings
abbrev RingF := Nightstream.SuperNeo.Concrete.RingF

/-- Exact number of family transitions in the production sequence. -/
def exactFamilyCount : Nat :=
  ProductionStreamingPiRlcInputBinding.familyCount

theorem exactFamilyCount_eq : exactFamilyCount = 110 := by
  rfl

/-- One accepted chain uses the same state value between adjacent phases and
opens each verifier-owned family from one common complete input array. -/
structure AcceptedRun
    (setup : InputBindingSetup) (inputs : InputRings) where
  state : Nat → FamilyState
  output : Family → RingF
  phase : ∀ ordinal : Fin exactFamilyCount,
    FamilyPhaseRelation setup
      (state ordinal.val) (state (ordinal.val + 1))
      (familyAtOrdinal ordinal)
      (fun source => inputs source (familyAtOrdinal ordinal))
      (output (familyAtOrdinal ordinal))

/-- The final PiRLC phase must observe the complete family cursor and a zero
rank-two residual. These are algebraic checks, not digest checks. -/
def FamilyFinishRelation (state : FamilyState) : Prop :=
  state.familyCursor = exactFamilyCount ∧
    state.inputResidual = zeroResidualFields

/-- The concrete residual contribution selected by one natural family
ordinal. Out-of-range indices are zero, which makes prefix sums total. -/
def phaseResidualAtNat
    (setup : InputBindingSetup) (inputs : InputRings)
    (index : Nat) (output : Fin (shape.rows * shape.degree)) :
    Nightstream.SuperNeo.Concrete.F :=
  if bound : index < exactFamilyCount then
    concretePhaseBinding setup (familyAtOrdinal ⟨index, bound⟩)
      (fun source => inputs source (familyAtOrdinal ⟨index, bound⟩))
      output
  else
    0

/-- Sum of the first `count` verifier-owned family contributions. -/
def residualPrefix
    (setup : InputBindingSetup) (inputs : InputRings)
    (count : Nat) : ResidualFields :=
  fun output =>
    ∑ index ∈ Finset.range count,
      phaseResidualAtNat setup inputs index output

theorem residualPrefix_zero
    (setup : InputBindingSetup) (inputs : InputRings) :
    residualPrefix setup inputs 0 = zeroResidualFields := by
  funext output
  simp [residualPrefix, zeroResidualFields]

theorem residualPrefix_succ
    (setup : InputBindingSetup) (inputs : InputRings)
    (ordinal : Fin exactFamilyCount) (output : Fin (shape.rows * shape.degree)) :
    residualPrefix setup inputs (ordinal.val + 1) output =
      residualPrefix setup inputs ordinal.val output +
        concretePhaseBinding setup (familyAtOrdinal ordinal)
          (fun source => inputs source (familyAtOrdinal ordinal)) output := by
  simp [residualPrefix, Finset.sum_range_succ, phaseResidualAtNat,
    ordinal.isLt]

namespace AcceptedRun

/-- Every phase sees the exact family ordinal selected by its position. -/
theorem cursor_before
    {setup : InputBindingSetup} {inputs : InputRings}
    (run : AcceptedRun setup inputs) (ordinal : Fin exactFamilyCount) :
    (run.state ordinal.val).familyCursor = ordinal.val := by
  simpa using (run.phase ordinal).1

/-- Each local transition advances to the next exact family cursor. -/
theorem cursor_after
    {setup : InputBindingSetup} {inputs : InputRings}
    (run : AcceptedRun setup inputs) (ordinal : Fin exactFamilyCount) :
    (run.state (ordinal.val + 1)).familyCursor = ordinal.val + 1 := by
  calc
    (run.state (ordinal.val + 1)).familyCursor =
        (run.state ordinal.val).familyCursor + 1 :=
      (run.phase ordinal).2.2.cursor
    _ = ordinal.val + 1 := by rw [run.cursor_before ordinal]

/-- A complete 110-phase chain reaches cursor 110. -/
theorem final_cursor
    {setup : InputBindingSetup} {inputs : InputRings}
    (run : AcceptedRun setup inputs) :
    (run.state exactFamilyCount).familyCursor = exactFamilyCount := by
  let last : Fin exactFamilyCount := ⟨109, by decide⟩
  have exact := run.cursor_after last
  simpa [last, exactFamilyCount] using exact

/-- Challenge rings are carried unchanged through every family prefix. -/
theorem challenges_eq_initial
    {setup : InputBindingSetup} {inputs : InputRings}
    (run : AcceptedRun setup inputs)
    (count : Nat) (bound : count ≤ exactFamilyCount) :
    (run.state count).challenges = (run.state 0).challenges := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      have countBound : count < exactFamilyCount := by omega
      let ordinal : Fin exactFamilyCount := ⟨count, countBound⟩
      calc
        (run.state (count + 1)).challenges =
            (run.state count).challenges :=
          (run.phase ordinal).2.2.challenges
        _ = (run.state 0).challenges :=
          inductionHypothesis (by omega)

/-- A row-derived start value makes the carried challenge rings
authoritative at every family boundary. -/
theorem challenges_eq_authoritative
    {setup : InputBindingSetup} {inputs : InputRings}
    (run : AcceptedRun setup inputs)
    {authoritativeChallenges : Source → RingF}
    {authoritativeResidual : InputResidual}
    (start : FamilyStartRelation (run.state 0)
      authoritativeChallenges authoritativeResidual)
    (count : Nat) (bound : count ≤ exactFamilyCount) :
    (run.state count).challenges = authoritativeChallenges := by
  exact (run.challenges_eq_initial count bound).trans start.2.2.2.1

/-- Every accepted family output uses the one challenge array placed by the
start phase. -/
theorem output_eq_authoritative
    {setup : InputBindingSetup} {inputs : InputRings}
    (run : AcceptedRun setup inputs)
    {authoritativeChallenges : Source → RingF}
    {authoritativeResidual : InputResidual}
    (start : FamilyStartRelation (run.state 0)
      authoritativeChallenges authoritativeResidual)
    (ordinal : Fin exactFamilyCount) :
    run.output (familyAtOrdinal ordinal) =
      combineOne authoritativeChallenges
        (fun source => inputs source (familyAtOrdinal ordinal)) := by
  calc
    run.output (familyAtOrdinal ordinal) =
        combineOne (run.state ordinal.val).challenges
          (fun source => inputs source (familyAtOrdinal ordinal)) :=
      (run.phase ordinal).2.1
    _ = combineOne authoritativeChallenges
          (fun source => inputs source (familyAtOrdinal ordinal)) := by
      rw [run.challenges_eq_authoritative start ordinal.val
        (Nat.le_of_lt ordinal.isLt)]

/-- The first residual equals the sum of the first `count` local openings
plus the residual carried after that prefix. -/
theorem residual_prefix_exact
    {setup : InputBindingSetup} {inputs : InputRings}
    (run : AcceptedRun setup inputs)
    (count : Nat) (bound : count ≤ exactFamilyCount) :
    (run.state 0).inputResidual =
      addResidualFields (residualPrefix setup inputs count)
        (run.state count).inputResidual := by
  induction count with
  | zero =>
      funext output
      simp [addResidualFields, residualPrefix]
  | succ count inductionHypothesis =>
      have countBound : count < exactFamilyCount := by omega
      let ordinal : Fin exactFamilyCount := ⟨count, countBound⟩
      have transition := (run.phase ordinal).2.2.inputResidual
      funext output
      have transitionAt := congrFun transition output
      calc
        (run.state 0).inputResidual output =
            residualPrefix setup inputs count output +
              (run.state count).inputResidual output := by
          exact congrFun (inductionHypothesis (by omega)) output
        _ = residualPrefix setup inputs count output +
              (concretePhaseBinding setup (familyAtOrdinal ordinal)
                  (fun source => inputs source (familyAtOrdinal ordinal))
                  output +
                (run.state (count + 1)).inputResidual output) := by
          rw [transitionAt]
          rfl
        _ = (residualPrefix setup inputs count output +
              concretePhaseBinding setup (familyAtOrdinal ordinal)
                (fun source => inputs source (familyAtOrdinal ordinal))
                output) +
              (run.state (count + 1)).inputResidual output := by
          rw [add_assoc]
        _ = residualPrefix setup inputs (count + 1) output +
              (run.state (count + 1)).inputResidual output := by
          rw [residualPrefix_succ setup inputs ordinal output]
        _ = addResidualFields (residualPrefix setup inputs (count + 1))
              (run.state (count + 1)).inputResidual output := rfl

/-- The natural prefix sum at 110 is the canonical `Fin 110` family sum in
the complete residual relation. -/
theorem residualPrefix_full
    (setup : InputBindingSetup) (inputs : InputRings) :
    residualPrefix setup inputs exactFamilyCount =
      fun output =>
        ∑ ordinal : Fin exactFamilyCount,
          concretePhaseBinding setup (familyAtOrdinal ordinal)
            (fun source => inputs source (familyAtOrdinal ordinal)) output := by
  funext output
  unfold residualPrefix
  rw [← Fin.sum_univ_eq_sum_range]
  simp [phaseResidualAtNat]

/-- The 110 local residual equations telescope to the exact complete
residual equation. -/
theorem concreteCompleteResidualRun
    {setup : InputBindingSetup} {inputs : InputRings}
    (run : AcceptedRun setup inputs) :
    ConcreteCompleteResidualRun setup
      (run.state 0).inputResidual
      (run.state exactFamilyCount).inputResidual inputs := by
  have aggregate := run.residual_prefix_exact exactFamilyCount (by rfl)
  rw [residualPrefix_full setup inputs] at aggregate
  exact aggregate

/-- Start authority plus a zero finish residual recovers every supplied
PiRLC input, or gives the named concrete Module-SIS failure. -/
theorem start_finish_recovers_inputs_or_failure
    {setup : InputBindingSetup}
    {authoritative supplied : InputRings}
    {authoritativeChallenges : Source → RingF}
    (run : AcceptedRun setup supplied)
    (start : FamilyStartRelation (run.state 0) authoritativeChallenges
      (concreteBinding setup authoritative))
    (finish : FamilyFinishRelation (run.state exactFamilyCount)) :
    Or (supplied = authoritative) (ConcreteBindingFailure setup) := by
  have complete := run.concreteCompleteResidualRun
  rw [finish.2] at complete
  exact concrete_complete_zero_recovers_inputs_or_failure setup
    authoritative supplied (run.state 0).inputResidual start.2.1 complete

/-- In the non-failure branch, every accepted output family is the exact
PiRLC combination of the authoritative PiCCS inputs and challenges. -/
theorem outputs_exact_or_failure
    {setup : InputBindingSetup}
    {authoritative supplied : InputRings}
    {authoritativeChallenges : Source → RingF}
    (run : AcceptedRun setup supplied)
    (start : FamilyStartRelation (run.state 0) authoritativeChallenges
      (concreteBinding setup authoritative))
    (finish : FamilyFinishRelation (run.state exactFamilyCount)) :
    Or
      (∀ family,
        run.output family =
          familyOutput authoritativeChallenges authoritative family)
      (ConcreteBindingFailure setup) := by
  rcases run.start_finish_recovers_inputs_or_failure start finish with
    exact | failure
  · subst supplied
    left
    intro family
    simpa [familyOutput] using
      run.output_eq_authoritative start (familyIndex family)
  · exact Or.inr failure

end AcceptedRun

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySequence
