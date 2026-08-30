import NightstreamFPrime.Circuit.SupportRange
import NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal

/-!
Owns variable-support propagation for the recipe-free Poseidon2 Duplex wiring
projection. The full compiler remains the semantic and row authority.
-/

namespace NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Circuit.SupportRange
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Layer

def KSupported (value : KExpr) (allowed : Nat → Prop) : Prop :=
  value.c0.VarsSatisfy allowed ∧ value.c1.VarsSatisfy allowed

def StateSupported (state : EState) (allowed : Nat → Prop) : Prop :=
  ∀ lane, (state lane).VarsSatisfy allowed

theorem KSupported.mono {value : KExpr} {allowed larger : Nat → Prop}
    (support : KSupported value allowed)
    (includes : ∀ index, allowed index → larger index) :
    KSupported value larger :=
  ⟨Expr.VarsSatisfy.mono value.c0 support.1 includes,
    Expr.VarsSatisfy.mono value.c1 support.2 includes⟩

theorem StateSupported.mono {state : EState} {allowed larger : Nat → Prop}
    (support : StateSupported state allowed)
    (includes : ∀ index, allowed index → larger index) :
    StateSupported state larger := by
  intro lane
  exact Expr.VarsSatisfy.mono (state lane) (support lane) includes

theorem sampleGetD_supported (samples : List KExpr) (index : Nat)
    (fallback : KExpr) (allowed : Nat → Prop)
    (support : ∀ sample ∈ samples, KSupported sample allowed)
    (indexBound : index < samples.length) :
    KSupported (samples.getD index fallback) allowed := by
  rw [List.getD_eq_get samples fallback ⟨index, indexBound⟩]
  exact support _ (List.get_mem samples ⟨index, indexBound⟩)

private theorem scheduleOutput_supported (allowed : Nat → Prop)
    (base start : Nat) (baseLeStart : base ≤ start) :
    StateSupported (Permutation.scheduleOutput start)
      (Extend allowed base (start + 592)) := by
  intro lane
  simp only [Permutation.scheduleOutput, Permutation.freshState,
    Expr.VarsSatisfy]
  apply NightstreamFPrime.Circuit.SupportRange.interval
  · omega
  · have laneBound := lane.isLt
    omega

private theorem compileAbsorbWiring_output_supported
    (allowed : Nat → Prop) (base start : Nat) (state : EState)
    (blocks : List (List Expr)) (baseLeStart : base ≤ start)
    (stateSupport : StateSupported state (Extend allowed base start)) :
    StateSupported (compileAbsorbWiring start state blocks).output
      (Extend allowed base (start + blocks.length * 592)) := by
  induction blocks generalizing start state with
  | nil =>
      simpa [compileAbsorbWiring] using stateSupport
  | cons block blocks inductionHypothesis =>
      have nextSupport := scheduleOutput_supported allowed base start baseLeStart
      have tail := inductionHypothesis
        (start := start + 592) (state := Permutation.scheduleOutput start)
        (by omega) nextSupport
      have endEq :
          start + (List.length (block :: blocks)) * 592 =
            (start + 592) + blocks.length * 592 := by
        simp only [List.length_cons]
        omega
      simpa only [compileAbsorbWiring, endEq] using tail

/-- Every exposed sample and final state uses only prior supported variables
or variables allocated inside this exact wiring interval. -/
theorem compileWiring_supported (allowed : Nat → Prop)
    (base start : Nat) (state : EState) (actions : List Action)
    (baseLeStart : base ≤ start)
    (stateSupport : StateSupported state (Extend allowed base start)) :
    (∀ sample ∈ (compileWiring start state actions).samples,
        KSupported sample (Extend allowed base (start + recipeCount actions))) ∧
      StateSupported (compileWiring start state actions).output
        (Extend allowed base (start + recipeCount actions)) := by
  induction actions generalizing start state with
  | nil =>
      constructor
      · intro sample member
        simp [compileWiring] at member
      · simpa [compileWiring, recipeCount] using stateSupport
  | cons action actions inductionHypothesis =>
      cases action with
      | absorb input =>
          let blocks := Hash.inputChunks input
          let absorbed := compileAbsorbWiring start state blocks
          have absorbedSupport : StateSupported absorbed.output
              (Extend allowed base absorbed.next) := by
            have support := compileAbsorbWiring_output_supported allowed base
              start state blocks baseLeStart stateSupport
            have nextEq := compileAbsorbWiring_next start state blocks
            rw [nextEq]
            exact support
          have baseLeNext : base ≤ absorbed.next := by
            dsimp [absorbed]
            rw [compileAbsorbWiring_next]
            omega
          have tail := inductionHypothesis
            (start := absorbed.next) (state := absorbed.output)
            baseLeNext absorbedSupport
          have finishEq :
              absorbed.next + recipeCount actions =
                start + recipeCount (.absorb input :: actions) := by
            dsimp [absorbed]
            rw [compileAbsorbWiring_next]
            simp only [recipeCount, List.map_cons, List.sum_cons,
              Action.recipeCount, blocks]
            omega
          simpa only [compileWiring, blocks, absorbed, finishEq] using tail
      | squeezeK expected =>
          let tailStart := start + 1184
          let tailState := Permutation.scheduleOutput (start + 592)
          have tailStateSupport : StateSupported tailState
              (Extend allowed base tailStart) := by
            have support := scheduleOutput_supported allowed base (start + 592)
              (by omega)
            simpa [tailState, tailStart, Nat.add_assoc] using support
          have tail := inductionHypothesis
            (start := tailStart) (state := tailState) (by
              unfold tailStart
              omega) tailStateSupport
          have finishEq :
              tailStart + recipeCount actions =
                start + recipeCount (.squeezeK expected :: actions) := by
            simp [tailStart, recipeCount, Action.recipeCount, Nat.add_assoc]
          have tailAtFinish :
              (∀ sample ∈ (compileWiring tailStart tailState actions).samples,
                  KSupported sample
                    (Extend allowed base
                      (start + recipeCount (.squeezeK expected :: actions)))) ∧
                StateSupported (compileWiring tailStart tailState actions).output
                  (Extend allowed base
                    (start + recipeCount (.squeezeK expected :: actions))) := by
            simpa only [finishEq] using tail
          constructor
          · intro sample member
            simp only [compileWiring, List.mem_cons] at member
            rcases member with rfl | member
            · constructor
              · apply Expr.VarsSatisfy.mono (state 0) (stateSupport 0)
                intro index support
                apply NightstreamFPrime.Circuit.SupportRange.mono_finish support
                simp [recipeCount, Action.recipeCount]
              · have firstSupport :
                    (Permutation.scheduleOutput start 0).VarsSatisfy
                      (Extend allowed base (start + 592)) :=
                    scheduleOutput_supported allowed base start baseLeStart 0
                apply Expr.VarsSatisfy.mono _ firstSupport
                intro index support
                apply NightstreamFPrime.Circuit.SupportRange.mono_finish support
                omega
            · exact tailAtFinish.1 sample member
          · exact tailAtFinish.2

end NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal
