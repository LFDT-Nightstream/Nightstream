import NightstreamFPrime.Layout.Poseidon2.Duplex
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness

/-!
Paper authority: SuperNeo v1_1, section 7.3, PiCCS Fiat–Shamir challenges.
Obligation: Derive all 25 `α` coordinates and `γ` from the exact labelled
Poseidon2 transcript schedule.

Inputs:
- the child-owned state produced by Statement absorption.

Outputs:
- 25 verifier-derived `α` values;
- one verifier-derived `γ` value;
- the child-owned outgoing transcript state.

Constraint groups:
- labelled constant absorptions;
- two Poseidon2 permutations for each extension-field squeeze;
- no expected-sample or state-copy rows.

Parent coverage:
- `Formal.opsAt`, child `piccs.v1_1.challenge_derivation`.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.ChallengeDerivation

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation
open NightstreamFPrime.Layout.Poseidon2
open NightstreamFPrime.Layout.Poseidon2.Duplex
open NightstreamFPrime.Layout.Polynomial.Horner
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth degreeBound : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The only external symbolic input is an affine incoming transcript state. -/
structure InputsAffine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation.Interface)
    (offset : Nat) : Prop where
  initialState : StateAffine (interface.initialState offset)

private theorem constantWords_affine (words : List F) :
    ListAffine (constantWords words) := by
  intro expression member
  rw [constantWords, List.mem_map] at member
  rcases member with ⟨word, _, rfl⟩
  exact R1CS.isAffine_const word

private theorem zero_affine : KExprAffine KExpr.zero := by
  exact ⟨R1CS.isAffine_const 0, R1CS.isAffine_const 0⟩

private theorem labelActions_affine
    (label : FiatShamir.ChallengeLabel productionShape)
    (expected : KExpr) (expectedAffine : KExprAffine expected) :
    ActionsAffine (labelActions label expected) := by
  apply ActionsAffine.cons
  · exact constantWords_affine _
  · apply ActionsAffine.cons
    · exact expectedAffine
    · intro action member
      simp at member

private theorem labelledActions_affine
    (labels : List (FiatShamir.ChallengeLabel productionShape))
    (samples : List KExpr)
    (samplesAffine : ∀ sample ∈ samples, KExprAffine sample) :
    ActionsAffine (labelledActions labels samples) := by
  induction labels generalizing samples with
  | nil => simp [labelledActions, ActionsAffine]
  | cons label labels inductionHypothesis =>
      cases samples with
      | nil => simp [labelledActions, ActionsAffine]
      | cons sample samples =>
          apply ActionsAffine.append
          · exact labelActions_affine label sample
              (samplesAffine sample (by simp))
          · exact inductionHypothesis samples (by
              intro current member
              exact samplesAffine current (by simp [member]))

private theorem replicatedZero_affine (count : Nat) :
    ∀ sample ∈ List.replicate count KExpr.zero, KExprAffine sample := by
  induction count with
  | zero => simp
  | succ count inductionHypothesis =>
      intro sample member
      simp only [List.replicate_succ, List.mem_cons] at member
      rcases member with rfl | member
      · exact zero_affine
      · exact inductionHypothesis sample member

private theorem layoutActions_affine : ActionsAffine layoutActions := by
  unfold layoutActions
  exact labelledActions_affine challengeLabels
    (List.replicate 26 KExpr.zero) (replicatedZero_affine 26)

private theorem layoutProgram_samples_affine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation.Interface)
    (offset : Nat) (inputs : InputsAffine interface offset) :
    ∀ sample ∈ (layoutProgram interface offset).samples,
      KExprAffine sample := by
  exact compile_samples_affine offset (interface.initialState offset)
    layoutActions inputs.initialState layoutActions_affine

private theorem layoutProgram_samples_linear
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation.Interface)
    (offset : Nat)
    (initialFresh : StateFresh (interface.initialState offset)) :
    ∀ sample ∈ (layoutProgram interface offset).samples,
      KExprLinear sample := by
  exact compile_samples_linear offset (interface.initialState offset)
    layoutActions initialFresh

theorem alpha_linear
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation.Interface)
    (offset : Nat)
    (initialFresh : StateFresh (interface.initialState offset))
    (coordinate : Fin productionShape.cubeVariables) :
    KExprLinear (alpha interface offset coordinate) := by
  apply layoutProgram_samples_linear interface offset initialFresh
  exact List.mem_of_mem_take (List.get_mem _ _)

theorem gamma_linear
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation.Interface)
    (offset : Nat)
    (initialFresh : StateFresh (interface.initialState offset)) :
    KExprLinear (gamma interface offset) := by
  apply layoutProgram_samples_linear interface offset initialFresh
  exact List.get_mem _ _

theorem finalState_fresh
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation.Interface)
    (offset : Nat)
    (initialFresh : StateFresh (interface.initialState offset)) :
    StateFresh (finalState interface offset) := by
  unfold finalState program
  exact compile_output_fresh offset (interface.initialState offset)
    (actions interface offset) initialFresh

theorem finalState_affine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation.Interface)
    (offset : Nat)
    (initialFresh : StateFresh (interface.initialState offset)) :
    StateAffine (finalState interface offset) :=
  (finalState_fresh interface offset initialFresh).affine

theorem actions_affine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation.Interface)
    (offset : Nat) (inputs : InputsAffine interface offset) :
    ActionsAffine (actions interface offset) := by
  rw [actions_eq_labelled]
  exact labelledActions_affine challengeLabels
    (layoutProgram interface offset).samples
    (layoutProgram_samples_affine interface offset inputs)

/-- Exact parent-facing physical footprint. Derived challenges and transcript
states use compiler output variables and need no boundary-copy rows. -/
def footprint
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (parentOffset : Nat)
    (inputs : ∀ offset,
      InputsAffine
        (Formal.challengeInterface (Formal.atOffset interface parentOffset)
          parentOffset) offset) :
    R1CS.CircuitFootprint (Formal.challengeCircuit interface parentOffset) :=
  let child := Formal.challengeInterface
    (Formal.atOffset interface parentOffset) parentOffset
  {
    freshColumnCount := fun _ => 0
    physicalRowCount := fun _ => 46176
    freshColumnCount_eq := by
      intro offset
      unfold Formal.challengeCircuit
      dsimp only
      rw [FormalCircuit.withConstantFootprint_main]
      change R1CS.totalFreshCount (flatConstraints
        (opsAt child offset)) = 0
      rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation.flatConstraints_opsAt]
      apply R1CS.recipeConstraints_totalFreshCount
      exact compile_recipes_direct offset (child.initialState offset)
        (actions child offset) (inputs offset).initialState
        (actions_affine child offset (inputs offset))
    physicalRowCount_eq := by
      intro offset
      unfold Formal.challengeCircuit
      dsimp only
      rw [FormalCircuit.withConstantFootprint_main]
      change R1CS.totalRowCount (flatConstraints
        (opsAt child offset)) = 46176
      rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation.flatConstraints_opsAt]
      rw [R1CS.recipeConstraints_totalRowCount]
      exact NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation.program_recipes_length
        child offset
      exact compile_recipes_direct offset (child.initialState offset)
        (actions child offset) (inputs offset).initialState
        (actions_affine child offset (inputs offset))
  }

theorem freshColumnCount_eq
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (parentOffset : Nat)
    (inputs : ∀ offset,
      InputsAffine
        (Formal.challengeInterface (Formal.atOffset interface parentOffset)
          parentOffset) offset)
    (offset : Nat) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Formal.challengeCircuit interface parentOffset).main offset)) = 0 :=
  (footprint interface parentOffset inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (parentOffset : Nat)
    (inputs : ∀ offset,
      InputsAffine
        (Formal.challengeInterface (Formal.atOffset interface parentOffset)
          parentOffset) offset)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Formal.challengeCircuit interface parentOffset).main offset)) =
        46176 :=
  (footprint interface parentOffset inputs).physicalRowCount_eq offset

end NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.ChallengeDerivation
