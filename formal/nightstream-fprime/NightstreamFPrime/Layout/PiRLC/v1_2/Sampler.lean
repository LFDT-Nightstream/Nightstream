import NightstreamFPrime.Layout.Sampling.WideReduction
import NightstreamFPrime.Layout.PiRLC.v1_2.Leaves.TranscriptAbsorption
import NightstreamFPrime.Layout.Poseidon2.PermutationOwned
import NightstreamFPrime.Lifecycle.PiRLC.v1_2.Sampler

/-! Exact physical cost of one scalar. Parent proofs use the certified child
costs; they do not expand the permutation or checked-range row lists. -/

namespace NightstreamFPrime.Layout.PiRLC.v1_2.Sampler

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Layout.Poseidon2

namespace Logical

abbrev Assumptions := NightstreamFPrime.Lifecycle.PiRLC.v1_2.Sampler.Assumptions
abbrev Interface := NightstreamFPrime.Lifecycle.PiRLC.v1_2.Sampler.Interface
abbrev SpecHolds := NightstreamFPrime.Lifecycle.PiRLC.v1_2.Sampler.SpecHolds
abbrev circuit := NightstreamFPrime.Lifecycle.PiRLC.v1_2.Sampler.circuit
abbrev logicalPrivateCount := NightstreamFPrime.Lifecycle.PiRLC.v1_2.Sampler.logicalPrivateCount
abbrev main := NightstreamFPrime.Lifecycle.PiRLC.v1_2.Sampler.main

end Logical

private theorem entered_fresh (interface : Logical.Interface) (coordinate offset : Nat) :
    Duplex.StateFresh (Lifecycle.PiRLC.v1_2.Sampler.enteredState interface coordinate offset) := by
  unfold Lifecycle.PiRLC.v1_2.Sampler.enteredState
    Lifecycle.PiRLC.v1_2.TranscriptAbsorption.output Lifecycle.PiRLC.v1_2.TranscriptAbsorption.ownedInterface
    Gadgets.Poseidon2.Duplex.Formal.Owned.output Gadgets.Poseidon2.Duplex.Formal.Owned.program
  apply Duplex.compile_output_fresh_of_head_absorb
  intro empty
  have lengths := congrArg List.length empty
  simp [Gadgets.Poseidon2.Hash.inputChunks, Lifecycle.PiRLC.v1_2.TranscriptAbsorption.constantWords,
    Lifecycle.PiRLC.v1_2.TranscriptAbsorption.frameWords, Spec.Poseidon2.rate] at lengths

theorem entered_affine (interface : Logical.Interface) (coordinate offset : Nat) :
    StateAffine (Lifecycle.PiRLC.v1_2.Sampler.enteredState interface coordinate offset) := (entered_fresh interface coordinate offset).affine

theorem output_affine (interface : Logical.Interface) (coordinate offset : Nat) :
    StateAffine (Lifecycle.PiRLC.v1_2.Sampler.outputState interface coordinate offset) := by
  intro lane
  exact R1CS.isAffine_var _

theorem child_constraints (name : String) (child : FormalCircuit) (offset : Nat) :
    (Sequence.childOp name child offset).flatConstraints = flatConstraints (Circuit.ops child.main offset) := rfl

theorem entry_fresh (interface : Logical.Interface) (coordinate offset : Nat)
    (inputs : ∀ current, StateAffine (interface.initialState current)) :
    R1CS.totalFreshCount ((Lifecycle.PiRLC.v1_2.Sampler.entryOp interface coordinate offset).flatConstraints) = 0 := by
  rw [Lifecycle.PiRLC.v1_2.Sampler.entryOp, child_constraints, Lifecycle.PiRLC.v1_2.Sampler.entry, FormalCircuit.withConstantFootprint_main]
  exact PiRLC.v1_2.Leaves.TranscriptAbsorption.freshColumnCount_eq interface coordinate (fun current => ⟨inputs current⟩) offset

theorem range_fresh (interface : Logical.Interface) (coordinate offset : Nat) :
    R1CS.totalFreshCount ((Lifecycle.PiRLC.v1_2.Sampler.rangeOp interface coordinate offset).flatConstraints) = 144 := by
  rw [Lifecycle.PiRLC.v1_2.Sampler.rangeOp, child_constraints]
  change R1CS.totalFreshCount (flatConstraints (WideReduction.Program.operations
    (Lifecycle.PiRLC.v1_2.Sampler.rangeInterface interface coordinate offset) (Lifecycle.PiRLC.v1_2.Sampler.rangeOffset offset))) = _
  rw [WideReduction.Program.constraints_eq]
  exact (Layout.Sampling.WideReduction.counts _ _ _ (fun lane => entered_affine interface coordinate offset (Lifecycle.PiRLC.v1_2.Sampler.rateLane lane))).1

theorem advance_fresh (interface : Logical.Interface) (coordinate offset : Nat) :
    R1CS.totalFreshCount ((Lifecycle.PiRLC.v1_2.Sampler.advanceOp interface coordinate offset).flatConstraints) = 0 := by
  rw [Lifecycle.PiRLC.v1_2.Sampler.advanceOp, child_constraints]
  exact PermutationOwned.totalFreshCount_eq (Lifecycle.PiRLC.v1_2.Sampler.advanceInterface interface coordinate offset)
    (Lifecycle.PiRLC.v1_2.Sampler.advanceOffset offset) ⟨entered_affine interface coordinate offset⟩

private theorem word_affine (start : Nat) (position : Fin ringDegree) :
    R1CS.IsAffine (WideReduction.Program.outputWord start position) := by
  unfold WideReduction.Program.outputWord WideReduction.linearExpr
  exact R1CS.IsAffine.add (R1CS.IsAffine.const_mul _ (R1CS.isAffine_var _))
    (R1CS.IsAffine.add (R1CS.IsAffine.const_mul _ (R1CS.isAffine_var _))
      (R1CS.IsAffine.add (R1CS.IsAffine.const_mul _ (R1CS.isAffine_var _)) (R1CS.isAffine_const _)))

theorem words_fresh (offset : Nat) :
    R1CS.totalFreshCount ((Lifecycle.PiRLC.v1_2.Sampler.wordsOp offset).flatConstraints) = 0 := by
  rw [Lifecycle.PiRLC.v1_2.Sampler.wordsOp, child_constraints,
    Lifecycle.PiRLC.v1_2.SamplerWords.circuit_ops, Lifecycle.PiRLC.v1_2.SamplerWords.constraints_eq]
  apply R1CS.recipeConstraints_totalFreshCount
  have all : ∀ recipe ∈ Lifecycle.PiRLC.v1_2.SamplerWords.recipes
      (Lifecycle.PiRLC.v1_2.Sampler.rangeOffset offset), R1CS.IsAffine recipe := by
    intro recipe member
    obtain ⟨position, rfl⟩ := List.mem_ofFn.mp member
    exact word_affine _ position
  generalize Lifecycle.PiRLC.v1_2.SamplerWords.recipes
    (Lifecycle.PiRLC.v1_2.Sampler.rangeOffset offset) = recipes at all ⊢
  generalize Lifecycle.PiRLC.v1_2.Sampler.wordsOffset offset = start
  induction recipes generalizing start with
  | nil => trivial
  | cons recipe rest ih =>
      exact ⟨R1CS.IsDirectRecipe.of_affine start (all recipe (by simp)),
        ih (fun recipe member => all recipe (by simp [member])) _⟩

theorem counts (interface : Logical.Interface) (coordinate offset : Nat)
    (inputs : ∀ current, StateAffine (interface.initialState current)) :
    R1CS.totalFreshCount (flatConstraints (Lifecycle.PiRLC.v1_2.Sampler.opsAt interface coordinate offset)) = 144 ∧
      R1CS.totalRowCount (flatConstraints (Lifecycle.PiRLC.v1_2.Sampler.opsAt interface coordinate offset)) = 3071 := by
  have fresh : R1CS.totalFreshCount (flatConstraints (Lifecycle.PiRLC.v1_2.Sampler.opsAt interface coordinate offset)) = 144 := by
    simp only [Lifecycle.PiRLC.v1_2.Sampler.opsAt, flatConstraints, List.flatMap_cons, List.flatMap_nil, List.append_nil,
      R1CS.totalFreshCount_append, entry_fresh interface coordinate offset inputs,
      range_fresh, advance_fresh, words_fresh, Nat.add_zero]
  refine ⟨fresh, ?_⟩
  rw [R1CS.totalRowCount_eq_fresh_add_length, fresh, Lifecycle.PiRLC.v1_2.Sampler.rowCount_eq, Lifecycle.PiRLC.v1_2.Sampler.counts.2]

/-- The scalar's external expressions are its incoming transcript state. -/
structure InputsAffine (interface : Logical.Interface) (offset : Nat) : Prop where
  initialState : StateAffine (interface.initialState offset)

def logicalConstraints (interface : Logical.Interface) (coordinate offset : Nat) : List Expr :=
  flatConstraints (Lifecycle.PiRLC.v1_2.Sampler.opsAt interface coordinate offset)

theorem totalFreshCount_eq (interface : Logical.Interface) (coordinate offset : Nat)
    (inputs : ∀ current, InputsAffine interface current) :
    R1CS.totalFreshCount (logicalConstraints interface coordinate offset) = 144 :=
  (counts interface coordinate offset (fun current => (inputs current).initialState)).1

theorem totalRowCount_eq (interface : Logical.Interface) (coordinate offset : Nat)
    (inputs : ∀ current, InputsAffine interface current) :
    R1CS.totalRowCount (logicalConstraints interface coordinate offset) = 3071 :=
  (counts interface coordinate offset (fun current => (inputs current).initialState)).2

def footprint (interface : Logical.Interface) (coordinate : Nat)
    (inputs : ∀ current, InputsAffine interface current) :
    R1CS.CircuitFootprint (Logical.circuit interface coordinate) where
  freshColumnCount := fun _ => 144
  physicalRowCount := fun _ => 3071
  freshColumnCount_eq := fun offset => totalFreshCount_eq interface coordinate offset inputs
  physicalRowCount_eq := fun offset => totalRowCount_eq interface coordinate offset inputs

def plan (interface : Logical.Interface) (coordinate offset : Nat) : R1CS.LoweringPlan where
  constraints := logicalConstraints interface coordinate offset
  firstFresh := offset + Logical.logicalPrivateCount

def PhysicalHolds (interface : Logical.Interface) (coordinate offset : Nat) (env : Env) : Prop :=
  R1CS.RowsHold env (plan interface coordinate offset).rows

theorem physical_implies_relation (interface : Logical.Interface) (coordinate offset : Nat)
    (env : Env) (inputs : Logical.Assumptions interface offset)
    (rows : PhysicalHolds interface coordinate offset env) :
    Logical.SpecHolds interface coordinate offset env := by
  exact Lifecycle.PiRLC.v1_2.Sampler.soundness interface coordinate env offset inputs
    (holdsFlat_implies_holds env _ (R1CS.LoweringPlan.sound (plan interface coordinate offset) env rows))

end NightstreamFPrime.Layout.PiRLC.v1_2.Sampler
