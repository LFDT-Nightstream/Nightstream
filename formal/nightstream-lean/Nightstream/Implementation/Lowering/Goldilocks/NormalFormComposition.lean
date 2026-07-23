import Nightstream.Implementation.Lowering.Goldilocks.PrimitiveNormalForm

/-!
Contract: compositional finite normal-form selection for an ordered list of
independent Goldilocks instruction classes.

Owns:
- heterogeneous packaging of one independently specified finite candidate
  class;
- pointwise admissibility and structural conjunction of local `Implements`
  relations;
- additive whole-selection cost;
- monotonicity of the fixed lexicographic cost order under a common prefix or
  suffix;
- the theorem that pointwise canonical selection is no more expensive than
  every pointwise member;
- a concrete two-instruction instantiation using the branch-join and
  gated-assertion classes.

Does not own: a caller-provided whole-program proposition, instruction
dependency analysis, call lowering, generated artifacts, Rust behavior, or
global minimality outside each explicitly supplied finite candidate class.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.Goldilocks.NormalFormComposition

open Nightstream.Implementation.Lowering.Typed

universe u v

/-! ## Additive monotonicity of the fixed cost order -/

theorem lexLe_add_same_suffix
    {left right : Cost}
    (suffix : Cost)
    (lessOrEqual : Cost.LexLe left right) :
    Cost.LexLe (left + suffix) (right + suffix) := by
  unfold Cost.LexLe at lessOrEqual ⊢
  simp only [Cost.add_recurringRows, Cost.add_committedColumns,
    Cost.add_publicColumns, Cost.add_auxiliaryColumns]
  omega

theorem lexLe_add_same_prefix
    (commonPrefix : Cost)
    {left right : Cost}
    (lessOrEqual : Cost.LexLe left right) :
    Cost.LexLe (commonPrefix + left) (commonPrefix + right) := by
  unfold Cost.LexLe at lessOrEqual ⊢
  simp only [Cost.add_recurringRows, Cost.add_committedColumns,
    Cost.add_publicColumns, Cost.add_auxiliaryColumns]
  omega

theorem lexLe_add
    {leftHead rightHead leftTail rightTail : Cost}
    (headLessOrEqual : Cost.LexLe leftHead rightHead)
    (tailLessOrEqual : Cost.LexLe leftTail rightTail) :
    Cost.LexLe
      (leftHead + leftTail)
      (rightHead + rightTail) := by
  exact NormalForm.lexLe_trans
    (lexLe_add_same_suffix leftTail headLessOrEqual)
    (lexLe_add_same_prefix rightHead tailLessOrEqual)

/-! ## Heterogeneous finite instruction classes -/

/-- One independently specified finite instruction-candidate class.

The local semantics and candidate proof are stored before the cost function;
cost therefore cannot serve as the semantic specification. -/
structure InstructionClass where
  Candidate : Type u
  Specification : Type v
  Implements : Candidate -> Specification -> Prop
  specification : Specification
  candidates :
    NormalForm.FiniteCandidates
      Candidate Specification Implements specification
  cost : Candidate -> Cost

/-- A heterogeneous pointwise selection for an ordered instruction list.

This type reduces to nested products, preserving the candidate type belonging
to each instruction position. -/
def Selection : List (InstructionClass.{u, v}) -> Type u
  | [] => PUnit
  | instruction :: tail =>
      instruction.Candidate × Selection tail

/-- Every selected candidate belongs to the finite class declared at the same
instruction position. -/
def Admissible :
    (classes : List (InstructionClass.{u, v})) ->
      Selection classes -> Prop
  | [], _ => True
  | instruction :: tail, selection =>
      selection.1 ∈ instruction.candidates.members ∧
        Admissible tail selection.2

/-- Whole-selection semantics is exactly the structural conjunction of the
independent local `Implements` relations.  There is no whole-program
proposition supplied by a caller. -/
def ImplementsAll :
    (classes : List (InstructionClass.{u, v})) ->
      Selection classes -> Prop
  | [], _ => True
  | instruction :: tail, selection =>
      instruction.Implements
          selection.1 instruction.specification ∧
        ImplementsAll tail selection.2

/-- Whole-selection cost is the ordered additive fold of local costs. -/
def totalCost :
    (classes : List (InstructionClass.{u, v})) ->
      Selection classes -> Cost
  | [], _ => Cost.zero
  | instruction :: tail, selection =>
      instruction.cost selection.1 +
        totalCost tail selection.2

/-- Choose each instruction's local canonical member independently. -/
def canonicalSelection :
    (classes : List (InstructionClass.{u, v})) ->
      Selection classes
  | [] => PUnit.unit
  | instruction :: tail =>
      (instruction.candidates.canonical instruction.cost,
        canonicalSelection tail)

theorem canonicalSelection_admissible
    (classes : List (InstructionClass.{u, v})) :
    Admissible classes (canonicalSelection classes) := by
  induction classes with
  | nil =>
      exact True.intro
  | cons instruction tail inductionHypothesis =>
      exact ⟨
        instruction.candidates.canonical_mem instruction.cost,
        inductionHypothesis⟩

/-- Every pointwise member implements the structural conjunction of its local
specifications. -/
theorem admissible_implies_implementsAll
    (classes : List (InstructionClass.{u, v}))
    (selection : Selection classes)
    (admissible : Admissible classes selection) :
    ImplementsAll classes selection := by
  induction classes with
  | nil =>
      exact True.intro
  | cons instruction tail inductionHypothesis =>
      exact ⟨
        instruction.candidates.correct selection.1 admissible.1,
        inductionHypothesis selection.2 admissible.2⟩

theorem canonicalSelection_implementsAll
    (classes : List (InstructionClass.{u, v})) :
    ImplementsAll classes (canonicalSelection classes) :=
  admissible_implies_implementsAll
    classes
    (canonicalSelection classes)
    (canonicalSelection_admissible classes)

/-- Pointwise finite-list minima compose under additive cost.

The conclusion is relative exactly to the ordered supplied classes and the
pointwise membership proof.  It is not a global arithmetization minimum. -/
theorem canonicalSelection_minimum
    (classes : List (InstructionClass.{u, v}))
    (selection : Selection classes)
    (admissible : Admissible classes selection) :
    Cost.LexLe
      (totalCost classes (canonicalSelection classes))
      (totalCost classes selection) := by
  induction classes with
  | nil =>
      exact NormalForm.lexLe_refl Cost.zero
  | cons instruction tail inductionHypothesis =>
      apply lexLe_add
      · exact instruction.candidates.canonical_minimum
          instruction.cost admissible.1
      · exact inductionHypothesis selection.2 admissible.2

/-! ## Existing concrete primitive classes -/

namespace ConcretePrimitives

open PrimitiveNormalForm

/-- Package the already proved branch-join candidate class without changing
its specification, semantics, candidates, or structurally folded cost. -/
def branchJoinClass
    (specification : BranchJoin.Specification) :
    InstructionClass where
  Candidate := BranchJoin.Candidate
  Specification := BranchJoin.Specification
  Implements := BranchJoin.Implements
  specification := specification
  candidates := BranchJoin.candidates specification
  cost := BranchJoin.Candidate.cost specification

/-- Package the already proved gated-assertion candidate class unchanged. -/
def gatedAssertionClass
    (specification : GatedAssertion.Specification) :
    InstructionClass where
  Candidate := GatedAssertion.Candidate
  Specification := GatedAssertion.Specification
  Implements := GatedAssertion.Implements
  specification := specification
  candidates := GatedAssertion.candidates specification
  cost := GatedAssertion.Candidate.cost specification

/-- A real two-position normal-form profile, parameterized by arbitrary valid
source specifications rather than artificial sample assignments. -/
def branchJoinThenAssertion
    (branchSpecification : BranchJoin.Specification)
    (assertionSpecification : GatedAssertion.Specification) :
    List InstructionClass :=
  [branchJoinClass branchSpecification,
    gatedAssertionClass assertionSpecification]

theorem branchJoinThenAssertion_canonical_eq
    (branchSpecification : BranchJoin.Specification)
    (assertionSpecification : GatedAssertion.Specification) :
    canonicalSelection
        (branchJoinThenAssertion
          branchSpecification assertionSpecification) =
      (BranchJoin.Candidate.selectedMux,
        (GatedAssertion.Candidate.direct, PUnit.unit)) :=
  rfl

/-- Exact cost is definitionally folded from the two selected concrete row and
allocation lists: one mux row plus one direct assertion row. -/
theorem branchJoinThenAssertion_canonical_cost
    (branchSpecification : BranchJoin.Specification)
    (assertionSpecification : GatedAssertion.Specification) :
    totalCost
        (branchJoinThenAssertion
          branchSpecification assertionSpecification)
        (canonicalSelection
          (branchJoinThenAssertion
            branchSpecification assertionSpecification)) =
      ⟨2, 0, 0, 0⟩ :=
  rfl

/-- The instantiated whole semantics exposes the two independent local
relations as a conjunction. -/
theorem branchJoinThenAssertion_canonical_implements
    (branchSpecification : BranchJoin.Specification)
    (assertionSpecification : GatedAssertion.Specification) :
    BranchJoin.Implements .selectedMux branchSpecification ∧
      GatedAssertion.Implements .direct assertionSpecification := by
  have composed :=
    canonicalSelection_implementsAll
      (branchJoinThenAssertion
        branchSpecification assertionSpecification)
  exact ⟨composed.1, composed.2.1⟩

/-- Concrete specialization of the compositional minimum theorem. -/
theorem branchJoinThenAssertion_minimum
    (branchSpecification : BranchJoin.Specification)
    (assertionSpecification : GatedAssertion.Specification)
    (selection :
      Selection
        (branchJoinThenAssertion
          branchSpecification assertionSpecification))
    (admissible :
      Admissible
        (branchJoinThenAssertion
          branchSpecification assertionSpecification)
        selection) :
    Cost.LexLe
      (totalCost
        (branchJoinThenAssertion
          branchSpecification assertionSpecification)
        (canonicalSelection
          (branchJoinThenAssertion
            branchSpecification assertionSpecification)))
      (totalCost
        (branchJoinThenAssertion
          branchSpecification assertionSpecification)
        selection) :=
  canonicalSelection_minimum
    (branchJoinThenAssertion
      branchSpecification assertionSpecification)
    selection admissible

end ConcretePrimitives

end Nightstream.Implementation.Lowering.Goldilocks.NormalFormComposition
