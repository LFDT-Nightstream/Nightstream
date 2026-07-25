import Nightstream.Implementation.Lowering.Goldilocks.NormalFormComposition

/-!
Contract: the finite local rewrite classes selected for the fixed-one step and
terminal physical encodings.

Owns:
- one two-member branch-join class for every physically joined running
  coordinate in the step program;
- the two retained gated assertions in the step program;
- the base endpoint and three recursive gated assertions in the terminal
  program;
- exact selected local costs and compositional minimum theorems in the order
  `(recurring rows, committed columns, public columns, auxiliary columns)`.

Does not own: call recipes, input/output allocation, branch-activation rows,
source-path alignment, a physical receipt program, Rust or R1CS behavior, or
global minimality over other arithmetizations.  Those resources are identical
across this finite class and are represented only by the common fixed-cost
prefix in the minimum theorems.

The admitted rewrites are exactly:
1. one selected-mux row versus two selector-gated equality rows for each
   nonempty branch-join coordinate; and
2. one direct gated-assertion row versus a materialized residual, zero pin,
   and one auxiliary column for each retained assertion.

Emits constraints: no.  The candidates contain exact row/allocation recipes,
but this module only selects and compares them.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.NormalForm

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.PrimitiveNormalForm
open Nightstream.Implementation.Lowering.Goldilocks.NormalFormComposition

/-! ## Step rewrite class -/

/-- Physical specifications for precisely the variable local sites in the
fixed-one step encoding.

`joinCoordinates` is ordered exactly as the selected running-value codec.  A
later source-alignment certificate proves that it is the complete coordinate
list of the unique nonempty step join. -/
structure StepSpecifications where
  joinCoordinates : List BranchJoin.Specification
  baseEndpoint : GatedAssertion.Specification
  recursivePriorLink : GatedAssertion.Specification

/-- One independent two-member class per joined coordinate. -/
def stepJoinClasses (specifications : StepSpecifications) :
    List InstructionClass :=
  specifications.joinCoordinates.map
    ConcretePrimitives.branchJoinClass

/-- The complete fixed-one step rewrite class.  Calls, literal construction,
branch activation, inputs, and result columns are deliberately absent because
they do not vary inside this class. -/
def stepClasses (specifications : StepSpecifications) :
    List InstructionClass :=
  stepJoinClasses specifications ++
    [ConcretePrimitives.gatedAssertionClass
        specifications.baseEndpoint,
      ConcretePrimitives.gatedAssertionClass
        specifications.recursivePriorLink]

private theorem joinCanonicalCost
    (specifications : List BranchJoin.Specification) :
    totalCost
        (specifications.map ConcretePrimitives.branchJoinClass)
        (canonicalSelection
          (specifications.map ConcretePrimitives.branchJoinClass)) =
      ⟨specifications.length, 0, 0, 0⟩ := by
  induction specifications with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, canonicalSelection, totalCost]
      rw [inductionHypothesis]
      change
        BranchJoin.Candidate.cost head (BranchJoin.canonical head) +
            ⟨tail.length, 0, 0, 0⟩ =
          ⟨(head :: tail).length, 0, 0, 0⟩
      rw [BranchJoin.canonical_cost]
      change
        ({ recurringRows := 1 + tail.length
           committedColumns := 0
           publicColumns := 0
           auxiliaryColumns := 0 } : Cost) =
          { recurringRows := tail.length + 1
            committedColumns := 0
            publicColumns := 0
            auxiliaryColumns := 0 }
      rw [Nat.add_comm 1 tail.length]

/-- The selected step-local cost is one mux row per joined coordinate and one
direct row for each of the two retained assertions. -/
theorem stepCanonicalLocalCost
    (specifications : StepSpecifications) :
    totalCost
        (stepClasses specifications)
        (canonicalSelection (stepClasses specifications)) =
      ⟨specifications.joinCoordinates.length + 2, 0, 0, 0⟩ := by
  unfold stepClasses stepJoinClasses
  induction specifications.joinCoordinates with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.cons_append, canonicalSelection,
        totalCost]
      rw [show
        totalCost
            (tail.map ConcretePrimitives.branchJoinClass ++
              [ConcretePrimitives.gatedAssertionClass
                  specifications.baseEndpoint,
                ConcretePrimitives.gatedAssertionClass
                  specifications.recursivePriorLink])
            (canonicalSelection
              (tail.map ConcretePrimitives.branchJoinClass ++
                [ConcretePrimitives.gatedAssertionClass
                    specifications.baseEndpoint,
                  ConcretePrimitives.gatedAssertionClass
                    specifications.recursivePriorLink])) =
          ⟨tail.length + 2, 0, 0, 0⟩ by
        exact inductionHypothesis]
      change
        BranchJoin.Candidate.cost head (BranchJoin.canonical head) +
            ⟨tail.length + 2, 0, 0, 0⟩ =
          ⟨(head :: tail).length + 2, 0, 0, 0⟩
      rw [BranchJoin.canonical_cost]
      change
        ({ recurringRows := 1 + (tail.length + 2)
           committedColumns := 0
           publicColumns := 0
           auxiliaryColumns := 0 } : Cost) =
          { recurringRows := (tail.length + 1) + 2
            committedColumns := 0
            publicColumns := 0
            auxiliaryColumns := 0 }
      congr 1
      omega

/-- Every selected step candidate implements its independently specified
local relation. -/
theorem stepCanonicalImplements
    (specifications : StepSpecifications) :
    ImplementsAll
      (stepClasses specifications)
      (canonicalSelection (stepClasses specifications)) :=
  canonicalSelection_implementsAll (stepClasses specifications)

/-- The selected local step encoding is least in the exact finite
coordinate-and-assertion rewrite class. -/
theorem stepCanonicalMinimum
    (specifications : StepSpecifications)
    (selection : Selection (stepClasses specifications))
    (admissible : Admissible (stepClasses specifications) selection) :
    Cost.LexLe
      (totalCost
        (stepClasses specifications)
        (canonicalSelection (stepClasses specifications)))
      (totalCost (stepClasses specifications) selection) :=
  canonicalSelection_minimum
    (stepClasses specifications) selection admissible

/-- Adding the identical cost of all fixed step instructions preserves the
finite-class minimum. -/
theorem stepCanonicalMinimumWithFixedCost
    (specifications : StepSpecifications)
    (fixedCost : Cost)
    (selection : Selection (stepClasses specifications))
    (admissible : Admissible (stepClasses specifications) selection) :
    Cost.LexLe
      (fixedCost +
        totalCost
          (stepClasses specifications)
          (canonicalSelection (stepClasses specifications)))
      (fixedCost +
        totalCost (stepClasses specifications) selection) :=
  lexLe_add_same_prefix fixedCost
    (stepCanonicalMinimum specifications selection admissible)

/-! ## Terminal rewrite class -/

/-- Physical specifications for the four and only four retained terminal
assertions.  The terminal join schema is empty, so there is no joined
coordinate and therefore no branch-join rewrite position. -/
structure TerminalSpecifications where
  baseEndpoint : GatedAssertion.Specification
  recursivePriorLink : GatedAssertion.Specification
  runningRelation : GatedAssertion.Specification
  freshRelation : GatedAssertion.Specification

/-- The complete fixed-one terminal rewrite class. -/
def terminalClasses (specifications : TerminalSpecifications) :
    List InstructionClass :=
  [ConcretePrimitives.gatedAssertionClass
      specifications.baseEndpoint,
    ConcretePrimitives.gatedAssertionClass
      specifications.recursivePriorLink,
    ConcretePrimitives.gatedAssertionClass
      specifications.runningRelation,
    ConcretePrimitives.gatedAssertionClass
      specifications.freshRelation]

/-- Canonical terminal selection uses the direct assertion recipe at all four
positions. -/
theorem terminalCanonicalSelection
    (specifications : TerminalSpecifications) :
    canonicalSelection (terminalClasses specifications) =
      (GatedAssertion.Candidate.direct,
        (GatedAssertion.Candidate.direct,
          (GatedAssertion.Candidate.direct,
            (GatedAssertion.Candidate.direct, PUnit.unit)))) :=
  rfl

/-- The selected terminal-local cost is exactly four rows and no
candidate-specific columns. -/
theorem terminalCanonicalLocalCost
    (specifications : TerminalSpecifications) :
    totalCost
        (terminalClasses specifications)
        (canonicalSelection (terminalClasses specifications)) =
      ⟨4, 0, 0, 0⟩ :=
  rfl

/-- Every selected terminal candidate implements its independently specified
local relation. -/
theorem terminalCanonicalImplements
    (specifications : TerminalSpecifications) :
    ImplementsAll
      (terminalClasses specifications)
      (canonicalSelection (terminalClasses specifications)) :=
  canonicalSelection_implementsAll (terminalClasses specifications)

/-- The selected local terminal encoding is least in the exact four-position
finite rewrite class. -/
theorem terminalCanonicalMinimum
    (specifications : TerminalSpecifications)
    (selection : Selection (terminalClasses specifications))
    (admissible : Admissible (terminalClasses specifications) selection) :
    Cost.LexLe
      (totalCost
        (terminalClasses specifications)
        (canonicalSelection (terminalClasses specifications)))
      (totalCost (terminalClasses specifications) selection) :=
  canonicalSelection_minimum
    (terminalClasses specifications) selection admissible

/-- Adding the identical cost of all fixed terminal instructions preserves
the finite-class minimum. -/
theorem terminalCanonicalMinimumWithFixedCost
    (specifications : TerminalSpecifications)
    (fixedCost : Cost)
    (selection : Selection (terminalClasses specifications))
    (admissible : Admissible (terminalClasses specifications) selection) :
    Cost.LexLe
      (fixedCost +
        totalCost
          (terminalClasses specifications)
          (canonicalSelection (terminalClasses specifications)))
      (fixedCost +
        totalCost (terminalClasses specifications) selection) :=
  lexLe_add_same_prefix fixedCost
    (terminalCanonicalMinimum specifications selection admissible)

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.NormalForm
