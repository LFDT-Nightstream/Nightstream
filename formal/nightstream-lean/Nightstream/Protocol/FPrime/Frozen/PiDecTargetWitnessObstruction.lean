import Nightstream.SuperNeo.Folding.PiDEC.PaperReduction

/-!
Kernel obstruction to deriving `Pi_DEC` target witnesses from public verifier
acceptance alone.

Owns: one fully typed, inhabited model satisfying the `Pi_DEC` algebra,
public-split, and evaluation-arity interfaces in which the Section-7.5 public
equations accept but no computed child belongs to the target `CE(b)` relation.

Does not own: a production relation, a commitment instantiation, an
extraction hardness statement, Fiat--Shamir, Rust, R1CS, artifacts,
minimality, or costs.

Emits constraints: no.

SuperNeo Appendix D.6 defines reduction success as public verifier acceptance
together with valid target child witnesses.  The zero-loss straight-line
extractor consumes that stronger event; it does not create child witnesses
from the public equations.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.Frozen.PiDecTargetWitnessObstruction

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding

/-- One-child parameters with a trivially valid Definition-14 inequality. -/
def params : GlobalParams where
  q := 3
  b := 2
  k := 1
  maxFresh := 0
  expansionT := 0
  rlc_bound := by decide

def firstChild : Fin params.k := ⟨0, by decide⟩

/-- All carriers are inhabited.  The target relation is empty solely because
`normBounded` is false, not because the assignment type has no inhabitants. -/
def semantics : RelationSemantics Unit Unit Unit Unit Unit Unit where
  commit := fun _ => ()
  projectPublicInput := fun _ => ()
  normBounded := fun _ _ => False
  ccsSatisfied := fun _ _ => False
  evaluationPointValid := fun _ _ => True
  evaluations := fun _ _ _ => #[]

def algebra : PiDEC.Algebra Unit Unit Unit Unit Unit Unit semantics params where
  splitAssignment := fun _ _ => ()
  recomposeAssignment := fun _ => ()
  recomposeCommitment := fun _ => ()
  recomposePublicInput := fun _ => ()
  recomposeEvaluations := fun _ => #[]
  split_recompose := by
    intro assignment
    cases assignment
    rfl
  split_norm := by
    intro _ norm
    exact False.elim norm
  recompose_norm := by
    intro assignments norms
    exact False.elim (norms firstChild)
  commit_hom := by
    intro assignments
    rfl
  publicInput_hom := by
    intro assignments
    rfl
  evaluations_hom := by
    intro system point assignments
    rfl

def publicSplit : PiDEC.PaperVerifier.PublicInputSplit algebra where
  split := fun _ _ => ()
  recompose_split := by
    intro input
    cases input
    rfl
  split_project := by
    intro assignment child
    cases assignment
    rfl

def evaluationArity : PiDEC.PaperVerifier.EvaluationArity semantics where
  count := fun _ => 0
  evaluations_size := by
    intro system assignment point
    rfl

def context : PiDEC.PaperReduction.Context Unit Unit Unit Unit Unit Unit where
  semantics := semantics
  params := params
  algebra := algebra
  publicSplit := publicSplit
  evaluationArity := evaluationArity
  kPositive := by decide

def parent : CE.Instance Unit Unit Unit Unit Unit where
  constraintSystem := ()
  commitment := ()
  publicInput := ()
  point := ()
  evaluations := #[]
  stage := .combined

def attempt : PiDEC.PaperVerifier.Attempt Unit Unit Unit Unit Unit params where
  parent := parent
  messages := fun _ => {
    commitment := ()
    evaluations := #[]
  }

/-- Every public Section-7.5 check succeeds. -/
theorem attemptAccepted :
    PiDEC.PaperVerifier.Accepted algebra evaluationArity attempt := by
  refine ⟨rfl, rfl, ?_, rfl, rfl⟩
  intro child
  rfl

/-- No assignment opens any computed target child because the selected
relation's norm predicate is false. -/
theorem child_not_target
    (child : Fin params.k)
    (assignment : Unit) :
    ¬ CE.Holds semantics params
      (PiDEC.PaperVerifier.children publicSplit attempt child)
      assignment := by
  simp [CE.Holds, Opening.Holds, semantics]

/-- Consequently no family of target witnesses exists for the accepted
public output. -/
theorem noTargetWitnessFamily :
    ¬ exists childAssignments : Fin params.k -> Unit,
      forall child,
        CE.Holds semantics params
          (PiDEC.PaperVerifier.children publicSplit attempt child)
          (childAssignments child) := by
  rintro ⟨childAssignments, valid⟩
  exact child_not_target firstChild
    (childAssignments firstChild) (valid firstChild)

/-- Headline obstruction: public acceptance is strictly weaker than the D.6
target-success event consumed by the zero-loss reduction. -/
theorem accepted_without_piDec_target_witness :
    PiDEC.PaperVerifier.Accepted algebra evaluationArity attempt /\
      ¬ exists childAssignments : Fin params.k -> Unit,
        forall child,
          CE.Holds semantics params
            (PiDEC.PaperVerifier.children publicSplit attempt child)
            (childAssignments child) := by
  exact ⟨attemptAccepted, noTargetWitnessFamily⟩

end Nightstream.Protocol.FPrime.Frozen.PiDecTargetWitnessObstruction
