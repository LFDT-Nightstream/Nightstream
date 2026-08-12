import Nightstream.Implementation.NebulaV2.Production.FPrime.Recursive.RelationRowsSoundFor

/-!
Contract: hostile models for incomplete claim-opening authority.

These examples show three deterministic defects that a sound F-prime theorem
must exclude. First, a commitment opening and relation satisfaction can use
different assignments. Second, an opening for one source claim says nothing
about a different receipt claim unless exact claim equality is proved. Third,
an accepted terminal assignment can satisfy a program that is not the program
selected by the verifier context.

The production lifetime excludes both defects with one `CCS.Holds` witness
per produced claim and exact receipt-to-source-claim equality.

The field-native claim now excludes profile and application-statement
sidecars. The prior deterministic substitution countermodel therefore cannot
be stated for the production claim type.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningCountermodels

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveRelationRowsSoundFor
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ProductionBatchedFPrime
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- An intentionally weak authority shape. It permits one assignment for the
commitment opening and another assignment for relation satisfaction. -/
structure DetachedAuthority
    (Assignment Bundle : Type)
    (commit : Assignment → Bundle)
    (satisfies : Assignment → Prop)
    (bundle : Bundle) where
  openingAssignment : Assignment
  opening : commit openingAssignment = bundle
  relationAssignment : Assignment
  relation : satisfies relationAssignment

def identityCommit (assignment : Bool) : Bool := assignment

def acceptsOnlyTrue (assignment : Bool) : Prop := assignment = true

/-- The weak shape is satisfiable: `false` opens the bundle while `true`
satisfies the relation. -/
def detachedAuthority :
    DetachedAuthority Bool Bool identityCommit acceptsOnlyTrue false where
  openingAssignment := false
  opening := rfl
  relationAssignment := true
  relation := rfl

theorem detached_authority_exists :
    Nonempty
      (DetachedAuthority Bool Bool identityCommit acceptsOnlyTrue false) :=
  ⟨detachedAuthority⟩

/-- No single assignment both opens that bundle and satisfies that relation.
Thus two detached witnesses cannot replace one `CCS.Holds` witness. -/
theorem detached_authority_has_no_common_witness :
    ¬ ∃ assignment : Bool,
      identityCommit assignment = false ∧ acceptsOnlyTrue assignment := by
  intro common
  rcases common with ⟨assignment, opening, relation⟩
  cases assignment
  · contradiction
  · contradiction

def claimBundle (claim : Bool) : Bool := claim

/-- An opening for a source claim does not open a different receipt claim.
The exact receipt-to-source equality in `ReceiptOpening` is necessary. -/
theorem source_opening_does_not_open_different_receipt :
    ∃ source receipt assignment : Bool,
      source ≠ receipt ∧
      identityCommit assignment = claimBundle source ∧
      identityCommit assignment ≠ claimBundle receipt := by
  exact ⟨false, true, false, by decide, rfl, by decide⟩

def satisfiesProgram (program assignment : Bool) : Prop :=
  assignment = program

/-- A terminal node that stores its own program can be valid for that program
while it is invalid for the program selected by the verifier context. -/
theorem floating_terminal_program_does_not_imply_fixed_program :
    ∃ acceptedProgram fixedProgram assignment : Bool,
      acceptedProgram ≠ fixedProgram ∧
      satisfiesProgram acceptedProgram assignment ∧
      ¬ satisfiesProgram fixedProgram assignment := by
  exact ⟨false, true, false, by decide, rfl, by
    intro impossible
    cases impossible⟩

end Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningCountermodels
