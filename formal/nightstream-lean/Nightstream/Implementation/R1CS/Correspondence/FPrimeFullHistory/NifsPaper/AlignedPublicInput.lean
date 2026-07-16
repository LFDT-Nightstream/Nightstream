import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.FreshAssignmentPacking
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PublicInputBoundary

/-!
Ring-aligned public boundary proposed for the fixed F' SuperNeo relation.

Owns: the semantic insertion of thirteen verifier-fixed zero coefficients
between the 257 logical F' public fields and the private witness; preservation
of the logical public projection and private suffix; exact recovery of the
legacy assignment; and the paper-required `270 = 54 * 5` dimension fact.

Does not own: a transformed CCS matrix, equivalence of CCS satisfaction,
Ajtai-key migration, Rust column insertion, generated R1CS rows, or permission
to change/remove constraints. Those are later refinement obligations.

Emits constraints: no.

Authority boundary: the thirteen coefficients are definitionally zero in the
semantic assignment. A prover-supplied zero-looking digest or packed carrier
does not establish this predicate.

| Branch | Mathematical obligation | Result | Assurance tier |
|---|---|---|---|
| dimensions | `257 + 13 = 270 = 54 * 5` | `aligned_dimensions` | kernel theorem |
| insertion | zeros occur before the private witness | `insertPublicPadding` | definition |
| logical public view | first 257 values are unchanged | `logicalProjection_preserved` | kernel theorem |
| aligned public view | first 270 values are logical input plus 13 zeros | `alignedProjection_exact` | kernel theorem |
| private witness | suffix after 270 equals old suffix after 257 | `privateSuffix_preserved` | kernel theorem |
| reversibility | deleting the fixed padding recovers the old assignment | `erase_insertPublicPadding` | kernel theorem |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedPublicInput

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FreshAssignmentPacking
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PublicInputBoundary

def logicalPublicWidth : Nat := 257

def paddingWidth : Nat := 13

def alignedPublicWidth : Nat := 270

def publicPadding : List F := List.replicate paddingWidth 0

/-- Insert the verifier-fixed public padding before the private witness. -/
def insertPublicPadding (assignment : List F) : List F :=
  assignment.take logicalPublicWidth ++ publicPadding ++
    assignment.drop logicalPublicWidth

/-- The complete aligned paper public input. -/
def alignedPublicInput (assignment : List F) : List F :=
  assignment.take logicalPublicWidth ++ publicPadding

/-- Delete the aligned public padding while preserving both owned regions. -/
def erasePublicPadding (assignment : List F) : List F :=
  assignment.take logicalPublicWidth ++ assignment.drop alignedPublicWidth

theorem aligned_dimensions :
    logicalPublicWidth = 257 ∧ paddingWidth = 13 ∧
      alignedPublicWidth = 270 ∧
      alignedPublicWidth = ringDegree * 5 := by
  decide

/-- Explicit bridge from the independent aligned dimensions to the currently
measured production profile. This equality does not authorize the production
binding; it only fixes the dimensions being repaired. -/
theorem dimensions_matchProduction :
    logicalPublicWidth = productionPublicWidth ∧
      paddingWidth = productionExtraCoefficients ∧
      alignedPublicWidth = productionPackedWidth := by
  decide

theorem publicPadding_length : publicPadding.length = paddingWidth := by
  simp [publicPadding]

theorem alignedPublicInput_length (assignment : List F)
    (hasLogicalPublic : logicalPublicWidth ≤ assignment.length) :
    (alignedPublicInput assignment).length = alignedPublicWidth := by
  have has257 : 257 ≤ assignment.length := by
    simpa [logicalPublicWidth] using hasLogicalPublic
  simp [alignedPublicInput, publicPadding, List.length_take,
    Nat.min_eq_left has257, alignedPublicWidth, logicalPublicWidth,
    paddingWidth]

theorem insertPublicPadding_length (assignment : List F)
    (hasLogicalPublic : logicalPublicWidth ≤ assignment.length) :
    (insertPublicPadding assignment).length = assignment.length + paddingWidth := by
  simp [insertPublicPadding, publicPadding, List.length_take,
    Nat.min_eq_left hasLogicalPublic, List.length_drop, paddingWidth]
  omega

/-- The existing 257-field F' interface is unchanged by the aligned semantic
insertion. -/
theorem logicalProjection_preserved (assignment : List F)
    (hasLogicalPublic : logicalPublicWidth ≤ assignment.length) :
    projectPublicInput logicalPublicWidth (insertPublicPadding assignment) =
      projectPublicInput logicalPublicWidth assignment := by
  simp [projectPublicInput_eq_take, insertPublicPadding, publicPadding,
    hasLogicalPublic]

/-- The paper-visible public input is exactly the logical F' input followed by
thirteen authoritative zeros. -/
theorem alignedProjection_exact (assignment : List F)
    (hasLogicalPublic : logicalPublicWidth ≤ assignment.length) :
    projectPublicInput alignedPublicWidth (insertPublicPadding assignment) =
      alignedPublicInput assignment := by
  change (alignedPublicInput assignment ++
    assignment.drop logicalPublicWidth).take alignedPublicWidth =
      alignedPublicInput assignment
  rw [← alignedPublicInput_length assignment hasLogicalPublic]
  exact List.take_append_length

/-- Inserting the aligned public block shifts but does not alter the private
witness. -/
theorem privateSuffix_preserved (assignment : List F)
    (hasLogicalPublic : logicalPublicWidth ≤ assignment.length) :
    (insertPublicPadding assignment).drop alignedPublicWidth =
      assignment.drop logicalPublicWidth := by
  change (alignedPublicInput assignment ++
    assignment.drop logicalPublicWidth).drop alignedPublicWidth =
      assignment.drop logicalPublicWidth
  rw [← alignedPublicInput_length assignment hasLogicalPublic]
  exact List.drop_append_length

/-- The padding adapter is lossless at the semantic boundary. -/
theorem erase_insertPublicPadding (assignment : List F)
    (hasLogicalPublic : logicalPublicWidth ≤ assignment.length) :
    erasePublicPadding (insertPublicPadding assignment) = assignment := by
  have head := logicalProjection_preserved assignment hasLogicalPublic
  change (insertPublicPadding assignment).take logicalPublicWidth =
    assignment.take logicalPublicWidth at head
  unfold erasePublicPadding
  rw [head, privateSuffix_preserved assignment hasLogicalPublic]
  exact List.take_append_drop logicalPublicWidth assignment

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedPublicInput
