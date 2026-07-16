import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.FreshAssignmentPacking
import Nightstream.SuperNeo.Folding.PiDEC

/-!
Contract: concrete model-level Ajtai collision boundary for a Π_DEC parent.

Owns: specialization of the generic Π_DEC parent-opening collision to the
Nightstream `ajtaiCommit` map, and fixed-width distinctness of the packed ring
witnesses.

Does not own: Ajtai key shape or serialization, correspondence with the
canonical `formal/superneo-lean` opening relation, cyclotomic multiplication
refinement, norm translation to canonical coefficient arrays, an MSIS kernel
witness, Rust conformance, or permission to remove rows.

Emits constraints: no.

Authority boundary: `context.ajtaiKey` and `assignmentWidth` must come from the
verifier-owned relation shape. Packing equality is authority only after both
scalar openings are proved to have that same width.

| Result | Source obligation | Guarantee | Remaining obligation | Permits row removal? |
|---|---|---|---|---|
| `AjtaiOpeningCollision` | two concrete `bigB`-bounded openings of one `ajtaiCommit` value | fixed-width packed witnesses are distinct | canonical key/ring/norm/MSIS and Rust refinement | no |
| `parentOpeningBindingCollision_to_ajtaiOpeningCollision` | generic Π_DEC collision plus two verifier-owned width equalities | constructs the concrete collision above | derive widths from accepted production assignment shape | no |
-/

namespace Nightstream.Implementation.R1CS.PiDecAjtaiOpeningCollision

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.Implementation.R1CS.FreshAssignmentPacking

/-- A concrete Nightstream Ajtai opening collision with fixed scalar width.

This is deliberately not the canonical Ajtai binding game event and carries no
hardness conclusion. Its extra `packedDistinct` field is the representation
fact needed before a later bridge can construct two distinct canonical opening
witness arrays. -/
structure AjtaiOpeningCollision
    (context : Context)
    (params : GlobalParams)
    (commitment : Commitment)
    (assignmentWidth : Nat) where
  opening1 : Assignment
  opening2 : Assignment
  opening1Width : opening1.length = assignmentWidth
  opening2Width : opening2.length = assignmentWidth
  opening1Commits : ajtaiCommit context.ajtaiKey opening1 = commitment
  opening2Commits : ajtaiCommit context.ajtaiKey opening2 = commitment
  opening1Norm : normBounded params.bigB opening1
  opening2Norm : normBounded params.bigB opening2
  packedDistinct : packAssignment opening1 ≠ packAssignment opening2

/-- Specialize the generic Π_DEC collision to the concrete Nightstream Ajtai
map and derive distinct packed witnesses from verifier-owned equal widths. -/
theorem parentOpeningBindingCollision_to_ajtaiOpeningCollision
    {context : Context}
    {params : GlobalParams}
    {commitment : Commitment}
    {assignmentWidth : Nat}
    (collision : PiDEC.ParentOpeningBindingCollision
      (relationSemantics context) params commitment)
    (parentWidth : collision.parentOpening.length = assignmentWidth)
    (recomposedWidth : collision.recomposedOpening.length = assignmentWidth) :
    Nonempty (AjtaiOpeningCollision context params commitment assignmentWidth) := by
  refine ⟨{
    opening1 := collision.parentOpening
    opening2 := collision.recomposedOpening
    opening1Width := parentWidth
    opening2Width := recomposedWidth
    opening1Commits := collision.parentCommits
    opening2Commits := collision.recomposedCommits
    opening1Norm := collision.parentNorm
    opening2Norm := collision.recomposedNorm
    packedDistinct := ?_
  }⟩
  intro samePacked
  apply collision.different
  exact packAssignment_injective_of_length_eq
    (parentWidth.trans recomposedWidth.symm) samePacked

end Nightstream.Implementation.R1CS.PiDecAjtaiOpeningCollision
