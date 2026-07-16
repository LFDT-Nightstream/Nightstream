import Nightstream.SuperNeo.Concrete.Relation

/-!
Contract: model-level ABI for packing the exact outer fresh CCS assignment.

Owns: identity-ordered flat coordinates, the production `D = 54` block/lane
map, zero padding in the final ring block, and public-input prefix projection.

Does not own: Rust constructor conformance, fixed-selector materialization,
Ajtai binding, Π_CCS soundness, or permission to remove encoding rows.

Emits constraints: no.

Authority boundary: these are definitional properties of the assignment
consumed by `CCS.Holds`. A constructor or generated digest is not authority;
the accepted fresh CCS opening must bind this same assignment.

| ABI branch | Mathematical obligation | Main result | Tier |
|---|---|---|---|
| flat coordinate | `i = (i / 54) * 54 + i % 54` | `packedCoeff_div_mod` | model-level |
| ring block | block `b` stores `z[b*54 + rho]` | `packAssignment_block` | model-level |
| exact coordinate | no permutation between `z[i]` and packed coefficient | `packAssignment_coordinate` | model-level |
| final padding | every packed cell beyond `z.length` is zero | `packAssignment_padding_zero` | model-level |
| fixed-width injectivity | equal-length assignments with equal packed images are equal | `packAssignment_injective_of_length_eq` | model-level |
| public prefix | public input is exactly `z.take publicWidth` | `projectPublicInput_eq_take` | model-level |
-/

namespace Nightstream.Implementation.R1CS.FreshAssignmentPacking

open Nightstream.SuperNeo.Concrete

/-- Division/modulus spelling of the production block/lane packing. -/
theorem packedCoeff_div_mod (z : List F) (i : Nat) :
    packedCoeff z (i / ringDegree)
        ⟨i % ringDegree, Nat.mod_lt _ (by simp [ringDegree])⟩ =
      z.getD i 0 := by
  unfold packedCoeff
  congr 1
  simpa [Nat.mul_comm] using Nat.div_add_mod i ringDegree

/-- Every in-range block is the corresponding `packedCoeff` function. -/
theorem packAssignment_block (z : List F) (block : Nat)
    (blockLt : block < (packAssignment z).length) :
    (packAssignment z).getD block ringFZero =
      fun rho => packedCoeff z block rho := by
  simp only [packAssignment, List.length_map, List.length_range] at blockLt
  simp [packAssignment, blockLt]

/-- The outer assignment is packed at the same flat index, without a
permutation or second serialization. -/
theorem packAssignment_coordinate (z : List F) (i : Nat)
    (iLt : i < z.length) :
    (packAssignment z).getD (i / ringDegree) ringFZero
        ⟨i % ringDegree, Nat.mod_lt _ (by simp [ringDegree])⟩ =
      z.getD i 0 := by
  have blockLt : i / ringDegree < (packAssignment z).length := by
    simp only [packAssignment, List.length_map, List.length_range]
    simp [ringDegree] at iLt ⊢
    omega
  rw [packAssignment_block z (i / ringDegree) blockLt]
  exact packedCoeff_div_mod z i

/-- A valid packed cell whose flat index lies beyond the assignment is the
canonical zero padding of the final block. -/
theorem packAssignment_padding_zero (z : List F) (block : Nat)
    (rho : Fin ringDegree)
    (blockLt : block < (packAssignment z).length)
    (padding : z.length ≤ block * ringDegree + rho.val) :
    (packAssignment z).getD block ringFZero rho = 0 := by
  rw [packAssignment_block z block blockLt]
  unfold packedCoeff List.getD
  change z[block * ringDegree + rho.val]?.getD 0 = 0
  rw [List.getElem?_eq_none padding]
  rfl

/-- Packing is injective once the verifier-owned scalar assignment width is
fixed. The width premise is load-bearing: trailing scalar zeros occupy the
same padded ring block and therefore cannot be recovered from the packed value
alone. -/
theorem packAssignment_injective_of_length_eq
    {left right : List F}
    (sameLength : left.length = right.length)
    (samePacked : packAssignment left = packAssignment right) :
    left = right := by
  apply List.ext_getElem sameLength
  intro i leftLt rightLt
  have coordinateEquality :
      (packAssignment left).getD (i / ringDegree) ringFZero
          ⟨i % ringDegree, Nat.mod_lt _ (by simp [ringDegree])⟩ =
        (packAssignment right).getD (i / ringDegree) ringFZero
          ⟨i % ringDegree, Nat.mod_lt _ (by simp [ringDegree])⟩ := by
    rw [samePacked]
  rw [packAssignment_coordinate left i leftLt,
    packAssignment_coordinate right i rightLt] at coordinateEquality
  simpa [List.getD_eq_getElem?_getD, leftLt, rightLt] using coordinateEquality

/-- The concrete public projection owns exactly the flat assignment prefix. -/
theorem projectPublicInput_eq_take (publicWidth : Nat) (z : List F) :
    projectPublicInput publicWidth z = z.take publicWidth := by
  rfl

end Nightstream.Implementation.R1CS.FreshAssignmentPacking
