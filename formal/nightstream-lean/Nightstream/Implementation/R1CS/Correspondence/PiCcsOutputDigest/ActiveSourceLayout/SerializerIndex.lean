import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveSourceLayout

/-!
Canonical serializer indices for active `Pi_CCS` `y_zcol` limbs.

Assurance tier: model-level representation correspondence.

Owns: the exact field-index formula for a `y_zcol` limb and the proof that
indexing `sourceRoles` at that position selects the corresponding typed role.

Does not own: Rust field indices, physical source columns, output authority,
SIS/Poseidon2, projection rows, transcript binding, costs, or row removal.

Emits constraints: no.

Authority boundary: the theorem derives indices only from the independent
shape-indexed serializer tree. A physical artifact must separately prove that
its raw serializer index equals this formula.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.output_message_hashes.digest.preimage.y_zcol.index.limb` | `c0 = 0`, `c1 = 1` within a two-limb lane | computed | `limbOffset` |
| `nifs.pi_ccs.output_message_hashes.digest.preimage.y_zcol.index.source` | source header `9` + complete `y_ring` region + width `1` + lane/limb offset | computed | `yZcolLimbSourceOffset` |
| `nifs.pi_ccs.output_message_hashes.digest.preimage.y_zcol.index.field` | outer header `8` + complete preceding source blocks + in-source offset | computed | `yZcolLimbFieldIndex` |
| `nifs.pi_ccs.output_message_hashes.digest.preimage.y_zcol.index.selection` | the formula is in bounds and selects exactly `SourceRole.yZcolLimb` | derived | `yZcolLimbFieldIndex_lt`, `sourceRoles_getElem_yZcolLimb` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-- Position of a Phi81 limb inside one two-field lane. -/
def limbOffset : Limb -> Nat
  | .c0 => 0
  | .c1 => 1

/-- Position of one `y_zcol` limb inside its source block:

`source header (9) + y_ring vectors + y_zcol width (1) + lane * 2 + limb`.
-/
def yZcolLimbSourceOffset
    (shape : SemanticShape)
    (lane : Fin ringDegree)
    (limb : Limb) : Nat :=
  9 + shape.matrixCount * Encoding.kVectorFieldCount ringDegree +
    1 + 2 * lane.val + limbOffset limb

/-- Position of one `y_zcol` limb inside the complete active serializer:

`outer header (8) + preceding fixed-width source blocks + source offset`.
-/
def yZcolLimbFieldIndex
    (shape : SemanticShape)
    (source : Fin shape.sourceCount)
    (lane : Fin ringDegree)
    (limb : Limb) : Nat :=
  8 + source.val * ActiveSemantics.sourceFieldCount shape.matrixCount +
    yZcolLimbSourceOffset shape lane limb

private theorem getD_flatten_ofFn_fixedWidth
    {Item : Type}
    {outer width : Nat}
    (blocks : Fin outer -> List Item)
    (blockLength : forall index, (blocks index).length = width)
    (outerIndex : Fin outer)
    (innerIndex : Fin width)
    (default : Item) :
    ((List.ofFn blocks).flatten).getD
        (outerIndex.val * width + innerIndex.val) default =
      (blocks outerIndex).getD innerIndex.val default := by
  induction outer with
  | zero => exact Fin.elim0 outerIndex
  | succ outer inductionHypothesis =>
      refine Fin.cases ?_ (fun index => ?_) outerIndex
      · simp only [List.ofFn_succ, List.flatten_cons, Fin.val_zero,
          Nat.zero_mul, Nat.zero_add]
        simp only [List.getD_eq_getElem?_getD]
        rw [List.getElem?_append_left (by
          rw [blockLength]
          exact innerIndex.isLt)]
      · simp only [List.ofFn_succ, List.flatten_cons, Fin.val_succ]
        simp only [List.getD_eq_getElem?_getD]
        rw [List.getElem?_append_right (by
          rw [blockLength]
          rw [Nat.add_mul, Nat.one_mul]
          omega)]
        rw [blockLength]
        have indexArithmetic :
            (index.val + 1) * width + innerIndex.val - width =
              index.val * width + innerIndex.val := by
          rw [Nat.add_mul, Nat.one_mul]
          omega
        rw [indexArithmetic]
        exact inductionHypothesis
          (fun position => blocks position.succ)
          (fun position => blockLength position.succ)
          index

private theorem twoLimbRoles_getD
    {shape : SemanticShape}
    (make : Fin ringDegree -> Limb -> SourceRole shape)
    (lane : Fin ringDegree)
    (limb : Limb)
    (default : SourceRole shape) :
    (twoLimbRoles make).getD
        (2 * lane.val + limbOffset limb) default = make lane limb := by
  cases limb with
  | c0 =>
      simpa [twoLimbRoles, limbOffset, Nat.mul_comm] using
        getD_flatten_ofFn_fixedWidth
          (width := 2)
          (fun position : Fin ringDegree =>
            [make position .c0, make position .c1])
          (by intro; rfl) lane (0 : Fin 2) default
  | c1 =>
      simpa [twoLimbRoles, limbOffset, Nat.mul_comm] using
        getD_flatten_ofFn_fixedWidth
          (width := 2)
          (fun position : Fin ringDegree =>
            [make position .c0, make position .c1])
          (by intro; rfl) lane (1 : Fin 2) default

private theorem yZcolRoles_getD
    {shape : SemanticShape}
    (source : Fin shape.sourceCount)
    (lane : Fin ringDegree)
    (limb : Limb)
    (default : SourceRole shape) :
    (yZcolRoles source).getD
        (1 + 2 * lane.val + limbOffset limb) default =
      .yZcolLimb source lane limb := by
  rw [yZcolRoles]
  have indexArithmetic :
      1 + 2 * lane.val + limbOffset limb =
        (2 * lane.val + limbOffset limb) + 1 := by
    omega
  rw [indexArithmetic, List.getD_cons_succ]
  exact twoLimbRoles_getD
    (fun position part => SourceRole.yZcolLimb source position part)
    lane limb default

private theorem yZcolLimbSourceOffset_lt
    {shape : SemanticShape}
    (lane : Fin ringDegree)
    (limb : Limb) :
    yZcolLimbSourceOffset shape lane limb <
      ActiveSemantics.sourceFieldCount shape.matrixCount := by
  cases limb <;>
    simp [yZcolLimbSourceOffset, limbOffset,
      ActiveSemantics.sourceFieldCount,
      ActiveSemantics.sourcePayloadFieldCount,
      Encoding.kVectorFieldCount] <;>
    omega

private theorem sourceBlockRoles_getD_yZcolLimb
    {shape : SemanticShape}
    (source : Fin shape.sourceCount)
    (lane : Fin ringDegree)
    (limb : Limb)
    (default : SourceRole shape) :
    (sourceBlockRoles source).getD
        (yZcolLimbSourceOffset shape lane limb) default =
      .yZcolLimb source lane limb := by
  rw [sourceBlockRoles]
  simp only [List.getD_eq_getElem?_getD]
  rw [List.getElem?_append_right (by
    simp [yZcolLimbSourceOffset]
    omega)]
  have removePrefix :
      yZcolLimbSourceOffset shape lane limb -
          (sourceHeaderRoles source ++ yRingRoles source).length =
        1 + 2 * lane.val + limbOffset limb := by
    simp [yZcolLimbSourceOffset]
    omega
  rw [removePrefix]
  exact yZcolRoles_getD source lane limb default

/-- The canonical serializer field formula selects exactly the requested
`y_zcol` limb role. This is a theorem about the independent role tree, not a
claim that any Rust-reported field index or R1CS column follows it. -/
theorem sourceRoles_getD_yZcolLimb
    {shape : SemanticShape}
    (source : Fin shape.sourceCount)
    (lane : Fin ringDegree)
    (limb : Limb)
    (default : SourceRole shape) :
    (sourceRoles shape).getD
        (yZcolLimbFieldIndex shape source lane limb) default =
      .yZcolLimb source lane limb := by
  rw [sourceRoles]
  simp only [List.getD_eq_getElem?_getD]
  rw [List.getElem?_append_right (by
    simp [yZcolLimbFieldIndex]
    omega)]
  rw [outerHeaderRoles_length]
  have removeOuterHeader :
      yZcolLimbFieldIndex shape source lane limb - 8 =
        source.val * ActiveSemantics.sourceFieldCount shape.matrixCount +
          yZcolLimbSourceOffset shape lane limb := by
    unfold yZcolLimbFieldIndex
    omega
  rw [removeOuterHeader]
  calc
    ((List.ofFn fun position : Fin shape.sourceCount =>
        sourceBlockRoles position).flatten).getD
          (source.val * ActiveSemantics.sourceFieldCount shape.matrixCount +
            yZcolLimbSourceOffset shape lane limb) default =
        (sourceBlockRoles source).getD
          (yZcolLimbSourceOffset shape lane limb) default :=
      getD_flatten_ofFn_fixedWidth
        (width := ActiveSemantics.sourceFieldCount shape.matrixCount)
        (fun position : Fin shape.sourceCount => sourceBlockRoles position)
        sourceBlockRoles_length source
        ⟨yZcolLimbSourceOffset shape lane limb,
          yZcolLimbSourceOffset_lt lane limb⟩ default
    _ = .yZcolLimb source lane limb :=
      sourceBlockRoles_getD_yZcolLimb source lane limb default

/-- The canonical field index is inside the complete serializer role list. -/
theorem yZcolLimbFieldIndex_lt
    {shape : SemanticShape}
    (source : Fin shape.sourceCount)
    (lane : Fin ringDegree)
    (limb : Limb) :
    yZcolLimbFieldIndex shape source lane limb <
      (sourceRoles shape).length := by
  have offsetLt := yZcolLimbSourceOffset_lt (shape := shape) lane limb
  have throughSource :
      source.val * ActiveSemantics.sourceFieldCount shape.matrixCount +
          yZcolLimbSourceOffset shape lane limb <
        shape.sourceCount *
          ActiveSemantics.sourceFieldCount shape.matrixCount := by
    calc
      source.val * ActiveSemantics.sourceFieldCount shape.matrixCount +
          yZcolLimbSourceOffset shape lane limb <
        source.val * ActiveSemantics.sourceFieldCount shape.matrixCount +
          ActiveSemantics.sourceFieldCount shape.matrixCount :=
        Nat.add_lt_add_left offsetLt _
      _ = (source.val + 1) *
          ActiveSemantics.sourceFieldCount shape.matrixCount := by
        rw [Nat.add_mul, Nat.one_mul]
      _ ≤ shape.sourceCount *
          ActiveSemantics.sourceFieldCount shape.matrixCount :=
        Nat.mul_le_mul_right _ (Nat.succ_le_iff.mpr source.isLt)
  rw [sourceRoles_length]
  simpa [yZcolLimbFieldIndex, ActiveSemantics.fieldCount, Nat.add_assoc] using
    Nat.add_lt_add_left throughSource 8

/-- Direct bounded indexing form of `sourceRoles_getD_yZcolLimb`. This is the
form consumed by artifact-side producer-index refinement. -/
theorem sourceRoles_getElem_yZcolLimb
    {shape : SemanticShape}
    (source : Fin shape.sourceCount)
    (lane : Fin ringDegree)
    (limb : Limb) :
    (sourceRoles shape)[yZcolLimbFieldIndex shape source lane limb]'
        (yZcolLimbFieldIndex_lt source lane limb) =
      .yZcolLimb source lane limb := by
  have selected := sourceRoles_getD_yZcolLimb
    source lane limb SourceRole.sourceCount
  simpa [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem
      (yZcolLimbFieldIndex_lt source lane limb)] using selected

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout
