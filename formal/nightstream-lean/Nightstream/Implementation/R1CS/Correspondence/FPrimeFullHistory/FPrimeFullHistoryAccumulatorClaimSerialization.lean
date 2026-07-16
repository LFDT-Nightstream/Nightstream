import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryParentCeSerialization

/-!
Contract: exact supported-profile serialization for Rust
`accumulator_ce_claim_digest/v1` and its full-history PiDEC-parent source schema.

| Component | Mathematical obligation | Rust owner | Emits constraints? |
|---|---|---|---|
| serializer | Order every accumulator-authority field in the v1 Poseidon2 preimage | `paper::digest::accumulator_ce_claim_digest` | no |
| source schema | Interleave verifier-owned constants with exact PiDEC parent columns | `enforce_accumulator_ce_claim_digest` | no |
| terminal binding | Decode the terminal checked parent through its production relabeling | terminal accumulator owner | no |
| recursive binding | Decode the recursive checked parent through its production relabeling | recursive accumulator owner | no |

Owns: the value-level v1 preimage, supported-profile rejection of `adv`, and
the exact assignment-index schema shared by terminal and recursive owners.
Does not own: Poseidon2 rows, constant-definition rows, PiDEC acceptance,
`y_zcol` validation, or accumulator-digest authority.
Authority boundary: this serializer is only a projection of a checked PiDEC
parent.  Its digest is compression, never authority by itself.

Assurance tier: model-level plus an exact assignment-index refinement theorem.
Artifact-checked terminal and recursive bindings belong in their accumulator
soundness modules after each generated artifact pins its constant columns.
-/

namespace Nightstream.Implementation.R1CS.AccumulatorCeClaimDigestV1

open PiDecStrictCompiler

/-- `pack_bytes_as_fields(b"neo.fold.clean/accumulator_ce_claim_digest/v1")`. -/
def domainTag : List Nat :=
  [45, 30521782141150574, 31069335676202596, 33052923221205295,
    27970967795163500, 30796639612657509, 32777976662287455, 3241519]

def pairLimbs (values : List (Nat × Nat)) : List Nat :=
  values.flatMap fun value => [value.1, value.2]

/-- Each decoded `yRing` row stores flat extension-field limbs.  The native
serializer prefixes each row by its number of extension-field elements. -/
def yRingFields (extensionLimbs : Nat) (rows : List (List Nat)) : List Nat :=
  rows.flatMap fun row => (row.length / extensionLimbs) :: row

/-- Exact accumulator-v1 preimage for the current clean profile.

The zero after `ct` encodes `aux_openings.len`, while the final three zeros
encode `c_step_coords.len`, `u_offset`, and `u_len`.  Those sidecars are
not represented by `DecodedClaim` because the strict PiDEC owner rejects them.
`y_zcol` is intentionally absent from this accumulator projection. -/
def supportedNoAdvPreimage (activeXColumns extensionLimbs : Nat)
    (claim : DecodedClaim) : List Nat :=
  domainTag ++
    [claim.commitment.d, claim.commitment.kappa,
      claim.commitment.data.length] ++
    claim.commitment.data ++
    [claim.xRows, claim.xWidth, activeXColumns] ++
    claim.xActive ++
    [claim.r.length] ++ pairLimbs claim.r ++
    [claim.sCol.length] ++ pairLimbs claim.sCol ++
    [claim.yRing.length] ++ yRingFields extensionLimbs claim.yRing ++
    [claim.ct.length] ++ pairLimbs claim.ct ++
    [0, claim.mIn] ++ claim.foldDigest ++
    [0, 0, 0]

/-- Safe public serializer for the supported non-Nebula profile.  A present
`adv` requires the Rust Nebula leaf-digest extension and is rejected here. -/
def serialize (activeXColumns extensionLimbs : Nat)
    (claim : DecodedClaim) : Option (List Nat) :=
  match claim.adv with
  | none => some (supportedNoAdvPreimage activeXColumns extensionLimbs claim)
  | some _ => none

theorem serialize_eq_some_iff (activeXColumns extensionLimbs : Nat)
    (claim : DecodedClaim) (preimage : List Nat) :
    serialize activeXColumns extensionLimbs claim = some preimage ↔
      claim.adv = none ∧
        preimage = supportedNoAdvPreimage activeXColumns extensionLimbs claim := by
  cases h : claim.adv with
  | none =>
      simp only [serialize, h, Option.some.injEq, true_and]
      exact eq_comm
  | some adv => simp [serialize, h]

end Nightstream.Implementation.R1CS.AccumulatorCeClaimDigestV1

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryAccumulatorClaimSerialization

open PiDecStrictCompiler
open FPrimeFullHistoryParentCeSerialization

private theorem decodeClaim_commitment_data (assignment : Nat → Nat)
    (layout : ClaimLayout) :
    (decodeClaim assignment layout).commitment.data =
      layout.commitment.dataCols.map assignment := by
  rfl

private theorem decodeClaim_xActive (assignment : Nat → Nat)
    (layout : ClaimLayout) :
    (decodeClaim assignment layout).xActive =
      layout.xActiveCols.map assignment := by
  rfl

private theorem decodeClaim_r (assignment : Nat → Nat)
    (layout : ClaimLayout) :
    (decodeClaim assignment layout).r =
      layout.rCols.map fun pair => (assignment pair.1, assignment pair.2) := by
  rfl

private theorem decodeClaim_sCol (assignment : Nat → Nat)
    (layout : ClaimLayout) :
    (decodeClaim assignment layout).sCol =
      layout.sColCols.map fun pair => (assignment pair.1, assignment pair.2) := by
  rfl

private theorem decodeClaim_yRing (assignment : Nat → Nat)
    (layout : ClaimLayout) :
    (decodeClaim assignment layout).yRing =
      layout.yRingCols.map fun columns => columns.map assignment := by
  rfl

private theorem decodeClaim_ct (assignment : Nat → Nat)
    (layout : ClaimLayout) :
    (decodeClaim assignment layout).ct =
      layout.ctCols.map fun pair => (assignment pair.1, assignment pair.2) := by
  rfl

private theorem decodeClaim_foldDigest (assignment : Nat → Nat)
    (layout : ClaimLayout) :
    (decodeClaim assignment layout).foldDigest =
      layout.foldDigestCols.map assignment := by
  rfl

/-- Exact accumulator-v1 preimage under an arbitrary generated PiDEC
local-to-owner column relabeling. -/
def accumulatorParentPreimageWith (columnMap : List Nat)
    (assignment : Nat → Nat) : List Nat :=
  AccumulatorCeClaimDigestV1.supportedNoAdvPreimage
    (PiDecStrictCompiler.activeColumns FPrimeFullHistoryPiDec.layout)
    FPrimeFullHistoryPiDec.layout.extensionLimbs
    (verifierParentWith columnMap assignment)

/-- Terminal checked-parent projection before Poseidon2 compression. -/
def terminalParentPreimage (assignment : Nat → Nat) : List Nat :=
  accumulatorParentPreimageWith
    FPrimeFullHistoryPiDec.terminalColumnMap assignment

/-- Recursive checked-parent projection before Poseidon2 compression. -/
def recursiveParentPreimage (assignment : Nat → Nat) : List Nat :=
  accumulatorParentPreimageWith
    FPrimeFullHistoryPiDec.recursiveColumnMap assignment

/-- Values allocated as constants by the exact supported-profile Rust gadget,
in allocation order. -/
def constantValues : List Nat :=
  AccumulatorCeClaimDigestV1.domainTag ++
    [ FPrimeFullHistoryPiDec.layout.ringDimension
    , FPrimeFullHistoryPiDec.layout.parent.commitment.dataCols.length /
        FPrimeFullHistoryPiDec.layout.ringDimension
    , FPrimeFullHistoryPiDec.layout.parent.commitment.dataCols.length
    , FPrimeFullHistoryPiDec.layout.parent.xRows
    , FPrimeFullHistoryPiDec.layout.parent.xWidth
    , PiDecStrictCompiler.activeColumns FPrimeFullHistoryPiDec.layout
    , FPrimeFullHistoryPiDec.layout.parent.rCols.length
    , FPrimeFullHistoryPiDec.layout.parent.sColCols.length
    , FPrimeFullHistoryPiDec.layout.parent.yRingCols.length
    , (FPrimeFullHistoryPiDec.layout.parent.yRingCols.getD 0 []).length /
        FPrimeFullHistoryPiDec.layout.extensionLimbs
    , (FPrimeFullHistoryPiDec.layout.parent.yRingCols.getD 1 []).length /
        FPrimeFullHistoryPiDec.layout.extensionLimbs
    , (FPrimeFullHistoryPiDec.layout.parent.yRingCols.getD 2 []).length /
        FPrimeFullHistoryPiDec.layout.extensionLimbs
    , FPrimeFullHistoryPiDec.layout.parent.ctCols.length
    , 0
    , FPrimeFullHistoryPiDec.layout.parent.mIn
    , 0
    , 0
    , 0 ]

theorem constantValues_value :
    constantValues =
      [45, 30521782141150574, 31069335676202596, 33052923221205295,
        27970967795163500, 30796639612657509, 32777976662287455, 3241519,
        54, 18, 972, 54, 257, 5, 1, 9, 3, 64, 64, 64, 3, 0, 257, 0, 0, 0] := by
  native_decide

private theorem activeColumns_value :
    PiDecStrictCompiler.activeColumns FPrimeFullHistoryPiDec.layout = 5 := by
  native_decide

/-- Positions of the 26 verifier-owned constants in the 1,682-field preimage. -/
def constantSourceIndices : List Nat :=
  List.range 11 ++
    [983, 984, 985, 1256, 1259, 1278, 1279, 1408, 1537, 1666,
      1673, 1674, 1679, 1680, 1681]

def constantColumnsFrom (sourceColumns : List Nat) : List Nat :=
  constantSourceIndices.map fun index => sourceColumns.getD index 0

/-- Exact source-column ordering emitted by the supported-profile accumulator
gadget.  `constantColumns` must be supplied in `constantValues` order. -/
def expectedSourceColumns (columnMap constantColumns : List Nat) : List Nat :=
  let claim := FPrimeFullHistoryPiDec.layout.parent
  constantColumns.take 11 ++
    relabelColumns columnMap claim.commitment.dataCols ++
    (constantColumns.drop 11).take 3 ++
    relabelColumns columnMap claim.xActiveCols ++
    (constantColumns.drop 14).take 1 ++
    relabelColumns columnMap (pairColumns claim.rCols) ++
    (constantColumns.drop 15).take 1 ++
    relabelColumns columnMap (pairColumns claim.sColCols) ++
    (constantColumns.drop 16).take 2 ++
    relabelColumns columnMap (claim.yRingCols.getD 0 []) ++
    (constantColumns.drop 18).take 1 ++
    relabelColumns columnMap (claim.yRingCols.getD 1 []) ++
    (constantColumns.drop 19).take 1 ++
    relabelColumns columnMap (claim.yRingCols.getD 2 []) ++
    (constantColumns.drop 20).take 1 ++
    relabelColumns columnMap (pairColumns claim.ctCols) ++
    (constantColumns.drop 21).take 2 ++
    relabelColumns columnMap claim.foldDigestCols ++
    (constantColumns.drop 23).take 3

theorem expectedSourceColumns_length
    {columnMap constantColumns : List Nat}
    (constantLength : constantColumns.length = 26) :
    (expectedSourceColumns columnMap constantColumns).length = 1682 := by
  simp [expectedSourceColumns, relabelColumns, pairColumns, constantLength,
    FPrimeFullHistoryPiDec.layout]

/-- Once an accumulator owner pins its freshly allocated constants, evaluating
the exact source schema yields the verifier-normalized PiDEC parent preimage. -/
theorem expectedSourceColumns_values
    {columnMap constantColumns : List Nat} {assignment : Nat → Nat}
    (constants : constantColumns.map assignment = constantValues) :
    (expectedSourceColumns columnMap constantColumns).map assignment =
      accumulatorParentPreimageWith columnMap assignment := by
  rw [constantValues_value] at constants
  unfold expectedSourceColumns accumulatorParentPreimageWith verifierParentWith
    decodedParentWith
  simp only [List.map_append, List.map_take, List.map_drop]
  rw [constants, activeColumns_value]
  simp [AccumulatorCeClaimDigestV1.domainTag,
    AccumulatorCeClaimDigestV1.supportedNoAdvPreimage,
    AccumulatorCeClaimDigestV1.yRingFields,
    AccumulatorCeClaimDigestV1.pairLimbs,
    decodeClaim_commitment_data, decodeClaim_xActive, decodeClaim_r,
    decodeClaim_sCol, decodeClaim_yRing, decodeClaim_ct,
    decodeClaim_foldDigest,
    relabelColumns, pairColumns, Relabel.assignment, Relabel.column,
    Function.comp_def, FPrimeFullHistoryPiDec.layout]

theorem terminalExpectedSourceColumns_values
    {constantColumns : List Nat} {assignment : Nat → Nat}
    (constants : constantColumns.map assignment = constantValues) :
    (expectedSourceColumns FPrimeFullHistoryPiDec.terminalColumnMap
        constantColumns).map assignment = terminalParentPreimage assignment :=
  expectedSourceColumns_values constants

theorem recursiveExpectedSourceColumns_values
    {constantColumns : List Nat} {assignment : Nat → Nat}
    (constants : constantColumns.map assignment = constantValues) :
    (expectedSourceColumns FPrimeFullHistoryPiDec.recursiveColumnMap
        constantColumns).map assignment = recursiveParentPreimage assignment :=
  expectedSourceColumns_values constants

theorem serialize_verifierParentWith (columnMap : List Nat)
    (assignment : Nat → Nat) :
    AccumulatorCeClaimDigestV1.serialize
        (PiDecStrictCompiler.activeColumns FPrimeFullHistoryPiDec.layout)
        FPrimeFullHistoryPiDec.layout.extensionLimbs
        (verifierParentWith columnMap assignment) =
      some (accumulatorParentPreimageWith columnMap assignment) := by
  rfl

/-- Under strict-PiDEC shape agreement, verifier normalization changes no
serialized value. -/
theorem accumulatorParentPreimage_eq_decodedWith
    {columnMap : List Nat} {assignment : Nat → Nat}
    (shape : ShapeAgreesWith columnMap assignment) :
    accumulatorParentPreimageWith columnMap assignment =
      AccumulatorCeClaimDigestV1.supportedNoAdvPreimage
        (PiDecStrictCompiler.activeColumns FPrimeFullHistoryPiDec.layout)
        FPrimeFullHistoryPiDec.layout.extensionLimbs
        (decodedParentWith columnMap assignment) := by
  simp [accumulatorParentPreimageWith, verifierParentWith, decodedParentWith,
    AccumulatorCeClaimDigestV1.supportedNoAdvPreimage]
  exact ⟨shape.commitmentD.symm, shape.commitmentKappa.symm,
    shape.xRows.symm, shape.xWidth.symm, shape.mIn.symm⟩

/-- A raw decoded parent serializes to the same projection once the PiDEC owner
establishes the static shape equalities. -/
theorem serialize_parentWith
    {columnMap : List Nat} {assignment : Nat → Nat}
    (shape : ShapeAgreesWith columnMap assignment) :
    AccumulatorCeClaimDigestV1.serialize
        (PiDecStrictCompiler.activeColumns FPrimeFullHistoryPiDec.layout)
        FPrimeFullHistoryPiDec.layout.extensionLimbs
        (decodedParentWith columnMap assignment) =
      some (accumulatorParentPreimageWith columnMap assignment) := by
  rw [AccumulatorCeClaimDigestV1.serialize, decodedParentWith_noAdv,
    ← accumulatorParentPreimage_eq_decodedWith shape]

theorem terminalParentPreimage_eq_decoded {assignment : Nat → Nat}
    (shape : ShapeAgrees assignment) :
    terminalParentPreimage assignment =
      AccumulatorCeClaimDigestV1.supportedNoAdvPreimage
        (PiDecStrictCompiler.activeColumns FPrimeFullHistoryPiDec.layout)
        FPrimeFullHistoryPiDec.layout.extensionLimbs
        (decodedParent assignment) :=
  accumulatorParentPreimage_eq_decodedWith shape

theorem serialize_terminalParent {assignment : Nat → Nat}
    (shape : ShapeAgrees assignment) :
    AccumulatorCeClaimDigestV1.serialize
        (PiDecStrictCompiler.activeColumns FPrimeFullHistoryPiDec.layout)
        FPrimeFullHistoryPiDec.layout.extensionLimbs
        (decodedParent assignment) =
      some (terminalParentPreimage assignment) :=
  serialize_parentWith shape

theorem serialize_recursiveParent {assignment : Nat → Nat}
    (shape : RecursiveShapeAgrees assignment) :
    AccumulatorCeClaimDigestV1.serialize
        (PiDecStrictCompiler.activeColumns FPrimeFullHistoryPiDec.layout)
        FPrimeFullHistoryPiDec.layout.extensionLimbs
        (decodedParentWith FPrimeFullHistoryPiDec.recursiveColumnMap assignment) =
      some (recursiveParentPreimage assignment) :=
  serialize_parentWith shape

/-- The exact supported-profile preimage has 1,682 Goldilocks field elements,
independently of the terminal or recursive production relabeling. -/
theorem accumulatorParentPreimageWith_length (columnMap : List Nat)
    (assignment : Nat → Nat) :
    (accumulatorParentPreimageWith columnMap assignment).length = 1682 := by
  simp [accumulatorParentPreimageWith,
    AccumulatorCeClaimDigestV1.supportedNoAdvPreimage,
    AccumulatorCeClaimDigestV1.domainTag,
    AccumulatorCeClaimDigestV1.pairLimbs,
    AccumulatorCeClaimDigestV1.yRingFields,
    verifierParentWith, decodedParentWith,
    decodeClaim_commitment_data, decodeClaim_xActive, decodeClaim_r,
    decodeClaim_sCol, decodeClaim_yRing, decodeClaim_ct,
    decodeClaim_foldDigest,
    PiDecStrictCompiler.activeColumns, FPrimeFullHistoryPiDec.layout]

theorem terminalParentPreimage_length (assignment : Nat → Nat) :
    (terminalParentPreimage assignment).length = 1682 :=
  accumulatorParentPreimageWith_length _ _

theorem recursiveParentPreimage_length (assignment : Nat → Nat) :
    (recursiveParentPreimage assignment).length = 1682 :=
  accumulatorParentPreimageWith_length _ _

end Nightstream.Implementation.R1CS.FPrimeFullHistoryAccumulatorClaimSerialization
