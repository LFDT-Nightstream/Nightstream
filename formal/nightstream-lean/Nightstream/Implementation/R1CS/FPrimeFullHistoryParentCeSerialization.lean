import Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDecArtifact

/-!
Contract: R1CS-layer decoding and canonical `ce_claim_digest/v2` serialization
for the exact full-history PiDEC parent claim.

This module owns values and ordering only.  It does not hash the preimage and
does not treat a digest as claim authority.  A generated accumulator artifact
can bind its ordered source columns to `parentPreimage`; the surrounding owner
must separately prove the generated constant columns and PiDEC relabeling.
-/

namespace Nightstream.Implementation.R1CS.PiDecStrictCompiler

/-- One verifier-visible commitment decoded from a strict-PiDEC layout. -/
structure DecodedCommitment where
  d : Nat
  kappa : Nat
  data : List Nat
deriving DecidableEq, Repr

/-- Optional Nebula product-commitment coordinates. -/
structure DecodedAdv where
  ops : DecodedCommitment
  is : DecodedCommitment
  fs : DecodedCommitment
deriving DecidableEq, Repr

/-- Public CE-claim fields carried by the strict-PiDEC wire layout. -/
structure DecodedClaim where
  commitment : DecodedCommitment
  adv : Option DecodedAdv
  xActive : List Nat
  xInactive : Nat
  xRows : Nat
  xWidth : Nat
  mIn : Nat
  yRing : List (List Nat)
  ct : List (Nat × Nat)
  r : List (Nat × Nat)
  sCol : List (Nat × Nat)
  foldDigest : List Nat
deriving DecidableEq, Repr

private def values (assignment : Nat → Nat) (columns : List Nat) : List Nat :=
  columns.map assignment

private def pairValues (assignment : Nat → Nat)
    (columns : List (Nat × Nat)) : List (Nat × Nat) :=
  columns.map fun pair => (assignment pair.1, assignment pair.2)

def decodeCommitment (assignment : Nat → Nat)
    (layout : CommitmentLayout) : DecodedCommitment where
  d := assignment layout.dCol
  kappa := assignment layout.kappaCol
  data := values assignment layout.dataCols

def decodeAdv (assignment : Nat → Nat)
    (layout : AdvLayout) : DecodedAdv where
  ops := decodeCommitment assignment layout.ops
  is := decodeCommitment assignment layout.is
  fs := decodeCommitment assignment layout.fs

/-- Decode the public CE claim directly from strict-PiDEC columns. -/
def decodeClaim (assignment : Nat → Nat)
    (layout : ClaimLayout) : DecodedClaim where
  commitment := decodeCommitment assignment layout.commitment
  adv := layout.adv.map (decodeAdv assignment)
  xActive := values assignment layout.xActiveCols
  xInactive := assignment layout.xInactiveCol
  xRows := assignment layout.xRowsCol
  xWidth := assignment layout.xWidthCol
  mIn := assignment layout.mInCol
  yRing := layout.yRingCols.map (values assignment)
  ct := pairValues assignment layout.ctCols
  r := pairValues assignment layout.rCols
  sCol := pairValues assignment layout.sColCols
  foldDigest := values assignment layout.foldDigestCols

end Nightstream.Implementation.R1CS.PiDecStrictCompiler

namespace Nightstream.Implementation.R1CS.CeClaimDigestV2

open PiDecStrictCompiler

/-- `pack_bytes_as_fields(b"neo.fold.clean/ce_claim_digest/v2")`. -/
def domainTag : List Nat :=
  [33, 30521782141150574, 31069335676202596, 27422246798975791,
    28542675000978793, 216731186291]

def pairLimbs (values : List (Nat × Nat)) : List Nat :=
  values.flatMap fun value => [value.1, value.2]

/-- Each decoded `yRing` row stores flat extension-field limbs.  The native
digest prefixes the row by its number of extension-field elements. -/
def yRingFields (extensionLimbs : Nat) (rows : List (List Nat)) : List Nat :=
  rows.flatMap fun row => (row.length / extensionLimbs) :: row

/-- Core v2 preimage for the supported profile where Nebula is absent. -/
def noAdvPreimage (activeXColumns extensionLimbs : Nat)
    (claim : DecodedClaim) : List Nat :=
  domainTag ++
    [claim.commitment.d, claim.commitment.kappa,
      claim.commitment.data.length] ++
    claim.commitment.data ++
    [claim.xRows, claim.xWidth, activeXColumns] ++
    claim.xActive ++
    [claim.r.length] ++ pairLimbs claim.r ++
    [claim.yRing.length] ++ yRingFields extensionLimbs claim.yRing ++
    [claim.mIn] ++ claim.foldDigest

/-- Safe public serializer.  Profiles carrying Nebula coordinates must add
the three Nebula leaf digests and therefore are rejected by this module. -/
def serialize (activeXColumns extensionLimbs : Nat)
    (claim : DecodedClaim) : Option (List Nat) :=
  match claim.adv with
  | none => some (noAdvPreimage activeXColumns extensionLimbs claim)
  | some _ => none

theorem serialize_eq_some_iff (activeXColumns extensionLimbs : Nat)
    (claim : DecodedClaim) (preimage : List Nat) :
    serialize activeXColumns extensionLimbs claim = some preimage ↔
      claim.adv = none ∧
        preimage = noAdvPreimage activeXColumns extensionLimbs claim := by
  cases h : claim.adv with
  | none =>
      simp only [serialize, h, Option.some.injEq, true_and]
      exact eq_comm
  | some adv => simp [serialize, h]

end Nightstream.Implementation.R1CS.CeClaimDigestV2

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryParentCeSerialization

open PiDecStrictCompiler

/-- Decode the parent through an exact local-to-owner column map. -/
def decodedParentWith (columnMap : List Nat) (assignment : Nat → Nat) :
    DecodedClaim :=
  decodeClaim (Relabel.assignment columnMap assignment)
    FPrimeFullHistoryPiDec.layout.parent

/-- Decode the PiDEC parent through the exact terminal local-to-production
column map used by the generated terminal verifier artifact. -/
def decodedParent (assignment : Nat → Nat) : DecodedClaim :=
  decodedParentWith FPrimeFullHistoryPiDec.terminalColumnMap assignment

/-- Nebula is absent independently of which generated owner relabeling is
used to decode this supported-profile parent. -/
theorem decodedParentWith_noAdv (columnMap : List Nat)
    (assignment : Nat → Nat) :
    (decodedParentWith columnMap assignment).adv = none := by
  rfl

/-- The supported full-history profile carries no Nebula product commitment. -/
theorem decodedParent_noAdv (assignment : Nat → Nat) :
    (decodedParent assignment).adv = none := by
  rfl

/-- Verifier-owned shape normalization used by the production digest gadget.
The gadget allocates constants for these fields; it does not read the earlier
PiDEC shape wires a second time. -/
def verifierParentWith (columnMap : List Nat) (assignment : Nat → Nat) :
    DecodedClaim :=
  let decoded := decodedParentWith columnMap assignment
  { decoded with
    commitment :=
      { decoded.commitment with
        d := FPrimeFullHistoryPiDec.layout.ringDimension
        kappa := FPrimeFullHistoryPiDec.layout.parent.commitment.dataCols.length /
          FPrimeFullHistoryPiDec.layout.ringDimension }
    xRows := FPrimeFullHistoryPiDec.layout.parent.xRows
    xWidth := FPrimeFullHistoryPiDec.layout.parent.xWidth
    mIn := FPrimeFullHistoryPiDec.layout.parent.mIn }

def verifierParent (assignment : Nat → Nat) : DecodedClaim :=
  verifierParentWith FPrimeFullHistoryPiDec.terminalColumnMap assignment

theorem verifierParent_noAdv (assignment : Nat → Nat) :
    (verifierParent assignment).adv = none := by
  rfl

/-- The separate strict-PiDEC owner must establish these wire-to-static-shape
equalities before the raw decoded claim and verifier-normalized claim coincide. -/
structure ShapeAgreesWith (columnMap : List Nat)
    (assignment : Nat → Nat) : Prop where
  commitmentD : (decodedParentWith columnMap assignment).commitment.d =
    FPrimeFullHistoryPiDec.layout.ringDimension
  commitmentKappa : (decodedParentWith columnMap assignment).commitment.kappa =
    FPrimeFullHistoryPiDec.layout.parent.commitment.dataCols.length /
      FPrimeFullHistoryPiDec.layout.ringDimension
  xRows : (decodedParentWith columnMap assignment).xRows =
    FPrimeFullHistoryPiDec.layout.parent.xRows
  xWidth : (decodedParentWith columnMap assignment).xWidth =
    FPrimeFullHistoryPiDec.layout.parent.xWidth
  mIn : (decodedParentWith columnMap assignment).mIn =
    FPrimeFullHistoryPiDec.layout.parent.mIn

abbrev ShapeAgrees (assignment : Nat → Nat) : Prop :=
  ShapeAgreesWith FPrimeFullHistoryPiDec.terminalColumnMap assignment

abbrev TerminalShapeAgrees (assignment : Nat → Nat) : Prop :=
  ShapeAgrees assignment

abbrev RecursiveShapeAgrees (assignment : Nat → Nat) : Prop :=
  ShapeAgreesWith FPrimeFullHistoryPiDec.recursiveColumnMap assignment

/-- Exact v2 raw preimage under an arbitrary generated PiDEC relabeling. -/
def parentPreimageWith (columnMap : List Nat)
    (assignment : Nat → Nat) : List Nat :=
  CeClaimDigestV2.noAdvPreimage
    (PiDecStrictCompiler.activeColumns FPrimeFullHistoryPiDec.layout)
    FPrimeFullHistoryPiDec.layout.extensionLimbs
    (verifierParentWith columnMap assignment)

/-- Exact native `ce_claim_digest/v2` raw preimage values for the decoded
terminal PiDEC parent.  This is the semantic target for exported source
columns, before shifted-ternary encoding, SIS accumulation, or Poseidon2. -/
def parentPreimage (assignment : Nat → Nat) : List Nat :=
  parentPreimageWith FPrimeFullHistoryPiDec.terminalColumnMap assignment

/-- Values allocated as constants by `enforce_ce_claim_digest` for this exact
profile, in their allocation order. -/
def constantValues : List Nat :=
  CeClaimDigestV2.domainTag ++
    [ FPrimeFullHistoryPiDec.layout.ringDimension
    , FPrimeFullHistoryPiDec.layout.parent.commitment.dataCols.length /
        FPrimeFullHistoryPiDec.layout.ringDimension
    , FPrimeFullHistoryPiDec.layout.parent.commitment.dataCols.length
    , FPrimeFullHistoryPiDec.layout.parent.xRows
    , FPrimeFullHistoryPiDec.layout.parent.xWidth
    , PiDecStrictCompiler.activeColumns FPrimeFullHistoryPiDec.layout
    , FPrimeFullHistoryPiDec.layout.parent.rCols.length
    , FPrimeFullHistoryPiDec.layout.parent.yRingCols.length
    , (FPrimeFullHistoryPiDec.layout.parent.yRingCols.getD 0 []).length /
        FPrimeFullHistoryPiDec.layout.extensionLimbs
    , (FPrimeFullHistoryPiDec.layout.parent.yRingCols.getD 1 []).length /
        FPrimeFullHistoryPiDec.layout.extensionLimbs
    , (FPrimeFullHistoryPiDec.layout.parent.yRingCols.getD 2 []).length /
        FPrimeFullHistoryPiDec.layout.extensionLimbs
    , FPrimeFullHistoryPiDec.layout.parent.mIn ]

theorem constantValues_value :
    constantValues =
      [33, 30521782141150574, 31069335676202596, 27422246798975791,
        28542675000978793, 216731186291, 54, 18, 972, 54, 257, 5,
        1, 3, 64, 64, 64, 257] := by
  native_decide

private theorem activeColumns_value :
    PiDecStrictCompiler.activeColumns FPrimeFullHistoryPiDec.layout = 5 := by
  native_decide

/-- Positions of those constants in the 1,650-field raw source list. -/
def constantSourceIndices : List Nat :=
  (List.range 9) ++ [981, 982, 983, 1254, 1257, 1258, 1387, 1516, 1645]

def constantColumnsFrom (sourceColumns : List Nat) : List Nat :=
  constantSourceIndices.map fun index => sourceColumns.getD index 0

def relabelColumns (columnMap columns : List Nat) : List Nat :=
  columns.map (Relabel.column columnMap)

def pairColumns (columns : List (Nat × Nat)) : List Nat :=
  columns.flatMap fun pair => [pair.1, pair.2]

/-- Exact raw source-column schema, parameterized only by the generated
PiDEC relabeling and the digest gadget's freshly allocated constant columns. -/
def expectedSourceColumns (columnMap constantColumns : List Nat) : List Nat :=
  let claim := FPrimeFullHistoryPiDec.layout.parent
  constantColumns.take 9 ++
    relabelColumns columnMap claim.commitment.dataCols ++
    (constantColumns.drop 9).take 3 ++
    relabelColumns columnMap claim.xActiveCols ++
    (constantColumns.drop 12).take 1 ++
    relabelColumns columnMap (pairColumns claim.rCols) ++
    (constantColumns.drop 13).take 2 ++
    relabelColumns columnMap (claim.yRingCols.getD 0 []) ++
    (constantColumns.drop 15).take 1 ++
    relabelColumns columnMap (claim.yRingCols.getD 1 []) ++
    (constantColumns.drop 16).take 1 ++
    relabelColumns columnMap (claim.yRingCols.getD 2 []) ++
    (constantColumns.drop 17).take 1 ++
    relabelColumns columnMap claim.foldDigestCols

/-- Once the ordinary owner pins its freshly allocated constants, the exact
source schema evaluates to the verifier-normalized decoded parent preimage. -/
theorem expectedSourceColumns_values
    {columnMap constantColumns : List Nat} {assignment : Nat → Nat}
    (constants : constantColumns.map assignment = constantValues) :
    (expectedSourceColumns columnMap constantColumns).map assignment =
      parentPreimageWith columnMap assignment := by
  rw [constantValues_value] at constants
  unfold expectedSourceColumns parentPreimageWith verifierParentWith
    decodedParentWith
  simp only [List.map_append, List.map_take, List.map_drop]
  rw [constants, activeColumns_value]
  simp [expectedSourceColumns, parentPreimageWith, verifierParentWith,
    decodedParentWith, CeClaimDigestV2.domainTag,
    CeClaimDigestV2.noAdvPreimage,
    CeClaimDigestV2.yRingFields, CeClaimDigestV2.pairLimbs,
    PiDecStrictCompiler.decodeClaim, PiDecStrictCompiler.decodeCommitment,
    PiDecStrictCompiler.values, PiDecStrictCompiler.pairValues,
    relabelColumns, pairColumns, Relabel.assignment, Relabel.column,
    Function.comp_def,
    constantValues, constantValues_value, activeColumns_value,
    List.map_take, List.map_drop, constants,
    FPrimeFullHistoryPiDec.layout]

theorem serialize_verifierParent (assignment : Nat → Nat) :
    CeClaimDigestV2.serialize
        (PiDecStrictCompiler.activeColumns FPrimeFullHistoryPiDec.layout)
        FPrimeFullHistoryPiDec.layout.extensionLimbs
        (verifierParent assignment) =
      some (parentPreimage assignment) := by
  rfl

theorem parentPreimage_eq_decodedWith
    {columnMap : List Nat} {assignment : Nat → Nat}
    (shape : ShapeAgreesWith columnMap assignment) :
    parentPreimageWith columnMap assignment =
      CeClaimDigestV2.noAdvPreimage
        (PiDecStrictCompiler.activeColumns FPrimeFullHistoryPiDec.layout)
        FPrimeFullHistoryPiDec.layout.extensionLimbs
        (decodedParentWith columnMap assignment) := by
  simp [parentPreimageWith, verifierParentWith, decodedParentWith,
    CeClaimDigestV2.noAdvPreimage,
    shape.commitmentD, shape.commitmentKappa, shape.xRows, shape.xWidth,
    shape.mIn]
  exact ⟨shape.commitmentD.symm, shape.commitmentKappa.symm,
    shape.xRows.symm, shape.xWidth.symm, shape.mIn.symm⟩

theorem serialize_parentWith
    {columnMap : List Nat} {assignment : Nat → Nat}
    (shape : ShapeAgreesWith columnMap assignment) :
    CeClaimDigestV2.serialize
        (PiDecStrictCompiler.activeColumns FPrimeFullHistoryPiDec.layout)
        FPrimeFullHistoryPiDec.layout.extensionLimbs
        (decodedParentWith columnMap assignment) =
      some (parentPreimageWith columnMap assignment) := by
  rw [CeClaimDigestV2.serialize, decodedParentWith_noAdv,
    ← parentPreimage_eq_decodedWith shape]

theorem parentPreimage_eq_decoded {assignment : Nat → Nat}
    (shape : ShapeAgrees assignment) :
    parentPreimage assignment =
      CeClaimDigestV2.noAdvPreimage
        (PiDecStrictCompiler.activeColumns FPrimeFullHistoryPiDec.layout)
        FPrimeFullHistoryPiDec.layout.extensionLimbs
        (decodedParent assignment) :=
  parentPreimage_eq_decodedWith shape

theorem serialize_parent {assignment : Nat → Nat}
    (shape : ShapeAgrees assignment) :
    CeClaimDigestV2.serialize
        (PiDecStrictCompiler.activeColumns FPrimeFullHistoryPiDec.layout)
        FPrimeFullHistoryPiDec.layout.extensionLimbs
        (decodedParent assignment) =
      some (parentPreimage assignment) :=
  serialize_parentWith shape

/-- The exact supported profile has five active X columns and quadratic
extension-field rows.  These are verifier-owned layout facts, not witness
claims. -/
theorem supportedShape :
    PiDecStrictCompiler.activeColumns FPrimeFullHistoryPiDec.layout = 5 ∧
      FPrimeFullHistoryPiDec.layout.extensionLimbs = 2 := by
  native_decide

/-- The native supported-profile preimage has exactly 1,650 field elements. -/
theorem parentPreimage_length (assignment : Nat → Nat) :
    (parentPreimage assignment).length = 1650 := by
  simp [parentPreimage, parentPreimageWith, CeClaimDigestV2.noAdvPreimage,
    CeClaimDigestV2.domainTag, CeClaimDigestV2.pairLimbs,
    CeClaimDigestV2.yRingFields, decodedParent,
    verifierParent, verifierParentWith, decodedParentWith,
    PiDecStrictCompiler.decodeClaim, PiDecStrictCompiler.decodeCommitment,
    PiDecStrictCompiler.values, PiDecStrictCompiler.pairValues,
    PiDecStrictCompiler.activeColumns, FPrimeFullHistoryPiDec.layout]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryParentCeSerialization
