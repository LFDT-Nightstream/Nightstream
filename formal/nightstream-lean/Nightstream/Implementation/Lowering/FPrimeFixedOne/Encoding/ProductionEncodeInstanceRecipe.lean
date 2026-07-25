import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Footprints

/-!
Contract: concrete physical affine recipe for the compact fixed-one
`encodeInstance` call.

Assurance tier: model-level.

Owns:
- five auxiliary source coordinates and six auxiliary output coordinates;
- one output row per coordinate, with caller-owned physical identities;
- exact active soundness and honest active/inactive completeness;
- exact row support, ownership, uniqueness, and cost.

Does not own: a complete lowering profile, an emitted whole-program receipt,
the nonlinear `freshPublic` call, generated rows, or compiled-Rust semantics.

Emits constraints: exactly six recurring rows and no temporary columns.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceRecipe

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed

namespace Native

abbrev Digest := ProductionDigestCodecs.Native.Digest
abbrev Encoded := ProductionDigestCodecs.Native.Encoded

end Native

/-- Physical placement supplied by the enclosing typed instruction.  The
semantic widths and ownership classes are fixed here; only identities and the
instruction-local row range remain caller-owned. -/
structure Placement where
  owner : PhysicalOwner
  firstOrdinal : Nat
  one : ColumnId
  active : ColumnId
  source : ColumnBundle (auxiliaryLayout 5)
  output : ColumnBundle (auxiliaryLayout 6)

/-- Exact six-row affine realization of the concrete coordinate map. -/
def recipe (placement : Placement) :
    AffineMapRecipe encodeInstanceAffineMap
      (auxiliaryLayout 5) (auxiliaryLayout 6) where
  owner := placement.owner
  firstOrdinal := placement.firstOrdinal
  one := placement.one
  active := placement.active
  source := placement.source
  output := placement.output
  sourceWidth := rfl
  targetWidth := rfl

/-- The recipe emits exactly six rows, computed from the target codec width. -/
theorem row_count (placement : Placement) :
    (recipe placement).rows.length = 6 := by
  simpa [adapterEncodedCodec] using
    (AffineMapRecipe.row_count (recipe placement))

/-- The selected intrinsic footprint is six recurring rows and no
temporaries. -/
theorem footprint_exact :
    affineFootprint adapterEncodedCodec.width =
      { recurringRows := 6, temporaries := [] } :=
  rfl

/-- Every emitted row belongs to the caller-selected instruction owner. -/
theorem rows_owned
    (placement : Placement)
    (row : OwnedRow)
    (member : row ∈ (recipe placement).rows) :
    row.id.owner = placement.owner :=
  AffineMapRecipe.rows_owned (recipe placement) row member

/-- The six emitted physical row identities are pairwise distinct. -/
theorem row_ids_nodup (placement : Placement) :
    ((recipe placement).rows.map fun row => row.id).Nodup :=
  AffineMapRecipe.row_ids_nodup (recipe placement)

/-- No row can mention a column outside constant one, activation, the five
source coordinates, and the six output coordinates. -/
theorem rows_supported
    (placement : Placement)
    (row : OwnedRow)
    (member : row ∈ (recipe placement).rows)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column = placement.one ∨ column = placement.active ∨
      column ∈ placement.source.ids ∨
      column ∈ placement.output.ids :=
  AffineMapRecipe.rows_supported
    (recipe placement) row member column columnMember

/-- Active physical satisfaction decodes exactly the native
`encodeInstance` result. -/
theorem active_sound
    (placement : Placement)
    (assignment : ColumnId -> Field)
    (digest : Option Native.Digest)
    (constantOne : assignment placement.one = 1)
    (activeOne : assignment placement.active = 1)
    (sourceDecoded :
      optionalDigestCodec.decode (placement.source.values assignment) =
        some digest)
    (holds : Satisfies (recipe placement).rows assignment) :
    adapterEncodedCodec.decode (placement.output.values assignment) =
      some
        (Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.encodeInstance
          digest) :=
  AffineMapRecipe.active_sound
    (recipe placement) assignment digest constantOne activeOne sourceDecoded
      holds

/-- Canonically encoded active inputs and outputs satisfy all six rows
without allocating a temporary. -/
theorem active_complete
    (placement : Placement)
    (assignment : ColumnId -> Field)
    (digest : Option Native.Digest)
    (constantOne : assignment placement.one = 1)
    (activeOne : assignment placement.active = 1)
    (sourceCoordinates :
      placement.source.values assignment =
        optionalDigestCodec.encode digest)
    (outputCoordinates :
      placement.output.values assignment =
        adapterEncodedCodec.encode
          (Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.encodeInstance
            digest)) :
    Satisfies (recipe placement).rows assignment :=
  AffineMapRecipe.active_complete
    (recipe placement) assignment digest constantOne activeOne
      sourceCoordinates outputCoordinates True.intro

/-- When the instruction is inactive, every assignment satisfies its six
gated rows. -/
theorem inactive_complete
    (placement : Placement)
    (assignment : ColumnId -> Field)
    (activeZero : assignment placement.active = 0) :
    Satisfies (recipe placement).rows assignment :=
  AffineMapRecipe.inactive_complete
    (recipe placement) assignment activeZero

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceRecipe
