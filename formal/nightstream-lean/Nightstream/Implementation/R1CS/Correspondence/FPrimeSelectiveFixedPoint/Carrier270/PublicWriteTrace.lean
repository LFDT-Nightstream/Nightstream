import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270.PhysicalPublicAssignment

/-!
Executable one-arm public-write contract for the production fixed-point
assignment.

Assurance tier: model-level pending one Rust-generated certificate.

Owns: a proof-free symbolic record for one public assignment write; an exact
`Fin 270` one-arm trace shape; fail-closed execution of that trace into the
11,725,506-coordinate production assignment; and derivation of the physical
public-source, padding-zero, padding-row, and typed-carrier conclusions.

Does not own: an inhabitant of `PendingProductionExporterCertificate`, the
active Rust call site, private assignment decoding, matrix semantics,
commitment-key alignment, protocol acceptance, or row removal.  In
particular, no theorem below accepts equality of assignment coordinates as a
premise.

Emits constraints: none.

The future exporter must produce one record per typed public coordinate.  The
typed index supplies the exact 270-record bound without evaluating one joined
list: coordinates `0..256` are the conventional constant plus direct source
writes, while coordinates `257..269` are untouched initialized zeros.

| Stable stage path | Obligation | Authority class | Lean owner |
|---|---|---|---|
| `f_prime.fixed_point.assignment.public_write_trace.schema` | one proof-free write instruction | model schema | `RawPublicWrite` |
| `f_prime.fixed_point.assignment.public_write_trace.execute` | execute one arm at physical columns `0..269` | computed | `executePhysical` |
| `f_prime.fixed_point.assignment.public_write_trace.certificate` | production exporter equals the canonical trace | pending artifact | `PendingProductionExporterCertificate` |
| `f_prime.fixed_point.assignment.public_write_trace.refinement` | trace execution refines the typed carrier | derived | `projectPhysical270_execute_eq_projectPublicInput` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicWriteTrace

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicAssignment
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PhysicalPublicAssignment
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPaddingRefinement

/-- Value source used by one symbolic public-assignment write. -/
inductive RawPublicWriteKind where
  | constantOne
  | directSource
  | initializedZero
  deriving DecidableEq, Repr

/-- Proof-free symbolic view of one write performed by the one-arm assignment
encoder.  Redundant geometry is intentional: the generated certificate must
pin the arm, normalized source, final target, width, centering, and alias
policy rather than relying on a stage label. -/
structure RawPublicWrite where
  schemaVersion : Nat
  arm : Nat
  logicalColumn : Nat
  normalizedSourceColumn : Option Nat
  finalColumn : Nat
  kind : RawPublicWriteKind
  width : Nat
  centered : Bool
  aliasSource : Option Nat
  deriving DecidableEq, Repr

/-- Exactly one proof-free record for each of the 270 public coordinates.
Generated files may store this lookup in bounded shards; no theorem here
requires normalizing a joined 270-record list. -/
abbrev OneArmTrace :=
  Fin PublicDecoder.alignedPublicWidth -> RawPublicWrite

/-- Canonical one-arm write record expected from the production encoder. -/
def expectedWrite (arm : Nat)
    (column : Fin PublicDecoder.alignedPublicWidth) : RawPublicWrite :=
  if isConstant : column.val = 0 then
    { schemaVersion := 1
      arm := arm
      logicalColumn := column.val
      normalizedSourceColumn := none
      finalColumn := column.val
      kind := .constantOne
      width := 1
      centered := false
      aliasSource := none }
  else if isSource : column.val < legacyPublicWidth then
    { schemaVersion := 1
      arm := arm
      logicalColumn := column.val
      normalizedSourceColumn := some column.val
      finalColumn := column.val
      kind := .directSource
      width := 1
      centered := false
      aliasSource := none }
  else
    { schemaVersion := 1
      arm := arm
      logicalColumn := column.val
      normalizedSourceColumn := none
      finalColumn := column.val
      kind := .initializedZero
      width := 0
      centered := false
      aliasSource := none }

/-- Exact structural certificate still missing from the production Rust
exporter.  This proposition is deliberately not inhabited in handwritten
Lean.  Its future artifact proof must decode bounded generated shards from the
actual encoder snapshot used by the active call site. -/
def PendingProductionExporterCertificate (arm : Nat)
    (trace : OneArmTrace) : Prop :=
  forall column, trace column = expectedWrite arm column

/-- Interpret the value source of one proof-free write.  Malformed direct
source indices fail closed to zero; the pending exact certificate excludes
that branch for all direct public writes. -/
def writeValue (source : Fin legacyPublicWidth -> F)
    (write : RawPublicWrite) : F :=
  match write.kind with
  | .constantOne => 1
  | .directSource =>
      match write.normalizedSourceColumn with
      | some sourceColumn =>
          if inRange : sourceColumn < legacyPublicWidth then
            source ⟨sourceColumn, inRange⟩
          else
            0
      | none => 0
  | .initializedZero => 0

/-- Fail-closed execution of one trace record at its typed public index. -/
def executePublic (arm : Nat) (trace : OneArmTrace)
    (source : Fin legacyPublicWidth -> F)
    (column : Fin PublicDecoder.alignedPublicWidth) : F :=
  let write := trace column
  if valid : write.schemaVersion = 1 /\
      write.arm = arm /\
      write.logicalColumn = column.val /\
      write.finalColumn = column.val then
    writeValue source write
  else
    0

/-- Execute the bounded public trace in the exact full production width and
leave all private coordinates to the caller-owned suffix. -/
def executePhysical (arm : Nat) (trace : OneArmTrace)
    (source : Fin legacyPublicWidth -> F)
    (suffix : Fin PublicPaddingRefinement.Artifact.relationColumns -> F) :
    Fin PublicPaddingRefinement.Artifact.relationColumns -> F :=
  fun column =>
    if isPublic : column.val < PublicDecoder.alignedPublicWidth then
      executePublic arm trace source ⟨column.val, isPublic⟩
    else
      suffix column

/-- A certified trace has exactly the conventional constant/direct prefix and
the thirteen initialized-zero records.  This is pointwise structural evidence;
it does not inspect or assume assignment values. -/
theorem certified_kind_partition
    (arm : Nat) (trace : OneArmTrace)
    (certificate : PendingProductionExporterCertificate arm trace) :
    (forall column : Fin PublicDecoder.alignedPublicWidth,
      column.val < legacyPublicWidth ->
        (trace column).kind =
          if column.val = 0 then .constantOne else .directSource) /\
    (forall column : Fin PublicDecoder.alignedPublicWidth,
      legacyPublicWidth <= column.val ->
        (trace column).kind = .initializedZero) := by
  constructor
  · intro column isSource
    rw [certificate column]
    by_cases isConstant : column.val = 0
    · simp [expectedWrite, isConstant]
    · simp [expectedWrite, isConstant, isSource]
  · intro column isPadding
    rw [certificate column]
    have notConstant : column.val ≠ 0 := by
      intro equal
      simp only [legacyPublicWidth] at isPadding
      omega
    have notSource : ¬ column.val < legacyPublicWidth :=
      Nat.not_lt.mpr isPadding
    simp [expectedWrite, notConstant, notSource]

/-- Kernel reduction of one exact trace record.  The source prefix is
`source[0..256]`; the only semantic premise is its conventional constant-one
coordinate. -/
theorem executePublic_exact
    (arm : Nat) (trace : OneArmTrace)
    (source : Fin legacyPublicWidth -> F)
    (certificate : PendingProductionExporterCertificate arm trace)
    (constantOne : source ⟨0, by decide⟩ = 1)
    (column : Fin PublicDecoder.alignedPublicWidth) :
    executePublic arm trace source column =
      if isSource : column.val < legacyPublicWidth then
        source ⟨column.val, isSource⟩
      else
        0 := by
  by_cases isConstant : column.val = 0
  · have columnZero : column = ⟨0, by decide⟩ := Fin.ext isConstant
    subst column
    rw [executePublic, certificate ⟨0, by decide⟩]
    simp [expectedWrite, legacyPublicWidth, writeValue]
    exact constantOne.symm
  · by_cases isSource : column.val < legacyPublicWidth
    · simp [executePublic, certificate column, expectedWrite, isConstant,
        isSource, writeValue]
    · simp [executePublic, certificate column, expectedWrite, isConstant,
        isSource, writeValue]

/-- Physical embedding of a public trace record. -/
theorem executePhysical_at_public
    (dimensions : Dimensions)
    (arm : Nat) (trace : OneArmTrace)
    (source : Fin legacyPublicWidth -> F)
    (suffix : Fin PublicPaddingRefinement.Artifact.relationColumns -> F)
    (column : Fin dimensions.shape.publicWidth) :
    executePhysical arm trace source suffix
        (physicalPublicColumn dimensions column) =
      executePublic arm trace source (artifactColumn dimensions column) := by
  have publicBound :
      (physicalPublicColumn dimensions column).val <
        PublicDecoder.alignedPublicWidth := by
    have columnBound := column.isLt
    simpa [Dimensions.shape_publicWidth, PublicDecoder.alignedPublicWidth]
      using columnBound
  rw [executePhysical, dif_pos publicBound]
  congr 1

/-- The exact generated write program supplies `PublicSourceDataflow` by
execution; no coordinate-value equality is accepted as a premise. -/
theorem executePhysical_sourceDataflow
    (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (arm : Nat) (trace : OneArmTrace)
    (suffix : Fin PublicPaddingRefinement.Artifact.relationColumns -> F)
    (certificate : PendingProductionExporterCertificate arm trace)
    (constantOne : SourceConstantOne dimensions legacy) :
    PublicSourceDataflow dimensions legacy
      (executePhysical arm trace (sourcePublicPrefix dimensions legacy)
        suffix) := by
  intro column isSource
  rw [executePhysical_at_public]
  have sourceOne :
      sourcePublicPrefix dimensions legacy ⟨0, by decide⟩ = 1 := by
    simpa [sourcePublicPrefix, SourceConstantOne] using constantOne
  have exact := executePublic_exact arm trace
    (sourcePublicPrefix dimensions legacy) certificate sourceOne
      (artifactColumn dimensions column)
  simpa [artifactColumn, sourcePublicPrefix, isSource] using exact

/-- The final thirteen physical public coordinates produced by the certified
trace are literal initialized zeros. -/
theorem executePhysical_padding_zero
    (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (arm : Nat) (trace : OneArmTrace)
    (suffix : Fin PublicPaddingRefinement.Artifact.relationColumns -> F)
    (certificate : PendingProductionExporterCertificate arm trace)
    (constantOne : SourceConstantOne dimensions legacy)
    (offset : Fin PublicPaddingRefinement.Artifact.paddingWidth) :
    executePhysical arm trace (sourcePublicPrefix dimensions legacy) suffix
        (PublicPaddingRefinement.paddingColumn offset) = 0 := by
  let column : Fin dimensions.shape.publicWidth :=
    ⟨legacyPublicWidth + offset.val, by
      have offsetBound := offset.isLt
      simp only [Dimensions.shape_publicWidth, legacyPublicWidth,
        PublicPaddingRefinement.Artifact.paddingWidth,
        Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicPadding.paddingWidth]
        at offsetBound ⊢
      omega⟩
  have physicalEq :
      physicalPublicColumn dimensions column =
        PublicPaddingRefinement.paddingColumn offset := by
    apply Fin.ext
    rfl
  rw [← physicalEq, executePhysical_at_public]
  have sourceOne :
      sourcePublicPrefix dimensions legacy ⟨0, by decide⟩ = 1 := by
    simpa [sourcePublicPrefix, SourceConstantOne] using constantOne
  have exact := executePublic_exact arm trace
    (sourcePublicPrefix dimensions legacy) certificate sourceOne
      (artifactColumn dimensions column)
  have notSource : ¬ column.val < legacyPublicWidth := by
    simp [column]
  simpa [artifactColumn, notSource] using exact

/-- The certified write trace satisfies all thirteen exact generated
public-padding equations. -/
theorem executePhysical_paddingRowsSatisfied
    (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (arm : Nat) (trace : OneArmTrace)
    (suffix : Fin PublicPaddingRefinement.Artifact.relationColumns -> F)
    (certificate : PendingProductionExporterCertificate arm trace)
    (constantOne : SourceConstantOne dimensions legacy) :
    PublicPaddingRefinement.GeneratedRowsSatisfied
      (executePhysical arm trace (sourcePublicPrefix dimensions legacy)
        suffix) := by
  let encoded :=
    executePhysical arm trace (sourcePublicPrefix dimensions legacy) suffix
  have sourceDataflow : PublicSourceDataflow dimensions legacy encoded :=
    executePhysical_sourceDataflow dimensions legacy arm trace suffix
      certificate constantOne
  have encodedOne : encoded PublicPaddingRefinement.constantColumn = 1 := by
    let column : Fin dimensions.shape.publicWidth :=
      ⟨0, by simp [Dimensions.shape_publicWidth]⟩
    have isSource : column.val < legacyPublicWidth := by
      simp [column, legacyPublicWidth]
    have source := sourceDataflow column isSource
    have physicalEq :
        physicalPublicColumn dimensions column =
          PublicPaddingRefinement.constantColumn := by
      apply Fin.ext
      rfl
    rw [physicalEq] at source
    simpa [SourceConstantOne, column] using source.trans constantOne
  rw [PublicPaddingRefinement.generatedRowsSatisfied_iff_padding_zero encoded
    encodedOne]
  intro offset
  exact executePhysical_padding_zero dimensions legacy arm trace suffix
    certificate constantOne offset

/-- Final model-level refinement contract.  Once the production exporter
supplies `PendingProductionExporterCertificate`, executing its exact one-arm
trace identifies the physical public prefix with the independently typed
270-coordinate carrier. -/
theorem projectPhysical270_execute_eq_projectPublicInput
    (dimensions : Dimensions)
    (legacy : LegacyAssignment dimensions)
    (arm : Nat) (trace : OneArmTrace)
    (suffix : Fin PublicPaddingRefinement.Artifact.relationColumns -> F)
    (certificate : PendingProductionExporterCertificate arm trace)
    (constantOne : SourceConstantOne dimensions legacy) :
    projectPhysical270 dimensions
        (executePhysical arm trace (sourcePublicPrefix dimensions legacy)
          suffix) =
      projectPublicInput (assignment dimensions legacy) := by
  exact projectPhysical270_eq_projectPublicInput dimensions legacy
    (executePhysical arm trace (sourcePublicPrefix dimensions legacy) suffix)
    (executePhysical_sourceDataflow dimensions legacy arm trace suffix
      certificate constantOne)
    constantOne
    (executePhysical_paddingRowsSatisfied dimensions legacy arm trace suffix
      certificate constantOne)

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PublicWriteTrace
