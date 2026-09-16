import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentPlan

/-!
Cache the two retained-source boundaries before constructing per-slot readers.
Existing source-assignment functions own both suffixes, and BlockKind.template
owns every block geometry. No retained value, slot encoding or order changes.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.CachedAssignmentPlan

open NightstreamFPrime.Spec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1.PerApplicationCanonicalAssignment
open NightstreamFPrime.Export.Stage1.PerApplicationAssignmentPlan (BlockKind)

/-- Numeric data, with erased links to the original source-width owners.
Keeping the boundaries in a record separates preparation from column reads. -/
structure Widths (application : Program) where
  baseWidth : Nat
  prefixWidth : Nat
  baseWidth_eq : baseWidth = PiRLCProductPlan.baseSourceWidth application
  prefixWidth_eq : prefixWidth = ProductRetainedBlock.sourceWidth baseWidth
    PiRLCProductSchedule.invocationCount

/-- This function returns data, not a curried column reader. The application
width is evaluated once; the product-prefix width reuses that stored number. -/
@[noinline] def prepareWidths (application : Program) : Widths application :=
  let baseWidth := PiRLCProductPlan.baseSourceWidth application
  { baseWidth := baseWidth
    prefixWidth := ProductRetainedBlock.sourceWidth baseWidth PiRLCProductSchedule.invocationCount
    baseWidth_eq := rfl
    prefixWidth_eq := rfl }

private theorem sourceWidth_eq {application : Program} (widths : Widths application) :
    FieldSuffixBlock.sourceWidth widths.prefixWidth PiRLCFirst54DirectSchedule.candidateCount =
      PiRLCRetainedGeometry.sourceWidth application := by
  rw [widths.prefixWidth_eq, widths.baseWidth_eq]
  rfl

/-- Read using numeric record fields. Fin casts affect only erased proofs;
both suffix dispatches retain their original implementations. -/
@[noinline] def read {application : Program} (widths : Widths application)
    (raw : RawValues application)
    (column : Fin (PiRLCRetainedGeometry.sourceWidth application)) : F :=
  FieldSuffixBlock.sourceAssignment widths.prefixWidth PiRLCFirst54DirectSchedule.candidateCount
    (fun prefixColumn => ProductRetainedBlock.sourceAssignment widths.baseWidth
      PiRLCProductSchedule.invocationCount
      (fun base => raw.base (Fin.cast widths.baseWidth_eq base)) raw.groupValue
      (Fin.cast widths.prefixWidth_eq prefixColumn))
    raw.products (Fin.cast (sourceWidth_eq widths).symm column)

/-- The cache changes only where the two numeric boundaries are computed. -/
theorem read_eq_retainedSource {application : Program} (widths : Widths application)
    (raw : RawValues application) : read widths raw = raw.retainedSource := by
  rcases widths with ⟨baseWidth, prefixWidth, baseWidth_eq, prefixWidth_eq⟩
  subst baseWidth
  subst prefixWidth
  rfl

private def usesRetainedSource : BlockKind → Bool
  | .applicationWitness | .applicationLocal => false
  | _ => true

/-- Select the source type without copying any of the thirty block builders. -/
private theorem template_retainedWidth (application : Program) (kind : BlockKind)
    (selected : usesRetainedSource kind = true) :
    (kind.template application).sourceWidth = PiRLCRetainedGeometry.sourceWidth application := by
  cases kind <;> first | rfl | simp [usesRetainedSource] at selected

private theorem template_retainedSource {application : Program} (raw : RawValues application)
    (kind : BlockKind) (selected : usesRetainedSource kind = true) :
    (kind.template application).source raw = fun column =>
      raw.retainedSource (Fin.cast (template_retainedWidth application kind selected) column) := by
  cases kind <;> first | rfl | simp [usesRetainedSource] at selected

/-- Reuse the existing block geometry and replace only its retained reader.
The two application-source blocks keep their original reader unchanged. -/
def block {application : Program} (widths : Widths application)
    (raw : RawValues application) (kind : BlockKind) : CanonicalBlockAssignment.BlockValue :=
  let template := kind.template application
  if selected : usesRetainedSource kind = true then
    Canonical.ofBlock template.block fun column =>
      read widths raw (Fin.cast (template_retainedWidth application kind selected) column)
  else Canonical.ofBlock template.block (template.source raw)

/-- Every cached block has exactly the existing kind, slots and source values. -/
theorem block_eq {application : Program} (widths : Widths application)
    (raw : RawValues application) (kind : BlockKind) :
    block widths raw kind = kind.expand raw := by
  unfold block BlockKind.expand
  dsimp only
  split_ifs with selected
  · rw [read_eq_retainedSource]
    exact congrArg (Canonical.ofBlock (kind.template application).block)
      (template_retainedSource raw kind selected).symm
  · rfl

/-- Build the same thirty-block schedule with one shared width cache. -/
def expand {application : Program} (widths : Widths application)
    (raw : RawValues application) : Canonical.Schedule :=
  PerApplicationAssignmentPlan.canonicalKinds.map (block widths raw)

/-- Exact schedule equality with the existing transport-plan interpreter. -/
theorem expand_eq {application : Program} (widths : Widths application)
    (raw : RawValues application) :
    expand widths raw = PerApplicationAssignmentPlan.expand raw := by
  unfold expand PerApplicationAssignmentPlan.expand
  apply List.map_congr_left
  intro kind _member
  exact block_eq widths raw kind

end NightstreamFPrime.Export.Stage1.CachedAssignmentPlan
