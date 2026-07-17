import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.DomainSeparation
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Parameters

/-!
Necessity of sizing the flat NC column cube for the complete Phi81 carrier.

Protocol: SuperNeo `Pi_CCS`.
Phase: SplitNc domain selection before polynomial construction.
Constraint family: flat-column coverage; this file emits no rows.

Owns: a kernel-checked proof that a Boolean column cube sized only for a
power-of-two logical CCS width cannot cover the completed 54-lane carrier,
even though a six-bit lane cube does cover all 54 Phi81 lanes. It also names
the first semantic carrier coordinate omitted by that column domain.

Does not own: a claim that production currently chooses this domain, Rust
dimension decoding, NC polynomial soundness, SumCheck, transcript replay,
R1CS rows, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: `SemanticShape.carrierWidth`, rather than the original
logical width, determines which assignment coordinates are authoritative.
Any executable flat-column model must prove `FlatNcDomain.Covers`; matching
the original logical width is insufficient.

| Protocol | Phase | Family | Mathematical obligation | Permits row removal? |
|---|---|---|---|---|
| coefficient embedding | carrier completion | 54-lane blocks | complete carrier width is not a Boolean-cube cardinality | no |
| paper `Pi_CCS` | square row/column domain | `ColumnLayout` | the paper's direct square carrier cannot instantiate the production shape | no |
| SplitNc | candidate domain | logical-width column cube | column count equals only the original power-of-two width | no |
| SplitNc | candidate domain | six-bit lane cube | all 54 real lanes are covered | no |
| SplitNc | necessity | first completed tail coordinate | one authoritative coordinate lies outside the column cube | no |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.CarrierCoverage

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-- Candidate flat NC domain obtained by reusing the logical-width cube and
using six bits for the 54 Phi81 lanes. This definition is diagnostic: no
production-refinement claim is attached to it. -/
def logicalWidthCube (variables : Nat) : FlatNcDomain where
  columnVariables := variables
  laneVariables := 6

@[simp] theorem logicalWidthCube_columnCount (variables : Nat) :
    (logicalWidthCube variables).columnCount = 2 ^ variables := by
  rfl

@[simp] theorem logicalWidthCube_laneCount (variables : Nat) :
    (logicalWidthCube variables).laneCount = 64 := by
  rfl

/-- Six lane bits are sufficient for every real Phi81 coefficient lane. -/
theorem logicalWidthCube_covers_lanes (variables : Nat) :
    ringDegree <= (logicalWidthCube variables).laneCount := by
  simp [ringDegree, logicalWidthCube, FlatNcDomain.laneCount]

/-- The paper's direct square row/column carrier cannot instantiate any
production `SemanticShape`: a complete Phi81 carrier is never a Boolean-cube
cardinality. A production proof must therefore refine the independent Section
7.3 relation through a rectangular arithmetization such as Split-NC, rather
than pretending the old `ColumnLayout` applies. -/
theorem no_paperColumnLayout_for_carrier
    (shape : SemanticShape) :
    ¬ Nonempty
      (PaperJoint.UnifiedSources.ColumnLayout
        shape.rowVariables shape.carrierWidth) := by
  simpa [SemanticShape.carrierWidth] using
    PaperJoint.Necessity.DomainSeparation.no_columnLayout_for_completeCarrier
      shape.logicalWidth shape.rowVariables

/-- When the original logical width is a Boolean-cube cardinality, carrier
completion strictly adds coordinates. Equality is impossible because every
complete Phi81 carrier is divisible by three and no power of two is. -/
theorem logicalWidth_lt_carrierWidth
    (shape : SemanticShape)
    (variables : Nat)
    (logicalWidthEq : shape.logicalWidth = 2 ^ variables) :
    shape.logicalWidth < shape.carrierWidth := by
  have notEqual : shape.logicalWidth ≠ shape.carrierWidth := by
    intro equal
    apply PaperJoint.Necessity.DomainSeparation.carrierWidth_ne_twoPow
      shape.logicalWidth variables
    calc
      PaperJoint.Phi81CarrierLayout.carrierWidth shape.logicalWidth =
          shape.carrierWidth := rfl
      _ = shape.logicalWidth := equal.symm
      _ = 2 ^ variables := logicalWidthEq
  exact Nat.lt_of_le_of_ne shape.logicalWidth_le_carrierWidth notEqual

/-- The first coordinate introduced by carrier completion. It is a real
semantic coordinate for carried assignments, not arithmetization padding. -/
def firstCompletedTail
    (shape : SemanticShape)
    (variables : Nat)
    (logicalWidthEq : shape.logicalWidth = 2 ^ variables) :
    Fin shape.carrierWidth :=
  ⟨shape.logicalWidth,
    logicalWidth_lt_carrierWidth shape variables logicalWidthEq⟩

@[simp] theorem firstCompletedTail_val
    (shape : SemanticShape)
    (variables : Nat)
    (logicalWidthEq : shape.logicalWidth = 2 ^ variables) :
    (firstCompletedTail shape variables logicalWidthEq).val =
      shape.logicalWidth := by
  rfl

/-- The logical-width column cube cannot index the first completed tail
coordinate. -/
theorem firstCompletedTail_outside_columnCube
    (shape : SemanticShape)
    (variables : Nat)
    (logicalWidthEq : shape.logicalWidth = 2 ^ variables) :
    ¬ ((firstCompletedTail shape variables logicalWidthEq).val <
      (logicalWidthCube variables).columnCount) := by
  rw [firstCompletedTail_val, logicalWidthCube_columnCount, <- logicalWidthEq]
  exact Nat.lt_irrefl shape.logicalWidth

/-- Inclusion-necessity theorem: covering all real lanes does not rescue a
column cube that was sized only for the original power-of-two width. -/
theorem logicalWidthCube_does_not_cover
    (shape : SemanticShape)
    (variables : Nat)
    (logicalWidthEq : shape.logicalWidth = 2 ^ variables) :
    ¬ ((logicalWidthCube variables).Covers shape) := by
  intro covers
  have completedTail :=
    logicalWidth_lt_carrierWidth shape variables logicalWidthEq
  have carrierAtMostLogical : shape.carrierWidth <= shape.logicalWidth := by
    calc
      shape.carrierWidth <= (logicalWidthCube variables).columnCount := covers.1
      _ = 2 ^ variables := logicalWidthCube_columnCount variables
      _ = shape.logicalWidth := logicalWidthEq.symm
  omega

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.CarrierCoverage
