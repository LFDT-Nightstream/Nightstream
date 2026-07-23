import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Refinement

/-!
Concrete generated-column refinement for the bounded running-`X` public-prefix
decoder.

Assurance tier: artifact-checked for the generated fixed public-prefix profile.

Owns: construction of a bounded 270-coordinate semantic fixture from the exact
generated `running[child].x` balanced-ternary assignment intervals; exact child
and carrier casts; the resulting primitive allocation equations; and the `54 × 5` live
/ ten-lane padding decoder.

Does not own: `CcsWitness.Z` or `CeWitness.Z`, the private witness suffix, the
full production assignment, combined-NC sparse rows, parent padding rows,
transcript sampling, recursive-state continuity, Ajtai binding, Poseidon2
internals, costs, or row-removal permission.

Emits constraints: none; correspondence theorem only.

| Stable stage path | Obligation | Authority |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed_projection.running_x_prefix_decoder.artifact_refinement` | Decode exact generated 41-digit intervals into the bounded public-prefix fixture | direct dataflow |

`decodedData` below is only a 270-coordinate fixture: the generator reads
`CeClaim.X`, so this construction cannot instantiate production
`Sources.Data.runningAssignments` when the committed witness has a larger
private suffix. The production refinement must instead decode the full packed
witness matrix `Z`. No `CeClaim.y_zcol`, output digest, or prover-carried child
sidecar occurs in this file.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.ArtifactRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

universe uState

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Inverse of the exact compact-child-to-production-child cast. -/
def childOfProduction
    (child : Fin productionGlobalParams.k) : Child :=
  Fin.cast childCount_eq_productionArity.symm child

@[simp] theorem productionChild_childOfProduction
    (child : Fin productionGlobalParams.k) :
    productionChild (childOfProduction child) = child := by
  apply Fin.ext
  rfl

@[simp] theorem childOfProduction_productionChild (child : Child) :
    childOfProduction (productionChild child) = child := by
  apply Fin.ext
  rfl

/-- Inverse of `Profile.semanticColumn`; this is an exact cast, not a
decoder lookup. -/
def logicalColumnOfSemantic
    (profile : Profile shape)
    (column : Fin shape.carrierWidth) : LogicalColumn :=
  Fin.cast profile.carrierWidth_eq column

@[simp] theorem semanticColumn_logicalColumnOfSemantic
    (profile : Profile shape)
    (column : Fin shape.carrierWidth) :
    profile.semanticColumn (logicalColumnOfSemantic profile column) =
      column := by
  apply Fin.ext
  rfl

@[simp] theorem logicalColumnOfSemantic_semanticColumn
    (profile : Profile shape)
    (column : LogicalColumn) :
    logicalColumnOfSemantic profile (profile.semanticColumn column) =
      column := by
  apply Fin.ext
  rfl

/-- Exact compact child corresponding to one semantic running source. -/
def childOfSemanticRunning
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (source : Fin shape.runningCount) : Child :=
  childOfProduction (context.alignment.productRunningIndex source)

/-- Bounded 270-coordinate running table decoded from generated public-`X`
columns. It is not the full committed production witness. -/
def decodedRunningAssignments
    (profile : Profile shape)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (assignment : PhysicalAssignment) :
    Fin shape.runningCount -> Fin shape.carrierWidth -> F :=
  fun source column =>
    decodedLogical Generated.sourceAllocationMap assignment
      (childOfSemanticRunning context source)
      (logicalColumnOfSemantic profile column)

/-- Construct a bounded semantic fixture from the public-`X` decoder. Every
non-running source component is retained from `template`; this definition is
not a decoder for the full production witness matrix. -/
def decodedData
    (profile : Profile shape)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (template : Sources.Data shape)
    (assignment : PhysicalAssignment) : Sources.Data shape :=
  { template with
    runningAssignments := decodedRunningAssignments profile context assignment }

@[simp] theorem decodedData_runningAssignments
    (profile : Profile shape)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (template : Sources.Data shape)
    (assignment : PhysicalAssignment)
    (source : Fin shape.runningCount)
    (column : Fin shape.carrierWidth) :
    (decodedData profile context template assignment).runningAssignments
        source column =
      decodedLogical Generated.sourceAllocationMap assignment
        (childOfSemanticRunning context source)
        (logicalColumnOfSemantic profile column) := by
  rfl

/-- The bounded running table of `decodedData` is exactly the generated
public-`X` physical assignment decoder. -/
theorem rawRunningAssignments_decodedData
    (profile : Profile shape)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (template : Sources.Data shape)
    (assignment : PhysicalAssignment)
    (child : Fin productionGlobalParams.k)
    (column : Fin shape.carrierWidth) :
    DelayedRawChildren.rawRunningAssignments context
        (decodedData profile context template assignment) child column =
      decodedLogical Generated.sourceAllocationMap assignment
        (childOfProduction child)
        (logicalColumnOfSemantic profile column) := by
  simp [DelayedRawChildren.rawRunningAssignments,
    DelayedRawChildren.rawRunningAssignment, decodedData,
    decodedRunningAssignments, childOfSemanticRunning]

/-- Because the bounded fixture is constructed from the generated public-`X`
map, its primitive source-column contract is derived definitionally. This is
not a theorem of full-witness authority. -/
theorem sourceAllocationRowsBind_decodedData
    (profile : Profile shape)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (template : Sources.Data shape)
    (assignment : PhysicalAssignment) :
    SourceAllocationRowsBind profile context
      (decodedData profile context template assignment)
      Generated.sourceAllocationMap assignment := by
  unfold SourceAllocationRowsBind SourceAllocationEquation
  intro child column
  have raw := rawRunningAssignments_decodedData profile context template
    assignment (productionChild child) (profile.semanticColumn column)
  change decodedLogical Generated.sourceAllocationMap assignment child column =
    DelayedRawChildren.rawRunningAssignments context
      (decodedData profile context template assignment)
      (productionChild child) (profile.semanticColumn column)
  simpa only [childOfProduction_productionChild,
    logicalColumnOfSemantic_semanticColumn] using raw.symm

/-- Exact reconstruction from the complete generated balanced-ternary
interval for one raw child coordinate. -/
theorem decodedScalar_eq_rawRunningAssignment
    (profile : Profile shape)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (template : Sources.Data shape)
    (assignment : PhysicalAssignment)
    (child : Child)
    (column : LogicalColumn) :
    (Generated.sourceAllocationMap.allocation child column).decode assignment =
      DelayedRawChildren.rawRunningAssignments context
        (decodedData profile context template assignment)
        (productionChild child) (profile.semanticColumn column) := by
  have raw := rawRunningAssignments_decodedData profile context template
    assignment (productionChild child) (profile.semanticColumn column)
  simpa [decodedLogical] using raw.symm

/-- Every live `(lane, block)` cell reads the exact generated final column
for logical coordinate `block * 54 + lane`. -/
theorem decodedVirtual_live_eq_rawRunningAssignment
    (profile : Profile shape)
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (template : Sources.Data shape)
    (assignment : PhysicalAssignment)
    (child : Child)
    (lane : PackedLane)
    (block : LiveBlock) :
    decodedVirtual Generated.sourceAllocationMap assignment child
        (virtualLaneOfLive lane) (virtualBlockOfLive block) =
      DelayedRawChildren.rawRunningAssignments context
        (decodedData profile context template assignment)
        (productionChild child)
        (profile.semanticColumn
          (logicalColumnAt { lane := lane, block := block })) := by
  rw [decodedVirtual_live]
  have raw := rawRunningAssignments_decodedData profile context template
    assignment (productionChild child)
      (profile.semanticColumn
        (logicalColumnAt { lane := lane, block := block }))
  simpa only [childOfProduction_productionChild,
    logicalColumnOfSemantic_semanticColumn] using raw.symm

/-- The ten Boolean-cube lane-padding positions are literal decoder zeros;
they do not read or alias any generated assignment column. -/
theorem decodedVirtual_paddingLane_zero
    (assignment : PhysicalAssignment)
    (child : Child)
    (lane : VirtualLane)
    (block : VirtualBlock)
    (padding : packedLaneCount <= lane.val) :
    decodedVirtual Generated.sourceAllocationMap assignment child lane
        block = 0 :=
  decodedVirtual_lanePadding_zero Generated.sourceAllocationMap
    assignment child lane block padding

/-- Exact physical ownership from the artifact: no two raw logical
coordinates use the same complete selectively lowered allocation. -/
theorem allocation_uniqueOwner :
    Function.Injective fun address : Child × LogicalColumn =>
      Generated.sourceAllocationMap.allocation address.1 address.2 :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Exact.sourceAllocationMap_injective

/-- Stronger physical ownership: distinct logical coordinates have disjoint
41-column final-assignment intervals. -/
theorem allocation_intervals_nonoverlap
    (left right : Child × LogicalColumn) (different : left ≠ right) :
    let leftRecord := Generated.allocationAt left.1 left.2
    let rightRecord := Generated.allocationAt right.1 right.2
    leftRecord.finalStart + leftRecord.width ≤ rightRecord.finalStart ∨
      rightRecord.finalStart + rightRecord.width ≤ leftRecord.finalStart :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Exact.finalIntervals_nonoverlap
    left right different

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.ArtifactRefinement
