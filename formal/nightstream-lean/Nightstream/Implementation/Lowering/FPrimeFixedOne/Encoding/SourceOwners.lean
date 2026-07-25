import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.NormalForm
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Step
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal
import Nightstream.Implementation.Lowering.Goldilocks.SourceAlignment

/-!
Contract: exact structural owner paths and normal-form site alignment for the
fixed-one step and terminal typed programs.

Owns:
- named owner paths for every primitive and branch node in the two frozen
  typed programs;
- definitional equality between those lists and the generic source-owner
  skeleton;
- exact coordinate-owner/ordinal alignment for the finite rewrite classes.

Does not own: physical receipts, call recipes, row satisfaction, Rust/R1CS
refinement, or generated artifacts.

This file derives paths from the typed program structure.  It does not inspect
Rust or an existing constraint matrix.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.SourceOwners

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.SourceAlignment
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

/-! ## Step owner paths -/

def stepApplyPath : OwnerPath :=
  .root

def stepSelectorPath : OwnerPath :=
  .rest stepApplyPath

def stepBranchPath : OwnerPath :=
  .rest stepSelectorPath

def stepBaseStateEqualPath : OwnerPath :=
  .trueArm stepBranchPath

def stepBaseAssertionPath : OwnerPath :=
  .rest stepBaseStateEqualPath

def stepBaseDefaultPath : OwnerPath :=
  .rest stepBaseAssertionPath

def stepRecursiveHashPriorPath : OwnerPath :=
  .falseArm stepBranchPath

def stepRecursiveFreshPublicPath : OwnerPath :=
  .rest stepRecursiveHashPriorPath

def stepRecursiveEncodePath : OwnerPath :=
  .rest stepRecursiveFreshPublicPath

def stepRecursiveEncodedEqualPath : OwnerPath :=
  .rest stepRecursiveEncodePath

def stepRecursiveAssertionPath : OwnerPath :=
  .rest stepRecursiveEncodedEqualPath

def stepRecursiveNifsPath : OwnerPath :=
  .rest stepRecursiveAssertionPath

def stepContinuationHashPath : OwnerPath :=
  .continuation stepBranchPath

/-- Exact non-input physical owner order of the fixed-one step body. -/
def stepBodyOwners : List PhysicalOwner :=
  [.typed (.instruction stepApplyPath),
    .typed (.instruction stepSelectorPath),
    .branchActivation stepBranchPath true,
    .branchActivation stepBranchPath false,
    .typed (.instruction stepBaseStateEqualPath),
    .typed (.instruction stepBaseAssertionPath),
    .typed (.instruction stepBaseDefaultPath),
    .typed (.instruction stepRecursiveHashPriorPath),
    .typed (.instruction stepRecursiveFreshPublicPath),
    .typed (.instruction stepRecursiveEncodePath),
    .typed (.instruction stepRecursiveEncodedEqualPath),
    .typed (.instruction stepRecursiveAssertionPath),
    .typed (.instruction stepRecursiveNifsPath),
    .typed (.branch stepBranchPath),
    .typed (.instruction stepContinuationHashPath)]

def stepOwners (parameters : Parameters) : List PhysicalOwner :=
  .prelude ::
    inputOwners (stepInputSchema parameters) ++
      stepBodyOwners

/-- Kernel reduction of the exact typed step AST yields the named physical
owner order. -/
theorem stepProgramOwnersExact
    (parameters : Parameters) :
    programOwners
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.program
          parameters) =
      stepOwners parameters :=
  rfl

/-! ## Terminal owner paths -/

def terminalSelectorPath : OwnerPath :=
  .root

def terminalBranchPath : OwnerPath :=
  .rest terminalSelectorPath

def terminalBaseStateEqualPath : OwnerPath :=
  .trueArm terminalBranchPath

def terminalBaseAssertionPath : OwnerPath :=
  .rest terminalBaseStateEqualPath

def terminalRecursiveHashPriorPath : OwnerPath :=
  .falseArm terminalBranchPath

def terminalRecursiveFreshPublicPath : OwnerPath :=
  .rest terminalRecursiveHashPriorPath

def terminalRecursiveEncodePath : OwnerPath :=
  .rest terminalRecursiveFreshPublicPath

def terminalRecursiveEncodedEqualPath : OwnerPath :=
  .rest terminalRecursiveEncodePath

def terminalRecursivePriorAssertionPath : OwnerPath :=
  .rest terminalRecursiveEncodedEqualPath

def terminalRecursiveRunningCheckPath : OwnerPath :=
  .rest terminalRecursivePriorAssertionPath

def terminalRecursiveRunningAssertionPath : OwnerPath :=
  .rest terminalRecursiveRunningCheckPath

def terminalRecursiveFreshCheckPath : OwnerPath :=
  .rest terminalRecursiveRunningAssertionPath

def terminalRecursiveFreshAssertionPath : OwnerPath :=
  .rest terminalRecursiveFreshCheckPath

/-- Exact non-input physical owner order of the fixed-one terminal body.  The
join owner remains present even though its result schema has zero coordinates.
-/
def terminalBodyOwners : List PhysicalOwner :=
  [.typed (.instruction terminalSelectorPath),
    .branchActivation terminalBranchPath true,
    .branchActivation terminalBranchPath false,
    .typed (.instruction terminalBaseStateEqualPath),
    .typed (.instruction terminalBaseAssertionPath),
    .typed (.instruction terminalRecursiveHashPriorPath),
    .typed (.instruction terminalRecursiveFreshPublicPath),
    .typed (.instruction terminalRecursiveEncodePath),
    .typed (.instruction terminalRecursiveEncodedEqualPath),
    .typed (.instruction terminalRecursivePriorAssertionPath),
    .typed (.instruction terminalRecursiveRunningCheckPath),
    .typed (.instruction terminalRecursiveRunningAssertionPath),
    .typed (.instruction terminalRecursiveFreshCheckPath),
    .typed (.instruction terminalRecursiveFreshAssertionPath),
    .typed (.branch terminalBranchPath)]

def terminalOwners (parameters : Parameters) : List PhysicalOwner :=
  .prelude ::
    inputOwners (terminalInputSchema parameters) ++
      terminalBodyOwners

/-- Kernel reduction of the exact typed terminal AST yields the named physical
owner order. -/
theorem terminalProgramOwnersExact
    (parameters : Parameters) :
    programOwners
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.program
          parameters) =
      terminalOwners parameters :=
  rfl

/-! ## Exact finite-class site alignment -/

/-- The step rewrite specifications name exactly every running-join
coordinate, in codec order, plus the two retained assertion instructions. -/
structure StepNormalFormAligned
    (parameters : Parameters)
    (specifications : NormalForm.StepSpecifications) : Prop where
  joinCoordinatesExact :
    specifications.joinCoordinates.map
        (fun specification =>
          (specification.owner, specification.firstOrdinal)) =
      (List.range parameters.widths.running).map
        (fun ordinal =>
          (PhysicalOwner.typed (.branch stepBranchPath), ordinal))
  baseEndpointOwner :
    specifications.baseEndpoint.owner =
      .typed (.instruction stepBaseAssertionPath)
  baseEndpointOrdinal :
    specifications.baseEndpoint.firstOrdinal = 0
  recursivePriorLinkOwner :
    specifications.recursivePriorLink.owner =
      .typed (.instruction stepRecursiveAssertionPath)
  recursivePriorLinkOrdinal :
    specifications.recursivePriorLink.firstOrdinal = 0

namespace StepNormalFormAligned

theorem joinCoordinateCount
    {parameters : Parameters}
    {specifications : NormalForm.StepSpecifications}
    (aligned : StepNormalFormAligned parameters specifications) :
    specifications.joinCoordinates.length =
      parameters.widths.running := by
  have lengths := congrArg List.length aligned.joinCoordinatesExact
  simpa only [List.length_map, List.length_range] using lengths

/-- Exact selected local step cost after source-site alignment. -/
theorem canonicalLocalCost
    {parameters : Parameters}
    {specifications : NormalForm.StepSpecifications}
    (aligned : StepNormalFormAligned parameters specifications) :
    Nightstream.Implementation.Lowering.Goldilocks.NormalFormComposition.totalCost
        (NormalForm.stepClasses specifications)
        (Nightstream.Implementation.Lowering.Goldilocks.NormalFormComposition.canonicalSelection
          (NormalForm.stepClasses specifications)) =
      ⟨parameters.widths.running + 2, 0, 0, 0⟩ := by
  rw [NormalForm.stepCanonicalLocalCost,
    aligned.joinCoordinateCount]

end StepNormalFormAligned

/-- The terminal rewrite specifications name exactly the four retained
assertion instructions. -/
structure TerminalNormalFormAligned
    (specifications : NormalForm.TerminalSpecifications) : Prop where
  baseEndpointOwner :
    specifications.baseEndpoint.owner =
      .typed (.instruction terminalBaseAssertionPath)
  baseEndpointOrdinal :
    specifications.baseEndpoint.firstOrdinal = 0
  recursivePriorLinkOwner :
    specifications.recursivePriorLink.owner =
      .typed (.instruction terminalRecursivePriorAssertionPath)
  recursivePriorLinkOrdinal :
    specifications.recursivePriorLink.firstOrdinal = 0
  runningRelationOwner :
    specifications.runningRelation.owner =
      .typed (.instruction terminalRecursiveRunningAssertionPath)
  runningRelationOrdinal :
    specifications.runningRelation.firstOrdinal = 0
  freshRelationOwner :
    specifications.freshRelation.owner =
      .typed (.instruction terminalRecursiveFreshAssertionPath)
  freshRelationOrdinal :
    specifications.freshRelation.firstOrdinal = 0

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.SourceOwners
