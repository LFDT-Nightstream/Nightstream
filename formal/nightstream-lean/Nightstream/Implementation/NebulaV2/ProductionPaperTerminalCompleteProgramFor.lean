import Nightstream.Implementation.NebulaV2.ProductionPaperTerminalOpeningRowsFor
import Nightstream.Implementation.NebulaV2.ProductionPaperTerminalTypedFoldRowsFor

/-!
Contract: one verifier-owned typed program for the complete paper terminal.

The program fixes the translated numeric terminal fold, all fourteen common-
witness bundle-opening layouts, all fourteen CE-core layouts, and every
zero-copy alias to the final NIFS carrier. `RowsSatisfied` reads one typed
Goldilocks assignment. Its fields are only satisfaction of the exact row
families selected by this program.

The program also emits one canonical finite typed row list. Satisfaction of
that list is proved equivalent to all row-family fields below. The main
constructor derives `ProductionPaperTerminalOpeningRowsFor.Family`. That
semantic family is not an acceptance input. This prevents a prover from
selecting terminal opening or CE layouts after the verifier key is fixed.

This module does not own finite numeric allocation of typed columns, a compact
terminal backend, generated artifact serialization, Rust refinement, or
cryptographic reductions.

Assurance tier: static terminal-program authority and row composition.
-/

set_option autoImplicit false
set_option maxHeartbeats 300000
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductionPaperTerminalCompleteProgramFor

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductNifsRunningCoordinatesFor
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.Protocol.NebulaV2.Terminal
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev FullShape
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductionPaperTerminalOpeningRowsFor.FullShape rowVariables logicalWidth
    publicFits

/-- Static rows and aliases for one post-PiDEC child. No assignment or row
satisfaction occurs in this object. -/
structure ChildProgram
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (manifest : SeedSchedule.Manifest)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (outputLayout : ProductionProductNifsOutputRowsFor.Layout rowVariables)
    (child : FoldedChild) where
  opening : TerminalBundleOpeningRows.Layout manifest
    (FullShape rowVariables logicalWidth publicFits) operationsShape snapshotShape
  configExact : TerminalProductCommitmentBridge.config opening = config
  oneAlias : opening.one = TerminalBundleOpeningRows.Layout.numericColumn 0
  commitmentAlias : forall component : Fin 4, forall row lane,
    opening.commitmentColumn (componentAt component) row lane =
      TerminalBundleOpeningRows.Layout.numericColumn
        (outputLayout.carrierColumn
          (ProductionProductNifsOutputRowsFor.commitmentIndex child component
            row lane))
  core : ProductionPaperTerminalCoreRowsFor.Layout opening
  coreAliases : ProductionPaperTerminalRunningPlacementFor.Aliases
    (ProductionPaperTerminalOpeningRowsFor.outputCarrier outputLayout) core child

/-- Exact row satisfaction for one statically selected child program. -/
structure ChildProgram.RowsSatisfied
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {manifest : SeedSchedule.Manifest}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {outputLayout : ProductionProductNifsOutputRowsFor.Layout rowVariables}
    {child : FoldedChild}
    (program : ChildProgram manifest config outputLayout child)
    (system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits))
    (assignment : ColumnId -> F) : Prop where
  opening : Nightstream.Implementation.Lowering.Goldilocks.Satisfies
    (TerminalBundleOpeningRows.rows program.opening) assignment
  publicProjection : Nightstream.Implementation.Lowering.Goldilocks.Satisfies
    (ProductionPaperTerminalCoreRowsFor.publicRows program.core) assignment
  evaluations :
    Nightstream.Implementation.R1CS.Phi81SharedEvaluationRows.RowsSatisfied
      program.core.evaluator system assignment

namespace ChildProgram

/-- One child's exact finite row list: common-witness opening, public
projection, and complete sparse Phi81 evaluation. -/
noncomputable def rows
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {manifest : SeedSchedule.Manifest}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {outputLayout : ProductionProductNifsOutputRowsFor.Layout rowVariables}
    {child : FoldedChild}
    (program : ChildProgram manifest config outputLayout child)
    (system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits)) : List OwnedRow :=
  TerminalBundleOpeningRows.rows program.opening ++
    ProductionPaperTerminalCoreRowsFor.publicRows program.core ++
    Nightstream.Implementation.R1CS.Phi81SharedEvaluationRows.rows
      program.core.evaluator system

/-- The finite child program is exact. No opening, public, tensor, product,
or output row family exists only in a quantified side condition. -/
theorem rows_satisfied_iff
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {manifest : SeedSchedule.Manifest}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {outputLayout : ProductionProductNifsOutputRowsFor.Layout rowVariables}
    {child : FoldedChild}
    (program : ChildProgram manifest config outputLayout child)
    (system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits))
    (assignment : ColumnId -> F) :
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies
        (program.rows system) assignment ↔
      program.RowsSatisfied system assignment := by
  rw [rows, satisfies_append_iff, satisfies_append_iff,
    Nightstream.Implementation.R1CS.Phi81SharedEvaluationRows.rows_satisfied_iff]
  constructor
  · rintro ⟨⟨opening, publicProjection⟩, evaluations⟩
    exact ⟨opening, publicProjection, evaluations⟩
  · intro satisfied
    exact ⟨⟨satisfied.opening, satisfied.publicProjection⟩,
      satisfied.evaluations⟩

/-- Row satisfaction builds the legacy local row record. No semantic opening
or terminal-relation result is supplied. -/
def childRows
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {manifest : SeedSchedule.Manifest}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {outputLayout : ProductionProductNifsOutputRowsFor.Layout rowVariables}
    {child : FoldedChild}
    {program : ChildProgram manifest config outputLayout child}
    {system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits)}
    {assignment : ColumnId -> F}
    (satisfied : program.RowsSatisfied system assignment) :
    ProductionPaperTerminalOpeningRowsFor.ChildRows
      (manifest := manifest) config outputLayout assignment child where
  layout := program.opening
  typedRows := satisfied.opening
  configExact := program.configExact
  oneAlias := program.oneAlias
  commitmentAlias := program.commitmentAlias

/-- Row satisfaction builds the exact CE-core row evidence for this child. -/
def coreRows
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {manifest : SeedSchedule.Manifest}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {outputLayout : ProductionProductNifsOutputRowsFor.Layout rowVariables}
    {child : FoldedChild}
    {program : ChildProgram manifest config outputLayout child}
    {system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits)}
    {assignment : ColumnId -> F}
    (satisfied : program.RowsSatisfied system assignment) :
    ProductionPaperTerminalCoreRowsFor.RowsEvidence program.core system
      assignment where
  publicRows := satisfied.publicProjection
  evaluationRows := satisfied.evaluations

end ChildProgram

/-- One complete static terminal program. The fold program owns the seed
manifest and output carrier used by every child. -/
structure Program
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (foldProgram : ProductionPaperTerminalFoldManifestFor.Program candidate
      rowVariables) where
  foldOwner : Nightstream.Implementation.Lowering.Goldilocks.PhysicalOwner
  foldFirstOrdinal : Nat
  child : (index : FoldedChild) ->
    ChildProgram foldProgram.fold.seedManifest config
      foldProgram.fold.nifsOutputLayout index

def Program.foldFrame
    {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {foldProgram : ProductionPaperTerminalFoldManifestFor.Program candidate
      rowVariables}
    (program : Program candidate config foldProgram) :
    ProductionPaperTerminalTypedFoldRowsFor.Frame candidate rowVariables where
  program := foldProgram
  owner := program.foldOwner
  firstOrdinal := program.foldFirstOrdinal

/-- The complete finite row suffix for all children. Children are enumerated
in their canonical `Fin 14` order. Keeping this list separate prevents proof
elaboration from expanding every child when a theorem only needs the common
fold prefix. -/
noncomputable def Program.childrenRows
    {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {foldProgram : ProductionPaperTerminalFoldManifestFor.Program candidate
      rowVariables}
    (program : Program candidate config foldProgram)
    (system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits)) : List OwnedRow :=
  (canonicalFinIndices foldedChildCount).flatMap fun child =>
    (program.child child).rows system

/-- The complete terminal program as one finite typed row list. -/
noncomputable def Program.rows
    {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {foldProgram : ProductionPaperTerminalFoldManifestFor.Program candidate
      rowVariables}
    (program : Program candidate config foldProgram)
    (system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits)) : List OwnedRow :=
  program.foldFrame.rows ++ program.childrenRows system

/-- Terminal row acceptance is the exact prefix/suffix partition of the one
finite verifier-owned program. Both parts are fixed finite row lists. There is
no caller-selected or quantified row-family authority. -/
structure SplitRowsSatisfied
    (foldRows childrenRows : List OwnedRow)
    (assignment : ColumnId -> F) : Prop where
  fold : Nightstream.Implementation.Lowering.Goldilocks.Satisfies
    foldRows assignment
  children : Nightstream.Implementation.Lowering.Goldilocks.Satisfies
    childrenRows assignment

/-- Exact generic equivalence between a concatenated finite row list and its
prefix/suffix proof carrier. -/
theorem splitRowsSatisfied_iff
    (foldRows childrenRows : List OwnedRow)
    (assignment : ColumnId -> F) :
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies
        (foldRows ++ childrenRows) assignment ↔
      SplitRowsSatisfied foldRows childrenRows assignment := by
  rw [satisfies_append_iff]
  constructor
  · intro satisfied
    exact ⟨satisfied.1, satisfied.2⟩
  · intro accepted
    exact ⟨accepted.fold, accepted.children⟩

/-- The generic split-row predicate instantiated by one static terminal
program. The small generic carrier prevents Lean from normalizing the complete
dependent program when projecting either proof. -/
def Program.RowsSatisfied
    {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {foldProgram : ProductionPaperTerminalFoldManifestFor.Program candidate
      rowVariables}
    (program : Program candidate config foldProgram)
    (system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits))
    (assignment : ColumnId -> F) : Prop :=
  SplitRowsSatisfied program.foldFrame.rows
    (program.childrenRows system) assignment

namespace Program

/-- Opaque terminal acceptance has exactly the fixed fold/children split. The
equivalence prevents large consumers from normalizing the complete dependent
program merely to project one proof. -/
theorem rowsSatisfied_iff_split
    {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {foldProgram : ProductionPaperTerminalFoldManifestFor.Program candidate
      rowVariables}
    (program : Program candidate config foldProgram)
    (system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits))
    (assignment : ColumnId -> F) :
    program.RowsSatisfied system assignment ↔
      SplitRowsSatisfied program.foldFrame.rows
        (program.childrenRows system) assignment := by
  rfl

private theorem satisfies_flatMap_member
    {Index : Type} {parts : List Index}
    {rowsOf : Index -> List OwnedRow}
    {assignment : ColumnId -> F}
    (satisfied : Nightstream.Implementation.Lowering.Goldilocks.Satisfies
      (parts.flatMap rowsOf) assignment)
    {part : Index} (member : part ∈ parts) :
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies
      (rowsOf part) assignment := by
  induction parts with
  | nil => simp at member
  | cons head tail inductionHypothesis =>
      rw [List.flatMap_cons, satisfies_append_iff] at satisfied
      rcases List.mem_cons.mp member with rfl | tailMember
      · exact satisfied.1
      · exact inductionHypothesis satisfied.2 tailMember

private theorem mem_canonicalFinIndices
    {count : Nat} (index : Fin count) :
    index ∈ canonicalFinIndices count := by
  simp [canonicalFinIndices]

/-- Finite terminal-row satisfaction implies the exact rows for the selected
child. Since the child is arbitrary, this covers all fourteen children. -/
theorem RowsSatisfied.child
    {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {foldProgram : ProductionPaperTerminalFoldManifestFor.Program candidate
      rowVariables}
    {program : Program candidate config foldProgram}
    {system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits)}
    {assignment : ColumnId -> F}
    (accepted : program.RowsSatisfied system assignment)
    (child : FoldedChild) :
    (program.child child).RowsSatisfied system assignment := by
  have split :=
    (rowsSatisfied_iff_split program system assignment).1 accepted
  apply (ChildProgram.rows_satisfied_iff
    (program.child child) system assignment).1
  exact satisfies_flatMap_member split.children
    (mem_canonicalFinIndices child)

/-- Exact row count of the selected complete typed terminal program. -/
noncomputable def rowCount
    {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {foldProgram : ProductionPaperTerminalFoldManifestFor.Program candidate
      rowVariables}
    (program : Program candidate config foldProgram)
    (system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits)) : Nat :=
  (program.rows system).length

/-- Build all fourteen opening and CE-core records from one static program and
one row-satisfying assignment. The final-running layout equality is retained
from the numeric terminal row theorem. -/
noncomputable def family
    {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {foldProgram : ProductionPaperTerminalFoldManifestFor.Program candidate
      rowVariables}
    (program : Program candidate config foldProgram)
    {headers : FPrime.ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {assignment : ColumnId -> F}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact foldProgram.fold.priorLayout
      (ProductionPaperTerminalOpeningRowsFor.pulledAssignment assignment)
      headers priorPrefix value proof}
    (outputExact : recursive.nifsOutputLayout =
      foldProgram.fold.nifsOutputLayout)
    (satisfied : program.RowsSatisfied artifact.system assignment) :
    ProductionPaperTerminalOpeningRowsFor.Family
      (manifest := foldProgram.fold.seedManifest)
      statementId config artifact foldProgram.fold.priorLayout headers
      priorPrefix value proof assignment recursive := by
  exact
    { outputLayout := foldProgram.fold.nifsOutputLayout
      outputLayoutExact := outputExact.symm
      child := fun index => (program.child index).childRows (satisfied.child index)
      coreLayout := fun index => (program.child index).core
      coreAliases := fun index => (program.child index).coreAliases
      coreRows := fun index => (program.child index).coreRows (satisfied.child index) }

end Program

end Nightstream.Implementation.NebulaV2.ProductionPaperTerminalCompleteProgramFor
