import Nightstream.SuperNeo.Concrete.Parameters
import Nightstream.SuperNeo.Folding.PiCCS
import Nightstream.SuperNeo.Folding.PiRLC
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryAffineSound
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryNestedOwners
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryPointBindingSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryParentCeSerialization
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Projection
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryProjectionSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryRecursiveAccumulatorSound
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistorySumcheckArtifact
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryTerminalAccumulatorArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryTranscriptSound
import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictSound
import Nightstream.Implementation.R1CS.Correspondence.Sumcheck.SumcheckRoundComplete
import Nightstream.Protocol.FPrime.Step

/-!
Concrete row-decoding boundary for the production F' NIFS verifier.

This module deliberately does not mention the configurable `Step.Semantics.nifsVerify`
callback.  Instead it records the exact verifier facts proved by the generated
recursive and terminal row families.  The two supported production batches have
their actual arities: one fresh input at bootstrap and one fresh input plus all
fourteen running inputs at the terminal fold.

The generated projection rows prove one accepted polynomial identity for each
native PiRLC output component.  Exact coefficient equality is then obtained, or
the result names the precise nonzero-polynomial root event.  Affine shape/glue,
strict PiDEC, and point binding are carried in the accepted certificate.

The verifier at the end of this module is executable. Its proof object is only
a field-valued wire assignment; it carries no acceptance bit or theorem. On
success the verifier decodes the strict-PiDEC parent and child claims into the
next running accumulator. Generated-row satisfaction is therefore the
authority for this row-decoded checklist. It is not yet authority for the
paper-level NIFS transition: a separate bridge must construct and discharge the
independent PiCCS/PiRLC/PiDEC semantics and their composition theorem.
-/

namespace Nightstream.Assurance.FPrimeConcreteNifs

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.ProjectionCheck
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles
open Nightstream.Implementation.R1CS.FPrimeFullHistorySumcheckArtifact

abbrev Digest := FPrimeFullHistoryTranscriptSound.Digest
abbrev Fresh := FPrimeFullHistoryTranscriptSound.Fresh

/-- Production recursive NIFS: one fresh statement and no synthetic running
statement on the first fold. -/
def recursiveArity : BatchArity Concrete.productionGlobalParams :=
  BatchArity.bootstrap Concrete.productionGlobalParams 1 (by decide) (by decide)

/-- Production terminal NIFS: one fresh statement and the full `k = 14`
running product. -/
def terminalArity : BatchArity Concrete.productionGlobalParams :=
  BatchArity.active Concrete.productionGlobalParams 1 (by decide) (by decide)

theorem recursive_total : recursiveArity.total = 1 := by
  rfl

theorem terminal_total : terminalArity.total = 15 := by
  rfl

/-- The order-sensitive connection between the generated projection census and
the Rust verifier's 31 output components.  `exact` is coefficient equality,
not merely equality at the sampled challenge. -/
structure ProjectionCertificate
    (arity : BatchArity Concrete.productionGlobalParams)
    (roles : List Role)
    (traces : List ProjectionProgram.ProjectionTrace)
    (assignment : Nat → Nat) : Prop where
  nativeOrder : roles = nativeVerifierOrder
  aligned : roles.length = traces.length
  pairArity : ∀ trace ∈ traces, trace.pairs.length = arity.total
  pairWidths : ∀ trace ∈ traces, ∀ pair ∈ trace.pairs,
    pair.rhoColumns.length = 54 ∧ pair.inputColumns.length = 54
  outputWidth : ∀ trace ∈ traces, trace.outputColumns.length = 54
  quotientWidth : ∀ trace ∈ traces, trace.quotientColumns.length = 53
  exact : BatchExact (ProjectionProgram.BatchIdentity traces assignment)

namespace ProjectionCertificate

/-- Transport one native role index to the identity at the same position. -/
def identityIndex
    {arity : BatchArity Concrete.productionGlobalParams}
    {roles : List Role}
    {traces : List ProjectionProgram.ProjectionTrace}
    {assignment : Nat → Nat}
    (certificate : ProjectionCertificate arity roles traces assignment)
    (index : Fin roles.length) :
    Fin (ProjectionProgram.BatchIdentity traces assignment).length :=
  ⟨index.val, by
    have aligned : roles.length =
        (ProjectionProgram.BatchIdentity traces assignment).length := by
      simpa [ProjectionProgram.BatchIdentity] using certificate.aligned
    exact aligned ▸ index.isLt⟩

/-- Every native component has coefficient equality at its aligned identity;
consumers need not re-use the weaker sampled-point equation. -/
theorem exactAt
    {arity : BatchArity Concrete.productionGlobalParams}
    {roles : List Role}
    {traces : List ProjectionProgram.ProjectionTrace}
    {assignment : Nat → Nat}
    (certificate : ProjectionCertificate arity roles traces assignment)
    (index : Fin roles.length) :
    ((ProjectionProgram.BatchIdentity traces assignment).get
      (certificate.identityIndex index)).Exact := by
  apply certificate.exact
  exact List.get_mem _ (certificate.identityIndex index)

end ProjectionCertificate

private theorem recursive_trace_count : recursiveTraces.length = 31 := by
  native_decide

private theorem terminal_trace_count : terminalTraces.length = 31 := by
  native_decide

private theorem recursive_pair_arity :
    ∀ trace ∈ recursiveTraces, trace.pairs.length = recursiveArity.total := by
  native_decide

private theorem terminal_pair_arity :
    ∀ trace ∈ terminalTraces, trace.pairs.length = terminalArity.total := by
  native_decide

theorem recursive_role_alignment (assignment : Nat → Nat) :
    recursiveRoles.length =
      (ProjectionProgram.BatchIdentity recursiveTraces assignment).length := by
  rw [role_census.1]
  simp only [ProjectionProgram.BatchIdentity, List.length_map]
  exact recursive_trace_count.symm

theorem terminal_role_alignment (assignment : Nat → Nat) :
    terminalRoles.length =
      (ProjectionProgram.BatchIdentity terminalTraces assignment).length := by
  rw [role_census.2]
  simp only [ProjectionProgram.BatchIdentity, List.length_map]
  exact terminal_trace_count.symm

private theorem recursive_pair_widths :
    ∀ trace ∈ recursiveTraces, ∀ pair ∈ trace.pairs,
      pair.rhoColumns.length = 54 ∧ pair.inputColumns.length = 54 := by
  intro trace traceMember pair pairMember
  exact trace_pair_widths trace
    (List.mem_append_left terminalTraces traceMember) pair pairMember

private theorem terminal_pair_widths :
    ∀ trace ∈ terminalTraces, ∀ pair ∈ trace.pairs,
      pair.rhoColumns.length = 54 ∧ pair.inputColumns.length = 54 := by
  intro trace traceMember pair pairMember
  exact trace_pair_widths trace
    (List.mem_append_right recursiveTraces traceMember) pair pairMember

private theorem recursive_output_width :
    ∀ trace ∈ recursiveTraces, trace.outputColumns.length = 54 := by
  intro trace traceMember
  rcases trace_layouts trace
      (List.mem_append_left terminalTraces traceMember) with
    ⟨_, _, _, _, _, _, _, _, _, _, _, _, outputWidth, _, _⟩
  exact outputWidth

private theorem recursive_quotient_width :
    ∀ trace ∈ recursiveTraces, trace.quotientColumns.length = 53 := by
  intro trace traceMember
  rcases trace_layouts trace
      (List.mem_append_left terminalTraces traceMember) with
    ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, quotientWidth, _⟩
  exact quotientWidth

private theorem terminal_output_width :
    ∀ trace ∈ terminalTraces, trace.outputColumns.length = 54 := by
  intro trace traceMember
  rcases trace_layouts trace
      (List.mem_append_right recursiveTraces traceMember) with
    ⟨_, _, _, _, _, _, _, _, _, _, _, _, outputWidth, _, _⟩
  exact outputWidth

private theorem terminal_quotient_width :
    ∀ trace ∈ terminalTraces, trace.quotientColumns.length = 53 := by
  intro trace traceMember
  rcases trace_layouts trace
      (List.mem_append_right recursiveTraces traceMember) with
    ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, quotientWidth, _⟩
  exact quotientWidth

abbrev Owner := OwnerCertificate.Owner

/-- Exact residual owners that sit between the named PiCCS/PiRLC phases.
Keeping the owners explicit prevents their rows from disappearing behind the
larger phase labels. -/
def recursiveResidualOwners : List Owner :=
  FPrimeFullHistoryNestedOwners.recursivePiCcsResidualOwners ++
    FPrimeFullHistoryNestedOwners.recursivePiRlcResidualOwners

def terminalResidualOwners : List Owner :=
  FPrimeFullHistoryNestedOwners.terminalPiCcsResidualOwners ++
    FPrimeFullHistoryNestedOwners.terminalPiRlcResidualOwners

def OwnersRows (owners : List Owner) (assignment : Nat → Nat) : Prop :=
  ∀ owner ∈ owners, Satisfies owner.rows assignment

def OwnersAccepted (owners : List Owner) (assignment : Nat → Nat) : Prop :=
  ∀ owner ∈ owners, owner.Accepted assignment

def OwnersExecution
    (field : CanonicalU64Complete.FieldInverse)
    (owners : List Owner) (assignment : Nat → Nat) : Type :=
  ∀ owner ∈ owners, owner.ExecutionWitness field assignment

theorem owners_sound {owners : List Owner} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (rows : OwnersRows owners assignment) :
    OwnersAccepted owners assignment := by
  intro owner member
  exact OwnerCertificate.Owner.sound canonical one (rows owner member)

theorem owners_complete {owners : List Owner} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : OwnersAccepted owners assignment) :
    OwnersRows owners assignment := by
  intro owner member
  exact OwnerCertificate.Owner.complete canonical one (accepted owner member)

theorem owners_execution_complete
    {field : CanonicalU64Complete.FieldInverse}
    {owners : List Owner} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (execution : OwnersExecution field owners assignment) :
    OwnersRows owners assignment := by
  intro owner member
  exact OwnerCertificate.Owner.execution_complete canonical one
    (execution owner member)

/-- The terminal PiCCS authority range contains a direct strict-PiDEC parent
check followed by exact affine pins. -/
structure TerminalAuthorityRows (assignment : Nat → Nat) : Prop where
  piDec : Satisfies FPrimeFullHistoryPiDec.terminalCeRows assignment
  tail : Satisfies FPrimeFullHistoryPiCcsTerminalAuthorityTail.rows assignment

structure TerminalAuthorityAccepted (assignment : Nat → Nat) : Prop where
  piDec : PiDecStrictCompiler.Accepted FPrimeFullHistoryPiDec.layout
    (Relabel.assignment FPrimeFullHistoryPiDec.terminalCeColumnMap assignment)
  tail : ∀ pin ∈ FPrimeFullHistoryPiCcsTerminalAuthorityTail.pins,
    AffinePins.Pin.Holds assignment pin

theorem terminalAuthority_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (rows : TerminalAuthorityRows assignment) :
    TerminalAuthorityAccepted assignment := {
  piDec := PiDecStrictSound.Exact.terminal_ce_sound prime canonical one rows.piDec
  tail := AffinePins.rows_sound
    FPrimeFullHistoryPiCcsTerminalAuthorityTail.pins_canonical canonical one rows.tail
}

/-- Exact row predicates owned by the recursive NIFS certificate. -/
structure RecursiveRows (assignment : Nat → Nat) : Prop where
  transcript : Satisfies
    FPrimeFullHistoryRecursiveTranscriptArtifact.ownerRows assignment
  affine : FPrimeFullHistoryAffineSound.Recursive.RowsSatisfy assignment
  projection : RecursiveHolds assignment
  projectionGlue : Satisfies recursiveGlueRows assignment
  feSumcheck : Satisfies recursiveFeRows assignment
  ncSumcheck : Satisfies recursiveNcRows assignment
  piDec : Satisfies
    FPrimeFullHistoryPiDec.recursiveRows assignment
  pointBinding : Satisfies
    FPrimeFullHistoryRecursivePointBinding.rows assignment
  accumulator : Satisfies
    FPrimeFullHistoryRecursiveAccumulator.rows assignment
  residual : OwnersRows recursiveResidualOwners assignment

/-- Exact row predicates owned by the terminal NIFS certificate. -/
structure TerminalRows (assignment : Nat → Nat) : Prop where
  transcript : Satisfies
    FPrimeFullHistoryTerminalTranscriptArtifact.ownerRows assignment
  affine : FPrimeFullHistoryAffineSound.Terminal.RowsSatisfy assignment
  projection : TerminalHolds assignment
  projectionGlue : Satisfies terminalGlueRows assignment
  feSumcheck : Satisfies terminalFeRows assignment
  ncSumcheck : Satisfies terminalNcRows assignment
  piDec : Satisfies
    FPrimeFullHistoryPiDec.terminalRows assignment
  pointBinding : Satisfies
    FPrimeFullHistoryTerminalPointBinding.rows assignment
  authority : TerminalAuthorityRows assignment
  residual : OwnersRows terminalResidualOwners assignment

/-- Recursive verifier checklist decoded from circuit wires. -/
structure RecursiveSemanticAccepted (assignment : Nat → Nat) : Prop where
  constantOne : assignment 0 = 1
  transcript :
    FPrimeFullHistoryRecursiveTranscriptArtifact.trace.Accepted assignment
  affine : FPrimeFullHistoryAffineSound.Recursive.Holds assignment
  projection : BatchAccepted ProjectionProgram.K.ops
    (ProjectionProgram.BatchIdentity recursiveTraces assignment)
  projectionGlue : ∀ pin ∈ recursiveGluePins,
    AffinePins.Pin.Holds assignment pin
  feSumcheck : Nightstream.SuperNeo.SumCheck.Accepted
    SumcheckChainSound.ops
    (SumcheckChainSound.transcript recursiveFeMaps assignment)
  ncSumcheck : Nightstream.SuperNeo.SumCheck.Accepted
    SumcheckChainSound.ops
    (SumcheckChainSound.transcript recursiveNcMaps assignment)
  piDec : PiDecStrictCompiler.Accepted FPrimeFullHistoryPiDec.layout
    (Relabel.assignment FPrimeFullHistoryPiDec.recursiveColumnMap assignment)
  pointBinding : FPrimeFullHistoryPointBindingSound.RecursiveHolds assignment
  accumulator :
    FPrimeFullHistoryRecursiveAccumulatorSound.Facts assignment
  residual : OwnersAccepted recursiveResidualOwners assignment

/-- Terminal-fold verifier checklist decoded from circuit wires. -/
structure TerminalSemanticAccepted (assignment : Nat → Nat) : Prop where
  constantOne : assignment 0 = 1
  transcript : FPrimeFullHistoryTranscriptSound.TerminalTranscriptAccepted
    assignment
  affine : FPrimeFullHistoryAffineSound.Terminal.Holds assignment
  projection : BatchAccepted ProjectionProgram.K.ops
    (ProjectionProgram.BatchIdentity terminalTraces assignment)
  projectionGlue : ∀ pin ∈ terminalGluePins,
    AffinePins.Pin.Holds assignment pin
  feSumcheck : Nightstream.SuperNeo.SumCheck.Accepted
    SumcheckChainSound.ops
    (SumcheckChainSound.transcript terminalFeMaps assignment)
  ncSumcheck : Nightstream.SuperNeo.SumCheck.Accepted
    SumcheckChainSound.ops
    (SumcheckChainSound.transcript terminalNcMaps assignment)
  piDec : PiDecStrictCompiler.Accepted FPrimeFullHistoryPiDec.layout
    (Relabel.assignment FPrimeFullHistoryPiDec.terminalColumnMap assignment)
  pointBinding : FPrimeFullHistoryPointBindingSound.TerminalHolds assignment
  authority : TerminalAuthorityAccepted assignment
  residual : OwnersAccepted terminalResidualOwners assignment

private theorem parent_kappa_value :
    FPrimeFullHistoryPiDec.layout.parent.commitment.dataCols.length /
      FPrimeFullHistoryPiDec.layout.ringDimension = 18 := by
  native_decide

/-- The recursive PiRLC shape owner pins the exact PiDEC parent shape used by
the accumulator serializer.  The relevant columns are the relabeled parent
`d`, `kappa`, `xRows`, `xWidth`, and `mIn` wires; digest equality is not used
as authority for any of them. -/
theorem RecursiveSemanticAccepted.parentShapeAgrees
    {assignment : Nat → Nat}
    (accepted : RecursiveSemanticAccepted assignment) :
    FPrimeFullHistoryParentCeSerialization.RecursiveShapeAgrees assignment := by
  refine {
    commitmentD := ?_
    commitmentKappa := ?_
    xRows := ?_
    xWidth := ?_
    mIn := ?_
  }
  · calc
      _ = assignment (Relabel.column FPrimeFullHistoryPiDec.recursiveColumnMap
          FPrimeFullHistoryPiDec.layout.parent.commitment.dCol) := rfl
      _ = 54 := accepted.affine.piRlcShape
        (.constant (Relabel.column FPrimeFullHistoryPiDec.recursiveColumnMap
          FPrimeFullHistoryPiDec.layout.parent.commitment.dCol) 54)
        (by native_decide)
      _ = _ := rfl
  · calc
      _ = assignment (Relabel.column FPrimeFullHistoryPiDec.recursiveColumnMap
          FPrimeFullHistoryPiDec.layout.parent.commitment.kappaCol) := rfl
      _ = 18 := accepted.affine.piRlcShape
        (.constant (Relabel.column FPrimeFullHistoryPiDec.recursiveColumnMap
          FPrimeFullHistoryPiDec.layout.parent.commitment.kappaCol) 18)
        (by native_decide)
      _ = _ := parent_kappa_value.symm
  · calc
      _ = assignment (Relabel.column FPrimeFullHistoryPiDec.recursiveColumnMap
          FPrimeFullHistoryPiDec.layout.parent.xRowsCol) := rfl
      _ = 54 := accepted.affine.piRlcShape
        (.constant (Relabel.column FPrimeFullHistoryPiDec.recursiveColumnMap
          FPrimeFullHistoryPiDec.layout.parent.xRowsCol) 54)
        (by native_decide)
      _ = _ := rfl
  · calc
      _ = assignment (Relabel.column FPrimeFullHistoryPiDec.recursiveColumnMap
          FPrimeFullHistoryPiDec.layout.parent.xWidthCol) := rfl
      _ = 257 := accepted.affine.piRlcShape
        (.constant (Relabel.column FPrimeFullHistoryPiDec.recursiveColumnMap
          FPrimeFullHistoryPiDec.layout.parent.xWidthCol) 257)
        (by native_decide)
      _ = _ := rfl
  · calc
      _ = assignment (Relabel.column FPrimeFullHistoryPiDec.recursiveColumnMap
          FPrimeFullHistoryPiDec.layout.parent.mInCol) := rfl
      _ = 257 := accepted.affine.piRlcShape
        (.constant (Relabel.column FPrimeFullHistoryPiDec.recursiveColumnMap
          FPrimeFullHistoryPiDec.layout.parent.mInCol) 257)
        (by native_decide)
      _ = _ := rfl

/-- Terminal counterpart of `parentShapeAgrees`, over the exact terminal
PiDEC relabeling. -/
theorem TerminalSemanticAccepted.parentShapeAgrees
    {assignment : Nat → Nat}
    (accepted : TerminalSemanticAccepted assignment) :
    FPrimeFullHistoryParentCeSerialization.TerminalShapeAgrees assignment := by
  refine {
    commitmentD := ?_
    commitmentKappa := ?_
    xRows := ?_
    xWidth := ?_
    mIn := ?_
  }
  · calc
      _ = assignment (Relabel.column FPrimeFullHistoryPiDec.terminalColumnMap
          FPrimeFullHistoryPiDec.layout.parent.commitment.dCol) := rfl
      _ = 54 := accepted.affine.piRlcShape
        (.constant (Relabel.column FPrimeFullHistoryPiDec.terminalColumnMap
          FPrimeFullHistoryPiDec.layout.parent.commitment.dCol) 54)
        (by native_decide)
      _ = _ := rfl
  · calc
      _ = assignment (Relabel.column FPrimeFullHistoryPiDec.terminalColumnMap
          FPrimeFullHistoryPiDec.layout.parent.commitment.kappaCol) := rfl
      _ = 18 := accepted.affine.piRlcShape
        (.constant (Relabel.column FPrimeFullHistoryPiDec.terminalColumnMap
          FPrimeFullHistoryPiDec.layout.parent.commitment.kappaCol) 18)
        (by native_decide)
      _ = _ := parent_kappa_value.symm
  · calc
      _ = assignment (Relabel.column FPrimeFullHistoryPiDec.terminalColumnMap
          FPrimeFullHistoryPiDec.layout.parent.xRowsCol) := rfl
      _ = 54 := accepted.affine.piRlcShape
        (.constant (Relabel.column FPrimeFullHistoryPiDec.terminalColumnMap
          FPrimeFullHistoryPiDec.layout.parent.xRowsCol) 54)
        (by native_decide)
      _ = _ := rfl
  · calc
      _ = assignment (Relabel.column FPrimeFullHistoryPiDec.terminalColumnMap
          FPrimeFullHistoryPiDec.layout.parent.xWidthCol) := rfl
      _ = 257 := accepted.affine.piRlcShape
        (.constant (Relabel.column FPrimeFullHistoryPiDec.terminalColumnMap
          FPrimeFullHistoryPiDec.layout.parent.xWidthCol) 257)
        (by native_decide)
      _ = _ := rfl
  · calc
      _ = assignment (Relabel.column FPrimeFullHistoryPiDec.terminalColumnMap
          FPrimeFullHistoryPiDec.layout.parent.mInCol) := rfl
      _ = 257 := accepted.affine.piRlcShape
        (.constant (Relabel.column FPrimeFullHistoryPiDec.terminalColumnMap
          FPrimeFullHistoryPiDec.layout.parent.mInCol) 257)
        (by native_decide)
      _ = _ := rfl

/-- The recursive raw-source preimage is exactly the safe serialization of
the decoded strict-PiDEC parent once the generated shape owner is applied. -/
theorem RecursiveSemanticAccepted.parentSerializes
    {assignment : Nat → Nat}
    (accepted : RecursiveSemanticAccepted assignment) :
    CeClaimDigestV2.serialize
        (PiDecStrictCompiler.activeColumns FPrimeFullHistoryPiDec.layout)
        FPrimeFullHistoryPiDec.layout.extensionLimbs
        (FPrimeFullHistoryParentCeSerialization.decodedParentWith
          FPrimeFullHistoryPiDec.recursiveColumnMap assignment) =
      some (FPrimeFullHistoryParentCeSerialization.parentPreimageWith
        FPrimeFullHistoryPiDec.recursiveColumnMap assignment) :=
  FPrimeFullHistoryParentCeSerialization.serialize_parentWith
    accepted.parentShapeAgrees

/-- Terminal counterpart of `parentSerializes`. -/
theorem TerminalSemanticAccepted.parentSerializes
    {assignment : Nat → Nat}
    (accepted : TerminalSemanticAccepted assignment) :
    CeClaimDigestV2.serialize
        (PiDecStrictCompiler.activeColumns FPrimeFullHistoryPiDec.layout)
        FPrimeFullHistoryPiDec.layout.extensionLimbs
        (FPrimeFullHistoryParentCeSerialization.decodedParentWith
          FPrimeFullHistoryPiDec.terminalColumnMap assignment) =
      some (FPrimeFullHistoryParentCeSerialization.parentPreimageWith
        FPrimeFullHistoryPiDec.terminalColumnMap assignment) :=
  FPrimeFullHistoryParentCeSerialization.serialize_parentWith
    accepted.parentShapeAgrees

/-- Honest compiler executions needed for recursive CIR-COMPLETE.  Decoded
semantic acceptance remains separate; these fields carry actual interpreter
sources and outputs for row families with compiler-only intermediate wires. -/
structure RecursiveExecutionWitness
    (field : CanonicalU64Complete.FieldInverse)
    (assignment : Nat → Nat) where
  transcript :
    FPrimeFullHistoryRecursiveTranscriptArtifact.trace.Accepted assignment
  affine : FPrimeFullHistoryAffineSound.Recursive.Holds assignment
  projection : ∀ trace ∈ recursiveTraces,
    trace.ExecutionWitness assignment
  projectionGlue : ∀ pin ∈ recursiveGluePins,
    AffinePins.Pin.Holds assignment pin
  feSumcheck : SumcheckChainSound.ExecutionWitness recursiveFeMaps assignment
  ncSumcheck : SumcheckChainSound.ExecutionWitness recursiveNcMaps assignment
  piDec : PiDecStrictSound.Exact.ExecutionWitness
    (Relabel.assignment FPrimeFullHistoryPiDec.recursiveColumnMap assignment)
  pointBinding : FPrimeFullHistoryPointBindingSound.RecursiveHolds assignment
  accumulator :
    FPrimeFullHistoryRecursiveAccumulatorSound.CompilerWitness assignment
  residual : OwnersExecution field recursiveResidualOwners assignment

/-- Terminal-fold counterpart of `RecursiveExecutionWitness`. -/
structure TerminalExecutionWitness
    (field : CanonicalU64Complete.FieldInverse)
    (assignment : Nat → Nat) where
  transcript : FPrimeFullHistoryTranscriptSound.TerminalTranscriptAccepted
    assignment
  affine : FPrimeFullHistoryAffineSound.Terminal.Holds assignment
  projection : ∀ trace ∈ terminalTraces,
    trace.ExecutionWitness assignment
  projectionGlue : ∀ pin ∈ terminalGluePins,
    AffinePins.Pin.Holds assignment pin
  feSumcheck : SumcheckChainSound.ExecutionWitness terminalFeMaps assignment
  ncSumcheck : SumcheckChainSound.ExecutionWitness terminalNcMaps assignment
  piDec : PiDecStrictSound.Exact.ExecutionWitness
    (Relabel.assignment FPrimeFullHistoryPiDec.terminalColumnMap assignment)
  pointBinding : FPrimeFullHistoryPointBindingSound.TerminalHolds assignment
  terminalCe : PiDecStrictSound.Exact.ExecutionWitness
    (Relabel.assignment FPrimeFullHistoryPiDec.terminalCeColumnMap assignment)
  authorityTail : ∀ pin ∈ FPrimeFullHistoryPiCcsTerminalAuthorityTail.pins,
    AffinePins.Pin.Holds assignment pin
  residual : OwnersExecution field terminalResidualOwners assignment

/-- Everything the currently interpreted generated recursive rows prove about
the concrete NIFS verifier. -/
structure RecursiveArtifactAccepted (assignment : Nat → Nat) : Prop where
  arity : recursiveArity.total = 1
  transcript :
    FPrimeFullHistoryRecursiveTranscriptArtifact.trace.Accepted assignment
  affine : FPrimeFullHistoryAffineSound.Recursive.Holds assignment
  projection : ProjectionCertificate recursiveArity recursiveRoles
    recursiveTraces assignment
  projectionGlue : ∀ pin ∈ recursiveGluePins,
    AffinePins.Pin.Holds assignment pin
  feSumcheck : Nightstream.SuperNeo.SumCheck.Accepted
    SumcheckChainSound.ops
    (SumcheckChainSound.transcript recursiveFeMaps assignment)
  ncSumcheck : Nightstream.SuperNeo.SumCheck.Accepted
    SumcheckChainSound.ops
    (SumcheckChainSound.transcript recursiveNcMaps assignment)
  piDec : PiDecStrictCompiler.Accepted FPrimeFullHistoryPiDec.layout
    (Relabel.assignment FPrimeFullHistoryPiDec.recursiveColumnMap assignment)
  pointBinding : FPrimeFullHistoryPointBindingSound.RecursiveHolds assignment
  accumulator :
    FPrimeFullHistoryRecursiveAccumulatorSound.Facts assignment
  residual : OwnersAccepted recursiveResidualOwners assignment

/-- Everything the currently interpreted generated terminal rows prove about
the concrete NIFS verifier. -/
structure TerminalArtifactAccepted (assignment : Nat → Nat) : Prop where
  arity : terminalArity.total = 15
  transcript : FPrimeFullHistoryTranscriptSound.TerminalTranscriptAccepted
    assignment
  affine : FPrimeFullHistoryAffineSound.Terminal.Holds assignment
  projection : ProjectionCertificate terminalArity terminalRoles
    terminalTraces assignment
  projectionGlue : ∀ pin ∈ terminalGluePins,
    AffinePins.Pin.Holds assignment pin
  feSumcheck : Nightstream.SuperNeo.SumCheck.Accepted
    SumcheckChainSound.ops
    (SumcheckChainSound.transcript terminalFeMaps assignment)
  ncSumcheck : Nightstream.SuperNeo.SumCheck.Accepted
    SumcheckChainSound.ops
    (SumcheckChainSound.transcript terminalNcMaps assignment)
  piDec : PiDecStrictCompiler.Accepted FPrimeFullHistoryPiDec.layout
    (Relabel.assignment FPrimeFullHistoryPiDec.terminalColumnMap assignment)
  pointBinding : FPrimeFullHistoryPointBindingSound.TerminalHolds assignment
  authority : TerminalAuthorityAccepted assignment
  residual : OwnersAccepted terminalResidualOwners assignment

/-- Row-decoded sampled-point verification yields coefficient-exact projection
equations, or the precise bad-root event. -/
theorem recursive_semantic_sound_or_badRoot
    {assignment : Nat → Nat}
    (accepted : RecursiveSemanticAccepted assignment) :
    RecursiveArtifactAccepted assignment ∨
      BatchBadRoot ProjectionProgram.K.ops
        (ProjectionProgram.BatchIdentity recursiveTraces assignment) := by
  rcases batchAccepted_implies_exact_or_badRoot _ _ accepted.projection with
    exact | bad
  · left
    exact {
      arity := recursive_total
      transcript := accepted.transcript
      affine := accepted.affine
      projection := {
        nativeOrder := recursive_roles_native_order
        aligned := by simpa using recursive_role_alignment assignment
        pairArity := recursive_pair_arity
        pairWidths := recursive_pair_widths
        outputWidth := recursive_output_width
        quotientWidth := recursive_quotient_width
        exact := exact
      }
      projectionGlue := accepted.projectionGlue
      feSumcheck := accepted.feSumcheck
      ncSumcheck := accepted.ncSumcheck
      piDec := accepted.piDec
      pointBinding := accepted.pointBinding
      accumulator := accepted.accumulator
      residual := accepted.residual
    }
  · exact Or.inr bad

/-- Terminal sampled-point verification has the same exact-or-bad-root
deterministic conclusion. -/
theorem terminal_semantic_sound_or_badRoot
    {assignment : Nat → Nat}
    (accepted : TerminalSemanticAccepted assignment) :
    TerminalArtifactAccepted assignment ∨
      BatchBadRoot ProjectionProgram.K.ops
        (ProjectionProgram.BatchIdentity terminalTraces assignment) := by
  rcases batchAccepted_implies_exact_or_badRoot _ _ accepted.projection with
    exact | bad
  · left
    exact {
      arity := terminal_total
      transcript := accepted.transcript
      affine := accepted.affine
      projection := {
        nativeOrder := terminal_roles_native_order
        aligned := by simpa using terminal_role_alignment assignment
        pairArity := terminal_pair_arity
        pairWidths := terminal_pair_widths
        outputWidth := terminal_output_width
        quotientWidth := terminal_quotient_width
        exact := exact
      }
      projectionGlue := accepted.projectionGlue
      feSumcheck := accepted.feSumcheck
      ncSumcheck := accepted.ncSumcheck
      piDec := accepted.piDec
      pointBinding := accepted.pointBinding
      authority := accepted.authority
      residual := accepted.residual
    }
  · exact Or.inr bad

/-- Exact recursive rows imply the row-decoded verifier checklist. -/
theorem recursive_rows_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (rows : RecursiveRows assignment) :
    RecursiveSemanticAccepted assignment := {
  constantOne := one
  transcript := (FPrimeFullHistoryTranscriptSound.recursive_sound canonical one
    rows.transcript).transcript
  affine := FPrimeFullHistoryAffineSound.Recursive.sound canonical one rows.affine
  projection := recursive_batchAccepted assignment canonical one rows.projection
  projectionGlue := recursive_glue_sound canonical one rows.projectionGlue
  feSumcheck := SumcheckChainSound.accepted recursiveFeMaps canonical one
    recursive_fe_shape.2.1 recursive_fe_shape.2.2 (by
      intro columnMap columnMapMember row rowMember
      exact rows.feSumcheck row
        (List.mem_flatMap.mpr ⟨columnMap, columnMapMember, rowMember⟩))
  ncSumcheck := SumcheckChainSound.accepted recursiveNcMaps canonical one
    recursive_nc_shape.2.1 recursive_nc_shape.2.2 (by
      intro columnMap columnMapMember row rowMember
      exact rows.ncSumcheck row
        (List.mem_flatMap.mpr ⟨columnMap, columnMapMember, rowMember⟩))
  piDec := PiDecStrictSound.Exact.recursive_sound prime canonical one rows.piDec
  pointBinding := FPrimeFullHistoryPointBindingSound.recursive_sound canonical one
    rows.pointBinding
  accumulator := FPrimeFullHistoryRecursiveAccumulatorSound.sound
    canonical one rows.accumulator
  residual := owners_sound canonical one rows.residual
}

/-- Exact terminal rows imply the row-decoded verifier checklist. -/
theorem terminal_rows_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (rows : TerminalRows assignment) :
    TerminalSemanticAccepted assignment := {
  constantOne := one
  transcript := FPrimeFullHistoryTranscriptSound.terminal_sound canonical one
    rows.transcript
  affine := FPrimeFullHistoryAffineSound.Terminal.sound canonical one rows.affine
  projection := terminal_batchAccepted assignment canonical one rows.projection
  projectionGlue := terminal_glue_sound canonical one rows.projectionGlue
  feSumcheck := SumcheckChainSound.accepted terminalFeMaps canonical one
    terminal_fe_shape.2.1 terminal_fe_shape.2.2 (by
      intro columnMap columnMapMember row rowMember
      exact rows.feSumcheck row
        (List.mem_flatMap.mpr ⟨columnMap, columnMapMember, rowMember⟩))
  ncSumcheck := SumcheckChainSound.accepted terminalNcMaps canonical one
    terminal_nc_shape.2.1 terminal_nc_shape.2.2 (by
      intro columnMap columnMapMember row rowMember
      exact rows.ncSumcheck row
        (List.mem_flatMap.mpr ⟨columnMap, columnMapMember, rowMember⟩))
  piDec := PiDecStrictSound.Exact.terminal_sound prime canonical one rows.piDec
  pointBinding := FPrimeFullHistoryPointBindingSound.terminal_sound canonical one
    rows.pointBinding
  authority := terminalAuthority_sound prime canonical one rows.authority
  residual := owners_sound canonical one rows.residual
}

private theorem sumcheck_rows_complete
    {maps : List SumcheckChainSound.ColumnMap}
    {assignment : Nat → Nat}
    (witness : SumcheckChainSound.ExecutionWitness maps assignment) :
    Satisfies (maps.flatMap SumcheckChainSound.Rows) assignment := by
  have rounds := SumcheckChainSound.complete witness
  intro row member
  rcases List.mem_flatMap.mp member with
    ⟨columnMap, columnMapMember, rowMember⟩
  exact rounds columnMap columnMapMember row rowMember

/-- CIR-COMPLETE for every currently modeled recursive NIFS row family.
Compiler intermediates come only from explicit interpreter executions. -/
theorem recursive_rows_complete
    {field : CanonicalU64Complete.FieldInverse}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (witness : RecursiveExecutionWitness field assignment) :
    RecursiveRows assignment := {
  transcript := FPrimeFullHistoryTranscriptSound.recursive_complete canonical
    one ⟨⟨rfl, rfl⟩, witness.transcript⟩
  affine := FPrimeFullHistoryAffineSound.Recursive.complete canonical
    one witness.affine
  projection := FPrimeFullHistoryProjection.recursive_complete
    witness.projection
  projectionGlue := recursive_glue_complete canonical
    one witness.projectionGlue
  feSumcheck := by
    simpa [recursiveFeRows] using sumcheck_rows_complete witness.feSumcheck
  ncSumcheck := by
    simpa [recursiveNcRows] using sumcheck_rows_complete witness.ncSumcheck
  piDec := PiDecStrictSound.Exact.recursive_native_complete witness.piDec
  pointBinding := FPrimeFullHistoryPointBindingSound.recursive_complete
    canonical one witness.pointBinding
  accumulator := FPrimeFullHistoryRecursiveAccumulatorSound.complete canonical
    one witness.accumulator
  residual := owners_execution_complete canonical one witness.residual
}

/-- CIR-COMPLETE for every currently modeled terminal NIFS row family. -/
theorem terminal_rows_complete
    {field : CanonicalU64Complete.FieldInverse}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (witness : TerminalExecutionWitness field assignment) :
    TerminalRows assignment := {
  transcript := FPrimeFullHistoryTranscriptSound.terminal_complete canonical
    one witness.transcript
  affine := FPrimeFullHistoryAffineSound.Terminal.complete canonical
    one witness.affine
  projection := FPrimeFullHistoryProjection.terminal_complete
    witness.projection
  projectionGlue := terminal_glue_complete canonical
    one witness.projectionGlue
  feSumcheck := by
    simpa [terminalFeRows] using sumcheck_rows_complete witness.feSumcheck
  ncSumcheck := by
    simpa [terminalNcRows] using sumcheck_rows_complete witness.ncSumcheck
  piDec := PiDecStrictSound.Exact.terminal_native_complete witness.piDec
  pointBinding := FPrimeFullHistoryPointBindingSound.terminal_complete
    canonical one witness.pointBinding
  authority := {
    piDec := by
      apply (Relabel.satisfies_mapped_iff FPrimeFullHistoryPiDec.rows
        FPrimeFullHistoryPiDec.terminalCeColumnMap assignment).mpr
      exact PiDecStrictSound.Exact.native_complete witness.terminalCe
    tail := AffinePins.rows_complete
      FPrimeFullHistoryPiCcsTerminalAuthorityTail.pins_canonical canonical
      one witness.authorityTail
  }
  residual := owners_execution_complete canonical one witness.residual
}

/-- Compatibility theorem phrased from exact rows. -/
theorem recursive_artifact_sound_or_badRoot
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (rows : RecursiveRows assignment) :
    RecursiveArtifactAccepted assignment ∨
      BatchBadRoot ProjectionProgram.K.ops
        (ProjectionProgram.BatchIdentity recursiveTraces assignment) :=
  recursive_semantic_sound_or_badRoot
    (recursive_rows_sound prime canonical one rows)

/-- Compatibility theorem phrased from exact terminal rows. -/
theorem terminal_artifact_sound_or_badRoot
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (rows : TerminalRows assignment) :
    TerminalArtifactAccepted assignment ∨
      BatchBadRoot ProjectionProgram.K.ops
        (ProjectionProgram.BatchIdentity terminalTraces assignment) :=
  terminal_semantic_sound_or_badRoot
    (terminal_rows_sound prime canonical one rows)

/-! ## Concrete production verifier

`Proof` is data only.  Using `Fin goldilocksP` for each wire makes canonical
field representation structural, while the verifier separately checks that
the distinguished constant wire is one.  The returned accumulator is decoded
from the exact strict-PiDEC parent/children layout used by Rust.
-/

/-- One commitment coordinate decoded from strict-PiDEC wires. -/
structure Commitment where
  d : Nat
  kappa : Nat
  data : List Nat
deriving DecidableEq, Repr

/-- Optional Nebula `(ops, is, fs)` product commitment. -/
structure AdvCommitment where
  ops : Commitment
  is : Commitment
  fs : Commitment
deriving DecidableEq, Repr

/-- Verifier-visible CE claim data.  There is deliberately no witness field. -/
structure Claim where
  commitment : Commitment
  adv : Option AdvCommitment
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

/-- Rust `RunningInstance` verifier view: PiDEC children plus their parent
authority. -/
structure Accumulator where
  parentAuthority : Option Claim
  claims : List Claim
  handle : Digest
deriving DecidableEq, Repr

/-- Rust's verifier-visible bootstrap accumulator. -/
def emptyAccumulator : Accumulator :=
  ⟨none, [], FPrimeFullHistoryBaseStepSound.emptyAccumulator⟩

private def values (assignment : Nat → Nat) (columns : List Nat) : List Nat :=
  columns.map assignment

private def pairValues (assignment : Nat → Nat)
    (columns : List (Nat × Nat)) : List (Nat × Nat) :=
  columns.map fun pair => (assignment pair.1, assignment pair.2)

private def decodeCommitment (assignment : Nat → Nat)
    (layout : PiDecStrictCompiler.CommitmentLayout) : Commitment where
  d := assignment layout.dCol
  kappa := assignment layout.kappaCol
  data := values assignment layout.dataCols

private def decodeAdvCommitment (assignment : Nat → Nat)
    (layout : PiDecStrictCompiler.AdvLayout) : AdvCommitment where
  ops := decodeCommitment assignment layout.ops
  is := decodeCommitment assignment layout.is
  fs := decodeCommitment assignment layout.fs

/-- Decode exactly the public CE-claim fields consumed by strict PiDEC. -/
def decodeClaim (assignment : Nat → Nat)
    (layout : PiDecStrictCompiler.ClaimLayout) : Claim where
  commitment := decodeCommitment assignment layout.commitment
  adv := layout.adv.map (decodeAdvCommitment assignment)
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

def decodeAccumulator (assignment : Nat → Nat)
    (layout : PiDecStrictCompiler.Layout)
    (handle : Digest) : Accumulator where
  parentAuthority := some (decodeClaim assignment layout.parent)
  claims := layout.children.map (decodeClaim assignment)
  handle := handle

/-- A production circuit proof is only a field assignment.  Acceptance is
recomputed by `recursiveCheck`/`terminalCheck`. -/
structure Proof where
  witness : Nat → ProjectionProgram.F

def Proof.assignment (proof : Proof) : Nat → Nat :=
  fun column => (proof.witness column).val

theorem Proof.canonical (proof : Proof) :
    ∀ column, proof.assignment column < goldilocksP :=
  fun column => (proof.witness column).isLt

/-- Canonical field assignment constructed from an R1CS assignment together
with the representation bound already required by artifact soundness. -/
def proofOfAssignment (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Proof where
  witness := fun column => ⟨assignment column, canonical column⟩

@[simp] theorem proofOfAssignment_assignment
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    (proofOfAssignment assignment canonical).assignment = assignment := by
  funext column
  rfl

private def pinCheck (assignment : Nat → Nat) : AffinePins.Pin → Bool
  | .zero column => decide (assignment column = 0)
  | .constant column value => decide (assignment column = value)
  | .equal left right => decide (assignment left = assignment right)

private theorem pinCheck_eq_true_iff (assignment : Nat → Nat)
    (pin : AffinePins.Pin) :
    pinCheck assignment pin = true ↔ pin.Holds assignment := by
  cases pin <;> simp [pinCheck, AffinePins.Pin.Holds, decide_eq_true_eq]

private def pinsCheck (assignment : Nat → Nat)
    (pins : List AffinePins.Pin) : Bool :=
  pins.all (pinCheck assignment)

private theorem pinsCheck_eq_true_iff (assignment : Nat → Nat)
    (pins : List AffinePins.Pin) :
    pinsCheck assignment pins = true ↔
      ∀ pin ∈ pins, pin.Holds assignment := by
  simp [pinsCheck, List.all_eq_true, pinCheck_eq_true_iff]

def ownersCheck (owners : List Owner) (assignment : Nat → Nat) : Bool :=
  owners.all fun owner => owner.check assignment

theorem ownersCheck_eq_true_iff (owners : List Owner)
    (assignment : Nat → Nat) :
    ownersCheck owners assignment = true ↔ OwnersAccepted owners assignment := by
  constructor
  · intro checked owner member
    exact (OwnerCertificate.Owner.check_eq_true_iff owner assignment).mp
      ((List.all_eq_true.mp checked) owner member)
  · intro accepted
    apply List.all_eq_true.mpr
    intro owner member
    exact (OwnerCertificate.Owner.check_eq_true_iff owner assignment).mpr
      (accepted owner member)

private def terminalAuthorityCheck (assignment : Nat → Nat) : Bool :=
  PiDecStrictCompiler.check FPrimeFullHistoryPiDec.layout
      (Relabel.assignment FPrimeFullHistoryPiDec.terminalCeColumnMap assignment) &&
    pinsCheck assignment FPrimeFullHistoryPiCcsTerminalAuthorityTail.pins

private theorem terminalAuthorityCheck_eq_true_iff (assignment : Nat → Nat) :
    terminalAuthorityCheck assignment = true ↔
      TerminalAuthorityAccepted assignment := by
  simp only [terminalAuthorityCheck, Bool.and_eq_true,
    PiDecStrictCompiler.check_eq_true_iff, pinsCheck_eq_true_iff]
  constructor
  · rintro ⟨piDec, tail⟩
    exact ⟨piDec, tail⟩
  · intro accepted
    exact ⟨accepted.piDec, accepted.tail⟩

private def projectionCheck
    (traces : List ProjectionProgram.ProjectionTrace)
    (assignment : Nat → Nat) : Bool :=
  (ProjectionProgram.BatchIdentity traces assignment).all fun identity =>
    decide (ProjectionCheck.Accepted ProjectionProgram.K.ops identity)

private theorem projectionCheck_eq_true_iff
    (traces : List ProjectionProgram.ProjectionTrace)
    (assignment : Nat → Nat) :
    projectionCheck traces assignment = true ↔
      BatchAccepted ProjectionProgram.K.ops
        (ProjectionProgram.BatchIdentity traces assignment) := by
  simp [projectionCheck, BatchAccepted, List.all_eq_true, decide_eq_true_eq]

private def recursivePointCheck (assignment : Nat → Nat) : Bool :=
  decide (assignment (FPrimeFullHistoryPointBindingSound.recursivePair 0).1 =
      assignment (FPrimeFullHistoryPointBindingSound.recursivePair 0).2) &&
    decide (assignment (FPrimeFullHistoryPointBindingSound.recursivePair 1).1 =
      assignment (FPrimeFullHistoryPointBindingSound.recursivePair 1).2)

private theorem recursivePointCheck_eq_true_iff (assignment : Nat → Nat) :
    recursivePointCheck assignment = true ↔
      FPrimeFullHistoryPointBindingSound.RecursiveHolds assignment := by
  simp only [recursivePointCheck, Bool.and_eq_true, decide_eq_true_eq]
  constructor
  · rintro ⟨low, high⟩
    exact ⟨fun limb limbLt => by
      rcases (show limb = 0 ∨ limb = 1 by omega) with rfl | rfl
      · exact low
      · exact high⟩
  · intro holds
    exact ⟨holds.point 0 (by decide), holds.point 1 (by decide)⟩

private def terminalPointCheck (assignment : Nat → Nat) : Bool :=
  decide (assignment (FPrimeFullHistoryPointBindingSound.terminalPair 0).1 =
      assignment (FPrimeFullHistoryPointBindingSound.terminalPair 0).2) &&
    decide (assignment (FPrimeFullHistoryPointBindingSound.terminalPair 1).1 =
      assignment (FPrimeFullHistoryPointBindingSound.terminalPair 1).2)

private theorem terminalPointCheck_eq_true_iff (assignment : Nat → Nat) :
    terminalPointCheck assignment = true ↔
      FPrimeFullHistoryPointBindingSound.TerminalHolds assignment := by
  simp only [terminalPointCheck, Bool.and_eq_true, decide_eq_true_eq]
  constructor
  · rintro ⟨low, high⟩
    exact ⟨fun limb limbLt => by
      rcases (show limb = 0 ∨ limb = 1 by omega) with rfl | rfl
      · exact low
      · exact high⟩
  · intro holds
    exact ⟨holds.point 0 (by decide), holds.point 1 (by decide)⟩

/-- Executable fixed-profile recursive NIFS verifier over decoded semantics. -/
def recursiveCheck (proof : Proof) : Bool :=
  let assignment := proof.assignment
  [decide (assignment 0 = 1),
   FPrimeFullHistoryRecursiveTranscriptArtifact.trace.check assignment,
   pinsCheck assignment FPrimeFullHistoryPiCcsRecursiveAllocation.pins,
   pinsCheck assignment FPrimeFullHistoryPiCcsRecursiveAuthority.pins,
   pinsCheck assignment FPrimeFullHistoryPiCcsRecursiveOutputBinding.pins,
   pinsCheck assignment FPrimeFullHistoryPiRlcRecursiveShape.pins,
   pinsCheck assignment FPrimeFullHistoryPiRlcRecursiveLinearFolds.pins,
   projectionCheck recursiveTraces assignment,
   pinsCheck assignment recursiveGluePins,
   Nightstream.SuperNeo.SumCheck.check SumcheckChainSound.ops
     (SumcheckChainSound.transcript recursiveFeMaps assignment),
   Nightstream.SuperNeo.SumCheck.check SumcheckChainSound.ops
     (SumcheckChainSound.transcript recursiveNcMaps assignment),
   PiDecStrictCompiler.check FPrimeFullHistoryPiDec.layout
     (Relabel.assignment FPrimeFullHistoryPiDec.recursiveColumnMap assignment),
   recursivePointCheck assignment,
   FPrimeFullHistoryRecursiveAccumulatorSound.nativeCheck assignment,
   ownersCheck recursiveResidualOwners assignment].all id

/-- Executable fixed-profile terminal NIFS verifier over decoded semantics. -/
def terminalCheck (proof : Proof) : Bool :=
  let assignment := proof.assignment
  [decide (assignment 0 = 1),
   FPrimeFullHistoryTranscriptSound.terminalCheck assignment,
   pinsCheck assignment FPrimeFullHistoryPiCcsTerminalAllocation.pins,
   pinsCheck assignment FPrimeFullHistoryPiCcsTerminalOutputBinding.pins,
   pinsCheck assignment FPrimeFullHistoryPiRlcTerminalShape.pins,
   pinsCheck assignment FPrimeFullHistoryPiRlcTerminalLinearFolds.pins,
   projectionCheck terminalTraces assignment,
   pinsCheck assignment terminalGluePins,
   Nightstream.SuperNeo.SumCheck.check SumcheckChainSound.ops
     (SumcheckChainSound.transcript terminalFeMaps assignment),
   Nightstream.SuperNeo.SumCheck.check SumcheckChainSound.ops
     (SumcheckChainSound.transcript terminalNcMaps assignment),
   PiDecStrictCompiler.check FPrimeFullHistoryPiDec.layout
     (Relabel.assignment FPrimeFullHistoryPiDec.terminalColumnMap assignment),
   terminalPointCheck assignment,
   terminalAuthorityCheck assignment,
   ownersCheck terminalResidualOwners assignment].all id

theorem recursiveCheck_eq_true_iff (proof : Proof) :
    recursiveCheck proof = true ↔
      RecursiveSemanticAccepted proof.assignment := by
  simp only [recursiveCheck, List.all_cons, List.all_nil, id_eq,
    Bool.and_eq_true, and_true, decide_eq_true_eq, pinsCheck_eq_true_iff,
    projectionCheck_eq_true_iff,
    TranscriptCertificate.Trace.check_eq_true_iff,
    Nightstream.SuperNeo.SumCheck.check_eq_true_iff_accepted,
    PiDecStrictCompiler.check_eq_true_iff, recursivePointCheck_eq_true_iff,
    FPrimeFullHistoryRecursiveAccumulatorSound.nativeCheck_eq_true_iff,
    ownersCheck_eq_true_iff]
  constructor
  · rintro ⟨one, transcript, allocation, authority, outputBinding, rlcShape,
      linearFolds, projection, glue, fe, nc, piDec, point, accumulator, residual⟩
    exact ⟨one, transcript, ⟨allocation, authority, outputBinding, rlcShape,
      linearFolds⟩, projection, glue, fe, nc, piDec, point, accumulator, residual⟩
  · intro accepted
    exact ⟨accepted.constantOne, accepted.transcript,
      accepted.affine.piCcsAllocation,
      accepted.affine.piCcsAuthority, accepted.affine.piCcsOutputBinding,
      accepted.affine.piRlcShape, accepted.affine.piRlcLinearFolds,
      accepted.projection, accepted.projectionGlue, accepted.feSumcheck,
      accepted.ncSumcheck, accepted.piDec, accepted.pointBinding,
      accepted.accumulator, accepted.residual⟩

theorem terminalCheck_eq_true_iff (proof : Proof) :
    terminalCheck proof = true ↔
      TerminalSemanticAccepted proof.assignment := by
  simp only [terminalCheck, List.all_cons, List.all_nil, id_eq,
    Bool.and_eq_true, and_true, decide_eq_true_eq, pinsCheck_eq_true_iff,
    projectionCheck_eq_true_iff,
    FPrimeFullHistoryTranscriptSound.terminalCheck_eq_true_iff,
    Nightstream.SuperNeo.SumCheck.check_eq_true_iff_accepted,
    PiDecStrictCompiler.check_eq_true_iff, terminalPointCheck_eq_true_iff,
    terminalAuthorityCheck_eq_true_iff, ownersCheck_eq_true_iff]
  constructor
  · rintro ⟨one, transcript, allocation, outputBinding, rlcShape, linearFolds,
      projection, glue, fe, nc, piDec, point, authority, residual⟩
    exact ⟨one, transcript, ⟨allocation, outputBinding, rlcShape, linearFolds⟩,
      projection, glue, fe, nc, piDec, point, authority, residual⟩
  · intro accepted
    exact ⟨accepted.constantOne, accepted.transcript,
      accepted.affine.piCcsAllocation,
      accepted.affine.piCcsOutputBinding, accepted.affine.piRlcShape,
      accepted.affine.piRlcLinearFolds, accepted.projection,
      accepted.projectionGlue, accepted.feSumcheck, accepted.ncSumcheck,
      accepted.piDec, accepted.pointBinding, accepted.authority,
      accepted.residual⟩

private def exactProjectionCheck
    (traces : List ProjectionProgram.ProjectionTrace)
    (assignment : Nat → Nat) : Bool :=
  (ProjectionProgram.BatchIdentity traces assignment).all fun identity =>
    decide identity.Exact

private theorem exactProjectionCheck_eq_true_iff
    (traces : List ProjectionProgram.ProjectionTrace)
    (assignment : Nat → Nat) :
    exactProjectionCheck traces assignment = true ↔
      BatchExact (ProjectionProgram.BatchIdentity traces assignment) := by
  simp [exactProjectionCheck, BatchExact, List.all_eq_true, decide_eq_true_eq]

/-- Coefficient-exact row-decoded checker used by the M3 NIFS callback. The
generated circuit only guarantees this check or `BadRoot`; paper-level NIFS
soundness additionally requires `FPR-NIFS-BRIDGE`. -/
def recursiveNativeCheck (proof : Proof) : Bool :=
  recursiveCheck proof && exactProjectionCheck recursiveTraces proof.assignment

theorem recursiveNativeCheck_eq_true_iff (proof : Proof) :
    recursiveNativeCheck proof = true ↔
      RecursiveSemanticAccepted proof.assignment ∧
        BatchExact (ProjectionProgram.BatchIdentity recursiveTraces
          proof.assignment) := by
  simp [recursiveNativeCheck, Bool.and_eq_true, recursiveCheck_eq_true_iff,
    exactProjectionCheck_eq_true_iff]

/-- Bind the verifier-owned recursive transcript prefix to the exact witness
assignment. -/
def recursiveContextCheck
    (context : Nightstream.Protocol.FPrime.Step.NifsContext Digest Unit)
    (proof : Proof) : Bool :=
  decide (context =
    FPrimeFullHistoryTranscriptSound.decodedContext proof.assignment)

theorem recursiveContextCheck_eq_true_iff
    (context : Nightstream.Protocol.FPrime.Step.NifsContext Digest Unit)
    (proof : Proof) :
    recursiveContextCheck context proof = true ↔
      context = FPrimeFullHistoryTranscriptSound.decodedContext
        proof.assignment := by
  simp [recursiveContextCheck, decide_eq_true_eq]

/-- Bind the carried singleton fresh claim to the assignment consumed by the
NIFS verifier. -/
def recursiveLatestCheck (latest : List Fresh) (proof : Proof) : Bool :=
  decide (latest =
    FPrimeFullHistoryTranscriptSound.decodedLatest proof.assignment)

theorem recursiveLatestCheck_eq_true_iff
    (latest : List Fresh) (proof : Proof) :
    recursiveLatestCheck latest proof = true ↔
      latest = FPrimeFullHistoryTranscriptSound.decodedLatest
        proof.assignment := by
  simp [recursiveLatestCheck, decide_eq_true_eq]

/-- Coefficient-exact terminal verifier.  It is the deterministic terminal
relation reached outside the explicit projection-root branch. -/
def terminalNativeCheck (proof : Proof) : Bool :=
  terminalCheck proof && exactProjectionCheck terminalTraces proof.assignment

theorem terminalNativeCheck_eq_true_iff (proof : Proof) :
    terminalNativeCheck proof = true ↔
      TerminalSemanticAccepted proof.assignment ∧
        BatchExact (ProjectionProgram.BatchIdentity terminalTraces
          proof.assignment) := by
  simp [terminalNativeCheck, Bool.and_eq_true, terminalCheck_eq_true_iff,
    exactProjectionCheck_eq_true_iff]

def recursiveAccumulator (proof : Proof) : Accumulator :=
  decodeAccumulator
    (Relabel.assignment FPrimeFullHistoryPiDec.recursiveColumnMap proof.assignment)
    FPrimeFullHistoryPiDec.layout
    (FPrimeFullHistoryRecursiveAccumulatorSound.handle proof.assignment)

def terminalAccumulator (proof : Proof) : Accumulator :=
  decodeAccumulator
    (Relabel.assignment FPrimeFullHistoryPiDec.terminalColumnMap proof.assignment)
    FPrimeFullHistoryPiDec.layout
    (FPrimeFullHistoryTerminalAccumulator.accumulatorDigestColumns.map
      proof.assignment)

/-- Rust-shaped result: reject with `none`, otherwise return the verifier-
decoded next running accumulator. -/
def recursiveVerify (proof : Proof) : Option Accumulator :=
  if recursiveCheck proof then some (recursiveAccumulator proof) else none

/-- Terminal-fold invocation of the same strict-PiDEC output decoder. -/
def terminalVerify (proof : Proof) : Option Accumulator :=
  if terminalCheck proof then some (terminalAccumulator proof) else none

/-- Coefficient-exact native semantics used by `Step.LocalHolds`. -/
def recursiveNativeVerify (proof : Proof) : Option Accumulator :=
  if recursiveNativeCheck proof then some (recursiveAccumulator proof) else none

/-- Coefficient-exact terminal relation used after discharging the terminal
projection-root alternative. -/
def terminalNativeVerify (proof : Proof) : Option Accumulator :=
  if terminalNativeCheck proof then some (terminalAccumulator proof) else none

theorem recursiveNativeVerify_of_exact
    {proof : Proof}
    (accepted : RecursiveSemanticAccepted proof.assignment)
    (exact : BatchExact
      (ProjectionProgram.BatchIdentity recursiveTraces proof.assignment)) :
    recursiveNativeVerify proof = some (recursiveAccumulator proof) := by
  have checked : recursiveNativeCheck proof = true :=
    (recursiveNativeCheck_eq_true_iff proof).2 ⟨accepted, exact⟩
  simp [recursiveNativeVerify, checked]

theorem terminalNativeVerify_of_exact
    {proof : Proof}
    (accepted : TerminalSemanticAccepted proof.assignment)
    (exact : BatchExact
      (ProjectionProgram.BatchIdentity terminalTraces proof.assignment)) :
    terminalNativeVerify proof = some (terminalAccumulator proof) := by
  have checked : terminalNativeCheck proof = true :=
    (terminalNativeCheck_eq_true_iff proof).2 ⟨accepted, exact⟩
  simp [terminalNativeVerify, checked]

theorem recursive_rows_verify
    (prime : EuclidPrime goldilocksP)
    (proof : Proof)
    (one : proof.assignment 0 = 1)
    (rows : RecursiveRows proof.assignment) :
    recursiveVerify proof = some (recursiveAccumulator proof) := by
  have accepted : recursiveCheck proof = true :=
    (recursiveCheck_eq_true_iff proof).2
      (recursive_rows_sound prime proof.canonical one rows)
  simp [recursiveVerify, accepted]

theorem terminal_rows_verify
    (prime : EuclidPrime goldilocksP)
    (proof : Proof)
    (one : proof.assignment 0 = 1)
    (rows : TerminalRows proof.assignment) :
    terminalVerify proof = some (terminalAccumulator proof) := by
  have accepted : terminalCheck proof = true :=
    (terminalCheck_eq_true_iff proof).2
      (terminal_rows_sound prime proof.canonical one rows)
  simp [terminalVerify, accepted]


end Nightstream.Assurance.FPrimeConcreteNifs
