import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package
import NightstreamFPrime.Export.Stage1.PerApplicationStreamingIdentity
import NightstreamFPrime.Layout.PiCCS.v1_1.Ownership
import NightstreamFPrime.Layout.Stage1.PiCCSStarts

/-!
Owns the compact row- and column-ownership sidecar for the physical PiCCS
packet in the selected Stage 1 package. The exported spans use global physical
indices. Their counts and starts are proved against the sole Lean layout, and
the artifact is bound to the selected package's structural identity.

This module does not define package data, matrix data, or a second ownership
classifier.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSOwnershipAudit

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

noncomputable abbrev application := Poseidon2HashChainV1Package.application
noncomputable abbrev logicalWidth := PerApplicationFixedPoint.logicalWidth application
abbrev publicFits := PerApplicationFixedPoint.publicFits application
noncomputable abbrev relation :=
  PerApplicationFixedPoint.relation application Poseidon2HashChainV1Package.fits
noncomputable abbrev interface : Formal.Interface logicalWidth 9 publicFits :=
  NightstreamFPrime.Layout.Stage1.PilotPiCCS.interface
noncomputable abbrev parentOffset : Nat :=
  NightstreamFPrime.Layout.Stage1.PilotPiCCS.piCcsOffset

/-- One closed half-open owner interval. -/
structure Span (Owner : Type) where
  owner : Owner
  start : Nat
  count : Nat
deriving Repr, DecidableEq

def Span.endExclusive {Owner : Type} (span : Span Owner) : Nat :=
  span.start + span.count

/-- Build adjacent spans from one base, one owner order, and its deltas. -/
def spansFrom {Owner : Type} : Nat → List Owner → List Nat → List (Span Owner)
  | _, [], _ => []
  | _, _, [] => []
  | start, owner :: owners, count :: counts =>
      ⟨owner, start, count⟩ :: spansFrom (start + count) owners counts

/-- Expand a span ledger to one owner value per covered point. -/
def expandSpans {Owner : Type} : List (Span Owner) → List Owner
  | [] => []
  | span :: spans =>
      List.replicate span.count span.owner ++ expandSpans spans

/-- Neighboring spans share an endpoint. Zero-count spans are retained. -/
def Adjacent {Owner : Type} : List (Span Owner) → Prop
  | [] => True
  | [_] => True
  | first :: second :: rest =>
      first.endExclusive = second.start ∧ Adjacent (second :: rest)

/-- The list partitions one complete half-open interval. -/
def CoversFrom {Owner : Type} (start finish : Nat) : List (Span Owner) → Prop
  | [] => start = finish
  | span :: spans =>
      span.start = start ∧ CoversFrom span.endExclusive finish spans

def rowCounts : List Nat :=
  [160, 224368, 51504, 149184, 116631, 424657, 8542, 109630,
    20794, 752, 130503, 4076512]

/-- The 12 global physical row spans in the sole parent order. -/
def rowSpans : List
    (Span NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.ChildOwner) :=
  spansFrom NightstreamFPrime.Layout.Stage1.PiCCSStarts.rowBase
    NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.childOrder rowCounts

theorem rowCounts_eq_layout :
    rowCounts = NightstreamFPrime.Layout.PiCCS.v1_1.physicalRowDeltas
      relation interface parentOffset := by
  exact (NightstreamFPrime.Layout.Stage1.PilotPiCCS.cumulativeFootprints_eq
    relation).1.symm

theorem rowSpans_exact :
    rowSpans =
      [⟨.statementBinding,
          NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementBindingRowStart,
          160⟩,
       ⟨.statementAbsorption,
          NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementAbsorptionRowStart,
          224368⟩,
       ⟨.challengeDerivation,
          NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeRowStart, 51504⟩,
       ⟨.roundTranscript,
          NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptRowStart,
          149184⟩,
       ⟨.initialClaim,
          NightstreamFPrime.Layout.Stage1.PiCCSStarts.initialClaimRowStart,
          116631⟩,
       ⟨.sumcheckChain,
          NightstreamFPrime.Layout.Stage1.PiCCSStarts.sumcheckRowStart, 424657⟩,
       ⟨.eval_K, NightstreamFPrime.Layout.Stage1.PiCCSStarts.evalKRowStart,
          8542⟩,
       ⟨.eval_A, NightstreamFPrime.Layout.Stage1.PiCCSStarts.evalARowStart,
          109630⟩,
       ⟨.ccsTerminal, NightstreamFPrime.Layout.Stage1.PiCCSStarts.ccsRowStart,
          20794⟩,
       ⟨.normTerminal,
          NightstreamFPrime.Layout.Stage1.PiCCSStarts.normRowStart, 752⟩,
       ⟨.finalIdentity,
          NightstreamFPrime.Layout.Stage1.PiCCSStarts.finalIdentityRowStart,
          130503⟩,
       ⟨.outputBinding,
          NightstreamFPrime.Layout.Stage1.PiCCSStarts.outputBindingRowStart,
          4076512⟩] := by
  rfl

theorem rowSpans_ownerOrder :
    rowSpans.map Span.owner =
      NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.childOrder := by
  rfl

theorem rowSpans_length : rowSpans.length = 12 := by
  rfl

theorem rowSpans_starts_eq_layout :
    rowSpans.map Span.start =
      NightstreamFPrime.Layout.Stage1.PiCCSStarts.rowStarts := by
  rfl

theorem rowSpans_adjacent : Adjacent rowSpans := by
  rw [rowSpans_exact]
  norm_num [Adjacent, Span.endExclusive,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementBindingRowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementAbsorptionRowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeRowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptRowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.initialClaimRowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.sumcheckRowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.evalKRowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.evalARowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.ccsRowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.normRowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.finalIdentityRowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.outputBindingRowStart]

theorem rowSpans_cover_layout :
    CoversFrom NightstreamFPrime.Layout.Stage1.PiCCSStarts.rowBase
      (NightstreamFPrime.Layout.Stage1.PilotPiCCS.physicalRowCount relation)
      rowSpans := by
  rw [rowSpans_exact,
    NightstreamFPrime.Layout.Stage1.PilotPiCCS.physicalRowCount_eq relation]
  norm_num [CoversFrom, Span.endExclusive,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementBindingRowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.statementAbsorptionRowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.challengeRowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.roundTranscriptRowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.initialClaimRowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.sumcheckRowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.evalKRowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.evalARowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.ccsRowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.normRowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.finalIdentityRowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.outputBindingRowStart,
    NightstreamFPrime.Layout.Stage1.PiCCSStarts.rowBase,
    NightstreamFPrime.Layout.PilotProduction.physicalRowCountValue_eq]

private def repeatOwners {Owner : Type} : List Owner → List Nat → List Owner
  | owner :: owners, count :: counts =>
      List.replicate count owner ++ repeatOwners owners counts
  | _, _ => []

private theorem expandSpans_spansFrom {Owner : Type}
    (base : Nat) (owners : List Owner) (counts : List Nat) :
    expandSpans (spansFrom base owners counts) =
      repeatOwners owners counts := by
  induction owners generalizing base counts with
  | nil => rfl
  | cons owner owners inductionHypothesis =>
      cases counts with
      | nil => rfl
      | cons count counts =>
          simp only [spansFrom, expandSpans, repeatOwners,
            inductionHypothesis]

private theorem map_const_eq_replicate {Source Owner : Type}
    (values : List Source) (owner : Owner) :
    values.map (fun _ => owner) = List.replicate values.length owner := by
  induction values with
  | nil => rfl
  | cons _ rest inductionHypothesis =>
      simp only [List.map_cons, List.length_cons, List.replicate_succ,
        inductionHypothesis]

private theorem ownersFor_childProjection
    (child : NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.ChildOwner)
    (start : Nat) (constraints : List Expr) :
    (NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.ownersFor child start
      constraints).map
        (fun owner => owner.child) =
      List.replicate (R1CS.totalRowCount constraints) child := by
  induction constraints generalizing start with
  | nil => rfl
  | cons expression rest inductionHypothesis =>
      cases result : R1CS.directConstraint expression with
      | none =>
          have multiplicationOwners :
              List.map
                  ((fun owner :
                      NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.RowOwner =>
                        owner.child) ∘
                    (fun ordinal =>
                      (⟨child, start, .multiplication ordinal⟩ :
                        NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.RowOwner)))
                  (List.range (R1CS.mulCount expression)) =
                List.replicate (R1CS.mulCount expression) child := by
            simpa only [Function.comp_apply, List.length_range] using
              (map_const_eq_replicate
                (List.range (R1CS.mulCount expression)) child)
          simp [NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.ownersFor,
            R1CS.totalRowCount,
            R1CS.constraintRowCount, result, inductionHypothesis,
            multiplicationOwners, List.replicate_add, Nat.add_assoc]
      | some direct =>
          simp [NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.ownersFor,
            R1CS.totalRowCount,
            R1CS.constraintRowCount, result, inductionHypothesis,
            List.replicate_add]

private theorem ownersForChildren_childProjection
    (children : List
      NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.ChildOwner)
    (constraintLists : List (List Expr)) :
    (NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.ownersForChildren children
      constraintLists).map
        (fun owner => owner.child) =
      repeatOwners children (constraintLists.map R1CS.totalRowCount) := by
  induction children generalizing constraintLists with
  | nil => rfl
  | cons child children inductionHypothesis =>
      cases constraintLists with
      | nil => rfl
      | cons constraints rest =>
          simp only [
            NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.ownersForChildren,
            List.map_append,
            ownersFor_childProjection, List.map_cons, repeatOwners]
          rw [inductionHypothesis rest]

/-- Run-length expansion agrees with every authoritative physical row owner.
The proof is structural and does not enumerate the concrete row list. -/
theorem rowSpans_pointwise_ownerAgreement :
    expandSpans rowSpans =
      (NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.rowOwners relation
        interface parentOffset).map
        (fun owner => owner.child) := by
  rw [rowSpans, rowCounts_eq_layout, expandSpans_spansFrom]
  unfold NightstreamFPrime.Layout.PiCCS.v1_1.physicalRowDeltas
    NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.rowOwners
  exact (ownersForChildren_childProjection
    NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.childOrder
    (NightstreamFPrime.Layout.PiCCS.v1_1.childConstraintLists relation
      interface parentOffset)).symm

/-- Stable column-owner vocabulary. -/
inductive ColumnFamily where
  | external
  | statementBinding
  | statementAbsorption
  | challengeDerivation
  | roundTranscript
  | initialClaim
  | sumcheckChain
  | eval_K
  | eval_A
  | ccsTerminal
  | normTerminal
  | finalIdentity
  | outputBinding
  | r1csIntermediate
deriving Repr, DecidableEq

def columnOrder : List ColumnFamily :=
  [.external, .statementBinding, .statementAbsorption, .challengeDerivation,
    .roundTranscript, .initialClaim, .sumcheckChain, .eval_K, .eval_A,
    .ccsTerminal, .normTerminal, .finalIdentity, .outputBinding,
    .r1csIntermediate]

def columnCounts : List Nat :=
  [14751804, 0, 224368, 51504, 149184, 25918, 0, 1836, 24300, 2,
    32, 27758, 4076512, 731605]

/-- The 14 global physical column spans. The two zero-count child families
remain in the ordered vocabulary. -/
def columnSpans : List (Span ColumnFamily) :=
  spansFrom 0 columnOrder columnCounts

theorem columnCounts_eq_layout :
    columnCounts =
      [parentOffset] ++
        NightstreamFPrime.Layout.PiCCS.v1_1.logicalPrivateDeltas relation
          interface parentOffset ++
        [NightstreamFPrime.Layout.PiCCS.v1_1.physicalFreshColumnCount relation
          interface parentOffset] := by
  simp only [interface, parentOffset,
    NightstreamFPrime.Layout.Stage1.PilotPiCCS.interface,
    NightstreamFPrime.Layout.Stage1.PilotPiCCS.piCcsOffset]
  have logical :=
    NightstreamFPrime.Layout.PiCCS.v1_1.logicalPrivateDeltas_eq_production
      relation
        (NightstreamFPrime.Layout.Stage1.PiCCSInputs.interface
          logicalWidth publicFits)
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
  have fresh :=
    NightstreamFPrime.Layout.PiCCS.v1_1.ProductionInputs.physicalFreshColumnCount_eq
      relation
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.interface
        logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.externalInputsLinear
        logicalWidth publicFits)
  rw [logical, fresh,
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]
  rfl

theorem columnSpans_exact :
    columnSpans =
      [⟨.external, 0, 14751804⟩,
       ⟨.statementBinding, 14751804, 0⟩,
       ⟨.statementAbsorption, 14751804, 224368⟩,
       ⟨.challengeDerivation, 14976172, 51504⟩,
       ⟨.roundTranscript, 15027676, 149184⟩,
       ⟨.initialClaim, 15176860, 25918⟩,
       ⟨.sumcheckChain, 15202778, 0⟩,
       ⟨.eval_K, 15202778, 1836⟩,
       ⟨.eval_A, 15204614, 24300⟩,
       ⟨.ccsTerminal, 15228914, 2⟩,
       ⟨.normTerminal, 15228916, 32⟩,
       ⟨.finalIdentity, 15228948, 27758⟩,
       ⟨.outputBinding, 15256706, 4076512⟩,
       ⟨.r1csIntermediate, 19333218, 731605⟩] := by
  rfl

theorem columnSpans_ownerOrder :
    columnSpans.map Span.owner = columnOrder := by
  rfl

theorem columnSpans_length : columnSpans.length = 14 := by
  rfl

theorem columnSpans_adjacent : Adjacent columnSpans := by
  rw [columnSpans_exact]
  norm_num [Adjacent, Span.endExclusive]

theorem columnSpans_cover_layout :
    CoversFrom 0
      (NightstreamFPrime.Layout.PiCCS.v1_1.physicalColumnCount relation
        interface parentOffset) columnSpans := by
  simp only [interface, parentOffset,
    NightstreamFPrime.Layout.Stage1.PilotPiCCS.interface,
    NightstreamFPrime.Layout.Stage1.PilotPiCCS.piCcsOffset]
  rw [NightstreamFPrime.Layout.PiCCS.v1_1.ProductionInputs.physicalColumnCount_eq
    relation
    (NightstreamFPrime.Layout.Stage1.PiCCSInputs.interface
      logicalWidth publicFits)
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset
    (NightstreamFPrime.Layout.Stage1.PiCCSInputs.externalInputsLinear
      logicalWidth publicFits),
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq,
    columnSpans_exact]
  norm_num [CoversFrom, Span.endExclusive]

/-- Forget only the local index from the authoritative column owner. -/
def columnFamily :
    NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.ColumnOwner → ColumnFamily
  | .external _ => .external
  | .statementBinding _ => .statementBinding
  | .statementAbsorption _ => .statementAbsorption
  | .challengeDerivation _ => .challengeDerivation
  | .roundTranscript _ => .roundTranscript
  | .initialClaim _ => .initialClaim
  | .sumcheckChain _ => .sumcheckChain
  | .eval_K _ => .eval_K
  | .eval_A _ => .eval_A
  | .ccsTerminal _ => .ccsTerminal
  | .normTerminal _ => .normTerminal
  | .finalIdentity _ => .finalIdentity
  | .outputBinding _ => .outputBinding
  | .r1csIntermediate _ => .r1csIntermediate

/-- Proof-side family classifier with the authoritative layout boundaries. -/
noncomputable def exportedColumnFamily (column : Nat) : ColumnFamily :=
  columnFamily
    (NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.columnOwnerAt relation
      interface parentOffset column)

/-- Read an ordered, adjacent span ledger by its recorded exclusive ends. -/
def familyAtSpanEnds {Owner : Type} (fallback : Owner) :
    List (Span Owner) → Nat → Owner
  | [], _ => fallback
  | span :: spans, column =>
      if column < span.endExclusive then span.owner
      else familyAtSpanEnds fallback spans column

/-- Column-family lookup derived directly from `columnSpans`. -/
def columnSpanFamily (column : Nat) : ColumnFamily :=
  familyAtSpanEnds .r1csIntermediate columnSpans column

private theorem statementAbsorptionOffset_eq :
    Formal.statementAbsorptionOffset
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.interface
        logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset = 14751804 := by
  rw [Formal.statementAbsorptionOffset_eq,
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]

private theorem challengeOffset_eq :
    Formal.challengeOffset
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.interface
        logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset = 14976172 := by
  rw [Formal.challengeOffset_eq,
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]

private theorem roundTranscriptOffset_eq :
    Formal.roundTranscriptOffset
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.interface
        logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset = 15027676 := by
  rw [Formal.roundTranscriptOffset_eq, challengeOffset_eq]

private theorem initialClaimOffset_eq :
    Formal.initialClaimOffset
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.interface
        logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset = 15176860 := by
  rw [Formal.initialClaimOffset_eq_initialClaimRowOffset,
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]
  norm_num [Formal.initialClaimRowOffset, productionShape,
    Phi81MatrixSource.phi81Shape, cubeVariables,
    RoundTranscript.perRoundRecipeCount]

private theorem sumcheckOffset_eq :
    Formal.sumcheckOffset
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.interface
        logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset = 15202778 := by
  rw [Formal.sumcheckOffset_eq_sumcheckRowOffset,
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]
  norm_num [Formal.sumcheckRowOffset, Formal.initialClaimRowOffset,
    InitialClaim.privateCount, productionShape, Phi81MatrixSource.phi81Shape,
    cubeVariables, RoundTranscript.perRoundRecipeCount]

private theorem evalKOffset_eq :
    Formal.evalKOffset
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.interface
        logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset = 15202778 := by
  rw [Formal.evalKOffset_eq_evalKRowOffset,
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]
  norm_num [Formal.evalKRowOffset, Formal.sumcheckRowOffset,
    Formal.initialClaimRowOffset, InitialClaim.privateCount, productionShape,
    Phi81MatrixSource.phi81Shape, cubeVariables,
    RoundTranscript.perRoundRecipeCount]

private theorem evalAOffset_eq :
    Formal.evalAOffset
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.interface
        logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset = 15204614 := by
  rw [Formal.evalAOffset_eq_evalARowOffset,
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]
  norm_num [Formal.evalARowOffset, Formal.evalKRowOffset,
    Formal.sumcheckRowOffset, Formal.initialClaimRowOffset,
    InitialClaim.privateCount, EvalKTerminal.privateCount, productionShape,
    Phi81MatrixSource.phi81Shape, cubeVariables,
    RoundTranscript.perRoundRecipeCount]

private theorem ccsOffset_eq :
    Formal.ccsOffset
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.interface
        logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset = 15228914 := by
  rw [Formal.ccsOffset_eq_ccsRowOffset,
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]
  norm_num [Formal.ccsRowOffset, Formal.evalARowOffset,
    Formal.evalKRowOffset, Formal.sumcheckRowOffset,
    Formal.initialClaimRowOffset, InitialClaim.privateCount,
    EvalKTerminal.privateCount, EvalATerminal.privateCount, productionShape,
    Phi81MatrixSource.phi81Shape, cubeVariables,
    RoundTranscript.perRoundRecipeCount]

private theorem normOffset_eq :
    Formal.normOffset relation
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.interface
        logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset = 15228916 := by
  rw [Formal.normOffset_eq_normRowOffset,
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]
  norm_num [Formal.normRowOffset, Formal.ccsRowOffset,
    Formal.evalARowOffset, Formal.evalKRowOffset, Formal.sumcheckRowOffset,
    Formal.initialClaimRowOffset, InitialClaim.privateCount,
    EvalKTerminal.privateCount, EvalATerminal.privateCount,
    CcsTerminal.privateCount, productionShape, Phi81MatrixSource.phi81Shape,
    cubeVariables, RoundTranscript.perRoundRecipeCount]

private theorem finalIdentityOffset_eq :
    Formal.finalIdentityOffset relation
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.interface
        logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset = 15228948 := by
  rw [Formal.finalIdentityOffset_eq_finalIdentityRowOffset,
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]
  norm_num [Formal.finalIdentityRowOffset, Formal.normRowOffset,
    Formal.ccsRowOffset, Formal.evalARowOffset, Formal.evalKRowOffset,
    Formal.sumcheckRowOffset, Formal.initialClaimRowOffset,
    InitialClaim.privateCount, EvalKTerminal.privateCount,
    EvalATerminal.privateCount, CcsTerminal.privateCount,
    NormTerminal.privateCount, productionShape, Phi81MatrixSource.phi81Shape,
    cubeVariables, RoundTranscript.perRoundRecipeCount]

private theorem outputBindingOffset_eq :
    Formal.outputBindingOffset relation
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.interface
        logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset = 15256706 := by
  rw [Formal.outputBindingOffset_eq_outputBindingRowOffset,
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]
  norm_num [Formal.outputBindingRowOffset, Formal.finalIdentityRowOffset,
    Formal.normRowOffset, Formal.ccsRowOffset, Formal.evalARowOffset,
    Formal.evalKRowOffset, Formal.sumcheckRowOffset,
    Formal.initialClaimRowOffset, InitialClaim.privateCount,
    EvalKTerminal.privateCount, EvalATerminal.privateCount,
    CcsTerminal.privateCount, NormTerminal.privateCount,
    FinalIdentity.privateCount, productionShape,
    Phi81MatrixSource.phi81Shape, cubeVariables,
    RoundTranscript.perRoundRecipeCount]

private theorem logicalColumnCount_eq :
    NightstreamFPrime.Layout.PiCCS.v1_1.logicalColumnCount relation
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.interface
        logicalWidth publicFits)
      NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset = 19333218 := by
  rw [← NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase_eq_layout
    relation]
  norm_num [NightstreamFPrime.Layout.Stage1.PiCCSStarts.logicalFreshBase,
    NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]

private theorem logicalColumnCount_literal_eq :
    NightstreamFPrime.Layout.PiCCS.v1_1.logicalColumnCount relation
      (NightstreamFPrime.Layout.Stage1.PiCCSInputs.interface
        logicalWidth publicFits) 14751804 = 19333218 := by
  simpa only [NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq] using
    logicalColumnCount_eq

private theorem columnSpanFamily_eq_exportedColumnFamily (column : Nat) :
    columnSpanFamily column = exportedColumnFamily column := by
  unfold exportedColumnFamily
    NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.columnOwnerAt
  simp only [interface, parentOffset,
    NightstreamFPrime.Layout.Stage1.PilotPiCCS.interface,
    NightstreamFPrime.Layout.Stage1.PilotPiCCS.piCcsOffset]
  rw [statementAbsorptionOffset_eq]
  rw [challengeOffset_eq]
  rw [roundTranscriptOffset_eq]
  rw [initialClaimOffset_eq]
  rw [sumcheckOffset_eq]
  rw [evalKOffset_eq]
  rw [evalAOffset_eq]
  rw [ccsOffset_eq]
  rw [normOffset_eq]
  rw [finalIdentityOffset_eq]
  rw [outputBindingOffset_eq]
  simp only [NightstreamFPrime.Layout.Stage1.PiCCSInputs.phaseOffset_eq]
  simp only [logicalColumnCount_literal_eq]
  simp [columnSpanFamily, familyAtSpanEnds, columnSpans, spansFrom,
    columnOrder, columnCounts, Span.endExclusive]
  rw [apply_ite columnFamily]
  rw [apply_ite columnFamily]
  rw [apply_ite columnFamily]
  rw [apply_ite columnFamily]
  rw [apply_ite columnFamily]
  rw [apply_ite columnFamily]
  rw [apply_ite columnFamily]
  rw [apply_ite columnFamily]
  rw [apply_ite columnFamily]
  rw [apply_ite columnFamily]
  rw [apply_ite columnFamily]
  rw [apply_ite columnFamily]
  rw [apply_ite columnFamily]
  rfl

/-- Every physical column receives the same family from the exported spans
and from the authoritative PiCCS owner function. -/
theorem columnSpans_pointwise_ownerAgreement
    (column : Fin (NightstreamFPrime.Layout.PiCCS.v1_1.physicalColumnCount
      relation interface parentOffset)) :
    columnFamily
        (NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.columnOwner relation
          interface parentOffset column) =
      exportedColumnFamily column.val := by
  unfold NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.columnOwner
    exportedColumnFamily
  rfl

/-- Direct lookup in the exported span ledger agrees with the authoritative
owner of every physical column. -/
theorem columnSpans_lookup_ownerAgreement
    (column : Fin (NightstreamFPrime.Layout.PiCCS.v1_1.physicalColumnCount
      relation interface parentOffset)) :
    columnSpanFamily column.val =
      columnFamily
        (NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.columnOwner relation
          interface parentOffset column) := by
  rw [columnSpanFamily_eq_exportedColumnFamily,
    columnSpans_pointwise_ownerAgreement]

def childOwnerTag :
    NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.ChildOwner → String
  | .statementBinding => "statement_binding"
  | .statementAbsorption => "statement_absorption"
  | .challengeDerivation => "challenge_derivation"
  | .roundTranscript => "round_transcript"
  | .initialClaim => "initial_claim"
  | .sumcheckChain => "sumcheck_chain"
  | .eval_K => "eval_k"
  | .eval_A => "eval_a"
  | .ccsTerminal => "ccs_terminal"
  | .normTerminal => "norm_terminal"
  | .finalIdentity => "final_identity"
  | .outputBinding => "output_binding"

def columnFamilyTag : ColumnFamily → String
  | .external => "external"
  | .statementBinding => "statement_binding"
  | .statementAbsorption => "statement_absorption"
  | .challengeDerivation => "challenge_derivation"
  | .roundTranscript => "round_transcript"
  | .initialClaim => "initial_claim"
  | .sumcheckChain => "sumcheck_chain"
  | .eval_K => "eval_k"
  | .eval_A => "eval_a"
  | .ccsTerminal => "ccs_terminal"
  | .normTerminal => "norm_terminal"
  | .finalIdentity => "final_identity"
  | .outputBinding => "output_binding"
  | .r1csIntermediate => "r1cs_intermediate"

theorem rowSpanTags_exact :
    rowSpans.map (fun span => childOwnerTag span.owner) =
      ["statement_binding", "statement_absorption", "challenge_derivation",
       "round_transcript", "initial_claim", "sumcheck_chain", "eval_k",
       "eval_a", "ccs_terminal", "norm_terminal", "final_identity",
       "output_binding"] := by
  rfl

theorem columnSpanTags_exact :
    columnSpans.map (fun span => columnFamilyTag span.owner) =
      ["external", "statement_binding", "statement_absorption",
       "challenge_derivation", "round_transcript", "initial_claim",
       "sumcheck_chain", "eval_k", "eval_a", "ccs_terminal",
       "norm_terminal", "final_identity", "output_binding",
       "r1cs_intermediate"] := by
  rfl

theorem structuralIdentity_eq_canonical :
    PerApplicationStreamingIdentity.structuralPackageIdentityDirect application
        Poseidon2HashChainV1Package.fits =
      PerApplicationCanonicalPackage.structuralPackageIdentity application
        Poseidon2HashChainV1Package.fits := by
  exact PerApplicationStreamingIdentity.structuralPackageIdentityDirect_eq
    application Poseidon2HashChainV1Package.fits

theorem structuralIdentityWords_length :
    (PerApplicationStreamingIdentity.structuralPackageIdentityDirect application
      Poseidon2HashChainV1Package.fits).toList.length = 4 := by
  exact VerifierContext.Digest4.toList_length _

private def quote (value : String) : String := "\"" ++ value ++ "\""

private def renderList (values : List String) : String :=
  "[" ++ String.intercalate "," values ++ "]"

private def renderSpan {Owner : Type} (tag : Owner → String)
    (span : Span Owner) : String :=
  "{\"owner\":" ++ quote (tag span.owner) ++
    ",\"start\":" ++ toString span.start ++
    ",\"count\":" ++ toString span.count ++ "}"

/-- Compact schema-1 JSON object for a consumer-recomputed structural identity.
Identity words are strings so consumers do not lose field precision; span
coordinates are natural-number JSON values. -/
def render (identity : VerifierContext.Digest4) : String :=
  "{\"schemaVersion\":1,\"structuralIdentity\":" ++
    renderList (identity.toList.map (fun word => quote (toString word.val))) ++
    ",\"rowSpans\":" ++ renderList (rowSpans.map (renderSpan childOwnerTag)) ++
    ",\"columnSpans\":" ++
      renderList (columnSpans.map (renderSpan columnFamilyTag)) ++ "}"

end NightstreamFPrime.Export.Stage1.PiCCSOwnershipAudit
