import Nightstream.Assurance.FPrimeFullHistory.TerminalShell
import Nightstream.Implementation.Lowering.Goldilocks.SourceAlignment
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryCurrentTerminalLinkPlacement
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.DiagnosticProfile
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.RelationProfile
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryManifest
import Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne

/-!
Contract: exact stop boundary for selecting the current fixed-one production
owner program.

Assurance tier: model-level obstruction plus artifact-range obstruction.

Owns:
- a kernel countermodel showing that structural receipt/source-owner alignment
  does not imply semantic refinement of the indexed typed program;
- the exact range obstruction between the historical complete full-history
  artifact and the bounded current 270-row terminal-link placement;
- the exact matrix-count mismatch between the bounded three-evaluation
  diagnostic fixture and the thirteen-matrix production relation;
- a kernel countermodel showing that the current terminal fact carrier does
  not select the independent running/fresh relations required by the frozen
  Construction-2 terminal verifier.

Does not own: a current whole-program artifact, a production phase
instruction vocabulary, phase-level semantic recipes, a concrete terminal
relation selection, source-authority reconstruction, Rust equality, or row
generation.

Emits constraints: no.
-/

namespace Nightstream.Assurance.FPrimeProductionOwnerProgramBoundary

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks

/-! ## Structural owner alignment is not semantic refinement -/

namespace AlignmentOpacity

/-- Small typed universe used only to expose the exact source-alignment
boundary. The semantic bit intentionally has no physical coordinates. -/
def types : TypeSystem where
  Field := Nat
  zero := 0
  add := Nat.add
  mul := Nat.mul
  Bit := Bool
  bitValue := id
  Data := Empty
  dataValue := fun tag => nomatch tag

/-- No opaque calls are needed for the countermodel. -/
def signature : Signature where
  types := types
  Call := Empty
  callInputs := fun call => nomatch call
  callOutputs := fun call => nomatch call
  callEval := fun call => nomatch call
  callFootprint := fun call => nomatch call

def bitPort : Port types where
  kind := .bit
  layout := ⟨[]⟩

abbrev Input : Schema types := [bitPort]
abbrev Output : Schema types := []

/-- The source program rejects when its semantic input bit is false. -/
def source : Program signature Input Output where
  body :=
    .step
      (.assertTrue (.here bitPort))
      (.yield .nil)

def falseInput : Schema.Values types Input :=
  .cons false .nil

/-- A structurally present owner may own no physical occurrence. This is
permitted by `AlignedReceiptProgram` and is exactly why owner equality alone
cannot establish instruction semantics. -/
def emptyReceipt
    (owner : PhysicalOwner)
    (kind : InstructionKind) :
    InstructionReceipt where
  owner := owner
  kind := kind
  allocations := []
  rows := []
  allocationsOwned := by simp
  rowsOwned := by simp

def receipts : List InstructionReceipt :=
  [ InstructionReceipt.prelude
  , emptyReceipt (.typed (.input 0)) .input
  , emptyReceipt (.typed (.instruction .root)) .assertion
  ]

def physical : ReceiptProgram source where
  receipts := receipts
  preludeMember := by simp [receipts]
  ownersNodup := by decide
  localColumnIdsNodup := by
    intro receipt member
    simp only [receipts, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with equal | equal | equal
    · subst receipt
      simp [InstructionReceipt.columnIds, InstructionReceipt.prelude,
        preludeColumns]
    · subst receipt
      simp [InstructionReceipt.columnIds, emptyReceipt]
    · subst receipt
      simp [InstructionReceipt.columnIds, emptyReceipt]
  localRowIdsNodup := by
    intro receipt member
    simp only [receipts, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with equal | equal | equal
    · subst receipt
      simp [InstructionReceipt.rowIds, InstructionReceipt.prelude]
    · subst receipt
      simp [InstructionReceipt.rowIds, emptyReceipt]
    · subst receipt
      simp [InstructionReceipt.rowIds, emptyReceipt]
  wellScoped := by
    simp [receipts, ReceiptsWellScoped, InstructionReceipt.WellScopedAfter,
      InstructionReceipt.referencedColumns, emptyReceipt]

/-- The receipt owners are occurrence-for-occurrence equal to the complete
typed source skeleton. -/
def aligned : SourceAlignment.AlignedReceiptProgram source where
  physical := physical
  ownersExact := rfl

def assignment : ColumnId -> Nightstream.SuperNeo.Concrete.F :=
  fun _ => 1

/-- Every selected physical row holds; there are no rows beyond the
verifier-owned constant-one prelude. -/
theorem physicalSatisfies :
    aligned.toEncoding.PhysicalSatisfies assignment := by
  constructor
  · rfl
  · change Satisfies [] assignment
    exact True.intro

/-- The indexed semantic source nevertheless rejects the chosen input. -/
theorem sourceRejects :
    source.exec falseInput = none :=
  rfl

/-- The strongest direct semantic bridge available from structural alignment
and physical satisfaction alone. -/
def AttemptedOwnerAlignmentBridge : Prop :=
  aligned.toEncoding.PhysicalSatisfies assignment ->
    ∃ result, source.exec falseInput = some result

/-- Structural owner equality and row conservation do not supply the missing
phase-level semantic refinement theorem. -/
theorem not_attemptedOwnerAlignmentBridge :
    ¬ AttemptedOwnerAlignmentBridge := by
  intro bridge
  rcases bridge physicalSatisfies with ⟨result, executed⟩
  rw [sourceRejects] at executed
  contradiction

end AlignmentOpacity

/-! ## The checked complete artifact is historical, not current -/

namespace CurrentArtifact

open Nightstream.Implementation.R1CS

/-- The bounded live terminal-link owner starts strictly after the final row
of the checked complete historical artifact. -/
theorem currentTerminalLink_starts_after_historicalProgram :
    FPrimeFullHistoryManifest.totalRows <
      FPrimeFullHistoryCurrentTerminalLinkPlacement.rowStart := by
  decide

/-- Consequently the current bounded range is not contained in the checked
historical whole-program interval. -/
theorem currentTerminalLink_not_in_historicalProgram :
    ¬ (FPrimeFullHistoryCurrentTerminalLinkPlacement.rowStart <
          FPrimeFullHistoryManifest.totalRows ∧
        FPrimeFullHistoryCurrentTerminalLinkPlacement.rowEnd <=
          FPrimeFullHistoryManifest.totalRows) := by
  decide

end CurrentArtifact

/-! ## The bounded current capture is not the selected production relation -/

namespace CurrentProfile

open Nightstream.Implementation.R1CS

/-- The bounded diagnostic is the direct-R1CS `A/B/C` structure. -/
theorem diagnosticMatrixCount_eq_three :
    FPrimeFullHistoryNifsPaper.PiRlc.DiagnosticProfile.matrixCount = 3 := by
  rfl

/-- The selected fixed-point relation owns all thirteen selective ports. -/
theorem activeProductionMatrixCount_eq_thirteen :
    FPrimeFullHistorySelectiveCcs.RelationProfile.matrixCount = 13 := by
  decide

/-- The bounded full-history terminal diagnostic and the independently
selected production relation have different CCS matrix arities `t`. The
fourteen terminal CE claims are a separate source-count dimension. This is a
capture-provenance obstruction, not a claim that either relation is unsound. -/
theorem diagnosticMatrixCount_ne_activeProduction :
    FPrimeFullHistoryNifsPaper.PiRlc.DiagnosticProfile.matrixCount ≠
      FPrimeFullHistorySelectiveCcs.RelationProfile.matrixCount := by
  decide

end CurrentProfile

/-! ## Terminal physical facts do not select paper relation checkers -/

namespace TerminalSelectionOpacity

open Nightstream.Protocol.FPrime.CanonicalTerminalVerifier

def setup : Setup Unit Unit Unit Unit 1 where
  verifierKeys := fun _ => ()
  nifs := { verify := fun _ _ _ _ => none }
  defaultRunning := ()

def machine : Machine Unit Unit Bool Unit Unit Unit Unit 1 where
  control := fun _ _ => ⟨0, by decide⟩
  step := fun _ state _ => state
  freshPublic := fun _ => ()
  encodeInstance := fun _ => ()
  hash := fun _ => ()

def acceptingRelations :
    TerminalRelations Unit Unit Unit Unit Unit 1 where
  runningHolds := fun _ _ _ _ => True
  freshHolds := fun _ _ _ _ => True

def acceptingChecks :
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
      acceptingRelations where
  runningCheck := fun _ _ _ _ => true
  freshCheck := fun _ _ _ _ => true
  runningCheck_iff := by simp [acceptingRelations]
  freshCheck_iff := by simp [acceptingRelations]

def rejectingRelations :
    TerminalRelations Unit Unit Unit Unit Unit 1 where
  runningHolds := fun _ _ _ _ => False
  freshHolds := fun _ _ _ _ => False

def rejectingChecks :
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
      rejectingRelations where
  runningCheck := fun _ _ _ _ => false
  freshCheck := fun _ _ _ _ => false
  runningCheck_iff := by simp [rejectingRelations]
  freshCheck_iff := by simp [rejectingRelations]

def statement : TerminalStatement Bool where
  iteration := 1
  z0 := false
  zi := false

def proof : FixedOne.Proof Unit Unit Unit Unit where
  running := ()
  runningWitness := ()
  fresh := ()
  freshWitness := ()

theorem acceptingSelection_accepts :
    FixedOne.Accepts setup machine acceptingRelations acceptingChecks
      statement proof := by
  change true = true
  rfl

theorem rejectingSelection_rejects :
    ¬ FixedOne.Accepts setup machine rejectingRelations rejectingChecks
      statement proof := by
  change ¬(false = true)
  simp

/-- The exact terminal fact carrier reconstructed from physical rows is
independent of the `TerminalRelations` and `RelationChecks` required by the
frozen terminal verifier. The same facts therefore cannot select which of
these two exact checker instantiations is the production one. -/
theorem terminalFacts_do_not_select_relationChecks
    {assignment : Nat -> Nat}
    (_ :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalShellSound.TerminalFacts
        assignment) :
    FixedOne.Accepts setup machine acceptingRelations acceptingChecks
        statement proof ∧
      ¬ FixedOne.Accepts setup machine rejectingRelations rejectingChecks
        statement proof :=
  ⟨acceptingSelection_accepts, rejectingSelection_rejects⟩

end TerminalSelectionOpacity

end Nightstream.Assurance.FPrimeProductionOwnerProgramBoundary
