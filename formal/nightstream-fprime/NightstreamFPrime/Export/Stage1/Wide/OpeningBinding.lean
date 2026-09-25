import NightstreamFPrime.Export.Stage1.Wide.SetupBinding
import NightstreamFPrime.Export.Stage1.Wide.PackageAuthority
import NightstreamFPrime.Layout.ProductionRelation.AcceptedOpening
import NightstreamFPrime.Lifecycle.Stage1.Terminal

/-! Connect the verifier's inputs to the wide step. Row acceptance is read
from the sealed matrix program that the verifier decodes, and a fresh CCS
opening supplies both those rows and the public input. The prepared package
is the authority; no separate Lean-only plan is trusted. -/

namespace NightstreamFPrime.Export.Stage1.Wide.OpeningBinding

open NightstreamFPrime.Spec NightstreamFPrime.Layout NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding.PiCCS.PaperJoint Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open ProductionRelation
open Spec.HyperNova.Construction2.Paper

/-- One decoded row as its 14 matrix ports; slot 13 is the empty form. -/
def portOf {logicalWidth : Nat}
    (forms : Fin Spec.ProductionRelation.meaningfulPortCount → SparseForm logicalWidth)
    (port : Fin Spec.ProductionRelation.matrixCount) : SparseForm logicalWidth :=
  match meaningfulPort? port with
  | some meaningful => forms meaningful
  | none => .empty

/-- Row acceptance read directly from a matrix program: every row decodes,
and its port evaluations satisfy the production polynomial. -/
def ProgramRowsZero (program : MatrixProgram.Program) (logicalWidth : Nat)
    (sourceRow : Nat → Option R1CS.Row) (assignment : Assignment F logicalWidth) : Prop :=
  ∀ row < program.rowCount, ∃ forms, program.row? logicalWidth sourceRow row = some forms ∧
    evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
      (fun port => (portOf forms port).eval assignment) = 0

/-- An exact program accepts an assignment exactly when its plan does. -/
theorem programRowsZero_iff {logicalWidth : Nat} {program : MatrixProgram.Program}
    {plan : ProductionRelation.Plan logicalWidth} {sourceRow : Nat → Option R1CS.Row}
    (exact : MatrixProgram.Exact program plan sourceRow) (assignment : Assignment F logicalWidth) :
    ProgramRowsZero program logicalWidth sourceRow assignment ↔ plan.RowsZero assignment := by
  have image (row : Fin plan.rowCount) :
      plan.rowImage assignment (plan.rowLayout.toVertex row) =
        fun port => (portOf (plan.forms row) port).eval assignment := by
    rw [Plan.rowImage_toVertex]
    rfl
  constructor
  · intro rows row
    have bounded : row.val < program.rowCount := by
      rw [exact.rowCount]
      exact row.isLt
    obtain ⟨forms, decoded, zero⟩ := rows row.val bounded
    rw [exact.row? row, Option.some.injEq] at decoded
    rw [image, decoded]
    exact zero
  · intro rows row bounded
    have live : row < plan.rowCount := by
      rw [← exact.rowCount]
      exact bounded
    refine ⟨plan.forms ⟨row, live⟩, exact.row? ⟨row, live⟩, ?_⟩
    have zero := rows ⟨row, live⟩
    rw [image] at zero
    exact zero

/-- The step or the collision follows from the rows of the sealed matrix
program, read through that package's own source archive. -/
theorem step_or_collision_of_matrix (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (parts : AuthorityStream.Parts) (built : AuthorityStream.prepare compiled = .ok parts)
    (assignment : Assignment F (RetainedLayout.logicalWidth SetupBinding.application))
    (claimed : HashPreimage (logicalWidth := RetainedLayout.logicalWidth SetupBinding.application)
      (publicFits := FixedPoint.publicFits SetupBinding.application))
    (publicDigest : Digest) (digestLength : publicDigest.length = 4)
    (publicEqual : PublicBinding.publicInput SetupBinding.application assignment = encHash publicDigest)
    (checkedDigest : publicDigest =
      stateHash { claimed with verifierKeys := fun _ => SetupBinding.contextKey (SetupBinding.descriptor parts) })
    (rows : ProgramRowsZero parts.matrix (RetainedLayout.logicalWidth SetupBinding.application)
      (PackageSourceRows.packageSourceRow? parts.package) assignment) :
    Lifecycle.Stage1.Wide.Relation.StepHoldsFor
        (FixedPoint.relation SetupBinding.application compiled SetupBinding.fits)
        SetupBinding.productionAjtaiKey (SetupBinding.contextKey (SetupBinding.descriptor parts))
        SetupBinding.application
        (FixedPointSoundness.input SetupBinding.application assignment
          (FixedPoint.relation SetupBinding.application compiled SetupBinding.fits))
        (FixedPointSoundness.output SetupBinding.application assignment
          (RetainedLayout.logicalWidth SetupBinding.application)
          (FixedPoint.publicFits SetupBinding.application)) ∨
      Layout.Stage1.PiCCSSecurity.StateHashCollision
        (ContextBinding.decodedNext SetupBinding.application assignment compiled SetupBinding.fits
          SetupBinding.productionAjtaiKey)
        { claimed with verifierKeys := fun _ => SetupBinding.contextKey (SetupBinding.descriptor parts) } :=
  SetupBinding.step_or_collision compiled parts built assignment claimed publicDigest digestLength
    publicEqual checkedDigest
    ((programRowsZero_iff (PackageAuthority.matrix_exact compiled parts built) assignment).mp rows)

theorem publicLogicalFits (program : RetainedLayout.Program) :
    ringDegree * publicRingColumns ≤ RetainedLayout.logicalWidth program := by
  rw [RetainedLayout.logicalWidth_eq]
  change 270 ≤ _
  omega

/-- A fresh CCS opening for the candidate relation gives its logical rows and
the actual public input, for arbitrary values in the carrier padding. -/
theorem freshHolds_implies_rowsAndPublic (program : RetainedLayout.Program)
    (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 program)
    (key : AjtaiKey (logicalWidth := RetainedLayout.logicalWidth program)
      (publicFits := FixedPoint.publicFits program))
    (fresh : Fresh (logicalWidth := RetainedLayout.logicalWidth program)
      (publicFits := FixedPoint.publicFits program))
    (assignment : Assignment F (Phi81CarrierLayout.carrierWidth (RetainedLayout.logicalWidth program)))
    (accepted : CCS.Holds (semantics key) productionGlobalParams
      (freshStatement (FixedPoint.relation program compiled fits) fresh) assignment) :
    (FixedPoint.structuralPlan program compiled fits).RowsZero (Plan.logicalAssignment assignment) ∧
      PublicBinding.publicInput program (Plan.logicalAssignment assignment) =
        fresh.publicInputs ⟨0, by decide⟩ :=
  Plan.freshHolds_implies_rowsAndPublic (FixedPoint.structuralPlan program compiled fits)
    (FixedPoint.carrierFits program fits) key fresh assignment (publicLogicalFits program) accepted

abbrev TerminalPayload (program : RetainedLayout.Program) := TerminalProof
  (Running (logicalWidth := RetainedLayout.logicalWidth program) (publicFits := FixedPoint.publicFits program))
  (Stage1.Terminal.RunningWitness (logicalWidth := RetainedLayout.logicalWidth program)
    (publicFits := FixedPoint.publicFits program))
  (Fresh (logicalWidth := RetainedLayout.logicalWidth program) (publicFits := FixedPoint.publicFits program))
  (Stage1.Terminal.FreshWitness (logicalWidth := RetainedLayout.logicalWidth program)
    (publicFits := FixedPoint.publicFits program)) slotCount

/-- The preimage hashed by the recursive terminal public check. -/
def terminalPreimage (program : RetainedLayout.Program) (vk : KeyDigest)
    (statement : TerminalStatement AppState) (payload : TerminalPayload program) :
    HashPreimage (logicalWidth := RetainedLayout.logicalWidth program)
      (publicFits := FixedPoint.publicFits program) where
  verifierKeys := fun _ => vk
  iteration := statement.iteration
  z0 := statement.z0
  current := statement.zi
  running := payload.running
  pc := payload.pc

/-- An accepted recursive terminal for the candidate relation gives the
HyperNova step of its fresh opening, or the named state-hash collision. The
terminal check supplies the rows, the public input and the checked digest. -/
theorem terminal_implies_stepOrCollision (program : RetainedLayout.Program)
    (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 program)
    (ajtai : AjtaiKey (logicalWidth := RetainedLayout.logicalWidth program)
      (publicFits := FixedPoint.publicFits program))
    (vk : KeyDigest) (vkLength : vk.length = 4)
    (statement : TerminalStatement AppState) (payload : TerminalPayload program)
    (terminal : Stage1.Terminal.HoldsFor (FixedPoint.relation program compiled fits) ajtai vk
      program statement (.recursive payload)) :
    Lifecycle.Stage1.Wide.Relation.StepHoldsFor (FixedPoint.relation program compiled fits) ajtai vk program
        (FixedPointSoundness.input program (Plan.logicalAssignment payload.freshWitness)
          (FixedPoint.relation program compiled fits))
        (FixedPointSoundness.output program (Plan.logicalAssignment payload.freshWitness)
          (RetainedLayout.logicalWidth program) (FixedPoint.publicFits program)) ∨
      Layout.Stage1.PiCCSSecurity.StateHashCollision
        (ContextBinding.decodedNext program (Plan.logicalAssignment payload.freshWitness) compiled fits ajtai)
        { terminalPreimage program vk statement payload with verifierKeys := fun _ => vk } := by
  rcases (Stage1.Terminal.holdsFor_recursive_iff (FixedPoint.relation program compiled fits) ajtai vk
    program statement payload).mp terminal with
    ⟨_statementValid, _pcValid, _positive, publicLink, _runningValid, freshValid⟩
  obtain ⟨rows, publicEqual⟩ :=
    freshHolds_implies_rowsAndPublic program compiled fits ajtai payload.fresh payload.freshWitness freshValid
  change payload.fresh.publicInputs ⟨0, by decide⟩ =
    encHash (stateHash (terminalPreimage program vk statement payload)) at publicLink
  exact ContextBinding.step_or_collision program (Plan.logicalAssignment payload.freshWitness) compiled fits
    ajtai vk vkLength (terminalPreimage program vk statement payload)
    (stateHash (terminalPreimage program vk statement payload))
    (Layout.Stage1.StateEncoding.stateHash_length _) (publicEqual.trans publicLink) rfl rows

end NightstreamFPrime.Export.Stage1.Wide.OpeningBinding
