import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationCertification
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalEncodingRealization
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepPhysicalCompleteness
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalTerminalPhysicalCompleteness

/-!
Contract: complete profile-indexed canonical encoding for the fixed-one Step
and Terminal checkers.

HyperNova Construction 2 supplies the application circuit, while the NIFS
setup supplies its verifier.  Their physical encodings therefore enter this
boundary as complete `CallRecipe` programs, not as acceptance propositions or
caller-provided semantic conclusions.

Owns: the final `step` and `nifsVerify` recipe boundary, assembly of all
eleven recipes, the canonical Step and Terminal encodings, their exact
checker correspondence, honest satisfying assignments, receipt ownership,
conservation, and program-derived cost.

Does not own: a deployment application selection, Rust, generated rows,
`BatchBadRoot` suppression, Fiat--Shamir, or cryptographic security.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

/-- The two application/setup-provided physical programs needed after
canonical Phases 3 and 4.

Both fields are the existing deep `CallRecipe` contract: each contains an
executable row program, exact footprint and receipt, active soundness, honest
active completeness, and inactive satisfiability.  A selected deployment must
construct these programs on the Lean-owned encoding path; equality with Rust
is a later refinement theorem, never an input to this boundary. -/
structure Phase5CallCertification
    (parameters : Parameters)
    (profile : Poseidon23ApplicationProfile parameters) where
  step :
    CallRecipe (signature parameters)
      (profile.family parameters) Call.step
  nifsVerify :
    CallRecipe (signature parameters)
      (profile.family parameters) Call.nifsVerify

/-- Complete profile-indexed certificate for the canonical physical Step and
Terminal programs.  The default running value must lie in the selected
codec's honest domain because the base branch materializes it without prover
advice. -/
structure CompleteApplicationCertification (parameters : Parameters) where
  profile : Poseidon23ApplicationProfile parameters
  phase5 : Phase5CallCertification parameters profile
  defaultRunningAdmissible :
    ((profile.family parameters).codecFor (.data .running)).Admissible
      (defaultRunning parameters)

namespace CompleteApplicationCertification

def directProfile
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    DirectCalls.DirectProfile parameters :=
  certificate.profile.toTerminalEqualityProfile.toDirectProfile

def baseProfile
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    Profile parameters :=
  certificate.directProfile.toProfile

def phase34
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    ApplicationCertification parameters :=
  ApplicationCertification.poseidon23 parameters certificate.profile

/-- The exact six-call remainder consumed by `DirectCalls.allRecipes`. -/
def remainingRecipes
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    DirectCalls.RemainingRecipes parameters certificate.directProfile where
  step := certificate.phase5.step
  hashPrior := certificate.phase34.hashPrior
  hashNext := certificate.phase34.hashNext
  nifsVerify := certificate.phase5.nifsVerify
  runningCheck := certificate.phase34.runningCheck
  freshCheck := certificate.phase34.freshCheck

/-- All eleven certified physical call programs. -/
def allRecipes
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    CallRecipes (signature parameters)
      (certificate.baseProfile.family parameters) :=
  DirectCalls.allRecipes parameters certificate.directProfile
    certificate.remainingRecipes

/-- The complete recipe-family domain, in constructor order. -/
def recipeCalls : List Call :=
  [.iterationZero, .stateEqual, .step, .hashPrior, .hashNext,
    .freshPublic, .encodeInstance, .encodedEqual, .nifsVerify,
    .runningCheck, .freshCheck]

theorem recipeCalls_exact :
    recipeCalls =
      [.iterationZero, .stateEqual, .step, .hashPrior, .hashNext,
        .freshPublic, .encodeInstance, .encodedEqual, .nifsVerify,
        .runningCheck, .freshCheck] :=
  rfl

/-- Every call has exactly one recipe position in the closed family.  This is
not a claim about occurrences in either typed program. -/
theorem recipe_family_multiplicities :
    recipeCalls.count Call.iterationZero = 1 ∧
      recipeCalls.count Call.stateEqual = 1 ∧
      recipeCalls.count Call.step = 1 ∧
      recipeCalls.count Call.hashPrior = 1 ∧
      recipeCalls.count Call.hashNext = 1 ∧
      recipeCalls.count Call.freshPublic = 1 ∧
      recipeCalls.count Call.encodeInstance = 1 ∧
      recipeCalls.count Call.encodedEqual = 1 ∧
      recipeCalls.count Call.nifsVerify = 1 ∧
      recipeCalls.count Call.runningCheck = 1 ∧
      recipeCalls.count Call.freshCheck = 1 := by
  decide

/-- Calls mentioned by one typed primitive. -/
private def callsOfPrimitive
    {signature : Signature}
    {input output : Schema signature.types} :
    Primitive signature input output -> List signature.Call
  | .invoke call _ => [call]
  | _ => []

/-- Calls mentioned structurally by a typed block, including both private
branch arms.  This is the source-owned multiplicity extractor used below. -/
private def callsOfBlock
    {signature : Signature} :
    {input output : Schema signature.types} ->
      Block signature input output -> List signature.Call
  | _, _, .yield _ => []
  | _, _, .step primitive rest =>
      callsOfPrimitive primitive ++ callsOfBlock rest
  | _, _, .branch _ onTrue onFalse continuation =>
      callsOfBlock onTrue ++ callsOfBlock onFalse ++
        callsOfBlock continuation

/-- Exact call occurrences derived from the typed Step AST. -/
def stepProgramCalls (parameters : Parameters) : List Call :=
  callsOfBlock
    (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.program
      parameters).body

/-- Exact call occurrences derived from the typed Terminal AST. -/
def terminalProgramCalls (parameters : Parameters) : List Call :=
  callsOfBlock
    (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.program
      parameters).body

theorem stepProgramCalls_exact (parameters : Parameters) :
    stepProgramCalls parameters =
      [.step, .iterationZero, .stateEqual, .hashPrior, .freshPublic,
        .encodeInstance, .encodedEqual, .nifsVerify, .hashNext] :=
  rfl

theorem terminalProgramCalls_exact (parameters : Parameters) :
    terminalProgramCalls parameters =
      [.iterationZero, .stateEqual, .hashPrior, .freshPublic,
        .encodeInstance, .encodedEqual, .runningCheck, .freshCheck] :=
  rfl

/-- Step multiplicities are computed from the typed source program. -/
theorem step_call_multiplicities (parameters : Parameters) :
    (stepProgramCalls parameters).count Call.step = 1 ∧
      (stepProgramCalls parameters).count Call.iterationZero = 1 ∧
      (stepProgramCalls parameters).count Call.stateEqual = 1 ∧
      (stepProgramCalls parameters).count Call.hashPrior = 1 ∧
      (stepProgramCalls parameters).count Call.freshPublic = 1 ∧
      (stepProgramCalls parameters).count Call.encodeInstance = 1 ∧
      (stepProgramCalls parameters).count Call.encodedEqual = 1 ∧
      (stepProgramCalls parameters).count Call.nifsVerify = 1 ∧
      (stepProgramCalls parameters).count Call.hashNext = 1 ∧
      (stepProgramCalls parameters).count Call.runningCheck = 0 ∧
      (stepProgramCalls parameters).count Call.freshCheck = 0 := by
  rw [stepProgramCalls_exact]
  decide

/-- Terminal multiplicities are computed from the typed source program. -/
theorem terminal_call_multiplicities (parameters : Parameters) :
    (terminalProgramCalls parameters).count Call.iterationZero = 1 ∧
      (terminalProgramCalls parameters).count Call.stateEqual = 1 ∧
      (terminalProgramCalls parameters).count Call.hashPrior = 1 ∧
      (terminalProgramCalls parameters).count Call.freshPublic = 1 ∧
      (terminalProgramCalls parameters).count Call.encodeInstance = 1 ∧
      (terminalProgramCalls parameters).count Call.encodedEqual = 1 ∧
      (terminalProgramCalls parameters).count Call.runningCheck = 1 ∧
      (terminalProgramCalls parameters).count Call.freshCheck = 1 ∧
      (terminalProgramCalls parameters).count Call.step = 0 ∧
      (terminalProgramCalls parameters).count Call.hashNext = 0 ∧
      (terminalProgramCalls parameters).count Call.nifsVerify = 0 := by
  rw [terminalProgramCalls_exact]
  decide

/-- A fixed physical coordinate string has at most one typed schema
interpretation.  This rules out a ghost result value in the final
checker-equivalence theorem. -/
private theorem decodedValues_unique
    {types : TypeSystem}
    (family : Family types)
    (assignment : ColumnId -> Field)
    {schema : Schema types}
    (bundles : SchemaBundles schema)
    {left right : Schema.Values types schema}
    (leftDecoded : bundles.Decodes family assignment left)
    (rightDecoded : bundles.Decodes family assignment right) :
    left = right := by
  induction bundles with
  | nil =>
      cases left
      cases right
      rfl
  | @cons port tail head rest inductionHypothesis =>
      cases left with
      | cons leftHead leftTail =>
          cases right with
          | cons rightHead rightTail =>
              have headEqual : leftHead = rightHead :=
                (family.codecFor port.kind).decoded_value_unique
                  leftDecoded.1 rightDecoded.1
              have tailEqual : leftTail = rightTail :=
                inductionHypothesis leftDecoded.2 rightDecoded.2
              cases headEqual
              cases tailEqual
              rfl

@[simp] theorem allRecipes_step
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.allRecipes.recipe Call.step = certificate.phase5.step :=
  rfl

@[simp] theorem allRecipes_iterationZero
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.allRecipes.recipe Call.iterationZero =
      DirectCalls.iterationZeroRecipe parameters certificate.directProfile :=
  rfl

@[simp] theorem allRecipes_stateEqual
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.allRecipes.recipe Call.stateEqual =
      DirectCalls.stateEqualRecipe parameters certificate.directProfile :=
  rfl

@[simp] theorem allRecipes_nifsVerify
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.allRecipes.recipe Call.nifsVerify =
      certificate.phase5.nifsVerify :=
  rfl

@[simp] theorem allRecipes_hashPrior
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.allRecipes.recipe Call.hashPrior =
      certificate.phase34.hashPrior :=
  rfl

@[simp] theorem allRecipes_hashNext
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.allRecipes.recipe Call.hashNext =
      certificate.phase34.hashNext :=
  rfl

@[simp] theorem allRecipes_freshPublic
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.allRecipes.recipe Call.freshPublic =
      DirectCalls.freshPublicRecipe parameters certificate.directProfile :=
  rfl

@[simp] theorem allRecipes_encodeInstance
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.allRecipes.recipe Call.encodeInstance =
      DirectCalls.encodeInstanceRecipe parameters certificate.directProfile :=
  rfl

@[simp] theorem allRecipes_encodedEqual
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.allRecipes.recipe Call.encodedEqual =
      DirectCalls.encodedEqualRecipe parameters certificate.directProfile :=
  rfl

@[simp] theorem allRecipes_runningCheck
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.allRecipes.recipe Call.runningCheck =
      certificate.phase34.runningCheck :=
  rfl

@[simp] theorem allRecipes_freshCheck
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.allRecipes.recipe Call.freshCheck =
      certificate.phase34.freshCheck :=
  rfl

/-- Constructive obligation-10 Step certificate for the complete recipe
family. -/
def canonicalStep
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    CanonicalEncoding.Step parameters :=
  CanonicalEncodingRealization.step parameters certificate.baseProfile
    certificate.allRecipes certificate.defaultRunningAdmissible

/-- Constructive obligation-10 Terminal certificate for the same complete
recipe family. -/
def canonicalTerminal
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    CanonicalEncoding.Terminal parameters :=
  CanonicalEncodingRealization.terminal parameters certificate.baseProfile
    certificate.allRecipes

theorem step_obligation10
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    CanonicalEncoding.Step.Claims certificate.canonicalStep :=
  CanonicalEncodingRealization.stepObligation10 parameters
    certificate.baseProfile certificate.allRecipes
    certificate.defaultRunningAdmissible

theorem terminal_obligation10
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    CanonicalEncoding.Terminal.Claims certificate.canonicalTerminal :=
  CanonicalEncodingRealization.terminalObligation10 parameters
    certificate.baseProfile certificate.allRecipes

/-- Physical Step satisfaction reaches the exact frozen checker and binds
its result to the decoded result columns. -/
theorem step_sound
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters)
    (assignment : ColumnId -> Field)
    (input :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (physical :
      (CanonicalStepSoundness.encoding parameters certificate.baseProfile
        certificate.allRecipes certificate.defaultRunningAdmissible
      ).PhysicalSatisfies assignment)
    (inputDecoded :
      Columns.Decodes
        (certificate.baseProfile.family parameters)
        (CanonicalContexts.Step.input parameters) assignment
        (stepInputValues parameters input)) :
    ∃ output :
        Nightstream.HyperNova.Construction2.Paper.Output
          parameters.Digest parameters.State parameters.Running 1,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Accepts
          parameters input output ∧
        Columns.Decodes
          (certificate.baseProfile.family parameters)
          (CanonicalContexts.Step.result parameters) assignment
          (stepResultValues parameters output) :=
  CanonicalStepSoundness.physicalSoundAligned parameters
    certificate.baseProfile certificate.allRecipes
    certificate.defaultRunningAdmissible
    certificate.directProfile.fieldLaws assignment input physical
    inputDecoded

/-- Every admissible accepted Step has an honest satisfying assignment for
the exact complete recipe program. -/
theorem step_complete
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters)
    (input :
      CanonicalStepCompleteness.StepInputFor
        parameters)
    (output :
      CanonicalStepCompleteness.StepOutputFor
        parameters)
    (accepted :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Accepts
        parameters input output)
    (admissible :
      CanonicalStepCompleteness.AdmissibleExecution parameters
        certificate.baseProfile input
        (CanonicalStepCompleteness.selectedRunning output)) :
    ∃ assignment : ColumnId -> Field,
      (CanonicalStepSoundness.encoding parameters certificate.baseProfile
          certificate.allRecipes certificate.defaultRunningAdmissible
        ).PhysicalSatisfies assignment ∧
        Columns.Encodes
          (certificate.baseProfile.family parameters)
          (CanonicalContexts.Step.input parameters) assignment
          (stepInputValues parameters input) ∧
        Columns.Encodes
          (certificate.baseProfile.family parameters)
          (CanonicalContexts.Step.result parameters) assignment
          (stepResultValues parameters output) :=
  CanonicalStepCompleteness.physicalComplete parameters
    certificate.baseProfile certificate.allRecipes
    certificate.defaultRunningAdmissible
    certificate.directProfile.fieldLaws input output accepted admissible

/-- The complete physical Step program accepts one exact typed input/output
pair.  Both sides are bound through the canonical codecs; no digest or
acceptance bit is used as authority. -/
def StepPhysicalAccepts
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters)
    (input : CanonicalStepCompleteness.StepInputFor parameters)
    (output : CanonicalStepCompleteness.StepOutputFor parameters) : Prop :=
  ∃ assignment : ColumnId -> Field,
    (CanonicalStepSoundness.encoding parameters certificate.baseProfile
        certificate.allRecipes certificate.defaultRunningAdmissible
      ).PhysicalSatisfies assignment ∧
      Columns.Encodes
        (certificate.baseProfile.family parameters)
        (CanonicalContexts.Step.input parameters) assignment
        (stepInputValues parameters input) ∧
      Columns.Encodes
        (certificate.baseProfile.family parameters)
        (CanonicalContexts.Step.result parameters) assignment
        (stepResultValues parameters output)

/-- Exact checker correspondence for one admissible typed Step execution. -/
theorem stepPhysicalAccepts_iff
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters)
    (input : CanonicalStepCompleteness.StepInputFor parameters)
    (output : CanonicalStepCompleteness.StepOutputFor parameters)
    (admissible :
      CanonicalStepCompleteness.AdmissibleExecution parameters
        certificate.baseProfile input
        (CanonicalStepCompleteness.selectedRunning output)) :
    certificate.StepPhysicalAccepts input output ↔
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Accepts
        parameters input output := by
  constructor
  · rintro ⟨assignment, physical, inputEncoded, outputEncoded⟩
    have inputDecoded :
        Columns.Decodes
          (certificate.baseProfile.family parameters)
          (CanonicalContexts.Step.input parameters) assignment
          (stepInputValues parameters input) :=
      (CanonicalContexts.Step.input parameters
        ).toSchemaBundles.decodes_of_encodes
          (certificate.baseProfile.family parameters) assignment
          (stepInputValues parameters input) inputEncoded
    rcases certificate.step_sound assignment input physical inputDecoded with
      ⟨computed, computedAccepted, computedDecoded⟩
    have outputDecoded :
        Columns.Decodes
          (certificate.baseProfile.family parameters)
          (CanonicalContexts.Step.result parameters) assignment
          (stepResultValues parameters output) :=
      (CanonicalContexts.Step.result parameters
        ).toSchemaBundles.decodes_of_encodes
          (certificate.baseProfile.family parameters) assignment
          (stepResultValues parameters output) outputEncoded
    have valuesEqual :
        stepResultValues parameters computed =
          stepResultValues parameters output :=
      decodedValues_unique
        (certificate.baseProfile.family parameters) assignment
        (CanonicalContexts.Step.result parameters).toSchemaBundles
        computedDecoded outputDecoded
    have computedEqual : computed = output :=
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.stepResultValues_injective
        parameters valuesEqual
    simpa only [computedEqual] using computedAccepted
  · intro accepted
    exact certificate.step_complete input output accepted admissible

/-- Physical Terminal satisfaction reaches the exact frozen terminal
checker. -/
theorem terminal_sound
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters)
    (assignment : ColumnId -> Field)
    (statement :
      Nightstream.HyperNova.Construction2.Paper.TerminalStatement
        parameters.State)
    (proof :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.FixedOneTerminal.Proof
        parameters)
    (physical :
      (CanonicalTerminalSoundness.encoding parameters
        certificate.baseProfile certificate.allRecipes
      ).PhysicalSatisfies assignment)
    (inputDecoded :
      Columns.Decodes
        (certificate.baseProfile.family parameters)
        (CanonicalContexts.Terminal.input parameters) assignment
        (terminalInputValues parameters statement proof)) :
    Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.Accepts
      parameters statement proof :=
  CanonicalTerminalSoundness.physicalSound parameters
    certificate.baseProfile certificate.allRecipes
    certificate.directProfile.fieldLaws assignment statement proof physical
    inputDecoded

/-- Every admissible accepted Terminal statement has an honest satisfying
assignment for the exact complete recipe program. -/
theorem terminal_complete
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters)
    (statement :
      CanonicalTerminalCompleteness.TerminalStatementFor
        parameters)
    (proof :
      CanonicalTerminalCompleteness.TerminalProofFor
        parameters)
    (accepted :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.Accepts
        parameters statement proof)
    (admissible :
      CanonicalTerminalCompleteness.AdmissibleExecution parameters
        certificate.baseProfile statement proof) :
    ∃ assignment : ColumnId -> Field,
      (CanonicalTerminalSoundness.encoding parameters
          certificate.baseProfile certificate.allRecipes
        ).PhysicalSatisfies assignment ∧
        Columns.Encodes
          (certificate.baseProfile.family parameters)
          (CanonicalContexts.Terminal.input parameters) assignment
          (terminalInputValues parameters statement proof) :=
  CanonicalTerminalCompleteness.physicalComplete parameters
    certificate.baseProfile certificate.allRecipes
    certificate.directProfile.fieldLaws statement proof accepted admissible

/-- The complete physical Terminal program accepts one exact typed terminal
statement/proof pair. -/
def TerminalPhysicalAccepts
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters)
    (statement :
      CanonicalTerminalCompleteness.TerminalStatementFor parameters)
    (proof : CanonicalTerminalCompleteness.TerminalProofFor parameters) :
    Prop :=
  ∃ assignment : ColumnId -> Field,
    (CanonicalTerminalSoundness.encoding parameters
        certificate.baseProfile certificate.allRecipes
      ).PhysicalSatisfies assignment ∧
      Columns.Encodes
        (certificate.baseProfile.family parameters)
        (CanonicalContexts.Terminal.input parameters) assignment
        (terminalInputValues parameters statement proof)

/-- Exact checker correspondence for one admissible typed Terminal
execution. -/
theorem terminalPhysicalAccepts_iff
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters)
    (statement :
      CanonicalTerminalCompleteness.TerminalStatementFor parameters)
    (proof : CanonicalTerminalCompleteness.TerminalProofFor parameters)
    (admissible :
      CanonicalTerminalCompleteness.AdmissibleExecution parameters
        certificate.baseProfile statement proof) :
    certificate.TerminalPhysicalAccepts statement proof ↔
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.Accepts
        parameters statement proof := by
  constructor
  · rintro ⟨assignment, physical, inputEncoded⟩
    have inputDecoded :
        Columns.Decodes
          (certificate.baseProfile.family parameters)
          (CanonicalContexts.Terminal.input parameters) assignment
          (terminalInputValues parameters statement proof) :=
      (CanonicalContexts.Terminal.input parameters
        ).toSchemaBundles.decodes_of_encodes
          (certificate.baseProfile.family parameters) assignment
          (terminalInputValues parameters statement proof) inputEncoded
    exact certificate.terminal_sound assignment statement proof physical
      inputDecoded
  · intro accepted
    exact certificate.terminal_complete statement proof accepted admissible

/-- Exact Step cost computed from the complete physical receipt program. -/
def stepCost
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) : Cost :=
  certificate.canonicalStep.encoding.cost

/-- Exact Terminal cost computed from the complete physical receipt
program. -/
def terminalCost
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) : Cost :=
  certificate.canonicalTerminal.encoding.cost

theorem stepCost_eq_receiptFold
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.stepCost =
      Cost.sum
        (certificate.canonicalStep.program.physical.receipts.map
          InstructionReceipt.cost) :=
  certificate.canonicalStep.costIsReceiptFold

theorem terminalCost_eq_receiptFold
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.terminalCost =
      Cost.sum
        (certificate.canonicalTerminal.program.physical.receipts.map
          InstructionReceipt.cost) :=
  certificate.canonicalTerminal.costIsReceiptFold

/-- Exact selected Step cost: the receipt-computed fixed program plus one
mux row per running coordinate and two direct assertion rows. -/
theorem stepCost_exact
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.stepCost =
      CanonicalEncoding.stepFixedCost certificate.canonicalStep.program +
        ⟨parameters.widths.running + 2, 0, 0, 0⟩ :=
  certificate.canonicalStep.exactCost

/-- Exact selected Terminal cost: the receipt-computed fixed program plus
the four retained direct assertion rows. -/
theorem terminalCost_exact
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.terminalCost =
      CanonicalEncoding.terminalFixedCost
          certificate.canonicalTerminal.program +
        ⟨4, 0, 0, 0⟩ :=
  certificate.canonicalTerminal.exactCost

theorem step_everyColumn_has_exact_owner
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters)
    (column : ColumnId)
    (member : column ∈ certificate.canonicalStep.encoding.columnIds) :
    ∃ receipt,
      receipt ∈ certificate.canonicalStep.program.physical.receipts ∧
        receipt.owner ∈ SourceOwners.stepOwners parameters ∧
        column ∈ receipt.columnIds ∧
          ∀ candidate,
            candidate ∈
                certificate.canonicalStep.program.physical.receipts ->
            column ∈ candidate.columnIds ->
            candidate = receipt :=
  certificate.canonicalStep.everyColumnHasExactlyOneSourceOwner column member

theorem step_everyRow_has_exact_owner
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters)
    (row : RowId)
    (member : row ∈ certificate.canonicalStep.encoding.rowIds) :
    ∃ receipt,
      receipt ∈ certificate.canonicalStep.program.physical.receipts ∧
        receipt.owner ∈ SourceOwners.stepOwners parameters ∧
        row ∈ receipt.rowIds ∧
          ∀ candidate,
            candidate ∈
                certificate.canonicalStep.program.physical.receipts ->
            row ∈ candidate.rowIds ->
            candidate = receipt :=
  certificate.canonicalStep.everyRowHasExactlyOneSourceOwner row member

theorem terminal_everyColumn_has_exact_owner
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters)
    (column : ColumnId)
    (member : column ∈ certificate.canonicalTerminal.encoding.columnIds) :
    ∃ receipt,
      receipt ∈ certificate.canonicalTerminal.program.physical.receipts ∧
        receipt.owner ∈ SourceOwners.terminalOwners parameters ∧
        column ∈ receipt.columnIds ∧
          ∀ candidate,
            candidate ∈
                certificate.canonicalTerminal.program.physical.receipts ->
            column ∈ candidate.columnIds ->
            candidate = receipt :=
  certificate.canonicalTerminal.everyColumnHasExactlyOneSourceOwner
    column member

theorem terminal_everyRow_has_exact_owner
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters)
    (row : RowId)
    (member : row ∈ certificate.canonicalTerminal.encoding.rowIds) :
    ∃ receipt,
      receipt ∈ certificate.canonicalTerminal.program.physical.receipts ∧
        receipt.owner ∈ SourceOwners.terminalOwners parameters ∧
        row ∈ receipt.rowIds ∧
          ∀ candidate,
            candidate ∈
                certificate.canonicalTerminal.program.physical.receipts ->
            row ∈ candidate.rowIds ->
            candidate = receipt :=
  certificate.canonicalTerminal.everyRowHasExactlyOneSourceOwner row member

theorem step_rows_conserved
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.canonicalStep.encoding.rows =
      certificate.canonicalStep.program.physical.receipts.flatMap
        (fun receipt => receipt.rows) :=
  certificate.canonicalStep.rowsConserved

theorem step_columns_conserved
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.canonicalStep.encoding.columns =
      certificate.canonicalStep.program.physical.receipts.flatMap
        (fun receipt => receipt.allocations) :=
  certificate.canonicalStep.columnsConserved

theorem terminal_rows_conserved
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.canonicalTerminal.encoding.rows =
      certificate.canonicalTerminal.program.physical.receipts.flatMap
        (fun receipt => receipt.rows) :=
  certificate.canonicalTerminal.rowsConserved

theorem terminal_columns_conserved
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.canonicalTerminal.encoding.columns =
      certificate.canonicalTerminal.program.physical.receipts.flatMap
        (fun receipt => receipt.allocations) :=
  certificate.canonicalTerminal.columnsConserved

end CompleteApplicationCertification

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
