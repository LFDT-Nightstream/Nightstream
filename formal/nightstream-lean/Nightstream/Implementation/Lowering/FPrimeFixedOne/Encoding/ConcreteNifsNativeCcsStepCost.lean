import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationStepCostSplit
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsStep

/-!
Contract: exact cost reduction for native CCS activation in fixed-one Step.

Assurance tier: model-level.

Owns:
- the exact old-versus-native selected-NIFS receipt cost equation;
- the exact whole-Step cost equation after replacing that one receipt;
- proof that the reduction is one row and one auxiliary column per intrinsic
  selected-verifier row.

Does not own: a concrete benchmark, recursive fixed-point dimensions, JSON,
or Rust.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsStepCost

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev TranscriptState := Poseidon2Duplex.State

private theorem cost_ext
    {left right : Cost}
    (rows : left.recurringRows = right.recurringRows)
    (committed : left.committedColumns = right.committedColumns)
    (publicEq : left.publicColumns = right.publicColumns)
    (auxiliary : left.auxiliaryColumns = right.auxiliaryColumns) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem cost_add_comm (left right : Cost) :
    left + right = right + left := by
  apply cost_ext <;>
    simp only [Cost.add_recurringRows, Cost.add_committedColumns,
      Cost.add_publicColumns, Cost.add_auxiliaryColumns] <;>
    omega

private theorem cost_reorder
    (first second third : Cost) :
    (first + second) + third = (first + third) + second := by
  rw [Cost.add_assoc, cost_add_comm second third, ← Cost.add_assoc]

private theorem auxiliaryLayout_cost (width : Nat) :
    (auxiliaryLayout width).cost = ⟨0, 0, 0, width⟩ := by
  induction width with
  | zero =>
      rfl
  | succ width inductionHypothesis =>
      change
        Cost.oneColumn .auxiliaryColumn +
            (auxiliaryLayout width).cost =
          ⟨0, 0, 0, width + 1⟩
      rw [inductionHypothesis]
      apply cost_ext <;> simp [Cost.oneColumn] <;> omega

section CompleteStep

variable {shape : SemanticShape}
variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}
variable {keys : Fin 1 →
  SelectedKey shape TranscriptState publicRingColumns publicFits verifierRows}
variable {defaultRunning :
  SelectedRunning shape publicRingColumns publicFits verifierRows}
variable {machine :
  Machine
    (SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows)
    Digest AppState Witness
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    Encoded 1}
variable {terminalRelations :
  TerminalRelations
    (SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows)
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    RunningWitness
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    FreshWitness 1}
variable {terminalChecks :
  Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
    terminalRelations}
variable {widths : Widths} {footprints : Footprints}

local notation "Selected" =>
  ConcreteNifsParameters.selected keys defaultRunning machine
    terminalRelations terminalChecks widths footprints

private abbrev StepRecipeFor
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected) :=
  CallRecipe (signature Selected)
    (application.profile.family Selected) Call.step

private abbrev DefaultAdmissibleFor
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected) :=
  ((application.profile.family Selected).codecFor (.data .running)).Admissible
    defaultRunning

def overhead
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) : Cost :=
  ActivatedRawProgram.overheadCost
    (ConcreteNifsRawProgram.cost application.profile nifs.operational
      (ConcreteNifsNativeCcsStep.invokePlan
        application nifs step defaultAdmissible).frame).recurringRows

theorem targetReceipt_cost_exact
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (ConcreteNifsNativeCcsStep.targetReceipt
      application nifs step defaultAdmissible).cost =
      (signature Selected).callCost Call.nifsVerify := by
  change
    (InstructionReceipt.ofCall
      (ConcreteNifsNativeCcsStep.invokePlan
        application nifs step defaultAdmissible).recipe
      (ConcreteNifsNativeCcsStep.invokePlan
        application nifs step defaultAdmissible).frame).cost =
      (signature Selected).callCost Call.nifsVerify
  exact ApplicationStepCostSplit.callReceipt_cost_exact
    (ConcreteNifsNativeCcsStep.invokePlan
      application nifs step defaultAdmissible).recipe
    (ConcreteNifsNativeCcsStep.invokePlan
      application nifs step defaultAdmissible).frame

theorem nativeReceipt_cost_exact
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (ConcreteNifsNativeCcsStep.nativeReceipt
      application nifs step defaultAdmissible).cost =
      ((signature Selected).callOutputs Call.nifsVerify).cost +
        ConcreteNifsRawProgram.cost application.profile nifs.operational
          (ConcreteNifsNativeCcsStep.invokePlan
            application nifs step defaultAdmissible).frame :=
  ConcreteNifsNativeCcsProgram.selectedReceipt_cost_exact
    application.profile nifs
    (ConcreteNifsNativeCcsStep.invokePlan
      application nifs step defaultAdmissible).frame

/-- The old selected receipt is exactly the native receipt plus the removed
activation overhead. -/
theorem target_cost_eq_native_add_overhead
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (ConcreteNifsNativeCcsStep.targetReceipt
        application nifs step defaultAdmissible).cost =
      (ConcreteNifsNativeCcsStep.nativeReceipt
          application nifs step defaultAdmissible).cost +
        overhead application nifs step defaultAdmissible := by
  rw [targetReceipt_cost_exact, nativeReceipt_cost_exact]
  unfold Signature.callCost
  have footprintExact :=
    ConcreteNifsActivatedProgram.selected_footprint_exact
      application.profile nifs.operational nifs.footprint
      (ConcreteNifsNativeCcsStep.invokePlan
        application nifs step defaultAdmissible).frame
  rw [footprintExact]
  unfold ConcreteNifsActivatedProgram.footprint
    CallFootprint.cost ConcreteNifsActivatedProgram.cost
    ConcreteNifsActivatedProgram.intrinsicCost overhead
    ActivatedRawProgram.cost
  simp only [List.map_cons, List.map_nil, Cost.sum]
  rw [auxiliaryLayout_cost]
  apply cost_ext <;>
    simp only [Cost.add_recurringRows, Cost.add_committedColumns,
      Cost.add_publicColumns, Cost.add_auxiliaryColumns,
      Cost.zero, ActivatedRawProgram.overheadCost,
      ConcreteNifsRawProgram.cost,
      ConcreteNifsRawProgram.claimedValueCost,
      ConcreteNifsProofCanonicalityRows.cost,
      ConcreteNifsRunningAuthorityRows.cost,
      ConcreteNifsOperationalSampler.cost,
      KSplitNcOperationalRows.cost,
      KSplitNcOperationalRows.endpointCost,
      KSplitNcTranscript.cost,
      SymbolicDuplex.cost,
      KSplitNcBlockLaneRows.cost,
      KSplitNcFeRows.cost,
      KSplitNcNcRows.cost,
      KFixedPhaseSumCheck.chainCost,
      PiRlcCanonicalSamplerProgram.cost,
      ConcreteNifsOperationalSampler.challengeCost,
      ConcreteNifsPiRlcPointRows.cost,
      ConcreteNifsPiRlcActionRows.cost,
      ConcreteNifsPiDecRows.cost,
      ConcreteNifsOutputRows.cost] <;>
    omega

private theorem replace_cost_eq_of_no_target
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (receipts : List InstructionReceipt)
    (noTarget :
      ∀ receipt, receipt ∈ receipts →
        receipt.owner ≠ ConcreteNifsNativeCcsStep.targetOwner) :
    Cost.sum
        ((receipts.map
          (ConcreteNifsNativeCcsStep.replace
            application nifs step defaultAdmissible)).map
              SelectedReceipt.cost) =
      Cost.sum (receipts.map InstructionReceipt.cost) := by
  induction receipts with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      have headOther :=
        noTarget head List.mem_cons_self
      have tailOther :
          ∀ receipt, receipt ∈ tail →
            receipt.owner ≠ ConcreteNifsNativeCcsStep.targetOwner := by
        intro receipt member
        exact noTarget receipt (List.mem_cons_of_mem head member)
      simp only [List.map_cons, Cost.sum,
        ConcreteNifsNativeCcsStep.replace_other
          application nifs step defaultAdmissible head headOther,
        SelectedReceipt.cost]
      rw [inductionHypothesis tailOther]

private theorem source_cost_eq_selected_add_overhead
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (receipts : List InstructionReceipt)
    (targetMember :
      ConcreteNifsNativeCcsStep.targetReceipt
        application nifs step defaultAdmissible ∈ receipts)
    (ownersNodup :
      (receipts.map fun receipt => receipt.owner).Nodup) :
    Cost.sum (receipts.map InstructionReceipt.cost) =
      Cost.sum
          ((receipts.map
            (ConcreteNifsNativeCcsStep.replace
              application nifs step defaultAdmissible)).map
                SelectedReceipt.cost) +
        overhead application nifs step defaultAdmissible := by
  induction receipts with
  | nil =>
      simp at targetMember
  | cons head tail inductionHypothesis =>
      have splitOwners :
          head.owner ∉ tail.map (fun receipt => receipt.owner) ∧
            (tail.map fun receipt => receipt.owner).Nodup := by
        simpa only [List.map_cons, List.nodup_cons] using ownersNodup
      rcases List.mem_cons.1 targetMember with headTarget | tailTarget
      · subst head
        have noTailTarget :
            ∀ receipt, receipt ∈ tail →
              receipt.owner ≠ ConcreteNifsNativeCcsStep.targetOwner := by
          intro receipt member equal
          apply splitOwners.1
          exact List.mem_map.mpr
            ⟨receipt, member, equal⟩
        simp only [List.map_cons, Cost.sum,
          ConcreteNifsNativeCcsStep.replace_target]
        rw [target_cost_eq_native_add_overhead,
          replace_cost_eq_of_no_target application nifs step
            defaultAdmissible tail noTailTarget]
        exact cost_reorder
          (ConcreteNifsNativeCcsStep.nativeReceipt
            application nifs step defaultAdmissible).cost
          (overhead application nifs step defaultAdmissible)
          (Cost.sum (tail.map InstructionReceipt.cost))
      · have headOther :
            head.owner ≠ ConcreteNifsNativeCcsStep.targetOwner := by
          intro equal
          apply splitOwners.1
          exact List.mem_map.mpr
            ⟨ConcreteNifsNativeCcsStep.targetReceipt
                application nifs step defaultAdmissible,
              tailTarget,
              by
                rw [ConcreteNifsNativeCcsStep.targetReceipt_owner]
                exact equal.symm⟩
        simp only [List.map_cons, Cost.sum,
          ConcreteNifsNativeCcsStep.replace_other
            application nifs step defaultAdmissible head headOther,
          SelectedReceipt.cost]
        rw [inductionHypothesis tailTarget splitOwners.2,
          Cost.add_assoc]

/-- The complete legacy Step cost is exactly the native Step cost plus the
one-for-one activation overhead. -/
theorem sourceCost_eq_nativeCost_add_overhead
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (ConcreteNifsNativeCcsStep.certificate
        application nifs step defaultAdmissible).stepCost =
      (ConcreteNifsNativeCcsStep.program
          application nifs step defaultAdmissible).cost +
        overhead application nifs step defaultAdmissible := by
  rw [(ConcreteNifsNativeCcsStep.certificate
    application nifs step defaultAdmissible).stepCost_eq_receiptFold]
  change
    Cost.sum
        ((ConcreteNifsNativeCcsStep.sourceReceipts
          application nifs step defaultAdmissible).map
            InstructionReceipt.cost) =
      Cost.sum
          ((ConcreteNifsNativeCcsStep.selectedReceipts
            application nifs step defaultAdmissible).map
              SelectedReceipt.cost) +
        overhead application nifs step defaultAdmissible
  exact
    source_cost_eq_selected_add_overhead
      application nifs step defaultAdmissible
      (ConcreteNifsNativeCcsStep.sourceReceipts
        application nifs step defaultAdmissible)
      (ConcreteNifsNativeCcsStep.targetReceipt_member
        application nifs step defaultAdmissible)
      (ConcreteNifsNativeCcsStep.certificate
        application nifs step defaultAdmissible
        ).canonicalStep.program.physical.ownersNodup

end CompleteStep

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsStepCost
