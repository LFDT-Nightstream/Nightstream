import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalNormalFormSites

/-!
Contract: constructive realization of the canonical fixed-one Step and
Terminal obligation-10 certificates.

Owns:
- exact finite-class specifications at every selected rewrite site;
- exact receipt witnesses for the Step mux, all retained assertions, and the
  mandatory empty Terminal join;
- inhabited canonical Step and Terminal certificates;
- unconditional obligation-10 claims for supplied canonical call recipes.

Does not own: semantic soundness or honest assignment construction; those are
proved downstream for this exact realization by
`Canonical{Step,Terminal}Soundness` and
`Canonical{Step,Terminal}PhysicalCompleteness`.  It also does not own the
still-open production codec/recipe instantiation, Rust emission, generated
artifact equality, or source-to-R1CS refinement.

Emits constraints: no; every selected receipt comes from the already
conserved physical programs.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.PrimitiveNormalForm
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

namespace CanonicalEncodingRealization

/-! ## Step finite sites -/

def stepJoinRecipe
    (parameters : Parameters)
    (profile : Profile parameters) :
    MuxRecipe (Ports.committedRunning parameters).layout :=
  CanonicalBranchPlan.onePortJoinRecipe
    SourceOwners.stepBranchPath
    (CanonicalContexts.Step.selector parameters profile)
    (Ports.committedRunning parameters)
    (CanonicalContexts.Step.baseRunning parameters)
    (CanonicalContexts.Step.recursiveRunning parameters)

def stepBaseAssertionRecipe
    (parameters : Parameters)
    (profile : Profile parameters) :
    BoolAssertRecipe :=
  (CanonicalPrimitivePlan.assertion profile
    (.here (Ports.auxiliaryBit parameters))
    SourceOwners.stepBaseAssertionPath
    (CanonicalContexts.Step.afterBaseEquality parameters)
    oneColumn
    (activationColumn SourceOwners.stepBranchPath true)
    (CanonicalContexts.Step.afterBaseEqualityWidths
      parameters profile)).recipe

def stepRecursiveAssertionRecipe
    (parameters : Parameters)
    (profile : Profile parameters) :
    BoolAssertRecipe :=
  (CanonicalPrimitivePlan.assertion profile
    (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.encodedEqual
      parameters)
    SourceOwners.stepRecursiveAssertionPath
    (CanonicalContexts.Step.afterEncodedEquality parameters)
    oneColumn
    (activationColumn SourceOwners.stepBranchPath false)
    (CanonicalContexts.Step.afterEncodedEqualityWidths
      parameters profile)).recipe

def stepSpecifications
    (parameters : Parameters)
    (profile : Profile parameters) :
    NormalForm.StepSpecifications where
  joinCoordinates :=
    CanonicalNormalFormSites.joinSpecifications
      (stepJoinRecipe parameters profile)
  baseEndpoint :=
    CanonicalNormalFormSites.assertionSpecification
      (stepBaseAssertionRecipe parameters profile)
  recursivePriorLink :=
    CanonicalNormalFormSites.assertionSpecification
      (stepRecursiveAssertionRecipe parameters profile)

theorem stepSitesAligned
    (parameters : Parameters)
    (profile : Profile parameters) :
    SourceOwners.StepNormalFormAligned parameters
      (stepSpecifications parameters profile) where
  joinCoordinatesExact := by
    rw [show
      (stepSpecifications parameters profile).joinCoordinates =
          CanonicalNormalFormSites.joinSpecifications
            (stepJoinRecipe parameters profile) by rfl]
    rw [CanonicalNormalFormSites.joinOwnerOrdinalsExact]
    simp [stepJoinRecipe,
      CanonicalBranchPlan.onePortJoinRecipe,
      bundleOwnedColumns,
      Ports.committedRunning, dataPort, committedLayout, ownedLayout,
      List.range_eq_range']
  baseEndpointOwner := rfl
  baseEndpointOrdinal := rfl
  recursivePriorLinkOwner := rfl
  recursivePriorLinkOrdinal := rfl

theorem stepJoinReceipt_mem
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    CanonicalBranchPlan.onePortJoinReceipt
        SourceOwners.stepBranchPath
        (CanonicalContexts.Step.selector parameters profile)
        (Ports.committedRunning parameters)
        (CanonicalContexts.Step.baseRunning parameters)
        (CanonicalContexts.Step.recursiveRunning parameters) ∈
      (CanonicalStepPlan.aligned parameters profile recipes
        defaultAdmissible).physical.receipts := by
  simp [CanonicalStepPlan.aligned, CanonicalStepPlan.physical,
    CanonicalStepPlan.receipts, CanonicalStepPlan.bodyReceipts]

theorem stepBaseAssertionReceipt_mem
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    (CanonicalStepPlan.baseAssertionPlan.{0}
        parameters profile).receipt ∈
      (CanonicalStepPlan.aligned parameters profile recipes
        defaultAdmissible).physical.receipts := by
  simp [CanonicalStepPlan.aligned, CanonicalStepPlan.physical,
    CanonicalStepPlan.receipts, CanonicalStepPlan.bodyReceipts]

theorem stepRecursiveAssertionReceipt_mem
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    (CanonicalStepPlan.recursiveAssertionPlan.{0}
        parameters profile).receipt ∈
      (CanonicalStepPlan.aligned parameters profile recipes
        defaultAdmissible).physical.receipts := by
  simp [CanonicalStepPlan.aligned, CanonicalStepPlan.physical,
    CanonicalStepPlan.receipts, CanonicalStepPlan.bodyReceipts]

def stepBranchJoinExact
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    CanonicalEncoding.ExactReceipt
      (CanonicalStepPlan.aligned parameters profile recipes
        defaultAdmissible)
      (.typed (.branch SourceOwners.stepBranchPath))
      .branchJoin
      (schemaOwnedColumns
        (branchJoinColumns SourceOwners.stepBranchPath
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.joinedSchema
            parameters)))
      ((stepSpecifications parameters profile).joinCoordinates.flatMap
        (fun specification =>
          BranchJoin.Candidate.rows .selectedMux specification)) where
  receipt :=
    CanonicalBranchPlan.onePortJoinReceipt
      SourceOwners.stepBranchPath
      (CanonicalContexts.Step.selector parameters profile)
      (Ports.committedRunning parameters)
      (CanonicalContexts.Step.baseRunning parameters)
      (CanonicalContexts.Step.recursiveRunning parameters)
  member :=
    stepJoinReceipt_mem parameters profile recipes defaultAdmissible
  ownerExact := rfl
  kindExact := rfl
  allocationsExact := by
    simpa [CanonicalBranchPlan.onePortJoinReceipt,
      InstructionReceipt.ofMux,
      CanonicalBranchPlan.onePortJoinRecipe,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.joinedSchema]
      using
        (CanonicalNormalFormSites.singletonSchemaOwnedColumns
          (branchJoinColumns SourceOwners.stepBranchPath
            [Ports.committedRunning parameters])).symm
  rowsExact := by
    simpa [stepJoinRecipe, stepSpecifications,
      CanonicalBranchPlan.onePortJoinReceipt,
      InstructionReceipt.ofMux] using
      CanonicalNormalFormSites.joinRowsExact
        (stepJoinRecipe parameters profile)

def stepBaseAssertionExact
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    CanonicalEncoding.ExactReceipt
      (CanonicalStepPlan.aligned parameters profile recipes
        defaultAdmissible)
      (.typed (.instruction SourceOwners.stepBaseAssertionPath))
      .assertion
      []
      (GatedAssertion.Candidate.rows .direct
        (stepSpecifications parameters profile).baseEndpoint) where
  receipt :=
    (CanonicalStepPlan.baseAssertionPlan.{0}
      parameters profile).receipt
  member :=
    stepBaseAssertionReceipt_mem
      parameters profile recipes defaultAdmissible
  ownerExact := rfl
  kindExact := rfl
  allocationsExact := rfl
  rowsExact := rfl

def stepRecursiveAssertionExact
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    CanonicalEncoding.ExactReceipt
      (CanonicalStepPlan.aligned parameters profile recipes
        defaultAdmissible)
      (.typed (.instruction SourceOwners.stepRecursiveAssertionPath))
      .assertion
      []
      (GatedAssertion.Candidate.rows .direct
        (stepSpecifications parameters profile).recursivePriorLink) where
  receipt :=
    (CanonicalStepPlan.recursiveAssertionPlan.{0}
      parameters profile).receipt
  member :=
    stepRecursiveAssertionReceipt_mem
      parameters profile recipes defaultAdmissible
  ownerExact := rfl
  kindExact := rfl
  allocationsExact := rfl
  rowsExact := rfl

/-- Constructive canonical Step certificate for the exact recipe-selected
physical program. -/
def step
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    CanonicalEncoding.Step parameters where
  profile := profile
  specifications := stepSpecifications parameters profile
  sitesAligned := stepSitesAligned parameters profile
  program :=
    CanonicalStepPlan.aligned parameters profile recipes
      defaultAdmissible
  branchJoin :=
    stepBranchJoinExact parameters profile recipes defaultAdmissible
  baseEndpoint :=
    stepBaseAssertionExact parameters profile recipes defaultAdmissible
  recursivePriorLink :=
    stepRecursiveAssertionExact
      parameters profile recipes defaultAdmissible

/-! ## Terminal finite sites -/

def terminalBaseAssertionRecipe
    (parameters : Parameters)
    (profile : Profile parameters) :
    BoolAssertRecipe :=
  (CanonicalPrimitivePlan.assertion profile
    (.here (Ports.auxiliaryBit parameters))
    SourceOwners.terminalBaseAssertionPath
    (CanonicalContexts.Terminal.afterBaseEquality parameters)
    oneColumn
    (activationColumn SourceOwners.terminalBranchPath true)
    (CanonicalContexts.Terminal.afterBaseEqualityWidths
      parameters profile)).recipe

def terminalPriorAssertionRecipe
    (parameters : Parameters)
    (profile : Profile parameters) :
    BoolAssertRecipe :=
  (CanonicalPrimitivePlan.assertion profile
    (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.encodedEqual
      parameters)
    SourceOwners.terminalRecursivePriorAssertionPath
    (CanonicalContexts.Terminal.afterEncodedEquality parameters)
    oneColumn
    (activationColumn SourceOwners.terminalBranchPath false)
    (CanonicalContexts.Terminal.afterEncodedEqualityWidths
      parameters profile)).recipe

def terminalRunningAssertionRecipe
    (parameters : Parameters)
    (profile : Profile parameters) :
    BoolAssertRecipe :=
  (CanonicalPrimitivePlan.assertion profile
    (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.runningAccepted
      parameters)
    SourceOwners.terminalRecursiveRunningAssertionPath
    (CanonicalContexts.Terminal.afterRunningCheck parameters)
    oneColumn
    (activationColumn SourceOwners.terminalBranchPath false)
    (CanonicalContexts.Terminal.afterRunningCheckWidths
      parameters profile)).recipe

def terminalFreshAssertionRecipe
    (parameters : Parameters)
    (profile : Profile parameters) :
    BoolAssertRecipe :=
  (CanonicalPrimitivePlan.assertion profile
    (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.freshAccepted
      parameters)
    SourceOwners.terminalRecursiveFreshAssertionPath
    (CanonicalContexts.Terminal.afterFreshCheck parameters)
    oneColumn
    (activationColumn SourceOwners.terminalBranchPath false)
    (CanonicalContexts.Terminal.afterFreshCheckWidths
      parameters profile)).recipe

def terminalSpecifications
    (parameters : Parameters)
    (profile : Profile parameters) :
    NormalForm.TerminalSpecifications where
  baseEndpoint :=
    CanonicalNormalFormSites.assertionSpecification
      (terminalBaseAssertionRecipe parameters profile)
  recursivePriorLink :=
    CanonicalNormalFormSites.assertionSpecification
      (terminalPriorAssertionRecipe parameters profile)
  runningRelation :=
    CanonicalNormalFormSites.assertionSpecification
      (terminalRunningAssertionRecipe parameters profile)
  freshRelation :=
    CanonicalNormalFormSites.assertionSpecification
      (terminalFreshAssertionRecipe parameters profile)

theorem terminalSitesAligned
    (parameters : Parameters)
    (profile : Profile parameters) :
    SourceOwners.TerminalNormalFormAligned
      (terminalSpecifications parameters profile) where
  baseEndpointOwner := rfl
  baseEndpointOrdinal := rfl
  recursivePriorLinkOwner := rfl
  recursivePriorLinkOrdinal := rfl
  runningRelationOwner := rfl
  runningRelationOrdinal := rfl
  freshRelationOwner := rfl
  freshRelationOrdinal := rfl

theorem terminalBaseAssertionReceipt_mem
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    (CanonicalTerminalPlan.baseAssertionPlan.{0}
        parameters profile).receipt ∈
      (CanonicalTerminalPlan.aligned
        parameters profile recipes).physical.receipts := by
  simp [CanonicalTerminalPlan.aligned, CanonicalTerminalPlan.physical,
    CanonicalTerminalPlan.receipts,
    CanonicalTerminalPlan.bodyReceipts]

theorem terminalPriorAssertionReceipt_mem
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    (CanonicalTerminalPlan.recursivePriorAssertionPlan.{0}
        parameters profile).receipt ∈
      (CanonicalTerminalPlan.aligned
        parameters profile recipes).physical.receipts := by
  simp [CanonicalTerminalPlan.aligned, CanonicalTerminalPlan.physical,
    CanonicalTerminalPlan.receipts,
    CanonicalTerminalPlan.bodyReceipts]

theorem terminalRunningAssertionReceipt_mem
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    (CanonicalTerminalPlan.recursiveRunningAssertionPlan.{0}
        parameters profile).receipt ∈
      (CanonicalTerminalPlan.aligned
        parameters profile recipes).physical.receipts := by
  simp [CanonicalTerminalPlan.aligned, CanonicalTerminalPlan.physical,
    CanonicalTerminalPlan.receipts,
    CanonicalTerminalPlan.bodyReceipts]

theorem terminalFreshAssertionReceipt_mem
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    (CanonicalTerminalPlan.recursiveFreshAssertionPlan.{0}
        parameters profile).receipt ∈
      (CanonicalTerminalPlan.aligned
        parameters profile recipes).physical.receipts := by
  simp [CanonicalTerminalPlan.aligned, CanonicalTerminalPlan.physical,
    CanonicalTerminalPlan.receipts,
    CanonicalTerminalPlan.bodyReceipts]

theorem terminalEmptyJoinReceipt_mem
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    CanonicalBranchPlan.emptyJoinReceipt
        SourceOwners.terminalBranchPath ∈
      (CanonicalTerminalPlan.aligned
        parameters profile recipes).physical.receipts := by
  simp [CanonicalTerminalPlan.aligned, CanonicalTerminalPlan.physical,
    CanonicalTerminalPlan.receipts,
    CanonicalTerminalPlan.bodyReceipts]

def terminalEmptyJoinExact
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    CanonicalEncoding.ExactReceipt
      (CanonicalTerminalPlan.aligned parameters profile recipes)
      (.typed (.branch SourceOwners.terminalBranchPath))
      .branchJoin [] [] where
  receipt :=
    CanonicalBranchPlan.emptyJoinReceipt
      SourceOwners.terminalBranchPath
  member :=
    terminalEmptyJoinReceipt_mem parameters profile recipes
  ownerExact := rfl
  kindExact := rfl
  allocationsExact := rfl
  rowsExact := rfl

def terminalBaseAssertionExact
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    CanonicalEncoding.ExactReceipt
      (CanonicalTerminalPlan.aligned parameters profile recipes)
      (.typed (.instruction SourceOwners.terminalBaseAssertionPath))
      .assertion []
      (GatedAssertion.Candidate.rows .direct
        (terminalSpecifications parameters profile).baseEndpoint) where
  receipt :=
    (CanonicalTerminalPlan.baseAssertionPlan.{0}
      parameters profile).receipt
  member :=
    terminalBaseAssertionReceipt_mem parameters profile recipes
  ownerExact := rfl
  kindExact := rfl
  allocationsExact := rfl
  rowsExact := rfl

def terminalPriorAssertionExact
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    CanonicalEncoding.ExactReceipt
      (CanonicalTerminalPlan.aligned parameters profile recipes)
      (.typed (.instruction
        SourceOwners.terminalRecursivePriorAssertionPath))
      .assertion []
      (GatedAssertion.Candidate.rows .direct
        (terminalSpecifications parameters profile).recursivePriorLink) where
  receipt :=
    (CanonicalTerminalPlan.recursivePriorAssertionPlan.{0}
      parameters profile).receipt
  member :=
    terminalPriorAssertionReceipt_mem parameters profile recipes
  ownerExact := rfl
  kindExact := rfl
  allocationsExact := rfl
  rowsExact := rfl

def terminalRunningAssertionExact
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    CanonicalEncoding.ExactReceipt
      (CanonicalTerminalPlan.aligned parameters profile recipes)
      (.typed (.instruction
        SourceOwners.terminalRecursiveRunningAssertionPath))
      .assertion []
      (GatedAssertion.Candidate.rows .direct
        (terminalSpecifications parameters profile).runningRelation) where
  receipt :=
    (CanonicalTerminalPlan.recursiveRunningAssertionPlan.{0}
      parameters profile).receipt
  member :=
    terminalRunningAssertionReceipt_mem parameters profile recipes
  ownerExact := rfl
  kindExact := rfl
  allocationsExact := rfl
  rowsExact := rfl

def terminalFreshAssertionExact
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    CanonicalEncoding.ExactReceipt
      (CanonicalTerminalPlan.aligned parameters profile recipes)
      (.typed (.instruction
        SourceOwners.terminalRecursiveFreshAssertionPath))
      .assertion []
      (GatedAssertion.Candidate.rows .direct
        (terminalSpecifications parameters profile).freshRelation) where
  receipt :=
    (CanonicalTerminalPlan.recursiveFreshAssertionPlan.{0}
      parameters profile).receipt
  member :=
    terminalFreshAssertionReceipt_mem parameters profile recipes
  ownerExact := rfl
  kindExact := rfl
  allocationsExact := rfl
  rowsExact := rfl

/-- Constructive canonical Terminal certificate for the exact
recipe-selected physical program. -/
def terminal
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    CanonicalEncoding.Terminal parameters where
  profile := profile
  specifications := terminalSpecifications parameters profile
  sitesAligned := terminalSitesAligned parameters profile
  program := CanonicalTerminalPlan.aligned parameters profile recipes
  emptyJoin := terminalEmptyJoinExact parameters profile recipes
  baseEndpoint := terminalBaseAssertionExact parameters profile recipes
  recursivePriorLink :=
    terminalPriorAssertionExact parameters profile recipes
  runningRelation :=
    terminalRunningAssertionExact parameters profile recipes
  freshRelation :=
    terminalFreshAssertionExact parameters profile recipes

/-! ## Realized obligation-10 claims -/

theorem stepObligation10
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    CanonicalEncoding.Step.Claims
      (step parameters profile recipes defaultAdmissible) :=
  CanonicalEncoding.Step.obligation10_of_certificate
    (step parameters profile recipes defaultAdmissible)

theorem terminalObligation10
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    CanonicalEncoding.Terminal.Claims
      (terminal parameters profile recipes) :=
  CanonicalEncoding.Terminal.obligation10_of_certificate
    (terminal parameters profile recipes)

end CanonicalEncodingRealization

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
