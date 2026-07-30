import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalContexts
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ReceiptScoping

/-!
Contract: canonical local primitive plans and exact receipt order for the
fixed-one Step program.

Owns:
- one constructive physical plan for every Step primitive occurrence;
- exact parent, true-arm, false-arm, join, and continuation controls;
- the ordered Step receipt list after the canonical prelude and inputs.

Does not own: whole-program receipt conservation, normal-form minimality,
row satisfaction, Rust emission, generated artifacts, or terminal lowering.

The only semantic inputs are the closed call recipes and admissibility of the
verifier-owned default running literal.  All columns and row identities are
derived from typed contexts and structural paths.

Emits constraints: the exact receipts selected by the Step source program.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

namespace CanonicalStepPlan

theorem one_excludes_instruction (path : OwnerPath) :
    oneColumn.owner ≠ .typed (.instruction path) := by
  simp [oneColumn]

theorem activation_excludes_instruction
    (branchPath target : OwnerPath)
    (selected : Bool) :
    (activationColumn branchPath selected).owner ≠
      .typed (.instruction target) := by
  simp [activationColumn]

theorem afterStep_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (applyDifferent : SourceOwners.stepApplyPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Step.afterStep parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.stepApplyPath target
      [Ports.committedState parameters] applyDifferent)
    (CanonicalPrimitivePlan.ContextExcludesOwner.input
      (stepInputSchema parameters) target)

theorem common_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (selectorDifferent : SourceOwners.stepSelectorPath ≠ target)
    (applyDifferent : SourceOwners.stepApplyPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Step.common parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.stepSelectorPath target
      [Ports.auxiliaryBit parameters] selectorDifferent)
    (afterStep_excludes parameters target applyDifferent)

theorem afterBaseEquality_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (baseEqualityDifferent :
      SourceOwners.stepBaseStateEqualPath ≠ target)
    (selectorDifferent : SourceOwners.stepSelectorPath ≠ target)
    (applyDifferent : SourceOwners.stepApplyPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Step.afterBaseEquality parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.stepBaseStateEqualPath target
      [Ports.auxiliaryBit parameters] baseEqualityDifferent)
    (common_excludes parameters target selectorDifferent applyDifferent)

theorem afterHash_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (hashDifferent : SourceOwners.stepRecursiveHashPriorPath ≠ target)
    (selectorDifferent : SourceOwners.stepSelectorPath ≠ target)
    (applyDifferent : SourceOwners.stepApplyPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Step.afterHash parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.stepRecursiveHashPriorPath target
      [Ports.auxiliaryDigest parameters] hashDifferent)
    (common_excludes parameters target selectorDifferent applyDifferent)

theorem afterFreshPublic_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (freshDifferent : SourceOwners.stepRecursiveFreshPublicPath ≠ target)
    (hashDifferent : SourceOwners.stepRecursiveHashPriorPath ≠ target)
    (selectorDifferent : SourceOwners.stepSelectorPath ≠ target)
    (applyDifferent : SourceOwners.stepApplyPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Step.afterFreshPublic parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.stepRecursiveFreshPublicPath target
      [Ports.auxiliaryEncoded parameters] freshDifferent)
    (afterHash_excludes parameters target hashDifferent
      selectorDifferent applyDifferent)

theorem afterEncode_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (encodeDifferent : SourceOwners.stepRecursiveEncodePath ≠ target)
    (freshDifferent : SourceOwners.stepRecursiveFreshPublicPath ≠ target)
    (hashDifferent : SourceOwners.stepRecursiveHashPriorPath ≠ target)
    (selectorDifferent : SourceOwners.stepSelectorPath ≠ target)
    (applyDifferent : SourceOwners.stepApplyPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Step.afterEncode parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.stepRecursiveEncodePath target
      [Ports.auxiliaryEncoded parameters] encodeDifferent)
    (afterFreshPublic_excludes parameters target freshDifferent
      hashDifferent selectorDifferent applyDifferent)

theorem afterEncodedEquality_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (equalityDifferent :
      SourceOwners.stepRecursiveEncodedEqualPath ≠ target)
    (encodeDifferent : SourceOwners.stepRecursiveEncodePath ≠ target)
    (freshDifferent : SourceOwners.stepRecursiveFreshPublicPath ≠ target)
    (hashDifferent : SourceOwners.stepRecursiveHashPriorPath ≠ target)
    (selectorDifferent : SourceOwners.stepSelectorPath ≠ target)
    (applyDifferent : SourceOwners.stepApplyPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Step.afterEncodedEquality parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.stepRecursiveEncodedEqualPath target
      [Ports.auxiliaryBit parameters] equalityDifferent)
    (afterEncode_excludes parameters target encodeDifferent
      freshDifferent hashDifferent selectorDifferent applyDifferent)

theorem continuationInput_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (selectorDifferent : SourceOwners.stepSelectorPath ≠ target)
    (applyDifferent : SourceOwners.stepApplyPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Step.continuationInput parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.branch
      SourceOwners.stepBranchPath target
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.joinedSchema
        parameters))
    (common_excludes parameters target selectorDifferent applyDifferent)

def applyPlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    PrimitivePlan parameters profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.stepCall
        parameters)
      SourceOwners.stepApplyPath
      (CanonicalContexts.Step.input parameters)
      oneColumn oneColumn :=
  .invoke
    (CanonicalPrimitivePlan.invoke profile recipes
      .step
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.zi
          parameters)
        (.cons
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.witness
            parameters)
          .nil))
      SourceOwners.stepApplyPath
      (CanonicalContexts.Step.input parameters)
      oneColumn oneColumn
      (CanonicalContexts.Step.inputWidths parameters profile)
      (one_excludes_instruction SourceOwners.stepApplyPath)
      (one_excludes_instruction SourceOwners.stepApplyPath)
      (CanonicalPrimitivePlan.ContextExcludesOwner.input
        (stepInputSchema parameters) SourceOwners.stepApplyPath))

def selectorPlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    PrimitivePlan parameters profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.iterationZeroCall
        parameters)
      SourceOwners.stepSelectorPath
      (CanonicalContexts.Step.afterStep parameters)
      oneColumn oneColumn :=
  .invoke
    (CanonicalPrimitivePlan.invoke profile recipes
      .iterationZero
      (.cons
        (.there
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.iteration
            parameters))
        .nil)
      SourceOwners.stepSelectorPath
      (CanonicalContexts.Step.afterStep parameters)
      oneColumn oneColumn
      (CanonicalContexts.Step.afterStepWidths parameters profile)
      (one_excludes_instruction SourceOwners.stepSelectorPath)
      (one_excludes_instruction SourceOwners.stepSelectorPath)
      (afterStep_excludes parameters SourceOwners.stepSelectorPath
        (by decide)))

def baseEqualityPlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    PrimitivePlan parameters profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.baseStateEqualCall
        parameters)
      SourceOwners.stepBaseStateEqualPath
      (CanonicalContexts.Step.common parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath true) :=
  .invoke
    (CanonicalPrimitivePlan.invoke profile recipes
      .stateEqual
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.z0
          parameters)
        (.cons
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.zi
            parameters)
          .nil))
      SourceOwners.stepBaseStateEqualPath
      (CanonicalContexts.Step.common parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath true)
      (CanonicalContexts.Step.commonWidths parameters profile)
      (one_excludes_instruction SourceOwners.stepBaseStateEqualPath)
      (activation_excludes_instruction SourceOwners.stepBranchPath
        SourceOwners.stepBaseStateEqualPath true)
      (common_excludes parameters SourceOwners.stepBaseStateEqualPath
        (by decide) (by decide)))

def baseAssertionPlan
    (parameters : Parameters)
    (profile : Profile parameters) :
    PrimitivePlan parameters profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.baseAssertion
        parameters)
      SourceOwners.stepBaseAssertionPath
      (CanonicalContexts.Step.afterBaseEquality parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath true) :=
  .assertTrue
    (CanonicalPrimitivePlan.assertion profile
      (.here (Ports.auxiliaryBit parameters))
      SourceOwners.stepBaseAssertionPath
      (CanonicalContexts.Step.afterBaseEquality parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath true)
      (CanonicalContexts.Step.afterBaseEqualityWidths parameters profile))

def baseLiteralPlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (admissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    PrimitivePlan parameters profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.baseDefaultCall
        parameters)
      SourceOwners.stepBaseDefaultPath
      (CanonicalContexts.Step.afterBaseEquality parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath true) :=
  .literal
    (CanonicalPrimitivePlan.literal profile
      (Ports.committedRunning parameters)
      (defaultRunning parameters)
      SourceOwners.stepBaseDefaultPath
      (CanonicalContexts.Step.afterBaseEquality parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath true)
      (profile.committedRunning_widthsAgree parameters)
      admissible)

def recursiveHashPlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    PrimitivePlan parameters profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.hashPriorCall
        parameters)
      SourceOwners.stepRecursiveHashPriorPath
      (CanonicalContexts.Step.common parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath false) :=
  .invoke
    (CanonicalPrimitivePlan.invoke profile recipes
      .hashPrior
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.iteration
          parameters)
        (.cons
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.z0
            parameters)
          (.cons
            (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.zi
              parameters)
            (.cons
              (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.running
                parameters)
              .nil))))
      SourceOwners.stepRecursiveHashPriorPath
      (CanonicalContexts.Step.common parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath false)
      (CanonicalContexts.Step.commonWidths parameters profile)
      (one_excludes_instruction SourceOwners.stepRecursiveHashPriorPath)
      (activation_excludes_instruction SourceOwners.stepBranchPath
        SourceOwners.stepRecursiveHashPriorPath false)
      (common_excludes parameters SourceOwners.stepRecursiveHashPriorPath
        (by decide) (by decide)))

def recursiveFreshPublicPlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    PrimitivePlan parameters profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.freshPublicCall
        parameters)
      SourceOwners.stepRecursiveFreshPublicPath
      (CanonicalContexts.Step.afterHash parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath false) :=
  .invoke
    (CanonicalPrimitivePlan.invoke profile recipes
      .freshPublic
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.freshAfterHash
          parameters)
        .nil)
      SourceOwners.stepRecursiveFreshPublicPath
      (CanonicalContexts.Step.afterHash parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath false)
      (CanonicalContexts.Step.afterHashWidths parameters profile)
      (one_excludes_instruction SourceOwners.stepRecursiveFreshPublicPath)
      (activation_excludes_instruction SourceOwners.stepBranchPath
        SourceOwners.stepRecursiveFreshPublicPath false)
      (afterHash_excludes parameters
        SourceOwners.stepRecursiveFreshPublicPath
        (by decide) (by decide) (by decide)))

def recursiveEncodePlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    PrimitivePlan parameters profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.encodeInstanceCall
        parameters)
      SourceOwners.stepRecursiveEncodePath
      (CanonicalContexts.Step.afterFreshPublic parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath false) :=
  .invoke
    (CanonicalPrimitivePlan.invoke profile recipes
      .encodeInstance
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.hashAfterFreshPublic
          parameters)
        .nil)
      SourceOwners.stepRecursiveEncodePath
      (CanonicalContexts.Step.afterFreshPublic parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath false)
      (CanonicalContexts.Step.afterFreshPublicWidths parameters profile)
      (one_excludes_instruction SourceOwners.stepRecursiveEncodePath)
      (activation_excludes_instruction SourceOwners.stepBranchPath
        SourceOwners.stepRecursiveEncodePath false)
      (afterFreshPublic_excludes parameters
        SourceOwners.stepRecursiveEncodePath
        (by decide) (by decide) (by decide) (by decide)))

def recursiveEncodedEqualityPlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    PrimitivePlan parameters profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.encodedEqualCall
        parameters)
      SourceOwners.stepRecursiveEncodedEqualPath
      (CanonicalContexts.Step.afterEncode parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath false) :=
  .invoke
    (CanonicalPrimitivePlan.invoke profile recipes
      .encodedEqual
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.encoded
          parameters)
        (.cons
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.freshPublic
            parameters)
          .nil))
      SourceOwners.stepRecursiveEncodedEqualPath
      (CanonicalContexts.Step.afterEncode parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath false)
      (CanonicalContexts.Step.afterEncodeWidths parameters profile)
      (one_excludes_instruction SourceOwners.stepRecursiveEncodedEqualPath)
      (activation_excludes_instruction SourceOwners.stepBranchPath
        SourceOwners.stepRecursiveEncodedEqualPath false)
      (afterEncode_excludes parameters
        SourceOwners.stepRecursiveEncodedEqualPath
        (by decide) (by decide) (by decide) (by decide) (by decide)))

def recursiveAssertionPlan
    (parameters : Parameters)
    (profile : Profile parameters) :
    PrimitivePlan parameters profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.encodedAssertion
        parameters)
      SourceOwners.stepRecursiveAssertionPath
      (CanonicalContexts.Step.afterEncodedEquality parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath false) :=
  .assertTrue
    (CanonicalPrimitivePlan.assertion profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.encodedEqual
        parameters)
      SourceOwners.stepRecursiveAssertionPath
      (CanonicalContexts.Step.afterEncodedEquality parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath false)
      (CanonicalContexts.Step.afterEncodedEqualityWidths parameters profile))

/-- The exact invoked-call plan for the recursive NIFS instruction. This
named boundary exposes the selected recipe and physical frame without
reducing the complete Step plan. -/
def recursiveNifsInvokePlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :=
  CanonicalPrimitivePlan.invoke profile recipes
      .nifsVerify
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.running
          parameters)
        (.cons
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.fresh
            parameters)
          (.cons
            (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.nifsProof
              parameters)
            .nil)))
      SourceOwners.stepRecursiveNifsPath
      (CanonicalContexts.Step.afterEncodedEquality parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath false)
      (CanonicalContexts.Step.afterEncodedEqualityWidths parameters profile)
      (one_excludes_instruction SourceOwners.stepRecursiveNifsPath)
      (activation_excludes_instruction SourceOwners.stepBranchPath
        SourceOwners.stepRecursiveNifsPath false)
      (afterEncodedEquality_excludes parameters
        SourceOwners.stepRecursiveNifsPath
        (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide))

def recursiveNifsPlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    PrimitivePlan parameters profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.nifsVerifyCall
        parameters)
      SourceOwners.stepRecursiveNifsPath
      (CanonicalContexts.Step.afterEncodedEquality parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath false) :=
  .invoke (recursiveNifsInvokePlan parameters profile recipes)

/-- The recursive NIFS receipt contains the named selected recipe rows at the
named canonical call frame. -/
@[simp] theorem recursiveNifsPlan_receipt_rows
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    (recursiveNifsPlan parameters profile recipes).receipt.rows =
      (recursiveNifsInvokePlan parameters profile recipes).recipe.rows
        (recursiveNifsInvokePlan parameters profile recipes).frame :=
  rfl

def continuationHashPlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    PrimitivePlan parameters profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.hashNextCall
        parameters)
      SourceOwners.stepContinuationHashPath
      (CanonicalContexts.Step.continuationInput parameters)
      oneColumn oneColumn :=
  .invoke
    (CanonicalPrimitivePlan.invoke profile recipes
      .hashNext
      (.cons
        (.there
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.iteration
            parameters))
        (.cons
          (.there
            (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.z0
              parameters))
          (.cons
            (.there
              (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.zNext
                parameters))
            (.cons (.here (Ports.committedRunning parameters)) .nil))))
      SourceOwners.stepContinuationHashPath
      (CanonicalContexts.Step.continuationInput parameters)
      oneColumn oneColumn
      (CanonicalContexts.Step.continuationInputWidths parameters profile)
      (one_excludes_instruction SourceOwners.stepContinuationHashPath)
      (one_excludes_instruction SourceOwners.stepContinuationHashPath)
      (continuationInput_excludes parameters
        SourceOwners.stepContinuationHashPath
        (by decide) (by decide)))

/-- Exact non-input receipt order induced by the Step AST. -/
def bodyReceipts
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    List InstructionReceipt :=
  [ (applyPlan.{0} parameters profile recipes).receipt,
    (selectorPlan.{0} parameters profile recipes).receipt,
    CanonicalBranchPlan.trueActivationReceipt
      SourceOwners.stepBranchPath oneColumn oneColumn
      (CanonicalContexts.Step.selector parameters profile),
    CanonicalBranchPlan.falseActivationReceipt
      SourceOwners.stepBranchPath oneColumn oneColumn
      (CanonicalContexts.Step.selector parameters profile),
    (baseEqualityPlan.{0} parameters profile recipes).receipt,
    (baseAssertionPlan.{0} parameters profile).receipt,
    (baseLiteralPlan.{0} parameters profile defaultAdmissible).receipt,
    (recursiveHashPlan.{0} parameters profile recipes).receipt,
    (recursiveFreshPublicPlan.{0} parameters profile recipes).receipt,
    (recursiveEncodePlan.{0} parameters profile recipes).receipt,
    (recursiveEncodedEqualityPlan.{0} parameters profile recipes).receipt,
    (recursiveAssertionPlan.{0} parameters profile).receipt,
    (recursiveNifsPlan.{0} parameters profile recipes).receipt,
    CanonicalBranchPlan.onePortJoinReceipt
      SourceOwners.stepBranchPath
      (CanonicalContexts.Step.selector parameters profile)
      (Ports.committedRunning parameters)
      (CanonicalContexts.Step.baseRunning parameters)
      (CanonicalContexts.Step.recursiveRunning parameters),
    (continuationHashPlan.{0} parameters profile recipes).receipt ]

/-- The exact Step body is well-scoped from any allocation prefix that
contains the verifier one column and the complete typed input context. -/
theorem bodyWellScoped
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (available : List ColumnId)
    (oneAvailable : oneColumn ∈ available)
    (inputCovers :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.input parameters) available) :
    ReceiptsWellScoped available
      (bodyReceipts parameters profile recipes defaultAdmissible) := by
  let pApply := applyPlan.{0} parameters profile recipes
  let pSelector := selectorPlan.{0} parameters profile recipes
  let pBaseEquality := baseEqualityPlan.{0} parameters profile recipes
  let pBaseAssertion := baseAssertionPlan.{0} parameters profile
  let pBaseLiteral :=
    baseLiteralPlan.{0} parameters profile defaultAdmissible
  let pRecursiveHash := recursiveHashPlan.{0} parameters profile recipes
  let pRecursiveFresh :=
    recursiveFreshPublicPlan.{0} parameters profile recipes
  let pRecursiveEncode :=
    recursiveEncodePlan.{0} parameters profile recipes
  let pRecursiveEquality :=
    recursiveEncodedEqualityPlan.{0} parameters profile recipes
  let pRecursiveAssertion :=
    recursiveAssertionPlan.{0} parameters profile
  let pRecursiveNifs :=
    recursiveNifsPlan.{0} parameters profile recipes
  let pContinuation :=
    continuationHashPlan.{0} parameters profile recipes
  let trueReceipt :=
    CanonicalBranchPlan.trueActivationReceipt
      SourceOwners.stepBranchPath oneColumn oneColumn
      (CanonicalContexts.Step.selector parameters profile)
  let falseReceipt :=
    CanonicalBranchPlan.falseActivationReceipt
      SourceOwners.stepBranchPath oneColumn oneColumn
      (CanonicalContexts.Step.selector parameters profile)
  let joinReceipt :=
    CanonicalBranchPlan.onePortJoinReceipt
      SourceOwners.stepBranchPath
      (CanonicalContexts.Step.selector parameters profile)
      (Ports.committedRunning parameters)
      (CanonicalContexts.Step.baseRunning parameters)
      (CanonicalContexts.Step.recursiveRunning parameters)
  let afterApply := available ++ pApply.receipt.columnIds
  let afterSelector := afterApply ++ pSelector.receipt.columnIds
  let afterTrue := afterSelector ++ trueReceipt.columnIds
  let afterFalse := afterTrue ++ falseReceipt.columnIds
  let afterBaseEquality :=
    afterFalse ++ pBaseEquality.receipt.columnIds
  let afterBaseAssertion :=
    afterBaseEquality ++ pBaseAssertion.receipt.columnIds
  let afterBaseLiteral :=
    afterBaseAssertion ++ pBaseLiteral.receipt.columnIds
  let afterRecursiveHash :=
    afterBaseLiteral ++ pRecursiveHash.receipt.columnIds
  let afterRecursiveFresh :=
    afterRecursiveHash ++ pRecursiveFresh.receipt.columnIds
  let afterRecursiveEncode :=
    afterRecursiveFresh ++ pRecursiveEncode.receipt.columnIds
  let afterRecursiveEquality :=
    afterRecursiveEncode ++ pRecursiveEquality.receipt.columnIds
  let afterRecursiveAssertion :=
    afterRecursiveEquality ++ pRecursiveAssertion.receipt.columnIds
  let afterRecursiveNifs :=
    afterRecursiveAssertion ++ pRecursiveNifs.receipt.columnIds
  let afterJoin := afterRecursiveNifs ++ joinReceipt.columnIds
  change ReceiptsWellScoped available
    [pApply.receipt,
      pSelector.receipt,
      trueReceipt,
      falseReceipt,
      pBaseEquality.receipt,
      pBaseAssertion.receipt,
      pBaseLiteral.receipt,
      pRecursiveHash.receipt,
      pRecursiveFresh.receipt,
      pRecursiveEncode.receipt,
      pRecursiveEquality.receipt,
      pRecursiveAssertion.receipt,
      pRecursiveNifs.receipt,
      joinReceipt,
      pContinuation.receipt]

  have applyScoped :
      pApply.receipt.WellScopedAfter available :=
    PrimitivePlan.wellScopedAfter pApply available
      (ReceiptScoping.Covers.primitiveInputs
        inputCovers oneAvailable oneAvailable)
  have oneAfterApply : oneColumn ∈ afterApply :=
    List.mem_append_left _ oneAvailable
  have stepCovers :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.afterStep parameters) afterApply := by
    simpa only [pApply, applyPlan,
      ReceiptScoping.PrimitivePlan.ResultCoveredAfter,
      PrimitivePlan.receipt, CanonicalContexts.Step.afterStep] using
      ReceiptScoping.PrimitivePlan.resultCoveredAfter
        pApply available inputCovers
  constructor
  · exact applyScoped

  have selectorScoped :
      pSelector.receipt.WellScopedAfter afterApply :=
    PrimitivePlan.wellScopedAfter pSelector afterApply
      (ReceiptScoping.Covers.primitiveInputs
        stepCovers oneAfterApply oneAfterApply)
  have oneAfterSelector : oneColumn ∈ afterSelector :=
    List.mem_append_left _ oneAfterApply
  have commonCovers :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.common parameters) afterSelector := by
    simpa only [pSelector, selectorPlan,
      ReceiptScoping.PrimitivePlan.ResultCoveredAfter,
      PrimitivePlan.receipt, CanonicalContexts.Step.common] using
      ReceiptScoping.PrimitivePlan.resultCoveredAfter
        pSelector afterApply stepCovers
  have selectorAvailable :
      CanonicalContexts.Step.selector parameters profile ∈
        afterSelector := by
    apply commonCovers
    exact CanonicalPrimitivePlan.bitCoordinate_mem profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.iterationZero
        parameters)
      (CanonicalContexts.Step.common parameters)
      (CanonicalContexts.Step.commonWidths parameters profile)
  constructor
  · exact selectorScoped

  have trueScoped :
      trueReceipt.WellScopedAfter afterSelector := by
    apply CanonicalBranchPlan.trueActivationReceipt_wellScoped
    exact ⟨oneAfterSelector, oneAfterSelector, selectorAvailable⟩
  have oneAfterTrue : oneColumn ∈ afterTrue :=
    List.mem_append_left _ oneAfterSelector
  have selectorAfterTrue :
      CanonicalContexts.Step.selector parameters profile ∈ afterTrue :=
    List.mem_append_left _ selectorAvailable
  have trueAvailable :
      activationColumn SourceOwners.stepBranchPath true ∈ afterTrue := by
    exact ReceiptScoping.trueActivationAvailableAfter
      SourceOwners.stepBranchPath oneColumn oneColumn
      (CanonicalContexts.Step.selector parameters profile)
      afterSelector
  have commonAfterTrue :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.common parameters) afterTrue :=
    commonCovers.weaken trueReceipt.columnIds
  constructor
  · exact trueScoped

  have falseScoped :
      falseReceipt.WellScopedAfter afterTrue := by
    apply CanonicalBranchPlan.falseActivationReceipt_wellScoped
    exact ⟨oneAfterTrue, oneAfterTrue, selectorAfterTrue⟩
  have oneAfterFalse : oneColumn ∈ afterFalse :=
    List.mem_append_left _ oneAfterTrue
  have trueAfterFalse :
      activationColumn SourceOwners.stepBranchPath true ∈ afterFalse :=
    List.mem_append_left _ trueAvailable
  have falseAvailable :
      activationColumn SourceOwners.stepBranchPath false ∈ afterFalse := by
    exact ReceiptScoping.falseActivationAvailableAfter
      SourceOwners.stepBranchPath oneColumn oneColumn
      (CanonicalContexts.Step.selector parameters profile)
      afterTrue
  have commonAfterFalse :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.common parameters) afterFalse :=
    commonAfterTrue.weaken falseReceipt.columnIds
  constructor
  · exact falseScoped

  have baseEqualityScoped :
      pBaseEquality.receipt.WellScopedAfter afterFalse :=
    PrimitivePlan.wellScopedAfter pBaseEquality afterFalse
      (ReceiptScoping.Covers.primitiveInputs
        commonAfterFalse oneAfterFalse trueAfterFalse)
  have oneAfterBaseEquality : oneColumn ∈ afterBaseEquality :=
    List.mem_append_left _ oneAfterFalse
  have trueAfterBaseEquality :
      activationColumn SourceOwners.stepBranchPath true ∈
        afterBaseEquality :=
    List.mem_append_left _ trueAfterFalse
  have baseEqualityCovers :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.afterBaseEquality parameters)
        afterBaseEquality := by
    simpa only [pBaseEquality, baseEqualityPlan,
      ReceiptScoping.PrimitivePlan.ResultCoveredAfter,
      PrimitivePlan.receipt,
      CanonicalContexts.Step.afterBaseEquality] using
      ReceiptScoping.PrimitivePlan.resultCoveredAfter
        pBaseEquality afterFalse commonAfterFalse
  have commonAfterBaseEquality :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.common parameters)
        afterBaseEquality :=
    commonAfterFalse.weaken pBaseEquality.receipt.columnIds
  have falseAfterBaseEquality :
      activationColumn SourceOwners.stepBranchPath false ∈
        afterBaseEquality :=
    List.mem_append_left _ falseAvailable
  constructor
  · exact baseEqualityScoped

  have baseAssertionScoped :
      pBaseAssertion.receipt.WellScopedAfter afterBaseEquality :=
    PrimitivePlan.wellScopedAfter pBaseAssertion afterBaseEquality
      (ReceiptScoping.Covers.primitiveInputs
        baseEqualityCovers oneAfterBaseEquality trueAfterBaseEquality)
  have oneAfterBaseAssertion : oneColumn ∈ afterBaseAssertion :=
    List.mem_append_left _ oneAfterBaseEquality
  have trueAfterBaseAssertion :
      activationColumn SourceOwners.stepBranchPath true ∈
        afterBaseAssertion :=
    List.mem_append_left _ trueAfterBaseEquality
  have baseEqualityAfterAssertion :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.afterBaseEquality parameters)
        afterBaseAssertion :=
    baseEqualityCovers.weaken pBaseAssertion.receipt.columnIds
  have commonAfterBaseAssertion :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.common parameters)
        afterBaseAssertion :=
    commonAfterBaseEquality.weaken pBaseAssertion.receipt.columnIds
  have falseAfterBaseAssertion :
      activationColumn SourceOwners.stepBranchPath false ∈
        afterBaseAssertion :=
    List.mem_append_left _ falseAfterBaseEquality
  constructor
  · exact baseAssertionScoped

  have baseLiteralScoped :
      pBaseLiteral.receipt.WellScopedAfter afterBaseAssertion :=
    PrimitivePlan.wellScopedAfter pBaseLiteral afterBaseAssertion
      (ReceiptScoping.Covers.primitiveInputs
        baseEqualityAfterAssertion oneAfterBaseAssertion
          trueAfterBaseAssertion)
  have oneAfterBaseLiteral : oneColumn ∈ afterBaseLiteral :=
    List.mem_append_left _ oneAfterBaseAssertion
  have baseOutputCovers :
      ReceiptScoping.Covers
        (instructionColumns SourceOwners.stepBaseDefaultPath
          [Ports.committedRunning parameters])
        afterBaseLiteral := by
    simpa only [pBaseLiteral, baseLiteralPlan,
      ReceiptScoping.PrimitivePlan.FreshOutputsCoveredAfter,
      PrimitivePlan.receipt] using
      ReceiptScoping.PrimitivePlan.freshOutputsCoveredAfter
        pBaseLiteral afterBaseAssertion
  have commonAfterBaseLiteral :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.common parameters)
        afterBaseLiteral :=
    commonAfterBaseAssertion.weaken pBaseLiteral.receipt.columnIds
  have falseAfterBaseLiteral :
      activationColumn SourceOwners.stepBranchPath false ∈
        afterBaseLiteral :=
    List.mem_append_left _ falseAfterBaseAssertion
  constructor
  · exact baseLiteralScoped

  have recursiveHashScoped :
      pRecursiveHash.receipt.WellScopedAfter afterBaseLiteral :=
    PrimitivePlan.wellScopedAfter pRecursiveHash afterBaseLiteral
      (ReceiptScoping.Covers.primitiveInputs
        commonAfterBaseLiteral oneAfterBaseLiteral falseAfterBaseLiteral)
  have oneAfterRecursiveHash : oneColumn ∈ afterRecursiveHash :=
    List.mem_append_left _ oneAfterBaseLiteral
  have hashCovers :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.afterHash parameters)
        afterRecursiveHash := by
    simpa only [pRecursiveHash, recursiveHashPlan,
      ReceiptScoping.PrimitivePlan.ResultCoveredAfter,
      PrimitivePlan.receipt, CanonicalContexts.Step.afterHash] using
      ReceiptScoping.PrimitivePlan.resultCoveredAfter
        pRecursiveHash afterBaseLiteral commonAfterBaseLiteral
  have baseOutputAfterHash :
      ReceiptScoping.Covers
        (instructionColumns SourceOwners.stepBaseDefaultPath
          [Ports.committedRunning parameters])
        afterRecursiveHash :=
    baseOutputCovers.weaken pRecursiveHash.receipt.columnIds
  have commonAfterHash :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.common parameters)
        afterRecursiveHash :=
    commonAfterBaseLiteral.weaken pRecursiveHash.receipt.columnIds
  have falseAfterHash :
      activationColumn SourceOwners.stepBranchPath false ∈
        afterRecursiveHash :=
    List.mem_append_left _ falseAfterBaseLiteral
  constructor
  · exact recursiveHashScoped

  have recursiveFreshScoped :
      pRecursiveFresh.receipt.WellScopedAfter afterRecursiveHash :=
    PrimitivePlan.wellScopedAfter pRecursiveFresh afterRecursiveHash
      (ReceiptScoping.Covers.primitiveInputs
        hashCovers oneAfterRecursiveHash falseAfterHash)
  have oneAfterRecursiveFresh : oneColumn ∈ afterRecursiveFresh :=
    List.mem_append_left _ oneAfterRecursiveHash
  have freshCovers :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.afterFreshPublic parameters)
        afterRecursiveFresh := by
    simpa only [pRecursiveFresh, recursiveFreshPublicPlan,
      ReceiptScoping.PrimitivePlan.ResultCoveredAfter,
      PrimitivePlan.receipt,
      CanonicalContexts.Step.afterFreshPublic] using
      ReceiptScoping.PrimitivePlan.resultCoveredAfter
        pRecursiveFresh afterRecursiveHash hashCovers
  have baseOutputAfterFresh :
      ReceiptScoping.Covers
        (instructionColumns SourceOwners.stepBaseDefaultPath
          [Ports.committedRunning parameters])
        afterRecursiveFresh :=
    baseOutputAfterHash.weaken pRecursiveFresh.receipt.columnIds
  have commonAfterFresh :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.common parameters)
        afterRecursiveFresh :=
    commonAfterHash.weaken pRecursiveFresh.receipt.columnIds
  have falseAfterFresh :
      activationColumn SourceOwners.stepBranchPath false ∈
        afterRecursiveFresh :=
    List.mem_append_left _ falseAfterHash
  constructor
  · exact recursiveFreshScoped

  have recursiveEncodeScoped :
      pRecursiveEncode.receipt.WellScopedAfter afterRecursiveFresh :=
    PrimitivePlan.wellScopedAfter pRecursiveEncode afterRecursiveFresh
      (ReceiptScoping.Covers.primitiveInputs
        freshCovers oneAfterRecursiveFresh falseAfterFresh)
  have oneAfterRecursiveEncode : oneColumn ∈ afterRecursiveEncode :=
    List.mem_append_left _ oneAfterRecursiveFresh
  have encodeCovers :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.afterEncode parameters)
        afterRecursiveEncode := by
    simpa only [pRecursiveEncode, recursiveEncodePlan,
      ReceiptScoping.PrimitivePlan.ResultCoveredAfter,
      PrimitivePlan.receipt, CanonicalContexts.Step.afterEncode] using
      ReceiptScoping.PrimitivePlan.resultCoveredAfter
        pRecursiveEncode afterRecursiveFresh freshCovers
  have baseOutputAfterEncode :
      ReceiptScoping.Covers
        (instructionColumns SourceOwners.stepBaseDefaultPath
          [Ports.committedRunning parameters])
        afterRecursiveEncode :=
    baseOutputAfterFresh.weaken pRecursiveEncode.receipt.columnIds
  have commonAfterEncode :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.common parameters)
        afterRecursiveEncode :=
    commonAfterFresh.weaken pRecursiveEncode.receipt.columnIds
  have falseAfterEncode :
      activationColumn SourceOwners.stepBranchPath false ∈
        afterRecursiveEncode :=
    List.mem_append_left _ falseAfterFresh
  constructor
  · exact recursiveEncodeScoped

  have recursiveEqualityScoped :
      pRecursiveEquality.receipt.WellScopedAfter afterRecursiveEncode :=
    PrimitivePlan.wellScopedAfter pRecursiveEquality afterRecursiveEncode
      (ReceiptScoping.Covers.primitiveInputs
        encodeCovers oneAfterRecursiveEncode falseAfterEncode)
  have oneAfterRecursiveEquality : oneColumn ∈ afterRecursiveEquality :=
    List.mem_append_left _ oneAfterRecursiveEncode
  have equalityCovers :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.afterEncodedEquality parameters)
        afterRecursiveEquality := by
    simpa only [pRecursiveEquality, recursiveEncodedEqualityPlan,
      ReceiptScoping.PrimitivePlan.ResultCoveredAfter,
      PrimitivePlan.receipt,
      CanonicalContexts.Step.afterEncodedEquality] using
      ReceiptScoping.PrimitivePlan.resultCoveredAfter
        pRecursiveEquality afterRecursiveEncode encodeCovers
  have baseOutputAfterEquality :
      ReceiptScoping.Covers
        (instructionColumns SourceOwners.stepBaseDefaultPath
          [Ports.committedRunning parameters])
        afterRecursiveEquality :=
    baseOutputAfterEncode.weaken pRecursiveEquality.receipt.columnIds
  have commonAfterEquality :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.common parameters)
        afterRecursiveEquality :=
    commonAfterEncode.weaken pRecursiveEquality.receipt.columnIds
  have falseAfterEquality :
      activationColumn SourceOwners.stepBranchPath false ∈
        afterRecursiveEquality :=
    List.mem_append_left _ falseAfterEncode
  constructor
  · exact recursiveEqualityScoped

  have recursiveAssertionScoped :
      pRecursiveAssertion.receipt.WellScopedAfter
        afterRecursiveEquality :=
    PrimitivePlan.wellScopedAfter
      pRecursiveAssertion afterRecursiveEquality
      (ReceiptScoping.Covers.primitiveInputs
        equalityCovers oneAfterRecursiveEquality falseAfterEquality)
  have oneAfterRecursiveAssertion :
      oneColumn ∈ afterRecursiveAssertion :=
    List.mem_append_left _ oneAfterRecursiveEquality
  have equalityAfterAssertion :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.afterEncodedEquality parameters)
        afterRecursiveAssertion :=
    equalityCovers.weaken pRecursiveAssertion.receipt.columnIds
  have baseOutputAfterAssertion :
      ReceiptScoping.Covers
        (instructionColumns SourceOwners.stepBaseDefaultPath
          [Ports.committedRunning parameters])
        afterRecursiveAssertion :=
    baseOutputAfterEquality.weaken pRecursiveAssertion.receipt.columnIds
  have commonAfterAssertion :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.common parameters)
        afterRecursiveAssertion :=
    commonAfterEquality.weaken pRecursiveAssertion.receipt.columnIds
  have falseAfterAssertion :
      activationColumn SourceOwners.stepBranchPath false ∈
        afterRecursiveAssertion :=
    List.mem_append_left _ falseAfterEquality
  constructor
  · exact recursiveAssertionScoped

  have recursiveNifsScoped :
      pRecursiveNifs.receipt.WellScopedAfter afterRecursiveAssertion :=
    PrimitivePlan.wellScopedAfter pRecursiveNifs afterRecursiveAssertion
      (ReceiptScoping.Covers.primitiveInputs
        equalityAfterAssertion oneAfterRecursiveAssertion
          falseAfterAssertion)
  have oneAfterRecursiveNifs : oneColumn ∈ afterRecursiveNifs :=
    List.mem_append_left _ oneAfterRecursiveAssertion
  have recursiveOutputCovers :
      ReceiptScoping.Covers
        (instructionColumns SourceOwners.stepRecursiveNifsPath
          [Ports.committedRunning parameters])
        afterRecursiveNifs := by
    simpa only [pRecursiveNifs, recursiveNifsPlan,
      ReceiptScoping.PrimitivePlan.FreshOutputsCoveredAfter,
      PrimitivePlan.receipt] using
      ReceiptScoping.PrimitivePlan.freshOutputsCoveredAfter
        pRecursiveNifs afterRecursiveAssertion
  have baseOutputAfterNifs :
      ReceiptScoping.Covers
        (instructionColumns SourceOwners.stepBaseDefaultPath
          [Ports.committedRunning parameters])
        afterRecursiveNifs :=
    baseOutputAfterAssertion.weaken pRecursiveNifs.receipt.columnIds
  have commonAfterNifs :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.common parameters)
        afterRecursiveNifs :=
    commonAfterAssertion.weaken pRecursiveNifs.receipt.columnIds
  have selectorAfterNifs :
      CanonicalContexts.Step.selector parameters profile ∈
        afterRecursiveNifs := by
    apply commonAfterNifs
    exact CanonicalPrimitivePlan.bitCoordinate_mem profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.iterationZero
        parameters)
      (CanonicalContexts.Step.common parameters)
      (CanonicalContexts.Step.commonWidths parameters profile)
  constructor
  · exact recursiveNifsScoped

  have joinInputs :
      CanonicalBranchPlan.JoinInputsAvailable
        (CanonicalContexts.Step.selector parameters profile)
        (CanonicalContexts.Step.baseRunning parameters)
        (CanonicalContexts.Step.recursiveRunning parameters)
        afterRecursiveNifs := by
    intro column member
    rcases List.mem_append.mp member with
      selectorOrBase | recursiveMember
    · rcases List.mem_append.mp selectorOrBase with
        selectorMember | baseMember
      · simp only [List.mem_singleton] at selectorMember
        exact selectorMember ▸ selectorAfterNifs
      · apply baseOutputAfterNifs column
        rw [ReceiptScoping.singletonColumnsIds]
        simpa only [CanonicalContexts.Step.baseRunning] using baseMember
    · apply recursiveOutputCovers column
      rw [ReceiptScoping.singletonColumnsIds]
      simpa only [CanonicalContexts.Step.recursiveRunning] using
        recursiveMember
  have joinScoped :
      joinReceipt.WellScopedAfter afterRecursiveNifs :=
    CanonicalBranchPlan.onePortJoinReceipt_wellScoped
      SourceOwners.stepBranchPath
      (CanonicalContexts.Step.selector parameters profile)
      (Ports.committedRunning parameters)
      (CanonicalContexts.Step.baseRunning parameters)
      (CanonicalContexts.Step.recursiveRunning parameters)
      afterRecursiveNifs joinInputs
  have oneAfterJoin : oneColumn ∈ afterJoin :=
    List.mem_append_left _ oneAfterRecursiveNifs
  have joinedCovers :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.joined parameters) afterJoin := by
    simpa only [joinReceipt, CanonicalContexts.Step.joined] using
      ReceiptScoping.joinOutputCoversAfter
        SourceOwners.stepBranchPath
        (CanonicalContexts.Step.selector parameters profile)
        (Ports.committedRunning parameters)
        (CanonicalContexts.Step.baseRunning parameters)
        (CanonicalContexts.Step.recursiveRunning parameters)
        afterRecursiveNifs
  have commonAfterJoin :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.common parameters) afterJoin :=
    commonAfterNifs.weaken joinReceipt.columnIds
  have continuationCovers :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.continuationInput parameters) afterJoin := by
    simpa only [CanonicalContexts.Step.continuationInput] using
      ReceiptScoping.Covers.append joinedCovers commonAfterJoin
  constructor
  · exact joinScoped

  have continuationScoped :
      pContinuation.receipt.WellScopedAfter afterJoin :=
    PrimitivePlan.wellScopedAfter pContinuation afterJoin
      (ReceiptScoping.Covers.primitiveInputs
        continuationCovers oneAfterJoin oneAfterJoin)
  constructor
  · exact continuationScoped
  · trivial

/-- Prelude, exact input receipts, and the exact Step body receipt order. -/
def receipts
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    List InstructionReceipt :=
  InstructionReceipt.prelude ::
    InputReceipts.receipts (stepInputSchema parameters) ++
      bodyReceipts parameters profile recipes defaultAdmissible

/-- The constructed body receipts preserve the exact structural Step owner
order. -/
theorem bodyOwnersExact
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    (bodyReceipts parameters profile recipes defaultAdmissible).map
        (fun receipt => receipt.owner) =
      SourceOwners.stepBodyOwners := by
  simp [bodyReceipts, SourceOwners.stepBodyOwners,
    PrimitivePlan.receipt_owner]

/-- The complete constructed receipt list has exactly the source-derived
Step owner skeleton. -/
theorem ownersExact
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    (receipts parameters profile recipes defaultAdmissible).map
        (fun receipt => receipt.owner) =
      SourceOwners.stepOwners parameters := by
  simp [receipts, SourceOwners.stepOwners,
    SourceAlignment.inputOwners, InputReceipts.owners_exact,
    bodyOwnersExact, InstructionReceipt.prelude]

/-- Structural paths and input slots in the Step owner skeleton are
collision-free. -/
theorem sourceOwnersNodup (parameters : Parameters) :
    (SourceOwners.stepOwners parameters).Nodup := by
  rw [SourceOwners.stepOwners]
  constructor
  · intro owner member equal
    subst owner
    simp [SourceAlignment.inputOwners, SourceOwners.stepBodyOwners] at member
  · have tailNodup :
        (SourceAlignment.inputOwners (stepInputSchema parameters) ++
          SourceOwners.stepBodyOwners).Nodup := by
      rw [List.nodup_append]
      refine ⟨?_, ?_, ?_⟩
      · have inputNodup :=
          InputReceipts.ownersNodup (stepInputSchema parameters)
        rw [InputReceipts.owners_exact] at inputNodup
        exact inputNodup
      · decide
      · simp [SourceAlignment.inputOwners, SourceOwners.stepBodyOwners]
    exact tailNodup

theorem bodyLocalColumnIdsNodup
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    ∀ receipt,
      receipt ∈ bodyReceipts parameters profile recipes defaultAdmissible ->
        receipt.columnIds.Nodup := by
  intro receipt member
  simp only [bodyReceipts, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with
    rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl
  · exact PrimitivePlan.columnIdsNodup _
  · exact PrimitivePlan.columnIdsNodup _
  · exact CanonicalBranchPlan.trueActivationReceipt_columnIdsNodup _ _ _ _
  · exact CanonicalBranchPlan.falseActivationReceipt_columnIdsNodup _ _ _ _
  · exact PrimitivePlan.columnIdsNodup _
  · exact PrimitivePlan.columnIdsNodup _
  · exact PrimitivePlan.columnIdsNodup _
  · exact PrimitivePlan.columnIdsNodup _
  · exact PrimitivePlan.columnIdsNodup _
  · exact PrimitivePlan.columnIdsNodup _
  · exact PrimitivePlan.columnIdsNodup _
  · exact PrimitivePlan.columnIdsNodup _
  · exact PrimitivePlan.columnIdsNodup _
  · exact CanonicalBranchPlan.onePortJoinReceipt_columnIdsNodup _ _ _ _ _
  · exact PrimitivePlan.columnIdsNodup _

theorem bodyLocalRowIdsNodup
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    ∀ receipt,
      receipt ∈ bodyReceipts parameters profile recipes defaultAdmissible ->
        receipt.rowIds.Nodup := by
  intro receipt member
  simp only [bodyReceipts, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with
    rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl
  · exact PrimitivePlan.rowIdsNodup _
  · exact PrimitivePlan.rowIdsNodup _
  · exact CanonicalBranchPlan.trueActivationReceipt_rowIdsNodup _ _ _ _
  · exact CanonicalBranchPlan.falseActivationReceipt_rowIdsNodup _ _ _ _
  · exact PrimitivePlan.rowIdsNodup _
  · exact PrimitivePlan.rowIdsNodup _
  · exact PrimitivePlan.rowIdsNodup _
  · exact PrimitivePlan.rowIdsNodup _
  · exact PrimitivePlan.rowIdsNodup _
  · exact PrimitivePlan.rowIdsNodup _
  · exact PrimitivePlan.rowIdsNodup _
  · exact PrimitivePlan.rowIdsNodup _
  · exact PrimitivePlan.rowIdsNodup _
  · exact CanonicalBranchPlan.onePortJoinReceipt_rowIdsNodup _ _ _ _ _
  · exact PrimitivePlan.rowIdsNodup _

theorem localColumnIdsNodup
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    ∀ receipt,
      receipt ∈ receipts parameters profile recipes defaultAdmissible ->
        receipt.columnIds.Nodup := by
  intro receipt member
  rw [receipts] at member
  rcases List.mem_cons.mp member with prelude | tail
  · subst receipt
    simp [InstructionReceipt.prelude, InstructionReceipt.columnIds,
      preludeColumns]
  · rcases List.mem_append.mp tail with input | body
    · exact InputReceipts.localColumnIdsNodup
        (stepInputSchema parameters) receipt input
    · exact bodyLocalColumnIdsNodup
        parameters profile recipes defaultAdmissible receipt body

theorem localRowIdsNodup
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    ∀ receipt,
      receipt ∈ receipts parameters profile recipes defaultAdmissible ->
        receipt.rowIds.Nodup := by
  intro receipt member
  rw [receipts] at member
  rcases List.mem_cons.mp member with prelude | tail
  · subst receipt
    simp [InstructionReceipt.prelude, InstructionReceipt.rowIds]
  · rcases List.mem_append.mp tail with input | body
    · exact InputReceipts.localRowIdsNodup
        (stepInputSchema parameters) receipt input
    · exact bodyLocalRowIdsNodup
        parameters profile recipes defaultAdmissible receipt body

/-- Prelude, inputs, and the exact Step body form one premise-free scoped
receipt sequence. -/
theorem wellScoped
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    ReceiptsWellScoped []
      (receipts parameters profile recipes defaultAdmissible) := by
  let inputPrefix :=
    InstructionReceipt.prelude ::
      InputReceipts.receipts (stepInputSchema parameters)
  have prefixScoped : ReceiptsWellScoped [] inputPrefix := by
    simpa only [inputPrefix] using
      InputReceipts.wellScopedAfterPrelude
        (stepInputSchema parameters)
  have oneInPrefix :
      oneColumn ∈
        inputPrefix.flatMap InstructionReceipt.columnIds := by
    simp [inputPrefix, InstructionReceipt.prelude_columnIds]
  have inputCovered :
      ReceiptScoping.Covers
        (CanonicalContexts.Step.input parameters)
        (inputPrefix.flatMap InstructionReceipt.columnIds) := by
    intro column member
    simp only [inputPrefix, List.flatMap_cons,
      InstructionReceipt.prelude_columnIds]
    apply List.mem_append_right [oneColumn]
    rw [InputReceipts.columnIds_exact]
    exact member
  have bodyScoped :
      ReceiptsWellScoped
        (inputPrefix.flatMap InstructionReceipt.columnIds)
        (bodyReceipts parameters profile recipes defaultAdmissible) :=
    bodyWellScoped parameters profile recipes defaultAdmissible
      (inputPrefix.flatMap InstructionReceipt.columnIds)
      oneInPrefix inputCovered
  have combined :=
    ReceiptScoping.wellScoped_append
      [] inputPrefix
      (bodyReceipts parameters profile recipes defaultAdmissible)
      prefixScoped bodyScoped
  simpa only [inputPrefix, receipts] using combined

/-- Conserved physical Step program: every column and row is owned by exactly
one receipt, and the receipt sequence is scoped from the empty prefix. -/
def physical
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    ReceiptProgram
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.program
        parameters) where
  receipts := receipts parameters profile recipes defaultAdmissible
  preludeMember := by
    simp [receipts]
  ownersNodup := by
    rw [ownersExact]
    exact sourceOwnersNodup parameters
  localColumnIdsNodup :=
    localColumnIdsNodup parameters profile recipes defaultAdmissible
  localRowIdsNodup :=
    localRowIdsNodup parameters profile recipes defaultAdmissible
  wellScoped :=
    wellScoped parameters profile recipes defaultAdmissible

/-- The conserved physical Step program has exactly the owner skeleton
derived from the typed Step AST. -/
def aligned
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    SourceAlignment.AlignedReceiptProgram
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.program
        parameters) where
  physical := physical parameters profile recipes defaultAdmissible
  ownersExact := by
    rw [SourceOwners.stepProgramOwnersExact]
    exact ownersExact parameters profile recipes defaultAdmissible

end CanonicalStepPlan

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
