import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepPlan
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.VisibleCompletion

/-!
Contract: constructive views of the exact canonical Step primitive plans.

Owns:
- the concrete `InvokePlan` beneath each canonical Step call occurrence;
- the concrete literal plan beneath the base default-running occurrence.

Does not own: alternate recipes, receipt order, semantic execution,
temporary completion, Rust behavior, or generated artifacts.

Every view uses the same call, operands, structural path, exact context, and
controls as `CanonicalStepPlan`; it introduces no new encoding choice.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

namespace CanonicalStepConstructionPlans

def apply
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    InvokePlan parameters profile .step
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.zi
          parameters)
        (.cons
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.witness
            parameters)
          .nil))
      SourceOwners.stepApplyPath
      (CanonicalContexts.Step.input parameters)
      oneColumn oneColumn :=
  CanonicalPrimitivePlan.invoke profile recipes
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
    (CanonicalStepPlan.one_excludes_instruction SourceOwners.stepApplyPath)
    (CanonicalStepPlan.one_excludes_instruction SourceOwners.stepApplyPath)
    (CanonicalPrimitivePlan.ContextExcludesOwner.input
      (stepInputSchema parameters) SourceOwners.stepApplyPath)

def selector
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    InvokePlan parameters profile .iterationZero
      (.cons
        (.there
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.iteration
            parameters))
        .nil)
      SourceOwners.stepSelectorPath
      (CanonicalContexts.Step.afterStep parameters)
      oneColumn oneColumn :=
  CanonicalPrimitivePlan.invoke profile recipes
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
    (CanonicalStepPlan.one_excludes_instruction SourceOwners.stepSelectorPath)
    (CanonicalStepPlan.one_excludes_instruction SourceOwners.stepSelectorPath)
    (CanonicalStepPlan.afterStep_excludes parameters
      SourceOwners.stepSelectorPath (by decide))

def baseEquality
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    InvokePlan parameters profile .stateEqual
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.z0
          parameters)
        (.cons
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.zi
            parameters)
          .nil))
      SourceOwners.stepBaseStateEqualPath
      (CanonicalContexts.Step.common parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath true) :=
  CanonicalPrimitivePlan.invoke profile recipes
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
    (CanonicalStepPlan.one_excludes_instruction
      SourceOwners.stepBaseStateEqualPath)
    (CanonicalStepPlan.activation_excludes_instruction
      SourceOwners.stepBranchPath SourceOwners.stepBaseStateEqualPath true)
    (CanonicalStepPlan.common_excludes parameters
      SourceOwners.stepBaseStateEqualPath (by decide) (by decide))

def baseLiteral
    (parameters : Parameters)
    (profile : Profile parameters)
    (admissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    LiteralPlan parameters profile
      (Ports.committedRunning parameters)
      (defaultRunning parameters)
      SourceOwners.stepBaseDefaultPath
      (CanonicalContexts.Step.afterBaseEquality parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath true) :=
  CanonicalPrimitivePlan.literal profile
    (Ports.committedRunning parameters)
    (defaultRunning parameters)
    SourceOwners.stepBaseDefaultPath
    (CanonicalContexts.Step.afterBaseEquality parameters)
    oneColumn (activationColumn SourceOwners.stepBranchPath true)
    (profile.committedRunning_widthsAgree parameters)
    admissible

def recursiveHash
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    InvokePlan parameters profile .hashPrior
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
      oneColumn (activationColumn SourceOwners.stepBranchPath false) :=
  CanonicalPrimitivePlan.invoke profile recipes
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
    (CanonicalStepPlan.one_excludes_instruction
      SourceOwners.stepRecursiveHashPriorPath)
    (CanonicalStepPlan.activation_excludes_instruction
      SourceOwners.stepBranchPath
      SourceOwners.stepRecursiveHashPriorPath false)
    (CanonicalStepPlan.common_excludes parameters
      SourceOwners.stepRecursiveHashPriorPath (by decide) (by decide))

def recursiveFreshPublic
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    InvokePlan parameters profile .freshPublic
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.freshAfterHash
          parameters)
        .nil)
      SourceOwners.stepRecursiveFreshPublicPath
      (CanonicalContexts.Step.afterHash parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath false) :=
  CanonicalPrimitivePlan.invoke profile recipes
    .freshPublic
    (.cons
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.freshAfterHash
        parameters)
      .nil)
    SourceOwners.stepRecursiveFreshPublicPath
    (CanonicalContexts.Step.afterHash parameters)
    oneColumn (activationColumn SourceOwners.stepBranchPath false)
    (CanonicalContexts.Step.afterHashWidths parameters profile)
    (CanonicalStepPlan.one_excludes_instruction
      SourceOwners.stepRecursiveFreshPublicPath)
    (CanonicalStepPlan.activation_excludes_instruction
      SourceOwners.stepBranchPath
      SourceOwners.stepRecursiveFreshPublicPath false)
    (CanonicalStepPlan.afterHash_excludes parameters
      SourceOwners.stepRecursiveFreshPublicPath
      (by decide) (by decide) (by decide))

def recursiveEncode
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    InvokePlan parameters profile .encodeInstance
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.hashAfterFreshPublic
          parameters)
        .nil)
      SourceOwners.stepRecursiveEncodePath
      (CanonicalContexts.Step.afterFreshPublic parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath false) :=
  CanonicalPrimitivePlan.invoke profile recipes
    .encodeInstance
    (.cons
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.hashAfterFreshPublic
        parameters)
      .nil)
    SourceOwners.stepRecursiveEncodePath
    (CanonicalContexts.Step.afterFreshPublic parameters)
    oneColumn (activationColumn SourceOwners.stepBranchPath false)
    (CanonicalContexts.Step.afterFreshPublicWidths parameters profile)
    (CanonicalStepPlan.one_excludes_instruction
      SourceOwners.stepRecursiveEncodePath)
    (CanonicalStepPlan.activation_excludes_instruction
      SourceOwners.stepBranchPath
      SourceOwners.stepRecursiveEncodePath false)
    (CanonicalStepPlan.afterFreshPublic_excludes parameters
      SourceOwners.stepRecursiveEncodePath
      (by decide) (by decide) (by decide) (by decide))

def recursiveEncodedEquality
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    InvokePlan parameters profile .encodedEqual
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.encoded
          parameters)
        (.cons
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.freshPublic
            parameters)
          .nil))
      SourceOwners.stepRecursiveEncodedEqualPath
      (CanonicalContexts.Step.afterEncode parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath false) :=
  CanonicalPrimitivePlan.invoke profile recipes
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
    (CanonicalStepPlan.one_excludes_instruction
      SourceOwners.stepRecursiveEncodedEqualPath)
    (CanonicalStepPlan.activation_excludes_instruction
      SourceOwners.stepBranchPath
      SourceOwners.stepRecursiveEncodedEqualPath false)
    (CanonicalStepPlan.afterEncode_excludes parameters
      SourceOwners.stepRecursiveEncodedEqualPath
      (by decide) (by decide) (by decide) (by decide) (by decide))

def recursiveNifs
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    InvokePlan parameters profile .nifsVerify
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
      oneColumn (activationColumn SourceOwners.stepBranchPath false) :=
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
    (CanonicalStepPlan.one_excludes_instruction
      SourceOwners.stepRecursiveNifsPath)
    (CanonicalStepPlan.activation_excludes_instruction
      SourceOwners.stepBranchPath
      SourceOwners.stepRecursiveNifsPath false)
    (CanonicalStepPlan.afterEncodedEquality_excludes parameters
      SourceOwners.stepRecursiveNifsPath
      (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide))

def continuationHash
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    InvokePlan parameters profile .hashNext
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
      oneColumn oneColumn :=
  CanonicalPrimitivePlan.invoke profile recipes
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
    (CanonicalStepPlan.one_excludes_instruction
      SourceOwners.stepContinuationHashPath)
    (CanonicalStepPlan.one_excludes_instruction
      SourceOwners.stepContinuationHashPath)
    (CanonicalStepPlan.continuationInput_excludes parameters
      SourceOwners.stepContinuationHashPath (by decide) (by decide))

end CanonicalStepConstructionPlans

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
