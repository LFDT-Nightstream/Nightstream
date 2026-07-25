import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalContexts
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ReceiptScoping

/-!
Contract: canonical local primitive plans, exact receipt order, and conserved
physical program for the fixed-one Terminal verifier.

Owns:
- one constructive physical plan for every Terminal primitive occurrence;
- exact branch controls and the mandatory empty join receipt;
- premise-free whole-program receipt scoping and source-owner alignment.

Does not own: normal-form minimality, row satisfaction, Rust emission,
generated artifacts, or Step lowering.

All physical identities come from typed contexts and structural paths.

Emits constraints: the exact receipts selected by the Terminal source program.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

namespace CanonicalTerminalPlan

private theorem one_excludes_instruction (path : OwnerPath) :
    oneColumn.owner ≠ .typed (.instruction path) := by
  simp [oneColumn]

private theorem activation_excludes_instruction
    (branchPath target : OwnerPath)
    (selected : Bool) :
    (activationColumn branchPath selected).owner ≠
      .typed (.instruction target) := by
  simp [activationColumn]

private theorem branchInput_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (selectorDifferent : SourceOwners.terminalSelectorPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Terminal.branchInput parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.terminalSelectorPath target
      [Ports.auxiliaryBit parameters] selectorDifferent)
    (CanonicalPrimitivePlan.ContextExcludesOwner.input
      (terminalInputSchema parameters) target)

private theorem afterHash_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (hashDifferent :
      SourceOwners.terminalRecursiveHashPriorPath ≠ target)
    (selectorDifferent : SourceOwners.terminalSelectorPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Terminal.afterHash parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.terminalRecursiveHashPriorPath target
      [Ports.auxiliaryDigest parameters] hashDifferent)
    (branchInput_excludes parameters target selectorDifferent)

private theorem afterFreshPublic_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (freshDifferent :
      SourceOwners.terminalRecursiveFreshPublicPath ≠ target)
    (hashDifferent :
      SourceOwners.terminalRecursiveHashPriorPath ≠ target)
    (selectorDifferent : SourceOwners.terminalSelectorPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Terminal.afterFreshPublic parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.terminalRecursiveFreshPublicPath target
      [Ports.auxiliaryEncoded parameters] freshDifferent)
    (afterHash_excludes parameters target hashDifferent selectorDifferent)

private theorem afterEncode_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (encodeDifferent :
      SourceOwners.terminalRecursiveEncodePath ≠ target)
    (freshDifferent :
      SourceOwners.terminalRecursiveFreshPublicPath ≠ target)
    (hashDifferent :
      SourceOwners.terminalRecursiveHashPriorPath ≠ target)
    (selectorDifferent : SourceOwners.terminalSelectorPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Terminal.afterEncode parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.terminalRecursiveEncodePath target
      [Ports.auxiliaryEncoded parameters] encodeDifferent)
    (afterFreshPublic_excludes parameters target freshDifferent
      hashDifferent selectorDifferent)

private theorem afterEncodedEquality_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (equalityDifferent :
      SourceOwners.terminalRecursiveEncodedEqualPath ≠ target)
    (encodeDifferent :
      SourceOwners.terminalRecursiveEncodePath ≠ target)
    (freshDifferent :
      SourceOwners.terminalRecursiveFreshPublicPath ≠ target)
    (hashDifferent :
      SourceOwners.terminalRecursiveHashPriorPath ≠ target)
    (selectorDifferent : SourceOwners.terminalSelectorPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Terminal.afterEncodedEquality parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.terminalRecursiveEncodedEqualPath target
      [Ports.auxiliaryBit parameters] equalityDifferent)
    (afterEncode_excludes parameters target encodeDifferent
      freshDifferent hashDifferent selectorDifferent)

private theorem afterRunningCheck_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (runningDifferent :
      SourceOwners.terminalRecursiveRunningCheckPath ≠ target)
    (equalityDifferent :
      SourceOwners.terminalRecursiveEncodedEqualPath ≠ target)
    (encodeDifferent :
      SourceOwners.terminalRecursiveEncodePath ≠ target)
    (freshDifferent :
      SourceOwners.terminalRecursiveFreshPublicPath ≠ target)
    (hashDifferent :
      SourceOwners.terminalRecursiveHashPriorPath ≠ target)
    (selectorDifferent : SourceOwners.terminalSelectorPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Terminal.afterRunningCheck parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.terminalRecursiveRunningCheckPath target
      [Ports.auxiliaryBit parameters] runningDifferent)
    (afterEncodedEquality_excludes parameters target
      equalityDifferent encodeDifferent freshDifferent
      hashDifferent selectorDifferent)

def selectorInvokePlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    InvokePlan parameters profile .iterationZero
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.InputRef.iteration
          parameters)
        .nil)
      SourceOwners.terminalSelectorPath
      (CanonicalContexts.Terminal.input parameters)
      oneColumn oneColumn :=
  CanonicalPrimitivePlan.invoke profile recipes
    .iterationZero
    (.cons
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.InputRef.iteration
        parameters)
      .nil)
    SourceOwners.terminalSelectorPath
    (CanonicalContexts.Terminal.input parameters)
    oneColumn oneColumn
    (CanonicalContexts.Terminal.inputWidths parameters profile)
    (one_excludes_instruction SourceOwners.terminalSelectorPath)
    (one_excludes_instruction SourceOwners.terminalSelectorPath)
    (CanonicalPrimitivePlan.ContextExcludesOwner.input
      (terminalInputSchema parameters)
      SourceOwners.terminalSelectorPath)

def selectorPlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    PrimitivePlan parameters profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.iterationZeroCall
        parameters)
      SourceOwners.terminalSelectorPath
      (CanonicalContexts.Terminal.input parameters)
      oneColumn oneColumn :=
  .invoke (selectorInvokePlan parameters profile recipes)

def baseEqualityInvokePlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    InvokePlan parameters profile .stateEqual
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.BranchRef.zi
          parameters)
        (.cons
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.BranchRef.z0
            parameters)
          .nil))
      SourceOwners.terminalBaseStateEqualPath
      (CanonicalContexts.Terminal.branchInput parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath true) :=
  CanonicalPrimitivePlan.invoke profile recipes
    .stateEqual
    (.cons
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.BranchRef.zi
        parameters)
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.BranchRef.z0
          parameters)
        .nil))
    SourceOwners.terminalBaseStateEqualPath
    (CanonicalContexts.Terminal.branchInput parameters)
    oneColumn (activationColumn SourceOwners.terminalBranchPath true)
    (CanonicalContexts.Terminal.branchInputWidths parameters profile)
    (one_excludes_instruction SourceOwners.terminalBaseStateEqualPath)
    (activation_excludes_instruction SourceOwners.terminalBranchPath
      SourceOwners.terminalBaseStateEqualPath true)
    (branchInput_excludes parameters
      SourceOwners.terminalBaseStateEqualPath (by decide))

def baseEqualityPlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    PrimitivePlan parameters profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.baseStateEqualCall
        parameters)
      SourceOwners.terminalBaseStateEqualPath
      (CanonicalContexts.Terminal.branchInput parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath true) :=
  .invoke (baseEqualityInvokePlan parameters profile recipes)

def baseAssertionPlan
    (parameters : Parameters)
    (profile : Profile parameters) :
    PrimitivePlan parameters profile
      (Primitive.assertTrue (signature := signature parameters)
        (.here (Ports.auxiliaryBit parameters) :
          Ref (typeSystem parameters)
            (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterBaseEqualitySchema
              parameters) .bit))
      SourceOwners.terminalBaseAssertionPath
      (CanonicalContexts.Terminal.afterBaseEquality parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath true) :=
  .assertTrue
    (CanonicalPrimitivePlan.assertion profile
      (.here (Ports.auxiliaryBit parameters))
      SourceOwners.terminalBaseAssertionPath
      (CanonicalContexts.Terminal.afterBaseEquality parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath true)
      (CanonicalContexts.Terminal.afterBaseEqualityWidths
        parameters profile))

def recursiveHashInvokePlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    InvokePlan parameters profile .hashPrior
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.BranchRef.iteration
          parameters)
        (.cons
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.BranchRef.z0
            parameters)
          (.cons
            (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.BranchRef.zi
              parameters)
            (.cons
              (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.BranchRef.running
                parameters)
              .nil))))
      SourceOwners.terminalRecursiveHashPriorPath
      (CanonicalContexts.Terminal.branchInput parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath false) :=
  CanonicalPrimitivePlan.invoke profile recipes
    .hashPrior
    (.cons
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.BranchRef.iteration
        parameters)
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.BranchRef.z0
          parameters)
        (.cons
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.BranchRef.zi
            parameters)
          (.cons
            (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.BranchRef.running
              parameters)
            .nil))))
    SourceOwners.terminalRecursiveHashPriorPath
    (CanonicalContexts.Terminal.branchInput parameters)
    oneColumn (activationColumn SourceOwners.terminalBranchPath false)
    (CanonicalContexts.Terminal.branchInputWidths parameters profile)
    (one_excludes_instruction
      SourceOwners.terminalRecursiveHashPriorPath)
    (activation_excludes_instruction SourceOwners.terminalBranchPath
      SourceOwners.terminalRecursiveHashPriorPath false)
    (branchInput_excludes parameters
      SourceOwners.terminalRecursiveHashPriorPath (by decide))

def recursiveHashPlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    PrimitivePlan parameters profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.hashPriorCall
        parameters)
      SourceOwners.terminalRecursiveHashPriorPath
      (CanonicalContexts.Terminal.branchInput parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath false) :=
  .invoke (recursiveHashInvokePlan parameters profile recipes)

def recursiveFreshPublicInvokePlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    InvokePlan parameters profile .freshPublic
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.freshAfterHash
          parameters)
        .nil)
      SourceOwners.terminalRecursiveFreshPublicPath
      (CanonicalContexts.Terminal.afterHash parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath false) :=
  CanonicalPrimitivePlan.invoke profile recipes
    .freshPublic
    (.cons
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.freshAfterHash
        parameters)
      .nil)
    SourceOwners.terminalRecursiveFreshPublicPath
    (CanonicalContexts.Terminal.afterHash parameters)
    oneColumn (activationColumn SourceOwners.terminalBranchPath false)
    (CanonicalContexts.Terminal.afterHashWidths parameters profile)
    (one_excludes_instruction
      SourceOwners.terminalRecursiveFreshPublicPath)
    (activation_excludes_instruction SourceOwners.terminalBranchPath
      SourceOwners.terminalRecursiveFreshPublicPath false)
    (afterHash_excludes parameters
      SourceOwners.terminalRecursiveFreshPublicPath
      (by decide) (by decide))

def recursiveFreshPublicPlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    PrimitivePlan parameters profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.freshPublicCall
        parameters)
      SourceOwners.terminalRecursiveFreshPublicPath
      (CanonicalContexts.Terminal.afterHash parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath false) :=
  .invoke (recursiveFreshPublicInvokePlan parameters profile recipes)

def recursiveEncodeInvokePlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    InvokePlan parameters profile .encodeInstance
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.hashAfterFreshPublic
          parameters)
        .nil)
      SourceOwners.terminalRecursiveEncodePath
      (CanonicalContexts.Terminal.afterFreshPublic parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath false) :=
  CanonicalPrimitivePlan.invoke profile recipes
    .encodeInstance
    (.cons
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.hashAfterFreshPublic
        parameters)
      .nil)
    SourceOwners.terminalRecursiveEncodePath
    (CanonicalContexts.Terminal.afterFreshPublic parameters)
    oneColumn (activationColumn SourceOwners.terminalBranchPath false)
    (CanonicalContexts.Terminal.afterFreshPublicWidths
      parameters profile)
    (one_excludes_instruction SourceOwners.terminalRecursiveEncodePath)
    (activation_excludes_instruction SourceOwners.terminalBranchPath
      SourceOwners.terminalRecursiveEncodePath false)
    (afterFreshPublic_excludes parameters
      SourceOwners.terminalRecursiveEncodePath
      (by decide) (by decide) (by decide))

def recursiveEncodePlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    PrimitivePlan parameters profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.encodeInstanceCall
        parameters)
      SourceOwners.terminalRecursiveEncodePath
      (CanonicalContexts.Terminal.afterFreshPublic parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath false) :=
  .invoke (recursiveEncodeInvokePlan parameters profile recipes)

def recursiveEncodedEqualityInvokePlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    InvokePlan parameters profile .encodedEqual
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.freshPublic
          parameters)
        (.cons
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.encoded
            parameters)
          .nil))
      SourceOwners.terminalRecursiveEncodedEqualPath
      (CanonicalContexts.Terminal.afterEncode parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath false) :=
  CanonicalPrimitivePlan.invoke profile recipes
    .encodedEqual
    (.cons
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.freshPublic
        parameters)
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.encoded
          parameters)
        .nil))
    SourceOwners.terminalRecursiveEncodedEqualPath
    (CanonicalContexts.Terminal.afterEncode parameters)
    oneColumn (activationColumn SourceOwners.terminalBranchPath false)
    (CanonicalContexts.Terminal.afterEncodeWidths parameters profile)
    (one_excludes_instruction
      SourceOwners.terminalRecursiveEncodedEqualPath)
    (activation_excludes_instruction SourceOwners.terminalBranchPath
      SourceOwners.terminalRecursiveEncodedEqualPath false)
    (afterEncode_excludes parameters
      SourceOwners.terminalRecursiveEncodedEqualPath
      (by decide) (by decide) (by decide) (by decide))

def recursiveEncodedEqualityPlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    PrimitivePlan parameters profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.encodedEqualCall
        parameters)
      SourceOwners.terminalRecursiveEncodedEqualPath
      (CanonicalContexts.Terminal.afterEncode parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath false) :=
  .invoke (recursiveEncodedEqualityInvokePlan parameters profile recipes)

def recursivePriorAssertionPlan
    (parameters : Parameters)
    (profile : Profile parameters) :
    PrimitivePlan parameters profile
      (Primitive.assertTrue (signature := signature parameters)
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.encodedEqual
          parameters))
      SourceOwners.terminalRecursivePriorAssertionPath
      (CanonicalContexts.Terminal.afterEncodedEquality parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath false) :=
  .assertTrue
    (CanonicalPrimitivePlan.assertion profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.encodedEqual
        parameters)
      SourceOwners.terminalRecursivePriorAssertionPath
      (CanonicalContexts.Terminal.afterEncodedEquality parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath false)
      (CanonicalContexts.Terminal.afterEncodedEqualityWidths
        parameters profile))

def recursiveRunningCheckInvokePlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    InvokePlan parameters profile .runningCheck
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.running
          parameters)
        (.cons
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.runningWitness
            parameters)
          .nil))
      SourceOwners.terminalRecursiveRunningCheckPath
      (CanonicalContexts.Terminal.afterEncodedEquality parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath false) :=
  CanonicalPrimitivePlan.invoke profile recipes
    .runningCheck
    (.cons
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.running
        parameters)
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.runningWitness
          parameters)
        .nil))
    SourceOwners.terminalRecursiveRunningCheckPath
    (CanonicalContexts.Terminal.afterEncodedEquality parameters)
    oneColumn (activationColumn SourceOwners.terminalBranchPath false)
    (CanonicalContexts.Terminal.afterEncodedEqualityWidths
      parameters profile)
    (one_excludes_instruction
      SourceOwners.terminalRecursiveRunningCheckPath)
    (activation_excludes_instruction SourceOwners.terminalBranchPath
      SourceOwners.terminalRecursiveRunningCheckPath false)
    (afterEncodedEquality_excludes parameters
      SourceOwners.terminalRecursiveRunningCheckPath
      (by decide) (by decide) (by decide) (by decide) (by decide))

def recursiveRunningCheckPlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    PrimitivePlan parameters profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.runningCheckCall
        parameters)
      SourceOwners.terminalRecursiveRunningCheckPath
      (CanonicalContexts.Terminal.afterEncodedEquality parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath false) :=
  .invoke (recursiveRunningCheckInvokePlan parameters profile recipes)

def recursiveRunningAssertionPlan
    (parameters : Parameters)
    (profile : Profile parameters) :
    PrimitivePlan parameters profile
      (Primitive.assertTrue (signature := signature parameters)
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.runningAccepted
          parameters))
      SourceOwners.terminalRecursiveRunningAssertionPath
      (CanonicalContexts.Terminal.afterRunningCheck parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath false) :=
  .assertTrue
    (CanonicalPrimitivePlan.assertion profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.runningAccepted
        parameters)
      SourceOwners.terminalRecursiveRunningAssertionPath
      (CanonicalContexts.Terminal.afterRunningCheck parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath false)
      (CanonicalContexts.Terminal.afterRunningCheckWidths
        parameters profile))

def recursiveFreshCheckInvokePlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    InvokePlan parameters profile .freshCheck
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.fresh
          parameters)
        (.cons
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.freshWitness
            parameters)
          .nil))
      SourceOwners.terminalRecursiveFreshCheckPath
      (CanonicalContexts.Terminal.afterRunningCheck parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath false) :=
  CanonicalPrimitivePlan.invoke profile recipes
    .freshCheck
    (.cons
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.fresh
        parameters)
      (.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.freshWitness
          parameters)
        .nil))
    SourceOwners.terminalRecursiveFreshCheckPath
    (CanonicalContexts.Terminal.afterRunningCheck parameters)
    oneColumn (activationColumn SourceOwners.terminalBranchPath false)
    (CanonicalContexts.Terminal.afterRunningCheckWidths
      parameters profile)
    (one_excludes_instruction
      SourceOwners.terminalRecursiveFreshCheckPath)
    (activation_excludes_instruction SourceOwners.terminalBranchPath
      SourceOwners.terminalRecursiveFreshCheckPath false)
    (afterRunningCheck_excludes parameters
      SourceOwners.terminalRecursiveFreshCheckPath
      (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide))

def recursiveFreshCheckPlan
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    PrimitivePlan parameters profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.freshCheckCall
        parameters)
      SourceOwners.terminalRecursiveFreshCheckPath
      (CanonicalContexts.Terminal.afterRunningCheck parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath false) :=
  .invoke (recursiveFreshCheckInvokePlan parameters profile recipes)

def recursiveFreshAssertionPlan
    (parameters : Parameters)
    (profile : Profile parameters) :
    PrimitivePlan parameters profile
      (Primitive.assertTrue (signature := signature parameters)
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.freshAccepted
          parameters))
      SourceOwners.terminalRecursiveFreshAssertionPath
      (CanonicalContexts.Terminal.afterFreshCheck parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath false) :=
  .assertTrue
    (CanonicalPrimitivePlan.assertion profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.RecursiveRef.freshAccepted
        parameters)
      SourceOwners.terminalRecursiveFreshAssertionPath
      (CanonicalContexts.Terminal.afterFreshCheck parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath false)
      (CanonicalContexts.Terminal.afterFreshCheckWidths
        parameters profile))

/-- Exact non-input receipt order induced by the Terminal AST. -/
def bodyReceipts
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    List InstructionReceipt :=
  [ (selectorPlan.{0} parameters profile recipes).receipt,
    CanonicalBranchPlan.trueActivationReceipt
      SourceOwners.terminalBranchPath oneColumn oneColumn
      (CanonicalContexts.Terminal.selector parameters profile),
    CanonicalBranchPlan.falseActivationReceipt
      SourceOwners.terminalBranchPath oneColumn oneColumn
      (CanonicalContexts.Terminal.selector parameters profile),
    (baseEqualityPlan.{0} parameters profile recipes).receipt,
    (baseAssertionPlan.{0} parameters profile).receipt,
    (recursiveHashPlan.{0} parameters profile recipes).receipt,
    (recursiveFreshPublicPlan.{0} parameters profile recipes).receipt,
    (recursiveEncodePlan.{0} parameters profile recipes).receipt,
    (recursiveEncodedEqualityPlan.{0}
      parameters profile recipes).receipt,
    (recursivePriorAssertionPlan.{0} parameters profile).receipt,
    (recursiveRunningCheckPlan.{0} parameters profile recipes).receipt,
    (recursiveRunningAssertionPlan.{0} parameters profile).receipt,
    (recursiveFreshCheckPlan.{0} parameters profile recipes).receipt,
    (recursiveFreshAssertionPlan.{0} parameters profile).receipt,
    CanonicalBranchPlan.emptyJoinReceipt
      SourceOwners.terminalBranchPath ]

/-- Prelude, exact input receipts, and the exact Terminal body order. -/
def receipts
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    List InstructionReceipt :=
  InstructionReceipt.prelude ::
    InputReceipts.receipts (terminalInputSchema parameters) ++
      bodyReceipts parameters profile recipes

end CanonicalTerminalPlan

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
