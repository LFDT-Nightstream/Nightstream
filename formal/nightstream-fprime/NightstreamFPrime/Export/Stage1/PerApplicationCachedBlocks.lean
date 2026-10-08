import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransport
import NightstreamFPrime.Export.Stage1.PerApplicationCachedShift

/-!
Owns the executable form of the assignment transport that the emitter writes:
the canonical block plans, the Phi81 value sources, and the output-digest
expressions.

Some of their columns lie past the constant column, so they add the
application's private-column count. The generic forms recompute that count,
and with it the whole application circuit, for each such column. These forms
compute it once. The generic definitions stay the authority; each `csimp`
theorem is the bridge.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationCachedBlocks

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open PerApplicationAssignmentBlocks
open PerApplicationAssignmentPlan
open PerApplicationAssignmentTransport

/-- The plan of a field block whose slot `k` reads Spartan source column
`start + k`. -/
def fieldPlan (context : PerApplicationCachedShift.Context) (kind : BlockKind)
    (count start : Nat) : BlockPlan :=
  { opcode := kind
    slotKind := .field
    slotCount := count
    sourceDomain := .retained
    sourceRuns := AffineRuns.compressIndexedTR fun slot : Fin count =>
      context.column (Spartan.sourceToSpartan (start + slot.val)) }

/-- The three field blocks that read public columns use the cached count. -/
def cachedOfKind (application : Program)
    (context : PerApplicationCachedShift.Context) : BlockKind → BlockPlan
  | .piCcsFreshPublicInput =>
      fieldPlan context .piCcsFreshPublicInput 270 PilotProduction.priorPublicInputStart
  | .piCcsExpectedContext =>
      fieldPlan context .piCcsExpectedContext PiCCSInputs.expectedContextWords
        PiCCSInputs.expectedContextStart
  | .pilotOutputDigest =>
      fieldPlan context .pilotOutputDigest 4 PilotProduction.outputDigestStart
  | kind => BlockPlan.ofKind application kind

def cachedCanonical (application : Program) : List BlockPlan :=
  let context := PerApplicationCachedShift.Context.ofProgram application
  canonicalKinds.map (cachedOfKind application context)

theorem cachedOfKind_eq (application : Program) (kind : BlockKind) :
    cachedOfKind application (PerApplicationCachedShift.Context.ofProgram application) kind =
      BlockPlan.ofKind application kind := by
  cases kind
  case piCcsFreshPublicInput | piCcsExpectedContext | pilotOutputDigest =>
    rw [BlockPlan.ofKind, ← directSourceRunsFor_eq_sourceRunsFor, directSourceRunsFor]
    refine congrArg (BlockPlan.mk _ _ _ _)
      (congrArg AffineRuns.compressIndexedTR (funext fun slot => ?_))
    exact PerApplicationCachedShift.Context.column_eq_shiftColumn _ _
  all_goals rfl

@[csimp] theorem canonical_eq_cachedCanonical :
    @PerApplicationAssignmentBlocks.canonical = @cachedCanonical := by
  funext application
  unfold PerApplicationAssignmentBlocks.canonical cachedCanonical
  exact List.map_congr_left fun kind _ => (cachedOfKind_eq application kind).symm

def cachedPhi81QuotientRecipe (program : Program) : Phi81QuotientRecipe :=
  let context := PerApplicationCachedShift.Context.ofProgram program
  { ringDegree := 54
    middleDegree := 27
    foldOffset := 81
    quotientCount := 54
    familyShapes := phi81FamilyShapes
    challengeBlock := .challengeWords
    challengeSlotBase := 0
    challengeSourceStride := 54
    challengeShift := 2
    valueSources := AffineRuns.compressIndexedTR fun invocation :
        Fin PiRLCProductSchedule.invocationCount =>
      let descriptor := PiRLCProductSchedule.descriptor invocation
      context.column (Spartan.sourceToSpartan (descriptor.valueColumn descriptor.lane))
    quotientOutputBlock := .productGroup }

@[csimp] theorem phi81QuotientRecipe_eq_cached :
    @phi81QuotientRecipe = @cachedPhi81QuotientRecipe := by
  funext program
  rw [phi81QuotientRecipe, cachedPhi81QuotientRecipe, phi81ValueSources_eq_direct,
    directPhi81ValueSources]
  refine congrArg (fun sources => Phi81QuotientRecipe.mk _ _ _ _ _ _ _ _ _ sources _)
    (congrArg AffineRuns.compressIndexedTR (funext fun invocation => ?_))
  rw [PerApplicationCachedShift.Context.column_eq_shiftColumn]
  rfl

def cachedOutputDigestExpressions (program : Program) : List Circuit.Expr :=
  let context := PerApplicationCachedShift.Context.ofProgram program
  List.ofFn fun lane : Fin 4 =>
    PerApplicationCachedShift.shiftExpr context <|
      PermutationOutput.Readout.rewriteExpr PiCCSTranscriptReadout.phaseStart
        PiCCSOrdinarySourceSupport.transcriptInvocationCount <|
          CompactRows.renameExpr Spartan.sourceToSpartan <|
            PilotProduction.outputInterface.digest
              (Lifecycle.Pilot.outputOffset PilotProduction.interface
                PilotProduction.witnessOffset) lane

@[csimp] theorem outputDigestExpressions_eq_cached :
    @outputDigestExpressions = @cachedOutputDigestExpressions := by
  funext program
  unfold outputDigestExpressions cachedOutputDigestExpressions
  congr 1

end NightstreamFPrime.Export.Stage1.PerApplicationCachedBlocks
