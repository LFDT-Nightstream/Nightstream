import NightstreamFPrime.Export.Stage1.PermutationPlan
import NightstreamFPrime.Export.Stage1.WitnessPlan

/-!
Owns the compact, Lean-authored emission plan for the Stage 1 package.

The semantic package remains `Data.circuitPackage`. The plan removes repeated
invocation records from the artifact. Its expansion theorem fixes all bounds,
ordering, row starts, local starts, templates, and input ranges in Lean.
-/

namespace NightstreamFPrime.Export.Stage1.PackagePlan

open NightstreamFPrime.Export
open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Package

/-- One indexed block for the repeated PiRLC `First54` invocations. -/
structure First54InvocationBlock where
  sourceCount : Nat
  roundCount : Nat
deriving Repr

def First54InvocationBlock.format : Format First54InvocationBlock where
  encode := fun value => .array [
    .atom value.sourceCount,
    .atom value.roundCount]
  decode
    | .array [.atom sourceCount, .atom roundCount] =>
      .ok ⟨sourceCount, roundCount⟩
    | _ => .error "invalid First54 invocation block"
  decode_encode := by
    intro value
    cases value
    rfl

def First54InvocationBlock.expand
    (block : First54InvocationBlock) : List CompactRowInvocation :=
  (List.range block.sourceCount).flatMap fun source =>
    (List.range block.roundCount).flatMap
      (PiRLCFirst54Invocations.roundInvocations source)

def canonicalFirst54Block : First54InvocationBlock where
  sourceCount := PiRLCFirst54Invocations.sourceCount
  roundCount := PiRLCFirst54Invocations.roundCount

theorem canonicalFirst54Block_expand :
    canonicalFirst54Block.expand =
      PiRLCFirst54Invocations.invocations := by
  rfl

/-- Exact public parameters for one indexed PiRLC combination family. -/
structure CombinationFamilyBlock where
  logicalStart : Nat
  rowStart : Nat
  freshStart : Nat
  blockCount : Nat
  cellCount : Nat
  valueStride : Nat
deriving Repr

def CombinationFamilyBlock.format : Format CombinationFamilyBlock where
  encode := fun value => .array [
    .atom value.logicalStart,
    .atom value.rowStart,
    .atom value.freshStart,
    .atom value.blockCount,
    .atom value.cellCount,
    .atom value.valueStride]
  decode
    | .array [.atom logicalStart, .atom rowStart, .atom freshStart,
        .atom blockCount, .atom cellCount, .atom valueStride] =>
      .ok ⟨logicalStart, rowStart, freshStart, blockCount, cellCount,
        valueStride⟩
    | _ => .error "invalid combination family block"
  decode_encode := by
    intro value
    cases value
    rfl

/-- The four distinct combination families in their fixed package order. -/
structure CombinationInvocationBlock where
  sourceCount : Nat
  commitment : CombinationFamilyBlock
  publicInput : CombinationFamilyBlock
  evalK : CombinationFamilyBlock
  evalA : CombinationFamilyBlock
deriving Repr

def CombinationInvocationBlock.format : Format CombinationInvocationBlock where
  encode := fun value => .array [
    .atom value.sourceCount,
    CombinationFamilyBlock.format.encode value.commitment,
    CombinationFamilyBlock.format.encode value.publicInput,
    CombinationFamilyBlock.format.encode value.evalK,
    CombinationFamilyBlock.format.encode value.evalA]
  decode
    | .array [.atom sourceCount, commitment, publicInput, evalK, evalA] => do
      pure ⟨sourceCount,
        ← CombinationFamilyBlock.format.decode commitment,
        ← CombinationFamilyBlock.format.decode publicInput,
        ← CombinationFamilyBlock.format.decode evalK,
        ← CombinationFamilyBlock.format.decode evalA⟩
    | _ => .error "invalid combination invocation block"
  decode_encode := by
    intro value
    cases value
    simp only
    rw [CombinationFamilyBlock.format.decode_encode,
      CombinationFamilyBlock.format.decode_encode,
      CombinationFamilyBlock.format.decode_encode,
      CombinationFamilyBlock.format.decode_encode]
    rfl

def expandCombinationFamily (sourceCount : Nat)
    (block : CombinationFamilyBlock)
    (valueSourceStart : Nat → Nat → Nat → Nat) :
    List CompactRowInvocation :=
  (List.range sourceCount).flatMap fun source =>
    List.ofFn fun index : Fin
        (NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.privateCount
          block.blockCount block.cellCount) =>
      let coordinates :=
        NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.coordinates
          index
      PiRLCCombinationInvocations.invocation block.logicalStart
        block.rowStart block.freshStart block.blockCount block.cellCount
        block.valueStride source coordinates.1.val coordinates.2.1.val
          coordinates.2.2.val valueSourceStart

private theorem expandCombinationFamily_eq_familyInvocations
    (block : CombinationFamilyBlock)
    (valueSourceStart : Nat → Nat → Nat → Nat) :
    expandCombinationFamily PiRLCCombinationInvocations.sourceCount block
        valueSourceStart =
      PiRLCCombinationInvocations.familyInvocations block.logicalStart
        block.rowStart block.freshStart block.blockCount block.cellCount
        block.valueStride valueSourceStart := by
  rfl

def CombinationInvocationBlock.expand
    (block : CombinationInvocationBlock) : List CompactRowInvocation :=
  expandCombinationFamily block.sourceCount block.commitment
      PiRLCCombinationInvocations.commitmentValueSourceStart ++
    expandCombinationFamily block.sourceCount block.publicInput
      PiRLCCombinationInvocations.publicInputValueSourceStart ++
    expandCombinationFamily block.sourceCount block.evalK
      PiRLCCombinationInvocations.evalKValueSourceStart ++
    expandCombinationFamily block.sourceCount block.evalA
      PiRLCCombinationInvocations.evalAValueSourceStart

def canonicalCombinationBlock : CombinationInvocationBlock where
  sourceCount := PiRLCCombinationInvocations.sourceCount
  commitment := ⟨
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.commitmentLogicalStart,
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.commitmentRowStart,
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.commitmentFreshStart,
    22, 1, 1⟩
  publicInput := ⟨
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.publicInputLogicalStart,
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.publicInputRowStart,
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.publicInputFreshStart,
    5, 1, 1⟩
  evalK := ⟨
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.evalKLogicalStart,
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.evalKRowStart,
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.evalKFreshStart,
    1, 2, 2⟩
  evalA := ⟨
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.evalALogicalStart,
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.evalARowStart,
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.evalAFreshStart,
    14, 2, 2⟩

theorem canonicalCombinationBlock_expand :
    canonicalCombinationBlock.expand =
      PiRLCCombinationInvocations.invocations := by
  unfold CombinationInvocationBlock.expand
  dsimp only [canonicalCombinationBlock]
  rw [expandCombinationFamily_eq_familyInvocations,
    expandCombinationFamily_eq_familyInvocations,
    expandCombinationFamily_eq_familyInvocations,
    expandCombinationFamily_eq_familyInvocations]
  change PiRLCCombinationInvocations.commitmentInvocations ++
      PiRLCCombinationInvocations.publicInputInvocations ++
      PiRLCCombinationInvocations.evalKInvocations ++
      PiRLCCombinationInvocations.evalAInvocations =
    PiRLCCombinationInvocations.invocations
  rfl

/-- A compact invocation-plan block. Each tag has one fixed Lean expansion. -/
inductive CompactInvocationBlock where
  | first54 (block : First54InvocationBlock)
  | combination (block : CombinationInvocationBlock)
deriving Repr

def CompactInvocationBlock.expand :
  CompactInvocationBlock → List CompactRowInvocation
  | .first54 block => block.expand
  | .combination block => block.expand

def CompactInvocationBlock.format : Format CompactInvocationBlock where
  encode
    | .first54 block => .array [
        .atom 0,
        First54InvocationBlock.format.encode block]
    | .combination block => .array [
        .atom 1,
        CombinationInvocationBlock.format.encode block]
  decode
    | .array [.atom 0, block] => do
      pure (.first54 (← First54InvocationBlock.format.decode block))
    | .array [.atom 1, block] => do
      pure (.combination (← CombinationInvocationBlock.format.decode block))
    | _ => .error "invalid compact invocation block"
  decode_encode := by
    intro value
    cases value <;> simp [Format.decode_encode]

/-- Schema 8 transports the exact package as static fields plus generative
invocation blocks. `expand` is the semantic interpretation. -/
structure Plan where
  schemaVersion : Nat
  staticPackage : CircuitPackage
  permutationBlocks : List PermutationPlan.Block
  compactInvocationBlocks : List CompactInvocationBlock
  witnessBlocks : List WitnessPlan.Block

def Plan.format : Format Plan where
  encode := fun value => .array [
    .atom value.schemaVersion,
    CircuitPackage.format.encode value.staticPackage,
    (list PermutationPlan.Block.format).encode value.permutationBlocks,
    (list CompactInvocationBlock.format).encode
      value.compactInvocationBlocks,
    (list WitnessPlan.Block.format).encode value.witnessBlocks]
  decode
    | .array [.atom schemaVersion, staticPackage,
        permutationBlocks, compactInvocationBlocks, witnessBlocks] => do
      pure ⟨schemaVersion,
        ← CircuitPackage.format.decode staticPackage,
        ← (list PermutationPlan.Block.format).decode permutationBlocks,
        ← (list CompactInvocationBlock.format).decode
          compactInvocationBlocks,
        ← (list WitnessPlan.Block.format).decode witnessBlocks⟩
    | _ => .error "invalid Stage 1 package plan"
  decode_encode := by
    intro value
    cases value
    simp only
    rw [CircuitPackage.format.decode_encode,
      (list PermutationPlan.Block.format).decode_encode,
      (list CompactInvocationBlock.format).decode_encode,
      (list WitnessPlan.Block.format).decode_encode]
    rfl

def withoutPlannedData (package : CircuitPackage) : CircuitPackage :=
  { package with
    permutationInvocations := []
    compactRowInvocations := []
    witnessBatches :=
      Data.liftPilotBatches (PilotData.priorWordBatches ()) ++
        WitnessProgram.piCcsBatches Data.logicalWidth Data.publicFits }

/-- Build the static payload directly. This avoids constructing either
explicit invocation list only to clear it. -/
def staticComponents (_unit : Unit) : Data.Components :=
  Data.Components.of (Data.arithmeticRows ()) []

def staticPackage (_unit : Unit) : CircuitPackage :=
  Data.circuitPackageOf (Data.arithmeticRows ()) [] []
    (WitnessProgram.piCcsBatches Data.logicalWidth Data.publicFits)

private theorem withoutPlannedData_circuitPackageOf
    (arithmeticRows : List Rows.CompiledRow)
    (permutations : List PermutationInvocation)
    (compact : List CompactRowInvocation)
    (batches : List NightstreamFPrime.Circuit.WitnessBatch) :
    withoutPlannedData
        (Data.circuitPackageOf arithmeticRows permutations compact batches) =
      Data.circuitPackageOf arithmeticRows [] []
        (WitnessProgram.piCcsBatches Data.logicalWidth Data.publicFits) := by
  rfl

theorem staticPackage_eq :
    staticPackage () =
      withoutPlannedData (Data.circuitPackage ()) := by
  symm
  change withoutPlannedData
      (Data.circuitPackageOf (Data.arithmeticRows ())
        (Data.permutationInvocations ()) (Data.compactRowInvocations ())
        (WitnessProgram.batches Data.logicalWidth Data.publicFits)) =
    Data.circuitPackageOf (Data.arithmeticRows ()) [] []
      (WitnessProgram.piCcsBatches Data.logicalWidth Data.publicFits)
  exact withoutPlannedData_circuitPackageOf _ _ _ _

def Plan.expand (plan : Plan) : CircuitPackage :=
  { plan.staticPackage with
    permutationInvocations :=
      plan.permutationBlocks.flatMap PermutationPlan.Block.expand
    compactRowInvocations :=
      plan.compactInvocationBlocks.flatMap
        CompactInvocationBlock.expand
    witnessBatches := plan.staticPackage.witnessBatches ++
      plan.witnessBlocks.flatMap WitnessPlan.Block.expand }

def canonicalCompactBlocks : List CompactInvocationBlock :=
  [.first54 canonicalFirst54Block,
   .combination canonicalCombinationBlock]

def canonical (_unit : Unit) : Plan where
  schemaVersion := 8
  staticPackage := staticPackage ()
  permutationBlocks := PermutationPlan.canonicalBlocks ()
  compactInvocationBlocks := canonicalCompactBlocks
  witnessBlocks :=
    WitnessPlan.canonicalBlocks Data.logicalWidth Data.publicFits

theorem canonicalCompactBlocks_expand :
    canonicalCompactBlocks.flatMap CompactInvocationBlock.expand =
      Data.compactRowInvocations () := by
  rw [Data.compactRowInvocations_eq]
  simp only [canonicalCompactBlocks, List.flatMap_cons, List.flatMap_nil,
    CompactInvocationBlock.expand, List.append_nil]
  exact congrArg₂ (fun left right => left ++ right)
    canonicalFirst54Block_expand canonicalCombinationBlock_expand

private theorem restorePlannedInvocations
    (package : CircuitPackage)
    (permutations : List PermutationInvocation)
    (compact : List CompactRowInvocation)
    (witnesses : List NightstreamFPrime.Circuit.WitnessBatch)
    (permutationEqual : permutations = package.permutationInvocations)
    (compactEqual : compact = package.compactRowInvocations)
    (witnessEqual : witnesses = package.witnessBatches) :
    ({ withoutPlannedData package with
        permutationInvocations := permutations
        compactRowInvocations := compact
        witnessBatches := witnesses } : CircuitPackage) = package := by
  subst permutations
  subst compact
  subst witnesses
  cases package
  rfl

/-- Expanding the emitted plan gives exactly the existing Lean semantic
package. Rust can execute the plan but cannot select any schedule value. -/
theorem canonical_expand : (canonical ()).expand = Data.circuitPackage () := by
  change ({ staticPackage () with
      permutationInvocations :=
        (canonical ()).permutationBlocks.flatMap
          PermutationPlan.Block.expand
      compactRowInvocations :=
        (canonical ()).compactInvocationBlocks.flatMap
          CompactInvocationBlock.expand
      witnessBatches := (staticPackage ()).witnessBatches ++
        (canonical ()).witnessBlocks.flatMap
          WitnessPlan.Block.expand } : CircuitPackage) =
    Data.circuitPackage ()
  rw [staticPackage_eq]
  apply restorePlannedInvocations
  · exact PermutationPlan.canonicalBlocks_expand.trans
      Data.circuitPackage_permutationInvocations.symm
  · calc
      (canonical ()).compactInvocationBlocks.flatMap
            CompactInvocationBlock.expand =
          canonicalCompactBlocks.flatMap
            CompactInvocationBlock.expand := by rfl
      _ = Data.compactRowInvocations () := canonicalCompactBlocks_expand
      _ = (Data.circuitPackage ()).compactRowInvocations :=
        Data.circuitPackage_compactRowInvocations.symm
  · change
      (Data.liftPilotBatches (PilotData.priorWordBatches ()) ++
        WitnessProgram.piCcsBatches Data.logicalWidth Data.publicFits) ++
          (WitnessPlan.canonicalBlocks
            Data.logicalWidth Data.publicFits).flatMap
              WitnessPlan.Block.expand =
        (Data.circuitPackage ()).witnessBatches
    rw [WitnessPlan.canonicalBlocks_expand,
      Data.circuitPackage_witnessBatches,
      WitnessProgram.batches_eq, List.append_assoc]

theorem canonical_decode_encode :
    Plan.format.decode (Plan.format.encode (canonical ())) = .ok (canonical ()) :=
  Plan.format.decode_encode (canonical ())

theorem canonical_expand_relationIdentifier :
    Package.relationIdentifier ((canonical ()).expand) =
      Data.relationIdentifier () := by
  rw [canonical_expand]
  rfl

end NightstreamFPrime.Export.Stage1.PackagePlan
