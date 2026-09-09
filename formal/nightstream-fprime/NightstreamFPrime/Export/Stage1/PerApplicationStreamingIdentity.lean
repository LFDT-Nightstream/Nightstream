import NightstreamFPrime.Export.Stage1.OrdinaryRowPlan
import NightstreamFPrime.Export.Stage1.PackagePlan
import NightstreamFPrime.Export.Stage1.PerApplicationCachedShift
import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage
import NightstreamFPrime.Export.Stage1.PiCCSPackets
import NightstreamFPrime.Export.NativePoseidon2

/-!
Owns the direct bounded-memory traversal of the final Stage 1 sealed package.
The traversal follows the codec order, but it expands large package fields one
proved block at a time. `PerApplicationCanonicalPackage.sealedPackageValue`
and `Package.relationIdentifierValue` remain the semantic authority.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationStreamingIdentity

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec

abbrev Program := Lifecycle.Stage1.Application.Program
abbrev FitsTwoPow28 (program : Program) :=
  PerApplicationCanonicalPackage.FitsTwoPow28 program

private theorem map_eq_of_pointwise {Alpha Beta : Type}
    (values : List Alpha) (left right : Alpha → Beta)
    (equal : ∀ value, left value = right value) :
    values.map left = values.map right := by
  induction values with
  | nil => rfl
  | cons value rest inductionHypothesis =>
      simp only [List.map_cons, equal value, inductionHypothesis]

private theorem map_compose {Alpha Beta Gamma : Type}
    (values : List Alpha) (first : Alpha → Beta) (second : Beta → Gamma) :
    values.map (fun value => second (first value)) =
      (values.map first).map second := by
  induction values with
  | nil => rfl
  | cons value rest inductionHypothesis =>
      simp only [List.map_cons, inductionHypothesis]

private theorem map_liftPilotInstructions_append
    (context : PerApplicationCachedShift.Context)
    (pilot rest : List WitnessInstruction) :
    (Data.liftPilotInstructions pilot ++ rest).map
        (PerApplicationCachedShift.shiftWitnessInstruction context) =
      pilot.map (fun instruction =>
        PerApplicationCachedShift.shiftWitnessInstruction context
          (Data.liftPilotInstruction instruction)) ++
        rest.map
          (PerApplicationCachedShift.shiftWitnessInstruction context) := by
  unfold Data.liftPilotInstructions
  rw [List.map_append, List.map_map]
  rfl

/-! ## Generic mapped streams -/

/-- Process a mapped typed list without constructing the mapped list. -/
@[inline, specialize push format transform] def processMappedEncodedItemsWith
    {State Alpha Beta : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (format : Format Beta) (transform : Alpha → Beta) : List Alpha → State
  | [] => state
  | value :: rest =>
      processMappedEncodedItemsWith push
        (StreamingIdentity.processValueWith push
          (format.encode (transform value)) state)
        format transform rest

theorem processMappedEncodedItemsWith_eq_items
    {State Alpha Beta : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (format : Format Beta) (transform : Alpha → Beta)
    (values : List Alpha) :
    processMappedEncodedItemsWith push state format transform values =
      StreamingIdentity.processEncodedItemsWith push state format
        (values.map transform) := by
  induction values generalizing state with
  | nil => rfl
  | cons value rest inductionHypothesis =>
      simp [processMappedEncodedItemsWith,
        StreamingIdentity.processEncodedItemsWith, inductionHypothesis]

/-- Process one mapped typed list under one array header. -/
@[inline, specialize push format transform] def processMappedEncodedListWith
    {State Alpha Beta : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (format : Format Beta) (transform : Alpha → Beta)
    (values : List Alpha) : State :=
  processMappedEncodedItemsWith push
    (push state ⟨1, values.length⟩) format transform values

theorem processMappedEncodedListWith_eq_processValueWith
    {State Alpha Beta : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (format : Format Beta) (transform : Alpha → Beta)
    (values : List Alpha) :
    processMappedEncodedListWith push state format transform values =
      StreamingIdentity.processValueWith push
        ((list format).encode (values.map transform)) state := by
  unfold processMappedEncodedListWith
  rw [processMappedEncodedItemsWith_eq_items]
  rw [show values.length = (values.map transform).length by simp]
  exact StreamingIdentity.processEncodedListWith_eq_processValueWith
    push state format (values.map transform)

/-- Process mapped block expansions without constructing either the flattened
expansion or its mapped result. -/
@[inline, specialize push format transform expand]
def processMappedFlatMapItemsWith {State Block Alpha Beta : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (format : Format Beta) (transform : Alpha → Beta)
    (expand : Block → List Alpha) : List Block → State
  | [] => state
  | block :: rest =>
      processMappedFlatMapItemsWith push
        (processMappedEncodedItemsWith push state format transform
          (expand block))
        format transform expand rest

theorem processMappedFlatMapItemsWith_eq_items
    {State Block Alpha Beta : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (format : Format Beta) (transform : Alpha → Beta)
    (expand : Block → List Alpha) (blocks : List Block) :
    processMappedFlatMapItemsWith push state format transform expand blocks =
      StreamingIdentity.processEncodedItemsWith push state format
        ((blocks.flatMap expand).map transform) := by
  induction blocks generalizing state with
  | nil => rfl
  | cons block rest inductionHypothesis =>
      rw [processMappedFlatMapItemsWith,
        processMappedEncodedItemsWith_eq_items, inductionHypothesis,
        List.flatMap_cons, List.map_append,
        StreamingIdentity.processEncodedItemsWith_append]

/-- Process mapped block expansions under one array header. -/
@[inline, specialize push format transform expand]
def processMappedFlatMapWith {State Block Alpha Beta : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (format : Format Beta) (transform : Alpha → Beta)
    (expand : Block → List Alpha) (blocks : List Block) : State :=
  processMappedFlatMapItemsWith push
    (push state ⟨1, StreamingIdentity.encodedFlatMapLength expand blocks⟩)
    format transform expand blocks

theorem processMappedFlatMapWith_eq_processValueWith
    {State Block Alpha Beta : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (format : Format Beta) (transform : Alpha → Beta)
    (expand : Block → List Alpha) (blocks : List Block) :
    processMappedFlatMapWith push state format transform expand blocks =
      StreamingIdentity.processValueWith push
        ((list format).encode ((blocks.flatMap expand).map transform)) state := by
  unfold processMappedFlatMapWith
  rw [processMappedFlatMapItemsWith_eq_items,
    StreamingIdentity.encodedFlatMapLength_eq_flatMap_length]
  rw [show (blocks.flatMap expand).length =
      ((blocks.flatMap expand).map transform).length by simp]
  exact StreamingIdentity.processEncodedListWith_eq_processValueWith
    push state format ((blocks.flatMap expand).map transform)

private theorem processEncodedItemsWith_header_eq_processValueWith
    {State Alpha : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (format : Format Alpha) (values : List Alpha) :
    StreamingIdentity.processEncodedItemsWith push
        (push state ⟨1, values.length⟩) format values =
      StreamingIdentity.processValueWith push
        ((list format).encode values) state := by
  simpa only [StreamingIdentity.processEncodedListWith] using
    (StreamingIdentity.processEncodedListWith_eq_processValueWith
      push state format values)

/-! ## Proof-only flattened package views -/

def piCcsPacketBatches (_ : Unit) : List WitnessBatch :=
  (PiCCSPackets.initialClaim Data.logicalWidth Data.publicFits).batches ++
    ((PiCCSPackets.sumcheck Data.logicalWidth Data.publicFits).batches ++
    ((PiCCSPackets.evalK Data.logicalWidth Data.publicFits).batches ++
    ((PiCCSPackets.evalA Data.logicalWidth Data.publicFits).batches ++
    ((PiCCSPackets.ccs Data.logicalWidth Data.publicFits).batches ++
    ((PiCCSPackets.norm Data.logicalWidth Data.publicFits).batches ++
      (PiCCSPackets.finalIdentity
        Data.logicalWidth Data.publicFits).batches)))))

theorem piCcsPacketBatches_eq :
    piCcsPacketBatches () =
      WitnessProgram.piCcsBatches Data.logicalWidth Data.publicFits := by
  unfold piCcsPacketBatches WitnessProgram.piCcsBatches
  rw [PiCCSPackets.initialClaim_batches, PiCCSPackets.sumcheck_batches,
    PiCCSPackets.evalK_batches, PiCCSPackets.evalA_batches,
    PiCCSPackets.ccs_batches, PiCCSPackets.norm_batches,
    PiCCSPackets.finalIdentity_batches]
  simp only [List.append_assoc]

def directBaseWitnessBatches (_ : Unit) : List WitnessBatch :=
  Data.liftPilotBatches (PilotData.priorWordBatches ()) ++
    (piCcsPacketBatches () ++
      (WitnessPlan.canonicalBlocks
        Data.logicalWidth Data.publicFits).flatMap WitnessPlan.Block.expand)

theorem directBaseWitnessBatches_eq :
    directBaseWitnessBatches () = (Data.circuitPackage ()).witnessBatches := by
  unfold directBaseWitnessBatches
  rw [piCcsPacketBatches_eq, WitnessPlan.canonicalBlocks_expand,
    Data.circuitPackage_witnessBatches, WitnessProgram.batches_eq]

def directBaseWitnessInstructions (_ : Unit) : List WitnessInstruction :=
  Data.liftPilotInstructions (PilotData.witnessInstructions ()) ++
    (OrdinaryRowPlan.canonicalBlocks ()).flatMap fun block =>
      Rows.witnessInstructionsTR
        (block.rows Data.logicalWidth Data.publicFits)

theorem directBaseWitnessInstructions_eq :
    directBaseWitnessInstructions () =
      (Data.circuitPackage ()).witnessInstructions := by
  unfold directBaseWitnessInstructions
  rw [OrdinaryRowPlan.canonicalWitnessInstructions_expand,
    Data.circuitPackage_witnessInstructions]
  unfold Data.Components.witnessInstructions
  rw [Data.components_arithmeticRows]

def directBaseAssertionRows (_ : Unit) : List SparseRow :=
  Data.liftPilotRows (PilotData.assertionRows ()) ++
    (OrdinaryRowPlan.canonicalBlocks ()).flatMap fun block =>
      Rows.assertionRowsTR (block.rows Data.logicalWidth Data.publicFits)

theorem directBaseAssertionRows_eq :
    directBaseAssertionRows () = (Data.circuitPackage ()).assertionRows := by
  unfold directBaseAssertionRows
  rw [OrdinaryRowPlan.canonicalAssertionRows_expand,
    Data.circuitPackage_assertionRows]
  unfold Data.Components.assertionRows Data.Components.arithmeticAssertionRows
  rw [Data.components_arithmeticRows]

def directHashChains (context : PerApplicationCachedShift.Context) :
    List HashChain :=
  [Data.priorChain, Data.outputChain].map
    (PerApplicationCachedShift.shiftHashChain context)

theorem directHashChains_eq (context : PerApplicationCachedShift.Context) :
    directHashChains context =
      (PerApplicationPackage.package context.program).hashChains := by
  calc
    directHashChains context =
        [Data.priorChain, Data.outputChain].map
          (PerApplicationPackage.shiftHashChain context.program) := by
      unfold directHashChains
      exact map_eq_of_pointwise _ _ _
        (PerApplicationCachedShift.shiftHashChain_eq context)
    _ = (Data.circuitPackage ()).hashChains.map
          (PerApplicationPackage.shiftHashChain context.program) := by
      rw [Data.circuitPackage_hashChains]
    _ = (PerApplicationPackage.package context.program).hashChains := by
      simpa only [PerApplicationPackage.basePackage] using
        (PerApplicationPackage.package_hashChains context.program).symm

def directPermutationInvocations
    (context : PerApplicationCachedShift.Context) :
    List PermutationInvocation :=
  ((PermutationPlan.canonicalBlocks ()).flatMap
      PermutationPlan.Block.expand).map
    (PerApplicationCachedShift.shiftPermutationInvocation context)

theorem directPermutationInvocations_eq
    (context : PerApplicationCachedShift.Context) :
    directPermutationInvocations context =
      (PerApplicationPackage.package context.program).permutationInvocations := by
  calc
    directPermutationInvocations context =
        ((PermutationPlan.canonicalBlocks ()).flatMap
          PermutationPlan.Block.expand).map
            (PerApplicationPackage.shiftPermutationInvocation
              context.program) := by
      unfold directPermutationInvocations
      exact map_eq_of_pointwise _ _ _
        (PerApplicationCachedShift.shiftPermutationInvocation_eq context)
    _ = (Data.permutationInvocations ()).map
          (PerApplicationPackage.shiftPermutationInvocation
            context.program) := by
      rw [PermutationPlan.canonicalBlocks_expand]
    _ = (Data.circuitPackage ()).permutationInvocations.map
          (PerApplicationPackage.shiftPermutationInvocation
            context.program) := by
      rw [Data.circuitPackage_permutationInvocations,
        Data.components_permutationInvocations]
    _ = (PerApplicationPackage.package
          context.program).permutationInvocations := by
      simpa only [PerApplicationPackage.basePackage] using
        (PerApplicationPackage.package_permutationInvocations
          context.program).symm

def directCompactRowInvocations
    (context : PerApplicationCachedShift.Context) :
    List CompactRowInvocation :=
  (PackagePlan.canonicalCompactBlocks.flatMap
      PackagePlan.CompactInvocationBlock.expand).map
    (PerApplicationCachedShift.shiftCompactRowInvocation context)

theorem directCompactRowInvocations_eq
    (context : PerApplicationCachedShift.Context) :
    directCompactRowInvocations context =
      (PerApplicationPackage.package context.program).compactRowInvocations := by
  calc
    directCompactRowInvocations context =
        (PackagePlan.canonicalCompactBlocks.flatMap
          PackagePlan.CompactInvocationBlock.expand).map
            (PerApplicationPackage.shiftCompactRowInvocation
              context.program) := by
      unfold directCompactRowInvocations
      exact map_eq_of_pointwise _ _ _
        (PerApplicationCachedShift.shiftCompactRowInvocation_eq context)
    _ = (Data.compactRowInvocations ()).map
          (PerApplicationPackage.shiftCompactRowInvocation
            context.program) := by
      rw [PackagePlan.canonicalCompactBlocks_expand]
    _ = (Data.circuitPackage ()).compactRowInvocations.map
          (PerApplicationPackage.shiftCompactRowInvocation
            context.program) := by
      rw [Data.circuitPackage_compactRowInvocations]
    _ = (PerApplicationPackage.package
          context.program).compactRowInvocations := by
      simpa only [PerApplicationPackage.basePackage] using
        (PerApplicationPackage.package_compactRowInvocations
          context.program).symm

def directWitnessBatches (context : PerApplicationCachedShift.Context)
    (application : ApplicationPackage.Plan) : List WitnessBatch :=
  (directBaseWitnessBatches ()).map
      (PerApplicationCachedShift.shiftBatch context) ++
    application.witnessBatches

theorem directWitnessBatches_eq
    (context : PerApplicationCachedShift.Context) :
    directWitnessBatches context
        (PerApplicationPackage.directApplicationPlan context.program) =
      (PerApplicationPackage.package context.program).witnessBatches := by
  calc
    directWitnessBatches context
        (PerApplicationPackage.directApplicationPlan context.program) =
      (directBaseWitnessBatches ()).map
          (PerApplicationPackage.shiftBatch context.program) ++
        (PerApplicationPackage.directApplicationPlan
          context.program).witnessBatches := by
      unfold directWitnessBatches
      exact congrArg₂ (fun left right => left ++ right)
        (map_eq_of_pointwise _ _ _
          (PerApplicationCachedShift.shiftBatch_eq context)) rfl
    _ = (Data.circuitPackage ()).witnessBatches.map
          (PerApplicationPackage.shiftBatch context.program) ++
        (PerApplicationPackage.applicationPlan
          context.program).witnessBatches := by
      rw [directBaseWitnessBatches_eq,
        PerApplicationPackage.directApplicationPlan_eq_applicationPlan]
    _ = (PerApplicationPackage.package context.program).witnessBatches := by
      simpa only [PerApplicationPackage.basePackage] using
        (PerApplicationPackage.package_witnessBatches context.program).symm

def directWitnessInstructions (context : PerApplicationCachedShift.Context)
    (application : ApplicationPackage.Plan) : List WitnessInstruction :=
  (directBaseWitnessInstructions ()).map
      (PerApplicationCachedShift.shiftWitnessInstruction context) ++
    application.witnessInstructions

private theorem directWitnessInstructions_segments
    (context : PerApplicationCachedShift.Context)
    (application : ApplicationPackage.Plan) :
    directWitnessInstructions context application =
      (PilotData.witnessInstructions ()).map (fun instruction =>
          PerApplicationCachedShift.shiftWitnessInstruction context
            (Data.liftPilotInstruction instruction)) ++
        (((OrdinaryRowPlan.canonicalBlocks ()).flatMap fun block =>
          Rows.witnessInstructionsTR
            (block.rows Data.logicalWidth Data.publicFits)).map
              (PerApplicationCachedShift.shiftWitnessInstruction context) ++
          application.witnessInstructions) := by
  unfold directWitnessInstructions directBaseWitnessInstructions
  rw [map_liftPilotInstructions_append]
  exact List.append_assoc _ _ _

theorem directWitnessInstructions_eq
    (context : PerApplicationCachedShift.Context) :
    directWitnessInstructions context
        (PerApplicationPackage.directApplicationPlan context.program) =
      (PerApplicationPackage.package context.program).witnessInstructions := by
  calc
    directWitnessInstructions context
        (PerApplicationPackage.directApplicationPlan context.program) =
      (directBaseWitnessInstructions ()).map
          (PerApplicationPackage.shiftWitnessInstruction context.program) ++
        (PerApplicationPackage.directApplicationPlan
          context.program).witnessInstructions := by
      unfold directWitnessInstructions
      exact congrArg₂ (fun left right => left ++ right)
        (map_eq_of_pointwise _ _ _
          (PerApplicationCachedShift.shiftWitnessInstruction_eq context)) rfl
    _ = (Data.circuitPackage ()).witnessInstructions.map
          (PerApplicationPackage.shiftWitnessInstruction context.program) ++
        (PerApplicationPackage.applicationPlan
          context.program).witnessInstructions := by
      rw [directBaseWitnessInstructions_eq,
        PerApplicationPackage.directApplicationPlan_eq_applicationPlan]
    _ = (PerApplicationPackage.package
          context.program).witnessInstructions := by
      simpa only [PerApplicationPackage.basePackage] using
        (PerApplicationPackage.package_witnessInstructions
          context.program).symm

def directAssertionRows (context : PerApplicationCachedShift.Context)
    (application : ApplicationPackage.Plan) : List SparseRow :=
  ((directBaseAssertionRows ()).map
      (PerApplicationCachedShift.shiftSparseRow context) ++
    application.assertionRows) ++
    NextPreimagePackage.assertionRows
      (PerApplicationPackage.directNextPreimageRowStart context.program)

private theorem directAssertionRows_segments
    (context : PerApplicationCachedShift.Context)
    (application : ApplicationPackage.Plan) :
    directAssertionRows context application =
      (PilotData.assertionRows ()).map (fun row =>
          PerApplicationCachedShift.shiftSparseRow context
            (Data.liftPilotRow row)) ++
        ((((OrdinaryRowPlan.canonicalBlocks ()).flatMap fun block =>
          Rows.assertionRowsTR
            (block.rows Data.logicalWidth Data.publicFits)).map
              (PerApplicationCachedShift.shiftSparseRow context) ++
          application.assertionRows) ++
          NextPreimagePackage.assertionRows
            (PerApplicationPackage.directNextPreimageRowStart
              context.program)) := by
  unfold directAssertionRows directBaseAssertionRows Data.liftPilotRows
  rw [List.map_append]
  rw [← map_compose _ Data.liftPilotRow
    (PerApplicationCachedShift.shiftSparseRow context)]
  simp only [List.append_assoc]

theorem directAssertionRows_eq
    (context : PerApplicationCachedShift.Context) :
    directAssertionRows context
        (PerApplicationPackage.directApplicationPlan context.program) =
      (PerApplicationPackage.package context.program).assertionRows := by
  calc
    directAssertionRows context
        (PerApplicationPackage.directApplicationPlan context.program) =
      ((directBaseAssertionRows ()).map
          (PerApplicationPackage.shiftSparseRow context.program) ++
        (PerApplicationPackage.directApplicationPlan
          context.program).assertionRows) ++
        NextPreimagePackage.assertionRows
          (PerApplicationPackage.directNextPreimageRowStart
            context.program) := by
      unfold directAssertionRows
      exact congrArg₂ (fun left right => left ++ right)
        (congrArg₂ (fun left right => left ++ right)
          (map_eq_of_pointwise _ _ _
            (PerApplicationCachedShift.shiftSparseRow_eq context)) rfl) rfl
    _ = ((Data.circuitPackage ()).assertionRows.map
          (PerApplicationPackage.shiftSparseRow context.program) ++
        (PerApplicationPackage.applicationPlan
          context.program).assertionRows) ++
        NextPreimagePackage.assertionRows
          (PerApplicationPackage.nextPreimageRowStart context.program) := by
      rw [directBaseAssertionRows_eq,
        PerApplicationPackage.directApplicationPlan_eq_applicationPlan,
        PerApplicationPackage.directNextPreimageRowStart_eq_nextPreimageRowStart]
    _ = (PerApplicationPackage.package context.program).assertionRows := by
      simpa only [PerApplicationPackage.basePackage] using
        (PerApplicationPackage.package_assertionRows context.program).symm

/-! ## Allocation-bounded package fields -/

def directWitnessBatchCount (application : ApplicationPackage.Plan) : Nat :=
  (PilotData.priorWordBatches ()).length +
    (PiCCSPackets.initialClaim Data.logicalWidth Data.publicFits).batches.length +
    (PiCCSPackets.sumcheck Data.logicalWidth Data.publicFits).batches.length +
    (PiCCSPackets.evalK Data.logicalWidth Data.publicFits).batches.length +
    (PiCCSPackets.evalA Data.logicalWidth Data.publicFits).batches.length +
    (PiCCSPackets.ccs Data.logicalWidth Data.publicFits).batches.length +
    (PiCCSPackets.norm Data.logicalWidth Data.publicFits).batches.length +
    (PiCCSPackets.finalIdentity
      Data.logicalWidth Data.publicFits).batches.length +
    StreamingIdentity.encodedFlatMapLength WitnessPlan.Block.expand
      (WitnessPlan.canonicalBlocks Data.logicalWidth Data.publicFits) +
    application.witnessBatches.length

theorem directWitnessBatchCount_eq
    (context : PerApplicationCachedShift.Context)
    (application : ApplicationPackage.Plan) :
    directWitnessBatchCount application =
      (directWitnessBatches context application).length := by
  simp [directWitnessBatchCount, directWitnessBatches,
    directBaseWitnessBatches, piCcsPacketBatches, Data.liftPilotBatches,
    StreamingIdentity.encodedFlatMapLength_eq_flatMap_length]
  omega

@[inline, specialize push] def processWitnessBatchesWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (context : PerApplicationCachedShift.Context)
    (application : ApplicationPackage.Plan) : State :=
  let state := push state ⟨1, directWitnessBatchCount application⟩
  let state := processMappedEncodedItemsWith push state WitnessBatch.format
    (fun batch => PerApplicationCachedShift.shiftBatch context
      (Data.liftPilotBatch batch)) (PilotData.priorWordBatches ())
  let state := processMappedEncodedItemsWith push state WitnessBatch.format
    (PerApplicationCachedShift.shiftBatch context)
    (PiCCSPackets.initialClaim Data.logicalWidth Data.publicFits).batches
  let state := processMappedEncodedItemsWith push state WitnessBatch.format
    (PerApplicationCachedShift.shiftBatch context)
    (PiCCSPackets.sumcheck Data.logicalWidth Data.publicFits).batches
  let state := processMappedEncodedItemsWith push state WitnessBatch.format
    (PerApplicationCachedShift.shiftBatch context)
    (PiCCSPackets.evalK Data.logicalWidth Data.publicFits).batches
  let state := processMappedEncodedItemsWith push state WitnessBatch.format
    (PerApplicationCachedShift.shiftBatch context)
    (PiCCSPackets.evalA Data.logicalWidth Data.publicFits).batches
  let state := processMappedEncodedItemsWith push state WitnessBatch.format
    (PerApplicationCachedShift.shiftBatch context)
    (PiCCSPackets.ccs Data.logicalWidth Data.publicFits).batches
  let state := processMappedEncodedItemsWith push state WitnessBatch.format
    (PerApplicationCachedShift.shiftBatch context)
    (PiCCSPackets.norm Data.logicalWidth Data.publicFits).batches
  let state := processMappedEncodedItemsWith push state WitnessBatch.format
    (PerApplicationCachedShift.shiftBatch context)
    (PiCCSPackets.finalIdentity Data.logicalWidth Data.publicFits).batches
  let state := processMappedFlatMapItemsWith push state WitnessBatch.format
    (PerApplicationCachedShift.shiftBatch context)
    WitnessPlan.Block.expand
    (WitnessPlan.canonicalBlocks Data.logicalWidth Data.publicFits)
  StreamingIdentity.processEncodedItemsWith push state WitnessBatch.format
    application.witnessBatches

theorem processWitnessBatchesWith_eq_processValueWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (context : PerApplicationCachedShift.Context)
    (application : ApplicationPackage.Plan) :
    processWitnessBatchesWith push state context application =
      StreamingIdentity.processValueWith push
        ((list WitnessBatch.format).encode
          (directWitnessBatches context application)) state := by
  dsimp only [processWitnessBatchesWith]
  rw [directWitnessBatchCount_eq context application]
  rw [processMappedEncodedItemsWith_eq_items,
    processMappedEncodedItemsWith_eq_items,
    processMappedEncodedItemsWith_eq_items,
    processMappedEncodedItemsWith_eq_items,
    processMappedEncodedItemsWith_eq_items,
    processMappedEncodedItemsWith_eq_items,
    processMappedEncodedItemsWith_eq_items,
    processMappedEncodedItemsWith_eq_items,
    processMappedFlatMapItemsWith_eq_items]
  rw [← StreamingIdentity.processEncodedItemsWith_append,
    ← StreamingIdentity.processEncodedItemsWith_append,
    ← StreamingIdentity.processEncodedItemsWith_append,
    ← StreamingIdentity.processEncodedItemsWith_append,
    ← StreamingIdentity.processEncodedItemsWith_append,
    ← StreamingIdentity.processEncodedItemsWith_append,
    ← StreamingIdentity.processEncodedItemsWith_append,
    ← StreamingIdentity.processEncodedItemsWith_append,
    ← StreamingIdentity.processEncodedItemsWith_append]
  change StreamingIdentity.processEncodedItemsWith push
      (push state ⟨1, (directWitnessBatches context application).length⟩)
      WitnessBatch.format (directWitnessBatches context application) = _
  exact StreamingIdentity.processEncodedListWith_eq_processValueWith
    push state WitnessBatch.format (directWitnessBatches context application)

def directWitnessInstructionCount
    (application : ApplicationPackage.Plan) : Nat :=
  (PilotData.witnessInstructions ()).length +
    StreamingIdentity.encodedFlatMapLength
      (fun block => Rows.witnessInstructionsTR
        (block.rows Data.logicalWidth Data.publicFits))
      (OrdinaryRowPlan.canonicalBlocks ()) +
    application.witnessInstructions.length

theorem directWitnessInstructionCount_eq
    (context : PerApplicationCachedShift.Context)
    (application : ApplicationPackage.Plan) :
    directWitnessInstructionCount application =
      (directWitnessInstructions context application).length := by
  simp [directWitnessInstructionCount, directWitnessInstructions,
    directBaseWitnessInstructions, Data.liftPilotInstructions,
    StreamingIdentity.encodedFlatMapLength_eq_flatMap_length]
  omega

@[inline, specialize push] def processWitnessInstructionsWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (context : PerApplicationCachedShift.Context)
    (application : ApplicationPackage.Plan) : State :=
  let state := push state ⟨1, directWitnessInstructionCount application⟩
  let state := processMappedEncodedItemsWith push state WitnessInstruction.format
    (fun instruction =>
      PerApplicationCachedShift.shiftWitnessInstruction context
        (Data.liftPilotInstruction instruction))
    (PilotData.witnessInstructions ())
  let state := processMappedFlatMapItemsWith push state WitnessInstruction.format
    (PerApplicationCachedShift.shiftWitnessInstruction context)
    (fun block => Rows.witnessInstructionsTR
      (block.rows Data.logicalWidth Data.publicFits))
    (OrdinaryRowPlan.canonicalBlocks ())
  StreamingIdentity.processEncodedItemsWith push state
    WitnessInstruction.format application.witnessInstructions

theorem processWitnessInstructionsWith_eq_processValueWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (context : PerApplicationCachedShift.Context)
    (application : ApplicationPackage.Plan) :
    processWitnessInstructionsWith push state context application =
      StreamingIdentity.processValueWith push
        ((list WitnessInstruction.format).encode
          (directWitnessInstructions context application)) state := by
  dsimp only [processWitnessInstructionsWith]
  rw [directWitnessInstructionCount_eq context application,
    directWitnessInstructions_segments context application,
    processMappedEncodedItemsWith_eq_items,
    processMappedFlatMapItemsWith_eq_items,
    ← StreamingIdentity.processEncodedItemsWith_append,
    ← StreamingIdentity.processEncodedItemsWith_append]
  exact processEncodedItemsWith_header_eq_processValueWith
    push state WitnessInstruction.format _

def directAssertionRowCount (context : PerApplicationCachedShift.Context)
    (application : ApplicationPackage.Plan) : Nat :=
  (PilotData.assertionRows ()).length +
    StreamingIdentity.encodedFlatMapLength
      (fun block => Rows.assertionRowsTR
        (block.rows Data.logicalWidth Data.publicFits))
      (OrdinaryRowPlan.canonicalBlocks ()) +
    application.assertionRows.length +
    (NextPreimagePackage.assertionRows
      (PerApplicationPackage.directNextPreimageRowStart
        context.program)).length

theorem directAssertionRowCount_eq
    (context : PerApplicationCachedShift.Context)
    (application : ApplicationPackage.Plan) :
    directAssertionRowCount context application =
      (directAssertionRows context application).length := by
  simp [directAssertionRowCount, directAssertionRows,
    directBaseAssertionRows, Data.liftPilotRows,
    StreamingIdentity.encodedFlatMapLength_eq_flatMap_length]
  omega

@[inline, specialize push] def processAssertionRowsWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (context : PerApplicationCachedShift.Context)
    (application : ApplicationPackage.Plan) : State :=
  let state := push state ⟨1, directAssertionRowCount context application⟩
  let state := processMappedEncodedItemsWith push state SparseRow.format
    (fun row => PerApplicationCachedShift.shiftSparseRow context
      (Data.liftPilotRow row)) (PilotData.assertionRows ())
  let state := processMappedFlatMapItemsWith push state SparseRow.format
    (PerApplicationCachedShift.shiftSparseRow context)
    (fun block => Rows.assertionRowsTR
      (block.rows Data.logicalWidth Data.publicFits))
    (OrdinaryRowPlan.canonicalBlocks ())
  let state := StreamingIdentity.processEncodedItemsWith push state
    SparseRow.format application.assertionRows
  StreamingIdentity.processEncodedItemsWith push state SparseRow.format
    (NextPreimagePackage.assertionRows
      (PerApplicationPackage.directNextPreimageRowStart context.program))

theorem processAssertionRowsWith_eq_processValueWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (context : PerApplicationCachedShift.Context)
    (application : ApplicationPackage.Plan) :
    processAssertionRowsWith push state context application =
      StreamingIdentity.processValueWith push
        ((list SparseRow.format).encode
          (directAssertionRows context application)) state := by
  dsimp only [processAssertionRowsWith]
  rw [directAssertionRowCount_eq context application,
    directAssertionRows_segments context application,
    processMappedEncodedItemsWith_eq_items,
    processMappedFlatMapItemsWith_eq_items,
    ← StreamingIdentity.processEncodedItemsWith_append,
    ← StreamingIdentity.processEncodedItemsWith_append,
    ← StreamingIdentity.processEncodedItemsWith_append]
  simp only [List.append_assoc]
  exact processEncodedItemsWith_header_eq_processValueWith
    push state SparseRow.format _

@[inline, specialize push] def processHashChainsWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (context : PerApplicationCachedShift.Context) : State :=
  processMappedEncodedListWith push state HashChain.format
    (PerApplicationCachedShift.shiftHashChain context)
    [Data.priorChain, Data.outputChain]

theorem processHashChainsWith_eq_processValueWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (context : PerApplicationCachedShift.Context) :
    processHashChainsWith push state context =
      StreamingIdentity.processValueWith push
        ((list HashChain.format).encode (directHashChains context)) state := by
  exact processMappedEncodedListWith_eq_processValueWith push state
    HashChain.format (PerApplicationCachedShift.shiftHashChain context)
      [Data.priorChain, Data.outputChain]

@[inline, specialize push] def processPermutationInvocationsWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (context : PerApplicationCachedShift.Context) : State :=
  processMappedFlatMapWith push state PermutationInvocation.format
    (PerApplicationCachedShift.shiftPermutationInvocation context)
    PermutationPlan.Block.expand (PermutationPlan.canonicalBlocks ())

theorem processPermutationInvocationsWith_eq_processValueWith
    {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (context : PerApplicationCachedShift.Context) :
    processPermutationInvocationsWith push state context =
      StreamingIdentity.processValueWith push
        ((list PermutationInvocation.format).encode
          (directPermutationInvocations context)) state := by
  exact processMappedFlatMapWith_eq_processValueWith push state
    PermutationInvocation.format
    (PerApplicationCachedShift.shiftPermutationInvocation context)
    PermutationPlan.Block.expand (PermutationPlan.canonicalBlocks ())

@[inline, specialize push] def processCompactRowInvocationsWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (context : PerApplicationCachedShift.Context) : State :=
  processMappedFlatMapWith push state CompactRowInvocation.format
    (PerApplicationCachedShift.shiftCompactRowInvocation context)
    PackagePlan.CompactInvocationBlock.expand
    PackagePlan.canonicalCompactBlocks

theorem processCompactRowInvocationsWith_eq_processValueWith
    {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (context : PerApplicationCachedShift.Context) :
    processCompactRowInvocationsWith push state context =
      StreamingIdentity.processValueWith push
        ((list CompactRowInvocation.format).encode
          (directCompactRowInvocations context)) state := by
  exact processMappedFlatMapWith_eq_processValueWith push state
    CompactRowInvocation.format
    (PerApplicationCachedShift.shiftCompactRowInvocation context)
    PackagePlan.CompactInvocationBlock.expand
    PackagePlan.canonicalCompactBlocks

/-! ## Fixed codec records -/

@[specialize push] def processApplicationPlanWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (plan : ApplicationPackage.Plan) : State :=
  let state := push state ⟨1, 16⟩
  let state := push state ⟨0, plan.schemaVersion⟩
  let state := push state ⟨0, plan.witnessWordCount⟩
  let state := StreamingIdentity.processEncodedListWith push state nat
    plan.inputColumns
  let state := StreamingIdentity.processEncodedListWith push state nat
    plan.witnessColumns
  let state := StreamingIdentity.processEncodedListWith push state nat
    plan.outputColumns
  let state := push state ⟨0, plan.privateStart⟩
  let state := push state ⟨0, plan.privateCount⟩
  let state := push state ⟨0, plan.rowStart⟩
  let state := push state ⟨0, plan.rowCount⟩
  let state := StreamingIdentity.processEncodedListWith push state
    HashChain.format plan.hashChains
  let state := StreamingIdentity.processEncodedListWith push state
    PermutationInvocation.format plan.permutationInvocations
  let state := StreamingIdentity.processEncodedListWith push state
    CompactRowTemplate.format plan.compactRowTemplates
  let state := StreamingIdentity.processEncodedListWith push state
    CompactRowInvocation.format plan.compactRowInvocations
  let state := StreamingIdentity.processEncodedListWith push state
    WitnessBatch.format plan.witnessBatches
  let state := StreamingIdentity.processEncodedListWith push state
    WitnessInstruction.format plan.witnessInstructions
  StreamingIdentity.processEncodedListWith push state SparseRow.format
    plan.assertionRows

theorem processApplicationPlanWith_eq_processValueWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (plan : ApplicationPackage.Plan) :
    processApplicationPlanWith push state plan =
      StreamingIdentity.processValueWith push
        (ApplicationPackage.Plan.format.encode plan) state := by
  cases plan
  dsimp only [processApplicationPlanWith, ApplicationPackage.Plan.format]
  rw [StreamingIdentity.processEncodedListWith_eq_processValueWith,
    StreamingIdentity.processEncodedListWith_eq_processValueWith,
    StreamingIdentity.processEncodedListWith_eq_processValueWith,
    StreamingIdentity.processEncodedListWith_eq_processValueWith,
    StreamingIdentity.processEncodedListWith_eq_processValueWith,
    StreamingIdentity.processEncodedListWith_eq_processValueWith,
    StreamingIdentity.processEncodedListWith_eq_processValueWith,
    StreamingIdentity.processEncodedListWith_eq_processValueWith,
    StreamingIdentity.processEncodedListWith_eq_processValueWith,
    StreamingIdentity.processEncodedListWith_eq_processValueWith]
  simp only [StreamingIdentity.processValueWith, List.foldl_cons,
    List.foldl_nil, List.length_cons, List.length_nil]

@[inline, specialize push] def processMatrixProgramWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (program : MatrixProgram.Program) : State :=
  StreamingIdentity.processEncodedListWith push state
    MatrixProgram.Block.format program.blocks

theorem processMatrixProgramWith_eq_processValueWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (program : MatrixProgram.Program) :
    processMatrixProgramWith push state program =
      StreamingIdentity.processValueWith push
        (MatrixProgram.Program.format.encode program) state := by
  cases program
  exact StreamingIdentity.processEncodedListWith_eq_processValueWith
    push state MatrixProgram.Block.format _

@[inline, specialize push] def processAssignmentTransportWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (program : Program) : State :=
  let state := push state ⟨1, 8⟩
  let state := push state
    ⟨0, PerApplicationAssignmentTransport.schema⟩
  let state := StreamingIdentity.processEncodedListWith push state
    PerApplicationAssignmentBlocks.BlockPlan.format
    (PerApplicationAssignmentBlocks.canonical program)
  let state := StreamingIdentity.processValueWith push
    (PerApplicationAssignmentTransport.Phi81GroupRecipe.format.encode
      (PerApplicationAssignmentTransport.phi81GroupRecipe program)) state
  let state := StreamingIdentity.processValueWith push
    (PerApplicationAssignmentTransport.First54ProductRecipe.format.encode
      PerApplicationAssignmentTransport.first54ProductRecipe) state
  let state := StreamingIdentity.processValueWith push
    (PerApplicationAssignmentPlan.BlockKind.format.encode .piCcsPayload) state
  let state := StreamingIdentity.processEncodedListWith push state exprFormat
    (PerApplicationAssignmentTransport.materializedPayloadExpressions program)
  let state := StreamingIdentity.processValueWith push
    (PerApplicationAssignmentPlan.BlockKind.format.encode
      .pilotOutputDigest) state
  StreamingIdentity.processEncodedListWith push state exprFormat
    (PerApplicationAssignmentTransport.outputDigestExpressions program)

theorem processAssignmentTransportWith_eq_processValueWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (program : Program) :
    processAssignmentTransportWith push state program =
      StreamingIdentity.processValueWith push
        (PerApplicationAssignmentTransport.Plan.format.encode
          (PerApplicationAssignmentTransport.canonical program)) state := by
  dsimp only [processAssignmentTransportWith,
    PerApplicationAssignmentTransport.Plan.format,
    PerApplicationAssignmentTransport.canonical,
    PerApplicationAssignmentBlocks.format]
  rw [PerApplicationAssignmentTransport.materializedPayloadExpressions_eq,
    StreamingIdentity.processEncodedListWith_eq_processValueWith,
    StreamingIdentity.processEncodedListWith_eq_processValueWith,
    StreamingIdentity.processEncodedListWith_eq_processValueWith]
  simp only [StreamingIdentity.processValueWith, List.foldl_cons,
    List.foldl_nil, List.length_cons, List.length_nil]

private theorem processValueWith_array {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (values : List Value) :
    StreamingIdentity.processValueWith push (.array values) state =
      values.foldl
        (fun current value =>
          StreamingIdentity.processValueWith push value current)
        (push state ⟨1, values.length⟩) := by
  simp only [StreamingIdentity.processValueWith]

/-- Proof view of the exact 14-child package codec value. The executable
traversal below does not reference this materialized value. -/
def directInnerValue (context : PerApplicationCachedShift.Context)
    (application : ApplicationPackage.Plan) : Value :=
  .array [
    .atom 8,
    Profile.format.encode PilotData.profile,
    PoseidonSchedule.format.encode PilotData.poseidonSchedule,
    PhysicalLayout.format.encode
      (PerApplicationPackage.directFinalLayout context.program),
    CcsRelation.format.encode
      (PerApplicationCanonicalPackage.directRecursiveRelation context.program),
    PermutationTemplate.format.encode (PilotData.permutationTemplate ()),
    (list HashChain.format).encode (directHashChains context),
    (list PermutationInvocation.format).encode
      (directPermutationInvocations context),
    (list CompactRowTemplate.format).encode (Data.compactRowTemplates ()),
    (list CompactRowInvocation.format).encode
      (directCompactRowInvocations context),
    (list WitnessBatch.format).encode
      (directWitnessBatches context application),
    (list WitnessInstruction.format).encode
      (directWitnessInstructions context application),
    (list SparseRow.format).encode (directAssertionRows context application),
    (option TerminalLayout.format).encode
      (some (PerApplicationCanonicalPackage.directTerminalLayout
        context.program))]

private theorem directSchemaVersion_eq
    (program : Program) (fits : FitsTwoPow28 program) :
    8 = (PerApplicationCanonicalPackage.package program fits).schemaVersion := by
  rfl

private theorem directProfile_eq
    (program : Program) (fits : FitsTwoPow28 program) :
    PilotData.profile =
      (PerApplicationCanonicalPackage.package program fits).profile := by
  rfl

private theorem directPoseidon_eq
    (program : Program) (fits : FitsTwoPow28 program) :
    PilotData.poseidonSchedule =
      (PerApplicationCanonicalPackage.package program fits).poseidon := by
  rfl

private theorem directLayout_eq
    (program : Program) (fits : FitsTwoPow28 program) :
    PerApplicationPackage.directFinalLayout program =
      (PerApplicationCanonicalPackage.package program fits).layout := by
  rw [PerApplicationPackage.directFinalLayout_eq_finalLayout]
  rfl

private theorem directRelation_eq
    (program : Program) (fits : FitsTwoPow28 program) :
    PerApplicationCanonicalPackage.directRecursiveRelation program =
      (PerApplicationCanonicalPackage.package program fits).relation := by
  rw [PerApplicationCanonicalPackage.directRecursiveRelation_eq_recursiveRelation]
  rfl

private theorem directPermutation_eq
    (program : Program) (fits : FitsTwoPow28 program) :
    PilotData.permutationTemplate () =
      (PerApplicationCanonicalPackage.package program fits).permutation := by
  rfl

private theorem installedReplaced_hashChains (source : CircuitPackage)
    (relation : CcsRelation) :
    (TerminalPackage.install
      (PerApplicationCanonicalPackage.replaceRelation
        source relation)).hashChains = source.hashChains := by
  cases source
  rfl

private theorem installedReplaced_permutationInvocations
    (source : CircuitPackage) (relation : CcsRelation) :
    (TerminalPackage.install
      (PerApplicationCanonicalPackage.replaceRelation
        source relation)).permutationInvocations =
      source.permutationInvocations := by
  cases source
  rfl

private theorem installedReplaced_compactRowInvocations
    (source : CircuitPackage) (relation : CcsRelation) :
    (TerminalPackage.install
      (PerApplicationCanonicalPackage.replaceRelation
        source relation)).compactRowInvocations =
      source.compactRowInvocations := by
  cases source
  rfl

private theorem installedReplaced_witnessBatches (source : CircuitPackage)
    (relation : CcsRelation) :
    (TerminalPackage.install
      (PerApplicationCanonicalPackage.replaceRelation
        source relation)).witnessBatches = source.witnessBatches := by
  cases source
  rfl

private theorem canonicalPackage_hashChains_eq_source
    (program : Program) (fits : FitsTwoPow28 program) :
    (PerApplicationCanonicalPackage.package program fits).hashChains =
      (PerApplicationPackage.package program).hashChains := by
  change (TerminalPackage.install
      (PerApplicationCanonicalPackage.replaceRelation
        (PerApplicationPackage.package program)
        (PerApplicationCanonicalPackage.recursiveRelation
          program fits))).hashChains = _
  exact installedReplaced_hashChains _ _

private theorem canonicalPackage_permutationInvocations_eq_source
    (program : Program) (fits : FitsTwoPow28 program) :
    (PerApplicationCanonicalPackage.package
        program fits).permutationInvocations =
      (PerApplicationPackage.package program).permutationInvocations := by
  change (TerminalPackage.install
      (PerApplicationCanonicalPackage.replaceRelation
        (PerApplicationPackage.package program)
        (PerApplicationCanonicalPackage.recursiveRelation
          program fits))).permutationInvocations = _
  exact installedReplaced_permutationInvocations _ _

private theorem canonicalPackage_compactRowInvocations_eq_source
    (program : Program) (fits : FitsTwoPow28 program) :
    (PerApplicationCanonicalPackage.package
        program fits).compactRowInvocations =
      (PerApplicationPackage.package program).compactRowInvocations := by
  change (TerminalPackage.install
      (PerApplicationCanonicalPackage.replaceRelation
        (PerApplicationPackage.package program)
        (PerApplicationCanonicalPackage.recursiveRelation
          program fits))).compactRowInvocations = _
  exact installedReplaced_compactRowInvocations _ _

private theorem canonicalPackage_witnessBatches_eq_source
    (program : Program) (fits : FitsTwoPow28 program) :
    (PerApplicationCanonicalPackage.package program fits).witnessBatches =
      (PerApplicationPackage.package program).witnessBatches := by
  change (TerminalPackage.install
      (PerApplicationCanonicalPackage.replaceRelation
        (PerApplicationPackage.package program)
        (PerApplicationCanonicalPackage.recursiveRelation
          program fits))).witnessBatches = _
  exact installedReplaced_witnessBatches _ _

private theorem canonicalPackage_witnessInstructions_eq_source
    (program : Program) (fits : FitsTwoPow28 program) :
    (PerApplicationCanonicalPackage.package
        program fits).witnessInstructions =
      (PerApplicationPackage.package program).witnessInstructions := by
  change (TerminalPackage.install
      (PerApplicationCanonicalPackage.replaceRelation
        (PerApplicationPackage.package program)
        (PerApplicationCanonicalPackage.recursiveRelation
          program fits))).witnessInstructions = _
  exact (TerminalPackage.install_witnessInstructions _).trans
    (PerApplicationCanonicalPackage.replaceRelation_witnessInstructions _ _)

private theorem canonicalPackage_assertionRows_eq_source
    (program : Program) (fits : FitsTwoPow28 program) :
    (PerApplicationCanonicalPackage.package program fits).assertionRows =
      (PerApplicationPackage.package program).assertionRows := by
  change (TerminalPackage.install
      (PerApplicationCanonicalPackage.replaceRelation
        (PerApplicationPackage.package program)
        (PerApplicationCanonicalPackage.recursiveRelation
          program fits))).assertionRows = _
  exact (TerminalPackage.install_assertionRows _).trans
    (PerApplicationCanonicalPackage.replaceRelation_assertionRows _ _)

private theorem directHashChains_eq_canonical
    (context : PerApplicationCachedShift.Context)
    (fits : FitsTwoPow28 context.program) :
    directHashChains context =
      (PerApplicationCanonicalPackage.package
        context.program fits).hashChains := by
  exact (directHashChains_eq context).trans
    (canonicalPackage_hashChains_eq_source context.program fits).symm

private theorem directPermutationInvocations_eq_canonical
    (context : PerApplicationCachedShift.Context)
    (fits : FitsTwoPow28 context.program) :
    directPermutationInvocations context =
      (PerApplicationCanonicalPackage.package
        context.program fits).permutationInvocations := by
  exact (directPermutationInvocations_eq context).trans
    (canonicalPackage_permutationInvocations_eq_source
      context.program fits).symm

private theorem directCompactRowTemplates_eq
    (program : Program) (fits : FitsTwoPow28 program) :
    Data.compactRowTemplates () =
      (PerApplicationCanonicalPackage.package
        program fits).compactRowTemplates := by
  rfl

private theorem directCompactRowInvocations_eq_canonical
    (context : PerApplicationCachedShift.Context)
    (fits : FitsTwoPow28 context.program) :
    directCompactRowInvocations context =
      (PerApplicationCanonicalPackage.package
        context.program fits).compactRowInvocations := by
  exact (directCompactRowInvocations_eq context).trans
    (canonicalPackage_compactRowInvocations_eq_source
      context.program fits).symm

private theorem directWitnessBatches_eq_canonical
    (context : PerApplicationCachedShift.Context)
    (fits : FitsTwoPow28 context.program) :
    directWitnessBatches context
        (PerApplicationPackage.directApplicationPlan context.program) =
      (PerApplicationCanonicalPackage.package
        context.program fits).witnessBatches := by
  exact (directWitnessBatches_eq context).trans
    (canonicalPackage_witnessBatches_eq_source context.program fits).symm

private theorem directWitnessInstructions_eq_canonical
    (context : PerApplicationCachedShift.Context)
    (fits : FitsTwoPow28 context.program) :
    directWitnessInstructions context
        (PerApplicationPackage.directApplicationPlan context.program) =
      (PerApplicationCanonicalPackage.package
        context.program fits).witnessInstructions := by
  exact (directWitnessInstructions_eq context).trans
    (canonicalPackage_witnessInstructions_eq_source
      context.program fits).symm

private theorem directAssertionRows_eq_canonical
    (context : PerApplicationCachedShift.Context)
    (fits : FitsTwoPow28 context.program) :
    directAssertionRows context
        (PerApplicationPackage.directApplicationPlan context.program) =
      (PerApplicationCanonicalPackage.package
        context.program fits).assertionRows := by
  exact (directAssertionRows_eq context).trans
    (canonicalPackage_assertionRows_eq_source context.program fits).symm

private theorem directTerminal_eq
    (program : Program) (fits : FitsTwoPow28 program) :
    some (PerApplicationCanonicalPackage.directTerminalLayout program) =
      (PerApplicationCanonicalPackage.package program fits).terminal := by
  rw [PerApplicationCanonicalPackage.package_terminal,
    PerApplicationCanonicalPackage.directTerminalLayout_eq_layoutFor_package]

theorem directInnerValue_eq
    (context : PerApplicationCachedShift.Context)
    (fits : FitsTwoPow28 context.program) :
    directInnerValue context
        (PerApplicationPackage.directApplicationPlan context.program) =
      CircuitPackage.format.encode
        (PerApplicationCanonicalPackage.package context.program fits) := by
  unfold directInnerValue CircuitPackage.format
  rw [directSchemaVersion_eq, directProfile_eq, directPoseidon_eq,
    directLayout_eq, directRelation_eq, directPermutation_eq,
    directHashChains_eq_canonical, directPermutationInvocations_eq_canonical,
    directCompactRowTemplates_eq,
    directCompactRowInvocations_eq_canonical,
    directWitnessBatches_eq_canonical,
    directWitnessInstructions_eq_canonical,
    directAssertionRows_eq_canonical, directTerminal_eq]

@[specialize push] def processInnerPackageWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (context : PerApplicationCachedShift.Context)
    (application : ApplicationPackage.Plan) : State :=
  let state := push state ⟨1, 14⟩
  let state := StreamingIdentity.processValueWith push (.atom 8) state
  let state := StreamingIdentity.processValueWith push
    (Profile.format.encode PilotData.profile) state
  let state := StreamingIdentity.processValueWith push
    (PoseidonSchedule.format.encode PilotData.poseidonSchedule) state
  let state := StreamingIdentity.processValueWith push
    (PhysicalLayout.format.encode
      (PerApplicationPackage.directFinalLayout context.program)) state
  let state := StreamingIdentity.processValueWith push
    (CcsRelation.format.encode
      (PerApplicationCanonicalPackage.directRecursiveRelation
        context.program)) state
  let state := StreamingIdentity.processValueWith push
    (PermutationTemplate.format.encode (PilotData.permutationTemplate ())) state
  let state := processHashChainsWith push state context
  let state := processPermutationInvocationsWith push state context
  let state := StreamingIdentity.processEncodedListWith push state
    CompactRowTemplate.format (Data.compactRowTemplates ())
  let state := processCompactRowInvocationsWith push state context
  let state := processWitnessBatchesWith push state context application
  let state := processWitnessInstructionsWith push state context application
  let state := processAssertionRowsWith push state context application
  StreamingIdentity.processValueWith push
    ((option TerminalLayout.format).encode
      (some (PerApplicationCanonicalPackage.directTerminalLayout
        context.program))) state

theorem processInnerPackageWith_eq_processValueWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (context : PerApplicationCachedShift.Context)
    (fits : FitsTwoPow28 context.program) :
    processInnerPackageWith push state context
        (PerApplicationPackage.directApplicationPlan context.program) =
      StreamingIdentity.processValueWith push
        (CircuitPackage.format.encode
          (PerApplicationCanonicalPackage.package context.program fits)) state := by
  rw [← directInnerValue_eq context fits]
  dsimp only [processInnerPackageWith]
  rw [processHashChainsWith_eq_processValueWith,
    processPermutationInvocationsWith_eq_processValueWith,
    StreamingIdentity.processEncodedListWith_eq_processValueWith,
    processCompactRowInvocationsWith_eq_processValueWith,
    processWitnessBatchesWith_eq_processValueWith,
    processWitnessInstructionsWith_eq_processValueWith,
    processAssertionRowsWith_eq_processValueWith]
  unfold directInnerValue
  rw [processValueWith_array]
  rfl

theorem processInnerPackageOfProgramWith_eq_processValueWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (state : State)
    (program : Program) (fits : FitsTwoPow28 program) :
    processInnerPackageWith push state
        (PerApplicationCachedShift.Context.ofProgram program)
        (PerApplicationPackage.directApplicationPlan program) =
      StreamingIdentity.processValueWith push
        (CircuitPackage.format.encode
          (PerApplicationCanonicalPackage.package program fits)) state := by
  simpa only [PerApplicationCachedShift.Context.ofProgram] using
    (processInnerPackageWith_eq_processValueWith push state
      (PerApplicationCachedShift.Context.ofProgram program) fits)

/-! ## Sealed root traversal -/

/-- Direct traversal of the seven-child sealed root. It is generic in the
state representation and in the one authoritative node transition supplied
by the caller. -/
@[specialize push] def directStateWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (initial : State)
    (program : Program) : State :=
  let context := PerApplicationCachedShift.Context.ofProgram program
  let application := PerApplicationPackage.directApplicationPlan program
  let state := push initial ⟨1, 7⟩
  let state := StreamingIdentity.processValueWith push
    (.atom PerApplicationCanonicalPackage.sealedPackageSchema) state
  let state := processInnerPackageWith push state context application
  let state := processMatrixProgramWith push state
    (PerApplicationMatrixProgram.matrixProgram program)
  let state := processApplicationPlanWith push state application
  let state := processAssignmentTransportWith push state program
  let state := StreamingIdentity.processValueWith push
    (MatrixProgram.IndexRange.format.encode
      (PerApplicationCanonicalPackage.nextPreimageRange program)) state
  StreamingIdentity.processValueWith push
    (.atom PerApplicationCanonicalPackage.logicalPublicInputCount) state

theorem directStateWith_eq_processValueWith {State : Type}
    (push : State → StreamingIdentity.Node → State) (initial : State)
    (program : Program) (fits : FitsTwoPow28 program) :
    directStateWith push initial program =
      StreamingIdentity.processValueWith push
        (PerApplicationCanonicalPackage.sealedPackageValue program fits)
        initial := by
  dsimp only [directStateWith]
  rw [processInnerPackageOfProgramWith_eq_processValueWith,
    processMatrixProgramWith_eq_processValueWith,
    processApplicationPlanWith_eq_processValueWith,
    processAssignmentTransportWith_eq_processValueWith]
  unfold PerApplicationCanonicalPackage.sealedPackageValue
  rw [processValueWith_array]
  rfl

def semanticState (program : Program) : StreamingIdentity.HashState :=
  directStateWith StreamingIdentity.pushNode
    StreamingIdentity.initialState program

theorem semanticState_eq
    (program : Program) (fits : FitsTwoPow28 program) :
    semanticState program =
      StreamingIdentity.processValue
        (PerApplicationCanonicalPackage.sealedPackageValue program fits)
        StreamingIdentity.initialState := by
  unfold semanticState StreamingIdentity.processValue
  exact directStateWith_eq_processValueWith
    StreamingIdentity.pushNode StreamingIdentity.initialState program fits

/-- Semantic streaming digest. This name is separate from the final native
entry point so the executable boundary has one monomorphic state type. -/
def structuralPackageIdentityStream (program : Program)
    (_fits : FitsTwoPow28 program) : VerifierContext.Digest4 :=
  VerifierContext.Digest4.ofList
    (StreamingIdentity.finalize (semanticState program))

theorem structuralPackageIdentityStream_eq
    (program : Program) (fits : FitsTwoPow28 program) :
    structuralPackageIdentityStream program fits =
      PerApplicationCanonicalPackage.structuralPackageIdentity
        program fits := by
  unfold structuralPackageIdentityStream
  rw [semanticState_eq program fits]
  change VerifierContext.Digest4.ofList
      (StreamingIdentity.relationIdentifierValueFast
        (PerApplicationCanonicalPackage.sealedPackageValue program fits)) = _
  rw [StreamingIdentity.relationIdentifierValueFast_eq]
  rfl

/-! ## Monomorphic native boundary -/

/-- First-order native traversal. Each bounded child is compiled separately,
so the compiler does not normalize the complete generic traversal at once. -/
def directState (program : Program) : NativePoseidon2.HashState64 :=
  let context := PerApplicationCachedShift.Context.ofProgram program
  let application := PerApplicationPackage.directApplicationPlan program
  let state := NativePoseidon2.pushNode64 NativePoseidon2.initialState64
    ⟨1, 7⟩
  let state := StreamingIdentity.processValueWith NativePoseidon2.pushNode64
    (.atom PerApplicationCanonicalPackage.sealedPackageSchema) state
  let state := processInnerPackageWith NativePoseidon2.pushNode64 state context
    application
  let state := processMatrixProgramWith NativePoseidon2.pushNode64 state
    (PerApplicationMatrixProgram.matrixProgram program)
  let state := processApplicationPlanWith NativePoseidon2.pushNode64 state
    application
  let state := processAssignmentTransportWith NativePoseidon2.pushNode64 state
    program
  let state := StreamingIdentity.processValueWith NativePoseidon2.pushNode64
    (MatrixProgram.IndexRange.format.encode
      (PerApplicationCanonicalPackage.nextPreimageRange program)) state
  StreamingIdentity.processValueWith NativePoseidon2.pushNode64
    (.atom PerApplicationCanonicalPackage.logicalPublicInputCount) state

private theorem directState_eq_directStateWith (program : Program) :
    directState program =
      directStateWith NativePoseidon2.pushNode64
        NativePoseidon2.initialState64 program := by
  rfl

/-- One whole-traversal refinement. Runtime stays in `HashState64`; the proof
maps only the final state to the semantic Poseidon2 state. -/
theorem directState_denote (program : Program) (fits : FitsTwoPow28 program) :
    (directState program).denote = semanticState program := by
  rw [directState_eq_directStateWith]
  unfold semanticState
  rw [directStateWith_eq_processValueWith NativePoseidon2.pushNode64
      NativePoseidon2.initialState64 program fits,
    directStateWith_eq_processValueWith StreamingIdentity.pushNode
      StreamingIdentity.initialState program fits]
  have simulation := StreamingIdentity.processValueWith_simulates
    NativePoseidon2.HashState64.denote NativePoseidon2.pushNode64
    StreamingIdentity.pushNode NativePoseidon2.pushNode64_denote
    (PerApplicationCanonicalPackage.sealedPackageValue program fits)
    NativePoseidon2.initialState64
  rw [NativePoseidon2.initialState64_denote] at simulation
  exact simulation

/-- Allocation-bounded executable identity for the final sealed package. -/
@[inline] def structuralPackageIdentityDirect (program : Program)
    (_fits : FitsTwoPow28 program) : VerifierContext.Digest4 :=
  VerifierContext.Digest4.ofList
    (NativePoseidon2.finalize64 (directState program)).denote

/-- The native result is the canonical relation identifier. The canonical
package value remains the semantic authority. -/
theorem structuralPackageIdentityDirect_eq
    (program : Program) (fits : FitsTwoPow28 program) :
    structuralPackageIdentityDirect program fits =
      PerApplicationCanonicalPackage.structuralPackageIdentity program fits := by
  unfold structuralPackageIdentityDirect
  rw [NativePoseidon2.finalize64_denote, directState_denote program fits]
  exact structuralPackageIdentityStream_eq program fits

/-- Replace the former artifact-sized executable traversal only after both
executables have been proved equal to the canonical structural identity. -/
@[csimp] theorem structuralPackageIdentityFast_eq_direct :
    @PerApplicationCanonicalPackage.structuralPackageIdentityFast =
      @structuralPackageIdentityDirect := by
  funext program fits
  exact
    (PerApplicationCanonicalPackage.structuralPackageIdentityFast_eq
      program fits).trans (structuralPackageIdentityDirect_eq program fits).symm

end NightstreamFPrime.Export.Stage1.PerApplicationStreamingIdentity
