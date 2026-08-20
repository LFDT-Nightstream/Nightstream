import Nightstream.Implementation.R1CS.Canonical.KMulHonest
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerDecodeRowSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpenTransition
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalSourceBindingRowSound

/-!
Contract: bind the exact terminal source-assignment delayed-open bit to the
column consumed by the Rust-emitted terminal finalizer.

The source bit remains subject to semantic fresh-witness authority. This
module proves only its exact source-to-final row ownership.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerSourceBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalSourceBinding
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalSourceBinding.Artifact
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpenTransition
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerDecodeRowSound
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionRowSound
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalSourceBindingRowSound

private abbrev sourceArtifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalSourceBinding.rawArtifact

private abbrev finalizerArtifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer.rawArtifact

abbrev SourceSatisfied (assignment : Nat → Nat) : Prop :=
  sourceArtifact.Satisfied assignment

abbrev DecodeSatisfied (assignment : Nat → Nat) : Prop :=
  finalizerArtifact.DecodeSatisfied assignment

def delayedOpenSource (assignment : Nat → Nat) : Bool :=
  assignment delayedOpenSourceColumn == 1

/-- Recompose one post-phase lane field directly from the final selective
assignment columns emitted for its authoritative Rust source field. -/
def nebulaLaneSourceValue (assignment : Nat → Nat) (index : Nat) : Nat :=
  if index < 4 then
    lcEval assignment (nebulaProgramDigestBlock.termsAt index)
  else if index < 5 then
    lcEval assignment (nebulaOpenBlock.termsAt (index - 4))
  else if index < 7 then
    lcEval assignment (nebulaCountersBlock.termsAt (index - 5))
  else if index < 8 then
    lcEval assignment (nebulaTimestampBlock.termsAt (index - 7))
  else
    lcEval assignment (nebulaStateBlock.termsAt (index - 8))

/-- Typed post-phase Nebula lane reconstructed from the exact final selective
source encodings, before the finalizer consumes its decoded aliases. -/
def sourceLane (assignment : Nat → Nat) :
    StreamingTerminalFullFinalizerTransitionRelation.Lane :=
  laneAt (nebulaLaneSourceValue assignment) (List.range laneFields)

def payloadSourceAssignment (assignment : Nat → Nat) (column : Nat) : Nat :=
  assignment (22126657 + (column - 28041985))

def sourceStepWordValue (assignment : Nat → Nat) (index : Nat) : Nat :=
  stepWordValue (payloadSourceAssignment assignment) index

def sourceStepFieldAt (assignment : Nat → Nat) (index : Nat) : F :=
  fieldValue (sourceStepWordValue assignment) index

def sourceStepKAt (assignment : Nat → Nat) (start : Nat) : K where
  c0 := sourceStepFieldAt assignment start
  c1 := sourceStepFieldAt assignment (start + 1)

/-- Typed delayed step reconstructed from the exact final selective source
bits. The stack-free production profile supplies both stack arrays as zero. -/
def sourceStep (assignment : Nat → Nat) :
    StreamingTerminalFullFinalizerTransitionRelation.StepInput where
  segmentIndex := sourceStepWordValue assignment 0
  stepIndex := sourceStepWordValue assignment 1
  timestampIn := sourceStepWordValue assignment 2
  timestampOut := sourceStepWordValue assignment 3
  gamma := fun index => sourceStepKAt assignment (4 + 2 * index.val)
  productsIn := fun index => sourceStepKAt assignment (8 + 2 * index.val)
  productsOut := fun index => sourceStepKAt assignment (16 + 2 * index.val)
  stackPointersIn := fun _ => 0
  stackPointersOut := fun _ => 0

def sourceDPreWordValue (assignment : Nat → Nat) (index : Nat) : Nat :=
  dPreWordValue (payloadSourceAssignment assignment) index

def sourceDPre (assignment : Nat → Nat) : Fin 3 →
    StreamingTerminalFullFinalizerTransitionRelation.Digest := fun lane coordinate =>
  fieldValue (sourceDPreWordValue assignment)
    (4 * lane.val + coordinate.val)

private theorem range_getD_eq
    (count index fallback : Nat) (bounded : index < count) :
    (List.range count).getD index fallback = index := by
  have inRange : index < (List.range count).length := by
    simpa using bounded
  rw [← List.getElem_eq_getD fallback]
  exact List.getElem_range inRange

private theorem stepInput_eq_of_fields
    {left right :
      StreamingTerminalFullFinalizerTransitionRelation.StepInput}
    (segmentIndex : left.segmentIndex = right.segmentIndex)
    (stepIndex : left.stepIndex = right.stepIndex)
    (timestampIn : left.timestampIn = right.timestampIn)
    (timestampOut : left.timestampOut = right.timestampOut)
    (gamma : left.gamma = right.gamma)
    (productsIn : left.productsIn = right.productsIn)
    (productsOut : left.productsOut = right.productsOut)
    (stackPointersIn : left.stackPointersIn = right.stackPointersIn)
    (stackPointersOut : left.stackPointersOut = right.stackPointersOut) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem rows_bind_nebulaLaneSourceValue
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (sourceSatisfied : SourceSatisfied assignment)
    (index : Nat) (bounded : index < laneFields) :
    assignment (28041931 + index) = nebulaLaneSourceValue assignment index := by
  have groups := rows_imply_decoder_groups assignment canonical one
    sourceSatisfied
  have programDigest : nebulaProgramDigestBlock.Holds assignment := by
    exact groups (DecoderGroup.block nebulaProgramDigestBlock)
      nebulaProgramDigestBlock_member
  have isOpen : nebulaOpenBlock.Holds assignment := by
    exact groups (DecoderGroup.block nebulaOpenBlock) nebulaOpenBlock_member
  have counters : nebulaCountersBlock.Holds assignment := by
    exact groups (DecoderGroup.block nebulaCountersBlock)
      nebulaCountersBlock_member
  have timestamp : nebulaTimestampBlock.Holds assignment := by
    exact groups (DecoderGroup.block nebulaTimestampBlock)
      nebulaTimestampBlock_member
  have state : nebulaStateBlock.Holds assignment := by
    exact groups (DecoderGroup.block nebulaStateBlock) nebulaStateBlock_member
  by_cases digestRange : index < 4
  · simpa [nebulaLaneSourceValue, digestRange, nebulaProgramDigestBlock] using
      programDigest index (by
        simpa [nebulaProgramDigestBlock, DecoderBlock.count, Range.length]
          using digestRange)
  by_cases openRange : index < 5
  · have localBound : index - 4 < nebulaOpenBlock.count := by
      norm_num [nebulaOpenBlock, DecoderBlock.count, Range.length]
      omega
    have value := isOpen (index - 4) localBound
    have columnExact :
        nebulaOpenBlock.decodedColumns.start + (index - 4) =
          28041931 + index := by
      norm_num [nebulaOpenBlock]
      omega
    rw [columnExact] at value
    simpa [nebulaLaneSourceValue, digestRange, openRange] using value
  by_cases counterRange : index < 7
  · have localBound : index - 5 < nebulaCountersBlock.count := by
      norm_num [nebulaCountersBlock, DecoderBlock.count, Range.length]
      omega
    have value := counters (index - 5) localBound
    have columnExact :
        nebulaCountersBlock.decodedColumns.start + (index - 5) =
          28041931 + index := by
      norm_num [nebulaCountersBlock]
      omega
    rw [columnExact] at value
    simpa [nebulaLaneSourceValue, digestRange, openRange, counterRange] using value
  by_cases timestampRange : index < 8
  · have localBound : index - 7 < nebulaTimestampBlock.count := by
      norm_num [nebulaTimestampBlock, DecoderBlock.count, Range.length]
      omega
    have value := timestamp (index - 7) localBound
    have columnExact :
        nebulaTimestampBlock.decodedColumns.start + (index - 7) =
          28041931 + index := by
      norm_num [nebulaTimestampBlock]
      omega
    rw [columnExact] at value
    simpa [nebulaLaneSourceValue, digestRange, openRange, counterRange,
      timestampRange] using value
  · have localBound : index - 8 < nebulaStateBlock.count := by
      norm_num [nebulaStateBlock, DecoderBlock.count, Range.length,
        laneFields] at bounded ⊢
      omega
    have value := state (index - 8) localBound
    have columnExact :
        nebulaStateBlock.decodedColumns.start + (index - 8) =
          28041931 + index := by
      norm_num [nebulaStateBlock]
      omega
    rw [columnExact] at value
    simpa [nebulaLaneSourceValue, digestRange, openRange, counterRange,
      timestampRange] using value

private theorem laneAt_eq_of_agree
    (left right : Nat → Nat) (leftColumns rightColumns : List Nat)
    (agree : ∀ index, index < laneFields →
      left (leftColumns.getD index 0) =
        right (rightColumns.getD index 0)) :
    laneAt left leftColumns = laneAt right rightColumns := by
  have fieldAgree (index : Nat) (bounded : index < laneFields) :
      fieldAt left leftColumns index = fieldAt right rightColumns index := by
    apply Fin.ext
    exact congrArg (fun value => value % goldilocksModulus)
      (agree index bounded)
  have digestAgree (start : Nat) (bounded : start + 3 < laneFields) :
      digestAt left leftColumns start = digestAt right rightColumns start := by
    funext coordinate
    exact fieldAgree (start + coordinate.val) (by omega)
  have kAgree (start : Nat) (bounded : start + 1 < laneFields) :
      kAt left leftColumns start = kAt right rightColumns start := by
    change K.mk (fieldAt left leftColumns start)
        (fieldAt left leftColumns (start + 1)) =
      K.mk (fieldAt right rightColumns start)
        (fieldAt right rightColumns (start + 1))
    rw [fieldAgree start (by omega), fieldAgree (start + 1) bounded]
  apply StreamingTerminalFullFinalizerTransitionRelation.Lane.ext
  · exact digestAgree 0 (by norm_num [laneFields])
  · exact congrArg (fun value => value == 1) (agree 4 (by norm_num [laneFields]))
  · exact agree 5 (by norm_num [laneFields])
  · exact agree 6 (by norm_num [laneFields])
  · exact agree 7 (by norm_num [laneFields])
  · funext index
    exact kAgree (8 + 2 * index.val) (by simp [laneFields] <;> omega)
  · funext index
    exact kAgree (12 + 2 * index.val) (by simp [laneFields] <;> omega)
  · funext index
    exact agree (20 + index.val) (by simp [laneFields] <;> omega)
  · funext index
    exact digestAgree (22 + 4 * index.val) (by simp [laneFields] <;> omega)
  · funext index
    exact digestAgree (34 + 4 * index.val) (by simp [laneFields] <;> omega)
  · exact digestAgree 46 (by norm_num [laneFields])

private theorem dPreWordValue_eq_source
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (sourceSatisfied : SourceSatisfied assignment)
    (index : Nat) (bounded : index < dPreWordCount) :
    dPreWordValue assignment index = sourceDPreWordValue assignment index := by
  unfold sourceDPreWordValue dPreWordValue
  apply Nightstream.Implementation.R1CS.Canonical.KMulHonest.lcEval_congr
  intro column mentioned
  simp only [Nightstream.Implementation.R1CS.Canonical.LinCombNormal.Mentions,
    RawArtifact.dPreWordTerms, RawArtifact.wordTerms, List.map_map,
    Function.comp_apply, Prod.fst] at mentioned
  rcases List.mem_map.mp mentioned with ⟨bit, bitMember, rfl⟩
  have bitBound : bit < 64 := by simpa using bitMember
  have payloadBound : dPreBitStart + 64 * index + bit < delayedPayloadFields := by
    norm_num [dPreBitStart, openBitIndex, stepBitFields, dPreWordCount,
      delayedPayloadFields] at bounded ⊢
    omega
  change assignment
      (finalizerArtifact.payloadColumn (dPreBitStart + 64 * index + bit)) =
    payloadSourceAssignment assignment
      (finalizerArtifact.payloadColumn (dPreBitStart + 64 * index + bit))
  rw [payload_column_exact _ payloadBound]
  unfold payloadSourceAssignment
  simp only [Nat.add_sub_cancel_left]
  exact rows_imply_delayedPayloadSource assignment canonical one
    sourceSatisfied _ (by
      simpa [delayedPayloadBlock, DecoderBlock.count, Range.length,
        delayedPayloadFields] using payloadBound)

private theorem stepWordValue_eq_source
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (sourceSatisfied : SourceSatisfied assignment)
    (index : Nat) (bounded : index < stepWordCount) :
    stepWordValue assignment index =
      sourceStepWordValue assignment index := by
  unfold sourceStepWordValue stepWordValue
  apply Nightstream.Implementation.R1CS.Canonical.KMulHonest.lcEval_congr
  intro column mentioned
  simp only [Nightstream.Implementation.R1CS.Canonical.LinCombNormal.Mentions,
    RawArtifact.stepWordTerms, RawArtifact.wordTerms, List.map_map,
    Function.comp_apply, Prod.fst] at mentioned
  rcases List.mem_map.mp mentioned with ⟨bit, bitMember, rfl⟩
  have bitBound : bit < stepWordWidths.getD index 0 := by
    simpa using bitMember
  have payloadBound :
      wordStart stepWordWidths index + bit < delayedPayloadFields := by
    norm_num [stepWordCount, stepWordWidths] at bounded
    interval_cases index <;>
      norm_num [wordStart, stepWordWidths, delayedPayloadFields]
        at bitBound ⊢ <;>
      omega
  change assignment
      (finalizerArtifact.payloadColumn
        (wordStart stepWordWidths index + bit)) =
    payloadSourceAssignment assignment
      (finalizerArtifact.payloadColumn
        (wordStart stepWordWidths index + bit))
  rw [payload_column_exact _ payloadBound]
  unfold payloadSourceAssignment
  simp only [Nat.add_sub_cancel_left]
  exact rows_imply_delayedPayloadSource assignment canonical one
    sourceSatisfied _ (by
      simpa [delayedPayloadBlock, DecoderBlock.count, Range.length,
        delayedPayloadFields] using payloadBound)

/-- The exact Rust source-binding rows make the finalizer consume the same
delayed-open bit as the final selective assignment. -/
theorem rows_bind_delayedOpen
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (sourceSatisfied : SourceSatisfied assignment) :
    delayedOpen assignment = delayedOpenSource assignment := by
  have valueExact := rows_imply_delayedOpenSource assignment canonical one
    sourceSatisfied
  simp only [delayedOpen, delayedOpenSource, open_column_exact]
  rw [valueExact]

/-- Exact terminal source-binding rows make the finalizer consume the typed
50-field Nebula lane reconstructed from the final selective source encoding. -/
theorem rows_bind_postPhaseLane
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (sourceSatisfied : SourceSatisfied assignment) :
    postPhaseLane assignment = sourceLane assignment := by
  unfold postPhaseLane sourceLane
  apply laneAt_eq_of_agree
  intro index bounded
  rw [lane_column_exact index bounded,
    range_getD_eq laneFields index 0 bounded]
  exact rows_bind_nebulaLaneSourceValue assignment canonical one
    sourceSatisfied index bounded

/-- Exact source-binding and finalizer decode rows recover the complete typed
delayed step from the final selective source bits. -/
theorem rows_bind_stepInput
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (sourceSatisfied : SourceSatisfied assignment)
    (decodeSatisfied : DecodeSatisfied assignment) :
    stepInput assignment = sourceStep assignment := by
  have decodeSound :=
    StreamingTerminalFullFinalizerDecodeRowSound.rows_sound assignment
      canonical one decodeSatisfied
  have wordExact (index : Nat) (bounded : index < stepWordCount) :
      assignment (finalizerArtifact.stepWordColumn index) =
        sourceStepWordValue assignment index :=
    (decodeSound.stepWords index bounded).trans
      (stepWordValue_eq_source assignment canonical one sourceSatisfied
        index bounded)
  have fieldExact (index : Nat) (bounded : index < stepWordCount) :
      stepFieldAt assignment index = sourceStepFieldAt assignment index := by
    apply Fin.ext
    exact congrArg (fun value => value % goldilocksModulus)
      (wordExact index bounded)
  have kExact (start : Nat) (bounded : start + 1 < stepWordCount) :
      stepKAt assignment start = sourceStepKAt assignment start := by
    exact congrArg₂ K.mk
      (fieldExact start (by omega))
      (fieldExact (start + 1) bounded)
  apply stepInput_eq_of_fields
  · exact wordExact 0 (by norm_num [stepWordCount, stepWordWidths])
  · exact wordExact 1 (by norm_num [stepWordCount, stepWordWidths])
  · exact wordExact 2 (by norm_num [stepWordCount, stepWordWidths])
  · exact wordExact 3 (by norm_num [stepWordCount, stepWordWidths])
  · funext index
    exact kExact (4 + 2 * index.val)
      (by simp [stepWordCount, stepWordWidths] <;> omega)
  · funext index
    exact kExact (8 + 2 * index.val)
      (by simp [stepWordCount, stepWordWidths] <;> omega)
  · funext index
    exact kExact (16 + 2 * index.val)
      (by simp [stepWordCount, stepWordWidths] <;> omega)
  · funext _
    exact decodeSound.constantZero
  · funext _
    exact decodeSound.constantZero

/-- Exact source-binding and finalizer decode rows recover the same delayed
`D_pre` digest family that the gamma mux consumes on the open branch. -/
theorem rows_bind_candidateDPre
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (sourceSatisfied : SourceSatisfied assignment)
    (decodeSatisfied : DecodeSatisfied assignment) :
    candidateDPre assignment = sourceDPre assignment := by
  have decodeSound :=
    StreamingTerminalFullFinalizerDecodeRowSound.rows_sound assignment
      canonical one decodeSatisfied
  funext lane coordinate
  simp only [candidateDPre, sourceDPre, digestAt, fieldAt, columnAt]
  let wordIndex := 4 * lane.val + coordinate.val
  have wordBound : wordIndex < dPreWordCount := by
    simp [wordIndex, dPreWordCount] <;> omega
  have columnExact :
      finalizerArtifact.gammaMuxOpenedColumns.getD (4 + wordIndex) 0 =
        finalizerArtifact.dPreWordColumn wordIndex := by
    simpa [wordIndex] using
      gamma_mux_opened_dPre_column ⟨wordIndex, wordBound⟩
  have indexExact :
      4 + 4 * lane.val + coordinate.val = 4 + wordIndex := by
    simp [wordIndex, Nat.add_assoc]
  rw [indexExact]
  change fieldValue assignment
      (finalizerArtifact.gammaMuxOpenedColumns.getD (4 + wordIndex) 0) =
    fieldValue (sourceDPreWordValue assignment) wordIndex
  rw [columnExact]
  apply Fin.ext
  change assignment (finalizerArtifact.dPreWordColumn wordIndex) %
      goldilocksModulus = sourceDPreWordValue assignment wordIndex %
        goldilocksModulus
  rw [decodeSound.dPreWords wordIndex wordBound,
    dPreWordValue_eq_source assignment canonical one sourceSatisfied
      wordIndex wordBound]

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerSourceBridge
