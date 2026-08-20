import Nightstream.Implementation.R1CS.Core.ConstantPins
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafOpeningRowSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingVariableHashRecipeConstantSound

/-!
Contract: reusable exact row soundness for one terminal commitment leaf.

The theorem owns verifier pins, canonical shifted-ternary openings, the
rank-two seeded map, the rank-one compression map, and the final Poseidon2
envelope. A `LeafValid` value supplies exact Rust-owned geometry and source
order. The digest is derived from checked rows and is never an authority
premise.

It does not own sampler no-rejection liveness, Module-SIS security, collision
resistance, or lifecycle closure.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.SeededPhi81
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernarySound
open Nightstream.Implementation.R1CS.ShiftedTernaryCanonicalWord
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipeCertificate
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingVariableHashRecipeConstantSound
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafOpeningRowSound

def consecutivePins (start : Nat) (values : List Nat) : List (Nat × Nat) :=
  (List.range' start values.length).zip values

def prefixPins (leaf : LeafHashArtifact) : List (Nat × Nat) :=
  consecutivePins leaf.prefixConstantStartColumn leaf.prefixConstantValues

def primaryMetadataPins (leaf : LeafHashArtifact) : List (Nat × Nat) :=
  consecutivePins leaf.primary.metadataStartColumn leaf.primary.metadataValues

def compressionMetadataPins (leaf : LeafHashArtifact) : List (Nat × Nat) :=
  consecutivePins leaf.compression.metadataStartColumn
    leaf.compression.metadataValues

private theorem consecutivePins_canonical
    (start : Nat) (values : List Nat)
    (canonical : ∀ value ∈ values, value < goldilocksP) :
    ConstantPins.ValuesCanonical (consecutivePins start values) := by
  intro pin member
  apply canonical pin.2
  exact (List.of_mem_zip (by
    simpa [consecutivePins] using member)).2

def openingRows (binding : SeededBindingArtifact) (count : Nat) : List Row :=
  ((List.range count).map fun index =>
    (openingPiece binding index).rows).flatten

private theorem opening_satisfied
    (binding : SeededBindingArtifact) (count : Nat)
    (assignment : Nat → Nat)
    (satisfied : Satisfies (openingRows binding count) assignment)
    (index : Nat) (bounded : index < count) :
    Satisfies (openingPiece binding index).rows assignment := by
  have pieces := (satisfies_flatten_iff
    ((List.range count).map fun current =>
      (openingPiece binding current).rows) assignment).mp (by
        simpa only [openingRows] using satisfied)
  exact pieces _ (List.mem_map.mpr
    ⟨index, List.mem_range.mpr bounded, rfl⟩)

def primaryRows (leaf : LeafHashArtifact) : List Row :=
  openingRows leaf.primary leafPrimaryFields ++ leaf.primary.block.rows

def compressionRows (leaf : LeafHashArtifact) : List Row :=
  openingRows leaf.compression leafCompressionFields ++
    leaf.compression.block.rows

abbrev recipe (leaf : LeafHashArtifact) : VariableHashRecipe :=
  leaf.envelopeRecipe

def envelopeRows (leaf : LeafHashArtifact) : List Row :=
  constantRows (recipe leaf) ++ (recipe leaf).trace.rows

def leafPieces (leaf : LeafHashArtifact) : List (List Row) :=
  [ConstantPins.rows (prefixPins leaf),
    ConstantPins.rows (primaryMetadataPins leaf),
    primaryRows leaf,
    ConstantPins.rows (compressionMetadataPins leaf),
    compressionRows leaf,
    envelopeRows leaf]

def rows (leaf : LeafHashArtifact) : List Row :=
  (leafPieces leaf).flatten

def Satisfied (leaf : LeafHashArtifact) (assignment : Nat → Nat) : Prop :=
  Satisfies (rows leaf) assignment

private theorem rowsIncluded_self (programRows : List Row) :
    rowsIncluded programRows programRows = true := by
  unfold rowsIncluded
  apply List.all_eq_true.mpr
  intro row member
  exact decide_eq_true member

private theorem all_pieces_satisfied
    (leaf : LeafHashArtifact) (assignment : Nat → Nat)
    (satisfied : Satisfied leaf assignment) :
    ∀ piece ∈ leafPieces leaf, Satisfies piece assignment := by
  apply (satisfies_flatten_iff (leafPieces leaf) assignment).mp
  simpa only [Satisfied, rows] using satisfied

private theorem primary_openings_satisfied
    (leaf : LeafHashArtifact) (assignment : Nat → Nat)
    (satisfied : Satisfies (primaryRows leaf) assignment) :
    Satisfies (openingRows leaf.primary leafPrimaryFields) assignment := by
  intro row member
  exact satisfied row (List.mem_append_left _ member)

private theorem primary_block_satisfied
    (leaf : LeafHashArtifact) (assignment : Nat → Nat)
    (satisfied : Satisfies (primaryRows leaf) assignment) :
    Satisfies leaf.primary.block.rows assignment := by
  intro row member
  exact satisfied row (List.mem_append_right _ member)

structure PrimarySound
    (leaf : LeafHashArtifact) (assignment : Nat → Nat) : Prop where
  openings : ∀ index, index < leafPrimaryFields →
    CanonicalOpening
      (localAssignment assignment
        (leaf.primary.sourceColumns.getD index 0)
        (leaf.primary.wordStart index))
  inputDigits : ∀ fieldIndex, fieldIndex < leafPrimaryFields →
    ∀ digitIndex, digitIndex < digitCount →
      assignment (leaf.primary.wordStart fieldIndex + digitIndex) =
        canonicalDigit
          (assignment (leaf.primary.sourceColumns.getD fieldIndex 0))
          digitIndex
  block : leaf.primary.block.Holds assignment
  outputs :
    ∀ (output : Fin leaf.primary.block.kappa)
      (coordinate : Fin dimension),
      assignment
          (leaf.primary.block.outputColumns.getD
            (output.val * dimension + coordinate.val) 0) =
        leaf.primary.block.linearValue assignment output.val coordinate.val

private theorem primary_rows_sound
    (leaf : LeafHashArtifact) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (primaryRows leaf) assignment) :
    PrimarySound leaf assignment := by
  have openingRowsSatisfied :=
    primary_openings_satisfied leaf assignment satisfied
  have blockRowsSatisfied := primary_block_satisfied leaf assignment satisfied
  have blockSound := SeededPhi81.sound canonical one blockRowsSatisfied
  exact {
    openings := fun index bounded =>
      opening_rows_sound leaf.primary index assignment canonical one
        (opening_satisfied leaf.primary leafPrimaryFields assignment
          openingRowsSatisfied index bounded)
    inputDigits := fun fieldIndex fieldBounded digitIndex digitBounded =>
      opening_digit_exact leaf.primary fieldIndex assignment canonical one
        (opening_satisfied leaf.primary leafPrimaryFields assignment
          openingRowsSatisfied fieldIndex fieldBounded)
        digitIndex digitBounded
    block := blockSound
    outputs := fun output coordinate =>
      leaf.primary.block.output_eq_linearValue blockSound output coordinate }

private theorem compression_openings_satisfied
    (leaf : LeafHashArtifact) (assignment : Nat → Nat)
    (satisfied : Satisfies (compressionRows leaf) assignment) :
    Satisfies (openingRows leaf.compression leafCompressionFields)
      assignment := by
  intro row member
  exact satisfied row (List.mem_append_left _ member)

private theorem compression_block_satisfied
    (leaf : LeafHashArtifact) (assignment : Nat → Nat)
    (satisfied : Satisfies (compressionRows leaf) assignment) :
    Satisfies leaf.compression.block.rows assignment := by
  intro row member
  exact satisfied row (List.mem_append_right _ member)

structure CompressionSound
    (leaf : LeafHashArtifact) (assignment : Nat → Nat) : Prop where
  openings : ∀ index, index < leafCompressionFields →
    CanonicalOpening
      (localAssignment assignment
        (leaf.compression.sourceColumns.getD index 0)
        (leaf.compression.wordStart index))
  inputDigits : ∀ fieldIndex, fieldIndex < leafCompressionFields →
    ∀ digitIndex, digitIndex < digitCount →
      assignment (leaf.compression.wordStart fieldIndex + digitIndex) =
        canonicalDigit
          (assignment (leaf.compression.sourceColumns.getD fieldIndex 0))
          digitIndex
  block : leaf.compression.block.Holds assignment
  outputs :
    ∀ (output : Fin leaf.compression.block.kappa)
      (coordinate : Fin dimension),
      assignment
          (leaf.compression.block.outputColumns.getD
            (output.val * dimension + coordinate.val) 0) =
        leaf.compression.block.linearValue assignment
          output.val coordinate.val

private theorem compression_rows_sound
    (leaf : LeafHashArtifact) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (compressionRows leaf) assignment) :
    CompressionSound leaf assignment := by
  have openingRowsSatisfied :=
    compression_openings_satisfied leaf assignment satisfied
  have blockRowsSatisfied :=
    compression_block_satisfied leaf assignment satisfied
  have blockSound := SeededPhi81.sound canonical one blockRowsSatisfied
  exact {
    openings := fun index bounded =>
      opening_rows_sound leaf.compression index assignment canonical one
        (opening_satisfied leaf.compression leafCompressionFields assignment
          openingRowsSatisfied index bounded)
    inputDigits := fun fieldIndex fieldBounded digitIndex digitBounded =>
      opening_digit_exact leaf.compression fieldIndex assignment canonical one
        (opening_satisfied leaf.compression leafCompressionFields assignment
          openingRowsSatisfied fieldIndex fieldBounded)
        digitIndex digitBounded
    block := blockSound
    outputs := fun output coordinate =>
      leaf.compression.block.output_eq_linearValue
        blockSound output coordinate }

private theorem envelope_constants_satisfied
    (leaf : LeafHashArtifact) (assignment : Nat → Nat)
    (satisfied : Satisfies (envelopeRows leaf) assignment) :
    Satisfies (constantRows (recipe leaf)) assignment := by
  intro row member
  exact satisfied row (List.mem_append_left _ member)

private theorem envelope_trace_satisfied
    (leaf : LeafHashArtifact) (assignment : Nat → Nat)
    (satisfied : Satisfies (envelopeRows leaf) assignment) :
    Satisfies (recipe leaf).trace.rows assignment := by
  intro row member
  exact satisfied row (List.mem_append_right _ member)

theorem envelope_input_length
    {leaf : LeafHashArtifact} {authoritativeColumns : List Nat}
    {phaseStart phaseStop : Nat}
    (valid : LeafValid leaf authoritativeColumns phaseStart phaseStop) :
    (recipe leaf).inputColumns.length = 64 := by
  change leaf.digestInputColumns.length = 64
  rw [valid.digestInputs, valid.compressionShape.2.2.2]
  norm_num [leafEnvelopeConstantFields, leafCompressionOutputs]

theorem envelope_absorbRounds_exact
    {leaf : LeafHashArtifact} {authoritativeColumns : List Nat}
    {phaseStart phaseStop : Nat}
    (valid : LeafValid leaf authoritativeColumns phaseStart phaseStop) :
    (recipe leaf).absorbRounds = 16 := by
  change ((recipe leaf).inputColumns.length + (rate - 1)) / rate = 16
  rw [envelope_input_length valid]
  decide

theorem envelope_trace_ownedValid
    {leaf : LeafHashArtifact} {authoritativeColumns : List Nat}
    {phaseStart phaseStop : Nat}
    (valid : LeafValid leaf authoritativeColumns phaseStart phaseStop) :
    (recipe leaf).trace.OwnedValid := by
  exact ownedValid (recipe leaf)
    (by rw [envelope_absorbRounds_exact valid]; decide)
    (by
      rw [envelope_input_length valid, envelope_absorbRounds_exact valid])
    valid.digestOutputExact

abbrev DigestValues := Fin digestFields → Nat

def computedDigest (leaf : LeafHashArtifact)
    (assignment : Nat → Nat) : DigestValues :=
  fun lane => runValueRounds (recipe leaf).trace.rounds
    ((recipe leaf).inputColumns.map assignment) (fun _ => 0) lane.val

def assignedDigest (leaf : LeafHashArtifact)
    (assignment : Nat → Nat) : DigestValues :=
  fun lane => assignment ((recipe leaf).outputColumns.getD lane.val 0)

structure EnvelopeSound
    (leaf : LeafHashArtifact) (assignment : Nat → Nat) : Prop where
  constants :
    (recipe leaf).constantColumns.map assignment =
      (recipe leaf).constantValues
  inputOrder :
    (recipe leaf).inputColumns =
      (recipe leaf).constantColumns ++
        leaf.compression.block.outputColumns
  compressionInputs :
    ((recipe leaf).inputColumns.drop leafEnvelopeConstantFields).map
        assignment =
      leaf.compression.block.outputColumns.map assignment
  hash : assignedDigest leaf assignment = computedDigest leaf assignment

private theorem envelope_rows_sound
    {leaf : LeafHashArtifact} {authoritativeColumns : List Nat}
    {phaseStart phaseStop : Nat}
    (valid : LeafValid leaf authoritativeColumns phaseStart phaseStop)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (envelopeRows leaf) assignment) :
    EnvelopeSound leaf assignment := by
  have constantRowsSatisfied :=
    envelope_constants_satisfied leaf assignment satisfied
  have traceRowsSatisfied := envelope_trace_satisfied leaf assignment satisfied
  have traceValid := envelope_trace_ownedValid valid
  refine {
    constants := constantRows_values (recipe leaf) assignment canonical one
      valid.envelopeCanonical constantRowsSatisfied
    inputOrder := ?_
    compressionInputs := ?_
    hash := ?_ }
  · change leaf.digestInputColumns =
      List.range' leaf.envelopeConstantStartColumn
          leaf.envelopeConstantValues.length ++
        leaf.compression.block.outputColumns
    rw [valid.envelopeCount]
    exact valid.digestInputs
  · rw [show (recipe leaf).inputColumns =
        (recipe leaf).constantColumns ++
          leaf.compression.block.outputColumns by
      change leaf.digestInputColumns =
        List.range' leaf.envelopeConstantStartColumn
            leaf.envelopeConstantValues.length ++
          leaf.compression.block.outputColumns
      rw [valid.envelopeCount]
      exact valid.digestInputs]
    have constantLength :
        (recipe leaf).constantColumns.length =
          leafEnvelopeConstantFields := by
      simpa [VariableHashRecipe.constantColumns] using valid.envelopeCount
    rw [← constantLength, List.drop_append_length]
  · funext lane
    exact ownedTrace_values_sound traceValid canonical one traceRowsSatisfied
      lane.val (by simpa [digestFields] using lane.isLt)

structure Sound
    (leaf : LeafHashArtifact) (authoritativeColumns : List Nat)
    (phaseStart phaseStop : Nat) (assignment : Nat → Nat) : Prop where
  geometry : LeafValid leaf authoritativeColumns phaseStart phaseStop
  prefixValues :
    ∀ pin ∈ prefixPins leaf, assignment pin.1 = pin.2
  primaryMetadata :
    ∀ pin ∈ primaryMetadataPins leaf, assignment pin.1 = pin.2
  primary : PrimarySound leaf assignment
  compressionMetadata :
    ∀ pin ∈ compressionMetadataPins leaf, assignment pin.1 = pin.2
  compression : CompressionSound leaf assignment
  envelope : EnvelopeSound leaf assignment

theorem rows_sound
    {leaf : LeafHashArtifact} {authoritativeColumns : List Nat}
    {phaseStart phaseStop : Nat}
    (valid : LeafValid leaf authoritativeColumns phaseStart phaseStop)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfied leaf assignment) :
    Sound leaf authoritativeColumns phaseStart phaseStop assignment := by
  have pieces := all_pieces_satisfied leaf assignment satisfied
  have prefixRows := pieces (ConstantPins.rows (prefixPins leaf))
    (by simp [leafPieces])
  have primaryMetadataRows :=
    pieces (ConstantPins.rows (primaryMetadataPins leaf))
      (by simp [leafPieces])
  have primaryStageRows := pieces (primaryRows leaf) (by simp [leafPieces])
  have compressionMetadataRows :=
    pieces (ConstantPins.rows (compressionMetadataPins leaf))
      (by simp [leafPieces])
  have compressionStageRows :=
    pieces (compressionRows leaf) (by simp [leafPieces])
  have envelopeStageRows := pieces (envelopeRows leaf) (by simp [leafPieces])
  exact {
    geometry := valid
    prefixValues := ConstantPins.sound
      (consecutivePins_canonical _ _ valid.prefixCanonical)
      (rowsIncluded_self _) canonical one prefixRows
    primaryMetadata := ConstantPins.sound
      (consecutivePins_canonical _ _ valid.primaryMetadataCanonical)
      (rowsIncluded_self _) canonical one primaryMetadataRows
    primary := primary_rows_sound leaf assignment canonical one primaryStageRows
    compressionMetadata := ConstantPins.sound
      (consecutivePins_canonical _ _ valid.compressionMetadataCanonical)
      (rowsIncluded_self _) canonical one compressionMetadataRows
    compression := compression_rows_sound leaf assignment canonical one
      compressionStageRows
    envelope := envelope_rows_sound valid assignment canonical one
      envelopeStageRows }

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafRowSound
