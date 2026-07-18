import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.SourceLayout
import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Sis.Refinement

/-!
Concrete two-layer SIS refinement for the historical three-matrix `Pi_CCS`
output-digest artifact.

Despite the legacy file name, this module fixes 15 sources, 3 matrices, and
6,683 serialized fields. It is not the active
15-source/13-matrix/23,033-field fixed-point target and cannot discharge
active-profile refinement.

Assurance tier: implementation/R1CS correspondence. Starting from accepted
owner rows, this file proves that the primary and compression output columns
are exactly the results of the assignment-free SIS linear-map semantics on
the independently typed terminal-output serialization.

Owns: the typed serialization-to-primary-word bridge; exact primary block
refinement; the primary-output-to-compression-word bridge; exact compression
block refinement; and their coordinate-wise and flattened composition.

Does not own: proof that dynamic serialization columns are authoritative
`Pi_CCS` verifier outputs; public-seed-to-coefficient conformance; Rust/ChaCha
stream equivalence; the Poseidon2 digest envelope; transcript authority;
collision resistance; row necessity; row removal; or cost totals.

Emits constraints: no.

Authority boundary: the right-hand side contains only the independently typed
message serialization and explicit `LinearMap` values projected from the two
diagnostic blocks. The coefficient functions are not yet called public-seed
derived; that is a separate cross-language theorem obligation.

| Protocol | Phase | Constraint family | Theorem | Exact guarantee |
|---|---|---|---|---|
| `Pi_CCS` | output digest | primary canonical words | `accepted_primaryWordAgreement` | all 6,683 block inputs are canonical digits of the typed serialization |
| `Pi_CCS` | output digest | primary SIS | `accepted_primaryOutputs` | all 108 primary outputs equal the independent rank-2 linear map |
| `Pi_CCS` | output digest | compression canonical words | `accepted_compressionWordAgreement` | all 108 compression inputs canonically encode those primary outputs |
| `Pi_CCS` | output digest | compression SIS | `accepted_composedOutputs` | all 54 compression outputs equal the exact two-map composition |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.ProductionBinding

open Nightstream.Implementation.R1CS

set_option maxRecDepth 1048576
set_option maxHeartbeats 8000000

abbrev primaryBlock : SeededPhi81.Block :=
  FPrimeFullHistorySeededPhi81.block8

abbrev compressionBlock : SeededPhi81.Block :=
  FPrimeFullHistorySeededPhi81.block9

/-- The independent typed terminal-output serialization, projected to
canonical natural-number representatives only at the SIS boundary. -/
def serializedValues
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) : List Nat :=
  (Semantics.serializeTerminalOutputs
    (SourceLayout.decodedOutputs assignment canonical)).map Fin.val

/-- The exact production primary-output columns, used as the compression
message before composition is substituted. -/
def primaryOutputValues (assignment : Nat -> Nat) : List Nat :=
  primaryBlock.outputColumns.map assignment

private theorem map_getD_eq_of_lt
    (values : List Nat) (assignment : Nat -> Nat) (index : Nat)
    (indexLt : index < values.length) :
    (values.map assignment).getD index 0 =
      assignment (values.getD index 0) := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_map,
    List.getElem?_eq_getElem indexLt]
  simp only [Option.map_some, Option.getD_some]
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem indexLt]
  simp

/-- Accepted source-layout rows identify the natural representatives of the
typed serialization with the exact 6,683 primary source-column values. -/
theorem accepted_serializedValues_eq_columns
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : EncodingSchedule.ArtifactAccepted assignment) :
    serializedValues assignment canonical =
      EncodingSchedule.mainFieldColumns.map assignment := by
  unfold serializedValues
  rw [SourceLayout.accepted_decodedSerialization canonical one accepted]
  simp [SourceLayout.fieldAt]

/-- Exact canonical-word contract for the primary rank-2 block. -/
theorem accepted_primaryWordAgreement
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : EncodingSchedule.ArtifactAccepted assignment) :
    Refinement.WordAgreement primaryBlock
      (serializedValues assignment canonical) assignment := by
  refine
    { wordWidth := ?_
      fieldCount := ?_
      digit := ?_ }
  · simpa [Sis.Semantics.digitCount] using
      EncodingSchedule.primaryWordWidth
  · have valuesEq :=
      accepted_serializedValues_eq_columns canonical one accepted
    calc
      (serializedValues assignment canonical).length =
          (EncodingSchedule.mainFieldColumns.map assignment).length :=
        congrArg List.length valuesEq
      _ = EncodingSchedule.mainFieldColumns.length := by simp
      _ = EncodingSchedule.mainFieldCount :=
        EncodingSchedule.mainFieldColumns_length
      _ = EncodingSchedule.mainDigitStarts.length :=
        EncodingSchedule.mainDigitStarts_length.symm
      _ = primaryBlock.wordStarts.length := by
        exact congrArg List.length
          EncodingSchedule.mainDigitStarts_eq_primaryWordStarts
  · intro wordIndex digitIndex wordIndexLt digitIndexLt
    have valuesEq :=
      accepted_serializedValues_eq_columns canonical one accepted
    have wordIndexLtCount : wordIndex < EncodingSchedule.mainFieldCount := by
      have bound := wordIndexLt
      rw [valuesEq, List.length_map,
        EncodingSchedule.mainFieldColumns_length] at bound
      exact bound
    have fieldIndexLt :
        wordIndex < EncodingSchedule.mainFieldColumns.length := by
      simpa [EncodingSchedule.mainFieldColumns_length] using wordIndexLtCount
    have sourceAt :
        (serializedValues assignment canonical).getD wordIndex 0 =
          assignment
            (EncodingSchedule.mainFieldColumns.getD wordIndex 0) := by
      rw [valuesEq]
      exact map_getD_eq_of_lt EncodingSchedule.mainFieldColumns assignment
        wordIndex fieldIndexLt
    have productionDigitLt :
        digitIndex < ShiftedTernaryCompiler.digitCount := by
      simpa [Sis.Semantics.digitCount,
        ShiftedTernaryCompiler.digitCount] using digitIndexLt
    have wordAt := EncodingSchedule.accepted_mainWordAt prime canonical one
      accepted wordIndex digitIndex wordIndexLtCount productionDigitLt
    change assignment (primaryBlock.wordStarts.getD wordIndex 0 + digitIndex) =
      Sis.Semantics.canonicalDigit
        ((serializedValues assignment canonical).getD wordIndex 0) digitIndex
    rw [← EncodingSchedule.mainDigitStarts_eq_primaryWordStarts, sourceAt,
      Refinement.abstractDigit_eq_productionDigit]
    exact wordAt

/-- Coordinate-level primary SIS refinement. -/
theorem accepted_primaryCoordinate
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : EncodingSchedule.ArtifactAccepted assignment)
    (output coordinate : Nat) (outputLt : output < primaryBlock.kappa)
    (coordinateLt : coordinate < Sis.Semantics.dimension) :
    assignment
        (primaryBlock.outputColumns.getD
          (output * SeededPhi81.dimension + coordinate) 0) =
      Sis.Semantics.applyCoordinate (Refinement.mapOfBlock primaryBlock)
        (serializedValues assignment canonical) output coordinate := by
  exact Refinement.outputCoordinate_eq
    (EncodingSchedule.accepted_primaryCommitment accepted)
    (accepted_primaryWordAgreement prime canonical one accepted)
    output coordinate outputLt coordinateLt

private theorem primaryOutputColumnAt (index : Nat) (indexLt : index < 108) :
    primaryBlock.outputColumns.getD index 0 = 1714499 + index := by
  change ((List.range 108).map fun position =>
    1714499 + position * 1).getD index 0 = 1714499 + index
  simp only [List.getD_eq_getElem?_getD, List.getElem?_map]
  rw [List.getElem?_range indexLt]
  simp

/-- Flattened exact primary-map result, in production output-column order. -/
theorem accepted_primaryOutputs
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : EncodingSchedule.ArtifactAccepted assignment) :
    primaryOutputValues assignment =
      Sis.Semantics.apply (Refinement.mapOfBlock primaryBlock)
        (serializedValues assignment canonical) := by
  change
    (List.range 108).map (fun index => assignment (1714499 + index)) =
      (List.range 2).flatMap fun output =>
        (List.range Sis.Semantics.dimension).map fun coordinate =>
          Sis.Semantics.applyCoordinate (Refinement.mapOfBlock primaryBlock)
            (serializedValues assignment canonical) output coordinate
  have outputZero :
      (List.range 54).map (fun coordinate =>
        assignment (1714499 + coordinate)) =
      (List.range 54).map (fun coordinate =>
        Sis.Semantics.applyCoordinate (Refinement.mapOfBlock primaryBlock)
          (serializedValues assignment canonical) 0 coordinate) := by
    apply List.map_congr_left
    intro coordinate coordinateMember
    have coordinateLt := List.mem_range.mp coordinateMember
    have refined := accepted_primaryCoordinate prime canonical one accepted
      0 coordinate (by decide)
      (by simpa [Sis.Semantics.dimension] using coordinateLt)
    rw [show 0 * SeededPhi81.dimension + coordinate = coordinate by simp,
      primaryOutputColumnAt coordinate (by omega)] at refined
    exact refined
  have outputOne :
      (List.range 54).map
        ((fun index => assignment (1714499 + index)) ∘
          fun coordinate => 54 + coordinate) =
      (List.range 54).map (fun coordinate =>
        Sis.Semantics.applyCoordinate (Refinement.mapOfBlock primaryBlock)
          (serializedValues assignment canonical) 1 coordinate) := by
    apply List.map_congr_left
    intro coordinate coordinateMember
    have coordinateLt := List.mem_range.mp coordinateMember
    have refined := accepted_primaryCoordinate prime canonical one accepted
      1 coordinate (by decide)
      (by simpa [Sis.Semantics.dimension] using coordinateLt)
    rw [show 1 * SeededPhi81.dimension + coordinate = 54 + coordinate by
        simp [SeededPhi81.dimension, SeededPhi81Sampler.dimension],
      primaryOutputColumnAt (54 + coordinate) (by omega)] at refined
    exact refined
  rw [show 108 = 54 + 54 by decide, List.range_add, List.map_append,
    List.map_map]
  rw [show List.range 2 = [0, 1] by decide]
  simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil]
  simp only [Sis.Semantics.dimension]
  rw [outputZero, outputOne]

/-- Exact canonical-word contract for the rank-1 compression block. -/
theorem accepted_compressionWordAgreement
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : EncodingSchedule.ArtifactAccepted assignment) :
    Refinement.WordAgreement compressionBlock
      (primaryOutputValues assignment) assignment := by
  refine
    { wordWidth := ?_
      fieldCount := ?_
      digit := ?_ }
  · simpa [Sis.Semantics.digitCount] using
      EncodingSchedule.compressionWordWidth
  · simp [primaryOutputValues]
    decide
  · intro wordIndex digitIndex wordIndexLt digitIndexLt
    have valuesLength :
        (primaryOutputValues assignment).length =
          EncodingSchedule.compressionFieldCount := by
      simp [primaryOutputValues, EncodingSchedule.compressionFieldCount]
      decide
    have wordIndexLtCount :
        wordIndex < EncodingSchedule.compressionFieldCount := by
      simpa [valuesLength] using wordIndexLt
    have primaryIndexLt : wordIndex < primaryBlock.outputColumns.length := by
      simpa [primaryOutputValues] using wordIndexLt
    have sourceAt :
        (primaryOutputValues assignment).getD wordIndex 0 =
          assignment (primaryBlock.outputColumns.getD wordIndex 0) := by
      unfold primaryOutputValues
      exact map_getD_eq_of_lt primaryBlock.outputColumns assignment
        wordIndex primaryIndexLt
    have productionDigitLt :
        digitIndex < ShiftedTernaryCompiler.digitCount := by
      simpa [Sis.Semantics.digitCount,
        ShiftedTernaryCompiler.digitCount] using digitIndexLt
    have wordAt := EncodingSchedule.accepted_compressionWordAt prime
      canonical one accepted wordIndex digitIndex wordIndexLtCount
      productionDigitLt
    change assignment
        (compressionBlock.wordStarts.getD wordIndex 0 + digitIndex) =
      Sis.Semantics.canonicalDigit
        ((primaryOutputValues assignment).getD wordIndex 0) digitIndex
    rw [← EncodingSchedule.compressionDigitStarts_eq_compressionWordStarts,
      sourceAt, ← EncodingSchedule.compressionFieldColumns_eq_primaryOutputs,
      Refinement.abstractDigit_eq_productionDigit]
    exact wordAt

/-- Coordinate-level compression SIS refinement. -/
theorem accepted_compressionCoordinate
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : EncodingSchedule.ArtifactAccepted assignment)
    (output coordinate : Nat) (outputLt : output < compressionBlock.kappa)
    (coordinateLt : coordinate < Sis.Semantics.dimension) :
    assignment
        (compressionBlock.outputColumns.getD
          (output * SeededPhi81.dimension + coordinate) 0) =
      Sis.Semantics.applyCoordinate (Refinement.mapOfBlock compressionBlock)
        (primaryOutputValues assignment) output coordinate := by
  exact Refinement.outputCoordinate_eq
    (EncodingSchedule.accepted_compressionCommitment accepted)
    (accepted_compressionWordAgreement prime canonical one accepted)
    output coordinate outputLt coordinateLt

private theorem compressionOutputColumnAt
    (index : Nat) (indexLt : index < 54) :
    compressionBlock.outputColumns.getD index 0 = 2529935 + index := by
  change ((List.range 54).map fun position =>
    2529935 + position * 1).getD index 0 = 2529935 + index
  simp only [List.getD_eq_getElem?_getD, List.getElem?_map]
  rw [List.getElem?_range indexLt]
  simp

/-- Flattened exact compression-map result before substituting the primary
map. -/
theorem accepted_compressionOutputs
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : EncodingSchedule.ArtifactAccepted assignment) :
    compressionBlock.outputColumns.map assignment =
      Sis.Semantics.apply (Refinement.mapOfBlock compressionBlock)
        (primaryOutputValues assignment) := by
  change
    (List.range 54).map (fun index => assignment (2529935 + index)) =
      (List.range 1).flatMap fun output =>
        (List.range Sis.Semantics.dimension).map fun coordinate =>
          Sis.Semantics.applyCoordinate
            (Refinement.mapOfBlock compressionBlock)
            (primaryOutputValues assignment) output coordinate
  rw [show List.range 1 = [0] by decide, List.flatMap_cons,
    List.flatMap_nil, List.append_nil]
  apply List.map_congr_left
  intro coordinate coordinateMember
  have coordinateLt := List.mem_range.mp coordinateMember
  have refined := accepted_compressionCoordinate prime canonical one accepted
    0 coordinate (by decide)
    (by simpa [Sis.Semantics.dimension] using coordinateLt)
  rw [show 0 * SeededPhi81.dimension + coordinate = coordinate by simp,
    compressionOutputColumnAt coordinate coordinateLt] at refined
  exact refined

/-- End-to-end two-layer SIS equality. This is still conditional on the
explicit coefficient functions of `block8` and `block9`; public-seed/Rust
conformance and the Poseidon2 envelope remain separate obligations. -/
theorem accepted_composedOutputs
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : EncodingSchedule.ArtifactAccepted assignment) :
    compressionBlock.outputColumns.map assignment =
      Sis.Semantics.apply (Refinement.mapOfBlock compressionBlock)
        (Sis.Semantics.apply (Refinement.mapOfBlock primaryBlock)
          (serializedValues assignment canonical)) := by
  rw [accepted_compressionOutputs prime canonical one accepted,
    accepted_primaryOutputs prime canonical one accepted]

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.ProductionBinding
