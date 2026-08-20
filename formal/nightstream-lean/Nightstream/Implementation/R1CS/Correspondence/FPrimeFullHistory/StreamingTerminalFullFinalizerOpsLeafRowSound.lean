import Nightstream.Implementation.R1CS.Core.ConstantPins
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafEnvelopeRowSound

/-!
Contract: complete ordered row soundness for the terminal `ops` leaf.

The exact order is verifier prefix pins, primary shape pins, primary canonical
openings and rank-two seeded rows, compression shape pins, compression
canonical openings and rank-one seeded rows, then the Poseidon2 envelope.

The result derives every digest output from named source fields and checked
rows. It does not own sampler no-rejection liveness, Module-SIS security,
collision resistance, the later `is` or `fs` leaves, or lifecycle closure.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer

namespace Primary

abbrev Sound :=
  Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafPrimaryRowSound.Sound

abbrev rowsSound :=
  Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafPrimaryRowSound.rows_sound

abbrev rows :=
  Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafPrimaryRowSound.opsPrimaryRows

end Primary

namespace Compression

abbrev Sound :=
  Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafCompressionRowSound.Sound

abbrev rowsSound :=
  Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafCompressionRowSound.rows_sound

abbrev rows :=
  Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafCompressionRowSound.opsCompressionRows

end Compression

namespace Envelope

abbrev Sound :=
  Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafEnvelopeRowSound.Sound

abbrev rowsSound :=
  Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafEnvelopeRowSound.rows_sound

abbrev rows :=
  Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafEnvelopeRowSound.opsEnvelopeRows

end Envelope

def consecutivePins (start : Nat) (values : List Nat) : List (Nat × Nat) :=
  (List.range' start values.length).zip values

def prefixPins : List (Nat × Nat) :=
  consecutivePins rawArtifact.opsLeaf.prefixConstantStartColumn
    rawArtifact.opsLeaf.prefixConstantValues

def primaryMetadataPins : List (Nat × Nat) :=
  consecutivePins rawArtifact.opsLeaf.primary.metadataStartColumn
    rawArtifact.opsLeaf.primary.metadataValues

def compressionMetadataPins : List (Nat × Nat) :=
  consecutivePins rawArtifact.opsLeaf.compression.metadataStartColumn
    rawArtifact.opsLeaf.compression.metadataValues

theorem prefixPins_canonical : ConstantPins.ValuesCanonical prefixPins := by
  decide

theorem primaryMetadataPins_canonical :
    ConstantPins.ValuesCanonical primaryMetadataPins := by
  decide

theorem compressionMetadataPins_canonical :
    ConstantPins.ValuesCanonical compressionMetadataPins := by
  decide

private theorem rowsIncluded_self (rows : List Row) :
    rowsIncluded rows rows = true := by
  unfold rowsIncluded
  apply List.all_eq_true.mpr
  intro row member
  exact decide_eq_true member

def opsLeafPieces : List (List Row) :=
  [ConstantPins.rows prefixPins,
    ConstantPins.rows primaryMetadataPins,
    Primary.rows,
    ConstantPins.rows compressionMetadataPins,
    Compression.rows,
    Envelope.rows]

def opsLeafRows : List Row := opsLeafPieces.flatten

def OpsLeafSatisfied (assignment : Nat → Nat) : Prop :=
  Satisfies opsLeafRows assignment

private theorem all_pieces_satisfied
    (assignment : Nat → Nat)
    (satisfied : OpsLeafSatisfied assignment) :
    ∀ piece ∈ opsLeafPieces, Satisfies piece assignment := by
  apply (satisfies_flatten_iff opsLeafPieces assignment).mp
  simpa only [OpsLeafSatisfied, opsLeafRows] using satisfied

structure Sound (assignment : Nat → Nat) : Prop where
  phaseStart :
    rawArtifact.opsLeaf.prefixPinRowStart =
      rawArtifact.coreRowStart + rawArtifact.openRowStop
  prefixValues : ∀ pin ∈ prefixPins, assignment pin.1 = pin.2
  primaryMetadata : ∀ pin ∈ primaryMetadataPins,
    assignment pin.1 = pin.2
  primary : Primary.Sound assignment
  compressionMetadata : ∀ pin ∈ compressionMetadataPins,
    assignment pin.1 = pin.2
  compression : Compression.Sound assignment
  envelope : Envelope.Sound assignment
  phaseStop :
    rawArtifact.opsLeaf.digestRowStop = rawArtifact.isLeaf.prefixPinRowStart

theorem rows_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : OpsLeafSatisfied assignment) :
    Sound assignment := by
  have pieces := all_pieces_satisfied assignment satisfied
  have prefixRows := pieces (ConstantPins.rows prefixPins)
    (by simp [opsLeafPieces])
  have primaryMetadataRows := pieces (ConstantPins.rows primaryMetadataPins)
    (by simp [opsLeafPieces])
  have primaryRows := pieces Primary.rows (by simp [opsLeafPieces])
  have compressionMetadataRows :=
    pieces (ConstantPins.rows compressionMetadataPins)
      (by simp [opsLeafPieces])
  have compressionRows := pieces Compression.rows (by simp [opsLeafPieces])
  have envelopeRows := pieces Envelope.rows (by simp [opsLeafPieces])
  exact {
    phaseStart := rawArtifact_valid.opsLeafStart
    prefixValues := ConstantPins.sound prefixPins_canonical
      (rowsIncluded_self _) canonical one prefixRows
    primaryMetadata := ConstantPins.sound primaryMetadataPins_canonical
      (rowsIncluded_self _) canonical one primaryMetadataRows
    primary := Primary.rowsSound assignment canonical one primaryRows
    compressionMetadata := ConstantPins.sound
      compressionMetadataPins_canonical (rowsIncluded_self _) canonical one
      compressionMetadataRows
    compression := Compression.rowsSound assignment canonical one compressionRows
    envelope := Envelope.rowsSound assignment canonical one envelopeRows
    phaseStop := rawArtifact_valid.opsLeafStop }

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafRowSound
