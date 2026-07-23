import Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
import Nightstream.Implementation.Lowering.Goldilocks.Codec

/-!
Contract: codec-derived widths for the fixed-one step and terminal lowering.

Owns: one explicit semantic codec for every fixed-one data tag, the resulting
Goldilocks codec family, and derivation of all eleven logical widths.

Does not own: call footprints, physical rows, protocol-call recipes, Rust
layouts, or generated artifacts.  A later production profile must instantiate
these codecs for its concrete data types.

The `Vocabulary.Parameters.widths` field is accepted only with equality to
`derivedWidths`; it is never an independent cost authority.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

/-- Selected Goldilocks-coordinate codecs for every semantic carrier used by
the fixed-one verifier.  Naturals and Booleans use the canonical codecs and
therefore are deliberately absent from this record. -/
structure DataCodecs (parameters : Parameters) where
  field : Codec parameters.Field
  digest : Codec parameters.Digest
  state : Codec parameters.State
  witness : Codec parameters.Witness
  running : Codec parameters.Running
  fresh : Codec parameters.Fresh
  nifsProof : Codec parameters.NifsProof
  encoded : Codec parameters.Encoded
  runningWitness : Codec parameters.RunningWitness
  freshWitness : Codec parameters.FreshWitness

namespace DataCodecs

/-- Complete family selected by semantic kind, never by a column number. -/
def family (parameters : Parameters) (codecs : DataCodecs parameters) :
    Family (typeSystem parameters) where
  field := codecs.field
  bit := boolCodec
  data := fun tag =>
    match tag with
    | .nat => boundedNatCodec
    | .digest => codecs.digest
    | .state => codecs.state
    | .witness => codecs.witness
    | .running => codecs.running
    | .fresh => codecs.fresh
    | .nifsProof => codecs.nifsProof
    | .encoded => codecs.encoded
    | .runningWitness => codecs.runningWitness
    | .freshWitness => codecs.freshWitness

/-- Every logical width is computed from the selected codec. -/
def derivedWidths (parameters : Parameters)
    (codecs : DataCodecs parameters) : Widths where
  iteration := boundedNatCodec.width
  state := codecs.state.width
  witness := codecs.witness.width
  running := codecs.running.width
  fresh := codecs.fresh.width
  nifsProof := codecs.nifsProof.width
  digest := codecs.digest.width
  encoded := codecs.encoded.width
  runningWitness := codecs.runningWitness.width
  freshWitness := codecs.freshWitness.width
  bit := boolCodec.width

@[simp] theorem derived_iteration_width
    (parameters : Parameters) (codecs : DataCodecs parameters) :
    (codecs.derivedWidths parameters).iteration = 1 :=
  rfl

@[simp] theorem derived_bit_width
    (parameters : Parameters) (codecs : DataCodecs parameters) :
    (codecs.derivedWidths parameters).bit = 1 :=
  rfl

end DataCodecs

/-- Concrete width-selection boundary for one fixed-one parameter set. -/
structure Profile (parameters : Parameters) where
  codecs : DataCodecs parameters
  widthsExact : parameters.widths = codecs.derivedWidths parameters

namespace Profile

def family (parameters : Parameters) (profile : Profile parameters) :
    Family (typeSystem parameters) :=
  profile.codecs.family parameters

theorem iteration_width_eq_one
    (parameters : Parameters) (profile : Profile parameters) :
    parameters.widths.iteration = 1 := by
  rw [profile.widthsExact]
  rfl

theorem bit_width_eq_one
    (parameters : Parameters) (profile : Profile parameters) :
    parameters.widths.bit = 1 := by
  rw [profile.widthsExact]
  rfl

theorem state_width_eq_codec
    (parameters : Parameters) (profile : Profile parameters) :
    parameters.widths.state = profile.codecs.state.width := by
  rw [profile.widthsExact]
  rfl

theorem witness_width_eq_codec
    (parameters : Parameters) (profile : Profile parameters) :
    parameters.widths.witness = profile.codecs.witness.width := by
  rw [profile.widthsExact]
  rfl

theorem running_width_eq_codec
    (parameters : Parameters) (profile : Profile parameters) :
    parameters.widths.running = profile.codecs.running.width := by
  rw [profile.widthsExact]
  rfl

theorem fresh_width_eq_codec
    (parameters : Parameters) (profile : Profile parameters) :
    parameters.widths.fresh = profile.codecs.fresh.width := by
  rw [profile.widthsExact]
  rfl

theorem nifsProof_width_eq_codec
    (parameters : Parameters) (profile : Profile parameters) :
    parameters.widths.nifsProof = profile.codecs.nifsProof.width := by
  rw [profile.widthsExact]
  rfl

theorem digest_width_eq_codec
    (parameters : Parameters) (profile : Profile parameters) :
    parameters.widths.digest = profile.codecs.digest.width := by
  rw [profile.widthsExact]
  rfl

theorem encoded_width_eq_codec
    (parameters : Parameters) (profile : Profile parameters) :
    parameters.widths.encoded = profile.codecs.encoded.width := by
  rw [profile.widthsExact]
  rfl

theorem runningWitness_width_eq_codec
    (parameters : Parameters) (profile : Profile parameters) :
    parameters.widths.runningWitness =
      profile.codecs.runningWitness.width := by
  rw [profile.widthsExact]
  rfl

theorem freshWitness_width_eq_codec
    (parameters : Parameters) (profile : Profile parameters) :
    parameters.widths.freshWitness = profile.codecs.freshWitness.width := by
  rw [profile.widthsExact]
  rfl

end Profile

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
