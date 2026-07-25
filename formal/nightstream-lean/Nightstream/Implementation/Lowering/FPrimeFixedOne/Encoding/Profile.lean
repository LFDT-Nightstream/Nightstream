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
open Nightstream.Implementation.Lowering.Typed
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

private theorem ownedDataPort_widthsAgree
    (parameters : Parameters)
    (profile : Profile parameters)
    (tag : DataTag)
    (ownership : Ownership)
    (width : Nat)
    (widthExact :
      ((profile.family parameters).codecFor (.data tag)).width = width) :
    PortWidthAgrees (profile.family parameters)
      (dataPort parameters tag (ownedLayout ownership width)) := by
  unfold PortWidthAgrees dataPort ownedLayout
  simpa using widthExact

private theorem ownedBitPort_widthsAgree
    (parameters : Parameters)
    (profile : Profile parameters)
    (ownership : Ownership)
    (width : Nat)
    (widthExact :
      ((profile.family parameters).codecFor .bit).width = width) :
    PortWidthAgrees (profile.family parameters)
      (bitPort parameters (ownedLayout ownership width)) := by
  unfold PortWidthAgrees bitPort ownedLayout
  simpa using widthExact

private theorem iteration_codec_width
    (parameters : Parameters)
    (profile : Profile parameters) :
    ((profile.family parameters).codecFor (.data .nat)).width =
      parameters.widths.iteration := by
  simpa [family, DataCodecs.family, Family.codecFor] using
    (profile.iteration_width_eq_one parameters).symm

private theorem bit_codec_width
    (parameters : Parameters)
    (profile : Profile parameters) :
    ((profile.family parameters).codecFor .bit).width =
      parameters.widths.bit := by
  simpa [family, DataCodecs.family, Family.codecFor] using
    (profile.bit_width_eq_one parameters).symm

private theorem state_codec_width
    (parameters : Parameters)
    (profile : Profile parameters) :
    ((profile.family parameters).codecFor (.data .state)).width =
      parameters.widths.state := by
  simpa [family, DataCodecs.family, Family.codecFor] using
    (profile.state_width_eq_codec parameters).symm

private theorem witness_codec_width
    (parameters : Parameters)
    (profile : Profile parameters) :
    ((profile.family parameters).codecFor (.data .witness)).width =
      parameters.widths.witness := by
  simpa [family, DataCodecs.family, Family.codecFor] using
    (profile.witness_width_eq_codec parameters).symm

private theorem running_codec_width
    (parameters : Parameters)
    (profile : Profile parameters) :
    ((profile.family parameters).codecFor (.data .running)).width =
      parameters.widths.running := by
  simpa [family, DataCodecs.family, Family.codecFor] using
    (profile.running_width_eq_codec parameters).symm

private theorem fresh_codec_width
    (parameters : Parameters)
    (profile : Profile parameters) :
    ((profile.family parameters).codecFor (.data .fresh)).width =
      parameters.widths.fresh := by
  simpa [family, DataCodecs.family, Family.codecFor] using
    (profile.fresh_width_eq_codec parameters).symm

private theorem nifsProof_codec_width
    (parameters : Parameters)
    (profile : Profile parameters) :
    ((profile.family parameters).codecFor (.data .nifsProof)).width =
      parameters.widths.nifsProof := by
  simpa [family, DataCodecs.family, Family.codecFor] using
    (profile.nifsProof_width_eq_codec parameters).symm

private theorem digest_codec_width
    (parameters : Parameters)
    (profile : Profile parameters) :
    ((profile.family parameters).codecFor (.data .digest)).width =
      parameters.widths.digest := by
  simpa [family, DataCodecs.family, Family.codecFor] using
    (profile.digest_width_eq_codec parameters).symm

private theorem encoded_codec_width
    (parameters : Parameters)
    (profile : Profile parameters) :
    ((profile.family parameters).codecFor (.data .encoded)).width =
      parameters.widths.encoded := by
  simpa [family, DataCodecs.family, Family.codecFor] using
    (profile.encoded_width_eq_codec parameters).symm

private theorem runningWitness_codec_width
    (parameters : Parameters)
    (profile : Profile parameters) :
    ((profile.family parameters).codecFor
      (.data .runningWitness)).width =
        parameters.widths.runningWitness := by
  simpa [family, DataCodecs.family, Family.codecFor] using
    (profile.runningWitness_width_eq_codec parameters).symm

private theorem freshWitness_codec_width
    (parameters : Parameters)
    (profile : Profile parameters) :
    ((profile.family parameters).codecFor
      (.data .freshWitness)).width =
        parameters.widths.freshWitness := by
  simpa [family, DataCodecs.family, Family.codecFor] using
    (profile.freshWitness_width_eq_codec parameters).symm

theorem committedNat_widthsAgree
    (parameters : Parameters) (profile : Profile parameters) :
    PortWidthAgrees (profile.family parameters)
      (Ports.committedNat parameters) :=
  ownedDataPort_widthsAgree parameters profile .nat .committedColumn _
    (iteration_codec_width parameters profile)

theorem publicNat_widthsAgree
    (parameters : Parameters) (profile : Profile parameters) :
    PortWidthAgrees (profile.family parameters)
      (Ports.publicNat parameters) :=
  ownedDataPort_widthsAgree parameters profile .nat .publicColumn _
    (iteration_codec_width parameters profile)

theorem committedState_widthsAgree
    (parameters : Parameters) (profile : Profile parameters) :
    PortWidthAgrees (profile.family parameters)
      (Ports.committedState parameters) :=
  ownedDataPort_widthsAgree parameters profile .state .committedColumn _
    (state_codec_width parameters profile)

theorem publicState_widthsAgree
    (parameters : Parameters) (profile : Profile parameters) :
    PortWidthAgrees (profile.family parameters)
      (Ports.publicState parameters) :=
  ownedDataPort_widthsAgree parameters profile .state .publicColumn _
    (state_codec_width parameters profile)

theorem committedWitness_widthsAgree
    (parameters : Parameters) (profile : Profile parameters) :
    PortWidthAgrees (profile.family parameters)
      (Ports.committedWitness parameters) :=
  ownedDataPort_widthsAgree parameters profile .witness .committedColumn _
    (witness_codec_width parameters profile)

theorem committedRunning_widthsAgree
    (parameters : Parameters) (profile : Profile parameters) :
    PortWidthAgrees (profile.family parameters)
      (Ports.committedRunning parameters) :=
  ownedDataPort_widthsAgree parameters profile .running .committedColumn _
    (running_codec_width parameters profile)

theorem committedFresh_widthsAgree
    (parameters : Parameters) (profile : Profile parameters) :
    PortWidthAgrees (profile.family parameters)
      (Ports.committedFresh parameters) :=
  ownedDataPort_widthsAgree parameters profile .fresh .committedColumn _
    (fresh_codec_width parameters profile)

theorem committedNifsProof_widthsAgree
    (parameters : Parameters) (profile : Profile parameters) :
    PortWidthAgrees (profile.family parameters)
      (Ports.committedNifsProof parameters) :=
  ownedDataPort_widthsAgree parameters profile .nifsProof .committedColumn _
    (nifsProof_codec_width parameters profile)

theorem publicDigest_widthsAgree
    (parameters : Parameters) (profile : Profile parameters) :
    PortWidthAgrees (profile.family parameters)
      (Ports.publicDigest parameters) :=
  ownedDataPort_widthsAgree parameters profile .digest .publicColumn _
    (digest_codec_width parameters profile)

theorem auxiliaryDigest_widthsAgree
    (parameters : Parameters) (profile : Profile parameters) :
    PortWidthAgrees (profile.family parameters)
      (Ports.auxiliaryDigest parameters) :=
  ownedDataPort_widthsAgree parameters profile .digest .auxiliaryColumn _
    (digest_codec_width parameters profile)

theorem auxiliaryEncoded_widthsAgree
    (parameters : Parameters) (profile : Profile parameters) :
    PortWidthAgrees (profile.family parameters)
      (Ports.auxiliaryEncoded parameters) :=
  ownedDataPort_widthsAgree parameters profile .encoded .auxiliaryColumn _
    (encoded_codec_width parameters profile)

theorem committedRunningWitness_widthsAgree
    (parameters : Parameters) (profile : Profile parameters) :
    PortWidthAgrees (profile.family parameters)
      (Ports.committedRunningWitness parameters) :=
  ownedDataPort_widthsAgree parameters profile .runningWitness
    .committedColumn _ (runningWitness_codec_width parameters profile)

theorem committedFreshWitness_widthsAgree
    (parameters : Parameters) (profile : Profile parameters) :
    PortWidthAgrees (profile.family parameters)
      (Ports.committedFreshWitness parameters) :=
  ownedDataPort_widthsAgree parameters profile .freshWitness
    .committedColumn _ (freshWitness_codec_width parameters profile)

theorem auxiliaryBit_widthsAgree
    (parameters : Parameters) (profile : Profile parameters) :
    PortWidthAgrees (profile.family parameters)
      (Ports.auxiliaryBit parameters) :=
  ownedBitPort_widthsAgree parameters profile .auxiliaryColumn _
    (bit_codec_width parameters profile)

theorem schemaWidthAgrees_cons
    {parameters : Parameters}
    {port : Port (typeSystem parameters)}
    {tail : Schema (typeSystem parameters)}
    {profile : Profile parameters}
    (head : PortWidthAgrees (profile.family parameters) port)
    (rest : SchemaWidthAgrees (profile.family parameters) tail) :
    SchemaWidthAgrees (profile.family parameters) (port :: tail) := by
  intro candidate member
  rcases List.mem_cons.mp member with equal | tailMember
  · subst candidate
    exact head
  · exact rest candidate tailMember

theorem schemaWidthAgrees_append
    {parameters : Parameters}
    {left right : Schema (typeSystem parameters)}
    {profile : Profile parameters}
    (leftAgrees : SchemaWidthAgrees (profile.family parameters) left)
    (rightAgrees : SchemaWidthAgrees (profile.family parameters) right) :
    SchemaWidthAgrees (profile.family parameters) (left ++ right) := by
  intro port member
  rcases List.mem_append.mp member with leftMember | rightMember
  · exact leftAgrees port leftMember
  · exact rightAgrees port rightMember

theorem stepInputSchema_widthsAgree
    (parameters : Parameters) (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (stepInputSchema parameters) := by
  exact schemaWidthAgrees_cons
    (committedNat_widthsAgree parameters profile)
    (schemaWidthAgrees_cons
      (committedState_widthsAgree parameters profile)
      (schemaWidthAgrees_cons
        (committedState_widthsAgree parameters profile)
        (schemaWidthAgrees_cons
          (committedRunning_widthsAgree parameters profile)
          (schemaWidthAgrees_cons
            (committedFresh_widthsAgree parameters profile)
            (schemaWidthAgrees_cons
              (committedWitness_widthsAgree parameters profile)
              (schemaWidthAgrees_cons
                (committedNifsProof_widthsAgree parameters profile)
                (by intro port member; simp at member)))))))

theorem terminalInputSchema_widthsAgree
    (parameters : Parameters) (profile : Profile parameters) :
    SchemaWidthAgrees (profile.family parameters)
      (terminalInputSchema parameters) := by
  exact schemaWidthAgrees_cons
    (publicNat_widthsAgree parameters profile)
    (schemaWidthAgrees_cons
      (publicState_widthsAgree parameters profile)
      (schemaWidthAgrees_cons
        (publicState_widthsAgree parameters profile)
        (schemaWidthAgrees_cons
          (committedRunning_widthsAgree parameters profile)
          (schemaWidthAgrees_cons
            (committedRunningWitness_widthsAgree parameters profile)
            (schemaWidthAgrees_cons
              (committedFresh_widthsAgree parameters profile)
              (schemaWidthAgrees_cons
                (committedFreshWitness_widthsAgree parameters profile)
                (by intro port member; simp at member)))))))

theorem callOutputs_widthsAgree
    (parameters : Parameters)
    (profile : Profile parameters)
    (call : Call) :
    SchemaWidthAgrees (profile.family parameters)
      ((signature parameters).callOutputs call) := by
  cases call with
  | iterationZero =>
      exact schemaWidthAgrees_cons
        (auxiliaryBit_widthsAgree parameters profile)
        (by intro port member; simp at member)
  | stateEqual =>
      exact schemaWidthAgrees_cons
        (auxiliaryBit_widthsAgree parameters profile)
        (by intro port member; simp at member)
  | step =>
      exact schemaWidthAgrees_cons
        (committedState_widthsAgree parameters profile)
        (by intro port member; simp at member)
  | hashPrior =>
      exact schemaWidthAgrees_cons
        (auxiliaryDigest_widthsAgree parameters profile)
        (by intro port member; simp at member)
  | hashNext =>
      exact schemaWidthAgrees_cons
        (publicDigest_widthsAgree parameters profile)
        (by intro port member; simp at member)
  | freshPublic =>
      exact schemaWidthAgrees_cons
        (auxiliaryEncoded_widthsAgree parameters profile)
        (by intro port member; simp at member)
  | encodeInstance =>
      exact schemaWidthAgrees_cons
        (auxiliaryEncoded_widthsAgree parameters profile)
        (by intro port member; simp at member)
  | encodedEqual =>
      exact schemaWidthAgrees_cons
        (auxiliaryBit_widthsAgree parameters profile)
        (by intro port member; simp at member)
  | nifsVerify =>
      exact schemaWidthAgrees_cons
        (committedRunning_widthsAgree parameters profile)
        (by intro port member; simp at member)
  | runningCheck =>
      exact schemaWidthAgrees_cons
        (auxiliaryBit_widthsAgree parameters profile)
        (by intro port member; simp at member)
  | freshCheck =>
      exact schemaWidthAgrees_cons
        (auxiliaryBit_widthsAgree parameters profile)
        (by intro port member; simp at member)

end Profile

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
