import Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt
import Nightstream.Implementation.NebulaV2.ProductCommitmentAlgebra
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Soundness

/-!
Contract: exact selected-verifier adapter for the paper SuperNeo NIFS with the
mandatory V2 product commitment.

Assurance tier: executable semantic verifier and exact claim-codec boundary.

Owns a fail-closed decoder from the complete V2 claim block to the paper
running/fresh input pair, a setup-owned bounded-sampler gate, a selected
verifier whose Boolean is definitionally that gate followed by the paper
verifier graph, deterministic soundness modulo the paper's five named bad
events, and conditional graph completeness outside sampler shortfall.

Does not own the final generated claim decoder, the concrete V2 paper key,
Poseidon2 random-oracle bounds, Ajtai/Module-SIS binding, generated verifier
rows, Rust, or the deployed parser. Those values must instantiate the exact
types in this module. No caller supplies a NIFS soundness or completeness
premise.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductPaperNifsSelection

open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt
open Nightstream.Protocol.NebulaV2
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

abbrev JointShape :=
  Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape

namespace Paper

abbrev Running
    (fullShape : Phi81Relation.Shape) (shape : JointShape) :=
  Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Running
    K ProductCommitmentAlgebra.BundleValue (PublicInput fullShape) shape

abbrev Fresh
    (fullShape : Phi81Relation.Shape) (shape : JointShape) :=
  Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Fresh
    ProductCommitmentAlgebra.BundleValue (PublicInput fullShape) shape

abbrev Proof (shape : JointShape) (degreeBound : Nat) :=
  Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Proof
    K ProductCommitmentAlgebra.BundleValue shape degreeBound

abbrev Key
    (TranscriptState : Type) (fullShape : Phi81Relation.Shape)
    (shape : JointShape) (columns blockCount degreeBound : Nat) :=
  Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Key
    K ProductCommitmentAlgebra.BundleValue (PublicInput fullShape) RingF
      TranscriptState shape columns blockCount degreeBound

end Paper

/-- A fail-closed parser for the complete authority-bearing claim block.
`decode_block` is a codec theorem, not a protocol-soundness premise: it only
states what the parser returns for the canonical encoding of one typed V2
claim. -/
structure ClaimDecoder
    (widths : CompilerWidths) (fullShape : Phi81Relation.Shape)
    (shape : JointShape) where
  /-- The exact codec-specific language. This predicate is stronger than the
  generic envelope predicate: in particular, it must include successful
  decoding of the complete running claim and exact fresh-claim projection. -/
  WellFormed : Value widths → Prop
  decode : FixedBits.Word widths.totalBits →
    Option (Paper.Running fullShape shape × Paper.Fresh fullShape shape)
  runningOf : Value widths → Paper.Running fullShape shape
  freshOf : Value widths → Paper.Fresh fullShape shape
  decode_block : ∀ value : Value widths, WellFormed value →
    decode value.block = some (runningOf value, freshOf value)

/-- All setup-owned data needed to construct the selected V2 verifier. The
paper key fixes the relation, transcript, product commitment, and algebra.
The decoder fixes the interpretation of every complete claim bit. -/
structure Configuration
    (widths : CompilerWidths) (TranscriptState : Type)
    (fullShape : Phi81Relation.Shape) (shape : JointShape)
    (columns blockCount degreeBound : Nat) where
  verifierKeyDigest : Digest.Value
  relationManifestDigest : Digest.Value
  key : Paper.Key TranscriptState fullShape shape columns blockCount degreeBound
  decoder : ClaimDecoder widths fullShape shape
  /-- A bounded concrete sampler must reject before the total paper response
  can substitute its internal default. This gate can only reject. It cannot
  cause paper-verifier acceptance. The exact V2 constructor installs the
  selected three-attempt Poseidon2 sampler check. -/
  samplerCheck : Paper.Running fullShape shape → Paper.Fresh fullShape shape →
    Paper.Proof shape degreeBound → Bool

/-- Exact Boolean graph of the executable paper verifier. Keeping this as one
named function prevents a later adapter from selecting a different equality
decision at the selected-verifier boundary. -/
noncomputable def paperAccepts
    {widths : CompilerWidths} {TranscriptState : Type}
    {fullShape : Phi81Relation.Shape} {shape : JointShape}
    {columns blockCount degreeBound : Nat}
    (config : Configuration widths TranscriptState fullShape shape
      columns blockCount degreeBound)
    (running : Paper.Running fullShape shape)
    (fresh : Paper.Fresh fullShape shape)
    (proof : Paper.Proof shape degreeBound)
    (output : Paper.Running fullShape shape) : Bool :=
  @decide
    (Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.verify
      config.key running fresh proof = some output)
    (Classical.propDecidable _)

@[simp] theorem paperAccepts_eq_true_iff
    {widths : CompilerWidths} {TranscriptState : Type}
    {fullShape : Phi81Relation.Shape} {shape : JointShape}
    {columns blockCount degreeBound : Nat}
    (config : Configuration widths TranscriptState fullShape shape
      columns blockCount degreeBound)
    (running : Paper.Running fullShape shape)
    (fresh : Paper.Fresh fullShape shape)
    (proof : Paper.Proof shape degreeBound)
    (output : Paper.Running fullShape shape) :
    paperAccepts config running fresh proof output = true ↔
      Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.verify
        config.key running fresh proof = some output := by
  classical
  simp [paperAccepts]

/-- The only selected verifier constructed by this module. It rejects a
decode failure. On success, its Boolean is exact equality with the output of
the executable paper NIFS verifier. -/
noncomputable def selected
    {widths : CompilerWidths} {TranscriptState : Type}
    {fullShape : Phi81Relation.Shape} {shape : JointShape}
    {columns blockCount degreeBound : Nat}
    (config : Configuration widths TranscriptState fullShape shape
      columns blockCount degreeBound) : SelectedVerifier widths where
  Proof := Paper.Proof shape degreeBound
  Output := Paper.Running fullShape shape
  verifierKeyDigest := config.verifierKeyDigest
  relationManifestDigest := config.relationManifestDigest
  profile := Profile.v2
  profileExact := rfl
  verify := fun proof block output =>
    match config.decoder.decode block with
    | none => false
    | some input =>
        config.samplerCheck input.1 input.2 proof &&
          paperAccepts config input.1 input.2 proof output

@[simp] theorem selected_verify_block
    {widths : CompilerWidths} {TranscriptState : Type}
    {fullShape : Phi81Relation.Shape} {shape : JointShape}
    {columns blockCount degreeBound : Nat}
    (config : Configuration widths TranscriptState fullShape shape
      columns blockCount degreeBound)
    (value : Value widths) (wellFormed : config.decoder.WellFormed value)
    (proof : Paper.Proof shape degreeBound)
    (output : Paper.Running fullShape shape) :
    (selected config).verify proof value.block output =
      (config.samplerCheck (config.decoder.runningOf value)
          (config.decoder.freshOf value) proof &&
        paperAccepts config (config.decoder.runningOf value)
          (config.decoder.freshOf value) proof output) := by
  simp [selected, config.decoder.decode_block value wellFormed]

/-- A sampler rejection is final. Paper-verifier acceptance cannot override
the bounded-sampler gate or make the total default response authoritative. -/
theorem selected_verify_block_eq_false_of_sampler_rejected
    {widths : CompilerWidths} {TranscriptState : Type}
    {fullShape : Phi81Relation.Shape} {shape : JointShape}
    {columns blockCount degreeBound : Nat}
    (config : Configuration widths TranscriptState fullShape shape
      columns blockCount degreeBound)
    (value : Value widths) (wellFormed : config.decoder.WellFormed value)
    (proof : Paper.Proof shape degreeBound)
    (output : Paper.Running fullShape shape)
    (rejected : config.samplerCheck (config.decoder.runningOf value)
      (config.decoder.freshOf value) proof = false) :
    (selected config).verify proof value.block output = false := by
  rw [selected_verify_block config value wellFormed, rejected]
  rfl

/-- Acceptance by the selected adapter exposes the exact decoded paper input
and the exact executable paper-verifier equation. This is a deterministic
adapter theorem; it does not use the paper soundness reduction. -/
theorem verifyClaim_decodes_and_accepts_paper
    {widths : CompilerWidths} {TranscriptState : Type}
    {fullShape : Phi81Relation.Shape} {shape : JointShape}
    {columns blockCount degreeBound : Nat}
    (config : Configuration widths TranscriptState fullShape shape
      columns blockCount degreeBound)
    (claim : Claim (selected config))
    (proof : Paper.Proof shape degreeBound)
    (output : Paper.Running fullShape shape)
    (accepted : VerifyClaim (selected config) (proof, output) claim) :
    ∃ running fresh,
      config.decoder.decode (Value.ofProtocolClaim claim).block =
          some (running, fresh) ∧
        config.samplerCheck running fresh proof = true ∧
        Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.verify config.key
          running fresh proof = some output := by
  have checked := accepted.2
  change
    (match config.decoder.decode (Value.ofProtocolClaim claim).block with
      | none => false
      | some input =>
          config.samplerCheck input.1 input.2 proof &&
            paperAccepts config input.1 input.2 proof output) = true
    at checked
  cases decoded : config.decoder.decode (Value.ofProtocolClaim claim).block with
  | none => rw [decoded] at checked; contradiction
  | some input =>
      have both :
          config.samplerCheck input.1 input.2 proof = true ∧
            paperAccepts config input.1 input.2 proof output = true := by
        simpa only [decoded, Bool.and_eq_true] using checked
      exact ⟨input.1, input.2, by simpa only [Prod.eta] using decoded,
        both.1,
        (paperAccepts_eq_true_iff config _ _ proof output).1 both.2⟩

/-- Exact selected-verifier soundness. Acceptance of a canonical complete V2
claim gives the independent paper transition or one of the five closed paper
bad events. There is no caller-supplied reduction theorem. -/
theorem verifyClaim_sound
    {widths : CompilerWidths} {TranscriptState : Type}
    {fullShape : Phi81Relation.Shape} {shape : JointShape}
    {columns blockCount degreeBound : Nat}
    (config : Configuration widths TranscriptState fullShape shape
      columns blockCount degreeBound)
    (claim : Claim (selected config))
    (proof : Paper.Proof shape degreeBound)
    (output : Paper.Running fullShape shape)
    (accepted : VerifyClaim (selected config) (proof, output) claim) :
    ∃ running fresh,
      config.decoder.decode (Value.ofProtocolClaim claim).block =
          some (running, fresh) ∧
        config.samplerCheck running fresh proof = true ∧
        (Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Transition
            config.key running fresh output ∨
          Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.BadEvent
            config.key running fresh proof output) := by
  rcases verifyClaim_decodes_and_accepts_paper
      config claim proof output accepted with
    ⟨running, fresh, decoded, samplerAccepted, paperAccepted⟩
  exact ⟨running, fresh, decoded, samplerAccepted,
    Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.verify_sound
      config.key running fresh proof output paperAccepted⟩

/-- Exact graph completeness outside the named bounded-sampler rejection.
Every independent paper transition produces one paper proof. If its exact
setup-owned sampler gate succeeds, the selected verifier accepts the same
complete envelope and output. This theorem does not assume sampler success. -/
theorem verifyClaim_complete
    {widths : CompilerWidths} {TranscriptState : Type}
    {fullShape : Phi81Relation.Shape} {shape : JointShape}
    {columns blockCount degreeBound : Nat}
    (config : Configuration widths TranscriptState fullShape shape
      columns blockCount degreeBound)
    (value : Value widths) (canonical : value.Canonical)
    (wellFormed : config.decoder.WellFormed value)
    (output : Paper.Running fullShape shape)
    (transition :
      Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Transition
        config.key (config.decoder.runningOf value)
          (config.decoder.freshOf value) output) :
    ∃ proof : Paper.Proof shape degreeBound,
      Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.verify config.key
          (config.decoder.runningOf value) (config.decoder.freshOf value)
          proof = some output ∧
        (config.samplerCheck (config.decoder.runningOf value)
            (config.decoder.freshOf value) proof = true →
          VerifyClaim (selected config) (proof, output)
            (value.toProtocolClaim
              (NifsProof := PackedProof (selected config)))) := by
  rcases
      Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.verify_complete
        config.key (config.decoder.runningOf value)
        (config.decoder.freshOf value) output transition with
    ⟨proof, paperAccepted⟩
  refine ⟨proof, paperAccepted, ?_⟩
  intro samplerAccepted
  refine ⟨canonical, ?_⟩
  simp only [Value.of_toProtocolClaim]
  rw [selected_verify_block config value wellFormed]
  simp only [samplerAccepted, Bool.true_and]
  exact (paperAccepts_eq_true_iff config _ _ proof output).2 paperAccepted

end Nightstream.Implementation.NebulaV2.ProductPaperNifsSelection
