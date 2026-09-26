import NightstreamFPrime.Export.Stage1.Wide.AuthorityStream
import NightstreamFPrime.Export.Stage1.Wide.ContextBinding
import NightstreamFPrime.Export.Stage1.Wide.HashChainCounts
import NightstreamFPrime.Export.Stage1.SetupBinding

/-! Verifier authority for the selected wide package: its self-derived
relation, unchanged application, exact transcript schedule and same-seed Ajtai
setup. Its serializers bind the wide schedule; the baseline package keeps its
own serialization and identity pins. The sealed children remain authoritative. -/

namespace NightstreamFPrime.Export.Stage1.Wide.SetupBinding

open NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Export NightstreamFPrime.Export.Package NightstreamFPrime.Export.Codec
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding Spec.Folding.PiCCS.PaperJoint Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

abbrev application := Poseidon2HashChainV1Package.application
abbrev fits := Poseidon2HashChainV1Package.fits

def verifierRows : Nat := productionProfile.commitmentWidth

def messageColumns : Nat :=
  Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth (RetainedLayout.logicalWidth application))

theorem verifierRows_eq : verifierRows = 22 := rfl

theorem messageColumns_eq : messageColumns = 2543368 := by
  unfold messageColumns
  rw [HashChainCounts.committedCoordinates]
  norm_num [Phi81ColumnLayout.blockCount, ringDegree]

abbrev Setup := AjtaiSetupV1.Setup verifierRows messageColumns

/-- The approved seed is retained; the indexed dimensions come from the wide
fixed point. No expanded key is stored or evaluated. -/
def productionSetup : Setup where
  seed := Poseidon2HashChainV1SetupAuthority.productionSeed

def ajtaiKey (setup : Setup) : AjtaiKey
    (logicalWidth := RetainedLayout.logicalWidth application)
    (publicFits := FixedPoint.publicFits application) := setup.verifierKey

def productionAjtaiKey := ajtaiKey productionSetup

def commitmentKeyWords : List F := productionSetup.authorityWords

theorem approved_seed : productionSetup.seed = Poseidon2HashChainV1Setup.productionSeed := rfl

theorem commitmentKeyWords_length : commitmentKeyWords.length = 73 :=
  productionSetup.authorityWords_length

theorem ajtaiKey_eq_of_authorityWords (left right : Setup)
    (same : left.authorityWords = right.authorityWords) : ajtaiKey left = ajtaiKey right := by
  apply AjtaiSetupV1.Setup.verifierKey_eq_of_authorityWords left right _ _ same
  · rw [verifierRows_eq]; decide
  · rw [messageColumns_eq]; decide

/-- The smaller same-seed carrier reduces to the already approved fixed
instance by appending zero blocks. Its strict norm bound is unchanged. -/
def shortKernel_to_approvedMsis
    (witness : Phi81Relation.PiRLCAlgebra.Binding.ShortKernelVector productionAjtaiKey productionGlobalParams.msisNormBound) :
    Phi81Relation.PiRLCAlgebra.Binding.ShortKernelVector
      (shape := Poseidon2HashChainV1Setup.approvedMsisShape)
      Poseidon2HashChainV1Setup.approvedMsisSetup.verifierKey productionGlobalParams.msisNormBound := by
  apply AjtaiSetupV1.Prefix.extendShortKernel
    (smallShape := FullShape (RetainedLayout.logicalWidth application) (FixedPoint.publicFits application))
    (largeShape := Poseidon2HashChainV1Setup.approvedMsisShape)
    (small := productionSetup) (large := Poseidon2HashChainV1Setup.approvedMsisSetup) _ rfl witness
  change Phi81CarrierLayout.carrierWidth (RetainedLayout.logicalWidth application) ≤
    Poseidon2HashChainV1Setup.approvedMsisShape.carrierWidth
  rw [HashChainCounts.committedCoordinates, Poseidon2HashChainV1Setup.approvedMsis_carrierWidth]
  decide

noncomputable def nifsKey (compiled : PiRlcWideSampler.RangePlan.Compiled) :=
  FixedPoint.key application compiled fits productionAjtaiKey

theorem nifsKey_profile (compiled : PiRlcWideSampler.RangePlan.Compiled) :
    (nifsKey compiled).params = productionGlobalParams := rfl

theorem nifsKey_response (compiled : PiRlcWideSampler.RangePlan.Compiled) (state : Transcript.State) :
    (nifsKey compiled).piRlcResponse state =
      some (fun source => Nifs.NonInteractive.PiRlcWideSampler.Transcript.challengeAt state source.val) := rfl

private def byteValue (bytes : List Nat) : Value := (list nat).encode bytes

/-- Canonical schedule schema: the digest-only PiCCS tag and dimensions,
then scalar order, scalar-entry tag, draw width and modulus, base-five digit
count, one state advance, and the exact centered alphabet. The wide map reads
the four fields in little-endian base-p order and reduces modulo 5^54. -/
def scheduleValue : Value := .array [
  byteValue Transcript.piCcsDigestDomainTagBytes,
  byteValue [cubeVariables, 10, productionShape.sourceCount, productionShape.matrixCount,
    productionShape.coefficientCount],
  byteValue ("Nightstream/PiRLC/wide-reduction/v1".toUTF8.toList.map UInt8.toNat),
  byteValue [productionShape.sourceCount, 4, Nifs.NonInteractive.PiRlcWideSampler.drawWidth,
    goldilocksModulus, Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.alphabetSize,
    Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.coefficientCount, 1],
  byteValue (List.ofFn fun digit : Fin Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.alphabetSize =>
    (Phi81StrongSet.embedCoefficient digit).val)]

def scheduleWords : List F := Package.valuePreimage scheduleValue

theorem sampler_frame (state : Spec.Poseidon2.State) (coordinate : Nat) :
    Nifs.NonInteractive.PiRlcWideSampler.Transcript.enter state coordinate =
      Poseidon2.absorbBlock state [Poseidon2.ofNat 4, Poseidon2.ofNat coordinate] := rfl

theorem sampler_advance (state : Spec.Poseidon2.State) (coordinate : Nat) :
    Nifs.NonInteractive.PiRlcWideSampler.Transcript.stateAt state (coordinate + 1) =
      Poseidon2.permute (Nifs.NonInteractive.PiRlcWideSampler.Transcript.enter
        (Nifs.NonInteractive.PiRlcWideSampler.Transcript.stateAt state coordinate) coordinate) := rfl

theorem sampler_digit (state : Spec.Poseidon2.State) (coordinate : Nat)
    (position : Fin Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.coefficientCount) :
    (Nifs.NonInteractive.PiRlcWideSampler.Transcript.scalarAt state coordinate position).val =
      (Nifs.NonInteractive.PiRlcWideSampler.drawIndex
        (Nifs.NonInteractive.PiRlcWideSampler.Transcript.drawAt state coordinate)).val %
        Nifs.NonInteractive.PiRlcWideSampler.scalarCount /
          Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.alphabetSize ^ position.val %
            Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.alphabetSize := rfl

private def relationWordsFromStructural (parts : AuthorityStream.Parts) (structural : VerifierContext.Digest4) : List F :=
  Package.valuePreimage (CcsRelation.format.encode parts.package.relation) ++ structural.toList

def relationWords (parts : AuthorityStream.Parts) : List F :=
  relationWordsFromStructural parts (AuthorityStream.structuralIdentity parts)

/-- The application component binds the exact relocated plan in the sealed
envelope. Rust recomputes this component from that same child. -/
def applicationWords (parts : AuthorityStream.Parts) : List F :=
  Stage1.ApplicationPackage.authorityWords parts.application

private def keyWordsFromRelation (relationWords : List F) : List F :=
  PerApplicationCanonicalPackage.nifsKeyDomain ++
    VerifierContext.framed relationWords ++
    VerifierContext.framed VerifierContext.profileWords ++
    VerifierContext.framed scheduleWords ++
    VerifierContext.framed (VerifierContext.componentDigest 4 commitmentKeyWords).toList

def nifsKeyWords (parts : AuthorityStream.Parts) : List F := keyWordsFromRelation (relationWords parts)

def authority (parts : AuthorityStream.Parts) : VerifierContext.Authority where
  relationWords := relationWords parts
  applicationWords := applicationWords parts
  nifsKeyWords := nifsKeyWords parts
  commitmentKeyWords := commitmentKeyWords

private def descriptorFromStructural (parts : AuthorityStream.Parts) (structural : VerifierContext.Digest4) :
    VerifierContext.Descriptor :=
  let relation := relationWordsFromStructural parts structural
  {
    relation := VerifierContext.componentDigest 1 relation
    application := PerApplicationVerifierContextStreaming.applicationComponentDigestDirect
      parts.application
    nifsKey := VerifierContext.componentDigest 3 (keyWordsFromRelation relation)
    commitmentKey := VerifierContext.componentDigest 4 commitmentKeyWords }

/-- The native descriptor streams the structural envelope once. The
application component uses its existing proved native stream. -/
def descriptor (parts : AuthorityStream.Parts) : VerifierContext.Descriptor :=
  descriptorFromStructural parts (AuthorityStream.structuralIdentity parts)

theorem descriptor_recomputed (parts : AuthorityStream.Parts) :
    descriptor parts = VerifierContext.descriptor (authority parts) := by
  unfold descriptor descriptorFromStructural
  rw [PerApplicationVerifierContextStreaming.applicationComponentDigestDirect_eq]
  rfl

/-- The candidate uses the actual wide schedule at the outer context layer.
The baseline descriptor serializer is not used by these candidate functions. -/
def contextSerialize (value : VerifierContext.Descriptor) : List F :=
  VerifierContext.contextDomain ++ VerifierContext.framed VerifierContext.profileWords ++
    VerifierContext.framed scheduleWords ++
    VerifierContext.framed value.relation.toList ++ VerifierContext.framed value.application.toList ++
    VerifierContext.framed value.nifsKey.toList ++ VerifierContext.framed value.commitmentKey.toList

def contextDigest (context : VerifierContext.Descriptor) : VerifierContext.Digest4 :=
  VerifierContext.Digest4.ofList (Poseidon2.hash (contextSerialize context))

def contextKey (context : VerifierContext.Descriptor) : KeyDigest := (contextDigest context).toList

theorem contextKey_length (context : VerifierContext.Descriptor) : (contextKey context).length = 4 := rfl

theorem contextKey_recomputed (parts : AuthorityStream.Parts) :
    contextKey (descriptor parts) = (VerifierContext.Digest4.ofList
      (Poseidon2.hash (contextSerialize (VerifierContext.descriptor (authority parts))))).toList := by
  rw [contextKey, contextDigest, descriptor_recomputed]

/-- Component replay for parity. A caller-supplied digest is not package
authority; the selected constructor below recomputes each component. -/
def bindingFromStructural (structural : VerifierContext.Digest4)
    (context : VerifierContext.Descriptor) : Stage1.VerificationKey.Binding :=
  {
    packageIdentity := VerifierContext.Digest4.ofList (Poseidon2.hash
      (PerApplicationCanonicalPackage.packageIdentityDomain ++
        VerifierContext.framed structural.toList ++ VerifierContext.framed (contextSerialize context)))
    context := context }

def bindingValues (parts : AuthorityStream.Parts) : VerifierContext.Digest4 × Stage1.VerificationKey.Binding :=
  let structural := AuthorityStream.structuralIdentity parts
  (structural, bindingFromStructural structural (descriptorFromStructural parts structural))

def verificationKeyBinding (parts : AuthorityStream.Parts) : Stage1.VerificationKey.Binding :=
  (bindingValues parts).2

def bindingSerialize (binding : Stage1.VerificationKey.Binding) : List F :=
  Stage1.VerificationKey.domain ++ VerifierContext.framed binding.packageIdentity.toList ++
    VerifierContext.framed (contextSerialize binding.context)

def bindingDigest (binding : Stage1.VerificationKey.Binding) : VerifierContext.Digest4 :=
  VerifierContext.Digest4.ofList (Poseidon2.hash (bindingSerialize binding))

/-- Schema 1 binds the sealed envelope, full context and verification key.
The structural stream is evaluated once and shared by all six fields. -/
def bindingFixtureValue (values : VerifierContext.Digest4 × Stage1.VerificationKey.Binding) : Value :=
  let words (fields : List F) := Value.array (fields.map fun field => Value.atom field.val)
  .array [
    .atom 1,
    words values.1.toList,
    words values.2.packageIdentity.toList,
    words (contextSerialize values.2.context),
    words (bindingSerialize values.2),
    words (bindingDigest values.2).toList]

def bindingFixture (parts : AuthorityStream.Parts) : Value :=
  bindingFixtureValue (bindingValues parts)

theorem bindingValues_structural (parts : AuthorityStream.Parts) :
    (bindingValues parts).1 = AuthorityStream.structuralIdentity parts := rfl

theorem bindingValues_binding (parts : AuthorityStream.Parts) :
    (bindingValues parts).2 = verificationKeyBinding parts := rfl

theorem verificationKeyBinding_context (parts : AuthorityStream.Parts) :
    (verificationKeyBinding parts).context = descriptor parts := rfl

/-- A verifier-owned candidate build fixes the expected four-word context.
The accepted rows imply the step for that setup and context, or give the
existing state-hash collision. No source-row or honest-witness premise is added. -/
theorem step_or_collision (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (parts : AuthorityStream.Parts) (_built : AuthorityStream.prepare compiled = .ok parts)
    (assignment : Assignment F (RetainedLayout.logicalWidth application))
    (claimed : HashPreimage (logicalWidth := RetainedLayout.logicalWidth application)
      (publicFits := FixedPoint.publicFits application))
    (publicDigest : Digest) (digestLength : publicDigest.length = 4)
    (publicEqual : PublicBinding.publicInput application assignment = encHash publicDigest)
    (checkedDigest : publicDigest = stateHash { claimed with verifierKeys := fun _ => contextKey (descriptor parts) })
    (rows : (FixedPoint.structuralPlan application compiled fits).RowsZero assignment) :
    Lifecycle.Stage1.Wide.Relation.StepHoldsFor (FixedPoint.relation application compiled fits)
      productionAjtaiKey (contextKey (descriptor parts)) application
      (FixedPointSoundness.input application assignment (FixedPoint.relation application compiled fits))
      (FixedPointSoundness.output application assignment (RetainedLayout.logicalWidth application)
        (FixedPoint.publicFits application)) ∨
    Layout.Stage1.PiCCSSecurity.StateHashCollision
      (ContextBinding.decodedNext application assignment compiled fits productionAjtaiKey)
      { claimed with verifierKeys := fun _ => contextKey (descriptor parts) } :=
  ContextBinding.step_or_collision application assignment compiled fits productionAjtaiKey
    (contextKey (descriptor parts)) (contextKey_length (descriptor parts)) claimed publicDigest digestLength publicEqual checkedDigest rows

end NightstreamFPrime.Export.Stage1.Wide.SetupBinding
