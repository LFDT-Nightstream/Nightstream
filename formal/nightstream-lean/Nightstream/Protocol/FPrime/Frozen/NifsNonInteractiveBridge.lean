import Nightstream.Protocol.FPrime.Frozen.Obligations
import Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.FullOracleSoundness

/-!
Exact frozen-target bridge for the paper non-interactive SuperNeo NIFS.

Owns: specialization of the frozen quantitative
`NifsNonInteractiveSound` proposition to the complete finite correlated
prefix/post-prefix oracle experiment.

Does not own: the deterministic NIFS soundness/completeness theorem, any
interactive residual bound, any transcript-collision bound, Poseidon2,
Ajtai, Rust, R1CS, artifacts, minimality, or costs.

Emits constraints: no.

The premises are the existing event-by-event contracts on the actual
experiment. Challenge-sampling failure is proved impossible on the selected
support, and the multi-fork programming loss is proved internally with the
paper-selected `(ell + 1) / |C|` bound. The combined headline below also
instantiates the deterministic soundness/completeness target, so neither
frozen obligation is accepted as a premise.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.Frozen.NifsNonInteractiveBridge

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.Paper
open Nightstream.Protocol.FPrime.Frozen.Obligations

universe uExtension uCommitment uPublicInput uScalar uState

/-- Headline obligation-5 quantitative theorem for the exact full oracle
experiment. Its conclusion is literally the frozen
`NifsNonInteractiveSound` target. -/
theorem fullOracleMixtureNifsNonInteractiveSound
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    [DecidableEq Scalar]
    [DecidableEq State]
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      key.nifsPiRlcContext.semantics key.nifsPiRlcContext.params
      key.nifsPiRlcContext.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      key.nifsPiRlcContext.algebra.challengeValid)
    (prefixExperiment : PiCcsPrefixExperiment key)
    (alphabet : Support Scalar)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      key.piRlcAlgebra.challengeValid scalar)
    (extractionBudget : NifsExtractionErrorBudget Rat)
    (interactiveBudget : InteractiveErrorBudget Rat)
    (collisionBudget : PostPrefixCollisionBudget)
    (interactiveContract :
      FullOracleInteractiveResidualContract
        (fullOracleForkMixture prefixExperiment alphabet
          alphabetValid).toProbabilityExperiment
        extractionBudget interactiveBudget)
    (collisionContract :
      FullOracleCollisionContract prefixExperiment alphabet alphabetValid
        collisionBudget) :
    NifsNonInteractiveSound
      Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.scale
      (fullOracleForkMixture prefixExperiment alphabet
        alphabetValid).toProbabilityExperiment
      FullOracleAcceptedOutcome
      FullOracleTransitionOutcome
      (nonInteractiveTotal
        Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.scale
        extractionBudget interactiveBudget
        (postPrefixFiatShamirBudget key alphabet collisionBudget)) := by
  exact
    fullOracleMixtureAccepted_probability_sub_total_le_transition
      laws strongSet prefixExperiment alphabet alphabetValid
      extractionBudget interactiveBudget collisionBudget
      interactiveContract collisionContract

/-- Complete model-level obligation-5 theorem for the exact typed paper NIFS.

The first conjunct is deterministic soundness modulo the five paper NIFS
events together with honest completeness. The second conjunct is the frozen
subtractive non-interactive bound for the actual correlated oracle
experiment. Neither conjunct is a premise.

The remaining premises are precisely the permitted mathematical and
cryptographic boundaries: extraction algebra, the strong sampling set, one
accepted target-witness extraction bound, four interactive event bounds, and
four typed transcript-collision bounds. Poseidon2 realization of the typed
oracle remains a separate concrete-refinement theorem. -/
theorem paperNifsSoundCompleteAndNonInteractive
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    [DecidableEq Scalar]
    [DecidableEq State]
    {shape : PiCCS.PaperJoint.Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      key.nifsPiRlcContext.semantics key.nifsPiRlcContext.params
      key.nifsPiRlcContext.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      key.nifsPiRlcContext.algebra.challengeValid)
    (prefixExperiment : PiCcsPrefixExperiment key)
    (alphabet : Support Scalar)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      key.piRlcAlgebra.challengeValid scalar)
    (extractionBudget : NifsExtractionErrorBudget Rat)
    (interactiveBudget : InteractiveErrorBudget Rat)
    (collisionBudget : PostPrefixCollisionBudget)
    (interactiveContract :
      FullOracleInteractiveResidualContract
        (fullOracleForkMixture prefixExperiment alphabet
          alphabetValid).toProbabilityExperiment
        extractionBudget interactiveBudget)
    (collisionContract :
      FullOracleCollisionContract prefixExperiment alphabet alphabetValid
        collisionBudget) :
    NifsSoundAndCompleteModulo
        (Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs.nifsVerifier
          (Extension := Extension)
          (Commitment := Commitment)
          (PublicInput := PublicInput)
          (Scalar := Scalar)
          (TranscriptState := State)
          (shape := shape)
          (columns := columns)
          (blockCount := blockCount)
          (degreeBound := degreeBound))
        Transition BadEvent /\
      NifsNonInteractiveSound
        Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.scale
        (fullOracleForkMixture prefixExperiment alphabet
          alphabetValid).toProbabilityExperiment
        FullOracleAcceptedOutcome
        FullOracleTransitionOutcome
        (nonInteractiveTotal
          Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.scale
          extractionBudget interactiveBudget
          (postPrefixFiatShamirBudget key alphabet collisionBudget)) := by
  constructor
  · exact
      Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs.nifsSoundAndCompleteModulo
  · exact
      fullOracleMixtureNifsNonInteractiveSound laws strongSet prefixExperiment
        alphabet alphabetValid extractionBudget interactiveBudget
        collisionBudget interactiveContract collisionContract

end Nightstream.Protocol.FPrime.Frozen.NifsNonInteractiveBridge
