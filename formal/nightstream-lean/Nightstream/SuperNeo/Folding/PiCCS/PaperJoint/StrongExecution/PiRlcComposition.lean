import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrong
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcBatch
import Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Reindex
import Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition

/-!
Finite operational coupling of paper `Pi_CCS` to paper `Pi_RLC`.

Source: SuperNeo Theorem 6 and Appendix D.3.

Owns: the composed adversary's causal `Pi_CCS` prefix, the exact verifier-
constructed `K+k` batch passed to `Pi_RLC`, post-prefix coordinate-fork
extraction, the abort gate for rejected first-stage prefixes, and the finite
seed reindexings used by the strong--weak composition theorem.

Does not own: either component reduction theorem, `Pi_DEC`, asymptotic
rejection sampling, Fiat--Shamir, Rust, R1CS, artifacts, or costs.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcBatch
open Nightstream.SuperNeo.Folding.PiRLC.PaperCompleteness
open Nightstream.SuperNeo.Folding.PiRLC.PaperForkExtraction
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open MatrixCoefficientSource
open PaperLinearAlgebra

universe uExtension uCommitment uPublicInput uScalar uProverSeed uProverTape

/-- Randomness fixed before the `Pi_RLC` coordinate fork: one prover tape
seed and the complete verifier-owned `Pi_CCS` coin seed. -/
abbrev PrefixSeed
    (Extension : Type uExtension)
    (shape : Shape)
    (ProverSeed : Type uProverSeed) :=
  ProverSeed × VerifierCoins.Seed Extension shape.cubeVariables

/-- One sequential adversary.  The second-stage oracle receives the complete
causal prefix, never a target witness or extracted assignment family. -/
structure Adversary
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (ProverSeed : Type uProverSeed)
    (ProverTape : Type uProverTape) where
  proverSupport : Support ProverSeed
  strategy : Strategy Extension shape ProverTape
  proverTape : ProverSeed -> ProverTape
  oracle : PrefixExecution Extension shape ->
    (Fin context.arity.total -> Scalar) -> Assignment F columns

/-- Exact support sampled before entering `Pi_RLC`. -/
def prefixSupport
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (alphabet : Support Extension)
    (adversary : Adversary context ProverSeed ProverTape) :
    Support (PrefixSeed Extension shape ProverSeed) :=
  adversary.proverSupport.product
    (VerifierCoins.support alphabet shape.cubeVariables)

/-- Execute the first-stage causal prefix from exactly its two seed
coordinates. -/
def prefixExecution
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (adversary : Adversary context ProverSeed ProverTape)
    (seed : PrefixSeed Extension shape ProverSeed) :
    PrefixExecution Extension shape :=
  execute adversary.strategy (adversary.proverTape seed.1)
    (VerifierCoins.toPublicCoins seed.2)

/-- The literal `Pi_RLC` adversary selected by one completed `Pi_CCS`
prefix.  Its batch is verifier-constructed; only its assignment oracle is
adversarial. -/
def component
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (adversary : Adversary context ProverSeed ProverTape)
    (causalRun : PrefixExecution Extension shape) :
    Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.Adversary
      context.piRlc where
  batch := context.batchOfPrefix causalRun
  oracle := adversary.oracle causalRun

/-- Appendix D.3's weak adversary, including the mandatory abort when the
first verifier rejects. -/
def toWeak
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (alphabet : Support Extension)
    (adversary : Adversary context ProverSeed ProverTape) :
    Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.Adversary
      context.piRlc
      (PrefixSeed Extension shape ProverSeed) where
  prefixes := prefixSupport context alphabet adversary
  enabled := fun seed =>
    acceptedCheck context.piCcs (prefixExecution context adversary seed)
  component := fun seed =>
    component context adversary (prefixExecution context adversary seed)

/-- The coordinate-fork sample determined by one completed prefix and one
fork seed. -/
def forkSample
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (verifier :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform.VerifierData
        context.piRlc)
    (adversary : Adversary context ProverSeed ProverTape)
    (causalRun : PrefixExecution Extension shape)
    (seed : ForkSeed verifier.alphabet context.arity.total) :
    ForkSample Scalar context.arity.total :=
  (Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform.run
    (verifier.accepts (component context adversary causalRun)) seed.val).sample

/-- Reindex the coordinate extractor's `K+k` family into the exact
`Pi_CCS.OutputWitness` order. -/
def extractedWitness
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (laws : ExtractionAlgebra context.piRlc.semantics context.piRlc.params
      context.piRlc.algebra)
    (strongSet : StrongSetUnits laws.ring
      context.piRlc.algebra.challengeValid)
    (adversary : Adversary context ProverSeed ProverTape)
    (causalRun : PrefixExecution Extension shape)
    (sample : ForkSample Scalar context.arity.total)
    (accepted :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.AcceptedFork
        context.piRlc
      (component context adversary causalRun) sample) :
    OutputWitness shape columns where
  assignments := fun source =>
    Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.extractedFamily
      context.piRlc laws strongSet
      (component context adversary causalRun) sample accepted
      (Fin.cast context.total_eq_sourceCount.symm source)

/-- Ambient membership of the reindexed strong target is exactly ambient
membership of every coordinate returned by the weak extractor. -/
theorem extractedWitness_ambient_iff
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (laws : ExtractionAlgebra context.piRlc.semantics context.piRlc.params
      context.piRlc.algebra)
    (strongSet : StrongSetUnits laws.ring
      context.piRlc.algebra.challengeValid)
    (adversary : Adversary context ProverSeed ProverTape)
    (causalRun : PrefixExecution Extension shape)
    (sample : ForkSample Scalar context.arity.total)
    (accepted :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.AcceptedFork
        context.piRlc (component context adversary causalRun) sample) :
    AmbientOutputHolds context.piCcs.extensionOps context.piCcs.lift
        context.piCcs.openingMaps context.piCcs.params
        context.piCcs.statement causalRun.probe
        (extractedWitness context laws strongSet adversary causalRun sample
          accepted) <->
      forall coordinate,
        Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.CorrectedAmbientHolds
          context.piRlc.semantics context.piRlc.params
          ((context.batchOfPrefix causalRun).inputs coordinate)
          (Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.extractedFamily
            context.piRlc laws strongSet
            (component context adversary causalRun) sample accepted
            coordinate) := by
  constructor
  · intro ambient coordinate
    have atSource := ambient (context.sourceIndex coordinate)
    simpa [extractedWitness, CompatibleContext.sourceIndex,
      CompatibleContext.batchOfPrefix, component] using atSource
  · intro allCoordinates source
    have atCoordinate := allCoordinates
      (Fin.cast context.total_eq_sourceCount.symm source)
    simpa [extractedWitness, CompatibleContext.sourceIndex,
      CompatibleContext.batchOfPrefix, component] using atCoordinate

/-- The post-prefix target returns an extracted family exactly when the
coordinate fork accepts. -/
noncomputable def targetWitness
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (laws : ExtractionAlgebra context.piRlc.semantics context.piRlc.params
      context.piRlc.algebra)
    (strongSet : StrongSetUnits laws.ring
      context.piRlc.algebra.challengeValid)
    (verifier :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform.VerifierData
        context.piRlc)
    (adversary : Adversary context ProverSeed ProverTape)
    (seed : ForkSeed verifier.alphabet context.arity.total)
    (causalRun : PrefixExecution Extension shape) :
    Option (OutputWitness shape columns) := by
  classical
  let sample := forkSample context verifier adversary causalRun seed
  exact if accepted :
        Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.AcceptedFork
          context.piRlc
          (component context adversary causalRun) sample then
      some (extractedWitness context laws strongSet adversary causalRun sample
        accepted)
    else
      none

/-- The post-prefix strong target succeeds in the corrected ambient relation
exactly when the first verifier accepted and the coordinate fork extracted
all `K+k` ambient witnesses. -/
theorem targetAmbient_iff_extracts
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (laws : ExtractionAlgebra context.piRlc.semantics context.piRlc.params
      context.piRlc.algebra)
    (strongSet : StrongSetUnits laws.ring
      context.piRlc.algebra.challengeValid)
    (verifier :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform.VerifierData
        context.piRlc)
    (adversary : Adversary context ProverSeed ProverTape)
    (seed : ForkSeed verifier.alphabet context.arity.total)
    (causalRun : PrefixExecution Extension shape) :
    ambientCheck context.piCcs
          (attachWitness causalRun
            (targetWitness context laws strongSet verifier adversary seed
              causalRun)) = true <->
      acceptedCheck context.piCcs causalRun = true /\
        Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.ExtractsCorrectedAmbient
          context.piRlc laws strongSet (component context adversary causalRun)
          (forkSample context verifier adversary causalRun seed) := by
  let sample := forkSample context verifier adversary causalRun seed
  by_cases accepted :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.AcceptedFork
        context.piRlc (component context adversary causalRun) sample
  · have targetEq :
        targetWitness context laws strongSet verifier adversary seed causalRun =
          some (extractedWitness context laws strongSet adversary causalRun
            sample accepted) := by
      simp [targetWitness, sample, accepted]
    rw [ambientCheck_eq_true_iff]
    simp only [AmbientSuccess, targetEq]
    constructor
    · intro success
      refine ⟨(acceptedCheck_eq_true_iff context.piCcs causalRun).2 success.1,
        ?_⟩
      refine ⟨accepted, ?_⟩
      exact (extractedWitness_ambient_iff context laws strongSet adversary
        causalRun sample accepted).1 success.2
    · rintro ⟨prefixAccepted, ⟨otherAccepted, ambient⟩⟩
      have proofEqual : otherAccepted = accepted := Subsingleton.elim _ _
      subst otherAccepted
      exact ⟨(acceptedCheck_eq_true_iff context.piCcs causalRun).1
          prefixAccepted,
        (extractedWitness_ambient_iff context laws strongSet adversary
          causalRun sample accepted).2 ambient⟩
  · have targetEq :
        targetWitness context laws strongSet verifier adversary seed causalRun =
          none := by
      simp [targetWitness, sample, accepted]
    rw [ambientCheck_eq_true_iff]
    simp only [AmbientSuccess, attachWitness, targetEq]
    constructor
    · exact False.elim
    · rintro ⟨_prefixAccepted, extracted⟩
      rcases extracted with ⟨otherAccepted, _ambient⟩
      exact accepted otherAccepted

/-- The strong-game adversary produced after selecting the weak extractor.
The extractor value carries no extra randomness; the canonical coordinate
fork support is the target seed. -/
noncomputable def toStrong
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (laws : ExtractionAlgebra context.piRlc.semantics context.piRlc.params
      context.piRlc.algebra)
    (strongSet : StrongSetUnits laws.ring
      context.piRlc.algebra.challengeValid)
    (verifier :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform.VerifierData
        context.piRlc)
    (adversary : Adversary context ProverSeed ProverTape) :
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalExperiment.Adversary
      context.piCcs ProverSeed
      (ForkSeed verifier.alphabet context.arity.total) ProverTape where
  proverSupport := adversary.proverSupport
  targetSupport := forkSeedSupport verifier.alphabet context.arity.total
  strategy := adversary.strategy
  proverTape := adversary.proverTape
  target := targetWitness context laws strongSet verifier adversary

/-- Reassociate one strong-game seed into the prefix-then-fork order used by
the lifted weak game. -/
def strongToWeakSeed
    {Extension : Type uExtension}
    {Scalar : Type uScalar}
    {shape : Shape}
    {ProverSeed : Type uProverSeed}
    {coordinates : Nat}
    {alphabet : Support Scalar}
    (seed : OperationalExperiment.RunSeed Extension shape ProverSeed
      (ForkSeed alphabet coordinates)) :
    (PrefixSeed Extension shape ProverSeed) × ForkSeed alphabet coordinates :=
  ((seed.1, seed.2.2), seed.2.1)

/-- Inverse seed reassociation. -/
def weakToStrongSeed
    {Extension : Type uExtension}
    {Scalar : Type uScalar}
    {shape : Shape}
    {ProverSeed : Type uProverSeed}
    {coordinates : Nat}
    {alphabet : Support Scalar}
    (seed : (PrefixSeed Extension shape ProverSeed) ×
      ForkSeed alphabet coordinates) :
    OperationalExperiment.RunSeed Extension shape ProverSeed
      (ForkSeed alphabet coordinates) :=
  (seed.1.1, (seed.2, seed.1.2))

/-- The two enumerations contain exactly the same independent seed tuples.
This is a proved permutation, not a probability premise. -/
theorem strongToWeakSeed_supportPermutation
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (laws : ExtractionAlgebra context.piRlc.semantics context.piRlc.params
      context.piRlc.algebra)
    (strongSet : StrongSetUnits laws.ring
      context.piRlc.algebra.challengeValid)
    (extensionAlphabet : Support Extension)
    (verifier :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform.VerifierData
        context.piRlc)
    (adversary : Adversary context ProverSeed ProverTape) :
    ((OperationalExperiment.runSupport context.piCcs extensionAlphabet
        (toStrong context laws strongSet verifier adversary)).values.map
      strongToWeakSeed).Perm
      ((prefixSupport context extensionAlphabet adversary).product
        (forkSeedSupport verifier.alphabet context.arity.total)).values := by
  apply Support.map_values_perm_of_inverse
    (OperationalExperiment.runSupport context.piCcs extensionAlphabet
      (toStrong context laws strongSet verifier adversary))
    ((prefixSupport context extensionAlphabet adversary).product
      (forkSeedSupport verifier.alphabet context.arity.total))
    strongToWeakSeed weakToStrongSeed
  · intro seed
    rfl
  · intro seed
    rfl
  · intro seed
    rw [OperationalExperiment.mem_runSupport_iff]
    simp only [toStrong, strongToWeakSeed, prefixSupport,
      Support.mem_product_iff]
    constructor
    · rintro ⟨hp, hf, hc⟩
      exact ⟨⟨hp, hc⟩, hf⟩
    · rintro ⟨⟨hp, hc⟩, hf⟩
      exact ⟨hp, hf, hc⟩

/-- The product experiment underlying the weak fork mixture. -/
def weakForkProductExperiment
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (extensionAlphabet : Support Extension)
    (verifier :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform.VerifierData
        context.piRlc)
    (adversary : Adversary context ProverSeed ProverTape) :
    Experiment
      (PrefixSeed Extension shape ProverSeed ×
        ForkSample Scalar context.arity.total) where
  Seed := (PrefixSeed Extension shape ProverSeed) ×
    ForkSeed verifier.alphabet context.arity.total
  support := (prefixSupport context extensionAlphabet adversary).product
    (forkSeedSupport verifier.alphabet context.arity.total)
  outcome := fun seed =>
    (seed.1, forkSample context verifier adversary
      (prefixExecution context adversary seed.1) seed.2)

/-- The lifted weak fork mixture is literally the uniform experiment on the
prefix/fork Cartesian product.  This is an exact finite-support identity;
neither side changes multiplicity or assumes a probability coupling. -/
theorem forkMixture_probability_eq_product
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (extensionAlphabet : Support Extension)
    (verifier :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform.VerifierData
        context.piRlc)
    (adversary : Adversary context ProverSeed ProverTape)
    (event : PrefixSeed Extension shape ProverSeed ×
      ForkSample Scalar context.arity.total -> Prop) :
    (Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.forkMixture
        verifier (toWeak context extensionAlphabet adversary)).probability
      event =
    (weakForkProductExperiment context extensionAlphabet verifier adversary).probability
      event := by
  exact Mixture.sharedSupport_probability_eq_product
    (prefixSupport context extensionAlphabet adversary)
    (forkSeedSupport verifier.alphabet context.arity.total)
    (fun outer seed =>
      (outer, forkSample context verifier adversary
        (prefixExecution context adversary outer) seed))
    event

/-- Under the explicit seed reassociation, literal strong ambient success is
exactly the abort-gated weak extraction event. -/
theorem strongAmbientEvent_iff_weakExtracts
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (laws : ExtractionAlgebra context.piRlc.semantics context.piRlc.params
      context.piRlc.algebra)
    (strongSet : StrongSetUnits laws.ring
      context.piRlc.algebra.challengeValid)
    (extensionAlphabet : Support Extension)
    (verifier :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform.VerifierData
        context.piRlc)
    (adversary : Adversary context ProverSeed ProverTape)
    (seed : OperationalExperiment.RunSeed Extension shape ProverSeed
      (ForkSeed verifier.alphabet context.arity.total)) :
    OperationalExperiment.success context.piCcs
          ((OperationalExperiment.experiment context.piCcs extensionAlphabet
            (toStrong context laws strongSet verifier adversary)).outcome seed) =
        true <->
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.Extracts
        laws strongSet (toWeak context extensionAlphabet adversary)
        ((weakForkProductExperiment context extensionAlphabet verifier
          adversary).outcome (strongToWeakSeed seed)) := by
  simpa [OperationalExperiment.experiment, OperationalExperiment.success,
    OperationalExperiment.run, toStrong, strongToWeakSeed,
    weakForkProductExperiment,
    Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.Extracts,
    toWeak, prefixExecution] using
    (targetAmbient_iff_extracts context laws strongSet verifier adversary
      seed.2.1
      (prefixExecution context adversary (seed.1, seed.2.2)))

/-- The strong game's intermediate ambient-success probability is exactly the
weak game's abort-gated extraction probability.  The equality is derived
from one explicit support permutation and the pointwise event equivalence
above. -/
theorem intermediateProbability
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {ProverTape : Type uProverTape}
    (context : CompatibleContext Extension Commitment PublicInput Scalar
      shape columns blockCount)
    (laws : ExtractionAlgebra context.piRlc.semantics context.piRlc.params
      context.piRlc.algebra)
    (strongSet : StrongSetUnits laws.ring
      context.piRlc.algebra.challengeValid)
    (extensionAlphabet : Support Extension)
    (verifier :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform.VerifierData
        context.piRlc)
    (adversary : Adversary context ProverSeed ProverTape) :
    (OperationalExperiment.experiment context.piCcs extensionAlphabet
        (toStrong context laws strongSet verifier adversary)).probabilityBool
          (OperationalExperiment.success context.piCcs) =
      (Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.forkMixture
        verifier (toWeak context extensionAlphabet adversary)).probability
          (Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.Extracts
            laws strongSet (toWeak context extensionAlphabet adversary)) := by
  let strongExperiment :=
    OperationalExperiment.experiment context.piCcs extensionAlphabet
      (toStrong context laws strongSet verifier adversary)
  let weakProduct :=
    weakForkProductExperiment context extensionAlphabet verifier adversary
  calc
    strongExperiment.probabilityBool
          (OperationalExperiment.success context.piCcs) =
        strongExperiment.probability
          (fun execution =>
            OperationalExperiment.success context.piCcs execution = true) :=
      (strongExperiment.probability_bool_event
        (OperationalExperiment.success context.piCcs)).symm
    _ = weakProduct.probability
          (Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.Extracts
            laws strongSet (toWeak context extensionAlphabet adversary)) := by
      apply Experiment.probability_eq_of_reindex strongExperiment weakProduct
        strongToWeakSeed
        (strongToWeakSeed_supportPermutation context laws strongSet
          extensionAlphabet verifier adversary)
      intro seed _member
      exact strongAmbientEvent_iff_weakExtracts context laws strongSet
        extensionAlphabet verifier adversary seed
    _ = (Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.forkMixture
          verifier (toWeak context extensionAlphabet adversary)).probability
            (Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.Extracts
              laws strongSet (toWeak context extensionAlphabet adversary)) :=
      (forkMixture_probability_eq_product context extensionAlphabet verifier
        adversary _).symm

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition
