import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition

/-!
Two-run operational coupling for paper `Pi_CCS` followed by paper `Pi_RLC`.

Owns: the independent two-run seed reassociation, exact flattening of both
finite mixtures, and equality of the literal strong witness-disagreement and
weak paired-extractor events.

Does not own: either component reduction theorem, one-run extraction,
`Pi_DEC`, Fiat--Shamir, Rust, R1CS, artifacts, or costs.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.Repeated

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcBatch
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition
open Nightstream.SuperNeo.Folding.PiRLC.PaperCompleteness
open Nightstream.SuperNeo.Folding.PiRLC.PaperForkExtraction
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open MatrixCoefficientSource
open PaperLinearAlgebra

universe uExtension uCommitment uPublicInput uScalar uProverSeed uProverTape

/-- Both repetitions use the same sequential adversary, sampled
independently by the paired experiment. -/
def pairedWeak
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
    (adversary : Adversary context ProverSeed ProverTape) :
    Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.PairedAdversary
      context.piRlc (PrefixSeed Extension shape ProverSeed) where
  left := toWeak context extensionAlphabet adversary
  right := toWeak context extensionAlphabet adversary

/-- Flat Cartesian-product form of the strong iid-pair experiment. -/
noncomputable def strongPairProductExperiment
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
    Experiment
      (Execution Extension shape columns × Execution Extension shape columns) :=
  let strongAdversary := toStrong context laws strongSet verifier adversary
  let base := OperationalExperiment.experiment context.piCcs extensionAlphabet
    strongAdversary
  { Seed := base.Seed × base.Seed
    support := base.support.product base.support
    outcome := fun seed => (base.outcome seed.1, base.outcome seed.2) }

/-- Flat Cartesian-product form of the weak paired-prefix/fork experiment. -/
def weakPairProductExperiment
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
    (adversary : Adversary context ProverSeed ProverTape) :
    Experiment
      (((PrefixSeed Extension shape ProverSeed) ×
          (PrefixSeed Extension shape ProverSeed)) ×
        (ForkSample Scalar context.arity.total ×
          ForkSample Scalar context.arity.total)) where
  Seed :=
    ((PrefixSeed Extension shape ProverSeed) ×
      (PrefixSeed Extension shape ProverSeed)) ×
    (ForkSeed verifier.alphabet context.arity.total ×
      ForkSeed verifier.alphabet context.arity.total)
  support :=
    ((prefixSupport context extensionAlphabet adversary).product
      (prefixSupport context extensionAlphabet adversary)).product
    ((forkSeedSupport verifier.alphabet context.arity.total).product
      (forkSeedSupport verifier.alphabet context.arity.total))
  outcome := fun seed =>
    (seed.1,
      (forkSample context verifier adversary
          (prefixExecution context adversary seed.1.1) seed.2.1,
        forkSample context verifier adversary
          (prefixExecution context adversary seed.1.2) seed.2.2))

/-- The iid strong mixture is exactly its flat product enumeration. -/
theorem strongIid_probability_eq_product
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
    (adversary : Adversary context ProverSeed ProverTape)
    (event : Execution Extension shape columns ×
      Execution Extension shape columns -> Prop) :
    (OperationalExperiment.experiment context.piCcs extensionAlphabet
        (toStrong context laws strongSet verifier adversary)).iidPair.probability
      event =
    (strongPairProductExperiment context laws strongSet extensionAlphabet
      verifier adversary).probability event := by
  exact Mixture.sharedSupport_probability_eq_product
    (OperationalExperiment.experiment context.piCcs extensionAlphabet
      (toStrong context laws strongSet verifier adversary)).support
    (OperationalExperiment.experiment context.piCcs extensionAlphabet
      (toStrong context laws strongSet verifier adversary)).support
    (fun firstSeed secondSeed =>
      ((OperationalExperiment.experiment context.piCcs extensionAlphabet
        (toStrong context laws strongSet verifier adversary)).outcome firstSeed,
       (OperationalExperiment.experiment context.piCcs extensionAlphabet
        (toStrong context laws strongSet verifier adversary)).outcome secondSeed))
    event

/-- The weak paired mixture is exactly its flat prefix-pair/fork-pair
enumeration. -/
theorem weakPaired_probability_eq_product
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
    (event : ((PrefixSeed Extension shape ProverSeed) ×
        (PrefixSeed Extension shape ProverSeed)) ×
      (ForkSample Scalar context.arity.total ×
        ForkSample Scalar context.arity.total) -> Prop) :
    (Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.pairedMixture
        verifier (pairedWeak context extensionAlphabet adversary)).probability
      event =
    (weakPairProductExperiment context extensionAlphabet verifier
      adversary).probability event := by
  exact Mixture.sharedSupport_probability_eq_product
    ((prefixSupport context extensionAlphabet adversary).product
      (prefixSupport context extensionAlphabet adversary))
    ((forkSeedSupport verifier.alphabet context.arity.total).product
      (forkSeedSupport verifier.alphabet context.arity.total))
    (fun prefixes seeds =>
      (prefixes,
        (forkSample context verifier adversary
            (prefixExecution context adversary prefixes.1) seeds.1,
          forkSample context verifier adversary
            (prefixExecution context adversary prefixes.2) seeds.2)))
    event

/-- Reassociate two independent strong seeds into independent prefix and fork
pairs. -/
def strongPairToWeakSeed
    {Extension : Type uExtension}
    {Scalar : Type uScalar}
    {shape : Shape}
    {ProverSeed : Type uProverSeed}
    {coordinates : Nat}
    {alphabet : Support Scalar}
    (seed :
      (OperationalExperiment.RunSeed Extension shape ProverSeed
        (ForkSeed alphabet coordinates)) ×
      (OperationalExperiment.RunSeed Extension shape ProverSeed
        (ForkSeed alphabet coordinates))) :
    (((PrefixSeed Extension shape ProverSeed) ×
        (PrefixSeed Extension shape ProverSeed)) ×
      (ForkSeed alphabet coordinates × ForkSeed alphabet coordinates)) :=
  (((seed.1.1, seed.1.2.2), (seed.2.1, seed.2.2.2)),
    (seed.1.2.1, seed.2.2.1))

/-- Inverse two-run seed reassociation. -/
def weakPairToStrongSeed
    {Extension : Type uExtension}
    {Scalar : Type uScalar}
    {shape : Shape}
    {ProverSeed : Type uProverSeed}
    {coordinates : Nat}
    {alphabet : Support Scalar}
    (seed :
      (((PrefixSeed Extension shape ProverSeed) ×
          (PrefixSeed Extension shape ProverSeed)) ×
        (ForkSeed alphabet coordinates × ForkSeed alphabet coordinates))) :
    (OperationalExperiment.RunSeed Extension shape ProverSeed
        (ForkSeed alphabet coordinates)) ×
      (OperationalExperiment.RunSeed Extension shape ProverSeed
        (ForkSeed alphabet coordinates)) :=
  ((seed.1.1.1, (seed.2.1, seed.1.1.2)),
    (seed.1.2.1, (seed.2.2, seed.1.2.2)))

/-- The two flat supports differ only by the explicit seed reassociation. -/
theorem strongPairToWeakSeed_supportPermutation
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
    (((strongPairProductExperiment context laws strongSet extensionAlphabet
        verifier adversary).support.values.map strongPairToWeakSeed).Perm
      (weakPairProductExperiment context extensionAlphabet verifier
        adversary).support.values) := by
  apply Support.map_values_perm_of_inverse
    (strongPairProductExperiment context laws strongSet extensionAlphabet
      verifier adversary).support
    (weakPairProductExperiment context extensionAlphabet verifier
      adversary).support
    strongPairToWeakSeed weakPairToStrongSeed
  · intro seed
    rfl
  · intro seed
    rfl
  · intro seed
    constructor
    · intro member
      have runPair := (Support.mem_product_iff _ _ seed).mp member
      have leftRun :=
        (OperationalExperiment.mem_runSupport_iff context.piCcs
          extensionAlphabet (toStrong context laws strongSet verifier
            adversary) seed.1).mp runPair.1
      have rightRun :=
        (OperationalExperiment.mem_runSupport_iff context.piCcs
          extensionAlphabet (toStrong context laws strongSet verifier
            adversary) seed.2).mp runPair.2
      have leftPrefix :
          (seed.1.1, seed.1.2.2) ∈
            (prefixSupport context extensionAlphabet adversary).values := by
        apply (Support.mem_product_iff _ _ _).mpr
        exact ⟨by simpa [toStrong] using leftRun.1, leftRun.2.2⟩
      have rightPrefix :
          (seed.2.1, seed.2.2.2) ∈
            (prefixSupport context extensionAlphabet adversary).values := by
        apply (Support.mem_product_iff _ _ _).mpr
        exact ⟨by simpa [toStrong] using rightRun.1, rightRun.2.2⟩
      apply (Support.mem_product_iff _ _ (strongPairToWeakSeed seed)).mpr
      constructor
      · apply (Support.mem_product_iff _ _ _).mpr
        simpa [strongPairToWeakSeed] using And.intro leftPrefix rightPrefix
      · apply (Support.mem_product_iff _ _ _).mpr
        simpa [strongPairToWeakSeed] using And.intro leftRun.2.1 rightRun.2.1
    · intro member
      have outerInner :=
        (Support.mem_product_iff _ _ (strongPairToWeakSeed seed)).mp member
      have prefixPair := (Support.mem_product_iff _ _ _).mp outerInner.1
      have forkPair := (Support.mem_product_iff _ _ _).mp outerInner.2
      have leftPrefix :
          (seed.1.1, seed.1.2.2) ∈
            (prefixSupport context extensionAlphabet adversary).values := by
        simpa [strongPairToWeakSeed] using prefixPair.1
      have rightPrefix :
          (seed.2.1, seed.2.2.2) ∈
            (prefixSupport context extensionAlphabet adversary).values := by
        simpa [strongPairToWeakSeed] using prefixPair.2
      have leftPrefixParts := (Support.mem_product_iff _ _ _).mp leftPrefix
      have rightPrefixParts := (Support.mem_product_iff _ _ _).mp rightPrefix
      have leftRun : seed.1 ∈
          (OperationalExperiment.runSupport context.piCcs extensionAlphabet
            (toStrong context laws strongSet verifier adversary)).values := by
        apply (OperationalExperiment.mem_runSupport_iff context.piCcs
          extensionAlphabet (toStrong context laws strongSet verifier
            adversary) seed.1).mpr
        exact ⟨by simpa [toStrong] using leftPrefixParts.1,
          by simpa [strongPairToWeakSeed] using forkPair.1,
          leftPrefixParts.2⟩
      have rightRun : seed.2 ∈
          (OperationalExperiment.runSupport context.piCcs extensionAlphabet
            (toStrong context laws strongSet verifier adversary)).values := by
        apply (OperationalExperiment.mem_runSupport_iff context.piCcs
          extensionAlphabet (toStrong context laws strongSet verifier
            adversary) seed.2).mpr
        exact ⟨by simpa [toStrong] using rightPrefixParts.1,
          by simpa [strongPairToWeakSeed] using forkPair.2,
          rightPrefixParts.2⟩
      exact (Support.mem_product_iff _ _ seed).mpr ⟨leftRun, rightRun⟩

private theorem outputWitness_eq_of_assignments_eq
    {shape : Shape}
    {columns : Nat}
    (left right : OutputWitness shape columns)
    (equalAssignments : left.assignments = right.assignments) :
    left = right := by
  cases left
  cases right
  cases equalAssignments
  rfl

/-- Two reindexed extracted output witnesses differ exactly when one of their
`K+k` assignment coordinates differs. -/
theorem extractedWitness_ne_iff
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
    (leftPrefix rightPrefix : PrefixExecution Extension shape)
    (leftSample rightSample : ForkSample Scalar context.arity.total)
    (leftAccepted :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.AcceptedFork
        context.piRlc (component context adversary leftPrefix) leftSample)
    (rightAccepted :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.AcceptedFork
        context.piRlc (component context adversary rightPrefix) rightSample) :
    extractedWitness context laws strongSet adversary leftPrefix leftSample
          leftAccepted ≠
        extractedWitness context laws strongSet adversary rightPrefix
          rightSample rightAccepted <->
      exists coordinate,
        Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.extractedFamily
            context.piRlc laws strongSet
            (component context adversary leftPrefix) leftSample leftAccepted
            coordinate ≠
          Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.extractedFamily
            context.piRlc laws strongSet
            (component context adversary rightPrefix) rightSample rightAccepted
            coordinate := by
  classical
  constructor
  · intro different
    by_cases existsDifference : exists coordinate,
        Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.extractedFamily
            context.piRlc laws strongSet
            (component context adversary leftPrefix) leftSample leftAccepted
            coordinate ≠
          Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.extractedFamily
            context.piRlc laws strongSet
            (component context adversary rightPrefix) rightSample rightAccepted
            coordinate
    · exact existsDifference
    · exfalso
      apply different
      apply outputWitness_eq_of_assignments_eq
      funext source
      have equalAtCoordinate := Classical.not_not.mp
        ((not_exists.mp existsDifference)
          (Fin.cast context.total_eq_sourceCount.symm source))
      simpa [extractedWitness] using equalAtCoordinate
  · rintro ⟨coordinate, different⟩ equalWitness
    apply different
    have equalAtSource := congrArg
      (fun witness => witness.assignments (context.sourceIndex coordinate))
      equalWitness
    simpa [extractedWitness, CompatibleContext.sourceIndex] using equalAtSource

/-- Any accepted coordinate-fork proof selects the same concrete extracted
witness in the strong target; proof irrelevance removes the choice of proof. -/
theorem targetWitness_eq_some_of_accepted
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
    (causalRun : PrefixExecution Extension shape)
    (accepted :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.AcceptedFork
        context.piRlc (component context adversary causalRun)
        (forkSample context verifier adversary causalRun seed)) :
    targetWitness context laws strongSet verifier adversary seed causalRun =
      some (extractedWitness context laws strongSet adversary causalRun
        (forkSample context verifier adversary causalRun seed) accepted) := by
  simp [targetWitness, accepted]

/-- Pointwise equality of the two-run events before any probability
calculation.  Rejected coordinate forks make both sides false; accepted
forks compare the same reindexed assignment families. -/
theorem pairedEvent_iff
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
    (leftPrefixSeed rightPrefixSeed :
      PrefixSeed Extension shape ProverSeed)
    (leftForkSeed rightForkSeed :
      ForkSeed verifier.alphabet context.arity.total) :
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents.witnessDisagreement
        context.piCcs
        (attachWitness (prefixExecution context adversary leftPrefixSeed)
            (targetWitness context laws strongSet verifier adversary
              leftForkSeed
              (prefixExecution context adversary leftPrefixSeed)),
          attachWitness (prefixExecution context adversary rightPrefixSeed)
            (targetWitness context laws strongSet verifier adversary
              rightForkSeed
              (prefixExecution context adversary rightPrefixSeed))) = true <->
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.PairedDisagreement
        laws strongSet (pairedWeak context extensionAlphabet adversary)
        ((leftPrefixSeed, rightPrefixSeed),
          (forkSample context verifier adversary
              (prefixExecution context adversary leftPrefixSeed) leftForkSeed,
            forkSample context verifier adversary
              (prefixExecution context adversary rightPrefixSeed)
              rightForkSeed)) := by
  let leftPrefix := prefixExecution context adversary leftPrefixSeed
  let rightPrefix := prefixExecution context adversary rightPrefixSeed
  let leftSample :=
    forkSample context verifier adversary leftPrefix leftForkSeed
  let rightSample :=
    forkSample context verifier adversary rightPrefix rightForkSeed
  change
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents.witnessDisagreement
        context.piCcs
        (attachWitness leftPrefix
            (targetWitness context laws strongSet verifier adversary
              leftForkSeed leftPrefix),
          attachWitness rightPrefix
            (targetWitness context laws strongSet verifier adversary
              rightForkSeed rightPrefix)) = true <->
      acceptedCheck context.piCcs leftPrefix = true /\
        acceptedCheck context.piCcs rightPrefix = true /\
        exists leftAccepted :
          Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.AcceptedFork
            context.piRlc (component context adversary leftPrefix) leftSample,
        exists rightAccepted :
          Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.AcceptedFork
            context.piRlc (component context adversary rightPrefix) rightSample,
        exists coordinate,
          Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.extractedFamily
              context.piRlc laws strongSet
              (component context adversary leftPrefix) leftSample leftAccepted
              coordinate ≠
            Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.extractedFamily
              context.piRlc laws strongSet
              (component context adversary rightPrefix) rightSample
              rightAccepted coordinate
  by_cases leftAccepted :
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.AcceptedFork
        context.piRlc (component context adversary leftPrefix) leftSample
  · by_cases rightAccepted :
        Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.AcceptedFork
          context.piRlc (component context adversary rightPrefix) rightSample
    · have leftTarget := targetWitness_eq_some_of_accepted context laws
        strongSet verifier adversary leftForkSeed leftPrefix leftAccepted
      have rightTarget := targetWitness_eq_some_of_accepted context laws
        strongSet verifier adversary rightForkSeed rightPrefix rightAccepted
      have leftExtracts :=
        Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.acceptedFork_extracts_correctedAmbient
          context.piRlc laws strongSet (component context adversary leftPrefix)
          leftSample leftAccepted
      have rightExtracts :=
        Nightstream.SuperNeo.Folding.PiRLC.PaperWeakReduction.acceptedFork_extracts_correctedAmbient
          context.piRlc laws strongSet
          (component context adversary rightPrefix) rightSample rightAccepted
      constructor
      · intro disagreement
        have semantic :=
          (Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents.witnessDisagreement_eq_true_iff
            context.piCcs _).1 disagreement
        have leftAcceptedPrefix :=
          (targetAmbient_iff_extracts context laws strongSet verifier adversary
            leftForkSeed leftPrefix).1
            ((ambientCheck_eq_true_iff context.piCcs _).2 semantic.1.1)
        have rightAcceptedPrefix :=
          (targetAmbient_iff_extracts context laws strongSet verifier adversary
            rightForkSeed rightPrefix).1
            ((ambientCheck_eq_true_iff context.piCcs _).2 semantic.1.2)
        rcases semantic.2 with
          ⟨leftWitness, rightWitness, leftEquation, rightEquation,
            witnessesDifferent⟩
        have leftWitnessEquation :
            extractedWitness context laws strongSet adversary leftPrefix
              leftSample leftAccepted = leftWitness := by
          simpa [leftSample, leftTarget, attachWitness] using leftEquation
        have rightWitnessEquation :
            extractedWitness context laws strongSet adversary rightPrefix
              rightSample rightAccepted = rightWitness := by
          simpa [rightSample, rightTarget, attachWitness] using rightEquation
        subst leftWitness
        subst rightWitness
        exact ⟨leftAcceptedPrefix.1, rightAcceptedPrefix.1,
          leftAccepted, rightAccepted,
          (extractedWitness_ne_iff context laws strongSet adversary
            leftPrefix rightPrefix leftSample rightSample leftAccepted
            rightAccepted).1 witnessesDifferent⟩
      · rintro ⟨leftAcceptedPrefix, rightAcceptedPrefix, otherLeftAccepted,
          otherRightAccepted, coordinate, coordinateDifferent⟩
        have leftProofEqual : otherLeftAccepted = leftAccepted :=
          Subsingleton.elim _ _
        have rightProofEqual : otherRightAccepted = rightAccepted :=
          Subsingleton.elim _ _
        subst otherLeftAccepted
        subst otherRightAccepted
        have leftAmbientCheck :=
          (targetAmbient_iff_extracts context laws strongSet verifier adversary
            leftForkSeed leftPrefix).2 ⟨leftAcceptedPrefix, leftExtracts⟩
        have rightAmbientCheck :=
          (targetAmbient_iff_extracts context laws strongSet verifier adversary
            rightForkSeed rightPrefix).2
            ⟨rightAcceptedPrefix, rightExtracts⟩
        apply
          (Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents.witnessDisagreement_eq_true_iff
            context.piCcs _).2
        refine ⟨⟨(ambientCheck_eq_true_iff context.piCcs _).1
              leftAmbientCheck,
            (ambientCheck_eq_true_iff context.piCcs _).1
              rightAmbientCheck⟩,
          extractedWitness context laws strongSet adversary leftPrefix
            leftSample leftAccepted,
          extractedWitness context laws strongSet adversary rightPrefix
            rightSample rightAccepted,
          ?_, ?_, ?_⟩
        · simpa [leftSample] using leftTarget
        · simpa [rightSample] using rightTarget
        · exact (extractedWitness_ne_iff context laws strongSet adversary
            leftPrefix rightPrefix leftSample rightSample leftAccepted
            rightAccepted).2 ⟨coordinate, coordinateDifferent⟩
    · have rightTarget :
          targetWitness context laws strongSet verifier adversary rightForkSeed
            rightPrefix = none := by
        simp [targetWitness, rightSample, rightAccepted]
      constructor
      · intro disagreement
        have semantic :=
          (Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents.witnessDisagreement_eq_true_iff
            context.piCcs _).1 disagreement
        rcases semantic.2 with
          ⟨_leftWitness, rightWitness, _leftEquation, rightEquation, _⟩
        rw [rightTarget] at rightEquation
        contradiction
      · rintro ⟨_, _, _, otherRightAccepted, _⟩
        exact False.elim (rightAccepted otherRightAccepted)
  · have leftTarget :
        targetWitness context laws strongSet verifier adversary leftForkSeed
          leftPrefix = none := by
      simp [targetWitness, leftSample, leftAccepted]
    constructor
    · intro disagreement
      have semantic :=
        (Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents.witnessDisagreement_eq_true_iff
          context.piCcs _).1 disagreement
      rcases semantic.2 with
        ⟨leftWitness, _rightWitness, leftEquation, _rightEquation, _⟩
      rw [leftTarget] at leftEquation
      contradiction
    · rintro ⟨_, _, otherLeftAccepted, _⟩
      exact False.elim (leftAccepted otherLeftAccepted)

/-- `pairedEvent_iff` specialized to the two flat experiments and their
explicit seed reassociation. -/
theorem strongPairEvent_iff_weakPairEvent
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
    (seed : (strongPairProductExperiment context laws strongSet
      extensionAlphabet verifier adversary).Seed) :
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents.witnessDisagreement
        context.piCcs
        ((strongPairProductExperiment context laws strongSet extensionAlphabet
          verifier adversary).outcome seed) = true <->
      Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.PairedDisagreement
        laws strongSet (pairedWeak context extensionAlphabet adversary)
        ((weakPairProductExperiment context extensionAlphabet verifier
          adversary).outcome (strongPairToWeakSeed seed)) := by
  simpa [strongPairProductExperiment, weakPairProductExperiment,
    OperationalExperiment.experiment, OperationalExperiment.run, toStrong,
    strongPairToWeakSeed] using
    (pairedEvent_iff context laws strongSet extensionAlphabet verifier adversary
      (seed.1.1, seed.1.2.2) (seed.2.1, seed.2.2.2)
      seed.1.2.1 seed.2.2.1)

/-- The literal two-run strong disagreement probability is exactly the weak
paired-extractor disagreement probability. -/
theorem repeatedWitnessProbability
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
        (toStrong context laws strongSet verifier adversary)).iidPair.probabilityBool
          (Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents.witnessDisagreement
            context.piCcs) =
      (Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.pairedMixture
        verifier (pairedWeak context extensionAlphabet adversary)).probability
          (Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.PairedDisagreement
            laws strongSet (pairedWeak context extensionAlphabet adversary)) := by
  let strongBase := OperationalExperiment.experiment context.piCcs
    extensionAlphabet (toStrong context laws strongSet verifier adversary)
  let strongProduct := strongPairProductExperiment context laws strongSet
    extensionAlphabet verifier adversary
  let weakProduct := weakPairProductExperiment context extensionAlphabet
    verifier adversary
  let strongEvent :=
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents.witnessDisagreement
      context.piCcs
  let weakEvent :=
    Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.PairedDisagreement
      laws strongSet (pairedWeak context extensionAlphabet adversary)
  calc
    strongBase.iidPair.probabilityBool strongEvent =
        strongBase.iidPair.probability
          (fun executions => strongEvent executions = true) :=
      (strongBase.iidPair.probability_bool_event strongEvent).symm
    _ = strongProduct.probability
          (fun executions => strongEvent executions = true) :=
      strongIid_probability_eq_product context laws strongSet
        extensionAlphabet verifier adversary _
    _ = weakProduct.probability weakEvent := by
      apply Experiment.probability_eq_of_reindex strongProduct weakProduct
        strongPairToWeakSeed
        (strongPairToWeakSeed_supportPermutation context laws strongSet
          extensionAlphabet verifier adversary)
      intro seed _member
      exact strongPairEvent_iff_weakPairEvent context laws strongSet
        extensionAlphabet verifier adversary seed
    _ = (Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteMixture.pairedMixture
          verifier (pairedWeak context extensionAlphabet adversary)).probability
            weakEvent :=
      (weakPaired_probability_eq_product context extensionAlphabet verifier
        adversary weakEvent).symm

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.Repeated
