import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.CausalPrefixCoupling
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition

/-!
Causal Appendix-D.4 first-success bridge for the paper non-interactive NIFS.

Source: SuperNeo Definition 10, Section 7.3, and Appendices D.3--D.5.

Owns: the exact strong-game adversary obtained from the paper `Pi_RLC`
coordinate extractor; its independent prover/target/verifier seed support;
the replay-aligned NIFS interpretation of ambient success; and the exact
fixed-first-witness mixing-root/SumCheck event on a fresh second prefix.

Does not own: construction of an ideal random oracle, `Pi_DEC` target-witness
availability from public acceptance, collision bounds, the asymptotic
rejection sampler, Poseidon2, Ajtai, Rust, R1CS, artifacts, minimality, or
costs.

Emits constraints: no.

The first witness is always read from the first strong-game execution before
the second prefix is interpreted.  The NIFS event therefore cannot choose a
witness after seeing the fresh second verifier coins.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalExperiment
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

universe uExtension uCommitment uPublicInput uScalar uState

namespace CausalPrefixCouplingContract

/-- One D.4 execution seed: prover randomness, independently sampled
`Pi_RLC` target-extractor randomness, and independently sampled public
`Pi_CCS` verifier coins. -/
abbrev D4RunSeed
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    (contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet)
    (verifier : PiRLC.PaperWeakFiniteUniform.VerifierData
      (key.compatibleContext running fresh).piRlc) :=
  OperationalExperiment.RunSeed Extension shape contract.ProverSeed
    (ForkSeed verifier.alphabet key.arity.total)

/-- The exact paper D.4 strong adversary selected after applying the
`Pi_RLC` coordinate extractor to the causal sequential adversary. -/
noncomputable def d4Adversary
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    (contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet)
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      (key.compatibleContext running fresh).piRlc.semantics
      (key.compatibleContext running fresh).piRlc.params
      (key.compatibleContext running fresh).piRlc.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      (key.compatibleContext running fresh).piRlc.algebra.challengeValid)
    (verifier : PiRLC.PaperWeakFiniteUniform.VerifierData
      (key.compatibleContext running fresh).piRlc) :
    OperationalExperiment.Adversary
      (key.compatibleContext running fresh).piCcs
      contract.ProverSeed
      (ForkSeed verifier.alphabet key.arity.total)
      contract.ProverTape :=
  PiRlcComposition.toStrong
    (key.compatibleContext running fresh) laws strongSet verifier
    (causalPiRlcAdversary key running fresh contract.adversary)

/-- Forget the target-extractor seed and recover the exact prefix seed that
was fixed before `Pi_RLC` extraction. -/
def d4PrefixSeed
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    {contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet}
    {verifier : PiRLC.PaperWeakFiniteUniform.VerifierData
      (key.compatibleContext running fresh).piRlc}
    (seed : D4RunSeed contract verifier) :
    PiRlcComposition.PrefixSeed Extension shape contract.ProverSeed :=
  (seed.1, seed.2.2)

/-- Execute one literal paper strong-game run. -/
noncomputable def d4Execution
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    (contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet)
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      (key.compatibleContext running fresh).piRlc.semantics
      (key.compatibleContext running fresh).piRlc.params
      (key.compatibleContext running fresh).piRlc.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      (key.compatibleContext running fresh).piRlc.algebra.challengeValid)
    (verifier : PiRLC.PaperWeakFiniteUniform.VerifierData
      (key.compatibleContext running fresh).piRlc)
    (seed : D4RunSeed contract verifier) :
    Execution Extension shape columns :=
  OperationalExperiment.run
    (key.compatibleContext running fresh).piCcs
    (contract.d4Adversary laws strongSet verifier) seed

/-- The realized NIFS key selected by the seed's causal prefix. -/
noncomputable def d4PrefixKey
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    {contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet}
    {verifier : PiRLC.PaperWeakFiniteUniform.VerifierData
      (key.compatibleContext running fresh).piRlc}
    (seed : D4RunSeed contract verifier) :
    Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound :=
  contract.toPrefixExperiment.realizedKey (d4PrefixSeed seed)

/-- The exact NIFS proof replayed in the seed's realized prefix world. -/
noncomputable def d4PrefixProof
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    {contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet}
    {verifier : PiRLC.PaperWeakFiniteUniform.VerifierData
      (key.compatibleContext running fresh).piRlc}
    (seed : D4RunSeed contract verifier) :
    Proof Extension Commitment shape degreeBound :=
  (contract.toPrefixExperiment.prover (d4PrefixSeed seed)).baseProof
    running fresh

/-- Prefix-oracle realization changes only transcript fields, so the
paper-interactive context is unchanged. -/
@[simp] theorem d4PrefixKey_compatibleContext
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    {contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet}
    {verifier : PiRLC.PaperWeakFiniteUniform.VerifierData
      (key.compatibleContext running fresh).piRlc}
    (seed : D4RunSeed contract verifier) :
    (d4PrefixKey seed).compatibleContext running fresh =
      key.compatibleContext running fresh := by
  rfl

/-- The strong execution's causal prefix is definitionally the prefix fixed
by the NIFS coupling contract. -/
@[simp] theorem d4Execution_causalRun
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    (contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet)
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      (key.compatibleContext running fresh).piRlc.semantics
      (key.compatibleContext running fresh).piRlc.params
      (key.compatibleContext running fresh).piRlc.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      (key.compatibleContext running fresh).piRlc.algebra.challengeValid)
    (verifier : PiRLC.PaperWeakFiniteUniform.VerifierData
      (key.compatibleContext running fresh).piRlc)
    (seed : D4RunSeed contract verifier) :
    (contract.d4Execution laws strongSet verifier seed).causalRun =
      causalPrefixRun key running fresh contract.adversary
        (d4PrefixSeed seed) := by
  rfl

/-- Every D.4 seed retains the exact replay receipt of its NIFS prefix. -/
theorem d4PrefixAlignment
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    (contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet)
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      (key.compatibleContext running fresh).piRlc.semantics
      (key.compatibleContext running fresh).piRlc.params
      (key.compatibleContext running fresh).piRlc.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      (key.compatibleContext running fresh).piRlc.algebra.challengeValid)
    (verifier : PiRLC.PaperWeakFiniteUniform.VerifierData
      (key.compatibleContext running fresh).piRlc)
    (seed : D4RunSeed contract verifier) :
    CausalPrefixAlignment (d4PrefixKey seed) running fresh
      (d4PrefixProof seed)
      (contract.d4Execution laws strongSet verifier seed).causalRun := by
  simpa [d4PrefixKey, d4PrefixProof] using
    contract.prefixAligned (d4PrefixSeed seed)

/-- Replay alignment identifies the corrected ambient target relation on the
interactive prefix with the typed NIFS ambient-opening relation. -/
theorem d4AmbientOutputHolds_iff_nifs
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    (contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet)
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      (key.compatibleContext running fresh).piRlc.semantics
      (key.compatibleContext running fresh).piRlc.params
      (key.compatibleContext running fresh).piRlc.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      (key.compatibleContext running fresh).piRlc.algebra.challengeValid)
    (verifier : PiRLC.PaperWeakFiniteUniform.VerifierData
      (key.compatibleContext running fresh).piRlc)
    (seed : D4RunSeed contract verifier)
    (witness : OutputWitness shape columns) :
    AmbientOutputHolds
        (key.compatibleContext running fresh).piCcs.extensionOps
        (key.compatibleContext running fresh).piCcs.lift
        (key.compatibleContext running fresh).piCcs.openingMaps
        (key.compatibleContext running fresh).piCcs.params
        (key.compatibleContext running fresh).piCcs.statement
        (contract.d4Execution laws strongSet verifier seed).causalRun.probe
        witness <->
      AmbientTargetOpenings (d4PrefixKey seed) running fresh
        (d4PrefixProof seed) witness := by
  unfold AmbientTargetOpenings
  rw [(contract.d4PrefixAlignment laws strongSet verifier seed).probe_eq]
  rfl

/-- Typed NIFS interpretation of one D.4 ambient-success seed. -/
noncomputable def NifsD4AmbientSuccess
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    (contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet)
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      (key.compatibleContext running fresh).piRlc.semantics
      (key.compatibleContext running fresh).piRlc.params
      (key.compatibleContext running fresh).piRlc.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      (key.compatibleContext running fresh).piRlc.algebra.challengeValid)
    (verifier : PiRLC.PaperWeakFiniteUniform.VerifierData
      (key.compatibleContext running fresh).piRlc)
    (seed : D4RunSeed contract verifier) : Prop :=
  match (contract.d4Execution laws strongSet verifier seed).target with
  | none => False
  | some witness =>
      piCcsCheck (d4PrefixKey seed) running fresh (d4PrefixProof seed) =
          true /\
        AmbientTargetOpenings (d4PrefixKey seed) running fresh
          (d4PrefixProof seed) witness

private noncomputable def propositionEvent (proposition : Prop) : Bool :=
  @ite Bool proposition (Classical.propDecidable proposition) true false

@[simp] private theorem propositionEvent_eq_true
    (proposition : Prop) :
    propositionEvent proposition = true <-> proposition := by
  simp [propositionEvent]

/-- Boolean event used by the finite seed experiment. -/
noncomputable def nifsD4AmbientSuccessEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    (contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet)
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      (key.compatibleContext running fresh).piRlc.semantics
      (key.compatibleContext running fresh).piRlc.params
      (key.compatibleContext running fresh).piRlc.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      (key.compatibleContext running fresh).piRlc.algebra.challengeValid)
    (verifier : PiRLC.PaperWeakFiniteUniform.VerifierData
      (key.compatibleContext running fresh).piRlc) :
    D4RunSeed contract verifier -> Bool :=
  fun seed =>
    propositionEvent
      (contract.NifsD4AmbientSuccess laws strongSet verifier seed)

@[simp] theorem nifsD4AmbientSuccessEvent_eq_true
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    (contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet)
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      (key.compatibleContext running fresh).piRlc.semantics
      (key.compatibleContext running fresh).piRlc.params
      (key.compatibleContext running fresh).piRlc.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      (key.compatibleContext running fresh).piRlc.algebra.challengeValid)
    (verifier : PiRLC.PaperWeakFiniteUniform.VerifierData
      (key.compatibleContext running fresh).piRlc)
    (seed : D4RunSeed contract verifier) :
    contract.nifsD4AmbientSuccessEvent laws strongSet verifier seed = true <->
      contract.NifsD4AmbientSuccess laws strongSet verifier seed := by
  simp [nifsD4AmbientSuccessEvent]

/-- The paper strong-game success predicate and the typed replay-aligned NIFS
success predicate are pointwise identical. -/
theorem d4Success_iff_nifsD4AmbientSuccess
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    (contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet)
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      (key.compatibleContext running fresh).piRlc.semantics
      (key.compatibleContext running fresh).piRlc.params
      (key.compatibleContext running fresh).piRlc.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      (key.compatibleContext running fresh).piRlc.algebra.challengeValid)
    (verifier : PiRLC.PaperWeakFiniteUniform.VerifierData
      (key.compatibleContext running fresh).piRlc)
    (seed : D4RunSeed contract verifier) :
    OperationalExperiment.success
        (key.compatibleContext running fresh).piCcs
        (contract.d4Execution laws strongSet verifier seed) = true <->
      contract.NifsD4AmbientSuccess laws strongSet verifier seed := by
  rw [OperationalExperiment.success, ambientCheck_eq_true_iff]
  unfold AmbientSuccess NifsD4AmbientSuccess
  cases (contract.d4Execution laws strongSet verifier seed).target with
  | none => simp
  | some witness =>
      simp only
      let context := (key.compatibleContext running fresh).piCcs
      let alignment :=
        contract.d4PrefixAlignment laws strongSet verifier seed
      have gateEq :
          acceptedCheck context
              (contract.d4Execution laws strongSet verifier seed).causalRun =
            piCcsCheck (d4PrefixKey seed) running fresh
              (d4PrefixProof seed) := by
        simpa only [d4PrefixKey_compatibleContext] using
          acceptedCheck_eq_piCcsCheck alignment
      constructor
      · rintro ⟨accepted, ambient⟩
        have interactiveAccepted :
            acceptedCheck context
                (contract.d4Execution laws strongSet verifier seed).causalRun =
              true :=
          (acceptedCheck_eq_true_iff context _).2 accepted
        rw [gateEq] at interactiveAccepted
        exact ⟨interactiveAccepted,
          (contract.d4AmbientOutputHolds_iff_nifs laws strongSet verifier
            seed witness).1 ambient⟩
      · rintro ⟨accepted, ambient⟩
        have interactiveAccepted :
            acceptedCheck context
                (contract.d4Execution laws strongSet verifier seed).causalRun =
              true := by
          rw [gateEq]
          exact accepted
        exact ⟨(acceptedCheck_eq_true_iff context _).1 interactiveAccepted,
          (contract.d4AmbientOutputHolds_iff_nifs laws strongSet verifier
            seed witness).2 ambient⟩

/-- Boolean equality form used to reindex finite probabilities. -/
theorem nifsD4AmbientSuccessEvent_eq_operational
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    (contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet)
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      (key.compatibleContext running fresh).piRlc.semantics
      (key.compatibleContext running fresh).piRlc.params
      (key.compatibleContext running fresh).piRlc.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      (key.compatibleContext running fresh).piRlc.algebra.challengeValid)
    (verifier : PiRLC.PaperWeakFiniteUniform.VerifierData
      (key.compatibleContext running fresh).piRlc)
    (seed : D4RunSeed contract verifier) :
    contract.nifsD4AmbientSuccessEvent laws strongSet verifier seed =
      OperationalExperiment.success
        (key.compatibleContext running fresh).piCcs
        (contract.d4Execution laws strongSet verifier seed) := by
  apply Bool.eq_iff_iff.mpr
  rw [nifsD4AmbientSuccessEvent_eq_true]
  exact
    (contract.d4Success_iff_nifsD4AmbientSuccess laws strongSet verifier
      seed).symm

/-- The NIFS fixed-first event reads the target witness from the first run
and tests only the fresh second prefix. -/
noncomputable def NifsD4FixedFirstBad
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    (contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet)
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      (key.compatibleContext running fresh).piRlc.semantics
      (key.compatibleContext running fresh).piRlc.params
      (key.compatibleContext running fresh).piRlc.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      (key.compatibleContext running fresh).piRlc.algebra.challengeValid)
    (verifier : PiRLC.PaperWeakFiniteUniform.VerifierData
      (key.compatibleContext running fresh).piRlc)
    (seeds : D4RunSeed contract verifier × D4RunSeed contract verifier) :
    Prop :=
  match (contract.d4Execution laws strongSet verifier seeds.1).target with
  | none => False
  | some witness =>
      PiCcsMixingRoot (d4PrefixKey seeds.2) running fresh
          (d4PrefixProof seeds.2) witness \/
        PiCcsSumCheckCollision (d4PrefixKey seeds.2) running fresh
          (d4PrefixProof seeds.2) witness

/-- Boolean fixed-first event for finite experiments. -/
noncomputable def nifsD4FixedFirstBadEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    (contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet)
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      (key.compatibleContext running fresh).piRlc.semantics
      (key.compatibleContext running fresh).piRlc.params
      (key.compatibleContext running fresh).piRlc.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      (key.compatibleContext running fresh).piRlc.algebra.challengeValid)
    (verifier : PiRLC.PaperWeakFiniteUniform.VerifierData
      (key.compatibleContext running fresh).piRlc) :
    D4RunSeed contract verifier × D4RunSeed contract verifier -> Bool :=
  fun seeds =>
    propositionEvent
      (contract.NifsD4FixedFirstBad laws strongSet verifier seeds)

@[simp] theorem nifsD4FixedFirstBadEvent_eq_true
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    (contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet)
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      (key.compatibleContext running fresh).piRlc.semantics
      (key.compatibleContext running fresh).piRlc.params
      (key.compatibleContext running fresh).piRlc.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      (key.compatibleContext running fresh).piRlc.algebra.challengeValid)
    (verifier : PiRLC.PaperWeakFiniteUniform.VerifierData
      (key.compatibleContext running fresh).piRlc)
    (seeds : D4RunSeed contract verifier × D4RunSeed contract verifier) :
    contract.nifsD4FixedFirstBadEvent laws strongSet verifier seeds = true <->
      contract.NifsD4FixedFirstBad laws strongSet verifier seeds := by
  simp [nifsD4FixedFirstBadEvent]

/-- Pointwise identity between the interactive fixed-first event and the two
typed NIFS residual events for the fresh second prefix. -/
theorem fixedFirstBad_iff_nifsD4FixedFirstBad
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    (contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet)
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      (key.compatibleContext running fresh).piRlc.semantics
      (key.compatibleContext running fresh).piRlc.params
      (key.compatibleContext running fresh).piRlc.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      (key.compatibleContext running fresh).piRlc.algebra.challengeValid)
    (verifier : PiRLC.PaperWeakFiniteUniform.VerifierData
      (key.compatibleContext running fresh).piRlc)
    (seeds : D4RunSeed contract verifier × D4RunSeed contract verifier) :
    fixedFirstBad (key.compatibleContext running fresh).piCcs
        (contract.d4Execution laws strongSet verifier seeds.1,
          contract.d4Execution laws strongSet verifier seeds.2) = true <->
      contract.NifsD4FixedFirstBad laws strongSet verifier seeds := by
  cases firstTarget :
      (contract.d4Execution laws strongSet verifier seeds.1).target with
  | none =>
      simp [fixedFirstBad, NifsD4FixedFirstBad, firstTarget]
  | some witness =>
      have mixing :=
        mixingFailure_iff_piCcsMixingRoot
          (contract.d4PrefixAlignment laws strongSet verifier seeds.2)
          witness
      have sumCheck :=
        sumCheckFailure_iff_piCcsSumCheckCollision
          (contract.d4PrefixAlignment laws strongSet verifier seeds.2)
          witness
      have mixing' :
          MixingFailure (key.compatibleContext running fresh).piCcs
              (contract.d4Execution laws strongSet verifier seeds.2).causalRun
              witness <->
            PiCcsMixingRoot (d4PrefixKey seeds.2) running fresh
              (d4PrefixProof seeds.2) witness := by
        simpa only [d4PrefixKey_compatibleContext] using mixing
      have sumCheck' :
          SumCheckFailure (key.compatibleContext running fresh).piCcs
              (contract.d4Execution laws strongSet verifier seeds.2).causalRun
              witness <->
            PiCcsSumCheckCollision (d4PrefixKey seeds.2) running fresh
              (d4PrefixProof seeds.2) witness := by
        simpa only [d4PrefixKey_compatibleContext] using sumCheck
      have combined :
          (MixingFailure (key.compatibleContext running fresh).piCcs
                (contract.d4Execution laws strongSet verifier
                  seeds.2).causalRun witness \/
              SumCheckFailure (key.compatibleContext running fresh).piCcs
                (contract.d4Execution laws strongSet verifier
                  seeds.2).causalRun witness) <->
            (PiCcsMixingRoot (d4PrefixKey seeds.2) running fresh
                (d4PrefixProof seeds.2) witness \/
              PiCcsSumCheckCollision (d4PrefixKey seeds.2) running fresh
                (d4PrefixProof seeds.2) witness) := by
        constructor
        · rintro (root | collision)
          · exact Or.inl (mixing'.mp root)
          · exact Or.inr (sumCheck'.mp collision)
        · rintro (root | collision)
          · exact Or.inl (mixing'.mpr root)
          · exact Or.inr (sumCheck'.mpr collision)
      rw [fixedFirstBad_eq_mixing_or_sumCheck
        (key.compatibleContext running fresh).piCcs
        (contract.d4Execution laws strongSet verifier seeds.1)
        (contract.d4Execution laws strongSet verifier seeds.2)
        witness firstTarget]
      simpa [mixingRootEvent, sumCheckBadChallengeEvent,
        NifsD4FixedFirstBad, firstTarget] using combined

/-- Boolean equality form of the fixed-first event identity. -/
theorem nifsD4FixedFirstBadEvent_eq_operational
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {extensionAlphabet : Support Extension}
    (contract :
      CausalPrefixCouplingContract key running fresh extensionAlphabet)
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      (key.compatibleContext running fresh).piRlc.semantics
      (key.compatibleContext running fresh).piRlc.params
      (key.compatibleContext running fresh).piRlc.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      (key.compatibleContext running fresh).piRlc.algebra.challengeValid)
    (verifier : PiRLC.PaperWeakFiniteUniform.VerifierData
      (key.compatibleContext running fresh).piRlc)
    (seeds : D4RunSeed contract verifier × D4RunSeed contract verifier) :
    contract.nifsD4FixedFirstBadEvent laws strongSet verifier seeds =
      fixedFirstBad (key.compatibleContext running fresh).piCcs
        (contract.d4Execution laws strongSet verifier seeds.1,
          contract.d4Execution laws strongSet verifier seeds.2) := by
  apply Bool.eq_iff_iff.mpr
  rw [nifsD4FixedFirstBadEvent_eq_true]
  exact
    (contract.fixedFirstBad_iff_nifsD4FixedFirstBad laws strongSet verifier
      seeds).symm

section FiniteCoupling

variable {Extension : Type uExtension}
variable {Commitment : Type uCommitment}
variable {PublicInput : Type uPublicInput}
variable {Scalar : Type uScalar}
variable {State : Type uState}
variable [DecidableEq Extension]
variable {shape : Shape}
variable {columns blockCount degreeBound : Nat}
variable {key : Key Extension Commitment PublicInput Scalar State shape
  columns blockCount degreeBound}
variable {running : Running Extension Commitment PublicInput shape}
variable {fresh : Fresh Commitment PublicInput shape}
variable {extensionAlphabet : Support Extension}
variable
  (contract :
    CausalPrefixCouplingContract key running fresh extensionAlphabet)
variable
  (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
    (key.compatibleContext running fresh).piRlc.semantics
    (key.compatibleContext running fresh).piRlc.params
    (key.compatibleContext running fresh).piRlc.algebra)
variable
  (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
    (key.compatibleContext running fresh).piRlc.algebra.challengeValid)
variable
  (verifier : PiRLC.PaperWeakFiniteUniform.VerifierData
    (key.compatibleContext running fresh).piRlc)

private theorem mapId_perm
    {Element : Type _}
    (values : List Element) :
    (values.map id).Perm values := by
  have mapped : values.map id = values := List.map_id values
  rw [mapped]

/-- The original operational D.4 experiment specialized to the causal NIFS
contract and the paper `Pi_RLC` target extractor. -/
noncomputable def d4OperationalExperiment :
    Experiment (Execution Extension shape columns) :=
  OperationalExperiment.experiment
    (key.compatibleContext running fresh).piCcs extensionAlphabet
    (contract.d4Adversary laws strongSet verifier)

/-- The same one-run support with its seed retained as the outcome.  This is
the carrier on which the realized NIFS key and proof remain recoverable. -/
noncomputable def d4SeedExperiment :
    Experiment (D4RunSeed contract verifier) :=
  (contract.d4OperationalExperiment laws strongSet verifier).support.uniform

@[simp] theorem d4OperationalExperiment_outcome
    (seed : D4RunSeed contract verifier) :
    (contract.d4OperationalExperiment laws strongSet verifier).outcome seed =
      contract.d4Execution laws strongSet verifier seed := by
  rfl

/-- Reassociation exposes the exact independent
`(causal prefix seed) × (target extractor seed)` support. -/
theorem d4Seed_supportPermutation :
    (((contract.d4SeedExperiment laws strongSet verifier).support.values.map
      PiRlcComposition.strongToWeakSeed).Perm
        (contract.support.product
          (forkSeedSupport verifier.alphabet key.arity.total)).values) := by
  simpa [d4SeedExperiment, d4OperationalExperiment,
    CausalPrefixCouplingContract.support] using
    PiRlcComposition.strongToWeakSeed_supportPermutation
      (key.compatibleContext running fresh) laws strongSet extensionAlphabet
      verifier (causalPiRlcAdversary key running fresh contract.adversary)

/-- Ambient-success probability is unchanged when the operational execution
is replaced by its replay-aligned typed NIFS seed event. -/
theorem d4AmbientSuccess_probability_eq_nifs :
    (contract.d4OperationalExperiment laws strongSet verifier).probabilityBool
        (OperationalExperiment.success
          (key.compatibleContext running fresh).piCcs) =
      (contract.d4SeedExperiment laws strongSet verifier).probabilityBool
        (contract.nifsD4AmbientSuccessEvent laws strongSet verifier) := by
  apply Experiment.probabilityBool_eq_of_reindex
    (contract.d4OperationalExperiment laws strongSet verifier)
    (contract.d4SeedExperiment laws strongSet verifier)
    id
  · change
      ((contract.d4OperationalExperiment laws strongSet
        verifier).support.values.map id).Perm
        (contract.d4OperationalExperiment laws strongSet
          verifier).support.values
    exact mapId_perm _
  · intro seed _member
    simpa [d4OperationalExperiment, d4SeedExperiment] using
      (contract.nifsD4AmbientSuccessEvent_eq_operational laws strongSet
        verifier seed).symm

/-- Source extraction interpreted at the first realized NIFS prefix.  The
fresh second seed is present only because this is the D.4 two-run carrier. -/
noncomputable def NifsD4SourceExtracted
    (seeds : D4RunSeed contract verifier × D4RunSeed contract verifier) :
    Prop :=
  match (contract.d4Execution laws strongSet verifier seeds.1).target with
  | none => False
  | some witness =>
      SourceValid (d4PrefixKey seeds.1) running fresh witness

/-- Boolean source-extraction event for the finite seed product. -/
noncomputable def nifsD4SourceExtractedEvent :
    D4RunSeed contract verifier × D4RunSeed contract verifier -> Bool :=
  fun seeds =>
    propositionEvent
      (contract.NifsD4SourceExtracted laws strongSet verifier seeds)

@[simp] theorem nifsD4SourceExtractedEvent_eq_true
    (seeds : D4RunSeed contract verifier × D4RunSeed contract verifier) :
    contract.nifsD4SourceExtractedEvent laws strongSet verifier seeds = true <->
      contract.NifsD4SourceExtracted laws strongSet verifier seeds := by
  simp [nifsD4SourceExtractedEvent]

/-- Source membership is unchanged by prefix-oracle realization. -/
theorem d4SourceHolds_iff_nifs
    (seed : D4RunSeed contract verifier)
    (witness : OutputWitness shape columns) :
    SourceHolds
        (key.compatibleContext running fresh).piCcs.extensionOps
        (key.compatibleContext running fresh).piCcs.lift
        (key.compatibleContext running fresh).piCcs.openingMaps
        (key.compatibleContext running fresh).piCcs.params
        (key.compatibleContext running fresh).piCcs.statement witness <->
      SourceValid (d4PrefixKey seed) running fresh witness := by
  rfl

/-- Pointwise identity of the operational and typed-NIFS source-extraction
events. -/
theorem nifsD4SourceExtractedEvent_eq_operational
    (seeds : D4RunSeed contract verifier × D4RunSeed contract verifier) :
    contract.nifsD4SourceExtractedEvent laws strongSet verifier seeds =
      sourceExtracted (key.compatibleContext running fresh).piCcs
        (contract.d4Execution laws strongSet verifier seeds.1,
          contract.d4Execution laws strongSet verifier seeds.2) := by
  apply Bool.eq_iff_iff.mpr
  rw [nifsD4SourceExtractedEvent_eq_true]
  cases firstTarget :
      (contract.d4Execution laws strongSet verifier seeds.1).target with
  | none =>
      simp [NifsD4SourceExtracted, sourceExtracted, firstTarget]
  | some witness =>
      simpa [NifsD4SourceExtracted, sourceExtracted, firstTarget] using
        (contract.d4SourceHolds_iff_nifs verifier seeds.1 witness).symm

/-- Exact finite first-success/fresh-second seed carrier.  Its support is a
Cartesian product: the first coordinate is filtered by actual paper success,
and the second coordinate is a fresh copy of the complete unconditioned
support. -/
noncomputable def d4FirstSuccessFreshSecondSeeds
    (nonempty :
      (contract.d4OperationalExperiment laws strongSet verifier).support.values.filter
        (fun seed =>
          OperationalExperiment.success
            (key.compatibleContext running fresh).piCcs
            ((contract.d4OperationalExperiment laws strongSet
              verifier).outcome seed)) ≠ []) :
    Experiment
      (D4RunSeed contract verifier × D4RunSeed contract verifier) :=
  let base := contract.d4OperationalExperiment laws strongSet verifier
  let first := base.support.filterBool
    (fun seed =>
      OperationalExperiment.success
        (key.compatibleContext running fresh).piCcs (base.outcome seed))
    nonempty
  (first.product base.support).uniform

/-- Membership exposes the successful first seed and the independently fresh
second seed.  The first-success clause is stated in typed NIFS form. -/
theorem mem_d4FirstSuccessFreshSecondSeeds_iff
    (nonempty :
      (contract.d4OperationalExperiment laws strongSet verifier).support.values.filter
        (fun seed =>
          OperationalExperiment.success
            (key.compatibleContext running fresh).piCcs
            ((contract.d4OperationalExperiment laws strongSet
              verifier).outcome seed)) ≠ [])
    (seeds : D4RunSeed contract verifier × D4RunSeed contract verifier) :
    seeds ∈
        (contract.d4FirstSuccessFreshSecondSeeds laws strongSet verifier
          nonempty).support.values <->
      seeds.1 ∈
          (contract.d4OperationalExperiment laws strongSet verifier).support.values /\
        contract.NifsD4AmbientSuccess laws strongSet verifier seeds.1 /\
        seeds.2 ∈
          (contract.d4OperationalExperiment laws strongSet verifier).support.values := by
  change seeds ∈
      (((contract.d4OperationalExperiment laws strongSet
          verifier).support.filterBool
          (fun seed =>
            OperationalExperiment.success
              (key.compatibleContext running fresh).piCcs
              ((contract.d4OperationalExperiment laws strongSet
                verifier).outcome seed))
          nonempty).product
        (contract.d4OperationalExperiment laws strongSet verifier).support
      ).values <->
    _
  let base := contract.d4OperationalExperiment laws strongSet verifier
  let first := base.support.filterBool
    (fun seed =>
      OperationalExperiment.success
        (key.compatibleContext running fresh).piCcs (base.outcome seed))
    nonempty
  constructor
  · intro member
    have split := (Support.mem_product_iff first base.support seeds).mp member
    have firstFiltered :
        seeds.1 ∈ base.support.values.filter
          (fun seed =>
            OperationalExperiment.success
              (key.compatibleContext running fresh).piCcs
              (base.outcome seed)) := by
      exact split.1
    have firstFacts := List.mem_filter.mp firstFiltered
    exact ⟨firstFacts.1,
      (contract.d4Success_iff_nifsD4AmbientSuccess laws strongSet verifier
        seeds.1).mp firstFacts.2,
      split.2⟩
  · rintro ⟨firstMember, firstSuccess, secondMember⟩
    apply (Support.mem_product_iff first base.support seeds).mpr
    refine ⟨?_, secondMember⟩
    exact List.mem_filter.mpr ⟨firstMember,
      (contract.d4Success_iff_nifsD4AmbientSuccess laws strongSet verifier
        seeds.1).mpr firstSuccess⟩

/-- Generic event transport from the paper first-conditioned mixture to the
exact seed-retaining NIFS product. -/
theorem d4FirstSuccessFreshSecond_probability_eq
    (nonempty :
      (contract.d4OperationalExperiment laws strongSet verifier).support.values.filter
        (fun seed =>
          OperationalExperiment.success
            (key.compatibleContext running fresh).piCcs
            ((contract.d4OperationalExperiment laws strongSet
              verifier).outcome seed)) ≠ [])
    (operationalEvent :
      Execution Extension shape columns × Execution Extension shape columns ->
        Bool)
    (nifsEvent :
      D4RunSeed contract verifier × D4RunSeed contract verifier -> Bool)
    (agreement : forall seeds,
      nifsEvent seeds =
        operationalEvent
          (contract.d4Execution laws strongSet verifier seeds.1,
            contract.d4Execution laws strongSet verifier seeds.2)) :
    ((contract.d4OperationalExperiment laws strongSet verifier
        ).firstConditionedFreshSecond
          (OperationalExperiment.success
            (key.compatibleContext running fresh).piCcs)
          nonempty).probabilityBool operationalEvent =
      (contract.d4FirstSuccessFreshSecondSeeds laws strongSet verifier
        nonempty).probabilityBool nifsEvent := by
  let base := contract.d4OperationalExperiment laws strongSet verifier
  let first := base.support.filterBool
    (fun seed =>
      OperationalExperiment.success
        (key.compatibleContext running fresh).piCcs (base.outcome seed))
    nonempty
  let productExecutions : Experiment
      (Execution Extension shape columns × Execution Extension shape columns) :=
    {
      Seed := base.Seed × base.Seed
      support := first.product base.support
      outcome := fun seeds => (base.outcome seeds.1, base.outcome seeds.2)
    }
  have flatten :
      (base.firstConditionedFreshSecond
          (OperationalExperiment.success
            (key.compatibleContext running fresh).piCcs)
          nonempty).probabilityBool operationalEvent =
        productExecutions.probabilityBool operationalEvent := by
    simpa [base, first, productExecutions,
      Experiment.firstConditionedFreshSecond] using
      Mixture.sharedSupport_probabilityBool_eq_product first base.support
        (fun firstSeed secondSeed =>
          (base.outcome firstSeed, base.outcome secondSeed))
        operationalEvent
  rw [flatten]
  apply Experiment.probabilityBool_eq_of_reindex
    productExecutions
    (contract.d4FirstSuccessFreshSecondSeeds laws strongSet verifier nonempty)
    id
  · change
      ((first.product base.support).values.map id).Perm
        (first.product base.support).values
    exact mapId_perm _
  · intro seeds _member
    simpa [productExecutions, d4OperationalExperiment,
      d4FirstSuccessFreshSecondSeeds, base, first] using
      (agreement seeds).symm

/-- The exact fixed-first bad-event probability is preserved by the causal
NIFS two-run carrier. -/
theorem d4FixedFirstBad_probability_eq_nifs
    (nonempty :
      (contract.d4OperationalExperiment laws strongSet verifier).support.values.filter
        (fun seed =>
          OperationalExperiment.success
            (key.compatibleContext running fresh).piCcs
            ((contract.d4OperationalExperiment laws strongSet
              verifier).outcome seed)) ≠ []) :
    ((contract.d4OperationalExperiment laws strongSet verifier
        ).firstConditionedFreshSecond
          (OperationalExperiment.success
            (key.compatibleContext running fresh).piCcs)
          nonempty).probabilityBool
            (fixedFirstBad (key.compatibleContext running fresh).piCcs) =
      (contract.d4FirstSuccessFreshSecondSeeds laws strongSet verifier
        nonempty).probabilityBool
          (contract.nifsD4FixedFirstBadEvent laws strongSet verifier) := by
  exact contract.d4FirstSuccessFreshSecond_probability_eq laws strongSet
    verifier nonempty
    (fixedFirstBad (key.compatibleContext running fresh).piCcs)
    (contract.nifsD4FixedFirstBadEvent laws strongSet verifier)
    (contract.nifsD4FixedFirstBadEvent_eq_operational laws strongSet verifier)

/-- The exact source-extraction probability is preserved by the same
first-success/fresh-second carrier. -/
theorem d4SourceExtracted_probability_eq_nifs
    (nonempty :
      (contract.d4OperationalExperiment laws strongSet verifier).support.values.filter
        (fun seed =>
          OperationalExperiment.success
            (key.compatibleContext running fresh).piCcs
            ((contract.d4OperationalExperiment laws strongSet
              verifier).outcome seed)) ≠ []) :
    ((contract.d4OperationalExperiment laws strongSet verifier
        ).firstConditionedFreshSecond
          (OperationalExperiment.success
            (key.compatibleContext running fresh).piCcs)
          nonempty).probabilityBool
            (sourceExtracted (key.compatibleContext running fresh).piCcs) =
      (contract.d4FirstSuccessFreshSecondSeeds laws strongSet verifier
        nonempty).probabilityBool
          (contract.nifsD4SourceExtractedEvent laws strongSet verifier) := by
  exact contract.d4FirstSuccessFreshSecond_probability_eq laws strongSet
    verifier nonempty
    (sourceExtracted (key.compatibleContext running fresh).piCcs)
    (contract.nifsD4SourceExtractedEvent laws strongSet verifier)
    (contract.nifsD4SourceExtractedEvent_eq_operational laws strongSet
      verifier)

/-- Appendix-D.4's finite first-success extraction inequality transported to
the exact typed NIFS seed carrier.  The two probability premises are the
existing paper operational contracts; this theorem changes neither their
event nor their budget.  Its conclusion contains only the replay-aligned NIFS
success and source-extraction events.

This does not assert that NIFS public acceptance supplies the target witness
used by `d4Adversary`; that separate D.6 premise remains necessary. -/
theorem d4Extraction_after_first_success_nifs
    (successFloor rawMismatchBudget badBudget : Rat)
    (floorPos : 0 < successFloor)
    (floorBound :
      successFloor <=
        (contract.d4SeedExperiment laws strongSet verifier).probabilityBool
          (contract.nifsD4AmbientSuccessEvent laws strongSet verifier))
    (rawMismatchBound :
      (contract.d4OperationalExperiment laws strongSet verifier
        ).iidPair.probabilityBool
          (witnessDisagreement
            (key.compatibleContext running fresh).piCcs) <=
        rawMismatchBudget)
    (fixedBadBound :
      OperationalExperiment.FixedFirstBadBound
        (key.compatibleContext running fresh).piCcs extensionAlphabet
        (contract.d4Adversary laws strongSet verifier) badBudget) :
    exists nonempty :
        (contract.d4OperationalExperiment laws strongSet verifier
          ).support.values.filter
            (fun seed =>
              OperationalExperiment.success
                (key.compatibleContext running fresh).piCcs
                ((contract.d4OperationalExperiment laws strongSet
                  verifier).outcome seed)) ≠ [],
      (contract.d4SeedExperiment laws strongSet verifier).probabilityBool
            (contract.nifsD4AmbientSuccessEvent laws strongSet verifier) -
          (badBudget + rawMismatchBudget / successFloor) <=
        (contract.d4FirstSuccessFreshSecondSeeds laws strongSet verifier
          nonempty).probabilityBool
            (contract.nifsD4SourceExtractedEvent laws strongSet verifier) := by
  have successProbability :=
    contract.d4AmbientSuccess_probability_eq_nifs laws strongSet verifier
  have operationalFloor :
      successFloor <=
        (contract.d4OperationalExperiment laws strongSet verifier
          ).probabilityBool
            (OperationalExperiment.success
              (key.compatibleContext running fresh).piCcs) := by
    rw [successProbability]
    exact floorBound
  let nonempty :=
    OperationalExperiment.successfulSupport_nonempty_of_floor
      (key.compatibleContext running fresh).piCcs extensionAlphabet
      (contract.d4Adversary laws strongSet verifier) successFloor
      floorPos operationalFloor
  refine ⟨nonempty, ?_⟩
  have extracted :=
    OperationalExperiment.extraction_after_first_success
      (key.compatibleContext running fresh).piCcs extensionAlphabet
      (contract.d4Adversary laws strongSet verifier)
      successFloor rawMismatchBudget badBudget floorPos operationalFloor
      rawMismatchBound fixedBadBound
  dsimp only at extracted
  change
    (contract.d4OperationalExperiment laws strongSet verifier).probabilityBool
          (OperationalExperiment.success
            (key.compatibleContext running fresh).piCcs) -
        (badBudget + rawMismatchBudget / successFloor) <=
      ((contract.d4OperationalExperiment laws strongSet verifier
          ).firstConditionedFreshSecond
            (OperationalExperiment.success
              (key.compatibleContext running fresh).piCcs)
            nonempty).probabilityBool
        (sourceExtracted (key.compatibleContext running fresh).piCcs)
    at extracted
  rw [successProbability] at extracted
  rw [contract.d4SourceExtracted_probability_eq_nifs laws strongSet verifier
    nonempty] at extracted
  exact extracted

end FiniteCoupling

end CausalPrefixCouplingContract

end Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
