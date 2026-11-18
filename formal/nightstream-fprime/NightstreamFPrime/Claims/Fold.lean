import NightstreamFPrime.Export.BoundedDepth.FoldSponge
import NightstreamFPrime.Export.BoundedDepth.RandomOracleSetup
import NightstreamFPrime.Lifecycle.BoundedDepth.FoldCompleteness
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.Assumption1
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.FoldGate
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.LightKnowledge
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RefreshKnowledge
import NightstreamFPrime.Spec.Folding.Light.Completeness
import NightstreamFPrime.Spec.Folding.Light.Security
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Law
import NightstreamFPrime.Spec.Folding.Refresh.Completeness
import NightstreamFPrime.Spec.Folding.Refresh.Security
import NightstreamFPrime.Spec.Folding.Schedule

/-! # Claims: the light fold and the refresh fold

Completeness and knowledge of the two folds of the bounded-depth scheme.

- Paper: SuperNeo v1.2 Theorems 12 and 13, Lemmas 1, 4, 10 and 11, and Cyclo Theorem 3.
- `light_extracts` and `refresh_extracts` are deterministic: the fork of the challenges is an
  argument, and each gives the witnesses or a Π_CCS bad event.
- `light_knowledge`, `refresh_knowledge` and `assumption1_in_random_oracle_model` hold in the
  ideal-permutation or random-oracle model (`[P2-PERM]`, `[SPONGE]`).
- `light_extracts` and `refresh_extracts` also take `strongSet`, the units of the strong
  sampling set. It returns data, so the premise list does not show it.
  `Claims.Foundations.low_norm_invertible` is the result behind it.
- `assumption1_in_random_oracle_model` does not give the premise of the bounds of `Claims.Ivc`.
  Its adversary queries a uniform oracle. The IVC adversary has no oracle, because the step
  circuits evaluate the fixed Poseidon2 permutation (`SECURITY_MODEL.md` trust item 3). -/

namespace NightstreamFPrime.Claims.Fold

open Spec Spec.Folding Lifecycle Lifecycle.BoundedDepth Lifecycle.BoundedDepth.RandomOracle Export.BoundedDepth
attribute [local instance low] Classical.propDecidable

/-- Cyclo Theorem 3, item 1, for one fresh source and one running source. -/
theorem light_complete.{uCommitment, uPublicInput, uScalar, uState}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : PiCCS.PaperJoint.Shape}
    {columns : ℕ}
    {blockCount : ℕ}
    {width : ℕ}
    (key : Light.Key Commitment PublicInput Scalar State shape columns blockCount width)
    (running : Nifs.PaperNonInteractive.Running K Commitment PublicInput shape)
    (fresh : Nifs.PaperNonInteractive.Fresh Commitment PublicInput shape)
    (witness : PiCCS.PaperJoint.StrongReduction.OutputWitness shape columns)
    (bound : ℕ)
    (freshOpening : Opening.Holds (Structure :=
        PiCCS.PaperJoint.StrongReduction.RelationSource shape columns blockCount)
        (PiCCS.PaperJoint.StrongReduction.paperRelationSemantics key.baseOps key.extensionOps
          key.lift key.openingMaps)
        key.params.b ((key.statement running fresh).commitments key.freshSource)
        ((key.statement running fresh).publicInputs key.freshSource)
        (witness.assignments key.freshSource))
    (runningOpening : Opening.Holds (Structure :=
        PiCCS.PaperJoint.StrongReduction.RelationSource shape columns blockCount)
        (PiCCS.PaperJoint.StrongReduction.paperRelationSemantics key.baseOps key.extensionOps
          key.lift key.openingMaps)
        bound ((key.statement running fresh).commitments key.runningSource)
        ((key.statement running fresh).publicInputs key.runningSource)
        (witness.assignments key.runningSource))
    (truth : Light.PiCCS.SemanticTruth
        (PiCCS.PaperJoint.MatrixCoefficientSource.ConnectedInputs.toUnifiedInputs key.baseOps
          ((key.statement running fresh).sourceConnectedInputs witness))
        key.baseOps key.extensionOps key.lift) :
    ∃ (rounds : Fin shape.cubeVariables → SumCheck.Finite.FixedPolynomial K width),
        have proof :=
          { piCcsRounds := rounds,
            piCcsOutput :=
              PiCCS.PaperJoint.FullOutputCoordinates.FullOutput.honestAt key.baseOps
                key.extensionOps key.lift
                ((key.statement running fresh).sourceConnectedInputs witness)
                (key.coins running fresh rounds).roundPoint :
            Light.Proof shape width };
        ∀ (challenge : Scalar),
          key.challenge running fresh proof = some challenge →
            ∃ (result :
              CE.Instance
                (PiCCS.PaperJoint.StrongReduction.RelationSource shape columns blockCount)
                PublicInput (PiCCS.PaperJoint.CubePoint K shape.cubeVariables)
                (PiCCS.PaperJoint.StrongReduction.EvaluationFamily K shape) Commitment),
              Light.verify key running fresh proof = some result ∧
                Light.Rlc.Holds key.piRlcSemantics
                  (bound + key.params.expansionT * (key.params.b - 1)) result
                  (Light.Rlc.witness key.piRlcAlgebra key.one challenge
                    (witness.assignments key.freshSource) (witness.assignments key.runningSource)) :=
  @Spec.Folding.Light.complete Commitment PublicInput Scalar State shape columns blockCount width
    key running fresh witness bound freshOpening runningOpening truth

/-- SuperNeo v1.2 Theorem 12 for the light fold, with Cyclo Theorem 3, item 2. -/
theorem light_extracts.{uCommitment, uPublicInput, uScalar, uState}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : PiCCS.PaperJoint.Shape}
    {columns : ℕ}
    {blockCount : ℕ}
    {width : ℕ}
    (key : Light.Key Commitment PublicInput Scalar State shape columns blockCount width)
    (running : Nifs.PaperNonInteractive.Running K Commitment PublicInput shape)
    (fresh : Nifs.PaperNonInteractive.Fresh Commitment PublicInput shape)
    (proof : Light.Proof shape width)
    (result : CE.Instance (PiCCS.PaperJoint.StrongReduction.RelationSource shape columns blockCount)
        PublicInput (PiCCS.PaperJoint.CubePoint K shape.cubeVariables)
        (PiCCS.PaperJoint.StrongReduction.EvaluationFamily K shape) Commitment)
    (accepted : Light.verify key running fresh proof = some result)
    (freshBound : key.params.b = 2)
    (challengeSetSize : ℕ)
    (bound : ℕ)
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)
    (oneEq : key.one = laws.ring.one)
    (normLaw : Light.Rlc.ExtractionNormLaw laws)
    (fork : key.Fork running fresh proof bound) :
    have witness := key.extractedWitness running fresh proof bound laws strongSet fork;
      Light.Knowledge key running fresh bound witness ∨
        PiCCS.PaperJoint.SignedCoefficientObject.MixingRoot key.extensionOps
            (PiCCS.PaperJoint.ProtocolPolynomial.Data.toJointData key.extensionOps
              (Light.PiCCS.protocolData key.baseOps key.lift
                (PiCCS.PaperJoint.MatrixCoefficientSource.ConnectedInputs.toUnifiedInputs
                  key.baseOps ((key.statement running fresh).sourceConnectedInputs witness))))
            (key.probe running fresh proof).coins.alpha
            (key.probe running fresh proof).coins.gamma ∨
          Light.PiCCS.SumCheckFailure key.extensionOps key.lift (key.statement running fresh)
            width challengeSetSize (key.probe running fresh proof) witness :=
  @Spec.Folding.Light.accepted_implies_securityOutcome Commitment PublicInput Scalar State shape
    columns blockCount width key running fresh proof result accepted freshBound challengeSetSize
    bound laws strongSet oneEq normLaw fork

/-- SuperNeo v1.2 Theorem 13 and Lemmas 10 and 11 for the refresh fold. -/
theorem refresh_complete.{uCommitment, uPublicInput, uScalar, uState}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : PiCCS.PaperJoint.Shape}
    {columns : ℕ}
    {blockCount : ℕ}
    {degreeBound : ℕ}
    (key : Nifs.PaperNonInteractive.Key K Commitment PublicInput Scalar State shape columns blockCount
        degreeBound)
    (accumulator : Refresh.Accumulator Commitment PublicInput shape columns blockCount)
    (fresh : Nifs.PaperNonInteractive.Fresh Commitment PublicInput shape)
    (assignment : PiCCS.PaperJoint.PaperLinearAlgebra.Assignment F columns)
    (freshWitness : Fin shape.freshCount → PiCCS.PaperJoint.PaperLinearAlgebra.Assignment F columns)
    (system : accumulator.constraintSystem = key.relationSource)
    (combined : accumulator.stage = NormStage.combined)
    (accumulatorHolds : CE.Holds key.piRlcSemantics key.params accumulator assignment)
    (freshHolds : ∀ (source : Fin shape.freshCount),
        CCS.Holds
          (PiCCS.PaperJoint.StrongReduction.paperRelationSemantics key.baseOps key.extensionOps
            key.lift key.openingMaps)
          key.params
          (PiCCS.PaperJoint.SourceMembership.freshInstance
            (key.statement
              (Refresh.children key accumulator (Refresh.honestSplit key accumulator assignment))
              fresh)
            source)
          (freshWitness source)) :
    have running :=
        Refresh.children key accumulator (Refresh.honestSplit key accumulator assignment);
      ∃ (messages : Fin shape.cubeVariables → SumCheck.Finite.FixedPolynomial K degreeBound)
        (fullOutput : PiCCS.PaperJoint.FullOutputCoordinates.FullOutput K shape),
        have coins :=
          PiCCS.PaperJoint.FiatShamir.derive key.oracle.transcript
            { priorState := key.publicInputState running fresh,
              input :=
                PiCCS.PaperJoint.StrongReduction.Statement.verifierInput key.lift
                  (key.statement running fresh) :
              PiCCS.TranscriptReplay.Statement K State shape }
            { rounds := fun (round : Fin shape.cubeVariables) => (messages round).toMessage :
              PiCCS.PaperJoint.FiatShamir.Certificate K shape };
        ∀ (rho : Fin key.arity.total → Scalar),
          key.piRlcResponse (key.absorbPiCcsOutput coins.finalState fullOutput) = some rho →
            ∃ (proof : Refresh.Proof Commitment shape degreeBound) (parent :
              Refresh.Accumulator Commitment PublicInput shape columns blockCount) (parentWitness
              : PiCCS.PaperJoint.PaperLinearAlgebra.Assignment F columns),
              proof.split = Refresh.honestSplit key accumulator assignment ∧
                proof.nifs.piCcsRounds = messages ∧
                  Refresh.verify key accumulator fresh proof = some parent ∧
                    CE.Holds key.piRlcSemantics key.params parent parentWitness :=
  @Spec.Folding.Refresh.complete Commitment PublicInput Scalar State shape columns blockCount
    degreeBound key accumulator fresh assignment freshWitness system combined accumulatorHolds
    freshHolds

/-- SuperNeo v1.2 Lemma 1, Theorem 13 and Lemma 4 for the refresh fold. A fork at the output
stage bound gives the fork here through `NormFreeFork.forgetNorms`. -/
theorem refresh_extracts.{uCommitment, uPublicInput, uScalar, uState}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : PiCCS.PaperJoint.Shape}
    {columns : ℕ}
    {blockCount : ℕ}
    {degreeBound : ℕ}
    (key : Nifs.PaperNonInteractive.Key K Commitment PublicInput Scalar State shape columns blockCount
        degreeBound)
    (accumulator : Refresh.Accumulator Commitment PublicInput shape columns blockCount)
    (fresh : Nifs.PaperNonInteractive.Fresh Commitment PublicInput shape)
    (proof : Refresh.Proof Commitment shape degreeBound)
    (parent : Refresh.Accumulator Commitment PublicInput shape columns blockCount)
    (accepted : Refresh.verify key accumulator fresh proof = some parent)
    (system : accumulator.constraintSystem = key.relationSource)
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)
    (fork : PiRLC.PaperForkExtraction.CompleteFork (PiRLC.NormFreeFork.normFree key.piRlcSemantics)
        key.params (PiRLC.NormFreeFork.normFreeAlgebra key.piRlcAlgebra)
        (Nifs.PaperSecurityComposition.piRlcBatch key
          (Refresh.children key accumulator proof.split) fresh proof.nifs)) :
    let running := Refresh.children key accumulator proof.split;
      have witness := Refresh.forkWitness key running fresh proof.nifs laws strongSet fork;
      (∀ (source : Fin shape.freshCount),
            CCS.Holds
              (PiCCS.PaperJoint.StrongReduction.paperRelationSemantics key.baseOps
                key.extensionOps key.lift key.openingMaps)
              key.params
              (PiCCS.PaperJoint.SourceMembership.freshInstance (key.statement running fresh)
                source)
              (witness.assignments (PiCCS.PaperJoint.UnifiedSources.freshSourceIndex source))) ∧
          CE.Holds key.piRlcSemantics key.params accumulator
            (key.piDecAlgebra.recomposeAssignment fun (child : Fin key.params.k) =>
              witness.assignments
                (PiCCS.PaperJoint.UnifiedSources.runningSourceIndex
                  (Fin.cast key.outputCount_eq child))) ∨
        PiCCS.PaperJoint.SignedCoefficientObject.MixingRoot key.extensionOps
            (PiCCS.PaperJoint.ProtocolPolynomial.Data.toJointData key.extensionOps
              (PiCCS.PaperJoint.StrongReduction.Statement.sourceProtocolData key.lift
                (key.statement running fresh) witness))
            (key.piCcsProbe running fresh proof.nifs).coins.alpha
            (key.piCcsProbe running fresh proof.nifs).coins.gamma ∨
          PiCCS.PaperJoint.StrongReduction.FixedWidthSumCheckFailure key.extensionOps key.lift
            (key.statement running fresh) degreeBound key.challengeSetSize
            (key.piCcsProbe running fresh proof.nifs) witness :=
  @Spec.Folding.Refresh.accepted_implies_securityOutcome Commitment PublicInput Scalar State shape
    columns blockCount degreeBound key accumulator fresh proof parent accepted system laws
    strongSet fork

/-- The honest light fold of a slot claim is accepted at the production keys, and its output is a
slot claim that holds at the input bound plus `T(b − 1)`. Its witness is zero on each ring block
on which both source witnesses are zero. -/
theorem light_production_complete
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (key : CircuitKey logicalWidth publicFits)
    (start : SlotClaim logicalWidth publicFits)
    (fresh : StepFresh logicalWidth publicFits)
    (witness : PiCCS.PaperJoint.StrongReduction.OutputWitness LightShape
        (PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth))
    (bound : ℕ)
    (freshOpening : Opening.Holds (Structure :=
        PiCCS.PaperJoint.StrongReduction.RelationSource LightShape
          (PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth)
          (PiCCS.PaperJoint.ConcreteRingColumnLayout.blockCount
            (PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth)))
        (PiCCS.PaperJoint.StrongReduction.paperRelationSemantics
          (keys key.relation ajtai).light.baseOps (keys key.relation ajtai).light.extensionOps
          (keys key.relation ajtai).light.lift (keys key.relation ajtai).light.openingMaps)
        (keys key.relation ajtai).light.params.b
        (((keys key.relation ajtai).light.statement (LightFork.runningOf start)
              (HyperNova.Construction2.BoundedDepthFolds.freshToLight fresh)).commitments
          (keys key.relation ajtai).light.freshSource)
        (((keys key.relation ajtai).light.statement (LightFork.runningOf start)
              (HyperNova.Construction2.BoundedDepthFolds.freshToLight fresh)).publicInputs
          (keys key.relation ajtai).light.freshSource)
        (witness.assignments (keys key.relation ajtai).light.freshSource))
    (runningOpening : Opening.Holds (Structure :=
        PiCCS.PaperJoint.StrongReduction.RelationSource LightShape
          (PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth)
          (PiCCS.PaperJoint.ConcreteRingColumnLayout.blockCount
            (PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth)))
        (PiCCS.PaperJoint.StrongReduction.paperRelationSemantics
          (keys key.relation ajtai).light.baseOps (keys key.relation ajtai).light.extensionOps
          (keys key.relation ajtai).light.lift (keys key.relation ajtai).light.openingMaps)
        bound
        (((keys key.relation ajtai).light.statement (LightFork.runningOf start)
              (HyperNova.Construction2.BoundedDepthFolds.freshToLight fresh)).commitments
          (keys key.relation ajtai).light.runningSource)
        (((keys key.relation ajtai).light.statement (LightFork.runningOf start)
              (HyperNova.Construction2.BoundedDepthFolds.freshToLight fresh)).publicInputs
          (keys key.relation ajtai).light.runningSource)
        (witness.assignments (keys key.relation ajtai).light.runningSource))
    (truth : LightUniqueness.Truth ajtai key.relation (LightFork.runningOf start)
        (HyperNova.Construction2.BoundedDepthFolds.freshToLight fresh) witness) :
    ∃ (proof : Light.Proof LightShape 8) (folded : SlotClaim logicalWidth publicFits)
        (foldedWitness :
        PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)),
        HyperNova.Construction2.BoundedDepthFolds.folds Schedule.Step.light
              (keys key.relation ajtai) (toClaim (CircuitKey.system ajtai key) start) fresh
              (HyperNova.Construction2.BoundedDepthFolds.FoldProof.light proof) =
            some (toClaim (CircuitKey.system ajtai key) folded) ∧
          Light.Rlc.Holds (PaperAlgebra.semantics ajtai)
              (bound +
                (keys key.relation ajtai).light.params.expansionT *
                  ((keys key.relation ajtai).light.params.b - 1))
              (toClaim (CircuitKey.system ajtai key) folded) foldedWitness ∧
            ∀ (width : ℕ),
              BlockSupport.Supported width
                  (witness.assignments (keys key.relation ajtai).light.freshSource) →
                BlockSupport.Supported width
                    (witness.assignments (keys key.relation ajtai).light.runningSource) →
                  BlockSupport.Supported width foldedWitness :=
  @Lifecycle.BoundedDepth.FoldCompleteness.light_folds logicalWidth publicFits ajtai key start
    fresh witness bound freshOpening runningOpening truth

/-- The honest refresh fold of a slot claim is accepted at the production keys, and its output is
a slot claim that holds below `B`. -/
theorem refresh_production_complete
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (key : CircuitKey logicalWidth publicFits)
    (start : SlotClaim logicalWidth publicFits)
    (fresh : StepFresh logicalWidth publicFits)
    (assignment : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
    (freshWitness : Fin productionShape.freshCount →
        PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
    (accumulatorHolds : CE.Holds (refreshKey key.relation ajtai).piRlcSemantics
        (refreshKey key.relation ajtai).params (toClaim (CircuitKey.system ajtai key) start)
        assignment)
    (freshHolds : ∀ (source : Fin productionShape.freshCount),
        CCS.Holds
          (PiCCS.PaperJoint.StrongReduction.paperRelationSemantics
            (refreshKey key.relation ajtai).baseOps (refreshKey key.relation ajtai).extensionOps
            (refreshKey key.relation ajtai).lift (refreshKey key.relation ajtai).openingMaps)
          (refreshKey key.relation ajtai).params
          (PiCCS.PaperJoint.SourceMembership.freshInstance
            (Nifs.PaperNonInteractive.Key.statement (refreshKey key.relation ajtai)
              (Refresh.children (refreshKey key.relation ajtai)
                (toClaim (CircuitKey.system ajtai key) start)
                (Refresh.honestSplit (refreshKey key.relation ajtai)
                  (toClaim (CircuitKey.system ajtai key) start) assignment))
              fresh)
            source)
          (freshWitness source)) :
    ∃ (proof : RefreshFork.RefreshProof) (folded : SlotClaim logicalWidth publicFits)
        (foldedWitness :
        PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)),
        HyperNova.Construction2.BoundedDepthFolds.folds Schedule.Step.refresh
              (keys key.relation ajtai) (toClaim (CircuitKey.system ajtai key) start) fresh
              (HyperNova.Construction2.BoundedDepthFolds.FoldProof.refresh proof) =
            some (toClaim (CircuitKey.system ajtai key) folded) ∧
          Light.Rlc.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams.bigB
            (toClaim (CircuitKey.system ajtai key) folded) foldedWitness :=
  @Lifecycle.BoundedDepth.FoldCompleteness.refresh_folds logicalWidth publicFits ajtai key start
    fresh assignment freshWitness accumulatorHolds freshHolds

/-- Along an admissible run, the bound `accumulatorBound` of each reached depth stays below `B`.
This is arithmetic on the bound, not a statement about an accumulator's norm. -/
theorem honest_bounds
    (p : GlobalParams)
    {cap : ℕ}
    (capFits : 1 + cap ≤ p.maxFresh)
    (steps : List Schedule.Step)
    (depth : ℕ)
    (start : depth ≤ cap)
    (admissible : Schedule.Admissible cap depth steps)
    (reached : ℕ)
    (member : reached ∈ Schedule.depths depth steps) :
    p.accumulatorBound reached < p.bigB :=
  @Spec.Folding.Schedule.honest_bounds p cap capFits steps depth start admissible reached member

/-- The number `distance` is below `2^-173`. It bounds the difference between an expectation
under the Π_RLC sampler and under its ideal law
(`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.expect_difference_abs_le`). -/
theorem sampler_distance :
    Nifs.NonInteractive.PiRlcSampler.distance < 1 / 2 ^ 173 :=
  @Spec.Folding.Nifs.NonInteractive.PiRlcSampler.distance_lt

/-- The statistical term of the light fold with `Q` queries is at most `(Q + 1)/2^bits`. -/
theorem light_error_bits
    (queries : ℕ) :
    LightKnowledge.statisticalError queries ≤
        ((↑queries : ℝ) + 1) / 2 ^ FoldGate.bits FoldGate.production :=
  @Lifecycle.BoundedDepth.RandomOracle.FoldGate.light_statisticalError_le_bits queries

/-- The statistical term of the refresh fold with `Q` queries is at most `(Q + 1)/2^bits`. -/
theorem refresh_error_bits
    (queries : ℕ) :
    RefreshKnowledge.statisticalError queries ≤
        ((↑queries : ℝ) + 1) / 2 ^ FoldGate.bits FoldGate.production :=
  @Lifecycle.BoundedDepth.RandomOracle.FoldGate.refresh_statisticalError_le_bits queries

/-- Adversary 3 for the light fold in the ideal-permutation model: under `Spec.Sponge.Transfer`
with the light conditions, the wins of each `world` are at most the statistical error plus
`spongeError calls`. -/
theorem light_knowledge
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (key : CircuitKey logicalWidth publicFits)
    {queries : ℕ}
    {World : Type}
    (winsOf : World → ℝ)
    (calls : ℕ)
    (premise : Sponge.Transfer (Model := FoldSponge.LightAdversary logicalWidth publicFits queries)
        (FoldSponge.Conditions (LightSchedule.schedule logicalWidth publicFits) (width := 8)
          FoldSponge.lightUse)
        winsOf (FoldSponge.lightWins ajtai key) calls)
    (world : World) :
    winsOf world ≤ LightKnowledge.statisticalError queries + Sponge.spongeError calls :=
  @Export.BoundedDepth.FoldSponge.light_knowledge_le logicalWidth publicFits ajtai key queries
    World winsOf calls premise world

/-- Adversary 3 for the refresh fold in the ideal-permutation model, as `light_knowledge`. -/
theorem refresh_knowledge
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (key : CircuitKey logicalWidth publicFits)
    {queries : ℕ}
    {World : Type}
    (winsOf : World → ℝ)
    (calls : ℕ)
    (premise : Sponge.Transfer (Model := FoldSponge.RefreshAdversary logicalWidth publicFits queries)
        (FoldSponge.Conditions (RefreshSchedule.schedule logicalWidth publicFits) (width := 8)
          FoldSponge.refreshUse)
        winsOf (FoldSponge.refreshWins ajtai key) calls)
    (world : World) :
    winsOf world ≤ RefreshKnowledge.statisticalError queries + Sponge.spongeError calls :=
  @Export.BoundedDepth.FoldSponge.refresh_knowledge_le logicalWidth publicFits ajtai key queries
    World winsOf calls premise world

/-- HyperNova errata Assumption 1 for both bounded-depth folds in the random-oracle model, with the
budgeted extractor and the error `romError`. -/
theorem assumption1_in_random_oracle_model
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (queries : ℕ)
    (budget : ℕ)
    (positive : 0 < budget) :
    HyperNova.ReverseStages.Assumption1 (RomModel.Succeeds ajtai) (RomModel.Extracted ajtai)
        (Assumption1.RomAdmitted circuitKeys queries)
        (Assumption1.RomEfficient ajtai circuitKeys queries budget)
        (Assumption1.romError ajtai circuitKeys queries budget) :=
  @Lifecycle.BoundedDepth.RandomOracle.Assumption1.romAssumption1 logicalWidth publicFits ajtai
    circuitKeys queries budget positive

/-- The light fold's knowledge bound in the random-oracle model. -/
theorem light_knowledge_in_random_oracle_model
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (key : CircuitKey logicalWidth publicFits)
    (adversary : RandomOracle.OracleComp
        (RandomOracle.Test.Point (LightSchedule.schedule logicalWidth publicFits) 8)
        RandomOracle.Test.Answer (ForkLink.Attempt logicalWidth publicFits))
    {queries : ℕ}
    (bounded : adversary.QueryBound queries) :
    (Finset.expect Finset.univ
          fun
            (oracle :
              RandomOracle.Test.Point (LightSchedule.schedule logicalWidth publicFits) 8 →
                RandomOracle.Test.Answer) =>
          if RandomOracle.Fork.Succeeds (LightFork.game ajtai key) adversary oracle then 1
          else 0) ≤
        (Finset.expect Finset.univ
            fun
              (oracle :
                RandomOracle.Test.Point (LightSchedule.schedule logicalWidth publicFits) 8 →
                  RandomOracle.Test.Answer) =>
            ∑ retries :
              Fin (LightSchedule.schedule logicalWidth publicFits).rhoCount →
                RandomOracle.Test.Answer,
              RandomOracle.Uniqueness.weight (LightFork.game ajtai key) adversary oracle retries *
                if LightKnowledge.Extracts ajtai key adversary oracle retries then 1 else 0) +
          LightKnowledge.knowledgeError ajtai key adversary queries :=
  @Lifecycle.BoundedDepth.RandomOracle.LightKnowledge.knowledge_error_le logicalWidth publicFits
    ajtai key adversary queries bounded

/-- The refresh fold's knowledge bound in the random-oracle model. -/
theorem refresh_knowledge_in_random_oracle_model
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
        PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (key : CircuitKey logicalWidth publicFits)
    (adversary : RandomOracle.OracleComp
        (RandomOracle.Test.Point (RefreshSchedule.schedule logicalWidth publicFits) 8)
        RandomOracle.Test.Answer (ForkLink.Attempt logicalWidth publicFits))
    {queries : ℕ}
    (bounded : adversary.QueryBound queries) :
    (Finset.expect Finset.univ
          fun
            (oracle :
              RandomOracle.Test.Point (RefreshSchedule.schedule logicalWidth publicFits) 8 →
                RandomOracle.Test.Answer) =>
          if RandomOracle.Fork.Succeeds (RefreshFork.game ajtai key) adversary oracle then 1
          else 0) ≤
        (Finset.expect Finset.univ
            fun
              (oracle :
                RandomOracle.Test.Point (RefreshSchedule.schedule logicalWidth publicFits) 8 →
                  RandomOracle.Test.Answer) =>
            ∑ retries :
              Fin (RefreshSchedule.schedule logicalWidth publicFits).rhoCount →
                RandomOracle.Test.Answer,
              RandomOracle.Uniqueness.weight (RefreshFork.game ajtai key) adversary oracle
                  retries *
                if RefreshKnowledge.Extracts ajtai key adversary oracle retries then 1 else 0) +
          RefreshKnowledge.knowledgeError ajtai key adversary queries :=
  @Lifecycle.BoundedDepth.RandomOracle.RefreshKnowledge.knowledge_error_le logicalWidth publicFits
    ajtai key adversary queries bounded

/-- The light knowledge bound at the bounded-depth scheme's key width with both terms of named problems as the
advantage of an explicit algorithm: the MSIS solver and the Poseidon2 collision finder, each
with `budget` adversary runs. -/
theorem light_production_knowledge_error
    (key : (RandomOracleSetup.SetupIndex StepRelation.keyWidth StepRelation.keyPublicFits →
          AjtaiSetupV1.Rejection.Stream) →
        CircuitKey StepRelation.keyWidth StepRelation.keyPublicFits)
    (adversary : (RandomOracleSetup.SetupIndex StepRelation.keyWidth StepRelation.keyPublicFits →
          AjtaiSetupV1.Rejection.Stream) →
        RandomOracle.OracleComp
          (RandomOracle.Test.Point
            (LightSchedule.schedule StepRelation.keyWidth StepRelation.keyPublicFits) 8)
          RandomOracle.Test.Answer
          (ForkLink.Attempt StepRelation.keyWidth StepRelation.keyPublicFits))
    {queries : ℕ}
    (bounded : ∀
        (chunks :
          RandomOracleSetup.SetupIndex StepRelation.keyWidth StepRelation.keyPublicFits →
            AjtaiSetupV1.Rejection.Stream),
        (adversary chunks).QueryBound queries)
    {budget : ℕ}
    (positive : 0 < budget) :
    (Finset.expect Finset.univ
          fun
            (chunks :
              RandomOracleSetup.SetupIndex StepRelation.keyWidth StepRelation.keyPublicFits →
                AjtaiSetupV1.Rejection.Stream) =>
          RandomOracleSetup.lightSuccess key adversary chunks) <
        (Finset.expect Finset.univ
                    fun
                      (chunks :
                        RandomOracleSetup.SetupIndex StepRelation.keyWidth
                            StepRelation.keyPublicFits →
                          AjtaiSetupV1.Rejection.Stream) =>
                    RandomOracleSetup.lightExtraction key adversary chunks) +
                  LightKnowledge.statisticalError queries +
                RandomOracleSetup.lightSolverAdvantage key adversary budget +
              2 * RandomOracleSetup.lightHashAdvantage key adversary budget +
            (7 * (↑queries : ℝ) + 124) / (↑budget : ℝ) +
          1 / 2 ^ 1129 :=
  @Export.BoundedDepth.RandomOracleSetup.light_production_knowledge_error_lt_solvers key adversary
    queries bounded budget positive

/-- The refresh knowledge bound at the bounded-depth scheme's key width with both terms of named problems as
the advantage of an explicit algorithm. -/
theorem refresh_production_knowledge_error
    (key : (RandomOracleSetup.SetupIndex StepRelation.keyWidth StepRelation.keyPublicFits →
          AjtaiSetupV1.Rejection.Stream) →
        CircuitKey StepRelation.keyWidth StepRelation.keyPublicFits)
    (adversary : (RandomOracleSetup.SetupIndex StepRelation.keyWidth StepRelation.keyPublicFits →
          AjtaiSetupV1.Rejection.Stream) →
        RandomOracle.OracleComp
          (RandomOracle.Test.Point
            (RefreshSchedule.schedule StepRelation.keyWidth StepRelation.keyPublicFits) 8)
          RandomOracle.Test.Answer
          (ForkLink.Attempt StepRelation.keyWidth StepRelation.keyPublicFits))
    {queries : ℕ}
    (bounded : ∀
        (chunks :
          RandomOracleSetup.SetupIndex StepRelation.keyWidth StepRelation.keyPublicFits →
            AjtaiSetupV1.Rejection.Stream),
        (adversary chunks).QueryBound queries)
    {budget : ℕ}
    (positive : 0 < budget) :
    (Finset.expect Finset.univ
          fun
            (chunks :
              RandomOracleSetup.SetupIndex StepRelation.keyWidth StepRelation.keyPublicFits →
                AjtaiSetupV1.Rejection.Stream) =>
          RandomOracleSetup.refreshSuccess key adversary chunks) <
        (Finset.expect Finset.univ
                    fun
                      (chunks :
                        RandomOracleSetup.SetupIndex StepRelation.keyWidth
                            StepRelation.keyPublicFits →
                          AjtaiSetupV1.Rejection.Stream) =>
                    RandomOracleSetup.refreshExtraction key adversary chunks) +
                  RefreshKnowledge.statisticalError queries +
                RandomOracleSetup.refreshSolverAdvantage key adversary budget +
              16 * RandomOracleSetup.refreshHashAdvantage key adversary budget +
            (77 * (↑queries : ℝ) + 1286) / (↑budget : ℝ) +
          1 / 2 ^ 1129 :=
  @Export.BoundedDepth.RandomOracleSetup.refresh_production_knowledge_error_lt_solvers key
    adversary queries bounded budget positive

end NightstreamFPrime.Claims.Fold
