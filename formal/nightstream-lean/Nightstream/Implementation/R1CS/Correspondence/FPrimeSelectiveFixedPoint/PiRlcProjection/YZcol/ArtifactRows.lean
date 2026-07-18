import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol
import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveSourceLayout.ProjectionIdentity
import Nightstream.Implementation.R1CS.Correspondence.Projection.ArtifactProgram

/-!
Exact-row transport for the bounded tiny-fixture PiRLC `y_zcol` projection.

Owns: set-level coverage of the two reconstructed projection traces by the
checked artifact certificate, direct transport from satisfied selected source
rows, and the weaker convenience transport from embedded full source-R1CS
rows to the generic active `RowsSatisfied` interface.

Does not own: the fixture's structural census, full-program embedding or
satisfaction, assignment canonicality, constant-one enforcement, producer ↔
consumer binding, PiCCS/transcript/parent authority, security bounds, costs,
production-wide Rust conformance, or permission to remove rows.

Emits constraints: no.

| Branch | Mathematical obligation | Evidence boundary |
|---|---|---|
| shared definitions | ladder and rho equations occur once physically but in both trace views | set-level `Covers.definitionsIff` |
| limb definitions | every low/high input, product, output, quotient, and Phi row is covered | exact owner schedule |
| final checks | both assertion rows for each limb are covered | exact owner schedule |
| selected-row transport | satisfying exact selected source rows imply generic trace rows | explicit caller premises |
| full-source transport | embedded satisfying source rows imply generic trace rows | convenience corollary only |

Assurance tier: artifact-checked for this bounded tiny fixture only, assuming
the separately supplied handwritten-schema `StructureValid` proof. This file
makes no production-wide, security-reduced, or row-removal claim.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ArtifactRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram

private abbrev artifact := Checked.artifact
private abbrev certificate := artifact.certificate

private theorem map_snd_zip_of_length
    {Alpha Beta : Type} :
    ∀ (left : List Alpha) (right : List Beta),
      right.length ≤ left.length →
      (List.zip left right).map Prod.snd = right
  | _, [], _ => by simp
  | [], _ :: _, lengthLe => by simp at lengthLe
  | _ :: left, right :: rights, lengthLe => by
      simp only [List.zip_cons_cons, List.map_cons]
      rw [map_snd_zip_of_length left rights
        (Nat.le_of_succ_le_succ lengthLe)]

private theorem indexedDefinitions_values
    {rows : RowBlock} {definitions : List Program.Definition}
    (fits : rows.Fits definitions.length) :
    (rows.indexDefinitions definitions).map Prod.snd = definitions := by
  unfold RowBlock.indexDefinitions
  apply map_snd_zip_of_length
  simp [RowBlock.indices, fits.2]

private theorem indexedChecks_values
    {rows : RowBlock} {checks : List Row}
    (fits : rows.Fits checks.length) :
    (rows.indexChecks checks).map Prod.snd = checks := by
  unfold RowBlock.indexChecks
  apply map_snd_zip_of_length
  simp [RowBlock.indices, fits.2]

private theorem map_snd_flatMap
    {Owner Value : Type}
    (owners : List Owner)
    (indexed : Owner → List (Nat × Value))
    (values : Owner → List Value)
    (matching : ∀ owner ∈ owners,
      (indexed owner).map Prod.snd = values owner) :
    (owners.flatMap indexed).map Prod.snd = owners.flatMap values := by
  induction owners with
  | nil => rfl
  | cons owner owners inductionHypothesis =>
      simp only [List.flatMap_cons, List.map_append]
      rw [matching owner (by simp)]
      rw [inductionHypothesis]
      intro candidate member
      exact matching candidate (by simp [member])

private theorem evaluation_values
    {owner : EvaluationOwner} {coefficientCount : Nat}
    (valid : owner.Valid coefficientCount) :
    owner.indexedDefinitions.map Prod.snd = owner.trace.definitions := by
  exact indexedDefinitions_values valid.2.2

private theorem product_values
    {owner : KProductOwner} (valid : owner.Valid) :
    owner.indexedDefinitions.map Prod.snd = owner.trace.definitions := by
  exact indexedDefinitions_values valid.2

private theorem pairs_values
    (laneCount : Nat) :
    ∀ (pairs : List PairOwner) (rhos : List EvaluationOwner),
      pairs.length = rhos.length →
      (∀ entry ∈ List.zip pairs rhos,
        entry.1.Valid entry.2 laneCount) →
      (pairs.flatMap fun pair =>
          pair.inputEvaluation.indexedDefinitions ++
            pair.rhoProduct.indexedDefinitions).map Prod.snd =
        ((List.zip pairs rhos).map fun entry =>
          entry.1.trace entry.2).flatMap fun trace =>
            trace.inputEvaluation.definitions ++ trace.product.definitions
  | [], [], _, _ => rfl
  | [], _ :: _, lengthEq, _ => by simp at lengthEq
  | _ :: _, [], lengthEq, _ => by simp at lengthEq
  | pair :: pairs, rho :: rhos, lengthEq, valid => by
      have headValid : pair.Valid rho laneCount :=
        valid (pair, rho) (by simp)
      have tailValid : ∀ entry ∈ List.zip pairs rhos,
          entry.1.Valid entry.2 laneCount := by
        intro entry member
        exact valid entry (by simp [member])
      have tailLength : pairs.length = rhos.length :=
        Nat.succ.inj lengthEq
      simp only [List.flatMap_cons, List.map_append,
        List.zip_cons_cons, List.map_cons]
      rw [evaluation_values headValid.1,
        product_values headValid.2.1,
        pairs_values laneCount pairs rhos tailLength tailValid]
      rfl

private theorem pair_rho_values :
    ∀ (pairs : List PairOwner) (rhos : List EvaluationOwner),
      pairs.length = rhos.length →
      ((List.zip pairs rhos).map fun entry =>
          entry.1.trace entry.2).flatMap
            (fun trace => trace.rhoEvaluation.definitions) =
        rhos.flatMap fun rho => rho.trace.definitions
  | [], [], _ => rfl
  | [], _ :: _, lengthEq => by simp at lengthEq
  | _ :: _, [], lengthEq => by simp at lengthEq
  | _ :: pairs, rho :: rhos, lengthEq => by
      have tailLength : pairs.length = rhos.length :=
        Nat.succ.inj lengthEq
      simp only [List.zip_cons_cons, List.map_cons, List.flatMap_cons]
      rw [pair_rho_values pairs rhos tailLength]
      rfl

private def sharedDefinitions (shared : SharedOwner) :
    List Program.Definition :=
  shared.ladderTrace.definitions ++
    shared.rhoEvaluations.flatMap fun rho => rho.trace.definitions

private def limbDefinitions (limb : LimbOwner) (shared : SharedOwner) :
    List Program.Definition :=
  ((limb.pairTraces shared).flatMap fun pair =>
      pair.inputEvaluation.definitions ++ pair.product.definitions) ++
    limb.parentEvaluation.trace.definitions ++
    limb.quotientEvaluation.trace.definitions ++
    limb.quotientPhiProduct.trace.definitions

private theorem shared_values
    {scope : Scope} {shared : SharedOwner}
    (valid : shared.Valid scope) :
    shared.indexedDefinitions.map Prod.snd = sharedDefinitions shared := by
  rcases valid with
    ⟨_, _, _, _, _, _, ladderLayout, productsValid, evaluationsValid⟩
  cases powersEq : shared.powers with
  | nil =>
      have contradiction : False := by
        simpa [SharedOwner.ladderTrace, LadderTrace.LayoutValid,
          powersEq] using ladderLayout
      exact contradiction.elim
  | cons power powers =>
      have baseFits : shared.ladderBaseRows.Fits
          (shared.ladderTrace.definitions.take 2).length := by
        simp [RowBlock.Fits, RowBlock.count, SharedOwner.ladderBaseRows,
          SharedOwner.ladderTrace, LadderTrace.definitions, powersEq]
      have productValues := map_snd_flatMap shared.ladderProducts
        KProductOwner.indexedDefinitions
        (fun product => product.trace.definitions)
        (fun product member => product_values
          (productsValid product member))
      have evaluationValues := map_snd_flatMap shared.rhoEvaluations
        EvaluationOwner.indexedDefinitions
        (fun evaluation => evaluation.trace.definitions)
        (fun evaluation member => evaluation_values
          (evaluationsValid evaluation member).1)
      have baseValues : shared.ladderBaseDefinitions.map Prod.snd =
          shared.ladderTrace.definitions.take 2 := by
        simpa [SharedOwner.ladderBaseDefinitions] using
          indexedDefinitions_values baseFits
      simp only [SharedOwner.indexedDefinitions, List.map_append]
      rw [baseValues, productValues, evaluationValues]
      simp [sharedDefinitions, SharedOwner.ladderTrace,
        LadderTrace.definitions, powersEq, List.flatMap_map]

private theorem limb_values
    {scope : Scope} {shared : SharedOwner} {limb : LimbOwner}
    (sharedValid : shared.Valid scope)
    (limbValid : limb.Valid scope shared) :
    limb.indexedDefinitions.map Prod.snd =
      limbDefinitions limb shared := by
  rcases sharedValid with
    ⟨_, _, rhoCount, _, _, _, _, _, _⟩
  rcases limbValid with
    ⟨_, pairCount, _, pairsValid, parentValid, _, quotientValid, _,
      phiValid, _, _, _, _, _, _⟩
  have countEq : limb.pairs.length = shared.rhoEvaluations.length :=
    pairCount.trans rhoCount.symm
  have pairValues := pairs_values scope.laneCount limb.pairs
    shared.rhoEvaluations countEq pairsValid
  simp only [LimbOwner.indexedDefinitions, List.map_append]
  rw [pairValues, evaluation_values parentValid,
    evaluation_values quotientValid, product_values phiValid]
  simp [limbDefinitions, LimbOwner.pairTraces]

private theorem trace_definitions_decompose
    {scope : Scope} {shared : SharedOwner} {limb : LimbOwner}
    (sharedValid : shared.Valid scope)
    (limbValid : limb.Valid scope shared) :
    (limb.trace shared).definitions =
      sharedDefinitions shared ++ limbDefinitions limb shared := by
  rcases sharedValid with
    ⟨_, _, rhoCount, _, _, _, _, _, _⟩
  rcases limbValid with
    ⟨_, pairCount, _, _, _, _, _, _, _, _, _, _, _, _, _⟩
  have countEq : limb.pairs.length = shared.rhoEvaluations.length :=
    pairCount.trans rhoCount.symm
  have rhoValues :
      (limb.pairTraces shared).flatMap
          (fun pair => pair.rhoEvaluation.definitions) =
        shared.rhoEvaluations.flatMap
          (fun rho => rho.trace.definitions) := by
    simpa [LimbOwner.pairTraces] using
      pair_rho_values limb.pairs shared.rhoEvaluations countEq
  simp only [ProjectionTrace.definitions, LimbOwner.trace]
  rw [rhoValues]
  simp [sharedDefinitions, limbDefinitions, LimbOwner.pairTraces,
    List.append_assoc]

private theorem limb_check_values
    {scope : Scope} {shared : SharedOwner} {limb : LimbOwner}
    (valid : limb.Valid scope shared) :
    (limb.indexedChecks shared).map Prod.snd =
      (limb.trace shared).checks := by
  rcases valid with
    ⟨_, _, _, _, _, _, _, _, _, _, _, finalFits, _, _, _⟩
  exact indexedChecks_values finalFits

private theorem limbs_definition_values
    {scope : Scope} {shared : SharedOwner} (limbs : List LimbOwner)
    (sharedValid : shared.Valid scope)
    (limbsValid : ∀ limb ∈ limbs, limb.Valid scope shared) :
    (limbs.flatMap LimbOwner.indexedDefinitions).map Prod.snd =
      limbs.flatMap fun limb => limbDefinitions limb shared := by
  apply map_snd_flatMap
  intro limb member
  exact limb_values sharedValid (limbsValid limb member)

private theorem limbs_check_values
    {scope : Scope} {shared : SharedOwner} (limbs : List LimbOwner)
    (limbsValid : ∀ limb ∈ limbs, limb.Valid scope shared) :
    (limbs.flatMap fun limb => limb.indexedChecks shared).map Prod.snd =
      limbs.flatMap fun limb => (limb.trace shared).checks := by
  apply map_snd_flatMap
  intro limb member
  exact limb_check_values (limbsValid limb member)

private theorem trace_definition_values
    {scope : Scope} {shared : SharedOwner} (limbs : List LimbOwner)
    (sharedValid : shared.Valid scope)
    (limbsValid : ∀ limb ∈ limbs, limb.Valid scope shared) :
    ((limbs.map fun limb => limb.trace shared).flatMap
        ProjectionTrace.definitions) =
      limbs.flatMap fun limb =>
        sharedDefinitions shared ++ limbDefinitions limb shared := by
  induction limbs with
  | nil => rfl
  | cons limb limbs inductionHypothesis =>
      simp only [List.map_cons, List.flatMap_cons]
      rw [trace_definitions_decompose sharedValid
        (limbsValid limb (by simp))]
      rw [inductionHypothesis]
      intro candidate member
      exact limbsValid candidate (by simp [member])

/-- Set-level coverage intentionally allows the shared ladder/rho equations to
appear in both semantic traces while retaining one physical row owner. -/
theorem certificate_covers
    (census : artifact.StructureValid) :
    certificate.Covers Checked.traces := by
  rcases census with
    ⟨_, sharedValid, limbsLength, _, limbsValid, _, _, _⟩
  have tracesEq : Checked.traces = artifact.traces := by rfl
  rw [tracesEq]
  constructor
  · intro definition
    have certificateValues : certificate.definitions =
        sharedDefinitions artifact.shared ++
          artifact.limbs.flatMap fun limb =>
            limbDefinitions limb artifact.shared := by
      change artifact.indexedDefinitions.map Prod.snd = _
      simp only [Artifact.indexedDefinitions, List.map_append]
      rw [shared_values sharedValid,
        limbs_definition_values artifact.limbs sharedValid limbsValid]
    have traceValues :
        artifact.traces.flatMap ProjectionTrace.definitions =
          artifact.limbs.flatMap fun limb =>
            sharedDefinitions artifact.shared ++
              limbDefinitions limb artifact.shared := by
      unfold Artifact.traces
      exact trace_definition_values artifact.limbs sharedValid limbsValid
    rw [traceValues, certificateValues]
    constructor
    · intro member
      rcases List.mem_flatMap.mp member with
        ⟨limb, limbMember, definitionMember⟩
      rcases List.mem_append.mp definitionMember with
        sharedMember | limbDefinitionMember
      · exact List.mem_append_left _ sharedMember
      · apply List.mem_append_right
        exact List.mem_flatMap.mpr
          ⟨limb, limbMember, limbDefinitionMember⟩
    · intro member
      rcases List.mem_append.mp member with
        sharedMember | limbDefinitionMember
      · have limbsNonempty : artifact.limbs ≠ [] := by
          intro empty
          rw [empty] at limbsLength
          simp at limbsLength
        obtain ⟨limb, limbMember⟩ :=
          List.exists_mem_of_ne_nil artifact.limbs limbsNonempty
        exact List.mem_flatMap.mpr
          ⟨limb, limbMember, List.mem_append_left _ sharedMember⟩
      · rcases List.mem_flatMap.mp limbDefinitionMember with
          ⟨limb, limbMember, localMember⟩
        exact List.mem_flatMap.mpr
          ⟨limb, limbMember, List.mem_append_right _ localMember⟩
  · intro row
    have certificateValues : certificate.checks =
        artifact.limbs.flatMap fun limb =>
          (limb.trace artifact.shared).checks := by
      change artifact.indexedChecks.map Prod.snd = _
      unfold Artifact.indexedChecks
      exact limbs_check_values artifact.limbs limbsValid
    have traceValues : artifact.traces.flatMap ProjectionTrace.checks =
        artifact.limbs.flatMap fun limb =>
          (limb.trace artifact.shared).checks := by
      simp [Artifact.traces, List.flatMap_map]
    rw [traceValues, certificateValues]

/-- Exact selected source rows reach the generic artifact-independent row
interface through explicit satisfaction, canonical field representatives,
and the global constant-one invariant. This is the truthful handoff expected
from a future selective-lowering refinement. -/
theorem rowsSatisfied_of_sourceRows
    {shape : Nightstream.SuperNeo.Folding.PiCCS.SplitNc.SemanticShape}
    {pair : PiCcsOutputDigest.ActiveSourceLayout.ProjectionIdentity.TracePair
      shape}
    {assignment : Nat → Nat}
    (pairTraces : pair.traces = Checked.traces)
    (census : artifact.StructureValid)
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (sourceSatisfies : Satisfies certificate.sourceRowValues assignment) :
    PiCcsOutputDigest.ActiveSourceLayout.ProjectionIdentity.RowsSatisfied
      pair assignment := by
  have held := certificate.traceRowsHold_of_sourceRows Checked.exactRows
    (certificate_covers census) assignmentCanonical constantOne
    sourceSatisfies
  constructor
  · rw [pairTraces]
    exact held.1
  · rw [pairTraces]
    exact held.2

/-- Full source-R1CS embedding is one way—not the production selective
lowering—to establish the selected-row satisfaction premise. -/
theorem rowsSatisfied_of_embedded
    {shape : Nightstream.SuperNeo.Folding.PiCCS.SplitNc.SemanticShape}
    {pair : PiCcsOutputDigest.ActiveSourceLayout.ProjectionIdentity.TracePair
      shape}
    {fullRows : List Row} {assignment : Nat → Nat}
    (pairTraces : pair.traces = Checked.traces)
    (census : artifact.StructureValid)
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (embedded : certificate.EmbeddedIn fullRows)
    (fullSatisfies : Satisfies fullRows assignment) :
    PiCcsOutputDigest.ActiveSourceLayout.ProjectionIdentity.RowsSatisfied
      pair assignment := by
  apply rowsSatisfied_of_sourceRows pairTraces census assignmentCanonical
    constantOne
  exact ProjectionIndexedRows.sourceRows_satisfied_of_embedded
    embedded fullSatisfies

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ArtifactRows
