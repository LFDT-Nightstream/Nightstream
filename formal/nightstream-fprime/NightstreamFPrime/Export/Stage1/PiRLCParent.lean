import NightstreamFPrime.Export.Stage1.PiDECInputCheck
import Mathlib.Data.List.Forall2

/-!
Connect the exact materialized R trace to the paper's combined claim and
weak-suffix success predicate. PiDEC supplies the reconstructed opening;
its validity is not an extra premise on the R parent. This connection states
no Fiat–Shamir probability law and adds no extraction-work assumption.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiRLCParent

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open PiRLCPartialTrace PiRLCNonzero

abbrev Input := PiCCSInputCheck.Input
abbrev Batch := Transcript.PiRlcSampler.Batch SourceCount
abbrev Values := PiRLCInputCheck.ParentValues

private theorem finalParent_parts (point : PaperAlgebra.Point) (batch : Batch)
    (cs : List MaterializedCommitment) (xs : List MaterializedPublicInput)
    (ks : List MaterializedRingK) (asByMatrix : List (List MaterializedRingK))
    (values : Values)
    (returned : PiRLCInputCheck.finalParent point batch cs xs ks asByMatrix = some values) :
    values.point = point ∧ values.outgoing = batch.finalState ∧
      cs.getLast? = some values.commitment ∧ xs.getLast? = some values.publicInput ∧
      ks.getLast? = some values.evalK ∧
      asByMatrix.mapM List.getLast? = some values.evalA.toList := by
  simp only [PiRLCInputCheck.finalParent, Option.bind_eq_bind,
    Option.bind_eq_some_iff] at returned
  rcases returned with ⟨c, hc, x, hx, k, hk, a, ha, returned⟩
  split at returned
  · have same := Option.some.inj returned
    cases same
    exact ⟨rfl, rfl, hc, hx, hk, by simpa using ha⟩
  · cases returned

private theorem mapM_some_pairs {Alpha Beta : Type} (f : Alpha → Option Beta)
    (xs : List Alpha) (ys : List Beta) (returned : xs.mapM f = some ys) :
    List.Forall₂ (fun x y => f x = some y) xs ys := by
  induction xs generalizing ys with
  | nil =>
      have same : [] = ys := by simpa using returned
      cases same
      exact .nil
  | cons x xs ih =>
      cases hx : f x with
      | none => simp [List.mapM_cons, hx] at returned
      | some y =>
          cases hs : xs.mapM f with
          | none => simp [List.mapM_cons, hx, hs] at returned
          | some rest =>
              have same : y :: rest = ys := by simpa [List.mapM_cons, hx, hs] using returned
              cases same
              exact .cons hx (ih rest hs)

/-- The pure values materialized by the C/R check's independent tasks. -/
def computedParent (input : Input) (batch : Batch) : Option Values :=
  PiRLCInputCheck.finalParent (PiCCSInputCheck.execute input).point batch
    (commitmentPartials batch.challenges (PiRLCInputCheck.commitments input))
    (publicInputPartials batch.challenges (PiRLCInputCheck.publicInputs input))
    (evaluationPartials batch.challenges fun source => (PiRLCInputCheck.evaluations input source).pad)
    ((List.finRange productionShape.matrixCount).map fun matrix =>
      evaluationPartials batch.challenges fun source => (PiRLCInputCheck.evaluations input source).matrix matrix)

def sourceClaim (input : Input) (source : Fin SourceCount) : PiDECInputCheck.Claim where
  constraintSystem := Lifecycle.PiRLC.v1_1.InputBinding.relationSource PiDECInputCheck.relation
  commitment := PiRLCInputCheck.commitments input source
  publicInput := PiRLCInputCheck.publicInputs input source
  point := (PiCCSInputCheck.execute input).point
  evaluations := #[PiRLCInputCheck.evaluations input source]
  stage := .fresh

def inputBatch (input : Input) :
    PiRLC.PaperForkExtraction.InputBatch (PaperAlgebra.Structure PiDECInputCheck.logicalWidth)
      PiCCSInputCheck.PublicInput PaperAlgebra.Point PaperAlgebra.Evaluation
      PaperAlgebra.Commitment productionGlobalParams Nifs.PaperProfile.arity where
  system := Lifecycle.PiRLC.v1_1.InputBinding.relationSource PiDECInputCheck.relation
  point := (PiCCSInputCheck.execute input).point
  inputs := sourceClaim input
  sameSystem := fun _ => rfl
  samePoint := fun _ => rfl
  evaluationCount := 1
  evaluationsSize := fun _ => rfl

/-- This actual batch has the exact commitment projection used by the
existing strong-prefix/weak-suffix interface, for every probe. -/
theorem inputBatch_phi_eq_probe (input : Input)
    (probe : StrongReduction.Probe K productionShape) :
    PiRLC.phi (inputBatch input).inputs =
      PiRLC.phi (Nifs.PaperStrongInterface.piRlcBatchForProbe
        (ProductionKey.key PiDECInputCheck.relation Poseidon2HashChainV1Setup.productionAjtaiKey)
        (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input) probe).inputs := by
  funext source
  exact PiRLCInputCheck.commitments_eq_batch PiDECInputCheck.relation
    Poseidon2HashChainV1Setup.productionAjtaiKey input probe source

theorem computedParent_outgoing (input : Input) (batch : Batch) (values : Values)
    (returned : computedParent input batch = some values) :
    values.outgoing = batch.finalState :=
  (finalParent_parts _ _ _ _ _ _ _ returned).2.1

private theorem evaluation_ext (left right : PaperAlgebra.Evaluation)
    (pad : left.pad = right.pad) (matrix : left.matrix = right.matrix) : left = right := by
  cases left
  cases right
  simp_all

private theorem claim_ext (left right : PiDECInputCheck.Claim)
    (system : left.constraintSystem = right.constraintSystem)
    (commitment : left.commitment = right.commitment)
    (publicInput : left.publicInput = right.publicInput)
    (point : left.point = right.point)
    (evaluations : left.evaluations = right.evaluations)
    (stage : left.stage = right.stage) : left = right := by
  cases left
  cases right
  simp_all

/-- The public action depends on the five public rings, not on the width of
the private assignment carrier. Induction keeps the proof independent of
the selected package's column count. -/
private theorem combinePublicInputs_width
    (leftWidth rightWidth : Nat)
    (leftFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth leftWidth)
    (rightFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth rightWidth)
    {count : Nat} (challenges : Fin count → RingF) (inputs : Fin count → Fin 270 → F)
    (column : Fin 270) :
    Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs
        (shape := FullShape leftWidth leftFits) challenges inputs column =
      Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs
        (shape := FullShape rightWidth rightFits) challenges inputs column := by
  induction count with
  | zero => rfl
  | succ count ih =>
      simp only [Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs,
        Phi81Relation.PiRLCAlgebra.PublicInput.publicAdd]
      rw [ih (fun index => challenges index.succ) (fun index => inputs index.succ)]
      rfl

/-- The actual materialized endpoint is the paper R output, including both
evaluation families and all selected relation, point and stage fields. -/
theorem computedParent_eq_combined (input : Input) (batch : Batch) (values : Values)
    (returned : computedParent input batch = some values) :
    PiDECInputCheck.parent values =
      PiRLC.combinedOutput (arity := Nifs.PaperProfile.arity)
        (piRlcAlgebra Poseidon2HashChainV1Setup.productionAjtaiKey)
        (inputBatch input).system (inputBatch input).point (inputBatch input).inputs
        batch.challenges := by
  have parts := finalParent_parts _ _ _ _ _ _ _ returned
  have hc := commitmentPartials_getLast? batch.challenges (PiRLCInputCheck.commitments input)
  rw [List.getLast?_map, parts.2.2.1, Option.map_some] at hc
  have hx := publicInputPartials_getLast? batch.challenges (PiRLCInputCheck.publicInputs input)
  rw [List.getLast?_map, parts.2.2.2.1, Option.map_some] at hx
  have hk := evaluationPartials_getLast? batch.challenges
    (fun source => (PiRLCInputCheck.evaluations input source).pad)
  rw [List.getLast?_map, parts.2.2.2.2.1, Option.map_some] at hk
  have paired := mapM_some_pairs List.getLast? _ _ parts.2.2.2.2.2
  have matrices (matrix : Fin productionShape.matrixCount) :
      (values.evalA.get matrix).toRing =
        PiRLCFinite.combineEvaluation batch.challenges
          (fun source => (PiRLCInputCheck.evaluations input source).matrix matrix) := by
    have matrixReturned :
        (evaluationPartials batch.challenges
          (fun source => (PiRLCInputCheck.evaluations input source).matrix matrix)).getLast? =
          some (values.evalA.get matrix) := by
      simpa [List.get_eq_getElem] using paired.get (i := matrix.val)
        (by simpa using matrix.isLt) (by simpa using matrix.isLt)
    have hm := evaluationPartials_getLast? batch.challenges
      (fun source => (PiRLCInputCheck.evaluations input source).matrix matrix)
    rw [List.getLast?_map, matrixReturned, Option.map_some] at hm
    exact Option.some.inj hm
  have family : PiDECInputCheck.evaluation values =
      combineEvaluationFamily batch.challenges (PiRLCInputCheck.evaluations input) := by
    apply evaluation_ext
    · exact Option.some.inj hk
    · funext matrix
      exact matrices matrix
  have publicCanonical : values.publicInput.toPublicInput =
      Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs
        (shape := FullShape VerifierContext.candidateLogicalWidth VerifierContext.candidatePublicFits)
        batch.challenges (PiRLCInputCheck.publicInputs input) := Option.some.inj hx
  have widthEquality := combinePublicInputs_width
    VerifierContext.candidateLogicalWidth PiDECInputCheck.logicalWidth
    VerifierContext.candidatePublicFits PiDECInputCheck.publicFits
    batch.challenges (PiRLCInputCheck.publicInputs input)
  have publicSelected (column : Fin 270) : values.publicInput.get column =
      Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs
        (shape := FullShape PiDECInputCheck.logicalWidth PiDECInputCheck.publicFits)
        batch.challenges (PiRLCInputCheck.publicInputs input) column :=
    (congrFun publicCanonical column).trans (widthEquality column)
  apply claim_ext
  · rfl
  · exact Option.some.inj hc
  · funext column
    exact publicSelected column
  · exact parts.1
  · change #[PiDECInputCheck.evaluation values] =
      PaperAlgebra.combineEvaluations batch.challenges
        (fun source => #[PiRLCInputCheck.evaluations input source])
    rw [PaperAlgebra.combineEvaluations_singletons (by decide)]
    exact congrArg (fun value : PaperAlgebra.Evaluation => #[value]) family
  · rfl

/-- Accepted D children supply the exact weak-extraction success witness
for this checked C/R run. Sampler replay is connected without assigning it
an interactive or Fiat–Shamir coin law. -/
theorem checked_children_imply_rlc_success
    (input : Input) (batch : Batch) (values : Values)
    (messages : PiDECInputCheck.Messages)
    (assignments : Fin 16 → PaperAlgebra.Assignment
      (logicalWidth := PiDECInputCheck.logicalWidth)
      (publicFits := PiDECInputCheck.publicFits))
    (sampled : PiRLCInputCheck.sampled input = some batch)
    (returned : computedParent input batch = some values)
    (checked : PiDECInputCheck.accepted values messages = true)
    (childrenValid : ∀ child,
      CE.Holds (semantics Poseidon2HashChainV1Setup.productionAjtaiKey)
        productionGlobalParams (PiDECInputCheck.children values messages child) (assignments child)) :
    (PiCCSInputCheck.execute input).accepted = true ∧
      ProductionKey.piRlcResponse (PiCCSInputCheck.execute input).outgoing = some batch.challenges ∧
      PiRLC.PaperForkExtraction.Response.Success
        (semantics Poseidon2HashChainV1Setup.productionAjtaiKey) productionGlobalParams
        (piRlcAlgebra Poseidon2HashChainV1Setup.productionAjtaiKey) (inputBatch input)
        { challenges := batch.challenges, assignment := Radix.recomposeAssignment assignments } ∧
      ∀ probe : StrongReduction.Probe K productionShape,
        PiRLC.phi (inputBatch input).inputs =
          PiRLC.phi (Nifs.PaperStrongInterface.piRlcBatchForProbe
            (ProductionKey.key PiDECInputCheck.relation Poseidon2HashChainV1Setup.productionAjtaiKey)
            (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input) probe).inputs := by
  have response := PiRLCInputCheck.sampled_response input batch sampled
  refine ⟨response.1, response.2, ?_, inputBatch_phi_eq_probe input⟩
  change CE.Holds _ _ (PiRLC.combinedOutput _ _ _ _ _) _
  rw [← computedParent_eq_combined input batch values returned]
  exact PiDECInputCheck.accepted_reduces_knowledge values messages assignments checked childrenValid

end NightstreamFPrime.Export.Stage1.PiRLCParent
