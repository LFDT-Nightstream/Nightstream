import NightstreamFPrime.Export.Stage1.PilotDigestBindingPlan
import NightstreamFPrime.Export.Stage1.PilotPoseidonPreservation

/-!
Owns the complete semantic bridge for the direct pilot relation. It composes
the selective Poseidon2 chains, eight digest-custody rows, and the exact 1,330
ordinary rows into the canonical logical pilot specification.

This module does not claim that unused legacy permutation locals satisfy the
old expanded template rows.
-/

namespace NightstreamFPrime.Export.Stage1.PilotDirectSemantics

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def poseidonGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    PiRLCPoseidonGeometry.Geometry program logicalWidth :=
  PilotDigestBindingPlan.poseidonGeometry geometry

/-- Direct pilot acceptance facts imply the exact logical pilot builder
specification under the canonical pilot pullback environment. -/
theorem implies_spec
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (env : Env)
    (hashes : PilotPoseidonPreservation.HashFacts
      (poseidonGeometry geometry) assignment env)
    (ordinaryRows : R1CS.RowsHold env PilotOrdinaryDirectSource.sourceRows)
    (binding : PilotDigestBindingPlan.Matches geometry assignment)
    (priorDigestValues : ∀ lane : Fin 4,
      ((PilotOrdinaryDirectPlan.Location.priorDigest lane).form geometry).eval assignment =
        env (PilotSpartan.sourceToSpartan
          (PilotOrdinaryDirectPlan.Location.priorDigest lane).sourceColumn))
    (outputStateValues : ∀ lane : Fin 4,
      ((PilotOrdinaryDirectPlan.Location.outputState lane).form geometry).eval assignment =
        env (PilotSpartan.sourceToSpartan
          (PilotOrdinaryDirectPlan.Location.outputState lane).sourceColumn)) :
    Lifecycle.Pilot.SpecHolds PilotProduction.interface
      PilotProduction.witnessOffset
      (PilotSpartan.pullback
        env) := by
  have predicates :=
    PilotOrdinaryDirectSource.sourceRows_hold_implies_packagePredicates
      env ordinaryRows
  have priorLane (lane : Fin 4) :
      NightstreamFPrime.Export.Pilot.chainOutputState PilotData.priorChain
          PilotData.priorChain.absorbCount env
          ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩ =
        PilotPoseidonPreservation.directDigest PilotData.priorChain
          PilotPoseidonPreservation.priorInvocationCount_eq
          (PilotPoseidonPreservation.priorOutputValue
            (poseidonGeometry geometry) assignment) lane := by
    have legacy := PilotOrdinaryDirectPlan.priorDigest_form_eval_chainOutput
      geometry assignment env priorDigestValues lane
    have matched := binding.prior lane
    rw [PilotDigestBindingPlan.legacyForm_priorRow,
      PilotDigestBindingPlan.derivedForm_priorRow] at matched
    calc
      NightstreamFPrime.Export.Pilot.chainOutputState PilotData.priorChain
          PilotData.priorChain.absorbCount env
          ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩ =
          ((PilotOrdinaryDirectPlan.Location.priorDigest lane).form geometry).eval
            assignment := legacy.symm
      _ = ((PilotPoseidonPlan.priorInterface
            (poseidonGeometry geometry)).output
          PilotDigestBindingPlan.lastInvocation
          (PilotDigestBindingPlan.digestLane lane)).eval assignment := matched
      _ = PilotPoseidonPreservation.directDigest PilotData.priorChain
          PilotPoseidonPreservation.priorInvocationCount_eq
          (PilotPoseidonPreservation.priorOutputValue
            (poseidonGeometry geometry) assignment) lane := by
        unfold PilotPoseidonPreservation.directDigest
          PilotPoseidonPreservation.priorOutputValue Hash.digestF
        congr 2 <;> apply Fin.ext <;> rfl
  have outputLane (lane : Fin 4) :
      NightstreamFPrime.Export.Pilot.chainOutputState PilotData.outputChain
          PilotData.outputChain.absorbCount env
          ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩ =
        PilotPoseidonPreservation.directDigest PilotData.outputChain
          PilotPoseidonPreservation.outputInvocationCount_eq
          (PilotPoseidonPreservation.outputOutputValue
            (poseidonGeometry geometry) assignment) lane := by
    have legacy := PilotOrdinaryDirectPlan.outputState_form_eval_chainOutput
      geometry assignment env outputStateValues lane
    have matched := binding.output lane
    rw [PilotDigestBindingPlan.legacyForm_outputRow,
      PilotDigestBindingPlan.derivedForm_outputRow] at matched
    calc
      NightstreamFPrime.Export.Pilot.chainOutputState PilotData.outputChain
          PilotData.outputChain.absorbCount env
          ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩ =
          ((PilotOrdinaryDirectPlan.Location.outputState lane).form geometry).eval
            assignment := legacy.symm
      _ = ((PilotPoseidonPlan.outputInterface
            (poseidonGeometry geometry)).output
          PilotDigestBindingPlan.lastInvocation
          (PilotDigestBindingPlan.digestLane lane)).eval assignment := matched
      _ = PilotPoseidonPreservation.directDigest PilotData.outputChain
          PilotPoseidonPreservation.outputInvocationCount_eq
          (PilotPoseidonPreservation.outputOutputValue
            (poseidonGeometry geometry) assignment) lane := by
        unfold PilotPoseidonPreservation.directDigest
          PilotPoseidonPreservation.outputOutputValue Hash.digestF
        congr 2 <;> apply Fin.ext <;> rfl
  have priorChainHash :
      List.ofFn (fun lane : Fin 4 =>
        NightstreamFPrime.Export.Pilot.chainOutputState PilotData.priorChain
          PilotData.priorChain.absorbCount env
          ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩) =
        Spec.Poseidon2.hash
          (NightstreamFPrime.Export.Pilot.chainInputValues
            PilotData.priorChain env) := by
    calc
      List.ofFn (fun lane : Fin 4 =>
          NightstreamFPrime.Export.Pilot.chainOutputState PilotData.priorChain
            PilotData.priorChain.absorbCount env
            ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩) =
          List.ofFn (PilotPoseidonPreservation.directDigest
            PilotData.priorChain
            PilotPoseidonPreservation.priorInvocationCount_eq
            (PilotPoseidonPreservation.priorOutputValue
              (poseidonGeometry geometry) assignment)) := by
        exact congrArg List.ofFn (funext priorLane)
      _ = Spec.Poseidon2.hash
          (NightstreamFPrime.Export.Pilot.chainInputValues
            PilotData.priorChain env) := hashes.prior
  have outputChainHash :
      List.ofFn (fun lane : Fin 4 =>
        NightstreamFPrime.Export.Pilot.chainOutputState PilotData.outputChain
          PilotData.outputChain.absorbCount env
          ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩) =
        Spec.Poseidon2.hash
          (NightstreamFPrime.Export.Pilot.chainInputValues
            PilotData.outputChain env) := by
    calc
      List.ofFn (fun lane : Fin 4 =>
          NightstreamFPrime.Export.Pilot.chainOutputState PilotData.outputChain
            PilotData.outputChain.absorbCount env
            ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩) =
          List.ofFn (PilotPoseidonPreservation.directDigest
            PilotData.outputChain
            PilotPoseidonPreservation.outputInvocationCount_eq
            (PilotPoseidonPreservation.outputOutputValue
              (poseidonGeometry geometry) assignment)) := by
        exact congrArg List.ofFn (funext outputLane)
      _ = Spec.Poseidon2.hash
          (NightstreamFPrime.Export.Pilot.chainInputValues
            PilotData.outputChain env) := hashes.output
  have assertions : ∀ assertion ∈ PilotData.assertionRows (),
      assertion.Holds env := predicates.2
  have extraLogical : ConstraintsHold (PilotSpartan.pullback env)
      (PilotData.priorExtraConstraints ()) :=
    PilotOrdinaryDirectSource.sourceRows_hold_implies_priorConstraints
      env ordinaryRows
  have postHashRows : holdsFlat (PilotSpartan.pullback env)
      (PriorStateHash.wordOps PilotProduction.priorInterface
          PilotProduction.witnessOffset ++
        PriorStateHash.bindingAssertions PilotProduction.priorInterface
          PilotProduction.witnessOffset) := by
    rw [PilotData.priorExtraConstraints,
      PilotProduction.fastPriorWordOps_eq] at extraLogical
    simpa only [holdsFlat] using extraLogical
  have priorSpec := PriorStateHash.soundness_of_hash_and_postHash
    PilotProduction.priorInterface (PilotSpartan.pullback env)
    PilotProduction.witnessOffset
    (PilotProduction.layoutAssumptions (PilotSpartan.pullback env)).1
    (NightstreamFPrime.Export.Pilot.priorRawSpec_of_chainHash env
      priorChainHash) postHashRows
  have outputRows := NightstreamFPrime.Export.Pilot.canonicalAssertions_sound
    env assertions
  have outputHash :
      List.ofFn (fun lane : Fin 4 =>
        env (PilotData.outputChain.digestStart + lane.val)) =
        Spec.Poseidon2.hash
          (NightstreamFPrime.Export.Pilot.chainInputValues
            PilotData.outputChain env) := by
    calc
      List.ofFn (fun lane : Fin 4 =>
          env (PilotData.outputChain.digestStart + lane.val)) =
          List.ofFn (fun lane : Fin 4 =>
            NightstreamFPrime.Export.Pilot.chainOutputState
              PilotData.outputChain PilotData.outputChain.absorbCount env
              ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩) := by
        exact congrArg List.ofFn (funext outputRows)
      _ = Spec.Poseidon2.hash
          (NightstreamFPrime.Export.Pilot.chainInputValues
            PilotData.outputChain env) := outputChainHash
  exact NightstreamFPrime.Export.Pilot.hashFacts_imply_spec env
    ⟨priorSpec, outputHash⟩

end NightstreamFPrime.Export.Stage1.PilotDirectSemantics
