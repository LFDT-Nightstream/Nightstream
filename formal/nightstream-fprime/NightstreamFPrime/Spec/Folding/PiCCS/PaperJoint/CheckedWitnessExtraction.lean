import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessProjection
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongProbability

/-!
SuperNeo B.2's checked one-run output. The checker returns the acceptance bit
and its work together. Its specification is the existing fixed-width verifier
and corrected ambient output relation on the same returned probe and witness.
The only projection path is the costed field-access program. Fresh prefixes
are restored from the verifier statement through Phi81Relation's actual map.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CheckedWitnessExtraction

open NightstreamFPrime.Spec
open StrongReduction ConcreteCarrier WitnessProjection UnifiedSources
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

abbrev Outcome (shape : Shape) (carrier : Phi81Relation.Shape) :=
  Option (Probe K shape × OutputWitness shape carrier.carrierWidth)

/-- These are the existing production opening maps at the selected Phi81 prefix. -/
def openingMaps {Commitment : Type*} {carrier : Phi81Relation.Shape}
    (commit : Phi81Relation.Assignment carrier → Commitment) :
    OpeningMaps Commitment (Phi81Relation.PublicInput carrier) carrier.carrierWidth where
  commit := commit
  projectPublicInput := Phi81Relation.projectPublicInput

/-- Both operations expose the work of their actual invocation. -/
structure Program (shape : Shape) (carrier : Phi81Relation.Shape) where
  check : (Probe K shape × OutputWitness shape carrier.carrierWidth) → Result Bool
  access : CostedWitnessProjection.Accessor shape carrier

/-- Checker correctness includes both acceptance and the full ambient witness
check. It is an implementation-refinement premise, not a cryptographic assumption. -/
structure Correct {Commitment : Type*} {shape : Shape} {carrier : Phi81Relation.Shape}
    {blockCount width : Nat} (program : Program shape carrier)
    (commit : Phi81Relation.Assignment carrier → Commitment) (params : GlobalParams)
    (statement : Statement K Commitment (Phi81Relation.PublicInput carrier)
      shape carrier.carrierWidth blockCount baseOps) : Prop where
  check : ∀ probe witness,
    (program.check (probe, witness)).value = true ↔
      probe.FixedWidthAccepted extensionOps K.embed statement width ∧
        AmbientOutputHolds extensionOps K.embed (openingMaps commit) params statement probe witness
  access : CostedWitnessProjection.Correct program.access

/-- Shared checked return for semantic and stored witness representations. -/
private def finishChecked {Candidate : Type*} {shape : Shape} {carrier : Phi81Relation.Shape}
    (check : Candidate → Result Bool) (project : Candidate → Result (SourceWitness shape carrier)) :
    Option Candidate → Result (Option (SourceWitness shape carrier))
  | none => ⟨none, 1⟩
  | some candidate =>
      let checked := check candidate
      if checked.value then
        let projected := project candidate
        ⟨some projected.value, checked.work + projected.work + 2⟩
      else ⟨none, checked.work + 2⟩

/-- Abort returns immediately. Otherwise execute the checker once, and copy
the actual witness only on acceptance. Each dispatch and return is charged. -/
def finish {shape : Shape} {carrier : Phi81Relation.Shape}
    (program : Program shape carrier) : Outcome shape carrier → Result (Option (SourceWitness shape carrier)) :=
  finishChecked program.check (fun candidate => CostedWitnessProjection.project program.access candidate.2)

/-- Only a returned candidate is checked; abort has no hidden check call. -/
def checkerWork {shape : Shape} {carrier : Phi81Relation.Shape}
    (program : Program shape carrier) : Outcome shape carrier → Nat
  | none => 0
  | some candidate => (program.check candidate).work

theorem finish_return_iff {shape : Shape} {carrier : Phi81Relation.Shape}
    (program : Program shape carrier) (outcome : Outcome shape carrier)
    (values : SourceWitness shape carrier) :
    (finish program outcome).value = some values ↔
      ∃ probe witness, outcome = some (probe, witness) ∧
        (program.check (probe, witness)).value = true ∧
        (CostedWitnessProjection.project program.access witness).value = values := by
  cases outcome with
  | none => simp [finish, finishChecked]
  | some candidate =>
      rcases candidate with ⟨probe, witness⟩
      cases checked : (program.check (probe, witness)).value <;> simp [finish, finishChecked, checked]

variable {Commitment : Type*} {shape : Shape} {carrier : Phi81Relation.Shape}
  {blockCount width : Nat} (program : Program shape carrier)
  (commit : Phi81Relation.Assignment carrier → Commitment) (params : GlobalParams)
  (statement : Statement K Commitment (Phi81Relation.PublicInput carrier)
    shape carrier.carrierWidth blockCount baseOps)
  (correct : Correct (width := width) program commit params statement)

include correct in
/-- The ambient relation binds the exact Phi81 prefix; no extra prefix
representation premise is imposed on the selected consumer. -/
theorem ambient_implies_reconstruct (probe : Probe K shape)
    (witness : OutputWitness shape carrier.carrierWidth)
    (ambient : AmbientOutputHolds extensionOps K.embed (openingMaps commit) params statement probe witness) :
    reconstruct statement.publicInputs (CostedWitnessProjection.project program.access witness).value = witness := by
  rw [CostedWitnessProjection.project_value program.access correct.access]
  apply WitnessProjection.reconstruct_project
  intro source
  exact (ambient (freshSourceIndex source)).1.2.1

include correct in
/-- Every returned result is the projection of the same accepted ambient
witness. Restoring statement-owned prefixes recovers that exact witness. -/
theorem finish_returns_reconstruction (outcome : Outcome shape carrier)
    (values : SourceWitness shape carrier) (returned : (finish program outcome).value = some values) :
    ∃ probe witness, outcome = some (probe, witness) ∧
      probe.FixedWidthAccepted extensionOps K.embed statement width ∧
      AmbientOutputHolds extensionOps K.embed (openingMaps commit) params statement probe witness ∧
      reconstruct statement.publicInputs values = witness := by
  rcases (finish_return_iff program outcome values).mp returned with
    ⟨probe, witness, issued, checked, projected⟩
  have accepted := (correct.check probe witness).mp checked
  refine ⟨probe, witness, issued, accepted.1, accepted.2, ?_⟩
  rw [← projected]
  exact ambient_implies_reconstruct program commit params statement correct probe witness accepted.2

/-- A valid returned source witness is interpreted through the existing source
relation and the verifier-owned public prefix. -/
def SourceReturned (result : Option (SourceWitness shape carrier)) : Prop :=
  ∃ values, result = some values ∧
    SourceHolds extensionOps K.embed (openingMaps commit) params statement
      (reconstruct statement.publicInputs values)

include correct in
/-- The actual returned-source event is exactly the event measured by the
proved B.2 probability theorem. No successful output is selected by choice. -/
theorem finish_source_iff (outcome : Outcome shape carrier) :
    SourceReturned commit params statement (finish program outcome).value ↔
      StrongProbability.RelaxedSuccess (width := width) (openingMaps commit) params statement outcome ∧
        StrongProbability.SourceValid (openingMaps commit) params statement outcome := by
  constructor
  · rintro ⟨values, returned, source⟩
    rcases finish_returns_reconstruction program commit params statement correct outcome values returned with
      ⟨probe, witness, issued, accepted, ambient, reconstructed⟩
    refine ⟨⟨probe, witness, issued, accepted, ambient⟩, probe, witness, issued, ?_⟩
    rwa [reconstructed] at source
  · rintro ⟨⟨probe, witness, issued, accepted, ambient⟩, source⟩
    rcases source with ⟨sourceProbe, sourceWitness, sourceIssued, valid⟩
    have same := Option.some.inj (issued.symm.trans sourceIssued)
    obtain ⟨probeEq, witnessEq⟩ := Prod.mk.inj same
    subst sourceProbe
    subst sourceWitness
    refine ⟨(CostedWitnessProjection.project program.access witness).value, ?_, ?_⟩
    · exact (finish_return_iff program outcome _).mpr
        ⟨probe, witness, issued, (correct.check probe witness).mpr ⟨accepted, ambient⟩, rfl⟩
    · rw [ambient_implies_reconstruct program commit params statement correct probe witness ambient]
      exact valid

/-- The returned tails and full running values satisfy the existing source
CCS/CE products. The selected fresh profile has b=2. -/
theorem sourceReturned_iff_memberships (freshBound : params.b = 2)
    (result : Option (SourceWitness shape carrier)) :
    SourceReturned commit params statement result ↔
      ∃ values, result = some values ∧
        (∀ fresh, CCS.Holds (paperRelationSemantics baseOps extensionOps K.embed (openingMaps commit)) params
          (SourceMembership.freshInstance statement fresh)
          ((reconstruct statement.publicInputs values).assignments (freshSourceIndex fresh))) ∧
        (∀ running, CE.Holds (paperRelationSemantics baseOps extensionOps K.embed (openingMaps commit)) params
          (SourceMembership.runningInstance statement running)
          ((reconstruct statement.publicInputs values).assignments (runningSourceIndex running))) := by
  unfold SourceReturned
  simp only [SourceMembership.sourceHolds_iff_memberships extensionOps extensionLaws K.embed
    (openingMaps commit) params freshBound]

/-- The work bound includes rejected and aborted results as well as success. -/
theorem finish_work_le (accessBound : Nat)
    (bounded : CostedWitnessProjection.Bounded program.access accessBound)
    (outcome : Outcome shape carrier) :
    (finish program outcome).work ≤ checkerWork program outcome +
      CostedWitnessProjection.workBound shape carrier accessBound + 2 := by
  cases outcome with
  | none => simp only [finish, finishChecked, checkerWork, Nat.zero_add]; omega
  | some candidate =>
      have projection := CostedWitnessProjection.project_work_le program.access accessBound bounded candidate.2
      cases checked : (program.check candidate).value <;>
        simp only [finish, finishChecked, checkerWork, checked, Bool.false_eq_true, ↓reduceIte] <;> omega

/-- The producing call returns storage, not an erased function to be read
later at an unknown cost. Its clock includes constructing these arrays. -/
abbrev StoredOutcome (shape : Shape) (carrier : Phi81Relation.Shape) :=
  Option (Probe K shape × StoredWitnessProjection.StoredWitness shape carrier)

def finishStored {shape : Shape} {carrier : Phi81Relation.Shape}
    (check : (Probe K shape × StoredWitnessProjection.StoredWitness shape carrier) → Result Bool) :
    StoredOutcome shape carrier → Result (Option (SourceWitness shape carrier)) :=
  finishChecked check (fun candidate => StoredWitnessProjection.project candidate.2)

/-- Storage is erased only for the existing semantic success events. -/
def storedView {shape : Shape} {carrier : Phi81Relation.Shape}
    (outcome : StoredOutcome shape carrier) : Outcome shape carrier :=
  outcome.map (fun candidate => (candidate.1, StoredWitnessProjection.view candidate.2))

/-- The concrete projection bound applies on every branch. The checker work
remains charged to the actual call; it is not replaced by an assumed constant. -/
theorem finishStored_work_le {shape : Shape} {carrier : Phi81Relation.Shape}
    (check : (Probe K shape × StoredWitnessProjection.StoredWitness shape carrier) → Result Bool)
    (outcome : StoredOutcome shape carrier) :
    (finishStored check outcome).work ≤
      (match outcome with | none => 0 | some candidate => (check candidate).work) +
        CostedWitnessProjection.workBound shape carrier (1 + 1 + 1) + 2 := by
  cases outcome with
  | none => simp only [finishStored, finishChecked, Nat.zero_add]; omega
  | some candidate =>
      have projection := StoredWitnessProjection.project_work_le candidate.2
      cases checked : (check candidate).value <;>
        simp only [finishStored, finishChecked, checked, Bool.false_eq_true, ↓reduceIte] <;> omega

/-- The checked array return has the exact existing B.2 source-success event.
Only correctness of the actual public/ambient checker remains a premise;
the array access and source projection are implemented and proved here. -/
theorem finishStored_source_iff {Commitment : Type*} {shape : Shape} {carrier : Phi81Relation.Shape}
    {blockCount width : Nat}
    (check : (Probe K shape × StoredWitnessProjection.StoredWitness shape carrier) → Result Bool)
    (commit : Phi81Relation.Assignment carrier → Commitment) (params : GlobalParams)
    (statement : Statement K Commitment (Phi81Relation.PublicInput carrier)
      shape carrier.carrierWidth blockCount baseOps)
    (checked : ∀ probe stored, (check (probe, stored)).value = true ↔
      probe.FixedWidthAccepted extensionOps K.embed statement width ∧
        AmbientOutputHolds extensionOps K.embed (openingMaps commit) params statement probe
          (StoredWitnessProjection.view stored))
    (outcome : StoredOutcome shape carrier) :
    SourceReturned commit params statement (finishStored check outcome).value ↔
      StrongProbability.RelaxedSuccess (width := width) (openingMaps commit) params statement (storedView outcome) ∧
        StrongProbability.SourceValid (openingMaps commit) params statement (storedView outcome) := by
  cases outcome with
  | none =>
      simp [finishStored, finishChecked, SourceReturned, storedView,
        StrongProbability.RelaxedSuccess, StrongProbability.SourceValid]
  | some candidate =>
      rcases candidate with ⟨probe, stored⟩
      by_cases accepted : (check (probe, stored)).value = true
      · have valid := (checked probe stored).mp accepted
        have reconstructed := StoredWitnessProjection.reconstruct_project statement.publicInputs stored
          (fun source => (valid.2 (freshSourceIndex source)).1.2.1)
        simp [finishStored, finishChecked, accepted, SourceReturned, storedView,
          StrongProbability.RelaxedSuccess, StrongProbability.SourceValid,
          valid.1, valid.2, reconstructed]
      · have rejected : ¬ (probe.FixedWidthAccepted extensionOps K.embed statement width ∧
            AmbientOutputHolds extensionOps K.embed (openingMaps commit) params statement probe
              (StoredWitnessProjection.view stored)) := fun valid => accepted ((checked probe stored).mpr valid)
        simp [finishStored, finishChecked, accepted, SourceReturned, storedView,
          StrongProbability.RelaxedSuccess, StrongProbability.SourceValid, rejected]

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CheckedWitnessExtraction
