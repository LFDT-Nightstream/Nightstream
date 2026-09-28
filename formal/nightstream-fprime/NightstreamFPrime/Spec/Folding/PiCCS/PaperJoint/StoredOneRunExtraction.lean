import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CheckedWitnessExtraction

/-!
B.2 extraction from a call that returns stored probes and field arrays. The
call owns array creation and coin-reading work. The checker owns public and ambient
witness validation. Projection uses the proved array reader, so its work
bound has no arbitrary-function access premise.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredOneRunExtraction

open NightstreamFPrime.Spec
open StrongReduction ConcreteCarrier WitnessProjection CheckedWitnessExtraction
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

abbrev Call (Tape : Type*) (shape : Shape) (carrier : Phi81Relation.Shape) :=
  Tape → CubePoint K shape.cubeVariables → K → CubePoint K shape.cubeVariables →
    Result (StoredOutcome shape carrier)

abbrev Check (shape : Shape) (carrier : Phi81Relation.Shape) :=
  (StoredProbe shape × StoredWitnessProjection.StoredWitness shape carrier) → Result Bool

def run {Tape : Type*} {shape : Shape} {carrier : Phi81Relation.Shape}
    (call : Call Tape shape carrier) (check : Check shape carrier)
    (tape : Tape) (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables) : Result (Option (SourceWitness shape carrier)) :=
  let issued := call tape alpha gamma point
  let finished := finishStored check issued.value
  ⟨finished.value, issued.work + finished.work + 1⟩

def baseClock {Tape : Type*} {shape : Shape} {carrier : Phi81Relation.Shape}
    (call : Call Tape shape carrier) (check : Check shape carrier)
    (tape : Tape) (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables) : ℝ :=
  let issued := call tape alpha gamma point
  (issued.work + (match issued.value with | none => 0 | some candidate => (check candidate).work) : Nat)

def totalClock {Tape : Type*} {shape : Shape} {carrier : Phi81Relation.Shape}
    (call : Call Tape shape carrier) (check : Check shape carrier)
    (tape : Tape) (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables) : ℝ :=
  (run call check tape alpha gamma point).work

/-- The same actual stored call determines the source event and the work.
Abort and checker rejection retain their full call and check costs. -/
theorem run_work_le {Tape : Type*} {shape : Shape} {carrier : Phi81Relation.Shape}
    (call : Call Tape shape carrier) (check : Check shape carrier)
    (tape : Tape) (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables) :
    totalClock call check tape alpha gamma point ≤ baseClock call check tape alpha gamma point +
      ((CostedWitnessProjection.workBound shape carrier (1 + 1 + 1) + 3 : Nat) : ℝ) := by
  have bound := finishStored_work_le check (call tape alpha gamma point).value
  have natural : (run call check tape alpha gamma point).work ≤
      (call tape alpha gamma point).work +
        (match (call tape alpha gamma point).value with | none => 0 | some candidate => (check candidate).work) +
        (CostedWitnessProjection.workBound shape carrier (1 + 1 + 1) + 3) := by
    dsimp only [run]
    cases issued : (call tape alpha gamma point).value <;>
      simp only [issued] at bound ⊢ <;> omega
  change ((run call check tape alpha gamma point).work : ℝ) ≤
    (((call tape alpha gamma point).work +
      (match (call tape alpha gamma point).value with | none => 0 | some candidate => (check candidate).work) : Nat) : ℝ) +
    ((CostedWitnessProjection.workBound shape carrier (1 + 1 + 1) + 3 : Nat) : ℝ)
  exact_mod_cast natural

variable {Tape Commitment : Type*} {shape : Shape} {carrier : Phi81Relation.Shape}
  {blockCount width : Nat}
  (tapes : PMF Tape) (call : Call Tape shape carrier) (check : Check shape carrier)
  (prover : Tape → CausalExecution.Prover shape carrier.carrierWidth width)
  (callCorrect : ∀ tape alpha gamma point,
    storedView (call tape alpha gamma point).value = CausalExecution.run (prover tape) alpha gamma point)
  (commit : Phi81Relation.Assignment carrier → Commitment) (params : GlobalParams)
  (statement : Statement K Commitment (Phi81Relation.PublicInput carrier)
    shape carrier.carrierWidth blockCount baseOps)
  (checkCorrect : ∀ probe stored, (check (probe, stored)).value = true ↔
    probe.view.FixedWidthAccepted extensionOps K.embed statement width ∧
      AmbientOutputHolds extensionOps K.embed (openingMaps commit) params statement probe.view
        (StoredWitnessProjection.view stored))

include callCorrect checkCorrect in
theorem run_source_iff (tape : Tape) (alpha : CubePoint K shape.cubeVariables)
    (gamma : K) (point : CubePoint K shape.cubeVariables) :
    SourceReturned commit params statement (run call check tape alpha gamma point).value ↔
      StrongProbability.RelaxedSuccess (width := width) (openingMaps commit) params statement
        (CausalExecution.run (prover tape) alpha gamma point) ∧
      StrongProbability.SourceValid (openingMaps commit) params statement
        (CausalExecution.run (prover tape) alpha gamma point) := by
  change SourceReturned commit params statement
    (finishStored check (call tape alpha gamma point).value).value ↔ _
  rw [finishStored_source_iff check commit params statement checkCorrect, callCorrect]

/-- Summability of the total clock follows from the actual call/check mean.
The projection bound uses concrete arrays and needs no accessor hypothesis. -/
theorem expected_work_bound
    (baseSummable : Summable fun tape =>
      (tapes tape).toReal * StrongProbability.verifierMean (baseClock call check tape)) :
    Summable (fun tape => (tapes tape).toReal * StrongProbability.verifierMean (totalClock call check tape)) ∧
      StrongProbability.clockMean tapes (totalClock call check) ≤
        StrongProbability.clockMean tapes (baseClock call check) +
          ((CostedWitnessProjection.workBound shape carrier (1 + 1 + 1) + 3 : Nat) : ℝ) := by
  apply StrongProbability.clockMean_le_add_const tapes (baseClock call check) (totalClock call check) _
  · intro tape alpha gamma point
    exact Nat.cast_nonneg _
  · exact baseSummable
  · exact run_work_le call check

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredOneRunExtraction
