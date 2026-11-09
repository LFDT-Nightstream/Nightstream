import NightstreamFPrime.Spec.Folding.Light.Extraction

/-! The light Π_RLC output in module terms (Cyclo, ePrint 2026/359, Figure 3, step 6, on
SuperNeo v1.2 Section 7.4): each combined value is `running + ρ · fresh` in the module of
SuperNeo v1.2 Lemma 8's extraction algebra. A step circuit computes exactly this: one
combination step per value, with the running value as the prior (CW-2.3). -/

namespace NightstreamFPrime.Spec.Folding.Light.Rlc

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkAlgebra
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction (ExtractionAlgebra)

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment uScalar uValue

theorem linearCombination_pair_eq {Scalar : Type uScalar} {Value : Type uValue}
    {ring : CommutativeRingOps Scalar} {module : ModuleOps Scalar Value}
    (laws : ModuleLaws ring module) (challenge : Scalar) (fresh running : Value) :
    linearCombination ring module (pair challenge ring.one) (pair fresh running) =
      module.add running (module.smul challenge fresh) := by
  rw [linearCombination_pair, laws.one_smul, laws.add_zero, laws.add_comm]

variable {Structure : Type uStructure} {Assignment : Type uAssignment}
  {PublicInput : Type uPublicInput} {Point : Type uPoint} {Evaluation : Type uEvaluation}
  {Commitment : Type uCommitment} {Scalar : Type uScalar}
  {semantics : RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams}
  {algebra : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment Scalar
    semantics params}

theorem output_commitment (laws : ExtractionAlgebra semantics params algebra)
    (challenge : Scalar) (system : Structure) (point : Point)
    (fresh running : CE.Instance Structure PublicInput Point Evaluation Commitment) :
    (output algebra laws.ring.one challenge system point fresh running).commitment =
      laws.commitmentModule.add running.commitment
        (laws.commitmentModule.smul challenge fresh.commitment) := by
  simp only [output]
  rw [laws.combineCommitment_eq, linearCombination_pair_eq laws.commitmentLaws]

theorem output_publicInput (laws : ExtractionAlgebra semantics params algebra)
    (challenge : Scalar) (system : Structure) (point : Point)
    (fresh running : CE.Instance Structure PublicInput Point Evaluation Commitment) :
    (output algebra laws.ring.one challenge system point fresh running).publicInput =
      laws.publicInputModule.add running.publicInput
        (laws.publicInputModule.smul challenge fresh.publicInput) := by
  simp only [output]
  rw [laws.combinePublicInput_eq, linearCombination_pair_eq laws.publicInputLaws]

/-- Each evaluation family of the output is `running + ρ · fresh`, when both inputs carry the
same number of families. -/
theorem output_evaluation (laws : ExtractionAlgebra semantics params algebra)
    (challenge : Scalar) (system : Structure) (point : Point)
    (fresh running : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (size : fresh.evaluations.size = running.evaluations.size) (index : Nat) :
    (output algebra laws.ring.one challenge system point fresh running).evaluations.getD index
        laws.evaluationModule.zero =
      laws.evaluationModule.add (running.evaluations.getD index laws.evaluationModule.zero)
        (laws.evaluationModule.smul challenge
          (fresh.evaluations.getD index laws.evaluationModule.zero)) := by
  simp only [output]
  rw [laws.combineEvaluations_getD (pair challenge laws.ring.one)
    (pair fresh.evaluations running.evaluations) running.evaluations.size index (by decide)
    (fun source => by
      by_cases first : source.val = 0 <;> simp [pair, first, size])]
  rw [← linearCombination_pair_eq laws.evaluationLaws]
  congr 1
  funext source
  by_cases first : source.val = 0 <;> simp [pair, first]

end NightstreamFPrime.Spec.Folding.Light.Rlc
