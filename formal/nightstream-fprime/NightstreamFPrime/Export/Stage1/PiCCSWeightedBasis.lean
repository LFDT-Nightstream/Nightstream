import NightstreamFPrime.Export.Stage1.PiCCSSparseEvaluation

/-! Prepare extension-field weights for an existing family of sparse forms.
The coefficient vector evaluates every weighted output through one source
read per lane. Repeated entries and cancellation retain their existing
`SparseForm` meanings. Source reads may already contain weighted sources. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSWeightedBasis

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
open NightstreamFPrime.Layout.ProductionRelation

/-- Combine all output weights once, in canonical output order. -/
def prepare {outputs lanes : Nat} (forms : Fin outputs → SparseForm lanes)
    (weights : Fin outputs → K) : Vector K lanes :=
  Vector.ofFn fun lane =>
    sumMap extensionOps (canonicalFinIndices outputs) fun output =>
      extensionOps.mul (weights output) (K.embed ((forms output).coefficient lane))

/-- Evaluate the prepared coefficients on extension-field source reads. -/
def dotK {lanes : Nat} (prepared : Vector K lanes)
    (source : Fin lanes → K) : K :=
  sumMap extensionOps (canonicalFinIndices lanes) fun lane =>
    extensionOps.mul (prepared.get lane) (source lane)

/-- Preparation stores the exact coefficient sum; it accepts no supplied table. -/
theorem prepare_value {outputs lanes : Nat}
    (forms : Fin outputs → SparseForm lanes) (weights : Fin outputs → K)
    (lane : Fin lanes) :
    (prepare forms weights).get lane =
      sumMap extensionOps (canonicalFinIndices outputs) (fun output =>
        extensionOps.mul (weights output)
          (K.embed ((forms output).coefficient lane))) := by
  change (Vector.ofFn _)[lane.val] = _
  rw [Vector.getElem_ofFn]

private theorem pair_foldl {Index : Type} (indices : List Index)
    (left right : Index → F) (initial : K) :
    (⟨indices.foldl (fun total index => total + left index) initial.c0,
      indices.foldl (fun total index => total + right index) initial.c1⟩ : K) =
      extensionOps.add initial
        (sumMap extensionOps indices (fun index => ⟨left index, right index⟩)) := by
  induction indices generalizing initial with
  | nil => exact (extensionLaws.add_zero initial).symm
  | cons index indices inductionHypothesis =>
      simp only [List.foldl_cons]
      calc
        _ = extensionOps.add
            ⟨initial.c0 + left index, initial.c1 + right index⟩
            (sumMap extensionOps indices (fun current =>
              ⟨left current, right current⟩)) := inductionHypothesis _
        _ = _ := extensionLaws.add_assoc initial ⟨left index, right index⟩ _

/-- The two stored-entry evaluations equal the complete coefficient sum.
Repeated and cancelling entries need no separate hypothesis. -/
theorem evaluateK_eq_coefficients {lanes : Nat}
    (form : SparseForm lanes) (source : Fin lanes → K) :
    PiCCSSparseEvaluation.evaluateK form source =
      sumMap extensionOps (canonicalFinIndices lanes) (fun lane =>
        extensionOps.mul (K.embed (form.coefficient lane)) (source lane)) := by
  simp only [PiCCSSparseEvaluation.evaluateK, SparseForm.evalSparse_eq_eval,
    SparseForm.eval]
  erw [pair_foldl (canonicalFinIndices lanes)
    (fun lane => form.coefficient lane * (source lane).c0)
    (fun lane => form.coefficient lane * (source lane).c1) extensionOps.zero,
    extensionLaws.zero_add]
  apply sumMap_congr
  intro lane _
  change (⟨form.coefficient lane * (source lane).c0,
    form.coefficient lane * (source lane).c1⟩ : K) =
      K.mul (K.embed (form.coefficient lane)) (source lane)
  simp only [K.mul, K.embed, Fin.mul_zero, Fin.zero_mul, Fin.add_zero]

private theorem sumMap_mul_right {Index : Type} (indices : List Index)
    (value : Index → K) (factor : K) :
    sumMap extensionOps indices (fun index =>
        extensionOps.mul (value index) factor) =
      extensionOps.mul (sumMap extensionOps indices value) factor := by
  calc
    _ = sumMap extensionOps indices (fun index =>
        extensionOps.mul factor (value index)) :=
      sumMap_congr extensionOps indices _ _ (fun _ _ => extensionLaws.mul_comm _ _)
    _ = extensionOps.mul factor (sumMap extensionOps indices value) :=
      sumMap_mul_left extensionOps extensionLaws factor indices value
    _ = _ := extensionLaws.mul_comm _ _

/-- One prepared dot product equals all weighted extension-field sparse
outputs. This is total: it has no validity, norm, or support premise. -/
theorem dotK_prepare {outputs lanes : Nat}
    (forms : Fin outputs → SparseForm lanes) (weights : Fin outputs → K)
    (source : Fin lanes → K) :
    dotK (prepare forms weights) source =
      sumMap extensionOps (canonicalFinIndices outputs) (fun output =>
        extensionOps.mul (weights output)
          (PiCCSSparseEvaluation.evaluateK (forms output) source)) := by
  unfold dotK
  simp only [prepare_value]
  calc
    _ = sumMap extensionOps (canonicalFinIndices lanes) (fun lane =>
        sumMap extensionOps (canonicalFinIndices outputs) (fun output =>
          extensionOps.mul
            (extensionOps.mul (weights output)
              (K.embed ((forms output).coefficient lane)))
            (source lane))) := by
      apply sumMap_congr
      intro lane _
      exact (sumMap_mul_right (canonicalFinIndices outputs) _ (source lane)).symm
    _ = sumMap extensionOps (canonicalFinIndices outputs) (fun output =>
        sumMap extensionOps (canonicalFinIndices lanes) (fun lane =>
          extensionOps.mul
            (extensionOps.mul (weights output)
              (K.embed ((forms output).coefficient lane)))
            (source lane))) :=
      sumMap_swap extensionOps extensionLaws
        (canonicalFinIndices lanes) (canonicalFinIndices outputs) _
    _ = _ := by
      apply sumMap_congr
      intro output _
      calc
        _ = sumMap extensionOps (canonicalFinIndices lanes) (fun lane =>
            extensionOps.mul (weights output)
              (extensionOps.mul (K.embed ((forms output).coefficient lane))
                (source lane))) := by
          apply sumMap_congr
          intro lane _
          exact extensionLaws.mul_assoc _ _ _
        _ = extensionOps.mul (weights output)
            (sumMap extensionOps (canonicalFinIndices lanes) (fun lane =>
              extensionOps.mul (K.embed ((forms output).coefficient lane))
                (source lane))) :=
          sumMap_mul_left extensionOps extensionLaws (weights output)
            (canonicalFinIndices lanes) _
        _ = _ := by rw [← evaluateK_eq_coefficients]

end NightstreamFPrime.Export.Stage1.PiCCSWeightedBasis
