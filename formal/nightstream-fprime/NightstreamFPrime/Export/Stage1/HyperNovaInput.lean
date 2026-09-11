import NightstreamFPrime.Export.Stage1.PiDECInputCheck

/-!
Connect decoded HyperNova claims to the existing selected NIFS checker input.
The conversion preserves every claim and PiCCS message field. It introduces
no alternate statement, transcript, matrix source, or witness representation.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaInput

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction

abbrev Running := Lifecycle.Running
  (logicalWidth := PiDECInputCheck.logicalWidth)
  (publicFits := PiDECInputCheck.publicFits)

abbrev Fresh := Lifecycle.Fresh
  (logicalWidth := PiDECInputCheck.logicalWidth)
  (publicFits := PiDECInputCheck.publicFits)

private theorem vector_get_ofFn {α : Type} {n : Nat} (f : Fin n → α) (i : Fin n) :
    (Vector.ofFn f).get i = f i := Vector.getElem_ofFn i.isLt

private theorem running_ext {E C P : Type} {shape : Shape}
    (left right : Spec.Folding.Nifs.PaperNonInteractive.Running E C P shape)
    (point : left.point = right.point) (commitments : left.commitments = right.commitments)
    (publicInputs : left.publicInputs = right.publicInputs)
    (evaluations : left.evaluations = right.evaluations) : left = right := by
  cases left; cases right; cases point; cases commitments; cases publicInputs; cases evaluations; rfl

private theorem fresh_ext {C P : Type} {shape : Shape}
    (left right : Spec.Folding.Nifs.PaperNonInteractive.Fresh C P shape)
    (commitments : left.commitments = right.commitments)
    (publicInputs : left.publicInputs = right.publicInputs) : left = right := by
  cases left; cases right; cases commitments; cases publicInputs; rfl

private theorem point_ext {E : Type} {dimensionCount : Nat}
    (left right : CubePoint E dimensionCount) (coordinates : left.coordinates = right.coordinates) :
    left = right := by
  cases left; cases right; cases coordinates; rfl

private theorem evaluation_ext {E : Type} {shape : Shape}
    (left right : EvaluationFamily E shape) (pad : left.pad = right.pad)
    (matrix : left.matrix = right.matrix) : left = right := by
  cases left; cases right; cases pad; cases matrix; rfl

private def commitmentWords (value : PaperAlgebra.Commitment) : Vector F 1188 :=
  Vector.ofFn fun index => value
    ⟨index.val / 54, by have bound := index.isLt; change index.val / 54 < 22; omega⟩
    ⟨index.val % 54, Nat.mod_lt _ (by decide)⟩

/-- Serialize the existing typed running bundle into the checker's fixed arrays. -/
def runningInput (value : Running) : PiCCSInputCheck.RunningInput where
  point := ⟨value.point.coordinates.toArray, by simpa using value.point.dimension⟩
  commitments := Vector.ofFn fun source => commitmentWords (value.commitments source)
  publicInputs := Vector.ofFn fun source => Vector.ofFn (value.publicInputs source)
  evalK := Vector.ofFn fun source => Vector.ofFn (value.evaluations source).pad
  evalA := Vector.ofFn fun source => Vector.ofFn fun matrix =>
    Vector.ofFn ((value.evaluations source).matrix matrix)

/-- Recover the same typed bundle, including the complete Pad and matrix families. -/
theorem runningFromInput_runningInput (value : Running) :
    PiCCSInputCheck.runningFromInput (runningInput value) = value := by
  apply running_ext
  · apply point_ext
    rfl
  · funext source row coefficient
    simp only [PiCCSInputCheck.runningFromInput, runningInput, vector_get_ofFn, commitmentWords]
    apply congrArg₂ (value.commitments source)
    · apply Fin.ext
      have cb : coefficient.val < 54 := coefficient.isLt
      change (row.val * 54 + coefficient.val) / 54 = row.val
      omega
    · apply Fin.ext
      have cb : coefficient.val < 54 := coefficient.isLt
      change (row.val * 54 + coefficient.val) % 54 = coefficient.val
      omega
  · funext source column
    simp only [PiCCSInputCheck.runningFromInput, runningInput, vector_get_ofFn]
  · funext source
    apply evaluation_ext
    · funext coefficient
      simp only [PiCCSInputCheck.runningFromInput, runningInput, vector_get_ofFn]
    · funext matrix coefficient
      simp only [PiCCSInputCheck.runningFromInput, runningInput, vector_get_ofFn]

/-- Preserve the actual PiCCS messages when forming the existing checker input. -/
def ofClaims (running : Running) (fresh : Fresh) (proof : Lifecycle.Proof 9) :
    PiCCSInputCheck.Input where
  commitment := commitmentWords (fresh.commitments ⟨0, by decide⟩)
  publicInput := Vector.ofFn (fresh.publicInputs ⟨0, by decide⟩)
  rounds := Vector.ofFn fun round =>
    ⟨(proof.piCcsRounds round).coefficients.toArray, by
      simpa using (proof.piCcsRounds round).coefficients_length⟩
  evalK := Vector.ofFn fun source => Vector.ofFn (proof.piCcsOutput.padCoordinate source)
  evalA := Vector.ofFn fun source => Vector.ofFn fun matrix =>
    Vector.ofFn (proof.piCcsOutput.matrixCoordinate source matrix)
  running := runningInput running

/-- The checked source statement retains the exact decoded running claims. -/
theorem running_ofClaims (running : Running) (fresh : Fresh) (proof : Lifecycle.Proof 9) :
    PiCCSInputCheck.running (ofClaims running fresh proof) = running :=
  runningFromInput_runningInput running

/-- The checked source statement retains the exact decoded fresh claim. -/
theorem fresh_ofClaims (running : Running) (fresh : Fresh) (proof : Lifecycle.Proof 9) :
    PiCCSInputCheck.fresh (ofClaims running fresh proof) = fresh := by
  apply fresh_ext
  · funext source row coefficient
    have sourceEq : source = ⟨0, by decide⟩ := by
      apply Fin.ext
      have bound : source.val < 1 := source.isLt
      change source.val = 0
      omega
    rw [sourceEq]
    simp only [PiCCSInputCheck.fresh, PiCCSInputCheck.proofValues, ofClaims,
      commitmentWords, vector_get_ofFn]
    apply congrArg₂ (fresh.commitments ⟨0, by decide⟩)
    · apply Fin.ext
      have cb : coefficient.val < 54 := coefficient.isLt
      change (row.val * 54 + coefficient.val) / 54 = row.val
      omega
    · apply Fin.ext
      have cb : coefficient.val < 54 := coefficient.isLt
      change (row.val * 54 + coefficient.val) % 54 = coefficient.val
      omega
  · funext source
    have sourceEq : source = ⟨0, by decide⟩ := by
      apply Fin.ext
      have bound : source.val < 1 := source.isLt
      change source.val = 0
      omega
    rw [sourceEq]
    funext column
    simp only [PiCCSInputCheck.fresh, ofClaims, vector_get_ofFn]

end NightstreamFPrime.Export.Stage1.HyperNovaInput
