import NightstreamFPrime.Export.Stage1.PiCCSNativeDot
import NightstreamFPrime.Export.Stage1.PiRLCPartialTrace

/-! Aggregate original running sources and Phi81 output weights before a
matrix row reads a column. The prepared basis table is derived from the
existing sparse forms. Complete source blocks include every tail lane. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSCarriedRead

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace

/-- Signed source coefficients use additions; other coefficients retain
the exact scalar multiplication, so the function remains total. -/
def scale (weight : K) (scalar : F) : K :=
  if scalar = 0 then K.zero
  else if scalar = 1 then weight
  else if scalar = -1 then ⟨-weight.c0, -weight.c1⟩
  else ⟨weight.c0 * scalar, weight.c1 * scalar⟩

/-- The scalar fast cases preserve the original extension multiplication. -/
theorem scale_value (weight : K) (scalar : F) :
    scale weight scalar = extensionOps.mul weight (K.embed scalar) := by
  change scale weight scalar = K.mul weight (K.embed scalar)
  by_cases zero : scalar = 0
  · subst scalar
    simp [scale, K.zero, K.mul, K.embed]
  · by_cases one : scalar = 1
    · subst scalar
      simp [scale, K.mul, K.embed, zero]
    · by_cases negative : scalar = -1
      · subst scalar
        simp [scale, K.mul, K.embed, zero, one]
      · simp [scale, zero, one, negative, K.mul, K.embed]

/-- Materialize the weighted original source coefficients once per block. -/
def combine {sources lanes : Nat} (weights : Fin sources → K)
    (source : Fin sources → Fin lanes → F) : Vector K lanes :=
  Vector.ofFn fun lane =>
    sumMap extensionOps (canonicalFinIndices sources) fun index =>
      scale (weights index) (source index lane)

/-- Every combined lane includes exactly the declared source list. -/
theorem combine_value {sources lanes : Nat} (weights : Fin sources → K)
    (source : Fin sources → Fin lanes → F) (lane : Fin lanes) :
    (combine weights source).get lane =
      sumMap extensionOps (canonicalFinIndices sources) (fun index =>
        extensionOps.mul (weights index) (K.embed (source index lane))) := by
  change (Vector.ofFn _)[lane.val] = _
  rw [Vector.getElem_ofFn]
  apply sumMap_congr
  intro index _
  exact scale_value _ _

/-- Prepare one weighted coefficient vector for each existing Phi81 basis. -/
def prepare
    (forms : FixedArray (FixedArray (SparseForm ringDegree) ringDegree) ringDegree)
    (weights : Fin ringDegree → K) : FixedArray (Vector K ringDegree) ringDegree :=
  FixedArray.ofFn fun basis =>
    PiCCSWeightedBasis.prepare (fun output => (forms.get basis).get output) weights

/-- The column selects the existing complete block and basis convention. -/
def read {columns : Nat} (prepared : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree) (column : Fin columns) : K :=
  PiCCSWeightedBasis.dotK
    (prepared.get ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩)
    (blocks (column.val / ringDegree)).get

/-- Prepared column reads equal all weighted original sparse basis outputs.
No signedness, norm, opening, matrix, or expected-output premise is needed. -/
theorem read_prepare {sources columns : Nat}
    (forms : FixedArray (FixedArray (SparseForm ringDegree) ringDegree) ringDegree)
    (outputWeights : Fin ringDegree → K) (sourceWeights : Fin sources → K)
    (blocks : Fin sources → Nat → RingF) (column : Fin columns) :
    read (prepare forms outputWeights)
        (fun block => combine sourceWeights (fun source => blocks source block)) column =
      sumMap extensionOps (canonicalFinIndices ringDegree) (fun output =>
        extensionOps.mul (outputWeights output)
          (sumMap extensionOps (canonicalFinIndices sources) (fun source =>
            extensionOps.mul (sourceWeights source) (K.embed
              (((forms.get ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩).get output).evalSparse
                (blocks source (column.val / ringDegree))))))) := by
  rw [read, prepare, FixedArray.get_ofFn, PiCCSWeightedBasis.dotK_prepare]
  apply sumMap_congr
  intro output _
  apply congrArg (extensionOps.mul (outputWeights output))
  have combined :
      (combine sourceWeights (fun source => blocks source (column.val / ringDegree))).get =
        fun lane => sumMap extensionOps (canonicalFinIndices sources) (fun source =>
          extensionOps.mul (sourceWeights source)
            (K.embed (blocks source (column.val / ringDegree) lane))) := by
    funext lane
    exact combine_value sourceWeights _ lane
  rw [combined]
  exact (PiCCSSparseEvaluation.weighted_reads _ (canonicalFinIndices sources)
    sourceWeights (fun source => blocks source (column.val / ringDegree))).symm

end NightstreamFPrime.Export.Stage1.PiCCSCarriedRead
