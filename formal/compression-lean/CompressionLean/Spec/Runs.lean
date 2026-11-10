import CompressionLean.Spec.Parent
import CompressionLean.Spec.PoseidonRows
import CompressionLean.Production
import NightstreamFPrime.Export.Stage1.PiDECCanonicalSourceCache
import NightstreamFPrime.Export.Stage1.PiDECProductRow
import NightstreamFPrime.Export.Stage1.DirectLowNormFootprint
import NightstreamFPrime.Layout.PiDEC.v1_2.Ownership
import NightstreamFPrime.Layout.ProductionRelation.MultiplicationFamilyPlan
import NightstreamFPrime.Layout.Stage1.RunningTransitionSourceSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.Support.CcsTerminal
import NightstreamFPrime.Lifecycle.Stage1.Terminal
import NightstreamFPrime.Spec
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.GoldilocksCausal
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetConvention
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork

namespace CompressionLean.Spec

open CompressionLean.Setup CompressionLean.Spec.PoseidonRows
open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Layout NightstreamFPrime.Layout.ProductionRelation NightstreamFPrime.Layout.MatrixProgram

def matchRun {n : ℕ} (r : Gl) : ℕ → ℕ → Gl → List (SparseEntry n) → Option (List (SparseEntry n))
  | _, 0, _, es => some es
  | c, k + 1, v, e :: es => if e.column.val = c ∧ gl e.coefficient = v then matchRun r (c + 1) k (v * r) es else none
  | _, _ + 1, _, [] => none

def matchRuns {n : ℕ} : List Push → List (SparseEntry n) → Bool
  | [], es => es.isEmpty
  | p :: ps, es =>
      match matchRun p.ratio p.start p.len p.initial es with
      | some rest => matchRuns ps rest
      | none => false

def entriesNormal {n : ℕ} : List (SparseEntry n) → Bool
  | [] => true
  | [e] => e.coefficient != 0
  | e :: f :: rest => e.coefficient != 0 && e.column.val < f.column.val && entriesNormal (f :: rest)

def normalEntries {n : ℕ} (form : SparseForm n) : List (SparseEntry n) :=
  if entriesNormal form.entries then form.entries else PoseidonRows.normalize form

def rowFailure {n : ℕ} (forms : RowForms n) (runs : List Push) : Option ℕ :=
  (List.finRange Spec.ProductionRelation.meaningfulPortCount).findSome? fun j =>
    if matchRuns (runs.filter (·.matrix = j.val)) (normalEntries (forms j)) then none else some j.val

def runsByRow (rows : ℕ) (pushes : List Push) : Array (List Push) :=
  pushes.foldl (fun byRow p => byRow.modify p.row (p :: ·)) (Array.replicate rows [])

end CompressionLean.Spec
