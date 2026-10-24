import CompressionLean.Bridge.Formulas
import CompressionLean.Bridge.Run

/-!
Owns the model's parameters as the replay's relation and statement give
them: the sizes (`shapeOf`), the CE(B) statement (`statementOf`), the key's
run and slot structure (`structureOf`), the `bar` transform (`barOf`) and
`Φ81 = X⁵⁴ + X²⁷ + 1` (`phi`).
-/

namespace CompressionLean.Bridge

open Polynomial
open CompressionLean.Replay CompressionLean.Layer1

/-- The model's sizes of a replay relation. -/
def shapeOf (rel : Replay.Relation) : Layer1.Shape where
  D := Replay.D
  laneVars := Replay.laneVars
  blockVars := rel.blockVars
  blocks := rel.key.blocks
  kappa := rel.kappa
  matrices := rel.matrices
  publicBlocks := rel.publicBlocks
  normBound := rel.normBound
  runVars := rel.key.layout.runVariables
  rowVars := rel.key.layout.rowVariables
  slotVars := rel.key.layout.slotVariables
  matrixVars := rel.matrixVars
  setupVars := rel.setupVars
  queries := rel.queries

/-- The CE(B) statement of the replay's words. -/
def statementOf (st : Replay.Statement) : Layer1.Statement Gl where
  commitment ρ l := ((st.commitment[ρ]!)[l]!).toF
  publicInput b l := ((st.publicInput[b]!)[l]!).toF
  r := st.point.toList.map fun v => (v.1.toF, v.2.toF)
  evalK l := ((st.evalK[l]!).1.toF, (st.evalK[l]!).2.toF)
  evalA j l := (((st.evalA[j]!)[l]!).1.toF, ((st.evalA[j]!)[l]!).2.toF)

/-- The key's run and slot structure. -/
def structureOf (layout : Replay.Layout) : Layer1.Structure Gl where
  runBound j := layout.runBounds[j]!
  classLength c := (layout.classes[c]!).len
  classRatio c := (layout.classes[c]!).ratio.toF
  classes := layout.classes.size
  segmentFirst i := (layout.segments[i]!).first
  segmentClass i := (layout.segments[i]!).classIndex
  segmentLane i := (layout.segments[i]!).lane
  segments := layout.segments.size
  slots := layout.slots

/-- The `bar` transform of the F′ closed form, in `Gl`. -/
def barOf (p l : ℕ) : Gl := (Replay.bar p l).toF

/-- `Φ81 = X⁵⁴ + X²⁷ + 1`. -/
noncomputable def phi : Gl[X] := X ^ 54 + X ^ 27 + 1

theorem statement_point (st : Replay.Statement) : (statementOf st).point = quadPoint st.point.toList := by
  simp [Layer1.Statement.point, statementOf, quadPoint, Function.comp_def]

theorem statement_r_length (st : Replay.Statement) : (statementOf st).r.length = st.point.size := by
  simp [statementOf]

/-- What `readProof` checked. -/
structure ProofShape (rel : Replay.Relation) (proof : Replay.Proof) : Prop where
  histogram : proof.histogram.size = 2 * rel.normBound + 1
  pairs : proof.earlyValues.pairs.size = rel.matrices
  quotient : proof.quotient.size = D - 1
  linear : proof.linear.size = rel.cubeVars
  rounds : ∀ m ∈ proof.linear.toList, m.size = 2
  leaves : proof.leaves.size = rel.queries
  lateValues : proof.lateValues.size = rel.matrices
  finals : proof.finals.size = 3
  partials : proof.partials.size = 2
  partialSizes : ∀ m ∈ proof.partials.toList, m.size = 2 ^ foldVars
  early : readGkr proof.early rel.earlyShapes = .ok ()
  leafShapes : ∀ leaf ∈ proof.leaves.toList,
    leaf.values.size = (rel.kappa + runOracles) * 2 ^ foldVars ∧ leaf.path.size = rel.setupVars + 1 - foldVars
  late : readGkr proof.late rel.lateShapes = .ok ()

theorem readProof_ok {rel : Replay.Relation} {proof : Replay.Proof} (read : readProof rel proof = .ok ()) :
    ProofShape rel proof := by
  unfold readProof at read
  obtain ⟨_, shape, read⟩ := Except.bind_ok read
  obtain ⟨_, early, read⟩ := Except.bind_ok read
  obtain ⟨_, leaves, read⟩ := Except.bind_ok read
  obtain ⟨_, late, _⟩ := Except.bind_ok read
  have shape := Except.ensure_ok shape
  have leaves := Except.ensure_ok leaves
  simp only [Bool.and_eq_true, beq_iff_eq, ← Array.all_toList, List.all_eq_true] at shape leaves
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨histogram, pairs⟩, quotient⟩, linear⟩, rounds⟩, leafCount⟩, lateValues⟩, p0⟩, p1⟩, finals⟩,
    partials⟩, partialSizes⟩ := shape
  exact ⟨histogram, pairs, quotient, linear, rounds, leafCount, lateValues, finals, partials, partialSizes, early,
    leaves, late⟩

end CompressionLean.Bridge
