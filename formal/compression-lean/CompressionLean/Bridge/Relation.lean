import CompressionLean.Bridge.Formulas
import CompressionLean.Bridge.Run
import CompressionLean.Spec.Relation

/-!
Owns `Φ81 = X⁵⁴ + X²⁷ + 1` (`phi`), the point of the replay's statement
(`statement_point`), and what `readProof` checks (`ProofShape`,
`readProof_ok`). The model's sizes, statement and structure of a replay
relation are in `Spec/Relation.lean`.
-/

namespace CompressionLean.Bridge

open CompressionLean.Spec
open Polynomial
open CompressionLean.Replay CompressionLean.Layer1

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
