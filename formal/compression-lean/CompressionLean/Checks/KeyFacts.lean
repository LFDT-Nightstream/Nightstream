import CompressionLean.Circuit.Verifier
import CompressionLean.Bridge.EvalK
import CompressionLean.Bridge.Honest

/-!
Owns the facts about the layer-1 key that the final theorems assume
(`KeyFacts`, `WidthFits`), and their check on a key (`keyFailure`,
`poseidon2Fits`). The production key is read at run time, so
`lowerCheck layer1` and `terminalCheck` run the check on the key that they
read.
-/

namespace CompressionLean.Checks

open CompressionLean.Circuit (Layer1Input)
open CompressionLean.Bridge (EvalKBits HonestSizes shapeOf structureOf)

/-- The facts about the layer-1 key that the final theorems assume, with the
size conditions of the layer-1 argument at the Goldilocks modulus (`shape`). Rust's
`Structure::check` (`neo-spartan/src/matrix.rs:142`) enforces `short` and
`checked`. `Relation::new` (`neo-spartan/src/lib.rs:334`) enforces `high`,
and `publicFits` through `public_blocks ≤ blocks`. `norm` is PiRLC's guard
bound (`compression_bound`, `nightstream/src/lifecycle/finish.rs`). -/
structure KeyFacts (key : Layer1Input) : Prop where
  run : key.relation.key.layout.runVariables ≠ 0
  row : key.relation.key.layout.rowVariables ≠ 0
  slot : key.relation.key.layout.slotVariables + key.relation.matrixVars ≠ 0
  kappa : key.relation.kappa = 22
  blocks : key.relation.publicBlocks = 5
  point : key.relation.pointVariables = 28
  matrices : key.relation.matrices = 4
  root : key.relation.key.root.size = 4
  short : ∀ cl ∈ key.relation.key.layout.classes, cl.len ≤ Replay.D
  k1 : key.p1Key.batches.size = key.relation.p1Whir.batches.size
  k0 : key.p0Key.batches.size = key.relation.p0Whir.batches.size
  p1 : ∀ b ∈ key.relation.p1Whir.batches, b.size % 3 = 0
  p1count : (key.relation.p1Whir.batches.toList.map fun b => b.size / 3).sum = 2 * key.relation.queries + 4
  p0 : key.relation.p0Whir.batches.toList.map (·.size) =
    [1, 1, 2, 2 * key.relation.matrices, 1, 2 * key.relation.matrices + 1, 1]
  rowFits : key.relation.key.layout.rowVariables ≤ key.relation.pointVariables
  matricesFit : key.relation.matrices ≤ 2 ^ key.relation.matrixVars
  bits : EvalKBits key.relation
  publicFits : key.relation.publicBlocks ≤ 2 ^ key.relation.blockVars
  sizes : HonestSizes key.relation
  high : key.relation.setupVars + 1 ≤ 32
  checked : (structureOf key.relation.key.layout).Checked (shapeOf key.relation)
  norm : key.relation.normBound = 3672
  shape : (shapeOf key.relation).Sizes NightstreamFPrime.Spec.goldilocksModulus

/-- Each fact of `KeyFacts` by name, decided on `key`. -/
def keyChecks (key : Layer1Input) : List (String × Bool) :=
  let rel := key.relation
  let layout := rel.key.layout
  let lg := Nat.log2 (Replay.D * 2 ^ rel.blockVars).nextPowerOfTwo
  let s := structureOf layout
  [("run", decide (layout.runVariables ≠ 0)),
    ("row", decide (layout.rowVariables ≠ 0)),
    ("slot", decide (layout.slotVariables + rel.matrixVars ≠ 0)),
    ("kappa", decide (rel.kappa = 22)),
    ("blocks", decide (rel.publicBlocks = 5)),
    ("point", decide (rel.pointVariables = 28)),
    ("matrices", decide (rel.matrices = 4)),
    ("root", decide (rel.key.root.size = 4)),
    ("short", decide (∀ cl ∈ layout.classes, cl.len ≤ Replay.D)),
    ("k1", decide (key.p1Key.batches.size = rel.p1Whir.batches.size)),
    ("k0", decide (key.p0Key.batches.size = rel.p0Whir.batches.size)),
    ("p1", decide (∀ b ∈ rel.p1Whir.batches, b.size % 3 = 0)),
    ("p1count", decide ((rel.p1Whir.batches.toList.map fun b => b.size / 3).sum = 2 * rel.queries + 4)),
    ("p0", decide (rel.p0Whir.batches.toList.map (·.size) =
      [1, 1, 2, 2 * rel.matrices, 1, 2 * rel.matrices + 1, 1])),
    ("rowFits", decide (layout.rowVariables ≤ rel.pointVariables)),
    ("matricesFit", decide (rel.matrices ≤ 2 ^ rel.matrixVars)),
    ("bits", decide (rel.blockVars ≤ lg ∧ lg ≤ rel.pointVariables ∧ rel.key.blocks ≤ 2 ^ rel.blockVars ∧
      Replay.D * rel.key.blocks ≤ 2 ^ lg ∧ rel.key.blocks < 2 ^ lg)),
    ("publicFits", decide (rel.publicBlocks ≤ 2 ^ rel.blockVars)),
    ("sizes", decide (rel.matrices ≤ layout.runBounds.size ∧ 3 ≤ rel.setupVars)),
    ("high", decide (rel.setupVars + 1 ≤ 32)),
    ("checked", decide (s.runBound 0 = 0 ∧ (∀ j, j < rel.matrices → s.runBound j ≤ s.runBound (j + 1)) ∧
      (0 < s.segments → s.segmentFirst 0 = 0) ∧
      (∀ i, i < s.segments - 1 → s.segmentFirst i < s.segmentFirst (i + 1)) ∧
      (0 < s.segments → s.segmentFirst (s.segments - 1) < s.slots) ∧
      (∀ i, i < s.segments → s.segmentClass i < s.classes) ∧
      ∀ c, c < s.classes → 1 ≤ s.classLength c)),
    ("norm", decide (rel.normBound = 3672)),
    ("shape", decide (let sh := shapeOf rel; let N := NightstreamFPrime.Spec.goldilocksModulus
      5 ≤ N ∧
      2 ^ sh.cubeVars < N ∧
      2 ^ sh.runVars < N ∧
      2 ^ sh.rowVars ≤ N ∧
      2 ^ (sh.slotVars + sh.matrixVars) ≤ N ∧
      2 ^ sh.blockVars + 1 < N ∧
      sh.slotVars ≤ sh.runVars ∧
      sh.matrices ≤ 2 ^ sh.matrixVars ∧
      sh.publicBlocks ≤ sh.blocks ∧
      sh.D ≤ 2 ^ sh.laneVars ∧
      sh.blocks ≤ 2 ^ sh.blockVars ∧
      1 ≤ sh.cubeVars ∧
      1 ≤ sh.runVars ∧
      1 ≤ sh.rowVars ∧
      1 ≤ sh.slotVars + sh.matrixVars ∧
      1 ≤ sh.blockVars ∧
      3 ≤ sh.setupVars ∧
      sh.laneVars + sh.blockVars ≤ sh.setupVars ∧
      sh.runVars ≤ sh.setupVars))]

/-- The name of the first fact of `KeyFacts` that `key` fails. -/
def keyFailure (key : Layer1Input) : Option String :=
  ((keyChecks key).find? fun check => !check.2).map (·.1)

/-- A key that fails no check has `KeyFacts`. -/
theorem keyFacts_of_check {key : Layer1Input} (pass : keyFailure key = none) : KeyFacts key := by
  have all : ∀ check ∈ keyChecks key, check.2 = true := by
    simpa [keyFailure, List.find?_eq_none] using pass
  simp only [keyChecks, List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq,
    decide_eq_true_eq] at all
  obtain ⟨run, row, slot, kappa, blocks, point, matrices, root, short, k1, k0, p1, p1count, p0, rowFits,
    matricesFit, ⟨deep, long, blocks_le, fits, strict⟩, publicFits, ⟨bounds, setup⟩, high,
    ⟨run_zero, run_mono, segment_zero, segment_mono, segment_last, class_named, length_pos⟩, norm,
    ⟨big_nodes, big_cube, big_runs, big_rows, big_pairs, big_blocks, slot_le_run, matrices_le, public_le, D_le, blocks_le, deep_cube, deep_runs, deep_rows, deep_pairs, deep_blocks, fold_three, fold_key, fold_run⟩⟩ := all
  refine ⟨run, row, slot, kappa, blocks, point, matrices, root, short, k1, k0, p1, p1count, p0, rowFits,
    matricesFit, ⟨deep, long, blocks_le, fits, strict⟩, publicFits, ⟨bounds, setup⟩, high,
    ⟨run_zero, run_mono, segment_zero, fun i lt => segment_mono i (by omega), fun i last => ?_, class_named,
      length_pos⟩, norm, ⟨big_nodes, big_cube, big_runs, big_rows, big_pairs, big_blocks, slot_le_run, matrices_le, public_le, D_le, blocks_le, deep_cube, deep_runs, deep_rows, deep_pairs, deep_blocks, fold_three, fold_key, fold_run⟩⟩
  rw [show i = (structureOf key.relation.key.layout).segments - 1 by omega]
  exact segment_last (by omega)

/-- The key has the `⌈lw / 54⌉` blocks of an application of logical width
`lw` (`Key::derive`, `neo-spartan/src/lib.rs`). Unlike `KeyFacts`, this fact
names one application, so a key of another application, such as the toy key
of `lowering_matches_lean`, fails it. -/
def WidthFits (key : Layer1Input) (lw : ℕ) : Prop :=
  key.relation.key.blocks = (lw + 53) / 54

instance (key : Layer1Input) (lw : ℕ) : Decidable (WidthFits key lw) :=
  inferInstanceAs (Decidable (_ = _))

/-- The logical width of the Poseidon2 hash-chain application, the
application of the production key (F′'s
`Poseidon2HashChainV1Package.logicalWidth`). -/
def poseidon2Width : ℕ := 43963750

/-- `WidthFits` at `poseidon2Width`, decided on `key`. -/
def poseidon2Fits (key : Layer1Input) : Bool :=
  decide (WidthFits key poseidon2Width)

end CompressionLean.Checks
