import CompressionLean.Ports.Circuit.Verifier
import CompressionLean.Bridge.Main
import CompressionLean.Production
import CompressionLean.Checks.Fact

/-!
Owns the facts about the layer-1 key that the final theorems assume
(`KeyFacts`, with its parts `CircuitFacts`, `Bridge.RelationFacts` and
`ProductionConstants`, and `WidthFits`), and their check on a key
(`keyFailure`, `poseidon2Fits`). The production key is read at run time, so
`lowerCheck layer1` and `terminalCheck` run the check on the key that they
read.
-/

namespace CompressionLean.Checks

open CompressionLean.Circuit (Layer1Input WhirKey)
open CompressionLean.Bridge (EvalKBits HonestSizes RelationFacts shapeOf structureOf)

/-- The facts of the key that the in-circuit verifier reads and the replay
does not check (`verify_ensures`): nonempty run, row and slot points, a
four-word Merkle root, classes no longer than a block, and WHIR keys with
the batches of the relation's openings. Rust's `Structure::check`
(`neo-spartan/src/matrix.rs:142`) enforces `short`. -/
structure CircuitFacts (rel : Replay.Relation) (p1Key p0Key : WhirKey) : Prop where
  run : rel.key.layout.runVariables ≠ 0
  row : rel.key.layout.rowVariables ≠ 0
  slot : rel.key.layout.slotVariables + rel.matrixVars ≠ 0
  root : rel.key.root.size = 4
  short : ∀ cl ∈ rel.key.layout.classes, cl.len ≤ Replay.D
  k1 : p1Key.batches.size = rel.p1Whir.batches.size
  k0 : p0Key.batches.size = rel.p0Whir.batches.size
  p1 : ∀ b ∈ rel.p1Whir.batches, b.size % 3 = 0
  p1count : (rel.p1Whir.batches.toList.map fun b => b.size / 3).sum = 2 * rel.queries + 4
  p0 : rel.p0Whir.batches.toList.map (·.size) = [1, 1, 2, 2 * rel.matrices, 1, 2 * rel.matrices + 1, 1]

/-- The relation has the production constants (`Production`). `norm` is
PiRLC's guard bound (`compression_bound`, `nightstream/src/lifecycle/finish.rs`). -/
structure ProductionConstants (rel : Replay.Relation) : Prop where
  kappa : rel.kappa = Production.kappa
  blocks : rel.publicBlocks = Production.publicBlocks
  point : rel.pointVariables = Production.pointVariables
  matrices : rel.matrices = Production.matrices
  norm : rel.normBound = Production.normBound

/-- The facts about the layer-1 key that the final theorems assume: the facts
that the in-circuit verifier reads, the relation facts of the bridge, the
production constants, and the size conditions of the layer-1 argument at the
Goldilocks modulus. Rust's `Structure::check` (`neo-spartan/src/matrix.rs:142`)
enforces `checked`. `Relation::new` (`neo-spartan/src/lib.rs:334`) enforces
`high`, and `publicFits` through `public_blocks ≤ blocks`. -/
structure KeyFacts (key : Layer1Input) : Prop where
  circuit : CircuitFacts key.relation key.p1Key key.p0Key
  relation : RelationFacts key.relation
  constants : ProductionConstants key.relation
  shape : (shapeOf key.relation).Sizes NightstreamFPrime.Spec.goldilocksModulus

/-- Each fact of `KeyFacts` by name, in field order. -/
def keyChecks (key : Layer1Input) : List Fact :=
  let rel := key.relation
  let layout := rel.key.layout
  [⟨"run", layout.runVariables ≠ 0⟩,
    ⟨"row", layout.rowVariables ≠ 0⟩,
    ⟨"slot", layout.slotVariables + rel.matrixVars ≠ 0⟩,
    ⟨"root", rel.key.root.size = 4⟩,
    ⟨"short", ∀ cl ∈ layout.classes, cl.len ≤ Replay.D⟩,
    ⟨"k1", key.p1Key.batches.size = rel.p1Whir.batches.size⟩,
    ⟨"k0", key.p0Key.batches.size = rel.p0Whir.batches.size⟩,
    ⟨"p1", ∀ b ∈ rel.p1Whir.batches, b.size % 3 = 0⟩,
    ⟨"p1count", (rel.p1Whir.batches.toList.map fun b => b.size / 3).sum = 2 * rel.queries + 4⟩,
    ⟨"p0", rel.p0Whir.batches.toList.map (·.size) = [1, 1, 2, 2 * rel.matrices, 1, 2 * rel.matrices + 1, 1]⟩,
    ⟨"rowFits", layout.rowVariables ≤ rel.pointVariables⟩,
    ⟨"matricesFit", rel.matrices ≤ 2 ^ rel.matrixVars⟩,
    ⟨"bits", EvalKBits rel⟩,
    ⟨"publicFits", rel.publicBlocks ≤ 2 ^ rel.blockVars⟩,
    ⟨"sizes", HonestSizes rel⟩,
    ⟨"high", rel.setupVars + 1 ≤ 32⟩,
    ⟨"checked", (structureOf layout).Checked (shapeOf rel)⟩,
    ⟨"kappa", rel.kappa = Production.kappa⟩,
    ⟨"blocks", rel.publicBlocks = Production.publicBlocks⟩,
    ⟨"point", rel.pointVariables = Production.pointVariables⟩,
    ⟨"matrices", rel.matrices = Production.matrices⟩,
    ⟨"norm", rel.normBound = Production.normBound⟩,
    ⟨"shape", (shapeOf rel).Sizes NightstreamFPrime.Spec.goldilocksModulus⟩]

/-- The name of the first fact of `KeyFacts` that `key` fails. -/
def keyFailure (key : Layer1Input) : Option String :=
  firstFailure (keyChecks key)

/-- A key that fails no check has `KeyFacts`. -/
theorem keyFacts_of_check {key : Layer1Input} (pass : keyFailure key = none) : KeyFacts key := by
  have all := holds_of_firstFailure pass
  simp only [keyChecks, List.forall_mem_cons, List.not_mem_nil, IsEmpty.forall_iff, implies_true, and_true] at all
  obtain ⟨run, row, slot, root, short, k1, k0, p1, p1count, p0, rowFits, matricesFit, bits, publicFits, sizes, high,
    checked, kappa, blocks, point, matrices, norm, shape⟩ := all
  exact ⟨⟨run, row, slot, root, short, k1, k0, p1, p1count, p0⟩,
    ⟨rowFits, matricesFit, bits, publicFits, sizes, high, checked⟩, ⟨kappa, blocks, point, matrices, norm⟩, shape⟩

/-- The key has the `⌈lw / D⌉` blocks of an application of logical width
`lw` (`Key::derive`, `neo-spartan/src/lib.rs`). Unlike `KeyFacts`, this fact
names one application, so a key of another application, such as the toy key
of `lowering_matches_lean`, fails it. -/
def WidthFits (key : Layer1Input) (lw : ℕ) : Prop :=
  key.relation.key.blocks = NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81ColumnLayout.blockCount lw

instance (key : Layer1Input) (lw : ℕ) : Decidable (WidthFits key lw) :=
  inferInstanceAs (Decidable (_ = _))

/-- `WidthFits` at `Production.poseidon2Width`, decided on `key`. -/
def poseidon2Fits (key : Layer1Input) : Bool :=
  decide (WidthFits key Production.poseidon2Width)

end CompressionLean.Checks
