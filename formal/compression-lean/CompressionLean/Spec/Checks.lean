import CompressionLean.Spec.Relation
import CompressionLean.Ports.Circuit.Verifier
import CompressionLean.Ports.Setup
import CompressionLean.Production
import CompressionLean.Layer1.Soundness
import NightstreamFPrime.Export.PackageExpansion
import Std.Data.HashSet

/-!
Owns the facts that the claims take about Rust's files, and the checks that
the executables run to decide them. A check is a list of named decidable
facts (`Fact`); `firstFailure` names the first that fails.

- The layer-1 key: `KeyFacts`, with its parts `CircuitFacts`,
  `RelationFacts` (with `EvalKBits` and `HonestSizes`) and
  `ProductionConstants`, decided by `keyFailure`; `WidthFits` at an
  application's width, decided at the Poseidon2 hash chain by
  `poseidon2Fits`.
- The setup that Rust builds: `setupFailure` on the key and the runs, and
  the setup that `build` returns for them (`poseidon2Setup`).
- The terminal package: `templatesNamed`.

The proofs that a pass gives each premise are in `Checks/`.
-/

namespace CompressionLean.Spec

open CompressionLean.Setup
open NightstreamFPrime.Spec NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Export.Package
open CompressionLean.Replay (D)
open CompressionLean.Circuit (Layer1Key WhirKey)

/-! ## Checks -/

/-- A fact of a check: its name and a decidable proposition. -/
structure Fact where
  name : String
  holds : Prop
  [decidable : Decidable holds]

attribute [instance] Fact.decidable

/-- The name of the first fact that fails. -/
def firstFailure (facts : List Fact) : Option String :=
  (facts.find? fun f => !decide f.holds).map (·.name)

/-! ## The layer-1 key -/

/-- The relation's Eval_K bits: they cover the block bits and the blocks
and fit the point `r` (`eval_k`'s coordinate bits). -/
structure EvalKBits (rel : Replay.Relation) : Prop where
  deep : rel.blockVars ≤ Nat.log2 (D * 2 ^ rel.blockVars).nextPowerOfTwo
  long : Nat.log2 (D * 2 ^ rel.blockVars).nextPowerOfTwo ≤ rel.pointVariables
  blocks_le : rel.key.blocks ≤ 2 ^ rel.blockVars
  fits : D * rel.key.blocks ≤ 2 ^ Nat.log2 (D * 2 ^ rel.blockVars).nextPowerOfTwo
  strict : rel.key.blocks < 2 ^ Nat.log2 (D * 2 ^ rel.blockVars).nextPowerOfTwo

instance (rel : Replay.Relation) : Decidable (EvalKBits rel) :=
  decidable_of_iff (_ ∧ _ ∧ _ ∧ _ ∧ _) ⟨fun ⟨a, b, c, d, e⟩ => ⟨a, b, c, d, e⟩, fun ⟨a, b, c, d, e⟩ => ⟨a, b, c, d, e⟩⟩

/-- The relation's sizes the honest fold reads. -/
structure HonestSizes (rel : Replay.Relation) : Prop where
  bounds : rel.matrices ≤ rel.key.layout.runBounds.size
  setup : 3 ≤ rel.setupVars

instance (rel : Replay.Relation) : Decidable (HonestSizes rel) :=
  decidable_of_iff (_ ∧ _) ⟨fun ⟨a, b⟩ => ⟨a, b⟩, fun ⟨a, b⟩ => ⟨a, b⟩⟩

/-- The relation facts that the replay does not check: sizes, and the order
part of the key's structure. -/
structure RelationFacts (rel : Replay.Relation) : Prop where
  rowFits : rel.key.layout.rowVariables ≤ rel.pointVariables
  matricesFit : rel.matrices ≤ 2 ^ rel.matrixVars
  bits : EvalKBits rel
  publicFits : rel.publicBlocks ≤ 2 ^ rel.blockVars
  sizes : HonestSizes rel
  high : rel.setupVars + 1 ≤ 32
  checked : (structureOf rel.key.layout).Checked (shapeOf rel)

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
structure KeyFacts (key : Layer1Key) : Prop where
  circuit : CircuitFacts key.relation key.p1Key key.p0Key
  relation : RelationFacts key.relation
  constants : ProductionConstants key.relation
  shape : (shapeOf key.relation).Sizes NightstreamFPrime.Spec.goldilocksModulus

/-- Each fact of `KeyFacts` by name, in field order. -/
def keyChecks (key : Layer1Key) : List Fact :=
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
def keyFailure (key : Layer1Key) : Option String :=
  firstFailure (keyChecks key)

/-- The key has the `⌈lw / D⌉` blocks of an application of logical width
`lw` (`Key::derive`, `neo-spartan/src/lib.rs`). Unlike `KeyFacts`, this fact
names one application, so a key of another application, such as the toy key
of `lowering_matches_lean`, fails it. -/
def WidthFits (key : Layer1Key) (lw : ℕ) : Prop :=
  key.relation.key.blocks = NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81ColumnLayout.blockCount lw

instance (key : Layer1Key) (lw : ℕ) : Decidable (WidthFits key lw) :=
  inferInstanceAs (Decidable (_ = _))

/-- `WidthFits` at `Production.poseidon2Width`, decided on `key`. -/
def poseidon2Fits (key : Layer1Key) : Bool :=
  decide (WidthFits key Production.poseidon2Width)

/-! ## The setup -/

/-- The distinct values of `f` on the runs in increasing order, as
`Tables::from_rows` collects the classes (a `BTreeSet`) and the slot keys (a
sort and a `dedup`): a hash set of the values in one pass, then a sort of the
distinct ones. -/
def sortedValues (f : Push → ℕ) (pushes : List Push) : Array ℕ :=
  ((pushes.foldl (fun set p => set.insert (f p)) (∅ : Std.HashSet ℕ)).toList.mergeSort).toArray

/-- The columns of the Poseidon2 hash chain's carrier, `carrierWidth` at its
logical width: the `MatrixShape::columns` of the production circuit. -/
def poseidon2Columns : ℕ := Phi81CarrierLayout.carrierWidth Production.poseidon2Width

/-- The setup that `build` returns for `seed` and the runs at the Poseidon2
hash chain's carrier and F′'s four matrices (`Setup::build`). -/
def poseidon2Setup (seed : List ℕ) (pushes : List Push) : Layer1.Setup Gl :=
  setupOfRuns seed (poseidon2Columns / ringDegree) (runList Production.matrices pushes)

/-- Each fact of the check by name, on the key and the runs. -/
def setupChecks (key : Layer1Key) (pushes : List Push) : List Fact :=
  let rel := key.relation
  let layout := rel.key.layout
  let st := structureOf layout
  let m := Production.matrices
  let classes := (sortedValues Push.classCode pushes).toList
  let keys := sortedValues (slotKey classes) pushes
  [⟨"runs", ∀ p ∈ pushes, p.matrix < m ∧ p.len ≤ ringDegree ∧ p.start + p.len ≤ poseidon2Columns ∧
      p.row < 2 ^ layout.rowVariables⟩,
    ⟨"matrices", layout.matrices = m ∧ m ≤ 2 ^ rel.matrixVars⟩,
    ⟨"bounds", ∀ j, j < m + 1 → layout.runBounds[j]! = pushes.countP fun p => p.matrix < j⟩,
    ⟨"run count", layout.runBounds[m]! ≤ 2 ^ layout.runVariables⟩,
    ⟨"blocks", poseidon2Columns / ringDegree ≤ rel.key.blocks⟩,
    ⟨"slots", keys.size = layout.slots ∧ keys.size ≤ 2 ^ layout.slotVariables⟩,
    ⟨"cover", layout.segments.size = 0 → layout.slots = 0⟩,
    ⟨"slot sizes", ∀ w ∈ (0 : ℕ) :: keys.toList, codeLength (keyCode classes w) ≤ ringDegree ∧
      ringDegree * (keyStart w / ringDegree) + keyStart w % ringDegree + codeLength (keyCode classes w) ≤
        ringDegree * rel.key.blocks⟩,
    ⟨"slot shapes", ∀ i, i < st.segments → ∀ t, t < st.segmentStart (i + 1) - st.segmentStart i →
      let w := keys.getD (st.segmentStart i + t) 0
      codeLength (keyCode classes w) = st.classLength (st.segmentClass i) ∧
        codeRatio (keyCode classes w) = st.classRatio (st.segmentClass i) ∧ keyStart w % ringDegree = st.segmentLane i⟩]

/-- The name of the first fact of the check that the key and the runs fail. -/
def setupFailure (key : Layer1Key) (pushes : List Push) : Option String :=
  firstFailure (setupChecks key pushes)

/-! ## The terminal package -/

/-- Every compact row invocation names a template (`CompactTemplatesValid`),
decided. -/
def templatesNamed (package : CircuitPackage) : Bool :=
  package.compactRowInvocations.all fun invocation => (package.compactRowTemplates[invocation.templateIndex]?).isSome

end CompressionLean.Spec
