import CompressionLean.Spec.Relation
import CompressionLean.Ports.Circuit.Verifier
import CompressionLean.Ports.Setup
import CompressionLean.Production
import CompressionLean.Layer1.Soundness
import NightstreamFPrime.Export.PackageExpansion
import NightstreamFPrime.Export.OptionB.StepRelation
import Std.Data.HashSet
import NightstreamFPrime.Layout.OptionB.LightAssemblerCompleteness
import NightstreamFPrime.Layout.OptionB.RefreshAssemblerCompleteness
import NightstreamFPrime.Layout.OptionB.StepCircuits
import NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.OutputBinding
import NightstreamFPrime.Layout.ProductionRelation.CcsOpening
import NightstreamFPrime.Lifecycle.Nebula.FactorWords
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.Completeness
import NightstreamFPrime.Spec
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.GoldilocksCausal
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetConvention
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork
import NightstreamFPrime.Spec.Nebula

namespace CompressionLean.Spec

open CompressionLean.Setup
open NightstreamFPrime.Spec NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Export.Package
open CompressionLean.Replay (D)
open CompressionLean.Circuit (Layer1Key WhirKey)

/-- `List.decidableBAll` recurses once per element, so `setupCheck` overflows the stack on the
38,533,993 production runs. At runtime this decides the same proposition with `List.all`. -/
def decidableBAllTR {α : Type _} (p : α → Prop) [DecidablePred p] (l : List α) :
    Decidable (∀ x ∈ l, p x) :=
  decidable_of_iff (l.all fun x => decide (p x)) (by simp)

@[csimp] theorem decidableBAll_eq_decidableBAllTR : @List.decidableBAll = @decidableBAllTR := by
  funext α p _ l
  exact Subsingleton.elim _ _

structure Fact where
  name : String
  holds : Prop
  [decidable : Decidable holds]

attribute [instance] Fact.decidable

def firstFailure (facts : List Fact) : Option String :=
  (facts.find? fun f => !decide f.holds).map (·.name)

structure EvalKBits (rel : Replay.Relation) : Prop where
  deep : rel.blockVars ≤ Nat.log2 (D * 2 ^ rel.blockVars).nextPowerOfTwo
  long : Nat.log2 (D * 2 ^ rel.blockVars).nextPowerOfTwo ≤ rel.pointVariables
  blocks_le : rel.key.blocks ≤ 2 ^ rel.blockVars
  fits : D * rel.key.blocks ≤ 2 ^ Nat.log2 (D * 2 ^ rel.blockVars).nextPowerOfTwo
  strict : rel.key.blocks < 2 ^ Nat.log2 (D * 2 ^ rel.blockVars).nextPowerOfTwo

instance (rel : Replay.Relation) : Decidable (EvalKBits rel) :=
  decidable_of_iff (_ ∧ _ ∧ _ ∧ _ ∧ _) ⟨fun ⟨a, b, c, d, e⟩ => ⟨a, b, c, d, e⟩, fun ⟨a, b, c, d, e⟩ => ⟨a, b, c, d, e⟩⟩

structure HonestSizes (rel : Replay.Relation) : Prop where
  bounds : rel.matrices ≤ rel.key.layout.runBounds.size
  setup : 3 ≤ rel.setupVars

instance (rel : Replay.Relation) : Decidable (HonestSizes rel) :=
  decidable_of_iff (_ ∧ _) ⟨fun ⟨a, b⟩ => ⟨a, b⟩, fun ⟨a, b⟩ => ⟨a, b⟩⟩

structure RelationFacts (rel : Replay.Relation) : Prop where
  rowFits : rel.key.layout.rowVariables ≤ rel.pointVariables
  matricesFit : rel.matrices ≤ 2 ^ rel.matrixVars
  bits : EvalKBits rel
  publicFits : rel.publicBlocks ≤ 2 ^ rel.blockVars
  sizes : HonestSizes rel
  high : rel.setupVars + 1 ≤ 32
  checked : (structureOf rel.key.layout).Checked (shapeOf rel)

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

structure ProductionConstants (rel : Replay.Relation) : Prop where
  kappa : rel.kappa = Production.kappa
  blocks : rel.publicBlocks = Production.publicBlocks
  point : rel.pointVariables = Production.pointVariables
  matrices : rel.matrices = Production.matrices
  norm : rel.normBound < Production.normLimit

structure KeyFacts (key : Layer1Key) : Prop where
  circuit : CircuitFacts key.relation key.p1Key key.p0Key
  relation : RelationFacts key.relation
  constants : ProductionConstants key.relation
  shape : (shapeOf key.relation).Sizes NightstreamFPrime.Spec.goldilocksModulus

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
    ⟨"norm", rel.normBound < Production.normLimit⟩,
    ⟨"shape", (shapeOf rel).Sizes NightstreamFPrime.Spec.goldilocksModulus⟩]

def keyFailure (key : Layer1Key) : Option String :=
  firstFailure (keyChecks key)

/-- The key commits at most the blocks of `lw`. Option B's light key commits fewer blocks than its
key width, and its witness is zero past them (`Final.parent_holds`). -/
def KeyWithin (key : Layer1Key) (lw : ℕ) : Prop :=
  key.relation.key.blocks ≤ NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81ColumnLayout.blockCount lw

instance (key : Layer1Key) (lw : ℕ) : Decidable (KeyWithin key lw) :=
  inferInstanceAs (Decidable (_ ≤ _))

def optionBFits (key : Layer1Key) : Bool :=
  decide (KeyWithin key NightstreamFPrime.Export.OptionB.StepRelation.keyWidth)

def sortedValues (f : Push → ℕ) (pushes : List Push) : Array ℕ :=
  ((pushes.foldl (fun set p => set.insert (f p)) (∅ : Std.HashSet ℕ)).toList.mergeSort).toArray

/-- The layer-1 setup that `build` gives from `pushes` over `columns`. -/
def setupAt (seed : List ℕ) (columns : ℕ) (pushes : List Push) : Layer1.Setup Gl :=
  setupOfRuns seed (columns / ringDegree) (runList Production.matrices pushes)

/-- The columns of a key's relation: Rust's `Key::derive` gives `blocks = columns / D`. -/
def keyColumns (key : Layer1Key) : ℕ := ringDegree * key.relation.key.blocks

def setupChecksAt (columns : ℕ) (key : Layer1Key) (pushes : List Push) : List Fact :=
  let rel := key.relation
  let layout := rel.key.layout
  let st := structureOf layout
  let m := Production.matrices
  let classes := (sortedValues Push.classCode pushes).toList
  let keys := sortedValues (slotKey classes) pushes
  [⟨"columns", columns % ringDegree = 0 ∧ 0 < columns / ringDegree ∧ columns / ringDegree ≤ 4708530⟩,
    ⟨"runs", ∀ p ∈ pushes, p.matrix < m ∧ p.len ≤ ringDegree ∧ p.start + p.len ≤ columns ∧
      p.row < 2 ^ layout.rowVariables⟩,
    ⟨"matrices", layout.matrices = m ∧ m ≤ 2 ^ rel.matrixVars⟩,
    ⟨"bounds", ∀ j, j < m + 1 → layout.runBounds[j]! = pushes.countP fun p => p.matrix < j⟩,
    ⟨"run count", layout.runBounds[m]! ≤ 2 ^ layout.runVariables⟩,
    ⟨"blocks", columns / ringDegree = rel.key.blocks⟩,
    ⟨"slots", keys.size = layout.slots ∧ keys.size ≤ 2 ^ layout.slotVariables⟩,
    ⟨"cover", layout.segments.size = 0 → layout.slots = 0⟩,
    ⟨"slot sizes", ∀ w ∈ (0 : ℕ) :: keys.toList, codeLength (keyCode classes w) ≤ ringDegree ∧
      ringDegree * (keyStart w / ringDegree) + keyStart w % ringDegree + codeLength (keyCode classes w) ≤
        ringDegree * rel.key.blocks⟩,
    ⟨"slot shapes", ∀ i, i < st.segments → ∀ t, t < st.segmentStart (i + 1) - st.segmentStart i →
      let w := keys.getD (st.segmentStart i + t) 0
      codeLength (keyCode classes w) = st.classLength (st.segmentClass i) ∧
        codeRatio (keyCode classes w) = st.classRatio (st.segmentClass i) ∧ keyStart w % ringDegree = st.segmentLane i⟩]

def setupFailureAt (columns : ℕ) (key : Layer1Key) (pushes : List Push) : Option String :=
  firstFailure (setupChecksAt columns key pushes)

structure SetupColumns where
  row : Array Gl
  slot : Array Gl
  coef : Array Gl
  block : Array Gl

def columnChecks (pushes : List Push) (cols : SetupColumns) : List Fact :=
  let runs := (runList Production.matrices pushes).toArray
  let classes := (sortedValues Push.classCode pushes).toList
  let keys := sortedValues (slotKey classes) pushes
  [⟨"column sizes", cols.row.size = runs.size ∧ cols.slot.size = runs.size ∧ cols.coef.size = runs.size ∧
      cols.block.size = keys.size⟩,
    ⟨"rows", ∀ k, k < runs.size → cols.row.getD k 0 = ((runs.getD k Push.pad).row : Gl)⟩,
    ⟨"slots", ∀ k, k < runs.size → (cols.slot.getD k 0).val < keys.size ∧
      keys.getD (cols.slot.getD k 0).val 0 = slotKey classes (runs.getD k Push.pad)⟩,
    ⟨"coefficients", ∀ k, k < runs.size → cols.coef.getD k 0 = (runs.getD k Push.pad).initial⟩,
    ⟨"blocks", ∀ σ, σ < keys.size → cols.block.getD σ 0 = ((keyStart (keys.getD σ 0) / ringDegree : ℕ) : Gl)⟩]

def columnsFailure (pushes : List Push) (cols : SetupColumns) : Option String :=
  firstFailure (columnChecks pushes cols)

end CompressionLean.Spec
