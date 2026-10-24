import Lean.Data.Json
import CompressionLean.Ports.Replay.Transcript

/-!
Owns the data the replay reads from the Rust export (`neo-spartan`'s
`tests/internal/replay.rs`): the relation (the key, the sizes, the WHIR
profiles and the WHIR transcript constants), one statement and one layer-1
proof per variant. Words are parsed as canonical Goldilocks values, which is
the only validation at this boundary; `Stages.readProof` checks the proof's
shape as `ProofView::read` does.

Also owns what the verifier derives from the relation: the sizes of `lib.rs`
(`Shape`, `Key`), the GKR tree shapes (`matrix.rs`) and the transcript words
of the relation and of the statement (`Relation::words`, `ClaimWords::words`).
-/

namespace CompressionLean.Replay

open Lean NightstreamFPrime.Spec

instance : FromJson Word where
  fromJson? json := do
    let value ← json.getNat?
    if value < goldilocksModulus then pure (Word.ofNat value)
    else throw s!"word {value} is not a canonical Goldilocks value"

instance : FromJson Ext where
  fromJson? json := do
    let words : Array Word ← fromJson? json
    if h : words.size = 3 then pure ⟨words[0], words[1], words[2]⟩
    else throw "an extension value has three words"

/-- The ring degree (`neo_math::D`). `ports.json` names the alias in
`ring.rs`; the value is in an inline module of `neo-params`, so the Rust test
`relation_shape_has_the_lean_constants` (`neo-spartan`) pins it at 54. -/
def D : Nat := 54
def laneVars : Nat := 6
/-- Variables folded per proof (`setup::FOLD`). -/
def foldVars : Nat := 3
/-- Setup oracles after the key rows (`RUN_ORACLES`). -/
def runOracles : Nat := 4

/-- `(n.next_power_of_two().trailing_zeros()).max(1)` (`block_variables`). -/
def bitsAtLeastOne (n : Nat) : Nat := max 1 (Nat.log2 n.nextPowerOfTwo)

/-- A run shape: its length and its ratio (`matrix::Class`). -/
structure RunClass where
  len : Nat
  ratio : Word
deriving FromJson, Inhabited

/-- Slots from `first` share one class and one lane (`matrix::Segment`). -/
structure Segment where
  first : Nat
  classIndex : Nat
  lane : Nat
deriving FromJson, Inhabited

/-- The matrices as the verifier knows them (`matrix::Structure`). -/
structure Layout where
  matrices : Nat
  rowVariables : Nat
  runVariables : Nat
  slotVariables : Nat
  runBounds : Array Nat
  classes : Array RunClass
  segments : Array Segment
  slots : Nat
deriving FromJson, Inhabited

/-- The verifier key (`lib.rs::Key`). -/
structure Key where
  root : Array Word
  blocks : Nat
  rows : Nat
  layout : Layout
deriving FromJson, Inhabited

/-- One quadratic sum-check of a WHIR run and the separator it absorbs first. -/
structure SumcheckShape where
  rounds : Nat
  seed : Array Word
deriving FromJson, Inhabited

structure WhirRoundShape where
  oodSamples : Nat
  queryDraws : Nat
  sumcheck : SumcheckShape
deriving FromJson, Inhabited

/-- One opening batch: its separator and its column count. -/
structure WhirBatch where
  seed : Array Word
  size : Nat
deriving FromJson, Inhabited

/-- The transcript of one WHIR opening (`whir::verify`) as fixed constants:
every separator it absorbs and every count that fixes its draws. -/
structure WhirShape where
  virtualSeed : Array Word
  oodSamples : Nat
  batches : Array WhirBatch
  whirSeed : Array Word
  batchingSeed : Array Word
  initialSumcheck : SumcheckShape
  rounds : Array WhirRoundShape
  finalPolyLen : Nat
  finalQueryDraws : Nat
  finalSumcheck : SumcheckShape
deriving FromJson, Inhabited

/-- One fixed relation (`lib.rs::Relation`). -/
structure Relation where
  key : Key
  publicBlocks : Nat
  pointVariables : Nat
  normBound : Nat
  kappa : Nat
  queries : Nat
  p0Profile : Array Word
  p1Profile : Array Word
  /-- The separator `whir::observe_root` absorbs before a root. -/
  commitmentSeed : Array Word
  p0Whir : WhirShape
  p1Whir : WhirShape
deriving FromJson, Inhabited

/-- The CE(B) statement words (`ClaimWords`); a `K` value is `(re, im)`. -/
structure Statement where
  commitment : Array (Array Word)
  publicInput : Array (Array Word)
  point : Array (Word × Word)
  evalK : Array (Word × Word)
  evalA : Array (Array (Word × Word))
deriving FromJson, Inhabited

structure GkrLayer where
  rounds : Array (Array Ext)
  children : Array (Array Ext)
deriving FromJson, Inhabited

structure GkrProof where
  roots : Array (Ext × Ext)
  layers : Array GkrLayer
deriving FromJson, Inhabited

/-- The prover's early values (`matrix::EarlyValues`). -/
structure EarlyValues where
  eRe : Ext
  eIm : Ext
  row : Ext
  pairs : Array (Ext × Ext)
  block : Ext
deriving FromJson, Inhabited

/-- One queried setup leaf (`setup::Leaf`). -/
structure Leaf where
  values : Array Word
  path : Array (Array Word)
deriving FromJson, Inhabited

structure WhirRound where
  root : Array Word
  ood : Array Ext
  sumcheck : Array (Ext × Ext)
deriving FromJson, Inhabited

/-- The parts of one WHIR opening that its transcript absorbs. -/
structure Opening where
  initialOod : Array Ext
  evals : Array (Array Ext)
  initialSumcheck : Array (Ext × Ext)
  rounds : Array WhirRound
  finalPoly : Array Ext
  finalSumcheck : Array (Ext × Ext)
deriving FromJson, Inhabited

/-- A layer-1 proof (`lib.rs::Proof`), without the WHIR query rows and paths. -/
structure Proof where
  p0 : Array Word
  histogram : Array Word
  early : GkrProof
  earlyValues : EarlyValues
  quotient : Array Ext
  linear : Array (Array Ext)
  finals : Array Ext
  partials : Array (Array Ext)
  p1 : Array Word
  leaves : Array Leaf
  late : GkrProof
  lateValues : Array (Ext × Ext)
  p1Opening : Opening
  p0Opening : Opening
deriving FromJson, Inhabited

/-- One input to replay: the layer-1 seed, a statement and a proof. -/
structure Variant where
  name : String
  seed : Array Word
  statement : Statement
  proof : Proof
deriving FromJson, Inhabited

structure Export where
  relation : Relation
  variants : Array Variant
deriving FromJson, Inhabited

/-- What the verifier knows of a GKR tree before the proof (`gkr::TreeShape`). -/
structure TreeShape where
  depth : Nat
  factors : Nat
deriving Inhabited

namespace Relation

variable (rel : Relation)

def matrices : Nat := rel.key.layout.matrices
def blockVars : Nat := bitsAtLeastOne rel.key.blocks
def cubeVars : Nat := rel.blockVars + laneVars
/-- Every setup oracle has this many variables (`Key::setup_variables`). -/
def setupVars : Nat := max rel.cubeVars rel.key.layout.runVariables
def matrixVars : Nat := Nat.log2 rel.matrices.nextPowerOfTwo
/-- Commitment rows, Eval_K (2), Eval_A (2 per matrix), public blocks (`ring::row_count`). -/
def rowCount : Nat := rel.kappa + 2 + 2 * rel.matrices + rel.publicBlocks
/-- The norm tree over the cube, then the four matrix early trees. -/
def earlyShapes : Array TreeShape :=
  let l := rel.key.layout
  #[⟨rel.cubeVars, 0⟩, ⟨l.runVariables, 0⟩, ⟨l.rowVariables, 1⟩, ⟨l.runVariables, 2⟩,
    ⟨l.slotVariables + rel.matrixVars, 1⟩]

/-- The block pairs, then the blocks (`Structure::late_shapes`). -/
def lateShapes : Array TreeShape :=
  #[⟨rel.key.layout.slotVariables + 1, 2⟩, ⟨rel.blockVars, 1⟩]

def layoutWords : Array Word :=
  let l := rel.key.layout
  (#[l.matrices, l.rowVariables, l.runVariables, l.slotVariables, l.classes.size, l.segments.size, l.slots] ++
      l.runBounds).map Word.ofNat ++
    l.classes.flatMap (fun c => #[Word.ofNat c.len, c.ratio]) ++
    l.segments.flatMap (fun s => #[s.first, s.classIndex, s.lane].map Word.ofNat)

/-- The relation as transcript words: the shape, the key, both WHIR profiles
and the setup query count (`Relation::words`). -/
def words : Array Word :=
  #[rel.cubeVars, rel.key.blocks, rel.key.rows, rel.matrices, rel.kappa, rel.publicBlocks,
      rel.pointVariables, rel.normBound].map Word.ofNat ++
    rel.key.root ++ #[rel.key.blocks, rel.key.rows].map Word.ofNat ++ rel.layoutWords ++
    rel.p0Profile ++ rel.p1Profile ++ #[Word.ofNat rel.queries]

end Relation

/-- The statement as transcript words: the ring rows, then the `K` values
(`ClaimWords::words`). -/
def Statement.words (st : Statement) : Array Word :=
  st.commitment.flatten ++ st.publicInput.flatten ++
    (st.point ++ st.evalK ++ st.evalA.flatten).flatMap fun value => #[value.1, value.2]

/-- The statement has the relation's shape (`ClaimWords::check_shape`, and the
fixed ring sizes of `ClaimWords`). -/
def Statement.fits (st : Statement) (rel : Relation) : Bool :=
  st.commitment.size == rel.kappa && st.commitment.all (·.size == D) &&
    st.publicInput.size == rel.publicBlocks && st.publicInput.all (·.size == D) &&
    st.point.size == rel.pointVariables && st.evalK.size == D &&
    st.evalA.size == rel.matrices && st.evalA.all (·.size == D)

end CompressionLean.Replay
