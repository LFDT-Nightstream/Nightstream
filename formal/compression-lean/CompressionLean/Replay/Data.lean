import Lean.Data.Json
import CompressionLean.Replay.Transcript

namespace CompressionLean.Replay

open Lean NightstreamFPrime.Spec

instance : FromJson Gl where
  fromJson? json := do
    let value ← json.getNat?
    if value < goldilocksModulus then pure (Gl.ofNat value)
    else throw s!"word {value} is not a canonical Goldilocks value"

instance : FromJson Ext where
  fromJson? json := do
    let words : Array Gl ← fromJson? json
    if h : words.size = 3 then pure ⟨words[0], words[1], words[2]⟩
    else throw "an extension value has three words"

def D : Nat := 54
def laneVars : Nat := 6
def foldVars : Nat := 3
def runOracles : Nat := 4

def bitsAtLeastOne (n : Nat) : Nat := max 1 (Nat.log2 n.nextPowerOfTwo)

structure RunClass where
  len : Nat
  ratio : Gl
deriving FromJson, Inhabited

structure Segment where
  first : Nat
  classIndex : Nat
  lane : Nat
deriving FromJson, Inhabited

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

structure Key where
  root : Array Gl
  blocks : Nat
  rows : Nat
  layout : Layout
deriving FromJson, Inhabited

structure SumcheckShape where
  rounds : Nat
  seed : Array Gl
deriving FromJson, Inhabited

structure WhirRoundShape where
  oodSamples : Nat
  queryDraws : Nat
  sumcheck : SumcheckShape
deriving FromJson, Inhabited

structure WhirBatch where
  seed : Array Gl
  size : Nat
deriving FromJson, Inhabited

structure WhirShape where
  virtualSeed : Array Gl
  oodSamples : Nat
  batches : Array WhirBatch
  whirSeed : Array Gl
  batchingSeed : Array Gl
  initialSumcheck : SumcheckShape
  rounds : Array WhirRoundShape
  finalPolyLen : Nat
  finalQueryDraws : Nat
  finalSumcheck : SumcheckShape
deriving FromJson, Inhabited

structure Relation where
  key : Key
  publicBlocks : Nat
  pointVariables : Nat
  normBound : Nat
  kappa : Nat
  queries : Nat
  p0Profile : Array Gl
  p1Profile : Array Gl
  commitmentSeed : Array Gl
  p0Whir : WhirShape
  p1Whir : WhirShape
deriving FromJson, Inhabited

structure Statement where
  commitment : Array (Array Gl)
  publicInput : Array (Array Gl)
  point : Array (Gl × Gl)
  evalK : Array (Gl × Gl)
  evalA : Array (Array (Gl × Gl))
deriving FromJson, Inhabited

structure GkrLayer where
  rounds : Array (Array Ext)
  children : Array (Array Ext)
deriving FromJson, Inhabited

structure GkrProof where
  roots : Array (Ext × Ext)
  layers : Array GkrLayer
deriving FromJson, Inhabited

structure EarlyValues where
  eRe : Ext
  eIm : Ext
  row : Ext
  pairs : Array (Ext × Ext)
  block : Ext
deriving FromJson, Inhabited

structure Leaf where
  values : Array Gl
  path : Array (Array Gl)
deriving FromJson, Inhabited

structure WhirRound where
  root : Array Gl
  ood : Array Ext
  sumcheck : Array (Ext × Ext)
deriving FromJson, Inhabited

structure Opening where
  initialOod : Array Ext
  evals : Array (Array Ext)
  initialSumcheck : Array (Ext × Ext)
  rounds : Array WhirRound
  finalPoly : Array Ext
  finalSumcheck : Array (Ext × Ext)
deriving FromJson, Inhabited

structure Proof where
  p0 : Array Gl
  histogram : Array Gl
  early : GkrProof
  earlyValues : EarlyValues
  quotient : Array Ext
  linear : Array (Array Ext)
  finals : Array Ext
  partials : Array (Array Ext)
  p1 : Array Gl
  leaves : Array Leaf
  late : GkrProof
  lateValues : Array (Ext × Ext)
  p1Opening : Opening
  p0Opening : Opening
deriving FromJson, Inhabited

structure Variant where
  name : String
  seed : Array Gl
  statement : Statement
  proof : Proof
deriving FromJson, Inhabited

structure Export where
  relation : Relation
  variants : Array Variant
deriving FromJson, Inhabited

structure TreeShape where
  depth : Nat
  factors : Nat
deriving Inhabited

namespace Relation

variable (rel : Relation)

def matrices : Nat := rel.key.layout.matrices
def blockVars : Nat := bitsAtLeastOne rel.key.blocks
def cubeVars : Nat := rel.blockVars + laneVars
def setupVars : Nat := max rel.cubeVars rel.key.layout.runVariables
def matrixVars : Nat := Nat.log2 rel.matrices.nextPowerOfTwo
def rowCount : Nat := rel.kappa + 2 + 2 * rel.matrices + rel.publicBlocks
def earlyShapes : Array TreeShape :=
  let l := rel.key.layout
  #[⟨rel.cubeVars, 0⟩, ⟨l.runVariables, 0⟩, ⟨l.rowVariables, 1⟩, ⟨l.runVariables, 2⟩,
    ⟨l.slotVariables + rel.matrixVars, 1⟩]

def lateShapes : Array TreeShape :=
  #[⟨rel.key.layout.slotVariables + 1, 2⟩, ⟨rel.blockVars, 1⟩]

def layoutWords : Array Gl :=
  let l := rel.key.layout
  (#[l.matrices, l.rowVariables, l.runVariables, l.slotVariables, l.classes.size, l.segments.size, l.slots] ++
      l.runBounds).map Gl.ofNat ++
    l.classes.flatMap (fun c => #[Gl.ofNat c.len, c.ratio]) ++
    l.segments.flatMap (fun s => #[s.first, s.classIndex, s.lane].map Gl.ofNat)

def words : Array Gl :=
  #[rel.cubeVars, rel.key.blocks, rel.key.rows, rel.matrices, rel.kappa, rel.publicBlocks,
      rel.pointVariables, rel.normBound].map Gl.ofNat ++
    rel.key.root ++ #[rel.key.blocks, rel.key.rows].map Gl.ofNat ++ rel.layoutWords ++
    rel.p0Profile ++ rel.p1Profile ++ #[Gl.ofNat rel.queries]

end Relation

def Statement.words (st : Statement) : Array Gl :=
  st.commitment.flatten ++ st.publicInput.flatten ++
    (st.point ++ st.evalK ++ st.evalA.flatten).flatMap fun value => #[value.1, value.2]

def Statement.fits (st : Statement) (rel : Relation) : Bool :=
  st.commitment.size == rel.kappa && st.commitment.all (·.size == D) &&
    st.publicInput.size == rel.publicBlocks && st.publicInput.all (·.size == D) &&
    st.point.size == rel.pointVariables && st.evalK.size == D &&
    st.evalA.size == rel.matrices && st.evalA.all (·.size == D)

end CompressionLean.Replay
