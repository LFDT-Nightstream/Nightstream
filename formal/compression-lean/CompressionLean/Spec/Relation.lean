import CompressionLean.Ports.Replay.Stages
import CompressionLean.Fields
import CompressionLean.Layer1.KeyStructure

namespace CompressionLean.Spec

open CompressionLean.Replay CompressionLean.Layer1

variable {K : Type} [Field K] [Algebra Gl K] [Fintype K] [ChallengeField K]

local notation "ι" => algebraMap Gl K
local notation "x₀" => ChallengeField.root (K := K)

def _root_.CompressionLean.Replay.Ext.toK (a : Ext) : K := ι a.c0.toF + ι a.c1.toF * x₀ + ι a.c2.toF * x₀ ^ 2

def pointK (p : Array Ext) : List K := p.toList.map Ext.toK

def tableK (a : Array Ext) (i : ℕ) : K := (a[i]!).toK

def pairK (v : Ext × Ext) : K × K := (v.1.toK, v.2.toK)

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

def statementOf (st : Replay.Statement) : Layer1.Statement Gl where
  commitment ρ l := ((st.commitment[ρ]!)[l]!).toF
  publicInput b l := ((st.publicInput[b]!)[l]!).toF
  r := st.point.toList.map fun v => (v.1.toF, v.2.toF)
  evalK l := ((st.evalK[l]!).1.toF, (st.evalK[l]!).2.toF)
  evalA j l := (((st.evalA[j]!)[l]!).1.toF, ((st.evalA[j]!)[l]!).2.toF)

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

def barOf (p l : ℕ) : Gl := (Replay.bar p l).toF

/-- Plonky3's `two_adic_generator(bits)`. -/
def twoAdic (bits : ℕ) : Gl := (0x185629dcda58878c : Gl) ^ (2 ^ (32 - bits))

def leafY (n i : ℕ) : Gl := 7 * twoAdic (n + 1) ^ i

def queryPoint (n i : ℕ) : Gl := 7 ^ 8 * twoAdic (n - 2) ^ i

def queryAt (rel : Replay.Relation) (i : ℕ) : K := ι (queryPoint rel.setupVars i)

def setupOracle (rel : Replay.Relation) (S : Setup Gl) (o x : ℕ) : Gl :=
  if o < rel.kappa then S.key o x
  else if o = rel.kappa then (S.runRow x : Gl)
  else if o = rel.kappa + 1 then (S.runSlot x : Gl)
  else if o = rel.kappa + 2 then S.runCoef x
  else if x < S.slots then (S.slotBlock x : Gl) else 0

end CompressionLean.Spec
