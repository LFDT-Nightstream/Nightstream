import NightstreamFPrime.Lifecycle.Nebula.StepRows

/-! Owns the word layout of a memory-application step witness: the index of
every witness field in the application's witness words, and the decoding of a
word vector into a `StepWitness`. The circuit reads its witness wires and the
Rust prover writes its witness vector with this layout. It does not own any
row. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula

namespace Words

variable (p : Plan)

/-- Words of one operation slot: the lane bits of spec §6.3, then the O4
difference bits. -/
def opWords : ℕ := p.opWidth + p.wTs

def appIn (i : ℕ) : ℕ := i
def carryIn (i : ℕ) : ℕ := 2 + i
def carryOut (i : ℕ) : ℕ := 41 + i
def appOut (i : ℕ) : ℕ := 80 + i
def proposal (i : ℕ) : ℕ := 82 + i
def idle : ℕ := 90
def isOpen : ℕ := 91
def openInverse : ℕ := 92
def isClose : ℕ := 93
def closeInverse : ℕ := 94
def etaFresh (i : ℕ) : ℕ := 95 + i
def eta1Square (i : ℕ) : ℕ := 99 + i
def seenPrev (i : ℕ) : ℕ := 101 + i
def idxEff : ℕ := 113

/-- Word `k` of operation slot `j`. -/
def op (j k : ℕ) : ℕ := 114 + j * opWords p + k

def initialStart : ℕ := 114 + p.bOps * opWords p

/-- Word `k` of IS slot `j`. -/
def initial (j k : ℕ) : ℕ := initialStart p + j * p.scanWidth + k

def finalStart : ℕ := initialStart p + p.bScan * p.scanWidth

/-- Word `k` of FS slot `j`. -/
def final (j k : ℕ) : ℕ := finalStart p + j * p.scanWidth + k

def tsBits (k : ℕ) : ℕ := finalStart p + p.bScan * p.scanWidth + k
def segBits (k : ℕ) : ℕ := tsBits p p.wTs + k
def opsProducts (j c : ℕ) : ℕ := segBits p p.segWidth + 4 * j + c
def scanProducts (j c : ℕ) : ℕ := opsProducts p p.bOps 0 + 4 * j + c

/-- The number of witness words. -/
def count : ℕ := scanProducts p p.bScan 0

/-! Offsets inside an operation slot and a scan slot. -/

def opPad : ℕ := 0
def opIsWrite : ℕ := 1
def opIsRam : ℕ := 2
def opAddr (k : ℕ) : ℕ := 3 + k
def opVr (k : ℕ) : ℕ := 3 + p.μ + k
def opVw (k : ℕ) : ℕ := 3 + p.μ + 32 + k
def opRt (k : ℕ) : ℕ := 3 + p.μ + 64 + k
def opDiff (k : ℕ) : ℕ := p.opWidth + k
def scanValue (k : ℕ) : ℕ := k
def scanStamp (k : ℕ) : ℕ := 32 + k

end Words

variable {p : Plan}

/-- The operation slot `j` of a word vector. -/
def OpSlotBits.ofWords (p : Plan) (v : ℕ → F) (j : ℕ) : OpSlotBits p where
  pad := v (Words.op p j Words.opPad)
  isWrite := v (Words.op p j Words.opIsWrite)
  isRam := v (Words.op p j Words.opIsRam)
  addr k := v (Words.op p j (Words.opAddr k))
  vr k := v (Words.op p j (Words.opVr p k))
  vw k := v (Words.op p j (Words.opVw p k))
  rt k := v (Words.op p j (Words.opRt p k))
  diff k := v (Words.op p j (Words.opDiff p k))

/-- A scan slot of a word vector, at the slot's first word `base`. -/
def ScanSlotBits.ofWords (p : Plan) (v : ℕ → F) (base : ℕ) : ScanSlotBits p where
  value k := v (base + Words.scanValue k)
  stamp k := v (base + Words.scanStamp k)

/-- The step witness of a word vector. -/
def StepWitness.ofWords (p : Plan) (v : ℕ → F) : StepWitness p where
  appIn i := v (Words.appIn i)
  carryIn i := v (Words.carryIn i)
  carryOut i := v (Words.carryOut i)
  appOut i := v (Words.appOut i)
  proposal i := v (Words.proposal i)
  idle := v Words.idle
  isOpen := v Words.isOpen
  openInverse := v Words.openInverse
  isClose := v Words.isClose
  closeInverse := v Words.closeInverse
  etaFresh i := v (Words.etaFresh i)
  eta1Square i := v (Words.eta1Square i)
  seenPrev i := v (Words.seenPrev i)
  idxEff := v Words.idxEff
  ops j := OpSlotBits.ofWords p v j
  initial j := ScanSlotBits.ofWords p v (Words.initial p j 0)
  final j := ScanSlotBits.ofWords p v (Words.final p j 0)
  tsBits k := v (Words.tsBits p k)
  segBits k := v (Words.segBits p k)
  opsProducts j c := v (Words.opsProducts p j c)
  scanProducts j c := v (Words.scanProducts p j c)

end NightstreamFPrime.Lifecycle.Nebula
