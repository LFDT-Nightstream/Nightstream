import NightstreamFPrime.Lifecycle.Nebula.Carry
import NightstreamFPrime.Lifecycle.Nebula.Machine

/-! Owns the witness of one invocation of the first Nebula memory application
and the field equations that its circuit checks: the memory rows of spec §8,
the carry actions of §11.2 with the two arms of §12, the three record chains
and the `η` transcript of §9, the state digests of §11.1, and the machine's
port rows (§10). Every value the relation needs is a witness word, so
validity is exactly "these equations hold". The refinement to the typed model
(Ob4) and the honest witness (Ob5) are proved elsewhere. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula

/-- The witness of one invocation. Bit fields are field words that the rows
force into `{0, 1}`. -/
structure StepWitness (p : Plan) where
  appIn : Fin 2 → F
  carryIn : Fin 39 → F
  carryOut : Fin 39 → F
  appOut : Fin 2 → F
  proposal : Fin 8 → F
  idle : F
  isOpen : F
  openInverse : F
  isClose : F
  closeInverse : F
  etaFresh : Fin 4 → F
  eta1Square : Fin 2 → F
  seenPrev : Fin 12 → F
  idxEff : F
  opsBits : Fin p.bOps → Fin p.opWidth → F
  initialBits : Fin p.bScan → Fin p.scanWidth → F
  finalBits : Fin p.bScan → Fin p.scanWidth → F
  diffBits : Fin p.bOps → Fin p.wTs → F
  tsBits : Fin p.wTs → F
  segBits : Fin p.segWidth → F
  opsProducts : Fin p.bOps → Fin 4 → F
  scanProducts : Fin p.bScan → Fin 4 → F

/-- A field word is a bit. -/
def IsBit (x : F) : Prop := x = 0 ∨ x = 1

/-- The little-endian value of `width` bits from `start`, in the field. -/
def bitsWord {n : ℕ} (bits : Fin n → F) (start width : ℕ) : F :=
  ∑ k : Fin width, if h : start + k < n then (2 : F) ^ (k : ℕ) * bits ⟨start + k, h⟩ else 0

/-- A `K` element from two words. -/
def kOf (c0 c1 : F) : K := ⟨c0, c1⟩

/-- A field word as an element of `K`. -/
def embed (x : F) : K := ⟨x, 0⟩

/-- The fingerprint of spec §8.2 over trunk `K` operations. -/
def fingerprintK (η1 η2 η1sq : K) (t g v : F) : K :=
  K.sub (K.add (K.add (embed g) (K.mul η1 (embed v))) (K.mul η1sq (embed t))) η2

/-- The gated factor `pad + (1 − pad) · f` of rows O8 and O9. -/
def gatedK (pad : F) (f : K) : K := K.add (embed pad) (K.mul (embed (1 - pad)) f)

namespace StepWitness

variable {p : Plan} (w : StepWitness p)

/-! ### Carry fields -/

/-- Word `k` of a carry vector, or `0` beyond it. -/
def carryWord (v : Fin 39 → F) (k : ℕ) : F := if h : k < 39 then v ⟨k, h⟩ else 0

/-- The digest in carry words `start … start + 3`. -/
def carryDigest (v : Fin 39 → F) (start : ℕ) : Digest := fun i => carryWord v (start + i)

def cIn (k : ℕ) : F := carryWord w.carryIn k
def cOut (k : ℕ) : F := carryWord w.carryOut k

/-- `(η1, η2)` of the output carry: the challenges that this step uses. -/
def eta1 : K := kOf (w.cOut 3) (w.cOut 4)
def eta2 : K := kOf (w.cOut 5) (w.cOut 6)

/-- The square of `η1` that the witness supplies. -/
def eta1Sq : K := kOf (w.eta1Square 0) (w.eta1Square 1)

/-- The fresh challenges of the `η` transcript. -/
def freshEta : K × K := (kOf (w.etaFresh 0) (w.etaFresh 1), kOf (w.etaFresh 2) (w.etaFresh 3))

/-- The proposed ops and FS roots of the step's witness. -/
def proposalDigest (start : ℕ) : Digest := fun i =>
  if h : start + i < 8 then w.proposal ⟨start + i, h⟩ else 0

/-- The chain digest that link `lane` extends: `seenPrev` words. -/
def previousDigest (start : ℕ) : Digest := fun i =>
  if h : start + i < 12 then w.seenPrev ⟨start + i, h⟩ else 0

/-! ### Slot fields -/

/-- Bit `k` of operation slot `j`, or `0` beyond its width. -/
def opBit (j : Fin p.bOps) (k : ℕ) : F := if h : k < p.opWidth then w.opsBits j ⟨k, h⟩ else 0

def pad (j : Fin p.bOps) : F := w.opBit j 0
def isWrite (j : Fin p.bOps) : F := w.opBit j 1
def isRam (j : Fin p.bOps) : F := w.opBit j 2
def addr (j : Fin p.bOps) : F := bitsWord (w.opsBits j) 3 p.μ
def vr (j : Fin p.bOps) : F := bitsWord (w.opsBits j) (3 + p.μ) 32
def vw (j : Fin p.bOps) : F := bitsWord (w.opsBits j) (35 + p.μ) 32
def rt (j : Fin p.bOps) : F := bitsWord (w.opsBits j) (67 + p.μ) p.wTs
def diff (j : Fin p.bOps) : F := bitsWord (w.diffBits j) 0 p.wTs

/-- The active count of the slots before `k` (row O2): `cnt_{k−1}`. -/
def cntBefore (k : ℕ) : F := ∑ i : Fin p.bOps, if i.val < k then 1 - w.pad i else 0

/-- `wt_j = ts + cnt_j`. -/
def wt (j : Fin p.bOps) : F := w.cIn 2 + w.cntBefore (j.val + 1)

/-- `g_j = addr_j + is_ram_j · R`. -/
def globalIndex (j : Fin p.bOps) : F := w.addr j + w.isRam j * natWord p.romSize

def scanValue (bits : Fin p.bScan → Fin p.scanWidth → F) (j : Fin p.bScan) : F :=
  bitsWord (bits j) 0 32
def scanStamp (bits : Fin p.bScan → Fin p.scanWidth → F) (j : Fin p.bScan) : F :=
  bitsWord (bits j) 32 p.wTs

/-- The structural index `idx · B_scan + j` of scan slot `j`. -/
def scanIndex (j : Fin p.bScan) : F := w.idxEff * natWord p.bScan + natWord j

/-! ### Products -/

/-- The products at the start of the step: `1` after an open, the carried
products otherwise. Words `word, word + 1` of the input carry. -/
def startProduct (word : ℕ) : K :=
  K.add (embed w.isOpen) (K.mul (embed (1 - w.isOpen)) (kOf (w.cIn word) (w.cIn (word + 1))))

/-- Running product pair `pair` (0 read, 1 write) after operation slot `k − 1`;
`k = 0` is the start. -/
def opsProductAfter (pair : ℕ) (k : ℕ) : K :=
  if k = 0 then w.startProduct (7 + 2 * pair)
  else if h : k - 1 < p.bOps then
    kOf (w.opsProducts ⟨k - 1, h⟩ ⟨2 * pair % 4, Nat.mod_lt _ (by decide)⟩)
      (w.opsProducts ⟨k - 1, h⟩ ⟨(2 * pair + 1) % 4, Nat.mod_lt _ (by decide)⟩)
  else K.zero

/-- Running product pair `pair` (0 IS, 1 FS) after scan slot `k − 1`. -/
def scanProductAfter (pair : ℕ) (k : ℕ) : K :=
  if k = 0 then w.startProduct (11 + 2 * pair)
  else if h : k - 1 < p.bScan then
    kOf (w.scanProducts ⟨k - 1, h⟩ ⟨2 * pair % 4, Nat.mod_lt _ (by decide)⟩)
      (w.scanProducts ⟨k - 1, h⟩ ⟨(2 * pair + 1) % 4, Nat.mod_lt _ (by decide)⟩)
  else K.zero

/-! ### Lanes -/

/-- The bits of lane `bits`, slot-major (spec §6.3). -/
def laneBits {slots width : ℕ} (bits : Fin slots → Fin width → F) : List F :=
  (List.ofFn fun j => List.ofFn (bits j)).flatten

end StepWitness

/-- Little-endian value of a chunk of field bits. -/
def chunkWord : List F → F
  | [] => 0
  | b :: bs => b + 2 * chunkWord bs

/-- Spec §9.1 packing of field bits: chunks of 63, each read little-endian. -/
def packWords (bits : List F) : List F :=
  if _h : bits = [] then [] else chunkWord (bits.take 63) :: packWords (bits.drop 63)
termination_by bits.length
decreasing_by
  have : 0 < bits.length := List.length_pos_of_ne_nil _h
  simp only [List.length_drop]
  omega

/-- The four words of a Poseidon2 transcript digest over `blocks`. -/
def digestOfBlocks (blocks : List (List F)) : Digest := squeezeDigest (absorbed blocks)

/-- The four state words of spec §11.1 over raw application and carry words. -/
def stateWordsRaw (app carry : List F) : List F :=
  digestWords (digestOfBlocks [textWords "Nightstream/Nebula/v3/state", app, carry])

/-- A chain link of spec §9.2 over field words. -/
def chainLink (lane : Lane) (index : F) (previous : Digest) (packed : List F) : Digest :=
  digestOfBlocks [textWords (chainTag lane), index :: digestWords previous, packed]

/-- The header digests that an open resets the three chains to. -/
def headerWords (p : Plan) (i : ℕ) : F :=
  if i < 4 then hash (.header .ops (planDigest p)) ⟨i % 4, Nat.mod_lt _ (by decide)⟩
  else hash (.header .mem (planDigest p)) ⟨i % 4, Nat.mod_lt _ (by decide)⟩

/-- Every equation that the circuit of one memory-application invocation
checks, except the output state, which is the step function. `zIn` is the
input state. Indices follow spec §11.1 carry word order. -/
structure StepWitness.RowsHold {p : Plan} (w : StepWitness p) (zIn : List F) : Prop where
  -- O1, S1, and the auxiliary bits
  opsBits : ∀ j k, IsBit (w.opsBits j k)
  initialBits : ∀ j k, IsBit (w.initialBits j k)
  finalBits : ∀ j k, IsBit (w.finalBits j k)
  diffBits : ∀ j k, IsBit (w.diffBits j k)
  tsBits : ∀ k, IsBit (w.tsBits k)
  segBits : ∀ k, IsBit (w.segBits k)
  idleBit : IsBit w.idle
  openBit : IsBit w.isOpen
  closeBit : IsBit w.isClose
  -- §12 arm: reopen exactly when the input carry is closed
  openZero : (w.cIn 1 - natWord p.n) * w.isOpen = 0
  openTest : (w.cIn 1 - natWord p.n) * w.openInverse = 1 - w.isOpen
  -- §11.2 open: S_max, then reset to the proposals, fresh challenges, headers
  segRange : w.isOpen * (natWord (p.sMax - 1) - w.cIn 0 - bitsWord w.segBits 0 p.segWidth) = 0
  idxEff : w.idxEff = (1 - w.isOpen) * w.cIn 1
  seenPrev : ∀ i : Fin 12,
    w.seenPrev i = w.isOpen * headerWords p i + (1 - w.isOpen) * w.cIn (23 + i)
  proposed : ∀ i : Fin 8, w.cOut (15 + i) = w.isOpen * w.proposal i + (1 - w.isOpen) * w.cIn (15 + i)
  eta : ∀ i : Fin 4, w.cOut (3 + i) = w.isOpen * w.etaFresh i + (1 - w.isOpen) * w.cIn (3 + i)
  square : w.eta1Sq = K.mul w.eta1 w.eta1
  -- §8.1 operation rows
  readKeeps : ∀ j, (1 - w.isWrite j) * (w.vw j - w.vr j) = 0
  fresh : ∀ j, (1 - w.pad j) * (w.wt j - w.rt j - 1 - w.diff j) = 0
  noRomWrite : ∀ j, w.isWrite j * (1 - w.isRam j) = 0
  romRange : ∀ j k, p.r ≤ k → k < p.μ → (1 - w.isRam j) * w.opBit j (3 + k) = 0
  padZero : ∀ j k, 1 ≤ k → k < p.opWidth → w.pad j * w.opBit j k = 0
  readProduct : ∀ j, w.opsProductAfter 0 (j.val + 1) = K.mul (w.opsProductAfter 0 j.val)
    (gatedK (w.pad j) (fingerprintK w.eta1 w.eta2 w.eta1Sq (w.rt j) (w.globalIndex j) (w.vr j)))
  writeProduct : ∀ j, w.opsProductAfter 1 (j.val + 1) = K.mul (w.opsProductAfter 1 j.val)
    (gatedK (w.pad j) (fingerprintK w.eta1 w.eta2 w.eta1Sq (w.wt j) (w.globalIndex j) (w.vw j)))
  -- §8.3 scan rows
  initialProduct : ∀ j, w.scanProductAfter 0 (j.val + 1) = K.mul (w.scanProductAfter 0 j.val)
    (fingerprintK w.eta1 w.eta2 w.eta1Sq (StepWitness.scanStamp w.initialBits j) (w.scanIndex j)
      (StepWitness.scanValue w.initialBits j))
  finalProduct : ∀ j, w.scanProductAfter 1 (j.val + 1) = K.mul (w.scanProductAfter 1 j.val)
    (fingerprintK w.eta1 w.eta2 w.eta1Sq (StepWitness.scanStamp w.finalBits j) (w.scanIndex j)
      (StepWitness.scanValue w.finalBits j))
  -- §8.4 boundary rows
  tsOut : w.cOut 2 = w.cIn 2 + w.cntBefore p.bOps
  tsRange : w.cOut 2 = bitsWord w.tsBits 0 p.wTs
  idxOut : w.cOut 1 = w.idxEff + 1
  productsOut : kOf (w.cOut 7) (w.cOut 8) = w.opsProductAfter 0 p.bOps ∧
    kOf (w.cOut 9) (w.cOut 10) = w.opsProductAfter 1 p.bOps ∧
    kOf (w.cOut 11) (w.cOut 12) = w.scanProductAfter 0 p.bScan ∧
    kOf (w.cOut 13) (w.cOut 14) = w.scanProductAfter 1 p.bScan
  -- §11.2 close, exactly when the step sets idx = N
  closeZero : (w.cOut 1 - natWord p.n) * w.isClose = 0
  closeTest : (w.cOut 1 - natWord p.n) * w.closeInverse = 1 - w.isClose
  closeOps : ∀ i : Fin 4, w.isClose * (w.cOut (23 + i) - w.cOut (15 + i)) = 0
  closeInitial : ∀ i : Fin 4, w.isClose * (w.cOut (27 + i) - w.cIn (35 + i)) = 0
  closeFinal : ∀ i : Fin 4, w.isClose * (w.cOut (31 + i) - w.cOut (19 + i)) = 0
  closeProducts : K.mul (embed w.isClose)
    (K.sub (K.mul (kOf (w.cOut 11) (w.cOut 12)) (kOf (w.cOut 9) (w.cOut 10)))
      (K.mul (kOf (w.cOut 7) (w.cOut 8)) (kOf (w.cOut 13) (w.cOut 14)))) = K.zero
  segOut : w.cOut 0 = w.cIn 0 + w.isClose
  memOut : ∀ i : Fin 4,
    w.cOut (35 + i) = w.isClose * w.cOut (19 + i) + (1 - w.isClose) * w.cIn (35 + i)
  -- §9.2 chains and §9.3 challenges
  chainOps : StepWitness.carryDigest w.carryOut 23 =
    chainLink .ops w.idxEff (w.previousDigest 0) (packWords (StepWitness.laneBits w.opsBits))
  chainInitial : StepWitness.carryDigest w.carryOut 27 =
    chainLink .mem w.idxEff (w.previousDigest 4) (packWords (StepWitness.laneBits w.initialBits))
  chainFinal : StepWitness.carryDigest w.carryOut 31 =
    chainLink .mem w.idxEff (w.previousDigest 8) (packWords (StepWitness.laneBits w.finalBits))
  freshEta : etaChallenges ⟨planDigest p, (w.cIn 2).val, w.proposalDigest 0,
    StepWitness.carryDigest w.carryIn 35, w.proposalDigest 4⟩ = w.freshEta
  -- §11.1 input state
  stateIn : stateWordsRaw (List.ofFn w.appIn) (List.ofFn w.carryIn) = zIn

namespace StepWitness

variable {p : Plan} (w : StepWitness p) (two : p.bOps = 2)

/-- Port 0, the fetch port. -/
def fetchSlot : Fin p.bOps := ⟨0, by omega⟩
/-- Port 1, the data port. -/
def dataSlot : Fin p.bOps := ⟨1, by omega⟩

/-- Bit `k` of the fetched instruction word. -/
def instructionBit (k : ℕ) : F := w.opBit (fetchSlot two) (3 + p.μ + k)

def isLoad : F := w.instructionBit two 0 * (1 - w.instructionBit two 1)
def isStore : F := (1 - w.instructionBit two 0) * w.instructionBit two 1
def isLoadi : F := w.instructionBit two 0 * w.instructionBit two 1

/-- The instruction argument `v / 4`: bits 2–31 of the fetched word. -/
def argument : F := bitsWord (w.opsBits (fetchSlot two)) (3 + p.μ + 2) 30

/-- The machine's port rows (spec §10, Ob7) and its next state. -/
structure MachineRows : Prop where
  fetchPad : w.pad (fetchSlot two) = w.idle
  fetchRead : w.isWrite (fetchSlot two) = 0
  fetchRom : w.isRam (fetchSlot two) = 0
  fetchAddr : w.addr (fetchSlot two) = (1 - w.idle) * w.appIn 0
  dataPad : w.pad (dataSlot two) = 1 - (w.isLoad two + w.isStore two)
  dataWrite : w.isWrite (dataSlot two) = w.isStore two
  dataRam : w.isRam (dataSlot two) = w.isLoad two + w.isStore two
  dataAddr : w.addr (dataSlot two) = (w.isLoad two + w.isStore two) * w.argument two
  dataValue : w.vw (dataSlot two) =
    w.isStore two * w.appIn 1 + w.isLoad two * w.vr (dataSlot two)
  pcOut : w.appOut 0 = w.appIn 0 + w.isLoad two + w.isStore two + w.isLoadi two
  accOut : w.appOut 1 = w.isLoad two * w.vr (dataSlot two) + w.isLoadi two * w.argument two +
    (1 - w.isLoad two - w.isLoadi two) * w.appIn 1

end StepWitness

end NightstreamFPrime.Lifecycle.Nebula
