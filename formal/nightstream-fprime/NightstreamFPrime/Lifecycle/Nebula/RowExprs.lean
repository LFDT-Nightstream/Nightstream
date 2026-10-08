import NightstreamFPrime.Lifecycle.Nebula.WitnessWords
import NightstreamFPrime.Circuit.Quadratic
import NightstreamFPrime.Circuit.VariableSupport

/-! Owns the symbolic step witness: one expression per witness value over the
word variables of `Words` (variable `k` is witness word `k`), and the
evaluation of each against `StepWitness.ofWords`. A circuit substitutes its
witness wires for the word variables. It does not own the row list. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic

namespace Expr

@[simp] theorem eval_zero (env : Env) : (0 : Circuit.Expr).eval env = 0 := rfl
@[simp] theorem eval_one (env : Env) : (1 : Circuit.Expr).eval env = 1 := rfl
@[simp] theorem eval_two (env : Env) : (2 : Circuit.Expr).eval env = 2 := rfl

/-- Replace each variable `k` with `wires k`. -/
def subst (wires : ℕ → Circuit.Expr) : Circuit.Expr → Circuit.Expr
  | .var k => wires k
  | .const value => .const value
  | .add left right => .add (subst wires left) (subst wires right)
  | .mul left right => .mul (subst wires left) (subst wires right)

theorem eval_subst (wires : ℕ → Circuit.Expr) (env : Env) (e : Circuit.Expr) :
    (subst wires e).eval env = e.eval fun k => (wires k).eval env := by
  induction e with
  | var k => rfl
  | const value => rfl
  | add left right ihl ihr => simp [subst, Circuit.Expr.eval, ihl, ihr]
  | mul left right ihl ihr => simp [subst, Circuit.Expr.eval, ihl, ihr]

theorem varsSatisfy_subst (wires : ℕ → Circuit.Expr) (allowed : ℕ → Prop)
    (each : ∀ k, (wires k).VarsSatisfy allowed) (e : Circuit.Expr) :
    (subst wires e).VarsSatisfy allowed := by
  induction e with
  | var k => exact each k
  | const value => trivial
  | add left right ihl ihr => exact ⟨ihl, ihr⟩
  | mul left right ihl ihr => exact ⟨ihl, ihr⟩

/-- The sum of a list of expressions. -/
def sum : List Circuit.Expr → Circuit.Expr
  | [] => 0
  | e :: rest => e + sum rest

theorem eval_sum (env : Env) (l : List Circuit.Expr) :
    (sum l).eval env = (l.map (Circuit.Expr.eval env)).sum := by
  induction l with
  | nil => rfl
  | cons e rest ih => simp [sum, ih]

/-- The little-endian value of a list of bit expressions. -/
def chunk : List Circuit.Expr → Circuit.Expr
  | [] => 0
  | b :: bs => b + 2 * chunk bs

theorem eval_chunk (env : Env) (l : List Circuit.Expr) :
    (chunk l).eval env = chunkWord (l.map (Circuit.Expr.eval env)) := by
  induction l with
  | nil => rfl
  | cons b bs ih =>
    simp only [chunk, Circuit.Expr.eval_hadd, Circuit.Expr.eval_hmul, ih, List.map_cons, chunkWord]
    rfl

/-- The little-endian value of a bit vector of expressions. -/
def bits {n : ℕ} (f : Fin n → Circuit.Expr) : Circuit.Expr := chunk (List.ofFn f)

theorem eval_bits (env : Env) {n : ℕ} (f : Fin n → Circuit.Expr) :
    (bits f).eval env = bitsWord fun k => (f k).eval env := by
  rw [bits, eval_chunk, List.map_ofFn]
  rfl

/-- Spec §9.1 packing of bit expressions: chunks of 63. -/
def pack (l : List Circuit.Expr) : List Circuit.Expr :=
  if _h : l = [] then [] else chunk (l.take 63) :: pack (l.drop 63)
termination_by l.length
decreasing_by
  have : 0 < l.length := List.length_pos_of_ne_nil _h
  simp only [List.length_drop]
  omega

theorem eval_pack (env : Env) (l : List Circuit.Expr) :
    (pack l).map (Circuit.Expr.eval env) = packWords (l.map (Circuit.Expr.eval env)) := by
  induction l using pack.induct with
  | case1 => simp [pack, packWords]
  | case2 l nonempty ih =>
    have mapped : l.map (Circuit.Expr.eval env) ≠ [] := by simpa using nonempty
    rw [pack, dif_neg nonempty, packWords, dif_neg mapped, List.map_cons, ih, eval_chunk,
      List.map_take, List.map_drop]

end Expr

/-- A field word as a constant expression. -/
def constE (x : F) : Circuit.Expr := .const x

/-- `(x, 0)` in `K`. -/
def embedE (x : Circuit.Expr) : KExpr := ⟨x, 0⟩

theorem eval_embedE (env : Env) (x : Circuit.Expr) : (embedE x).eval env = embed (x.eval env) :=
  rfl

/-- A `K` element from two word expressions. -/
def kOfE (a b : Circuit.Expr) : KExpr := ⟨a, b⟩

theorem eval_kOfE (env : Env) (a b : Circuit.Expr) : (kOfE a b).eval env = kOf (a.eval env) (b.eval env) :=
  rfl

/-- The symbolic fingerprint of spec §8.2. -/
def fingerprintE (η1 η2 η1sq : KExpr) (t g v : Circuit.Expr) : KExpr :=
  KExpr.sub (KExpr.add (KExpr.add (embedE g) (KExpr.mul η1 (embedE v)))
    (KExpr.mul η1sq (embedE t))) η2

theorem eval_fingerprintE (env : Env) (η1 η2 η1sq : KExpr) (t g v : Circuit.Expr) :
    (fingerprintE η1 η2 η1sq t g v).eval env =
      fingerprintK (η1.eval env) (η2.eval env) (η1sq.eval env) (t.eval env) (g.eval env)
        (v.eval env) := by
  simp [fingerprintE, fingerprintK, eval_embedE]

/-- The symbolic O8/O9 gate `pad + (1 − pad) · f`. -/
def gatedE (pad : Circuit.Expr) (f : KExpr) : KExpr :=
  KExpr.add (embedE pad) (KExpr.mul (embedE (1 - pad)) f)

theorem eval_gatedE (env : Env) (pad : Circuit.Expr) (f : KExpr) :
    (gatedE pad f).eval env = gatedK (pad.eval env) (f.eval env) := by
  simp only [gatedE, gatedK, KExpr.eval_add, KExpr.eval_mul, eval_embedE, Circuit.Expr.eval_sub]
  rfl

namespace Sym

variable (p : Plan)

def word (k : ℕ) : Circuit.Expr := .var k

def carryWord (base : ℕ → ℕ) (k : ℕ) : Circuit.Expr := if k < 39 then word (base k) else 0

def cIn (k : ℕ) : Circuit.Expr := carryWord Words.carryIn k
def cOut (k : ℕ) : Circuit.Expr := carryWord Words.carryOut k

def eta1 : KExpr := kOfE (cOut 3) (cOut 4)
def eta2 : KExpr := kOfE (cOut 5) (cOut 6)
def eta1Sq : KExpr := kOfE (word (Words.eta1Square 0)) (word (Words.eta1Square 1))

def idle : Circuit.Expr := word Words.idle
def isOpen : Circuit.Expr := word Words.isOpen
def openInverse : Circuit.Expr := word Words.openInverse
def isClose : Circuit.Expr := word Words.isClose
def closeInverse : Circuit.Expr := word Words.closeInverse
def idxEff : Circuit.Expr := word Words.idxEff

def pad (j : ℕ) : Circuit.Expr := word (Words.op p j Words.opPad)
def isWrite (j : ℕ) : Circuit.Expr := word (Words.op p j Words.opIsWrite)
def isRam (j : ℕ) : Circuit.Expr := word (Words.op p j Words.opIsRam)
def addrBit (j k : ℕ) : Circuit.Expr := word (Words.op p j (Words.opAddr k))
def vrBit (j k : ℕ) : Circuit.Expr := word (Words.op p j (Words.opVr p k))
def vwBit (j k : ℕ) : Circuit.Expr := word (Words.op p j (Words.opVw p k))
def rtBit (j k : ℕ) : Circuit.Expr := word (Words.op p j (Words.opRt p k))
def diffBit (j k : ℕ) : Circuit.Expr := word (Words.op p j (Words.opDiff p k))

def addr (j : ℕ) : Circuit.Expr := Expr.bits fun k : Fin p.μ => addrBit p j k
def vr (j : ℕ) : Circuit.Expr := Expr.bits fun k : Fin 32 => vrBit p j k
def vw (j : ℕ) : Circuit.Expr := Expr.bits fun k : Fin 32 => vwBit p j k
def rt (j : ℕ) : Circuit.Expr := Expr.bits fun k : Fin p.wTs => rtBit p j k
def diff (j : ℕ) : Circuit.Expr := Expr.bits fun k : Fin p.wTs => diffBit p j k

/-- The lane bits of operation slot `j`, in spec §6.3 order. -/
def opLane (j : ℕ) : List Circuit.Expr :=
  [pad p j, isWrite p j, isRam p j] ++ List.ofFn (fun k : Fin p.μ => addrBit p j k) ++
    List.ofFn (fun k : Fin 32 => vrBit p j k) ++ List.ofFn (fun k : Fin 32 => vwBit p j k) ++
    List.ofFn (fun k : Fin p.wTs => rtBit p j k)

/-- Word `k` of the scan slot that starts at word `base`. -/
def scanValueBit (base k : ℕ) : Circuit.Expr := word (base + Words.scanValue k)
def scanStampBit (base k : ℕ) : Circuit.Expr := word (base + Words.scanStamp k)
def scanValue (base : ℕ) : Circuit.Expr := Expr.bits fun k : Fin 32 => scanValueBit base k
def scanStamp (base : ℕ) : Circuit.Expr := Expr.bits fun k : Fin p.wTs => scanStampBit base k

def scanLane (base : ℕ) : List Circuit.Expr :=
  List.ofFn (fun k : Fin 32 => scanValueBit base k) ++
    List.ofFn (fun k : Fin p.wTs => scanStampBit base k)

def tsWord : Circuit.Expr := Expr.bits fun k : Fin p.wTs => word (Words.tsBits p k)
def segWord : Circuit.Expr := Expr.bits fun k : Fin p.segWidth => word (Words.segBits p k)

/-- `cnt` before slot `k` (row O2). -/
def cntBefore (k : ℕ) : Circuit.Expr :=
  Expr.sum ((List.ofFn fun j : Fin p.bOps => 1 - pad p j).take k)

def wt (j : ℕ) : Circuit.Expr := cIn 2 + cntBefore p (j + 1)

def globalIndex (j : ℕ) : Circuit.Expr := addr p j + isRam p j * constE (natWord p.romSize)

def scanIndex (j : ℕ) : Circuit.Expr := idxEff * constE (natWord p.bScan) + constE (natWord j)

def startProduct (k : ℕ) : KExpr :=
  KExpr.add (embedE isOpen) (KExpr.mul (embedE (1 - isOpen)) (kOfE (cIn k) (cIn (k + 1))))

def opsProductAfter (pair k : ℕ) : KExpr :=
  if k = 0 then startProduct (7 + 2 * pair)
  else if k - 1 < p.bOps then
    kOfE (word (Words.opsProducts p (k - 1) (2 * pair % 4)))
      (word (Words.opsProducts p (k - 1) ((2 * pair + 1) % 4)))
  else KExpr.zero

def scanProductAfter (pair k : ℕ) : KExpr :=
  if k = 0 then startProduct (11 + 2 * pair)
  else if k - 1 < p.bScan then
    kOfE (word (Words.scanProducts p (k - 1) (2 * pair % 4)))
      (word (Words.scanProducts p (k - 1) ((2 * pair + 1) % 4)))
  else KExpr.zero

end Sym

/-! ### Evaluation against the decoded witness -/

namespace Sym

variable (p : Plan) (v : ℕ → F)

local notation "W" => StepWitness.ofWords p v

omit p in
theorem eval_carryWord (base : ℕ → ℕ) (k : ℕ) (c : Fin 39 → F)
    (same : ∀ i : Fin 39, c i = v (base i)) :
    (carryWord base k).eval v = StepWitness.carryWord c k := by
  unfold carryWord StepWitness.carryWord
  split <;> simp_all [word]

theorem eval_cIn (k : ℕ) : (cIn k).eval v = (W).cIn k :=
  eval_carryWord v _ k _ fun _ => rfl

theorem eval_cOut (k : ℕ) : (cOut k).eval v = (W).cOut k :=
  eval_carryWord v _ k _ fun _ => rfl

theorem eval_eta1 : (eta1).eval v = (W).eta1 := by
  rw [eta1, StepWitness.eta1, eval_kOfE, eval_cOut p, eval_cOut p]

theorem eval_eta2 : (eta2).eval v = (W).eta2 := by
  rw [eta2, StepWitness.eta2, eval_kOfE, eval_cOut p, eval_cOut p]

theorem eval_eta1Sq : (eta1Sq).eval v = (W).eta1Sq := rfl

theorem eval_addr (j : Fin p.bOps) : (addr p j).eval v = (W).addr j := by
  rw [addr, Expr.eval_bits]
  rfl

theorem eval_vr (j : Fin p.bOps) : (vr p j).eval v = (W).vr j := by
  rw [vr, Expr.eval_bits]
  rfl

theorem eval_vw (j : Fin p.bOps) : (vw p j).eval v = (W).vw j := by
  rw [vw, Expr.eval_bits]
  rfl

theorem eval_rt (j : Fin p.bOps) : (rt p j).eval v = (W).rt j := by
  rw [rt, Expr.eval_bits]
  rfl

theorem eval_diff (j : Fin p.bOps) : (diff p j).eval v = (W).diff j := by
  rw [diff, Expr.eval_bits]
  rfl

theorem eval_opLane (j : Fin p.bOps) : (opLane p j).map (Circuit.Expr.eval v) = ((W).ops j).lane := by
  simp [opLane, OpSlotBits.lane, List.map_ofFn, Function.comp_def, StepWitness.ofWords,
    OpSlotBits.ofWords, pad, isWrite, isRam, addrBit, vrBit, vwBit, rtBit, word]

theorem eval_initialLane (j : Fin p.bScan) :
    (scanLane p (Words.initial p j 0)).map (Circuit.Expr.eval v) = ((W).initial j).lane := by
  simp [scanLane, ScanSlotBits.lane, List.map_ofFn, Function.comp_def, StepWitness.ofWords,
    ScanSlotBits.ofWords, scanValueBit, scanStampBit, word]

theorem eval_finalLane (j : Fin p.bScan) :
    (scanLane p (Words.final p j 0)).map (Circuit.Expr.eval v) = ((W).final j).lane := by
  simp [scanLane, ScanSlotBits.lane, List.map_ofFn, Function.comp_def, StepWitness.ofWords,
    ScanSlotBits.ofWords, scanValueBit, scanStampBit, word]

theorem eval_initialValue (j : Fin p.bScan) :
    (scanValue (Words.initial p j 0)).eval v = bitsWord ((W).initial j).value := by
  rw [scanValue, Expr.eval_bits]
  rfl

theorem eval_initialStamp (j : Fin p.bScan) :
    (scanStamp p (Words.initial p j 0)).eval v = bitsWord ((W).initial j).stamp := by
  rw [scanStamp, Expr.eval_bits]
  rfl

theorem eval_finalValue (j : Fin p.bScan) :
    (scanValue (Words.final p j 0)).eval v = bitsWord ((W).final j).value := by
  rw [scanValue, Expr.eval_bits]
  rfl

theorem eval_finalStamp (j : Fin p.bScan) :
    (scanStamp p (Words.final p j 0)).eval v = bitsWord ((W).final j).stamp := by
  rw [scanStamp, Expr.eval_bits]
  rfl

theorem eval_tsWord : (tsWord p).eval v = bitsWord (W).tsBits := by
  rw [tsWord, Expr.eval_bits]
  rfl

theorem eval_segWord : (segWord p).eval v = bitsWord (W).segBits := by
  rw [segWord, Expr.eval_bits]
  rfl

theorem eval_cntBefore (k : ℕ) : (cntBefore p k).eval v = (W).cntBefore k := by
  rw [cntBefore, Expr.eval_sum, StepWitness.cntBefore, List.map_take, List.map_ofFn]
  simp [Function.comp_def, pad, word, StepWitness.pad, StepWitness.ofWords, OpSlotBits.ofWords]

theorem eval_wt (j : Fin p.bOps) : (wt p j).eval v = (W).wt j := by
  simp only [wt, StepWitness.wt, Circuit.Expr.eval_hadd, eval_cIn p, eval_cntBefore]

theorem eval_globalIndex (j : Fin p.bOps) : (globalIndex p j).eval v = (W).globalIndex j := by
  simp only [globalIndex, StepWitness.globalIndex, Circuit.Expr.eval_hadd, Circuit.Expr.eval_hmul,
    eval_addr, constE, Circuit.Expr.eval_const]
  rfl

theorem eval_scanIndex (j : Fin p.bScan) : (scanIndex p j).eval v = (W).scanIndex j := by
  simp only [scanIndex, StepWitness.scanIndex, Circuit.Expr.eval_hadd, Circuit.Expr.eval_hmul,
    constE, Circuit.Expr.eval_const]
  rfl

theorem eval_startProduct (k : ℕ) : (startProduct k).eval v = (W).startProduct k := by
  simp only [startProduct, StepWitness.startProduct, KExpr.eval_add, KExpr.eval_mul, eval_embedE,
    Circuit.Expr.eval_sub, eval_kOfE, eval_cIn p, Expr.eval_one]
  rfl

theorem eval_opsProductAfter (pair k : ℕ) :
    (opsProductAfter p pair k).eval v = (W).opsProductAfter pair k := by
  unfold opsProductAfter StepWitness.opsProductAfter
  split_ifs <;> first | exact eval_startProduct p v _ | rfl

theorem eval_scanProductAfter (pair k : ℕ) :
    (scanProductAfter p pair k).eval v = (W).scanProductAfter pair k := by
  unfold scanProductAfter StepWitness.scanProductAfter
  split_ifs <;> first | exact eval_startProduct p v _ | rfl

end Sym

end NightstreamFPrime.Lifecycle.Nebula
