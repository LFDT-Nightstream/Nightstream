import NightstreamFPrime.Lifecycle.Nebula.RowExprs

/-! Owns the polynomial rows of the memory-application circuit as expressions
over the witness words, and their meaning: the rows evaluate to zero on a word
vector exactly when its decoded witness satisfies `PolyRows` and
`MachineRows`. It does not own the hash rows or the circuit. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic

/-! ### Row-list algebra -/

theorem hold_append (env : Env) (l₁ l₂ : List Circuit.Expr) :
    ConstraintsHold env (l₁ ++ l₂) ↔ ConstraintsHold env l₁ ∧ ConstraintsHold env l₂ :=
  Circuit.constraintsHold_append env l₁ l₂

theorem hold_single (env : Env) (e : Circuit.Expr) : ConstraintsHold env [e] ↔ e.eval env = 0 := by
  simp [ConstraintsHold]

theorem hold_flatMap {α : Type} (env : Env) (l : List α) (f : α → List Circuit.Expr) :
    ConstraintsHold env (l.flatMap f) ↔ ∀ x ∈ l, ConstraintsHold env (f x) := by
  simp only [ConstraintsHold, List.mem_flatMap]
  constructor
  · intro all x member e inner
    exact all e ⟨x, member, inner⟩
  · rintro all e ⟨x, member, inner⟩
    exact all x member e inner

theorem hold_map {α : Type} (env : Env) (l : List α) (f : α → Circuit.Expr) :
    ConstraintsHold env (l.map f) ↔ ∀ x ∈ l, (f x).eval env = 0 := by
  simp [ConstraintsHold]

theorem hold_finRange {n : ℕ} (env : Env) (f : Fin n → List Circuit.Expr) :
    ConstraintsHold env ((List.finRange n).flatMap f) ↔ ∀ k, ConstraintsHold env (f k) := by
  rw [hold_flatMap]
  simp

theorem hold_finRange_map {n : ℕ} (env : Env) (f : Fin n → Circuit.Expr) :
    ConstraintsHold env ((List.finRange n).map f) ↔ ∀ k, (f k).eval env = 0 := by
  rw [hold_map]
  simp

theorem eval_sub_eq_zero (env : Env) (a b : Circuit.Expr) :
    (a - b).eval env = 0 ↔ a.eval env = b.eval env := by
  rw [Circuit.Expr.eval_sub, sub_eq_zero]

/-- The bit row `x · (x − 1)`. -/
def bitRow (x : Circuit.Expr) : Circuit.Expr := x * (x - 1)

/-- In `F`, `x · (x − 1) = 0` exactly for the bits. -/
theorem mul_sub_one_eq_zero_iff (x : F) : x * (x - 1) = 0 ↔ IsBit x := by
  letI : Fact (Nat.Prime goldilocksModulus) := ⟨GoldilocksPrime.goldilocks_natPrime⟩
  have key : ∀ y : ZMod goldilocksModulus, y * (y - 1) = 0 ↔ y = 0 ∨ y = 1 := fun y => by
    rw [mul_eq_zero, sub_eq_zero]
  exact key x

theorem bitRow_iff (env : Env) (x : Circuit.Expr) : (bitRow x).eval env = 0 ↔ IsBit (x.eval env) := by
  rw [bitRow, Circuit.Expr.eval_hmul, Circuit.Expr.eval_sub, Expr.eval_one, mul_sub_one_eq_zero_iff]

theorem hold_bits (env : Env) (l : List Circuit.Expr) :
    ConstraintsHold env (l.map bitRow) ↔ ∀ x ∈ l.map (Circuit.Expr.eval env), IsBit x := by
  rw [hold_map]
  simp [bitRow_iff]

namespace Rows

variable (p : Plan)

open Sym (word cIn cOut eta1 eta2 eta1Sq idle isOpen openInverse isClose closeInverse idxEff
  isWrite isRam addrBit vrBit vwBit rtBit diffBit addr vr vw rt diff opLane scanValue scanStamp
  scanLane tsWord segWord cntBefore wt globalIndex scanIndex opsProductAfter scanProductAfter)

/-! ### §8 bits and the flags -/

def bitRows : List Circuit.Expr :=
  (List.finRange p.bOps).flatMap (fun j : Fin p.bOps => (opLane p j.val).map bitRow) ++
  (List.finRange p.bOps).flatMap (fun j : Fin p.bOps =>
    (List.finRange p.wTs).map fun k : Fin p.wTs => bitRow (diffBit p j.val k.val)) ++
  (List.finRange p.bScan).flatMap (fun j : Fin p.bScan =>
    (scanLane p (Words.initial p j.val 0)).map bitRow) ++
  (List.finRange p.bScan).flatMap (fun j : Fin p.bScan =>
    (scanLane p (Words.final p j.val 0)).map bitRow) ++
  (List.finRange p.wTs).map (fun k : Fin p.wTs => bitRow (word (Words.tsBits p k.val))) ++
  (List.finRange p.segWidth).map (fun k : Fin p.segWidth =>
    bitRow (word (Words.segBits p k.val))) ++
  [bitRow idle, bitRow isOpen, bitRow isClose]

/-! ### §12 arm and §11.2 open -/

def armRows : List Circuit.Expr :=
  [(cIn 1 - constE (natWord p.n)) * isOpen,
    (cIn 1 - constE (natWord p.n)) * openInverse - (1 - isOpen),
    isOpen * (constE (natWord (p.sMax - 1)) - cIn 0 - segWord p),
    idxEff - (1 - isOpen) * cIn 1] ++
  (List.finRange 12).map (fun i : Fin 12 => word (Words.seenPrev i.val) -
    (isOpen * constE (headerWords p i.val) + (1 - isOpen) * cIn (23 + i.val))) ++
  (List.finRange 8).map (fun i : Fin 8 => cOut (15 + i.val) -
    (isOpen * word (Words.proposal i.val) + (1 - isOpen) * cIn (15 + i.val))) ++
  (List.finRange 4).map (fun i : Fin 4 => cOut (3 + i.val) -
    (isOpen * word (Words.etaFresh i.val) + (1 - isOpen) * cIn (3 + i.val))) ++
  KExpr.equalities eta1Sq (KExpr.mul eta1 eta1)

/-! ### §8.1 operation rows -/

def slotRows (j : Fin p.bOps) : List Circuit.Expr :=
  [(1 - isWrite p j) * (vw p j - vr p j),
    (1 - Sym.pad p j) * (wt p j - rt p j - 1 - diff p j),
    isWrite p j * (1 - isRam p j)] ++
  ((List.finRange p.μ).filter (fun k : Fin p.μ => p.r ≤ k.val)).map
    (fun k : Fin p.μ => (1 - isRam p j) * addrBit p j k.val) ++
  (opLane p j).tail.map (fun x => Sym.pad p j * x) ++
  KExpr.equalities (opsProductAfter p 0 (j.val + 1)) (KExpr.mul (opsProductAfter p 0 j.val)
    (gatedE (Sym.pad p j) (fingerprintE eta1 eta2 eta1Sq (rt p j) (globalIndex p j) (vr p j)))) ++
  KExpr.equalities (opsProductAfter p 1 (j.val + 1)) (KExpr.mul (opsProductAfter p 1 j.val)
    (gatedE (Sym.pad p j) (fingerprintE eta1 eta2 eta1Sq (wt p j) (globalIndex p j) (vw p j))))

/-! ### §8.3 scan rows -/

def scanRows (j : Fin p.bScan) : List Circuit.Expr :=
  KExpr.equalities (scanProductAfter p 0 (j.val + 1)) (KExpr.mul (scanProductAfter p 0 j.val)
    (fingerprintE eta1 eta2 eta1Sq (scanStamp p (Words.initial p j 0)) (scanIndex p j)
      (scanValue (Words.initial p j 0)))) ++
  KExpr.equalities (scanProductAfter p 1 (j.val + 1)) (KExpr.mul (scanProductAfter p 1 j.val)
    (fingerprintE eta1 eta2 eta1Sq (scanStamp p (Words.final p j 0)) (scanIndex p j)
      (scanValue (Words.final p j 0))))

/-! ### §8.4 boundary and §11.2 close -/

def closeRows : List Circuit.Expr :=
  [cOut 2 - (cIn 2 + cntBefore p p.bOps),
    cOut 2 - tsWord p,
    cOut 1 - (idxEff + 1)] ++
  KExpr.equalities (kOfE (cOut 7) (cOut 8)) (opsProductAfter p 0 p.bOps) ++
  KExpr.equalities (kOfE (cOut 9) (cOut 10)) (opsProductAfter p 1 p.bOps) ++
  KExpr.equalities (kOfE (cOut 11) (cOut 12)) (scanProductAfter p 0 p.bScan) ++
  KExpr.equalities (kOfE (cOut 13) (cOut 14)) (scanProductAfter p 1 p.bScan) ++
  [(cOut 1 - constE (natWord p.n)) * isClose,
    (cOut 1 - constE (natWord p.n)) * closeInverse - (1 - isClose)] ++
  (List.finRange 4).map (fun i : Fin 4 => isClose * (cOut (23 + i.val) - cOut (15 + i.val))) ++
  (List.finRange 4).map (fun i : Fin 4 => isClose * (cOut (27 + i.val) - cIn (35 + i.val))) ++
  (List.finRange 4).map (fun i : Fin 4 => isClose * (cOut (31 + i.val) - cOut (19 + i.val))) ++
  KExpr.equalities (KExpr.mul (embedE isClose)
    (KExpr.sub (KExpr.mul (kOfE (cOut 11) (cOut 12)) (kOfE (cOut 9) (cOut 10)))
      (KExpr.mul (kOfE (cOut 7) (cOut 8)) (kOfE (cOut 13) (cOut 14))))) KExpr.zero ++
  [cOut 0 - (cIn 0 + isClose)] ++
  (List.finRange 4).map (fun i : Fin 4 => cOut (35 + i.val) -
    (isClose * cOut (19 + i.val) + (1 - isClose) * cIn (35 + i.val)))

/-- Every polynomial memory row. -/
def memoryRows : List Circuit.Expr :=
  bitRows p ++ armRows p ++ (List.finRange p.bOps).flatMap (slotRows p) ++
    (List.finRange p.bScan).flatMap (scanRows p) ++ closeRows p

/-! ### §10 machine port rows (fetch slot 0, data slot 1) -/

def instructionBit (k : ℕ) : Circuit.Expr := vrBit p 0 k
def isLoad : Circuit.Expr := instructionBit p 0 * (1 - instructionBit p 1)
def isStore : Circuit.Expr := (1 - instructionBit p 0) * instructionBit p 1
def isLoadi : Circuit.Expr := instructionBit p 0 * instructionBit p 1
def argument : Circuit.Expr := Expr.chunk ((List.ofFn fun k : Fin 32 => vrBit p 0 k).drop 2)

def appIn (i : ℕ) : Circuit.Expr := word (Words.appIn i)
def appOut (i : ℕ) : Circuit.Expr := word (Words.appOut i)

def machineRows : List Circuit.Expr :=
  [Sym.pad p 0 - idle,
    isWrite p 0,
    isRam p 0,
    addr p 0 - (1 - idle) * appIn 0,
    Sym.pad p 1 - (1 - (isLoad p + isStore p)),
    isWrite p 1 - isStore p,
    isRam p 1 - (isLoad p + isStore p),
    addr p 1 - (isLoad p + isStore p) * argument p,
    vw p 1 - (isStore p * appIn 1 + isLoad p * vr p 1),
    appOut 0 - (appIn 0 + isLoad p + isStore p + isLoadi p),
    appOut 1 - (isLoad p * vr p 1 + isLoadi p * argument p +
      (1 - isLoad p - isLoadi p) * appIn 1)]

/-- Every polynomial row of the memory application. -/
def polyRows : List Circuit.Expr := memoryRows p ++ machineRows p

end Rows

end NightstreamFPrime.Lifecycle.Nebula
