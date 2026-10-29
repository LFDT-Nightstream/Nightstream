import NightstreamFPrime.Lifecycle.Nebula.RowExprs
import NightstreamFPrime.Lifecycle.Nebula.PortApp

/-! Owns the polynomial rows of the memory-application circuit as expressions
over the witness words: the memory rows, grouped by the spec check that names
them, and the application's rows with their local variables mapped to witness
words. Their meaning is in `RowMeaning`. It does not own the hash rows or the
circuit. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic

/-! ### Row-list algebra -/

/-- Rows grouped by the spec check that names them, in row order. A row list
joins its groups, and `RowNames` reads the names from the same groups, so each
name labels its own rows. -/
abbrev Checks := List (String × List Circuit.Expr)

/-- The rows of `checks`, joined from the left. -/
def joinChecks (checks : Checks) : List Circuit.Expr :=
  checks.foldl (fun rows check => rows ++ check.2) []

theorem joinChecks_length (checks : Checks) :
    (joinChecks checks).length = (checks.map fun check => check.2.length).sum := by
  have from_rows : ∀ rows : List Circuit.Expr,
      (checks.foldl (fun rows check => rows ++ check.2) rows).length =
        rows.length + (checks.map fun check => check.2.length).sum := by
    induction checks with
    | nil => simp
    | cons check rest ih =>
      intro rows
      simp only [List.foldl_cons, ih, List.length_append, List.map_cons, List.sum_cons]
      omega
  rw [joinChecks, from_rows, List.length_nil, Nat.zero_add]

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
  let : Fact (Nat.Prime goldilocksModulus) := ⟨GoldilocksPrime.goldilocks_natPrime⟩
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

variable (p : Geometry)

open _root_.NightstreamFPrime.Lifecycle.Nebula.Sym (word cIn cOut eta1 eta2 eta1Sq isOpen openInverse isClose closeInverse idxEff
  isWrite isRam addrBit vrBit vwBit rtBit diffBit addr vr vw rt diff opLane scanValue scanStamp
  scanLane tsWord segWord cntBefore wt globalIndex scanIndex opsProductAfter scanProductAfter)

/-! ### §8 bits and the flags -/

def bitChecks : Checks :=
  [("O1 operation lanes",
      (List.finRange p.bOps).flatMap (fun j : Fin p.bOps => (opLane p j.val).map bitRow)),
    ("O1 diff bits", (List.finRange p.bOps).flatMap (fun j : Fin p.bOps =>
      (List.finRange p.wTs).map fun k : Fin p.wTs => bitRow (diffBit p j.val k.val))),
    ("S1 IS lanes", (List.finRange p.bScan).flatMap (fun j : Fin p.bScan =>
      (scanLane p (Words.initial p j.val 0)).map bitRow)),
    ("S1 FS lanes", (List.finRange p.bScan).flatMap (fun j : Fin p.bScan =>
      (scanLane p (Words.final p j.val 0)).map bitRow)),
    ("ts bits", (List.finRange p.wTs).map (fun k : Fin p.wTs => bitRow (word (Words.tsBits p k.val)))),
    ("segment bits", (List.finRange p.segWidth).map (fun k : Fin p.segWidth =>
      bitRow (word (Words.segBits p k.val)))),
    ("flag bits", [bitRow isOpen, bitRow isClose])]

def bitRows : List Circuit.Expr := joinChecks (bitChecks p)

/-! ### §12 arm and §11.2 open -/

def armChecks : Checks :=
  [("arm open flag", [(cIn 1 - constE (natWord p.n)) * isOpen,
      (cIn 1 - constE (natWord p.n)) * openInverse - (1 - isOpen)]),
    ("arm segment bound", [isOpen * (constE (natWord (p.sMax - 1)) - cIn 0 - segWord p)]),
    ("arm idx", [idxEff - (1 - isOpen) * cIn 1]),
    ("arm D_seen headers", (List.finRange 12).map (fun i : Fin 12 => word (Words.seenPrev i.val) -
      (isOpen * constE (headerWords p i.val) + (1 - isOpen) * cIn (23 + i.val)))),
    ("arm D_pre", (List.finRange 8).map (fun i : Fin 8 => cOut (15 + i.val) -
      (isOpen * word (Words.proposal i.val) + (1 - isOpen) * cIn (15 + i.val)))),
    ("arm eta", (List.finRange 4).map (fun i : Fin 4 => cOut (3 + i.val) -
      (isOpen * word (Words.etaFresh i.val) + (1 - isOpen) * cIn (3 + i.val)))),
    ("arm eta1 square", KExpr.equalities eta1Sq (KExpr.mul eta1 eta1))]

def armRows : List Circuit.Expr := joinChecks (armChecks p)

/-! ### §8.1 operation rows -/

def slotChecks (j : Fin p.bOps) : Checks :=
  [(s!"O3 slot {j.val}", [(1 - isWrite p j) * (vw p j - vr p j)]),
    (s!"O4 slot {j.val}", [(1 - Sym.pad p j) * (wt p j - rt p j - 1 - diff p j)]),
    (s!"O5 slot {j.val}", [isWrite p j * (1 - isRam p j)]),
    (s!"O6 slot {j.val}", ((List.finRange p.μ).filter (fun k : Fin p.μ => p.r ≤ k.val)).map
      (fun k : Fin p.μ => (1 - isRam p j) * addrBit p j k.val)),
    (s!"O7 slot {j.val}", (opLane p j).tail.map (fun x => Sym.pad p j * x)),
    (s!"O8 slot {j.val}", KExpr.equalities (opsProductAfter p 0 (j.val + 1))
      (KExpr.mul (opsProductAfter p 0 j.val)
        (gatedE (Sym.pad p j) (fingerprintE eta1 eta2 eta1Sq (rt p j) (globalIndex p j) (vr p j))))),
    (s!"O9 slot {j.val}", KExpr.equalities (opsProductAfter p 1 (j.val + 1))
      (KExpr.mul (opsProductAfter p 1 j.val)
        (gatedE (Sym.pad p j) (fingerprintE eta1 eta2 eta1Sq (wt p j) (globalIndex p j) (vw p j)))))]

def slotRows (j : Fin p.bOps) : List Circuit.Expr := joinChecks (slotChecks p j)

/-! ### §8.3 scan rows -/

def scanChecks (j : Fin p.bScan) : Checks :=
  [(s!"S2 slot {j.val}", KExpr.equalities (scanProductAfter p 0 (j.val + 1))
      (KExpr.mul (scanProductAfter p 0 j.val)
        (fingerprintE eta1 eta2 eta1Sq (scanStamp p (Words.initial p j 0)) (scanIndex p j)
          (scanValue (Words.initial p j 0))))),
    (s!"S3 slot {j.val}", KExpr.equalities (scanProductAfter p 1 (j.val + 1))
      (KExpr.mul (scanProductAfter p 1 j.val)
        (fingerprintE eta1 eta2 eta1Sq (scanStamp p (Words.final p j 0)) (scanIndex p j)
          (scanValue (Words.final p j 0)))))]

def scanRows (j : Fin p.bScan) : List Circuit.Expr := joinChecks (scanChecks p j)

/-! ### §8.4 boundary and §11.2 close -/

def closeChecks : Checks :=
  [("boundary ts", [cOut 2 - (cIn 2 + cntBefore p p.bOps)]),
    ("boundary ts bits", [cOut 2 - tsWord p]),
    ("boundary idx", [cOut 1 - (idxEff + 1)]),
    ("boundary products",
      KExpr.equalities (kOfE (cOut 7) (cOut 8)) (opsProductAfter p 0 p.bOps) ++
      KExpr.equalities (kOfE (cOut 9) (cOut 10)) (opsProductAfter p 1 p.bOps) ++
      KExpr.equalities (kOfE (cOut 11) (cOut 12)) (scanProductAfter p 0 p.bScan) ++
      KExpr.equalities (kOfE (cOut 13) (cOut 14)) (scanProductAfter p 1 p.bScan)),
    ("close flag", [(cOut 1 - constE (natWord p.n)) * isClose,
      (cOut 1 - constE (natWord p.n)) * closeInverse - (1 - isClose)]),
    ("close D_seen.ops = D_pre.ops",
      (List.finRange 4).map (fun i : Fin 4 => isClose * (cOut (23 + i.val) - cOut (15 + i.val)))),
    ("close D_seen.is = D_mem",
      (List.finRange 4).map (fun i : Fin 4 => isClose * (cOut (27 + i.val) - cIn (35 + i.val)))),
    ("close D_seen.fs = D_pre.fs",
      (List.finRange 4).map (fun i : Fin 4 => isClose * (cOut (31 + i.val) - cOut (19 + i.val)))),
    ("close product equation", KExpr.equalities (KExpr.mul (embedE isClose)
      (KExpr.sub (KExpr.mul (kOfE (cOut 11) (cOut 12)) (kOfE (cOut 9) (cOut 10)))
        (KExpr.mul (kOfE (cOut 7) (cOut 8)) (kOfE (cOut 13) (cOut 14))))) KExpr.zero),
    ("close seg_idx", [cOut 0 - (cIn 0 + isClose)]),
    ("close D_mem", (List.finRange 4).map (fun i : Fin 4 => cOut (35 + i.val) -
      (isClose * cOut (19 + i.val) + (1 - isClose) * cIn (35 + i.val))))]

def closeRows : List Circuit.Expr := joinChecks (closeChecks p)

/-- Every polynomial memory row. -/
def memoryRows : List Circuit.Expr :=
  bitRows p ++ armRows p ++ (List.finRange p.bOps).flatMap (slotRows p) ++
    (List.finRange p.bScan).flatMap (scanRows p) ++ closeRows p

/-! ### §10 application rows -/

/-- The application's rows over the witness words. -/
def appRows (app : PortApp p) : List Circuit.Expr :=
  app.rows.map (Expr.subst fun k => word (Words.appVar p app.toAppShape k))

/-- Every polynomial row of the memory application. -/
def polyRows (app : PortApp p) : List Circuit.Expr := memoryRows p ++ appRows p app

end Rows

end NightstreamFPrime.Lifecycle.Nebula
