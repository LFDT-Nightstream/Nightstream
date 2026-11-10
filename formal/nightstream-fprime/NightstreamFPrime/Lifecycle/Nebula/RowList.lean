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
  scanLane tsWord segWord cnt cntBefore wt globalIndex scanIndex startWord opsProductAfter scanProductAfter
  opsFactor scanFactor)

/-! ### Fingerprint steps

Each row below is `word − recipe` with at most one product, so it lowers to
one R1CS row and no fresh column. A bit chunk is affine, so a fingerprint
reads its chunks once and allocates nothing. -/

/-- The rows of the fingerprint `f(t, g, v)` over factor words `a 0 … a 3`:
`a 0, a 1 = η1 · v` and `(a 2, a 3) = f` (`fingerprintWords`). -/
def fingerprintRows (a : ℕ → Circuit.Expr) (t g v : Circuit.Expr) : List Circuit.Expr :=
  [a 0 - cOut 3 * v, a 1 - cOut 4 * v,
    a 2 - (word (Words.eta1Square 0) * t + (g + a 0 - cOut 5)),
    a 3 - (word (Words.eta1Square 1) * t + (a 1 - cOut 6))]

/-- The rows of `next = prod · x` with the cross products `m = prod.c1 · x1`
and `n = prod.c1 · x0`. -/
def productRows (prod next : KExpr) (x0 x1 m n : Circuit.Expr) : List Circuit.Expr :=
  [m - prod.c1 * x1, n - prod.c1 * x0, next.c0 - (prod.c0 * x0 + constE 7 * m),
    next.c1 - (prod.c0 * x1 + n)]

/-- An ungated fingerprint step `next = prod · f` (rows S2 and S3) over the
factor words `a 0 … a 5` (`scanFactorWords`). -/
def factorRows (a : ℕ → Circuit.Expr) (prod next : KExpr) (t g v : Circuit.Expr) :
    List Circuit.Expr :=
  fingerprintRows a t g v ++ productRows prod next (a 2) (a 3) (a 4) (a 5)

/-- A gated fingerprint step `next = prod · (pad + (1 − pad) · f)` (rows O8
and O9) over the factor words `a 0 … a 7` (`opsFactorWords`). -/
def gatedRows (a : ℕ → Circuit.Expr) (pad : Circuit.Expr) (prod next : KExpr)
    (t g v : Circuit.Expr) : List Circuit.Expr :=
  fingerprintRows a t g v ++ [a 4 - ((1 - pad) * a 2 + pad), a 5 - (1 - pad) * a 3] ++
    productRows prod next (a 4) (a 5) (a 6) (a 7)

/-! ### Bits and flags -/

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

/-! ### Arm and open -/

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
    ("arm eta1 square", KExpr.equalities eta1Sq (KExpr.mul eta1 eta1)),
    ("arm start products", (List.finRange 4).flatMap (fun s : Fin 4 =>
      [startWord p s 0 - ((1 - isOpen) * cIn (7 + 2 * s.val) + isOpen),
        startWord p s 1 - (1 - isOpen) * cIn (8 + 2 * s.val)]))]

def armRows : List Circuit.Expr := joinChecks (armChecks p)

/-! ### Operation rows -/

def slotChecks (j : Fin p.bOps) : Checks :=
  [(s!"O2 slot {j.val}", [cnt p j - (cntBefore p j + (1 - Sym.pad p j))]),
    (s!"O3 slot {j.val}", [(1 - isWrite p j) * (vw p j - vr p j)]),
    (s!"O4 slot {j.val}", [(1 - Sym.pad p j) * (wt p j - rt p j - 1 - diff p j)]),
    (s!"O5 slot {j.val}", [isWrite p j * (1 - isRam p j)]),
    (s!"O6 slot {j.val}", ((List.finRange p.addrWidth).filter (fun k : Fin p.addrWidth => p.r ≤ k.val)).map
        (fun k : Fin p.addrWidth => (1 - isRam p j) * addrBit p j k.val) ++
      ((List.finRange p.addrWidth).filter (fun k : Fin p.addrWidth => p.μ ≤ k.val)).map
        (fun k : Fin p.addrWidth => isRam p j * addrBit p j k.val)),
    (s!"O7 slot {j.val}", (opLane p j).tail.map (fun x => Sym.pad p j * x)),
    (s!"O8 slot {j.val}", gatedRows (opsFactor p j 0) (Sym.pad p j) (opsProductAfter p 0 j.val)
      (opsProductAfter p 0 (j.val + 1)) (rt p j) (globalIndex p j) (vr p j)),
    (s!"O9 slot {j.val}", gatedRows (opsFactor p j 1) (Sym.pad p j) (opsProductAfter p 1 j.val)
      (opsProductAfter p 1 (j.val + 1)) (wt p j) (globalIndex p j) (vw p j))]

def slotRows (j : Fin p.bOps) : List Circuit.Expr := joinChecks (slotChecks p j)

/-! ### Scan rows -/

def scanChecks (j : Fin p.bScan) : Checks :=
  [(s!"S2 slot {j.val}", factorRows (scanFactor p j 0) (scanProductAfter p 0 j.val)
      (scanProductAfter p 0 (j.val + 1)) (scanStamp p (Words.initial p j 0)) (scanIndex p j)
      (scanValue (Words.initial p j 0))),
    (s!"S3 slot {j.val}", factorRows (scanFactor p j 1) (scanProductAfter p 1 j.val)
      (scanProductAfter p 1 (j.val + 1)) (scanStamp p (Words.final p j 0)) (scanIndex p j)
      (scanValue (Words.final p j 0)))]

def scanRows (j : Fin p.bScan) : List Circuit.Expr := joinChecks (scanChecks p j)

/-! ### Boundary rows and close -/

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

/-! ### Application rows -/

/-- The application's rows over the witness words. -/
def appRows (app : PortApp p) : List Circuit.Expr :=
  app.rows.map (Expr.subst fun k => word (Words.appVar p app.toAppShape k))

/-- Every polynomial row of the memory application. -/
def polyRows (app : PortApp p) : List Circuit.Expr := memoryRows p ++ appRows p app

end Rows

end NightstreamFPrime.Lifecycle.Nebula
