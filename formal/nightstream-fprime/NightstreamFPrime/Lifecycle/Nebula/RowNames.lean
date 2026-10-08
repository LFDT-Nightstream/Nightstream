import NightstreamFPrime.Lifecycle.Nebula.MemoryCircuit

/-! Owns the spec names of the memory application's assertion constraints, as
(name, count) pairs in constraint order. Conformance tests use them to name the
check that a rejected step fails. The counts restate the shapes of
`MemoryApp.assertions` and `Rows.polyRows`; the package emitter checks that
they cover the assertions exactly. They carry no protocol authority. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec.Nebula

namespace Rows

variable (p : Plan)

def opLaneLength : ℕ := 3 + p.μ + 64 + p.wTs

def scanLaneLength : ℕ := 32 + p.wTs

def bitNames : List (String × ℕ) :=
  [("O1 operation lanes", p.bOps * opLaneLength p), ("O1 diff bits", p.bOps * p.wTs),
    ("S1 IS lanes", p.bScan * scanLaneLength p), ("S1 FS lanes", p.bScan * scanLaneLength p),
    ("ts bits", p.wTs), ("segment bits", p.segWidth), ("flag bits", 3)]

def armNames : List (String × ℕ) :=
  [("arm open flag", 2), ("arm segment bound", 1), ("arm idx", 1), ("arm D_seen headers", 12),
    ("arm D_pre", 8), ("arm eta", 4), ("arm eta1 square", 2)]

def slotNames (j : ℕ) : List (String × ℕ) :=
  [(s!"O3 slot {j}", 1), (s!"O4 slot {j}", 1), (s!"O5 slot {j}", 1),
    (s!"O6 slot {j}", ((List.finRange p.μ).filter fun k : Fin p.μ => p.r ≤ k.val).length),
    (s!"O7 slot {j}", opLaneLength p - 1), (s!"O8 slot {j}", 2), (s!"O9 slot {j}", 2)]

def scanNames (j : ℕ) : List (String × ℕ) := [(s!"S2 slot {j}", 2), (s!"S3 slot {j}", 2)]

def closeNames : List (String × ℕ) :=
  [("boundary ts", 1), ("boundary ts bits", 1), ("boundary idx", 1), ("boundary products", 8),
    ("close flag", 2), ("close D_seen.ops = D_pre.ops", 4), ("close D_seen.is = D_mem", 4),
    ("close D_seen.fs = D_pre.fs", 4), ("close product equation", 2), ("close seg_idx", 1),
    ("close D_mem", 4)]

/-- The names of `polyRows`, in row order. -/
def names : List (String × ℕ) :=
  bitNames p ++ armNames ++ (List.range p.bOps).flatMap (slotNames p) ++
    (List.range p.bScan).flatMap scanNames ++ closeNames ++ [("machine", 11)]

end Rows

/-- The names of `MemoryApp.assertions`, in constraint order. -/
def MemoryApp.assertionNames (p : Plan) : List (String × ℕ) :=
  [("state_in", 4), ("state_out", 4), ("chain_ops", 4), ("chain_is", 4), ("chain_fs", 4),
    ("eta", 4)] ++ Rows.names p

end NightstreamFPrime.Lifecycle.Nebula

namespace NightstreamFPrime.Lifecycle.Nebula.Rows

open NightstreamFPrime.Spec.Nebula

theorem sum_flatMap_range (n : ℕ) (l : List ℕ) :
    ((List.range n).flatMap fun _ => l).sum = n * l.sum := by
  induction n with
  | zero => simp
  | succ n ih => simp [List.range_succ, List.flatMap_append, ih, Nat.succ_mul]

/-- The names cover the polynomial rows exactly. -/
theorem names_count (p : Plan) : ((names p).map Prod.snd).sum = (polyRows p).length := by
  simp only [names, bitNames, armNames, slotNames, scanNames, closeNames, List.map_append,
    List.map_flatMap, List.sum_append]
  simp [sum_flatMap_range, polyRows, memoryRows, bitRows, armRows, slotRows, scanRows, closeRows,
    machineRows, opLaneLength, scanLaneLength, Sym.opLane, Sym.scanLane,
    Circuit.Quadratic.KExpr.equalities, List.length_flatMap, List.map_const', List.sum_replicate,
    List.length_finRange, smul_eq_mul]
  ring

end NightstreamFPrime.Lifecycle.Nebula.Rows

namespace NightstreamFPrime.Lifecycle.Nebula.MemoryApp

open NightstreamFPrime.Spec.Nebula

/-- The names cover the assertions exactly. -/
theorem assertionNames_count (p : Plan) (i : AppInterface p) (offset : ℕ) :
    ((assertionNames p).map Prod.snd).sum = (assertions p i offset).length := by
  simp [assertionNames, assertions, Rows.names_count]
  omega

end NightstreamFPrime.Lifecycle.Nebula.MemoryApp
