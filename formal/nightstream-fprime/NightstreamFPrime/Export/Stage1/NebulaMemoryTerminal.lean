import NightstreamFPrime.Lifecycle.Nebula.RunLink

/-! Owns the spec §13 terminal-check fixture that the Rust memory verifier
must match: a fixed list of statements and final carries per plan, with each
`TerminalChecks` field decided (`terminalChecks_iff`), whether every statement
integer that becomes a field word is below `q` (Ob1), and the final state
words. It does not own the checks (`Lifecycle/Nebula/RunLink.lean`). -/

namespace NightstreamFPrime.Export.Stage1.NebulaMemoryTerminal

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Lifecycle (natWord)
open NightstreamFPrime.Lifecycle.Nebula

/-- The `TerminalChecks` fields after `state`, decided, under their fixture
names. -/
def decisions (p : Geometry) (carry : Fin 39 → F) (steps segments finalTs : ℕ) (finalRoot : Digest) :
    List (String × Bool) :=
  [("closed", decide ((decodeCarry carry).idx = p.n)),
    ("segments_range", decide (1 ≤ segments ∧ segments ≤ p.sMax)),
    ("step_count", decide (steps = segments * p.n)),
    ("seg_idx", decide ((decodeCarry carry).segIdx = segments)),
    ("ts", decide ((decodeCarry carry).ts = finalTs)),
    ("root", decide ((decodeCarry carry).memRoot = finalRoot))]

/-- The decisions are the fields of `TerminalChecks`, in order: all hold
exactly when the checks hold on the state words of `finalApp` and `carry`. -/
theorem terminalChecks_iff {width : ℕ} (p : Geometry) (finalApp : Fin width → F)
    (carry : Fin 39 → F) (steps segments finalTs : ℕ) (finalRoot : Digest) :
    TerminalChecks p (stateWords (List.ofFn finalApp) (List.ofFn carry)) finalApp carry steps
        segments finalTs finalRoot ↔
      ∀ d ∈ decisions p carry steps segments finalTs finalRoot, d.2 = true := by
  simp only [decisions, List.forall_mem_cons, List.mem_nil_iff, false_implies, implies_true,
    and_true, decide_eq_true_eq]
  exact ⟨fun h => ⟨h.closed, h.segmentsRange, h.stepCount, h.segIdx, h.ts, h.root⟩,
    fun ⟨closed, range, count, segIdx, ts, root⟩ => ⟨rfl, closed, range, count, segIdx, ts, root⟩⟩

/-- One fixture case: the statement, with the application words `(pc, acc)`
as integers, and the prover's final carry words. -/
structure Case where
  name : String
  steps : ℕ
  initial : ℕ × ℕ
  final : ℕ × ℕ
  segments : ℕ
  finalTs : ℕ
  finalRoot : List F
  carry : List F

/-- Carry words with these counters and `D_mem`. The 32 words between them
are distinct, so a Rust decoder that moves a word fails its round trip. -/
def carryWordsOf (segIdx idx ts : ℕ) (root : List F) : List F :=
  [natWord segIdx, natWord idx, natWord ts] ++ (List.range 32).map (fun i => natWord (i + 100)) ++
    root

/-- Both ends of the accepted segment range, one case for each decided field
where only that field fails, and four statements with an integer at or above
`q`. -/
def cases (p : Plan) : List Case :=
  let q := goldilocksModulus
  let root := initRootWords p
  let closed (segments : ℕ) : Case :=
    let ts := segments * p.n * p.bOps
    ⟨"", segments * p.n, (0, 0), (3, 5), segments, ts, root, carryWordsOf segments p.n ts root⟩
  let one := closed 1
  [{ one with name := "accept: one segment" },
    { closed p.sMax with name := "accept: S_max segments" },
    { one with name := "closed: idx = N - 1", carry := carryWordsOf 1 (p.n - 1) one.finalTs root },
    { closed 0 with name := "segments_range: no segment" },
    { closed (p.sMax + 1) with name := "segments_range: S_max + 1 segments" },
    { one with name := "step_count: S * N + 1 steps", steps := one.steps + 1 },
    { one with name := "seg_idx: carry one segment ahead", carry := carryWordsOf 2 p.n one.finalTs root },
    { one with name := "ts: statement timestamp + 1", finalTs := one.finalTs + 1 },
    { one with name := "root: one word changed", finalRoot := root.modifyHead (· + 1) },
    { one with name := "non-canonical: final pc = q", final := (q, 5) },
    { one with name := "non-canonical: initial acc = q", initial := (0, q) },
    { one with name := "non-canonical: final pc = honest + q", final := (3 + q, 5) },
    { one with name := "non-canonical: final_ts = q", finalTs := q }]

private def wordsJson (words : List F) : String :=
  "[" ++ ",".intercalate (words.map fun w => toString w.val) ++ "]"

private def natsJson (values : List ℕ) : String :=
  "[" ++ ",".intercalate (values.map toString) ++ "]"

/-- The fixture entry of one case. The application and carry words are
functions over word lists, as `TerminalChecks` reads them. -/
def Case.json (p : Geometry) (c : Case) : String :=
  let finalApp : Fin 2 → F := fun i => [natWord c.final.1, natWord c.final.2].getD i 0
  let carry : Fin 39 → F := fun i => c.carry.getD i 0
  let canonical :=
    [c.initial.1, c.initial.2, c.final.1, c.final.2, c.finalTs].all (· < goldilocksModulus)
  let checks := (decisions p carry c.steps c.segments c.finalTs (ofWords c.finalRoot)).map
    fun (name, holds) => "\"" ++ name ++ "\":" ++ toString holds
  "{\"name\":\"" ++ c.name ++ "\",\"statement\":{\"steps\":" ++ toString c.steps ++
    ",\"initial\":" ++ natsJson [c.initial.1, c.initial.2] ++
    ",\"final\":" ++ natsJson [c.final.1, c.final.2] ++
    ",\"segments\":" ++ toString c.segments ++ ",\"final_ts\":" ++ toString c.finalTs ++
    ",\"final_root\":" ++ wordsJson c.finalRoot ++ "},\"carry\":" ++ wordsJson c.carry ++
    ",\"canonical\":" ++ toString canonical ++ ",\"checks\":{" ++ ",".intercalate checks ++
    "},\"final_state\":" ++ wordsJson (stateWords (List.ofFn finalApp) (List.ofFn carry)) ++ "}"

/-- The fixture of `p`, one case per line. -/
def fixture (p : Plan) : String :=
  "{\"cases\":[\n" ++ ",\n".intercalate ((cases p).map (Case.json p.toGeometry)) ++ "\n]}\n"

end NightstreamFPrime.Export.Stage1.NebulaMemoryTerminal
