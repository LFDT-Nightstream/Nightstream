import CompressionLean.Circuit.Ir
import CompressionLean.Fields
import CompressionLean.Terminal.Outputs
import NightstreamFPrime.Export.PackageExpansion
import NightstreamFPrime.Layout.PilotProduction

/-!
Owns the terminal package as `Package::apply` reads it
(`crates/nightstream/src/lifecycle/terminal/package.rs:86-160`): the package
rows as one system, the source of each system column, and the output map in
system columns. Each definition is a port of the Rust that builds the same
value; the `terminalCheck` executable compares the two on the artifact.

- Rows (`ordered_r1cs`, `nightstream-fprime/src/package/r1cs.rs:305-383`):
  every producer of `scheduled_witnesses` (`package.rs:629-691`) by row
  start, each after the assertion rows that start before it. A producer's
  rows are F′'s instantiation (`RowSemantics.lean`), so they are the rows of
  `CircuitPackage.expandedRows`.
- Sides (`finish_row`, `r1cs.rs:236-252`): the constant on the constant
  column, the terms by column, equal columns summed, zero sums dropped.
- Columns: the package columns that some row names, in increasing order.
  System column `k` is the `k`-th of them.
-/

namespace CompressionLean.Circuit.Package

open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Layout
open CompressionLean.Shrink (R1csRow)
open CompressionLean.Replay (Word)

/-- Sum the coefficients of adjacent equal columns. -/
def collect : List (ℕ × NightstreamFPrime.Spec.F) → List (ℕ × NightstreamFPrime.Spec.F)
  | (c, x) :: (d, y) :: rest =>
      if c = d then collect ((c, x + y) :: rest) else (c, x) :: collect ((d, y) :: rest)
  | terms => terms
termination_by terms => terms.length

/-- One side as `finish_row` writes it. -/
def side (constantColumn : ℕ) (combination : R1CS.LinearCombination) : List (ℕ × Word) :=
  let terms := ((constantColumn, combination.constant) :: combination.terms).filter (·.2 ≠ 0)
  (collect (terms.mergeSort fun x y => x.1 ≤ y.1)).filterMap fun (c, x) =>
    if x = 0 then none else some (c, Word.ofF x)

def row (constantColumn : ℕ) (r : R1CS.Row) : R1csRow Word :=
  ⟨side constantColumn r.a, side constantColumn r.b, side constantColumn r.c⟩

/-- One producer of rows (`ScheduledWitness`,
`nightstream-fprime/src/package.rs:264-301`). The terminal package has no
application rows. -/
inductive Producer where
  /-- Invocation `ordinal` of a hash chain. -/
  | hash (chain : HashChain) (ordinal : ℕ)
  | explicit (invocation : PermutationInvocation)
  | compact (invocation : CompactRowInvocation)
  | generic (instruction : WitnessInstruction)

def Producer.rowStart (package : CircuitPackage) : Producer → ℕ
  | .hash chain ordinal => chain.rowStart + ordinal * package.permutation.rows.length
  | .explicit invocation => invocation.rowStart
  | .compact invocation => invocation.rowStart
  | .generic instruction => instruction.rowIndex

def Producer.rows (package : CircuitPackage) : Producer → List R1CS.Row
  | .hash chain ordinal => package.permutation.rows.map (instantiateRow package chain ordinal)
  | .explicit invocation => permutationInvocationRows package invocation
  | .compact invocation => compactInvocationRows package invocation
  | .generic instruction => [instruction.toR1CS]

/-- Every producer by row start (`scheduled_witnesses`). -/
def schedule (package : CircuitPackage) : List Producer :=
  let hashes := package.hashChains.flatMap fun chain =>
    (List.range (chain.absorbCount + 1)).map (Producer.hash chain)
  (hashes ++ package.permutationInvocations.map Producer.explicit ++
      package.compactRowInvocations.map Producer.compact ++
      package.witnessInstructions.map Producer.generic).mergeSort
    fun p q => p.rowStart package ≤ q.rowStart package

/-- The package rows in schedule order, over package columns
(`expand_matrix` with `Placement::Ordered`). -/
def orderedRows (package : CircuitPackage) : Array (R1csRow Word) := Id.run do
  let constant := package.layout.constantColumn
  let mut out := #[]
  let mut assertions := package.assertionRows
  for producer in schedule package do
    let start := producer.rowStart package
    for assertion in assertions.takeWhile (·.rowIndex < start) do
      out := out.push (row constant assertion.toR1CS)
    assertions := assertions.dropWhile (·.rowIndex < start)
    for r in producer.rows package do
      out := out.push (row constant r)
  for assertion in assertions do
    out := out.push (row constant assertion.toR1CS)
  return out

/-- A private segment whose columns the witness program writes
(`is_witness_role`, `nightstream-fprime/src/package/v1_2.rs:86-91`). -/
def witnessRole (role : ℕ) : Bool :=
  role == Data.Role.witness || role == Data.Role.piDecWitness ||
    role == Data.Role.runningTransitionWitness ||
    role == PerApplicationPackage.Role.applicationLocal

/-- The caller segments with the private input index of their first column. -/
def callerSegments (layout : PhysicalLayout) : List (Segment × ℕ) :=
  ((layout.privateSegments.filter fun s => !witnessRole s.role).foldl
    (fun (out, offset) s => (out.push (s, offset), offset + s.length)) (#[], 0)).1.toList

/-- The private input that holds `column` (`private_input_index`,
`nightstream-fprime/src/package.rs:373-385`). -/
def privateInput (segments : List (Segment × ℕ)) (column : ℕ) : Option ℕ :=
  (segments.find? fun (s, _) => s.start ≤ column ∧ column < s.start + s.length).map
    fun (s, offset) => offset + column - s.start

/-- Where the value of one system column comes from (`Source`,
`package.rs:44-52`). -/
inductive Source where
  | one
  /-- Private input `index` of `execute_witness`. -/
  | priv (index : ℕ)
  /-- Public input `index`. -/
  | pub (index : ℕ)
  /-- Hint output `index`. -/
  | computed (index : ℕ)

/-- The output map in system columns (`TerminalOutputs`). -/
structure OutputColumns where
  commitment : Array ℕ
  publicInput : Array ℕ
  point : Array ℕ
  evalK : Array ℕ
  evalA : Array ℕ
  state : Array ℕ

/-- The package as `apply` reads it (`Package`, `package.rs:54-62`). -/
structure Loaded where
  /-- The rows over system columns. -/
  system : Array (R1csRow Word)
  sources : Array Source
  /-- The package column of each computed value, in hint order. -/
  computed : Array ℕ
  outputs : OutputColumns

/-- The system columns of an output list: each entry is one package column
that some row names (`decode_outputs`, `terminal.rs`; `map`, `package.rs`). -/
def outputColumns (layout : PhysicalLayout) (index : Array (Option ℕ))
    (list : List NightstreamFPrime.Circuit.Expr) : Except String (Array ℕ) :=
  list.toArray.mapM fun
    | .var c =>
        if c < layout.totalColumnCount ∧ c ≠ layout.constantColumn then
          match index[c]! with
          | some k => pure k
          | none => throw "terminal output column without a row"
        else throw "terminal output column"
    | _ => throw "terminal output expression"

/-- Mark the columns of `terms`. -/
def mark (used : Array Bool) (terms : List (ℕ × Word)) : Array Bool :=
  terms.foldl (fun used t => used.set! t.1 true) used

/-- The package columns below `total` that some row names. -/
def named (total : ℕ) (rows : Array (R1csRow Word)) : Array Bool :=
  rows.foldl (fun used r => mark (mark (mark used r.a) r.b) r.c) (Array.replicate total false)

/-- The numbering so far: the system column of each package column, the source
of each system column, and the package column of each computed value. -/
abbrev Numbering := Array (Option ℕ) × Array Source × Array ℕ

/-- Number one package column: a named column gets the next system column and
its source. -/
def numberColumn (layout : PhysicalLayout) (segments : List (Segment × ℕ)) (used : Array Bool)
    (state : Numbering) (column : ℕ) : Numbering :=
  let (index, sources, computed) := state
  let constant := layout.privateColumnCount
  if used[column]! then
    let index := index.set! column (some sources.size)
    if column = constant then (index, sources.push .one, computed)
    else if column > constant then (index, sources.push (.pub (column - constant - 1)), computed)
    else match privateInput segments column with
      | some input => (index, sources.push (.priv input), computed)
      | none => (index, sources.push (.computed computed.size), computed.push column)
  else state

/-- The system columns: the named package columns in increasing order. -/
def number (layout : PhysicalLayout) (segments : List (Segment × ℕ)) (used : Array Bool) :
    Numbering := Id.run do
  let mut state : Numbering := (Array.replicate layout.totalColumnCount none, #[], #[])
  for column in [0:layout.totalColumnCount] do
    state := numberColumn layout segments used state column
  return state

/-- Terms over package columns, moved to system columns. -/
def reindex (index : Array (Option ℕ)) (terms : List (ℕ × Word)) : List (ℕ × Word) :=
  terms.map fun (c, x) => ((index[c]!).getD 0, x)

/-- `load` (`package.rs:86-160`) on the Lean package and its output map. -/
def load (package : CircuitPackage) (map : Terminal.Outputs) : Except String Loaded := do
  let layout := package.layout
  let segments := callerSegments layout
  -- `apply` reads the first caller inputs as the prior state preimage.
  unless ((segments.find? (·.1.role == Data.Role.priorPreimage)).map fun (s, offset) =>
      (offset, s.length)) == some (0, PilotProduction.stateHashWords) do
    throw "the prior state preimage is not the first caller input"
  let rows := orderedRows package
  let (index, sources, computed) := number layout segments (named layout.totalColumnCount rows)
  let outputs : OutputColumns := {
    commitment := ← outputColumns layout index map.commitment
    publicInput := ← outputColumns layout index map.publicInput
    point := ← outputColumns layout index map.point
    evalK := ← outputColumns layout index map.eval_K
    evalA := ← outputColumns layout index map.eval_A
    state := ← outputColumns layout index map.state }
  pure {
    system := rows.map fun r => ⟨reindex index r.a, reindex index r.b, reindex index r.c⟩
    sources, computed, outputs }

end CompressionLean.Circuit.Package
