import CompressionLean.Circuit.Ir
import CompressionLean.Terminal.Outputs
import NightstreamFPrime.Export.PackageExpansion
import NightstreamFPrime.Layout.PilotProduction

namespace CompressionLean.Circuit.Package

open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Layout
open CompressionLean.Shrink (R1csRow)
open CompressionLean.Replay (Gl)

def word (value : NightstreamFPrime.Spec.F) : Gl := Gl.ofNat value.val

def collect : List (ℕ × NightstreamFPrime.Spec.F) → List (ℕ × NightstreamFPrime.Spec.F)
  | (c, x) :: (d, y) :: rest =>
      if c = d then collect ((c, x + y) :: rest) else (c, x) :: collect ((d, y) :: rest)
  | terms => terms
termination_by terms => terms.length

def side (constantColumn : ℕ) (combination : R1CS.LinearCombination) : List (ℕ × Gl) :=
  let terms := ((constantColumn, combination.constant) :: combination.terms).filter (·.2 ≠ 0)
  (collect (terms.mergeSort fun x y => x.1 ≤ y.1)).filterMap fun (c, x) =>
    if x = 0 then none else some (c, word x)

def row (constantColumn : ℕ) (r : R1CS.Row) : R1csRow Gl :=
  ⟨side constantColumn r.a, side constantColumn r.b, side constantColumn r.c⟩

inductive Producer where
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

def schedule (package : CircuitPackage) : List Producer :=
  let hashes := package.hashChains.flatMap fun chain =>
    (List.range (chain.absorbCount + 1)).map (Producer.hash chain)
  (hashes ++ package.permutationInvocations.map Producer.explicit ++
      package.compactRowInvocations.map Producer.compact ++
      package.witnessInstructions.map Producer.generic).mergeSort
    fun p q => p.rowStart package ≤ q.rowStart package

def orderedRows (package : CircuitPackage) : Array (R1csRow Gl) := Id.run do
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

def witnessRole (role : ℕ) : Bool :=
  role == Data.Role.witness || role == Data.Role.piDecWitness ||
    role == Data.Role.runningTransitionWitness ||
    role == PerApplicationPackage.Role.applicationLocal

def callerSegments (layout : PhysicalLayout) : List (Segment × ℕ) :=
  ((layout.privateSegments.filter fun s => !witnessRole s.role).foldl
    (fun (out, offset) s => (out.push (s, offset), offset + s.length)) (#[], 0)).1.toList

def privateInput (segments : List (Segment × ℕ)) (column : ℕ) : Option ℕ :=
  (segments.find? fun (s, _) => s.start ≤ column ∧ column < s.start + s.length).map
    fun (s, offset) => offset + column - s.start

inductive Source where
  | one
  | priv (index : ℕ)
  | pub (index : ℕ)
  | computed (index : ℕ)

structure OutputColumns where
  commitment : Array ℕ
  publicInput : Array ℕ
  point : Array ℕ
  evalK : Array ℕ
  evalA : Array ℕ
  state : Array ℕ

structure Loaded where
  system : Array (R1csRow Gl)
  sources : Array Source
  computed : Array ℕ
  outputs : OutputColumns

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

def load (package : CircuitPackage) (map : Terminal.Outputs) : Except String Loaded := do
  let layout := package.layout
  let segments := callerSegments layout
  unless ((segments.find? (·.1.role == Data.Role.priorPreimage)).map fun (s, offset) =>
      (offset, s.length)) == some (0, PilotProduction.stateHashWords) do
    throw "the prior state preimage is not the first caller input"
  let rows := orderedRows package
  let total := layout.totalColumnCount
  let mark := fun (used : Array Bool) (terms : List (ℕ × Gl)) =>
    terms.foldl (fun used t => used.set! t.1 true) used
  let used := rows.foldl (fun used r => mark (mark (mark used r.a) r.b) r.c)
    (Array.replicate total false)
  let constant := layout.privateColumnCount
  let (index, sources, computed) := Id.run do
    let mut index : Array (Option ℕ) := Array.replicate total none
    let mut sources := #[]
    let mut computed := #[]
    for column in [0:total] do
      if used[column]! then
        index := index.set! column (some sources.size)
        if column = constant then
          sources := sources.push .one
        else if column > constant then
          sources := sources.push (.pub (column - constant - 1))
        else match privateInput segments column with
          | some input => sources := sources.push (.priv input)
          | none =>
              sources := sources.push (.computed computed.size)
              computed := computed.push column
    return (index, sources, computed)
  let reindex := fun (terms : List (ℕ × Gl)) => terms.map fun (c, x) => ((index[c]!).getD 0, x)
  let outputs : OutputColumns := {
    commitment := ← outputColumns layout index map.commitment
    publicInput := ← outputColumns layout index map.publicInput
    point := ← outputColumns layout index map.point
    evalK := ← outputColumns layout index map.eval_K
    evalA := ← outputColumns layout index map.eval_A
    state := ← outputColumns layout index map.state }
  pure {
    system := rows.map fun r => ⟨reindex r.a, reindex r.b, reindex r.c⟩
    sources, computed, outputs }

end CompressionLean.Circuit.Package
