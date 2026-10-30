import CompressionLean.Ports.Shrink.Layout
import NightstreamFPrime.Gadgets.Poseidon2.Permutation
import CompressionLean.Lib.List

namespace CompressionLean.Circuit.Block

open CompressionLean.Lib
open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Gadgets.Poseidon2
open CompressionLean.Shrink
open Fin.CommRing

structure Linear where
  terms : List (ℕ × F)
  constant : F

namespace Linear

def cell (k : ℕ) : Linear := ⟨[(k, 1)], 0⟩

def zero : Linear := ⟨[], 0⟩

def addTerm : List (ℕ × F) → ℕ × F → List (ℕ × F)
  | [], t => [t]
  | u :: rest, t => if u.1 = t.1 then (u.1, u.2 + t.2) :: rest else u :: addTerm rest t

def mergePart (acc : List (ℕ × F) × F) (part : Linear × F) : List (ℕ × F) × F :=
  (part.1.terms.foldl (fun terms t => addTerm terms (t.1, t.2 * part.2)) acc.1,
    acc.2 + part.1.constant * part.2)

def combine (parts : List (Linear × F)) : Linear :=
  let merged := parts.foldl mergePart ([], 0)
  ⟨(merged.1.filter fun t => t.2 ≠ 0).mergeSort fun a b => a.1 ≤ b.1, merged.2⟩

def eval (c : ℕ → F) (l : Linear) : F :=
  l.constant + (l.terms.map fun t => t.2 * c t.1).sum

end Linear

def m4 : List (List F) := [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]]

def mixLinear (state : List Linear) : List Linear :=
  (List.range 16).map fun lane =>
    Linear.combine ((List.range 4).map fun j =>
      (state.getD (4 * (lane / 4) + j) .zero, (m4.getD (lane % 4) []).getD j 0))

def externalLinear (state : List Linear) : List Linear :=
  let mixed := mixLinear state
  (List.range 16).map fun lane =>
    Linear.combine ((mixed.getD lane .zero, 1) ::
      (List.range 4).map fun chunk => (mixed.getD (4 * chunk + lane % 4) .zero, 1))

structure Build where
  state : List Linear
  inputs : List Linear

def fullRound (rows : List (List ℕ)) (round : ℕ) (b : Build) : Build :=
  ⟨externalLinear ((List.range 16).map fun lane => .cell (16 + b.inputs.length + lane)),
    b.inputs ++ (List.range 16).map fun lane =>
      ⟨(b.state.getD lane .zero).terms,
        (b.state.getD lane .zero).constant + Spec.Poseidon2.constantAt rows round lane⟩⟩

def partialLanes (b : Build) : List Linear :=
  (List.range 16).map fun lane =>
    if lane = 0 then .cell (16 + b.inputs.length) else b.state.getD lane .zero

def partialRound (round : ℕ) (b : Build) : Build :=
  let sum := Linear.combine ((partialLanes b).map fun form => (form, 1))
  ⟨(List.range 16).map fun lane =>
      Linear.combine [(sum, 1), ((partialLanes b).getD lane .zero,
        Spec.Poseidon2.ofNat (Spec.Poseidon2.internalDiagonal.getD lane 0))],
    b.inputs ++ [⟨(b.state.getD 0 .zero).terms, (b.state.getD 0 .zero).constant +
      Spec.Poseidon2.ofNat (Spec.Poseidon2.internalConstants.getD round 0)⟩]⟩

def step (b : Build) : Permutation.Step → Build
  | .initialLayer => ⟨externalLinear b.state, b.inputs⟩
  | .initialFullRound round => fullRound Spec.Poseidon2.initialConstants round b
  | .partialRound round => partialRound round b
  | .terminalFullRound round => fullRound Spec.Poseidon2.terminalConstants round b

def start : Build := ⟨(List.range 16).map .cell, []⟩

def build : Build := Permutation.schedule.foldl step start

def sboxes : ℕ := 150

def output : ℕ := 16 + sboxes

def push (matrix : Fin 8) (row : ℕ) (form : Linear) (sign : F) : List (TemplateEntry F) :=
  form.terms.map (fun t => ⟨matrix, row, some t.1, sign * t.2⟩) ++
    if form.constant ≠ 0 then [⟨matrix, row, none, sign * form.constant⟩] else []

def rowCell (row : ℕ) : ℕ := if row < sboxes then 16 + row else output + row - sboxes

def entries (sboxInputs outputs : List Linear) : List (TemplateEntry F) :=
  (sboxInputs.zipIdx.flatMap fun p => push 0 p.2 p.1 1) ++
    (outputs.zipIdx.flatMap fun p => push 7 (sboxes + p.2) p.1 (-1)) ++
    (List.range (sboxes + 16)).map fun row => ⟨7, row, some (rowCell row), 1⟩

def poseidonTemplate : Template F := ⟨sboxes + 16, output + 16, entries build.inputs build.state⟩

end CompressionLean.Circuit.Block
