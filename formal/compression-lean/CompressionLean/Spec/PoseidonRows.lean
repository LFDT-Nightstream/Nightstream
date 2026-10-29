import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage

namespace CompressionLean.Spec.PoseidonRows

open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Layout NightstreamFPrime.Layout.ProductionRelation NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Gadgets.Poseidon2
open Fin.CommRing

abbrev Sym := Vector F 167

namespace Sym

def zero : Sym := Vector.replicate 167 0

def add (s t : Sym) : Sym := Vector.zipWith (· + ·) s t

def scale (k : F) (s : Sym) : Sym := s.map (k * ·)

def atom (a : Nat) : Sym := Vector.ofFn fun i => if i.val = a then 1 else 0

def coef (v : Sym) (a : Nat) : F := if h : a < 167 then v[a] else 0

end Sym

abbrev State := Vector Sym 16

def get (state : State) (index : Nat) : Sym :=
  if h : index < 16 then state[index] else Sym.zero

def scaleNat (value : Nat) (s : Sym) : Sym := Sym.scale (Poseidon2.ofNat value) s

def addConstant (s : Sym) (value : F) : Sym := Sym.add s (Sym.scale value (Sym.atom 166))

def mat4 (state : State) (base lane : Nat) : Sym :=
  match lane with
  | 0 => Sym.add (Sym.add (Sym.add (scaleNat 2 (get state base))
      (scaleNat 3 (get state (base + 1)))) (get state (base + 2))) (get state (base + 3))
  | 1 => Sym.add (Sym.add (Sym.add (get state base)
      (scaleNat 2 (get state (base + 1)))) (scaleNat 3 (get state (base + 2)))) (get state (base + 3))
  | 2 => Sym.add (Sym.add (Sym.add (get state base) (get state (base + 1)))
      (scaleNat 2 (get state (base + 2)))) (scaleNat 3 (get state (base + 3)))
  | _ => Sym.add (Sym.add (Sym.add (scaleNat 3 (get state base))
      (get state (base + 1))) (get state (base + 2))) (scaleNat 2 (get state (base + 3)))

def block (state : State) (index : Nat) : Sym := mat4 state (4 * (index / 4)) (index % 4)

def sumForms (forms : List Sym) : Sym := forms.foldl Sym.add Sym.zero

def column (state : State) (index : Nat) : Sym :=
  sumForms ((List.range 4).map fun b => block state (4 * b + index % 4))

def external (state : State) : State :=
  Vector.ofFn fun lane => Sym.add (block state lane.val) (column state lane.val)

def sum (state : State) : Sym := sumForms ((List.range 16).map (get state))

def internal (state : State) : State :=
  let total := sum state
  Vector.ofFn fun lane => Sym.add
    (Sym.scale (Poseidon2.ofNat (Poseidon2.internalDiagonal.getD lane.val 0)) state[lane]) total

def sboxAt (index : Nat) : Sym := if index < 150 then Sym.atom (16 + index) else Sym.zero

structure Row where
  input : Sym
  output : Nat

def fullRows (constants : List (List Nat)) (round nextSbox : Nat) (state : State) : List Row :=
  List.ofFn fun lane : Fin 16 =>
    { input := addConstant state[lane] (Poseidon2.constantAt constants round lane.val)
      output := nextSbox + lane.val }

def fullOutput (nextSbox : Nat) : State := Vector.ofFn fun lane => sboxAt (nextSbox + lane.val)

def partialRows (round nextSbox : Nat) (state : State) : List Row :=
  [{ input := addConstant state[0] (Poseidon2.ofNat (Poseidon2.internalConstants.getD round 0))
     output := nextSbox }]

def partialState (nextSbox : Nat) (state : State) : State :=
  Vector.ofFn fun lane => if lane.val = 0 then sboxAt nextSbox else state[lane]

structure StepResult where
  nextSbox : Nat
  state : State
  rows : List Row

def compileStep (nextSbox : Nat) (state : State) : Permutation.Step → StepResult
  | .initialLayer => ⟨nextSbox, external state, []⟩
  | .initialFullRound round => ⟨nextSbox + 16, external (fullOutput nextSbox),
      fullRows Poseidon2.initialConstants round nextSbox state⟩
  | .partialRound round => ⟨nextSbox + 1, internal (partialState nextSbox state),
      partialRows round nextSbox state⟩
  | .terminalFullRound round => ⟨nextSbox + 16, external (fullOutput nextSbox),
      fullRows Poseidon2.terminalConstants round nextSbox state⟩

def compile : Nat → State → List Permutation.Step → StepResult
  | nextSbox, state, [] => ⟨nextSbox, state, []⟩
  | nextSbox, state, step :: rest =>
      let head := compileStep nextSbox state step
      let tail := compile head.nextSbox head.state rest
      ⟨tail.nextSbox, tail.state, head.rows ++ tail.rows⟩

def atomRows : Array Row :=
  (compile 0 (Vector.ofFn fun lane => Sym.atom lane.val) Permutation.schedule).rows.toArray

abbrev Entries (n : Nat) := List (SparseEntry n)

def merge {n : Nat} (xs ys : Entries n) : Entries n :=
  match xs, ys with
  | [], ys => ys
  | x :: xs, [] => x :: xs
  | x :: xs, y :: ys =>
      if x.column.val < y.column.val then x :: merge xs (y :: ys)
      else if y.column.val < x.column.val then y :: merge (x :: xs) ys
      else if x.coefficient + y.coefficient = 0 then merge xs ys
      else ⟨x.column, x.coefficient + y.coefficient⟩ :: merge xs ys
termination_by xs.length + ys.length

def combine {n : Nat} : Entries n → Entries n
  | [] => []
  | e :: rest =>
      match combine rest with
      | [] => if e.coefficient = 0 then [] else [e]
      | h :: t =>
          if h.column = e.column then
            if e.coefficient + h.coefficient = 0 then t else ⟨e.column, e.coefficient + h.coefficient⟩ :: t
          else if e.coefficient = 0 then h :: t else e :: h :: t

def normalize {n : Nat} (form : SparseForm n) : Entries n :=
  combine (form.entries.mergeSort fun a b => a.column.val ≤ b.column.val)

def scaled {n : Nat} (k : F) (xs : Entries n) : Entries n := (SparseForm.scale k ⟨xs⟩).entries

def powers : Vector F 41 := Vector.ofFn fun t => Phi81Relation.PiDECAlgebra.Radix.fieldOfNat 3 ^ t.val

def power (t : Nat) : F :=
  if h : t < 41 then powers[t] else Phi81Relation.PiDECAlgebra.Radix.fieldOfNat 3 ^ t

structure Atoms (n : Nat) where
  inputs : Vector (Entries n) 16
  sbox : Vector (Entries n) 150
  one : Fin n

def Atoms.form {n : Nat} (atoms : Atoms n) (a : Nat) : SparseForm n :=
  if a < 16 then ⟨atoms.inputs[a]?.getD []⟩
  else if a < 166 then ⟨atoms.sbox[a - 16]?.getD []⟩
  else SparseForm.singleton atoms.one 1

def Atoms.expand {n : Nat} (atoms : Atoms n) (v : Sym) : Entries n :=
  let inputs := (List.range 16).foldl (fun acc l =>
    if Sym.coef v l = 0 then acc else merge acc (scaled (Sym.coef v l) (atoms.inputs[l]?.getD []))) []
  let sboxes := (List.range 150).flatMap fun k =>
    if Sym.coef v (16 + k) = 0 then [] else scaled (Sym.coef v (16 + k)) (atoms.sbox[k]?.getD [])
  let one := if Sym.coef v 166 = 0 then [] else [⟨atoms.one, Sym.coef v 166⟩]
  merge inputs (merge sboxes one)

def Atoms.row {n : Nat} (atoms : Atoms n) (row : Row) : SboxRow.Forms n :=
  { input := ⟨atoms.expand row.input⟩, output := ⟨atoms.sbox[row.output]?.getD []⟩ }

def slotEntries {sourceWidth n : Nat} (block : LowNormBlock.Block sourceWidth) (start : Nat)
    (fits : start + block.coordinateCount ≤ n) (slot : Fin block.slotCount) : Entries n :=
  List.ofFn fun t : Fin block.kind.width => ⟨block.column start fits slot t, power t.val⟩

def atomsOf (block : Poseidon.Block) (n : Nat) (oneBound : block.oneColumn < n)
    (slotCountEq : block.retained.slotCount = block.invocationCount * 150)
    (fits : block.retained.start + block.retained.coordinateCount ≤ n)
    (input : PoseidonSboxPlan.State n) (inv : Fin block.invocationCount) : Atoms n where
  inputs := Vector.ofFn fun l => normalize (input l)
  sbox := Vector.ofFn fun k => slotEntries block.retained.semantic block.retained.start fits
    ⟨150 * inv.val + k.val, by
      show _ < block.retained.slotCount
      rw [slotCountEq]
      omega⟩
  one := ⟨block.oneColumn, oneBound⟩

def invocationRows? (block : Poseidon.Block) (n inv : Nat) : Option (Array (SboxRow.Forms n)) :=
  if invBound : inv < block.invocationCount then
    if oneBound : block.oneColumn < n then
      if block.retained.kind = .field then
        if slotCountEq : block.retained.slotCount = block.invocationCount * 150 then
          if fits : block.retained.start + block.retained.coordinateCount ≤ n then
            (block.input.state? n block.oneColumn inv).map fun input =>
              let atoms := atomsOf block n oneBound slotCountEq fits input ⟨inv, invBound⟩
              atomRows.map atoms.row
          else none
        else none
      else none
    else none
  else none

end CompressionLean.Spec.PoseidonRows
