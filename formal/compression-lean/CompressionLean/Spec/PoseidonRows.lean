import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage

/-!
Owns a fast evaluator of the rows of a Poseidon2 block of F′'s matrix program
(`MatrixProgram.Poseidon.Block.row?`): `invocationRows?` gives the 150 rows of
one invocation.

F′'s trace (`PoseidonSboxPlan.trace`) keeps each lane as a closure over
appended entry lists, so each partial round multiplies the lane sizes by 16.
Here the trace runs once, over the 167 atoms of an invocation: its 16 input
lanes, its 150 S-box outputs and the one column (`atomRows`). Each invocation
then expands its rows from the atoms into forms sorted by column, as Rust's
`normalize_terms` keeps its terms sorted (`matrix_program/form.rs`).

The run check (`Spec/Runs.lean`) reads these rows. The proof that they have
F′'s coefficients is in `Checks/PoseidonRows.lean`.
-/

namespace CompressionLean.Spec.PoseidonRows

open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Layout NightstreamFPrime.Layout.ProductionRelation NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Gadgets.Poseidon2
open Fin.CommRing

/-! ## The trace over the atoms -/

/-- A combination of the atoms of one invocation: input lane `l` is atom `l`,
S-box output `k` is atom `16 + k`, and the one column is atom `166`. -/
abbrev Sym := Vector F 167

namespace Sym

def zero : Sym := Vector.replicate 167 0

def add (s t : Sym) : Sym := Vector.zipWith (· + ·) s t

def scale (k : F) (s : Sym) : Sym := s.map (k * ·)

def atom (a : Nat) : Sym := Vector.ofFn fun i => if i.val = a then 1 else 0

/-- The coefficient of atom `a`, and zero past the atoms. -/
def coef (v : Sym) (a : Nat) : F := if h : a < 167 then v[a] else 0

end Sym

/-- The sixteen lanes of the trace. -/
abbrev State := Vector Sym 16

/-- `SparseLayer.get`. -/
def get (state : State) (index : Nat) : Sym :=
  if h : index < 16 then state[index] else Sym.zero

/-- `SparseLayer.scale`. -/
def scaleNat (value : Nat) (s : Sym) : Sym := Sym.scale (Poseidon2.ofNat value) s

/-- `SparseLayer.addConstant`. -/
def addConstant (s : Sym) (value : F) : Sym := Sym.add s (Sym.scale value (Sym.atom 166))

/-- `SparseLayer.mat4`. -/
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

/-- `SparseLayer.block`. -/
def block (state : State) (index : Nat) : Sym := mat4 state (4 * (index / 4)) (index % 4)

/-- `SparseLayer.sumForms`. -/
def sumForms (forms : List Sym) : Sym := forms.foldl Sym.add Sym.zero

/-- `SparseLayer.column`. -/
def column (state : State) (index : Nat) : Sym :=
  sumForms ((List.range 4).map fun b => block state (4 * b + index % 4))

/-- `SparseLayer.external`. -/
def external (state : State) : State :=
  Vector.ofFn fun lane => Sym.add (block state lane.val) (column state lane.val)

/-- `SparseLayer.sum`. -/
def sum (state : State) : Sym := sumForms ((List.range 16).map (get state))

/-- `SparseLayer.internal`. -/
def internal (state : State) : State :=
  let total := sum state
  Vector.ofFn fun lane => Sym.add
    (Sym.scale (Poseidon2.ofNat (Poseidon2.internalDiagonal.getD lane.val 0)) state[lane]) total

/-- `PoseidonSboxPlan.sboxOutputAt`. -/
def sboxAt (index : Nat) : Sym := if index < 150 then Sym.atom (16 + index) else Sym.zero

/-- One S-box row over the atoms: its input and the S-box output it names. -/
structure Row where
  input : Sym
  output : Nat

/-- `PoseidonSboxPlan.fullRows`. -/
def fullRows (constants : List (List Nat)) (round nextSbox : Nat) (state : State) : List Row :=
  List.ofFn fun lane : Fin 16 =>
    { input := addConstant state[lane] (Poseidon2.constantAt constants round lane.val)
      output := nextSbox + lane.val }

/-- `PoseidonSboxPlan.fullOutput`. -/
def fullOutput (nextSbox : Nat) : State := Vector.ofFn fun lane => sboxAt (nextSbox + lane.val)

/-- `PoseidonSboxPlan.partialRows`. -/
def partialRows (round nextSbox : Nat) (state : State) : List Row :=
  [{ input := addConstant state[0] (Poseidon2.ofNat (Poseidon2.internalConstants.getD round 0))
     output := nextSbox }]

/-- `PoseidonSboxPlan.partialState`. -/
def partialState (nextSbox : Nat) (state : State) : State :=
  Vector.ofFn fun lane => if lane.val = 0 then sboxAt nextSbox else state[lane]

/-- `PoseidonSboxPlan.StepResult`. -/
structure StepResult where
  nextSbox : Nat
  state : State
  rows : List Row

/-- `PoseidonSboxPlan.compileStep`. -/
def compileStep (nextSbox : Nat) (state : State) : Permutation.Step → StepResult
  | .initialLayer => ⟨nextSbox, external state, []⟩
  | .initialFullRound round => ⟨nextSbox + 16, external (fullOutput nextSbox),
      fullRows Poseidon2.initialConstants round nextSbox state⟩
  | .partialRound round => ⟨nextSbox + 1, internal (partialState nextSbox state),
      partialRows round nextSbox state⟩
  | .terminalFullRound round => ⟨nextSbox + 16, external (fullOutput nextSbox),
      fullRows Poseidon2.terminalConstants round nextSbox state⟩

/-- `PoseidonSboxPlan.compile`. -/
def compile : Nat → State → List Permutation.Step → StepResult
  | nextSbox, state, [] => ⟨nextSbox, state, []⟩
  | nextSbox, state, step :: rest =>
      let head := compileStep nextSbox state step
      let tail := compile head.nextSbox head.state rest
      ⟨tail.nextSbox, tail.state, head.rows ++ tail.rows⟩

/-- The rows of one permutation over the atoms (`PoseidonSboxPlan.trace`). -/
def atomRows : Array Row :=
  (compile 0 (Vector.ofFn fun lane => Sym.atom lane.val) Permutation.schedule).rows.toArray

/-! ## Sorted forms -/

/-- The entries of a form. -/
abbrev Entries (n : Nat) := List (SparseEntry n)

/-- The sum of two forms. When both are sorted by column, so is the sum. -/
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

/-- Adjacent entries of one column summed, and zero entries dropped. -/
def combine {n : Nat} : Entries n → Entries n
  | [] => []
  | e :: rest =>
      match combine rest with
      | [] => if e.coefficient = 0 then [] else [e]
      | h :: t =>
          if h.column = e.column then
            if e.coefficient + h.coefficient = 0 then t else ⟨e.column, e.coefficient + h.coefficient⟩ :: t
          else if e.coefficient = 0 then h :: t else e :: h :: t

/-- A form sorted by column, with one nonzero entry per column. -/
def normalize {n : Nat} (form : SparseForm n) : Entries n :=
  combine (form.entries.mergeSort fun a b => a.column.val ≤ b.column.val)

/-- The entries times `k`. -/
def scaled {n : Nat} (k : F) (xs : Entries n) : Entries n := (SparseForm.scale k ⟨xs⟩).entries

/-- `3^t`, the weight of coordinate `t` of a retained field slot (`RetainedSlot.recomposeForms`). -/
def powers : Vector F 41 := Vector.ofFn fun t => Phi81Relation.PiDECAlgebra.Radix.fieldOfNat 3 ^ t.val

/-- `3^t`. -/
def power (t : Nat) : F :=
  if h : t < 41 then powers[t] else Phi81Relation.PiDECAlgebra.Radix.fieldOfNat 3 ^ t

/-- The atoms of one invocation as sorted forms. -/
structure Atoms (n : Nat) where
  inputs : Vector (Entries n) 16
  sbox : Vector (Entries n) 150
  one : Fin n

/-- Atom `a` as a form. -/
def Atoms.form {n : Nat} (atoms : Atoms n) (a : Nat) : SparseForm n :=
  if a < 16 then ⟨atoms.inputs[a]?.getD []⟩
  else if a < 166 then ⟨atoms.sbox[a - 16]?.getD []⟩
  else SparseForm.singleton atoms.one 1

/-- The form of a combination of the atoms: the input lanes merged, the S-box
outputs in order, then the one column. -/
def Atoms.expand {n : Nat} (atoms : Atoms n) (v : Sym) : Entries n :=
  let inputs := (List.range 16).foldl (fun acc l =>
    if Sym.coef v l = 0 then acc else merge acc (scaled (Sym.coef v l) (atoms.inputs[l]?.getD []))) []
  let sboxes := (List.range 150).flatMap fun k =>
    if Sym.coef v (16 + k) = 0 then [] else scaled (Sym.coef v (16 + k)) (atoms.sbox[k]?.getD [])
  let one := if Sym.coef v 166 = 0 then [] else [⟨atoms.one, Sym.coef v 166⟩]
  merge inputs (merge sboxes one)

/-- One row as F′'s S-box row forms. -/
def Atoms.row {n : Nat} (atoms : Atoms n) (row : Row) : SboxRow.Forms n :=
  { input := ⟨atoms.expand row.input⟩, output := ⟨atoms.sbox[row.output]?.getD []⟩ }

/-! ## One invocation -/

/-- The coordinates of slot `slot` of a retained field block with their weights. -/
def slotEntries {sourceWidth n : Nat} (block : LowNormBlock.Block sourceWidth) (start : Nat)
    (fits : start + block.coordinateCount ≤ n) (slot : Fin block.slotCount) : Entries n :=
  List.ofFn fun t : Fin block.kind.width => ⟨block.column start fits slot t, power t.val⟩

/-- The atoms of invocation `inv` of `block`, for its input lanes `input`. -/
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

/-- The 150 rows of invocation `inv` of `block`, in F′'s order: row `r` is
ordinal `150 * inv + r` of `block.row?`. `none` where `block.row?` is. -/
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
