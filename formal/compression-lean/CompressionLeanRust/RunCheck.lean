import NeoVerified.Funs

namespace CompressionLeanRust.RunCheck

open Aeneas Aeneas.Std

def usize (n : ℕ) : Usize := ⟨BitVec.ofNat _ n⟩

def u32 (n : ℕ) : U32 := ⟨BitVec.ofNat _ n⟩

def u16 (n : ℕ) : U16 := ⟨BitVec.ofNat _ n⟩

def outcome {α : Type} (r : Result α) (render : α → String) : String :=
  match r.match with
  | .ok a => s!"ok {render a}"
  | .vis _ _ => "fail"
  | .div => "div"

def option {α : Type} (o : Option α) (render : α → String) : String :=
  match o with
  | some a => s!"some {render a}"
  | none => "none"

def value {ty : UScalarTy} (x : UScalar ty) : String := toString x.val

open neo_verified in
def shape (b r c p : ℕ) : layout.Shape := ⟨Aeneas.Std.Array.make 1#usize [usize b], usize r, usize c, usize p⟩

open neo_verified in
def shapeWords (s : layout.Shape) : String :=
  " ".intercalate ((s.blocks.val ++ [s.glue_rows, s.glue_cells, s.publics]).map value)

open neo_verified in
def wire : List ℕ → Option circuit.Wire
  | [0, c, _] => some (.Glue (u32 c))
  | [1, b, c] => some (.Block .Permutation (u32 b) (u16 c))
  | [2, i, _] => some (.Public (u32 i))
  | _ => none

open neo_verified in
def wireWords : circuit.Wire → String
  | .Glue c => s!"0 {value c} 0"
  | .Block _ b c => s!"1 {value b} {value c}"
  | .Public i => s!"2 {value i} 0"

open neo_verified in
def parts (r : Aeneas.Std.Array layout.Part 2#usize × Usize) : String :=
  " ".intercalate ((r.1.val.flatMap fun p => [p.start, p.stride]).map value ++ [value r.2])

open neo_verified in
/-- The translated layout functions, with the one block kind. -/
def layoutCase : String → List ℕ → Option String
  | "Kind.index", [] => outcome (circuit.Kind.index .Permutation) value
  | "Kind.rows", [] => outcome (circuit.Kind.rows .Permutation) value
  | "Kind.cells", [] => outcome (circuit.Kind.cells .Permutation) value
  | "place", [x, y] => outcome (layout.place (usize x) (usize y)) parts
  | "slot", [a, b, c, d] => outcome (layout.slot (usize a) (usize b) (usize c) (usize d)) value
  | "row_variables", [b, r, c, p] => outcome (layout.row_variables (shape b r c p)) value
  | "cell_variables", [b, r, c, p] => outcome (layout.cell_variables (shape b r c p)) value
  | "one", [b, r, c, p] => outcome (layout.one (shape b r c p)) value
  | "glue_row", [b, r, c, p, g] => outcome (layout.glue_row (shape b r c p) (usize g)) value
  | "block_row", [b, r, c, p, x, k] =>
    outcome (layout.block_row (shape b r c p) .Permutation (usize x) (usize k)) value
  | "block_cell", [b, r, c, p, x, k] =>
    outcome (layout.block_cell (shape b r c p) .Permutation (usize x) (usize k)) value
  | "column", [b, r, c, p, t, x, y] => (wire [t, x, y]).map fun w => outcome (layout.column (shape b r c p) w) value
  | "entry_column", [b, r, c, p, x, present, k] =>
    outcome (layout.entry_column (shape b r c p) .Permutation (usize x) (if present = 1 then some (usize k) else none))
      value
  | _, _ => none

open neo_verified in
/-- The models that `place`, `cell_variables` and the derived `Clone` and `PartialEq` impls call. -/
def modelCase : String → List ℕ → Option String
  | "Usize.checked_add", [x, y] => s!"ok {option (Usize.checked_add (usize x) (usize y)) value}"
  | "Usize.checked_mul", [x, y] => s!"ok {option (Usize.checked_mul (usize x) (usize y)) value}"
  | "core.cmp.impls.OrdUsize.max", [x, y] => s!"ok {value (core.cmp.impls.OrdUsize.max (usize x) (usize y))}"
  | "core.cmp.PartialEqUsize.eq", [x, y] => outcome (core.cmp.PartialEqUsize.eq (usize x) (usize y)) toString
  | "core.array.equality.PartialEqArray.eq", [x, y] =>
    outcome (core.array.equality.PartialEqArray.eq core.cmp.PartialEqUsize (Aeneas.Std.Array.make 1#usize [usize x])
      (Aeneas.Std.Array.make 1#usize [usize y])) toString
  | "Array.index_usize", [x, y, i] =>
    outcome (Array.index_usize (Aeneas.Std.Array.make 2#usize [usize x, usize y]) (usize i)) value
  | "core.cmp.impls.PartialEqU32.eq", [x, y] => s!"ok {core.cmp.impls.PartialEqU32.eq (u32 x) (u32 y)}"
  | "core.cmp.impls.PartialEqU16.eq", [x, y] => s!"ok {core.cmp.impls.PartialEqU16.eq (u16 x) (u16 y)}"
  | "core.clone.Clone.clone.wire", [t, x, y] =>
    (wire [t, x, y]).map fun w => outcome (core.clone.Clone.clone circuit.Wire.Insts.CoreCloneClone w) wireWords
  | "core.marker.Copy.cloneInst.wire", [t, x, y] =>
    (wire [t, x, y]).map fun w =>
      outcome (core.clone.Clone.clone (core.marker.Copy.cloneInst circuit.Wire.Insts.CoreMarkerCopy) w) wireWords
  | "core.marker.StructuralPartialEq.wire", [t, x, y] =>
    (wire [t, x, y]).map fun _ =>
      let _ : core.marker.StructuralPartialEq circuit.Wire := circuit.Wire.Insts.CoreMarkerStructuralPartialEq
      "ok"
  | "core.cmp.PartialEq.eq.wire", [t, x, y, t', x', y'] => do
    let a ← wire [t, x, y]
    let b ← wire [t', x', y']
    outcome (core.cmp.PartialEq.eq circuit.Wire.Insts.CoreCmpPartialEqWire a b) toString
  | "core.clone.Clone.clone.kind", [] =>
    outcome (Std.bind (core.clone.Clone.clone circuit.Kind.Insts.CoreCloneClone .Permutation) circuit.Kind.index) value
  | "core.cmp.PartialEq.eq.kind", [] =>
    outcome (core.cmp.PartialEq.eq circuit.Kind.Insts.CoreCmpPartialEqKind .Permutation .Permutation) toString
  | "core.clone.Clone.clone.shape", [b, r, c, p] =>
    outcome (core.clone.Clone.clone layout.Shape.Insts.CoreCloneClone (shape b r c p)) shapeWords
  | "core.cmp.PartialEq.eq.shape", [b, r, c, p, b', r', c', p'] =>
    outcome (core.cmp.PartialEq.eq layout.Shape.Insts.CoreCmpPartialEqShape (shape b r c p) (shape b' r' c' p'))
      toString
  | "core.clone.Clone.clone.part", [start, stride] =>
    outcome (core.clone.Clone.clone layout.Part.Insts.CoreCloneClone ⟨usize start, usize stride⟩)
      fun p => s!"{value p.start} {value p.stride}"
  | _, _ => none

def run : String → List ℕ → Option String
  | "split", [x] => outcome (neo_verified.layout.split (usize x)) fun a => " ".intercalate (a.val.map value)
  | "variables", [x] => outcome (neo_verified.layout.variables (usize x)) value
  | "core.num.Usize.leading_zeros", [x] => s!"ok {value (core.num.Usize.leading_zeros (usize x))}"
  | "Usize.checked_sub", [x, y] => s!"ok {option (Usize.checked_sub (usize x) (usize y)) value}"
  | "U32.checked_sub", [x, y] => s!"ok {option (U32.checked_sub (u32 x) (u32 y)) value}"
  | "core.num.Usize.wrapping_shl", [x, k] => s!"ok {value (core.num.Usize.wrapping_shl (usize x) (u32 k))}"
  | "UScalar.cast", [k] => s!"ok {value (UScalar.cast .Usize (u32 k))}"
  | "core.option.Option.unwrap.some", [x] => outcome (core.option.Option.unwrap (some (usize x))) value
  | "core.option.Option.unwrap.none", [] => outcome (core.option.Option.unwrap (none : Option Usize)) value
  | "core.num.Usize.BITS", [] => s!"ok {value core.num.Usize.BITS}"
  | "Array.make", [x, y] => s!"ok {" ".intercalate ((Aeneas.Std.Array.make 2#usize [usize x, usize y]).val.map value)}"
  | case, words => (layoutCase case words).orElse fun _ => modelCase case words

end CompressionLeanRust.RunCheck

open CompressionLeanRust.RunCheck in
def main (args : List String) : IO UInt32 := do
  let [path] := args | IO.eprintln "usage: RunCheck.lean <cases>" *> pure 2
  let mut differ : Array String := #[]
  let mut count := 0
  for line in (← IO.FS.readFile path).splitOn "\n" do
    if line.isEmpty then continue
    let [lhs, expected] := line.splitOn " -> " | throw (IO.userError s!"not a case: {line}")
    let words := lhs.splitOn " "
    let some actual := run (words.headD "") (words.tail.map String.toNat!)
      | throw (IO.userError s!"unknown case: {line}")
    count := count + 1
    if actual != expected then differ := differ.push s!"{lhs}: Rust {expected}, Lean {actual}"
  if differ.isEmpty then
    IO.println s!"rustCheck: {count} cases agree"
    return 0
  for d in differ do IO.eprintln d
  return 1
