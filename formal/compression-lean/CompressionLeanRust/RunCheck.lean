import NeoVerified.Funs

namespace CompressionLeanRust.RunCheck

open Aeneas Aeneas.Std

def usize (n : ℕ) : Usize := ⟨BitVec.ofNat _ n⟩

def u32 (n : ℕ) : U32 := ⟨BitVec.ofNat _ n⟩

def u16 (n : ℕ) : U16 := ⟨BitVec.ofNat _ n⟩

def u64 (n : ℕ) : U64 := ⟨BitVec.ofNat _ n⟩

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

open neo_verified in
def relation : List ℕ → Option degrees.Relation
  | [d, l, b, k, m, p, n, u, o, s, x] =>
    some ⟨usize d, usize l, usize b, usize k, usize m, usize p, u32 n, usize u, usize o, usize s, usize x⟩
  | _ => none

/-- The slice of `xs`, or none when it is longer than a slice can be. -/
def slice (xs : List ℕ) : Option (Slice Usize) :=
  if h : (xs.map usize).length ≤ Usize.max then some (Slice.from (xs.map usize) h) else none

def values {n : Usize} (a : Aeneas.Std.Array Usize n) : String := " ".intercalate (a.val.map value)

open neo_verified in
/-- The translated degrees: `relation` reads the eleven fields of a `degrees::Relation` in order. -/
def degreesCase : String → List ℕ → Option String
  | "row_count", [k, m, p] => outcome (degrees.row_count (usize k) (usize m) (usize p)) value
  | "gkr", xs => (slice xs).map fun d => outcome (degrees.gkr d) value
  | "add", [x, y] => outcome (degrees.add (usize x) (usize y)) value
  | "sub", [x, y] => outcome (degrees.sub (usize x) (usize y)) value
  | "mul", [x, y] => outcome (degrees.mul (usize x) (usize y)) value
  | "exp2", [x] => outcome (degrees.exp2 (usize x)) value
  | "FIXED", [] => outcome degrees.FIXED values
  | "cube_variables", xs => (relation xs).map fun r => outcome (degrees.cube_variables r) value
  | "early_terms", xs => (relation xs).map fun r => outcome (degrees.early_terms r) values
  | "linear_terms", xs => (relation xs).map fun r => outcome (degrees.linear_terms r) values
  | "late_terms", xs => (relation xs).map fun r => outcome (degrees.late_terms r) values
  | "early", xs => (relation xs).map fun r => outcome (degrees.early r) value
  | "linear", xs => (relation xs).map fun r => outcome (degrees.linear r) value
  | "late", xs => (relation xs).map fun r => outcome (degrees.late r) value
  | "layer1", xs => (relation xs).map fun r => outcome (degrees.layer1 r) value
  | "shrink_terms", [b, r, c, p] => outcome (degrees.shrink_terms (shape b r c p)) values
  | "shrink", [b, r, c, p] => outcome (degrees.shrink (shape b r c p)) value
  | "Array.to_slice", xs =>
    if h : xs.length = 3 then
      some s!"ok {" ".intercalate ((Array.to_slice (Aeneas.Std.Array.make 3#usize (xs.map usize)
        (by simp [h]))).val.map value)}"
    else none
  | "Slice.len", xs => (slice xs).map fun d => s!"ok {value (Slice.len d)}"
  | "Slice.index_usize", i :: xs => (slice xs).map fun d => outcome (Slice.index_usize d (usize i)) value
  | "UScalar.cast.u32", [x] => s!"ok {value (UScalar.cast .U32 (usize x))}"
  | _, _ => none

open neo_verified in
def ext : List ℕ → Option field.Ext
  | [a, b, c] => some ⟨u64 a, u64 b, u64 c⟩
  | _ => none

open neo_verified in
def extWords (e : field.Ext) : String := s!"{value e.c0} {value e.c1} {value e.c2}"

open neo_verified in
/-- The translated field, its private word functions and the models that they call. -/
def fieldCase : String → List ℕ → Option String
  | "MODULUS", [] => s!"ok {value field.MODULUS}"
  | "GL_ZERO", [] => s!"ok {value field.GL_ZERO}"
  | "GL_ONE", [] => s!"ok {value field.GL_ONE}"
  | "EXT_ZERO", [] => s!"ok {extWords field.EXT_ZERO}"
  | "EXT_ONE", [] => s!"ok {extWords field.EXT_ONE}"
  | "canonical", [x] => outcome (field.canonical (u64 x)) fun o => option o value
  | "reduce", [x] => outcome (field.reduce (u64 x)) value
  | "reduce_word", [x] => outcome (field.reduce_word (u64 x)) value
  | "gl_word", [x] => outcome (field.gl_word (u64 x)) value
  | "low_half", [x] => outcome (field.low_half (u64 x)) value
  | "high_half", [x] => outcome (field.high_half (u64 x)) value
  | "fold_high", [x] => outcome (field.fold_high (u64 x)) value
  | "add_words", [x, y] => outcome (field.add_words (u64 x) (u64 y)) value
  | "sub_words", [x, y] => outcome (field.sub_words (u64 x) (u64 y)) value
  | "mul_words", [x, y] => outcome (field.mul_words (u64 x) (u64 y)) value
  | "gl_add", [x, y] => outcome (field.gl_add (u64 x) (u64 y)) value
  | "gl_sub", [x, y] => outcome (field.gl_sub (u64 x) (u64 y)) value
  | "gl_mul", [x, y] => outcome (field.gl_mul (u64 x) (u64 y)) value
  | "gl_neg", [x] => outcome (field.gl_neg (u64 x)) value
  | "gl_pow", [x, e] => outcome (field.gl_pow (u64 x) (u64 e)) value
  | "gl_inv", [x] => outcome (field.gl_inv (u64 x)) value
  | "ext_of_gl", [x] => outcome (field.ext_of_gl (u64 x)) extWords
  | "ext_neg", xs => (ext xs).map fun a => outcome (field.ext_neg a) extWords
  | "ext_pow", [a, b, c, e] => (ext [a, b, c]).map fun x => outcome (field.ext_pow x (u64 e)) extWords
  | "ext_scale", [a, b, c, g] => (ext [a, b, c]).map fun x => outcome (field.ext_scale x (u64 g)) extWords
  | "ext_add", [a, b, c, d, e, f] => do
    let x ← ext [a, b, c]
    let y ← ext [d, e, f]
    outcome (field.ext_add x y) extWords
  | "ext_sub", [a, b, c, d, e, f] => do
    let x ← ext [a, b, c]
    let y ← ext [d, e, f]
    outcome (field.ext_sub x y) extWords
  | "ext_mul", [a, b, c, d, e, f] => do
    let x ← ext [a, b, c]
    let y ← ext [d, e, f]
    outcome (field.ext_mul x y) extWords
  | "UScalar.cast.u32", [x] => s!"ok {value (UScalar.cast .U32 (u64 x))}"
  | "UScalar.cast.u64", [x] => s!"ok {value (UScalar.cast .U64 (u32 x))}"
  | "core.num.U64.wrapping_shr", [x, k] => s!"ok {value (core.num.U64.wrapping_shr (u64 x) (u32 k))}"
  | "core.num.U64.wrapping_shl", [x, k] => s!"ok {value (core.num.U64.wrapping_shl (u64 x) (u32 k))}"
  | "core.num.U64.wrapping_add", [x, y] => s!"ok {value (core.num.U64.wrapping_add (u64 x) (u64 y))}"
  | "core.num.U64.wrapping_sub", [x, y] => s!"ok {value (core.num.U64.wrapping_sub (u64 x) (u64 y))}"
  | "core.num.U64.wrapping_mul", [x, y] => s!"ok {value (core.num.U64.wrapping_mul (u64 x) (u64 y))}"
  | "core.num.U64.is_multiple_of", [x, m] => outcome (core.num.U64.is_multiple_of (u64 x) (u64 m)) toString
  | "core.cmp.PartialEq.eq.gl", [x, y] =>
    outcome (core.cmp.PartialEq.eq field.Gl.Insts.CoreCmpPartialEqGl (u64 x) (u64 y)) toString
  | "core.cmp.PartialEq.eq.ext", [a, b, c, d, e, f] => do
    let x ← ext [a, b, c]
    let y ← ext [d, e, f]
    outcome (core.cmp.PartialEq.eq field.Ext.Insts.CoreCmpPartialEqExt x y) toString
  | "core.clone.Clone.clone.gl", [x] => outcome (core.clone.Clone.clone field.Gl.Insts.CoreCloneClone (u64 x)) value
  | "core.clone.Clone.clone.ext", xs =>
    (ext xs).map fun a => outcome (core.clone.Clone.clone field.Ext.Insts.CoreCloneClone a) extWords
  | "core.cmp.Eq.gl", [x] =>
    outcome (core.cmp.Eq.assert_fields_are_eq field.Gl.Insts.CoreCmpEq (u64 x)) fun _ => "unit"
  | "core.cmp.Eq.ext", xs =>
    (ext xs).map fun a => outcome (core.cmp.Eq.assert_fields_are_eq field.Ext.Insts.CoreCmpEq a) fun _ => "unit"
  | _, _ => none

open neo_verified in
/-- A state of sixteen words, canonical or not. -/
def state16 (xs : List ℕ) : Option (Aeneas.Std.Array field.Gl 16#usize) :=
  if h : (xs.map u64).length = 16 then some (Aeneas.Std.Array.make 16#usize (xs.map u64) (by simp [h])) else none

def words64 {n : Usize} (a : Aeneas.Std.Array U64 n) : String := " ".intercalate (a.val.map value)

open neo_verified in
/-- The translated permutation, its tables and the models that only it calls. -/
def poseidon2Case : String → List ℕ → Option String
  | "permute", xs => (state16 xs).map fun s => outcome (poseidon2.permute s) words64
  | "HALF_FULL_ROUNDS", [] => s!"ok {value poseidon2.HALF_FULL_ROUNDS}"
  | "PARTIAL_ROUNDS", [] => s!"ok {value poseidon2.PARTIAL_ROUNDS}"
  | "INITIAL", [] => s!"ok {" ".intercalate (poseidon2.INITIAL.val.map words64)}"
  | "PARTIAL", [] => s!"ok {words64 poseidon2.PARTIAL}"
  | "TERMINAL", [] => s!"ok {" ".intercalate (poseidon2.TERMINAL.val.map words64)}"
  | "DIAGONAL", [] => s!"ok {words64 poseidon2.DIAGONAL}"
  | "core.num.Usize.wrapping_add", [x, y] => s!"ok {value (core.num.Usize.wrapping_add (usize x) (usize y))}"
  | "Array.update", [x, y, i, v] =>
    outcome (Array.update (Aeneas.Std.Array.make 2#usize [usize x, usize y]) (usize i) (usize v)) values
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
  | case, words => ((((layoutCase case words).orElse fun _ => modelCase case words).orElse fun _ =>
    degreesCase case words).orElse fun _ => fieldCase case words).orElse fun _ => poseidon2Case case words

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
