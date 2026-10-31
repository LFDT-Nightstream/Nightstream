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

open neo_verified in
/-- Extension values from their words, three each. -/
def extList : List ℕ → List field.Ext
  | a :: b :: c :: rest => ⟨u64 a, u64 b, u64 c⟩ :: extList rest
  | _ => []

def sliceOf {α : Type} (xs : List α) : Option (Slice α) :=
  if h : xs.length ≤ Usize.max then some (Slice.from xs h) else none

def vecOf {α : Type} (xs : List α) : Option (alloc.vec.Vec α) :=
  if h : xs.length ≤ Usize.max then some (alloc.vec.Vec.from xs h) else none

open neo_verified in
def extsWords (xs : List field.Ext) : String := " ".intercalate (xs.map extWords)

open neo_verified in
def glsWords (xs : List field.Gl) : String := " ".intercalate (xs.map value)

open neo_verified in
abbrev native : backend.Backend native.Native field.Gl field.Ext := native.Native.Insts.Neo_verifiedBackendBackendGlExt

open neo_verified in
def array4 (xs : List ℕ) : Option (Aeneas.Std.Array field.Gl 4#usize) :=
  if h : (xs.map u64).length = 4 then some (Aeneas.Std.Array.make 4#usize (xs.map u64) (by simp [h])) else none

open neo_verified in
def verdict (r : core.result.Result Unit backend.Rejected) : String :=
  match r with
  | .Ok _ => "ok"
  | .Err _ => "err"

open neo_verified in
/-- A root, a bit count and its bits, a value count and its values, and one path node per bit. -/
def leafArgs (rest : List ℕ) : Option (Aeneas.Std.Array field.Gl 4#usize × Slice field.Gl × Slice field.Gl ×
    Slice (Aeneas.Std.Array field.Gl 4#usize)) := do
  let root ← array4 (rest.take 4)
  let n := rest.getD 4 0
  let bits ← sliceOf ((rest.drop 5).take n |>.map u64)
  let m := rest.getD (5 + n) 0
  let values ← sliceOf ((rest.drop (6 + n)).take m |>.map u64)
  let path ← sliceOf (((List.range n).map fun k => (rest.drop (6 + n + m + 4 * k)).take 4).filterMap array4)
  pure (root, bits, values, path)

open neo_verified in
/-- `count` groups, each a length and three words per value, and the words after them. -/
def groupArgs : ℕ → List ℕ → Option (List (alloc.vec.Vec field.Ext) × List ℕ)
  | 0, rest => some ([], rest)
  | count + 1, len :: rest => do
    let group ← vecOf (extList (rest.take (3 * len)))
    let (groups, tail) ← groupArgs count (rest.drop (3 * len))
    pure (group :: groups, tail)
  | _, [] => none

open neo_verified in
/-- The circuit helpers run on the translated `Native`, and the models that only the circuit core
calls. -/
def circuitCase : String → List ℕ → Option String
  | "ext_inv", xs => (ext xs).map fun a => outcome (field.ext_inv a) extWords
  | "eq_eval", n :: rest => do
    let x ← sliceOf (extList (rest.take (3 * n)))
    let y ← sliceOf (extList (rest.drop (3 * n)))
    outcome (algebra.eq_eval native () x y) fun p => extWords p.1
  | "dot", n :: rest => do
    let x ← sliceOf (extList (rest.take (3 * n)))
    let y ← sliceOf (extList (rest.drop (3 * n)))
    outcome (algebra.dot native () x y) fun p => extWords p.1
  | "interpolate", n :: rest => do
    let v ← sliceOf (extList (rest.take (3 * n)))
    let x ← ext (rest.drop (3 * n))
    outcome (algebra.interpolate native () v x) fun p => extWords p.1
  | "horner", n :: rest => do
    let v ← sliceOf (extList (rest.take (3 * n)))
    let x ← ext (rest.drop (3 * n))
    outcome (algebra.horner native () v x) fun p => extWords p.1
  | "eq_at", index :: rest => do
    let point ← sliceOf (extList rest)
    outcome (algebra.eq_at native () point (usize index)) fun p => extWords p.1
  | "eq_table", rest => do
    let point ← sliceOf (extList rest)
    outcome (algebra.eq_table native () point) fun p => extsWords p.1.val
  | "power_from_bits", g :: bits => do
    let b ← sliceOf (bits.map u64)
    outcome (algebra.power_from_bits native () (u64 g) b) fun p => value p.1
  | "hash_leaf", xs => do
    let v ← sliceOf (xs.map u64)
    outcome (hash.hash_leaf native () v) fun p => glsWords p.1.val
  | "compress", xs => do
    let l ← array4 (xs.take 4)
    let r ← array4 (xs.drop 4)
    outcome (hash.compress native () l r) fun p => glsWords p.1.val
  | "merkle_root", rest => do
    let leaf ← array4 (rest.take 4)
    let n := rest.getD 4 0
    let bits ← sliceOf ((rest.drop 5).take n |>.map u64)
    let path ← sliceOf (((List.range n).map fun k => (rest.drop (5 + n + 4 * k)).take 4).filterMap array4)
    outcome (hash.merkle_root native () leaf bits path) fun p => glsWords p.1.val
  | "two_adic_generator", [b] => outcome (field.two_adic_generator (usize b)) value
  | "path_ok", rest => do
    let (root, bits, values, path) ← leafArgs rest
    outcome (setup.path_ok native () root bits values path) fun p => verdict p.1
  | "check", vars :: rest => do
    let (root, bits, values, path) ← leafArgs rest
    let n := rest.getD 4 0
    let m := rest.getD (5 + n) 0
    let (groups, alpha) ← groupArgs (rest.getD (6 + n + m + 4 * n) 0) (rest.drop (7 + n + m + 4 * n))
    let groups ← sliceOf groups
    let alpha ← if h : (extList alpha).length = 3 then some (Aeneas.Std.Array.make 3#usize (extList alpha) h) else none
    outcome (setup.check native () root (usize vars) bits values path groups alpha) fun p =>
      match p.1 with
      | .Ok v => s!"ok {extsWords v.val}"
      | .Err _ => "err"
  | "alloc.vec.Vec.new", [] => s!"ok {glsWords (alloc.vec.Vec.new U64).val}"
  | "alloc.vec.Vec.push", xs => do
    let v ← vecOf ((xs.dropLast).map u64)
    outcome (alloc.vec.Vec.push v (u64 (xs.getLastD 0))) fun w => glsWords w.val
  | "alloc.vec.Vec.len", xs => (vecOf (xs.map u64)).map fun v => s!"ok {value (alloc.vec.Vec.len v)}"
  | "alloc.vec.Vec.deref", xs => (vecOf (xs.map u64)).map fun v => s!"ok {glsWords (alloc.vec.Vec.deref v).val}"
  | "alloc.vec.CloneVec.clone", xs =>
    (vecOf (xs.map u64)).map fun v => outcome (alloc.vec.CloneVec.clone core.clone.CloneU64 v) fun w => glsWords w.val
  | "alloc.vec.FromVecArray.from", [x, y] =>
    outcome (alloc.vec.FromVecArray.from (Aeneas.Std.Array.make 2#usize [u64 x, u64 y])) fun w => glsWords w.val
  | "alloc.vec.Vec.index", i :: xs => (vecOf (xs.map u64)).map fun v =>
    outcome (alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice U64) v (usize i)) value
  | "alloc.vec.Vec.index_mut", i :: x :: xs => (vecOf (xs.map u64)).map fun v =>
    outcome (alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice U64) v (usize i)) fun p =>
      glsWords (p.2 (u64 x)).val
  | "core.slice.index.SliceIndexUsizeSlice", i :: xs => (sliceOf (xs.map u64)).map fun v =>
    outcome ((core.slice.index.SliceIndexUsizeSlice U64).index (usize i) v) value
  | "Array.index_mut_usize", [x, y, i, v] =>
    outcome (Array.index_mut_usize (Aeneas.Std.Array.make 2#usize [u64 x, u64 y]) (usize i)) fun p =>
      words64 (p.2 (u64 v))
  | "core.slice.Slice.is_empty", xs => (sliceOf (xs.map u64)).map fun v => outcome (core.slice.Slice.is_empty v) toString
  | "Array.empty", [] => s!"ok {(Std.Array.empty U64).val.length}"
  | "Array.repeat", [x] => s!"ok {words64 (Array.repeat 3#usize (u64 x))}"
  | "core.clone.CloneArray", xs =>
    if h : (xs.map u64).length = 4 then
      outcome ((core.clone.CloneArray 4#usize core.clone.CloneU64).clone
        (Aeneas.Std.Array.make 4#usize (xs.map u64) (by simp [h]))) words64
    else none
  | "core.clone.CloneU64", [x] => outcome (core.clone.CloneU64.clone (u64 x)) value
  | "core.clone.impls.CloneUsize.clone", [x] => s!"ok {value (core.clone.impls.CloneUsize.clone (usize x))}"
  | "core.convert.FromSame", [x] => outcome ((core.convert.FromSame U64).from (u64 x)) value
  | "core.num.U32.wrapping_add", [x, y] => s!"ok {value (core.num.U32.wrapping_add (u32 x) (u32 y))}"
  | "core.num.Usize.is_multiple_of", [x, m] => outcome (core.num.Usize.is_multiple_of (usize x) (usize m)) toString
  | "core.num.Usize.wrapping_shr", [x, k] => s!"ok {value (core.num.Usize.wrapping_shr (usize x) (u32 k))}"
  | "core.num.Usize.wrapping_sub", [x, y] => s!"ok {value (core.num.Usize.wrapping_sub (usize x) (usize y))}"
  | "core.cmp.PartialEq.ne.trait_default", [t, x, y, t', x', y'] => do
    let a ← wire [t, x, y]
    let b ← wire [t', x', y']
    outcome (core.cmp.PartialEq.ne.trait_default circuit.Wire.Insts.CoreCmpPartialEqWire a b) toString
  | "core.option.Option.is_none", [0] => s!"ok {core.option.Option.is_none (none : Option U64)}"
  | "core.option.Option.is_none", [1, x] => s!"ok {core.option.Option.is_none (some (u64 x))}"
  | "core.ops.control_flow.ControlFlow.Continue", [x] =>
    match (core.ops.control_flow.ControlFlow.Continue (u64 x) : core.ops.control_flow.ControlFlow Unit U64) with
    | .Continue v => s!"ok continue {value v}"
    | .Break _ => "ok break"
  | "core.ops.control_flow.ControlFlow.Break", [] =>
    match (core.ops.control_flow.ControlFlow.Break () : core.ops.control_flow.ControlFlow Unit U64) with
    | .Continue v => s!"ok continue {value v}"
    | .Break _ => "ok break"
  | "core.result.Result.Ok", [x] => resultCase (.Ok (u64 x))
  | "core.result.Result.Err", [] => resultCase (.Err (toStr "x"))
  | "core.result.Result.Insts.CoreOpsTry.branch", [1, x] => tryCase (.Ok (u64 x))
  | "core.result.Result.Insts.CoreOpsTry.branch", [0] => tryCase (.Err (toStr "x"))
  | "core.result.Result.Insts.CoreOpsTry_traitFromResidualResult.from_residual", [] => tryCase (.Err (toStr "x"))
  | _, _ => none
where
  resultWord (r : core.result.Result U64 backend.Rejected) : String :=
    match r with
    | .Ok v => s!"ok {value v}"
    | .Err _ => "err"
  resultCase (r : core.result.Result U64 backend.Rejected) : String := s!"ok {resultWord r}"
  /-- `let v = r?; Ok(v)`, as the translation of `?` runs it. -/
  tryCase (r : core.result.Result U64 backend.Rejected) : String :=
    let run : Result (core.result.Result U64 backend.Rejected) := do
      let flow ← core.result.Result.Insts.CoreOpsTry.branch r
      match flow with
      | core.ops.control_flow.ControlFlow.Break residual =>
        core.result.Result.Insts.CoreOpsTry_traitFromResidualResult.from_residual U64
          (core.convert.FromSame backend.Rejected) residual
      | core.ops.control_flow.ControlFlow.Continue v => Result.ok (.Ok v)
    outcome run resultWord

open neo_verified in
/-- A sequence of extension values: its length, then three words per value; and the words after it. -/
def extSeq : List ℕ → Option (List field.Ext × List ℕ)
  | n :: rest => if 3 * n ≤ rest.length then some (extList (rest.take (3 * n)), rest.drop (3 * n)) else none
  | [] => none

open neo_verified in
/-- `k` tracks: each point as a sequence and its offset; and the words after them. -/
def tracksIn : ℕ → List ℕ → Option (List diagonal.Track × List ℕ)
  | 0, rest => some ([], rest)
  | k + 1, rest => do
    let (pts, rest) ← extSeq rest
    let off :: rest := rest | none
    let point ← vecOf pts
    let (more, rest) ← tracksIn k rest
    pure (⟨point, usize off⟩ :: more, rest)

open neo_verified in
def tracksAll : List ℕ → Option (Slice diagonal.Track)
  | k :: rest => do
    let (tracks, []) ← tracksIn k rest | none
    sliceOf tracks
  | [] => none

open neo_verified in
def statesWords (states : List (Aeneas.Std.Array field.Ext 2#usize)) : String :=
  " ".intercalate (states.map fun a => extsWords a.val)

open neo_verified in
def statesIn (words : List ℕ) : List (Aeneas.Std.Array field.Ext 2#usize) :=
  (List.range (words.length / 6)).map fun k =>
    let e := extList ((words.drop (6 * k)).take 6)
    Aeneas.Std.Array.make 2#usize [e.getD 0 field.EXT_ZERO, e.getD 1 field.EXT_ZERO]

open neo_verified in
/-- The sum-check rounds and the diagonal walk of the shrink verifier. -/
def shrinkCase : String → List ℕ → Option String
  | "ext_reduce", xs => (ext xs).map fun a => outcome (field.ext_reduce a) extWords
  | "round_claim", degree :: rest => do
    let (message, rest) ← extSeq rest
    let m ← sliceOf message
    let claim ← ext (rest.take 3)
    let r ← ext (rest.drop 3)
    outcome (sumcheck.round_claim (usize degree) m claim r) extWords
  | "replay_claim", degree :: count :: rest => do
    let (rounds, rest) ← (List.range count).foldlM (fun (acc : List (alloc.vec.Vec field.Ext) × List ℕ) _ => do
      let (m, rest) ← extSeq acc.2
      let v ← vecOf m
      pure (acc.1 ++ [v], rest)) ([], rest)
    let r ← sliceOf rounds
    let claim ← ext (rest.take 3)
    let (point, []) ← extSeq (rest.drop 3) | none
    let p ← sliceOf point
    outcome (sumcheck.replay_claim (usize degree) r claim p) extWords
  | "walk_bits", count :: rest => do
    let tracks ← tracksAll rest
    outcome (diagonal.walk_bits tracks (usize count)) value
  | "diagonal", count :: rest => do
    let tracks ← tracksAll rest
    outcome (diagonal.diagonal tracks (usize count)) extWords
  | "walk_states", count :: width :: rest => do
    let tracks ← tracksAll rest
    outcome (diagonal.walk_states tracks (usize count) (usize width)) fun v => statesWords v.val
  | "walk_step", count :: t :: k :: rest => do
    let (tracks, n :: words) ← tracksIn k rest | none
    let tr ← sliceOf tracks
    let st ← sliceOf (statesIn (words.take (6 * n)))
    outcome (diagonal.walk_step tr (usize count) (usize t) st) fun v => statesWords v.val
  | "tracks_step", t :: b :: carry :: rest => do
    let tracks ← tracksAll rest
    outcome (diagonal.tracks_step tracks (usize t) (usize b) (usize carry)) fun p => s!"{extWords p.1} {value p.2}"
  | "track_factor", t :: bit :: rest => do
    let (point, []) ← extSeq rest | none
    let pt ← sliceOf point
    outcome (diagonal.track_factor pt (usize t) (usize bit)) extWords
  | "next_below", [cb, b, below] => outcome (diagonal.next_below (usize cb) (usize b) (usize below)) value
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
  | case, words => (((((layoutCase case words).orElse fun _ => modelCase case words).orElse fun _ =>
    degreesCase case words).orElse fun _ => fieldCase case words).orElse fun _ => poseidon2Case case words).orElse
    fun _ => (circuitCase case words).orElse fun _ => shrinkCase case words

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
