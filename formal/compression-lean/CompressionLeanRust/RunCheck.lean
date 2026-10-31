import NeoVerified.Funs

namespace CompressionLeanRust.RunCheck

open Aeneas Aeneas.Std

def usize (n : ℕ) : Usize := ⟨BitVec.ofNat _ n⟩

def u32 (n : ℕ) : U32 := ⟨BitVec.ofNat _ n⟩

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
  | _, _ => none

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
