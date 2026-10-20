import NightstreamFPrime.Export.NativePoseidon2

/-!
Owns the replay's arithmetic: canonical Goldilocks words (`Gl`), the
layer-1 challenge field `Ext = Gl[X]/(X³ - X - 1)` (Plonky3's
`CubicTrinomialExtensionField`, `crates/neo-spartan/src/field.rs`) and
`Kx = Ext[u]/(u² - 7)`, the `K ⊗ Ext` values of `circuit/algebra.rs`.

A word is canonical by construction. The machine operations are the F′
package's proved Goldilocks words (`NativeGoldilocks`), so a later bridge can
read every operation through `UInt64.denote`.
-/

namespace CompressionLean.Replay

open NightstreamFPrime.Spec
open NightstreamFPrime.Export.NativePoseidon2

/-- A canonical Goldilocks word. -/
structure Gl where
  word : UInt64
  canonical : word.toNat < goldilocksModulus

namespace Gl

def ofNat (n : Nat) : Gl := ⟨ofNat64 n, ofNat64_canonical n⟩

instance (n : Nat) : OfNat Gl n := ⟨ofNat n⟩
instance : Inhabited Gl := ⟨0⟩
instance : Add Gl := ⟨fun a b => ⟨add64 a.word b.word, add64_canonical _ _ a.canonical b.canonical⟩⟩
instance : Sub Gl := ⟨fun a b => ⟨sub64 a.word b.word, sub64_canonical _ _ a.canonical b.canonical⟩⟩
instance : Mul Gl := ⟨fun a b => ⟨mul64 a.word b.word, mul64_canonical _ _⟩⟩
instance : Neg Gl := ⟨fun a => 0 - a⟩
instance : BEq Gl := ⟨fun a b => a.word == b.word⟩

def toNat (a : Gl) : Nat := a.word.toNat

/-- `a^e` by squaring. -/
def pow (a : Gl) (e : Nat) : Gl :=
  if e = 0 then 1
  else
    let half := pow (a * a) (e / 2)
    if e % 2 = 1 then half * a else half
termination_by e
decreasing_by omega

/-- The inverse by Fermat; zero maps to zero. -/
def inv (a : Gl) : Gl := a.pow (goldilocksModulus - 2)

end Gl

/-- Plonky3's `Goldilocks::GENERATOR`. -/
def generator : Gl := 7

/-- Plonky3's `two_adic_generator(bits)` for `bits ≤ 32`: the table entry 32
squared `32 - bits` times. -/
def twoAdicGenerator (bits : Nat) : Gl :=
  (List.range (32 - bits)).foldl (fun g _ => g * g) (Gl.ofNat 0x185629dcda58878c)

/-- `a₀ + a₁·X + a₂·X²` with `X³ = X + 1`. -/
structure Ext where
  c0 : Gl
  c1 : Gl
  c2 : Gl
deriving Inhabited

namespace Ext

/-- A base value in the extension (`algebra::lift`). -/
def ofGl (a : Gl) : Ext := ⟨a, 0, 0⟩

instance (n : Nat) : OfNat Ext n := ⟨ofGl (Gl.ofNat n)⟩
instance : Add Ext := ⟨fun a b => ⟨a.c0 + b.c0, a.c1 + b.c1, a.c2 + b.c2⟩⟩
instance : Sub Ext := ⟨fun a b => ⟨a.c0 - b.c0, a.c1 - b.c1, a.c2 - b.c2⟩⟩
instance : Neg Ext := ⟨fun a => 0 - a⟩
instance : BEq Ext := ⟨fun a b => a.c0 == b.c0 && a.c1 == b.c1 && a.c2 == b.c2⟩

/-- The product with `X³ = X + 1` and `X⁴ = X² + X`. -/
instance : Mul Ext := ⟨fun a b =>
  let d3 := a.c1 * b.c2 + a.c2 * b.c1
  let d4 := a.c2 * b.c2
  ⟨a.c0 * b.c0 + d3, a.c0 * b.c1 + a.c1 * b.c0 + d3 + d4, a.c0 * b.c2 + a.c1 * b.c1 + a.c2 * b.c0 + d4⟩⟩

/-- `a·by` for a base value (`Backend::ext_scale`). -/
def scale (a : Ext) (by_ : Gl) : Ext := ⟨a.c0 * by_, a.c1 * by_, a.c2 * by_⟩

def isZero (a : Ext) : Bool := a == 0

def pow (a : Ext) (e : Nat) : Ext :=
  if e = 0 then 1
  else
    let half := pow (a * a) (e / 2)
    if e % 2 = 1 then half * a else half
termination_by e
decreasing_by omega

/-- The inverse in the field of `p³` elements; zero maps to zero. -/
def inv (a : Ext) : Ext := a.pow (goldilocksModulus ^ 3 - 2)

/-- The basis element `X^c` for `c < 3`. -/
def basis (c : Nat) : Ext :=
  if c = 0 then ⟨1, 0, 0⟩ else if c = 1 then ⟨0, 1, 0⟩ else ⟨0, 0, 1⟩

end Ext

/-- A `K ⊗ Ext` value `re + im·u` with `u² = 7` (`algebra::Kx`). -/
structure Kx where
  re : Ext
  im : Ext

namespace Kx

def zero : Kx := ⟨0, 0⟩
def one : Kx := ⟨1, 0⟩

/-- A `K` value `[re, im]` (`Kx::from_k`). -/
def ofK (value : Gl × Gl) : Kx := ⟨Ext.ofGl value.1, Ext.ofGl value.2⟩

def add (a b : Kx) : Kx := ⟨a.re + b.re, a.im + b.im⟩

def mul (a b : Kx) : Kx :=
  ⟨a.re * b.re + (a.im * b.im).scale 7, a.re * b.im + a.im * b.re⟩

def scale (a : Kx) (by_ : Ext) : Kx := ⟨a.re * by_, a.im * by_⟩

def oneMinus (a : Kx) : Kx := ⟨1 - a.re, 0 - a.im⟩

end Kx

end CompressionLean.Replay
