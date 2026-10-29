import CompressionLean.Fields
import NightstreamFPrime.Export.NativePoseidon2

/-! `Ext` is Plonky3's `CubicTrinomialExtensionField`. -/

namespace CompressionLean.Replay

open NightstreamFPrime.Spec
open NightstreamFPrime.Export.NativePoseidon2

structure Word where
  word : UInt64
  canonical : word.toNat < goldilocksModulus

namespace Word

def ofNat (n : Nat) : Word := ⟨ofNat64 n, ofNat64_canonical n⟩

instance (n : Nat) : OfNat Word n := ⟨ofNat n⟩
instance : Inhabited Word := ⟨0⟩
instance : Add Word := ⟨fun a b => ⟨add64 a.word b.word, add64_canonical _ _ a.canonical b.canonical⟩⟩
instance : Sub Word := ⟨fun a b => ⟨sub64 a.word b.word, sub64_canonical _ _ a.canonical b.canonical⟩⟩
instance : Mul Word := ⟨fun a b => ⟨mul64 a.word b.word, mul64_canonical _ _⟩⟩
instance : Neg Word := ⟨fun a => 0 - a⟩
instance : BEq Word := ⟨fun a b => a.word == b.word⟩

def toNat (a : Word) : Nat := a.word.toNat

def pow (a : Word) (e : Nat) : Word :=
  if e = 0 then 1
  else
    let half := pow (a * a) (e / 2)
    if e % 2 = 1 then half * a else half
termination_by e
decreasing_by omega

def inv (a : Word) : Word := a.pow (goldilocksModulus - 2)

def toF (a : Word) : Gl := (a.word.toNat : Gl)

@[pp_nodot] def cast {F : Type} [CommRing F] (g : Word) : F := (g.toNat : F)

def ofF (value : F) : Word := Word.ofNat value.val

end Word

/-- Plonky3's `Goldilocks::GENERATOR`. -/
def generator : Word := 7

/-- Plonky3's `two_adic_generator(bits)`. -/
def twoAdicGenerator (bits : Nat) : Word :=
  (List.range (32 - bits)).foldl (fun g _ => g * g) (Word.ofNat 0x185629dcda58878c)

structure Ext where
  c0 : Word
  c1 : Word
  c2 : Word
deriving Inhabited

namespace Ext

def ofGl (a : Word) : Ext := ⟨a, 0, 0⟩

instance (n : Nat) : OfNat Ext n := ⟨ofGl (Word.ofNat n)⟩
instance : Add Ext := ⟨fun a b => ⟨a.c0 + b.c0, a.c1 + b.c1, a.c2 + b.c2⟩⟩
instance : Sub Ext := ⟨fun a b => ⟨a.c0 - b.c0, a.c1 - b.c1, a.c2 - b.c2⟩⟩
instance : Neg Ext := ⟨fun a => 0 - a⟩
instance : BEq Ext := ⟨fun a b => a.c0 == b.c0 && a.c1 == b.c1 && a.c2 == b.c2⟩

instance : Mul Ext := ⟨fun a b =>
  let d3 := a.c1 * b.c2 + a.c2 * b.c1
  let d4 := a.c2 * b.c2
  ⟨a.c0 * b.c0 + d3, a.c0 * b.c1 + a.c1 * b.c0 + d3 + d4, a.c0 * b.c2 + a.c1 * b.c1 + a.c2 * b.c0 + d4⟩⟩

def scale (a : Ext) (by_ : Word) : Ext := ⟨a.c0 * by_, a.c1 * by_, a.c2 * by_⟩

def isZero (a : Ext) : Bool := a == 0

def pow (a : Ext) (e : Nat) : Ext :=
  if e = 0 then 1
  else
    let half := pow (a * a) (e / 2)
    if e % 2 = 1 then half * a else half
termination_by e
decreasing_by omega

def inv (a : Ext) : Ext := a.pow (goldilocksModulus ^ 3 - 2)

def basis (c : Nat) : Ext :=
  if c = 0 then ⟨1, 0, 0⟩ else if c = 1 then ⟨0, 1, 0⟩ else ⟨0, 0, 1⟩

end Ext

structure Kx where
  re : Ext
  im : Ext
deriving Inhabited

namespace Kx

def zero : Kx := ⟨0, 0⟩
def one : Kx := ⟨1, 0⟩

def ofK (value : Word × Word) : Kx := ⟨Ext.ofGl value.1, Ext.ofGl value.2⟩

def add (a b : Kx) : Kx := ⟨a.re + b.re, a.im + b.im⟩

def mul (a b : Kx) : Kx :=
  ⟨a.re * b.re + (a.im * b.im).scale 7, a.re * b.im + a.im * b.re⟩

def scale (a : Kx) (by_ : Ext) : Kx := ⟨a.re * by_, a.im * by_⟩

def oneMinus (a : Kx) : Kx := ⟨1 - a.re, 0 - a.im⟩

end Kx

end CompressionLean.Replay
