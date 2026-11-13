namespace NightstreamFPrime.Spec

def fieldModulus : Nat := 18446744073709551557

instance : NeZero fieldModulus := ⟨by decide⟩

abbrev F := Fin fieldModulus

structure K where
  c0 : F
  c1 : F
deriving DecidableEq, Repr

namespace K

def zero : K := ⟨0, 0⟩
def one : K := ⟨1, 0⟩
def embed (x : F) : K := ⟨x, 0⟩
def add (a b : K) : K := ⟨a.c0 + b.c0, a.c1 + b.c1⟩
def sub (a b : K) : K := ⟨a.c0 - b.c0, a.c1 - b.c1⟩
def mul (a b : K) : K :=
  ⟨a.c0 * b.c0 + 7 * a.c1 * b.c1,
   a.c0 * b.c1 + a.c1 * b.c0⟩

end K

def ringDegree : Nat := 54

def ringMiddleDegree : Nat := 27

abbrev RingF := Fin ringDegree → F
abbrev RingK := Fin ringDegree → K

def ringFCoeff (a : RingF) (i : Nat) : F :=
  if h : i < ringDegree then a ⟨i, h⟩ else 0

def ringKCoeff (a : RingK) (i : Nat) : K :=
  if h : i < ringDegree then a ⟨i, h⟩ else K.zero

def ringFZero : RingF := fun _ => 0
def ringKZero : RingK := fun _ => K.zero

def ringFMonomial (degree : Nat) (coefficient : F) : RingF :=
  fun i => if i.val = degree then coefficient else 0

def ringFOne : RingF := ringFMonomial 0 1

def ringFAdd (a b : RingF) : RingF := fun i => a i + b i
def ringKAdd (a b : RingK) : RingK := fun i => K.add (a i) (b i)

def rawMulCoeffF (a b : RingF) (degree : Nat) : F :=
  (List.range ringDegree).foldl (fun acc i =>
    if i ≤ degree ∧ degree - i < ringDegree then
      acc + ringFCoeff a i * ringFCoeff b (degree - i)
    else acc) 0

def rawMulCoeffK (a b : RingK) (degree : Nat) : K :=
  (List.range ringDegree).foldl (fun acc i =>
    if i ≤ degree ∧ degree - i < ringDegree then
      K.add acc (K.mul (ringKCoeff a i) (ringKCoeff b (degree - i)))
    else acc) K.zero

def ringFMul (a b : RingF) : RingF := fun out =>
  let i := out.val
  let folded := if i < ringMiddleDegree then
      rawMulCoeffF a b (i + ringDegree)
    else
      rawMulCoeffF a b (i + ringMiddleDegree)
  let twice := if i + (ringDegree + ringMiddleDegree) ≤ 2 * (ringDegree - 1) then
      rawMulCoeffF a b (i + (ringDegree + ringMiddleDegree)) else 0
  rawMulCoeffF a b i - folded + twice

def ringKMul (a b : RingK) : RingK := fun out =>
  let i := out.val
  let folded := if i < ringMiddleDegree then
      rawMulCoeffK a b (i + ringDegree)
    else
      rawMulCoeffK a b (i + ringMiddleDegree)
  let twice := if i + (ringDegree + ringMiddleDegree) ≤ 2 * (ringDegree - 1) then
      rawMulCoeffK a b (i + (ringDegree + ringMiddleDegree)) else K.zero
  K.add (K.sub (rawMulCoeffK a b i) folded) twice

def centeredMagnitude (x : F) : Nat :=
  min x.val (fieldModulus - x.val)

def normBounded (bound : Nat) (z : List F) : Prop :=
  ∀ x ∈ z, centeredMagnitude x < bound

end NightstreamFPrime.Spec
