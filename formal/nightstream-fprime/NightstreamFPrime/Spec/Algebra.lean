/-!
Owns the concrete Goldilocks algebra of the production profile: the field
`F`, its quadratic extension `K = F[X]/(X² − 7)`, the cyclotomic rings
`F[X]/Φ₈₁` and `K[X]/Φ₈₁` with `Φ₈₁ = X⁵⁴ + X²⁷ + 1`, the centered ∞-norm,
column-major packing, and the Ajtai matrix action. Executable semantics only.

Provenance: copied from
`formal/nightstream-lean/Nightstream/SuperNeo/Concrete/Algebra.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; unchanged except this header.
-/

namespace NightstreamFPrime.Spec

/-- Production Goldilocks modulus `0xFFFFFFFF00000001`. -/
def goldilocksModulus : Nat := 18446744069414584321

instance : NeZero goldilocksModulus := ⟨by decide⟩

/-- Canonical Goldilocks residues. `Fin` arithmetic is reduction modulo `q`. -/
abbrev F := Fin goldilocksModulus

/-- Quadratic extension `K = F[X]/(X² − 7)`. -/
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

/-- Production `d = φ(81)`. -/
def ringDegree : Nat := 54

/-- Middle coefficient of `Φ₈₁(X) = X⁵⁴ + X²⁷ + 1`. -/
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

/-- Schoolbook multiplication reduced by `X⁵⁴ = −X²⁷ − 1`.
The final positive term is the second reduction of degrees 81 through 106. -/
def ringFMul (a b : RingF) : RingF := fun out =>
  let i := out.val
  let folded := if i < ringMiddleDegree then
      rawMulCoeffF a b (i + ringDegree)
    else
      rawMulCoeffF a b (i + ringMiddleDegree)
  let twice := if i + 81 ≤ 106 then rawMulCoeffF a b (i + 81) else 0
  rawMulCoeffF a b i - folded + twice

def ringKMul (a b : RingK) : RingK := fun out =>
  let i := out.val
  let folded := if i < ringMiddleDegree then
      rawMulCoeffK a b (i + ringDegree)
    else
      rawMulCoeffK a b (i + ringMiddleDegree)
  let twice := if i + 81 ≤ 106 then rawMulCoeffK a b (i + 81) else K.zero
  K.add (K.sub (rawMulCoeffK a b i) folded) twice

/-- Centered magnitude of a canonical residue. -/
def centeredMagnitude (x : F) : Nat :=
  min x.val (goldilocksModulus - x.val)

/-- Paper norm predicate `‖z‖∞ < bound`. -/
def normBounded (bound : Nat) (z : List F) : Prop :=
  ∀ x ∈ z, centeredMagnitude x < bound

/-- Coefficient `rho` of packed block `block`, with zero padding. -/
def packedCoeff (z : List F) (block : Nat) (rho : Fin ringDegree) : F :=
  z.getD (block * ringDegree + rho.val) 0

/-- Column-major `D × ⌈m/D⌉` assignment packing. -/
def packAssignment (z : List F) : List RingF :=
  (List.range ((z.length + ringDegree - 1) / ringDegree)).map
    (fun block rho => packedCoeff z block rho)

def ringFDot : List RingF → List RingF → RingF
  | a :: as, b :: bs => ringFAdd (ringFMul a b) (ringFDot as bs)
  | _, _ => ringFZero

/-- Verifier-owned Ajtai matrix, row-major over ring elements. -/
abbrev AjtaiKey := List (List RingF)
abbrev Commitment := List RingF

/-- Concrete module action `A · pack(z)`. -/
def ajtaiCommit (key : AjtaiKey) (z : List F) : Commitment :=
  let packed := packAssignment z
  key.map (fun row => ringFDot row packed)

/-- Public projection `x = z[0..n_in]`. -/
def projectPublicInput (publicWidth : Nat) (z : List F) : List F :=
  z.take publicWidth

end NightstreamFPrime.Spec
