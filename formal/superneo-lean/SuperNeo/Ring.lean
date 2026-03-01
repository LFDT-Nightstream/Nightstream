import SuperNeo.Field

namespace SuperNeo

open F

/-- Ring dimension parameter for `Rq = F_q[X]/(X^d + 1)` in SuperNeo. -/
def d : Nat := 54

theorem d_pos : 0 < d := by
  decide

/-- Coefficient vector representation for ring elements. -/
abbrev Coeffs := Array F

def vecAdd (a b : Array F) : Array F :=
  if h : a.size = b.size then
    Array.ofFn (fun i : Fin a.size =>
      a[i.1]'i.2 + b[i.1]'(by simpa [h] using i.2))
  else
    #[]

def vecScale (s : F) (a : Array F) : Array F :=
  a.map (fun x => s * x)

def linComb2Vec (ρ1 ρ2 : F) (z1 z2 : Array F) : Array F :=
  vecAdd (vecScale ρ1 z1) (vecScale ρ2 z2)

def ct (a : Coeffs) : F :=
  a.getD 0 0

def coeffAt (a : Coeffs) (i : Nat) : F :=
  a.getD i 0

/--
Schoolbook-style cyclic multiplication skeleton modulo `d`.

This is a minimal executable kernel for early theorem work; quotient-specific
identities are layered later.
-/
def mulRq (a b : Coeffs) : Coeffs :=
  Array.ofFn (fun i : Fin d =>
    (List.range d).foldl
      (fun acc j =>
        let k := (i.1 + d - j) % d
        acc + coeffAt a j * coeffAt b k)
      0)

def zeroRq : Coeffs :=
  Array.replicate d 0

def oneRq : Coeffs :=
  (Array.replicate d 0).set! 0 1

@[simp] theorem vecScale_size (s : F) (a : Array F) :
    (vecScale s a).size = a.size := by
  simp [vecScale]

theorem vecAdd_size_of_eq {a b : Array F} (h : a.size = b.size) :
    (vecAdd a b).size = a.size := by
  simp [vecAdd, h]

theorem vecAdd_size_of_ne {a b : Array F} (h : a.size ≠ b.size) :
    (vecAdd a b).size = 0 := by
  simp [vecAdd, h]

theorem linComb2Vec_size_of_eq
    {ρ1 ρ2 : F} {z1 z2 : Array F} (h : z1.size = z2.size) :
    (linComb2Vec ρ1 ρ2 z1 z2).size = z1.size := by
  unfold linComb2Vec
  simp [vecScale_size, vecAdd_size_of_eq, h]

theorem mulRq_size (a b : Coeffs) : (mulRq a b).size = d := by
  simp [mulRq]

theorem zeroRq_size : zeroRq.size = d := by
  simp [zeroRq, d]

theorem oneRq_size : oneRq.size = d := by
  simp [oneRq, d]

theorem ct_zeroRq : ct zeroRq = 0 := by
  simp [ct, zeroRq, d]

def allCanonical (a : Coeffs) : Bool :=
  a.all F.canonicalCheck

end SuperNeo
