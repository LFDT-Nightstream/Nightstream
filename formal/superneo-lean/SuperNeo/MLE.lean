import SuperNeo.EqPoly

namespace SuperNeo

open F

/-- χ_r(j) = Π_i (r_i if bit_i(j)=1 else (1-r_i)). -/
def chiWeight (r : Array F) (j : Nat) : F :=
  Id.run do
    let mut w : F := 1
    for i in [0:r.size] do
      let bit := (j / (2 ^ i)) % 2
      let ri := r[i]!
      let term := if bit = 1 then ri else (1 : F) - ri
      w := w * term
    return w

def rHat (r : Array F) (n : Nat) : Array F :=
  Id.run do
    let mut out := Array.replicate n (0 : F)
    for j in [0:n] do
      out := out.set! j (chiWeight r j)
    return out

def dotVec (a b : Array F) : F :=
  if a.size != b.size then
    0
  else
    Id.run do
      let mut acc : F := 0
      for i in [0:a.size] do
        acc := acc + a[i]! * b[i]!
      return acc

/-- MLE via inner product identity: v~(r) = <v, r_hat>. -/
def mleByInnerProduct (v r : Array F) : F :=
  dotVec v (rHat r v.size)

private def foldLayer (vals : Array F) (ri : F) : Array F :=
  Id.run do
    let pairs := vals.size / 2
    let mut out := Array.replicate pairs (0 : F)
    for i in [0:pairs] do
      let a := vals[2 * i]!
      let b := vals[2 * i + 1]!
      out := out.set! i (a * ((1 : F) - ri) + b * ri)
    return out

/-- MLE via iterative multilinear folding across coordinates. -/
def mleByFolding (v r : Array F) : F :=
  Id.run do
    let mut cur := v
    for i in [0:r.size] do
      cur := foldLayer cur r[i]!
    if cur.isEmpty then
      0
    else
      cur[0]!

def mleIdentity (v r : Array F) : Bool :=
  if v.size != 2 ^ r.size then
    false
  else
    decide (mleByInnerProduct v r = mleByFolding v r)

def mleSanity : Bool :=
  let v := #[3, 5, 7, 9]
  let r := #[F.ofNat 2, F.ofNat 1]
  mleIdentity v r

end SuperNeo
