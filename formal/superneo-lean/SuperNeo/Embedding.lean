import SuperNeo.CoeffMaps
import SuperNeo.Dimensions

namespace SuperNeo

open F

/-- Element embedding: F^d -> R_F via coefficient map inverse. -/
def embedElem (v : Array F) : Coeffs := cfInv v

/-- Inverse element embedding: R_F -> F^d via coefficients. -/
def unembedElem (a : Coeffs) : Array F := cf a

private def chunkExact (xs : Array F) (chunk : Nat) : Array (Array F) :=
  Id.run do
    let mut out : Array (Array F) := #[]
    let mut i := 0
    while i < xs.size do
      let stop := Nat.min (i + chunk) xs.size
      out := out.push (xs.extract i stop)
      i := i + chunk
    return out

private def flatten (blocks : Array (Array F)) : Array F :=
  blocks.foldl (fun acc blk => acc ++ blk) #[]

/-- Vector embedding: F^(d*n_R) -> (R_F)^n_R by chunking in d coefficients. -/
def embedVec (z : Array F) : Array Coeffs :=
  if z.size % d != 0 then
    #[]
  else
    (chunkExact z d).map cfInv

def unembedVec (zr : Array Coeffs) : Array F :=
  flatten (zr.map cf)

/-- Matrix embedding row-wise. -/
def embedMatrix (m : Array (Array F)) : Array (Array Coeffs) :=
  m.map embedVec

def unembedMatrix (mr : Array (Array Coeffs)) : Array (Array F) :=
  mr.map unembedVec

def embeddingVecRoundTrip (z : Array F) : Bool :=
  if z.size % d != 0 then
    false
  else
    decide (unembedVec (embedVec z) = z)

def embeddingMatrixRoundTrip (m : Array (Array F)) : Bool :=
  if !(m.all (fun row => row.size % d = 0)) then
    false
  else
    decide (unembedMatrix (embedMatrix m) = m)

def embeddingSanity : Bool :=
  let z := ((List.range (2 * d)).toArray).map F.ofNat
  embeddingVecRoundTrip z

end SuperNeo
