import NightstreamFPrime.Spec.Compression.Akita.Replay.Layout
import NightstreamFPrime.Spec.Compression.Akita.TrigBuild

/-! The fold challenge sampler of the pinned Akita code (revision `d50f700`,
`crates/akita-challenges/src/sampler/stream.rs`, `fold_draw.rs`, `sampler/op_norm.rs`,
`sampler/op_norm_accumulate.rs`; Akita, ePrint 2026/1983, revision of 2026-11-23, Section 3.2 and
Appendix H.1) on the squeezed message of one fold group.

- `readBits` is `read_bits`: bit `i` of the message is bit `i % 8` of byte `i / 8`. Candidate `j`
  is the `K`-bit number at bit `j·K`, `K = FoldGroup.candidateBits`.
- `candidate` is `CandidateDecoder::candidate`: the low `w = pm1 + pm2` bits are the signs, the
  other bits the rank, which is out of range at `rankSpace` or above. The rank is
  `r_a + C(d, pm1)·r_b`; `unrank` is the colexicographic unrank of `BinomialTable::unrank` (one
  downward `scan` for all elements), and `complement` is `map_to_complement`. Coefficient `i` has
  the magnitude 1 below `pm1` and 2 above, and the sign `−` when sign bit `i` is set.
- `binomials` is `binomial_table` with the entries of `FoldGroup.binomial`.
- `Filter.of d` has the tables of `build_tables(d, 48)` (`TrigTable.code`) and the threshold of
  the policy (`OperatorNormRejection`: 18 for the degree 64, 13 for the degree 128).
  `Filter.accepts` is `accept_strict_parts`: each of the `d/2` frequencies has
  `R² + I² + 2(|R| + |I|)·r + 2r² ≤ T²·2^96`, with `R_k = Σ c·C[(2k+1)·p mod 2d]`.
- `sample` is `decode_with` without position hints: the first `coordinates` candidates of an
  unfiltered group (`decode_all`), or the first `coordinates` accepted candidates of the pool of a
  filtered group (`first_accepted`), and `none` when a pool runs out or an unfiltered candidate is
  out of range. `foldChallenges` maps each position to the ring of `A`: times
  `stride = ring/degree` (`Challenges::embed_subring_positions`, `as_slice`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

/-- A sparse fold challenge: its pairs (position, coefficient) in the order of the code, the
coefficients `±1` first. -/
abbrev Sparse := List (ℕ × ℤ)

namespace FoldGroup

/-- The bits of the rank of a candidate. -/
def rankBits (G : FoldGroup) : ℕ :=
  if G.filtered then Nat.size (G.rankSpace - 1) else Nat.log 2 G.rankSpace

theorem candidateBits_eq (G : FoldGroup) : G.candidateBits = G.rankBits + (G.pm1 + G.pm2) := by
  unfold candidateBits rankBits
  omega

/-- The threshold `T` of the operator-norm policy of a degree. -/
def threshold (degree : ℕ) : ℕ := if degree = 64 then 18 else 13

end FoldGroup

/-- `read_bits`: `width` bits of the message from bit `offset`. -/
def readBits (bytes : ByteArray) (offset width : ℕ) : ℕ :=
  let first := offset / 8
  let value := (List.range ((offset + width + 7) / 8 - first)).foldr
    (fun i acc => acc * 256 + (bytes[first + i]?.getD 0).toNat) 0
  value / 2 ^ (offset % 8) % 2 ^ width

/-- `C(n, k)` for `n ≤ maxN` and `k ≤ maxK`, by column. -/
def binomials (maxN maxK : ℕ) : Array (Array ℕ) :=
  (Array.range (maxK + 1)).map fun k => (Array.range (maxN + 1)).map fun n => FoldGroup.binomial n k

/-- An entry of `binomials`. -/
def chooseAt (table : Array (Array ℕ)) (n k : ℕ) : ℕ := ((table[k]?.bind (·[n]?))).getD 0

/-- The largest `c < upper` with `C(c, i) ≤ rank`. -/
def scan (B : ℕ → ℕ → ℕ) (i rank : ℕ) : ℕ → Option ℕ
  | 0 => none
  | u + 1 => if B u i ≤ rank then some u else scan B i rank u

/-- The elements of the colexicographic unrank, the largest first: element `k` is the largest
`c < upper` with `C(c, k) ≤ rank`, and the rank left is 0 at the end. -/
def unrankDown (B : ℕ → ℕ → ℕ) : ℕ → ℕ → ℕ → Option (List ℕ)
  | 0, rank, _ => if rank = 0 then some [] else none
  | k + 1, rank, upper => (scan B (k + 1) rank upper).bind fun c =>
      (unrankDown B k (rank - B c (k + 1)) c).map (c :: ·)

/-- The `k`-subset of `0..n` of colexicographic rank `rank`, in increasing order. -/
def unrank (B : ℕ → ℕ → ℕ) (n k rank : ℕ) : Option (List ℕ) :=
  (unrankDown B k rank n).map List.reverse

/-- The elements of `0..d` that are not in `taken`, in increasing order. -/
def free (d : ℕ) (taken : List ℕ) : List ℕ := (List.range d).filter fun x => !taken.contains x

/-- Each slot `s` is the element `s` of `free d taken`, and no slot is at or above its length. -/
def complement (d : ℕ) (taken slots : List ℕ) : Option (List ℕ) :=
  if slots.all (· < (free d taken).length) then some (slots.map fun s => (free d taken).getD s 0)
  else none

/-- Coefficient `i`: magnitude 1 below `ones` and 2 above, negative when sign bit `i` is set. -/
def coefficient (ones signs i : ℕ) : ℤ :=
  (if signs.testBit i then -1 else 1) * (if i < ones then 1 else 2)

def pairs (ones signs : ℕ) (positions : List ℕ) : Sparse :=
  positions.mapIdx fun i p => (p, coefficient ones signs i)

/-- The challenge of the candidate `value`, or `none` when its rank is out of range. -/
def candidate (G : FoldGroup) (B : ℕ → ℕ → ℕ) (value : ℕ) : Option Sparse :=
  let rank := value / 2 ^ (G.pm1 + G.pm2)
  if G.rankSpace ≤ rank then none else
    (unrank B G.degree G.pm1 (rank % B G.degree G.pm1)).bind fun ones =>
      (unrank B (G.degree - G.pm1) G.pm2 (rank / B G.degree G.pm1)).bind fun slots =>
        (complement G.degree ones slots).map fun twos =>
          pairs G.pm1 (value % 2 ^ (G.pm1 + G.pm2)) (ones ++ twos)

/-- The tables and the threshold of the operator-norm predicate of one degree. -/
structure Filter where
  degree : ℕ
  threshold : ℕ
  cos : Array ℤ
  sin : Array ℤ
  error : ℕ

/-- The tables of `build_tables(d, 48)` and the threshold of the policy of the degree `d`. -/
def Filter.of (d : ℕ) : Filter where
  degree := d
  threshold := FoldGroup.threshold d
  cos := (Array.range (2 * d)).map fun t => (TrigBuild.entry d 48 t).1.1
  sin := (Array.range (2 * d)).map fun t => (TrigBuild.entry d 48 t).2.1
  error := (TrigBuild.epsRoot d 48).toNat

/-- `R_k` or `I_k` of a challenge for one table. -/
def Filter.sum (F : Filter) (table : Array ℤ) (c : Sparse) (k : ℕ) : ℤ :=
  (c.map fun x => x.2 * (table[(2 * k + 1) * x.1 % (2 * F.degree)]?.getD 0)).sum

/-- `accept_strict_parts`: `U_k ≤ T²·Δ²` for each `k < d/2`. -/
def Filter.accepts (F : Filter) (c : Sparse) : Bool :=
  let r : ℕ := (c.map fun x => x.2.natAbs).sum * F.error
  (List.range (F.degree / 2)).all fun k =>
    let re := F.sum F.cos c k
    let im := F.sum F.sin c k
    decide (re ^ 2 + im ^ 2 + 2 * (|re| + |im|) * (r : ℤ) + 2 * (r : ℤ) ^ 2 ≤
      (F.threshold : ℤ) ^ 2 * ((2 ^ 48 : ℕ) : ℤ) ^ 2)

/-- `first_accepted`: the accepted values of the indices from `index`, for at most `fuel` indices,
after `acc`, until `count` are accepted. -/
def firstAccepted {α : Type} (f : ℕ → Option α) (count : ℕ) : ℕ → ℕ → List α → List α
  | _, 0, acc => acc
  | index, fuel + 1, acc =>
      if acc.length = count then acc
      else firstAccepted f count (index + 1) fuel (acc ++ (f index).toList)

/-- The fold challenges of a group from its squeezed message, in the ring of degree
`G.degree`. -/
def sample (G : FoldGroup) (bytes : ByteArray) : Option (List Sparse) :=
  if bytes.size ≠ G.messageBytes then none else
    let B := chooseAt (binomials G.degree (max G.pm1 G.pm2))
    let decode (j : ℕ) := candidate G B (readBits bytes (j * G.candidateBits) G.candidateBits)
    if G.filtered then
      let F := Filter.of G.degree
      let accepted := firstAccepted (fun j => (decode j).filter F.accepts) G.coordinates 0
        G.candidates []
      if accepted.length = G.coordinates then some accepted else none
    else
      let all := (List.range G.coordinates).filterMap decode
      if all.length = G.coordinates then some all else none

/-- A challenge of the subring in the ring of `A`: each position times `ring/degree`. -/
def ambient (G : FoldGroup) (c : Sparse) : Sparse :=
  c.map fun x => (x.1 * (G.ring / G.degree), x.2)

/-- The fold challenges of a group from its squeezed message, in the ring of `A`, or `none` when
the operator-norm pool runs out (`VerifierFoldDraw`: the verifier rejects). -/
def foldChallenges (G : FoldGroup) (bytes : ByteArray) : Option (List Sparse) :=
  (sample G bytes).map fun cs => cs.map (ambient G)

end NightstreamFPrime.Spec.Compression.Akita.Replay
