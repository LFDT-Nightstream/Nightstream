import CompressionLean.Lib.Cube
import Mathlib.Data.Nat.Size

namespace CompressionLean.Shrink

open CompressionLean.Lib

variable {K : Type} [CommRing K]

abbrev Track (K : Type) := List K × ℕ

def walkBits (tracks : List (Track K)) (count : ℕ) : ℕ :=
  (tracks.map fun track => max track.1.length (Nat.size track.2)).foldr max (Nat.size count)

def trackFactor (point : List K) (t bit : ℕ) : K :=
  if t < point.length then (if bit = 1 then point.getD t 0 else 1 - point.getD t 0)
  else if bit = 1 then 0 else 1

def tracksStep (t b : ℕ) : List (Track K) → ℕ → K × ℕ
  | [], _ => (1, 0)
  | (point, offset) :: rest, carry =>
      (trackFactor point t ((offset / 2 ^ t % 2 + b + carry % 2) % 2) * (tracksStep t b rest (carry / 2)).1,
        (offset / 2 ^ t % 2 + b + carry % 2) / 2 + 2 * (tracksStep t b rest (carry / 2)).2)

def nextBelow (countBit b below : ℕ) : ℕ :=
  if b < countBit then 1 else if countBit < b then 0 else below

def walkStep (tracks : List (Track K)) (count t : ℕ) (states : ℕ → ℕ → K) (carry' below' : ℕ) : K :=
  ∑ carry ∈ Finset.range (2 ^ tracks.length), ∑ below ∈ Finset.range 2, ∑ b ∈ Finset.range 2,
    if (tracksStep t b tracks carry).2 = carry' ∧ nextBelow (count / 2 ^ t % 2) b below = below' then
      states carry below * (tracksStep t b tracks carry).1
    else 0

def walk (tracks : List (Track K)) (count : ℕ) : ℕ → ℕ → ℕ → K
  | 0 => fun carry below => if carry = 0 ∧ below = 0 then 1 else 0
  | t + 1 => walkStep tracks count t (walk tracks count t)

def diagonal (tracks : List (Track K)) (count : ℕ) : K :=
  walk tracks count (walkBits tracks count) 0 1

end CompressionLean.Shrink
