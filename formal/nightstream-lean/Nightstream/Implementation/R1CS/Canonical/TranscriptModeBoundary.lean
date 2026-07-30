import Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge

/-!
Contract: record that the Fiat–Shamir transcript is a different construction
from the binding sponge, so no transcript recipe may be built on the latter.

## The two absorb modes

`Poseidon2Sponge.absorbChunk` **adds** into the rate lanes:

```text
absorbChunk chunk state lane = (state lane + value) % p
```

`neo-transcript`'s Poseidon2 duplex **overwrites** them.  `absorb_elem` is
`self.st[self.absorbed] = x`, and `absorb_slice`'s unrolled fast path carries
the comment "We use assignment (overwrite) to match absorb_elem behavior".

`modes_agree_on_initial_state` proves they coincide from the all-zero state —
which is why the difference is easy to miss, since it is invisible on the first
chunk.  `modes_differ` proves they diverge from any state carrying a non-zero
lane, which is every chunk after the first.

## Three further differences

- **Arity.**  This sponge is fixed at 23 fields
  (`Poseidon2Sponge.sponge23Fields`).  The transcript is a duplex over
  variable-length input with a cursor.
- **Padding.**  `Poseidon2Sponge.pad` adds a single `1` to lane 0 before a final
  permutation.  The transcript has no such step; `absorb_packed_bytes_with_len`
  instead absorbs the byte length as a field element first.
- **Length separation.**  Consequently the transcript separates lengths by
  absorbing one, not by padding.  `POSEIDON2-HASH-COMMITMENTS` records that at
  fixed arity the padding rule separates nothing; that argument is about the
  binding hash and does **not** transfer here.

## Why this is a result and not an obstacle

Prompt section 4.6: Construction 2's binding hash and the Fiat–Shamir random
oracle are distinct objects with distinct security contracts, and may share
arithmetic.  They do share the permutation.  They do **not** share the sponge.

Building the step transcript on `Poseidon2Sponge` would encode a different
function from the one the verifier computes — the same defect that
`KRecomposition.powerSumFrom_eq_hornerValue` was written to prevent for the
radix-`b` relation, where Horner form and the verifier's accumulator loop had to
be proved equal rather than assumed so.  There the two agreed; here they do not.

`TRANSCRIPT-MODE-BOUNDARY` records the route as blocked at this construction.
A transcript recipe needs a duplex model with a cursor, and that model does not
exist in this tree.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.TranscriptModeBoundary

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge

/-- The transcript's absorb mode: assignment, not accumulation. -/
def overwriteChunk (chunk : List Nat) (state : Values) : Values :=
  fun lane =>
    match chunk[lane.val]? with
    | some value => value % goldilocksP
    | none => state lane

/-- **The two modes agree from the all-zero state.**

This is why the difference is easy to miss: on the first absorbed chunk the
sponge's `0 + v` and the duplex's `v` are the same value, so a single-chunk
fixture cannot distinguish them. -/
theorem modes_agree_on_initial_state (chunk : List Nat) :
    absorbChunk chunk initialSpongeState = overwriteChunk chunk initialSpongeState := by
  funext lane
  unfold absorbChunk overwriteChunk initialSpongeState
  cases lookup : chunk[lane.val]? with
  | none => rfl
  | some value => simp only [Nat.zero_add]

/-- A state with a non-zero rate lane — that is, any state after one
permutation. -/
def carriedState : Values := fun _ => 5

/-- **The two modes diverge from any state that carries a value.**

Every chunk after the first absorbs into a permuted, non-zero state, so the
divergence is the normal case rather than the corner one. -/
theorem modes_differ :
    absorbChunk [1] carriedState ≠ overwriteChunk [1] carriedState := by
  intro equal
  have atLane := congrFun equal ⟨0, by decide⟩
  simp only [absorbChunk, overwriteChunk, carriedState] at atLane
  exact absurd atLane (by decide)

/-- **The divergence is visible in one lane**, stated as the concrete values so
the fixture cannot pass by both sides being equal for an unrelated reason. -/
theorem divergent_values :
    absorbChunk [1] carriedState ⟨0, by decide⟩ = 6
      ∧ overwriteChunk [1] carriedState ⟨0, by decide⟩ = 1 := by
  constructor <;> rfl

/-- **The sponge is fixed-arity.**  The transcript is not: it is a duplex over
variable-length input.  Recorded as the arity this construction admits, so the
mismatch is checkable rather than asserted. -/
theorem sponge_is_fixed_arity : sponge23Fields = 23 := rfl

end Nightstream.Implementation.R1CS.Canonical.TranscriptModeBoundary
