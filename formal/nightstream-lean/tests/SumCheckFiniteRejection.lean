import Nightstream.SuperNeo.SumCheck.VerifierCertificate

/-!
Negative witnesses for the finite verifier-visible SumCheck certificate.

Property: `SUM-FINITE-CERT`.

Owns: one honest fixture plus one rejecting fixture for every rejection branch
of `Finite.checkChain`, so that "executable replay exactly matches the finite
relation" cannot be satisfied by a checker that accepts too much. Each negative
mutates exactly one field of the honest fixture, and the positive control keeps
the negatives from being vacuous.

Does not own: symbolic SumCheck soundness or bad-challenge extraction
(`SUM-CLAIM`, `SUM-SOUND`, covered in `tests/SumCheck.lean`), root counting,
PiCCS integration, or transcript replay.

| Rejection branch | Witness |
|---|---|
| empty coefficient list | `emptyMessage_rejected` |
| trailing-zero (non-canonical) shape | `trailingZero_rejected` |
| declared degree above the verifier cap | `degreeTwo_accepted_at_cap_two` + `degreeAboveCap_rejected` |
| claimed initial not `p(0) + p(1)` | `brokenInitialClaim_rejected` |
| terminal not the replayed evaluation | `brokenTerminal_rejected` |
| fewer challenges than messages | `missingChallenge_rejected` |
| more challenges than messages | `extraChallenge_rejected` |
-/

set_option autoImplicit false

namespace NightstreamTests.SumCheckFiniteRejection

open Nightstream.SuperNeo.SumCheck.Finite

/-- Concrete finite operations. `Nat` is sufficient: every rejection below is a
shape or claimed-chain failure, not a field-theoretic one. -/
def natOps : Ops Nat where
  zero := 0
  one := 1
  add := Nat.add
  mul := Nat.mul

/-- Verifier-owned degree cap for the honest fixture. -/
def maxDegree : Nat := 1

/-- Honest round polynomial `p(X) = 1 + 2X`, constant-first. -/
def honestMessage : Message Nat := ⟨[1, 2]⟩

def honestCertificate : Certificate Nat := ⟨[honestMessage]⟩

/-- The verifier's single challenge. -/
def honestChallenges : List Nat := [3]

/-- `p(0) + p(1) = 1 + 3 = 4`. -/
def honestInitial : Nat := 4

/-- `p(3) = 1 + 2 * 3 = 7`. -/
def honestTerminal : Nat := 7

/-! ## Positive control

Without this, every rejection below could hold for the trivial reason that the
fixture was never acceptable. -/

theorem honest_accepted :
    check natOps maxDegree honestInitial honestChallenges honestTerminal
      honestCertificate = true := by
  decide

/-- The executable acceptance above is the finite relation, not a weaker
surrogate. -/
theorem honest_chain :
    Accepted natOps maxDegree honestInitial honestChallenges honestTerminal
      honestCertificate :=
  (check_eq_true_iff_accepted natOps maxDegree honestInitial honestChallenges
    honestTerminal honestCertificate).1 honest_accepted

/-! ## One rejection per branch -/

/-- A raw message may parse with no coefficients. Acceptance must reject it
rather than treating it as the zero polynomial — note the claimed chain is
chosen to close (`0 = 0 + 0`, replay `0`), so canonical shape is the only
failing branch. -/
theorem emptyMessage_rejected :
    check natOps maxDegree 0 honestChallenges 0
      ⟨[Message.mk []]⟩ = false := by
  decide

/-- A trailing zero coefficient is a second encoding of a lower-degree
polynomial. Canonical shape rejects it, so degree accounting stays unique. The
cap is raised here so this isolates canonicality rather than the degree check. -/
theorem trailingZero_rejected :
    check natOps 2 honestInitial honestChallenges honestTerminal
      ⟨[Message.mk [1, 2, 0]]⟩ = false := by
  decide

/-- `q(X) = 1 + 2X + X²` has `q(0) + q(1) = 5` and `q(3) = 16`, so its claimed
chain closes at cap `2`. This pins that the cap is the only thing separating
the two results below. -/
theorem degreeTwo_accepted_at_cap_two :
    check natOps 2 5 honestChallenges 16 ⟨[Message.mk [1, 2, 1]]⟩ = true := by
  decide

/-- The same certificate is rejected at cap `1`. Because the chain provably
closes above, this isolates the verifier-owned degree cap. -/
theorem degreeAboveCap_rejected :
    check natOps maxDegree 5 honestChallenges 16
      ⟨[Message.mk [1, 2, 1]]⟩ = false := by
  decide

/-- The claimed initial value must equal `p(0) + p(1)`. A prover-chosen initial
is rejected. -/
theorem brokenInitialClaim_rejected :
    check natOps maxDegree 5 honestChallenges honestTerminal
      honestCertificate = false := by
  decide

/-- The terminal must equal the replayed evaluation at the challenge, so the
final claim cannot be restated freely. -/
theorem brokenTerminal_rejected :
    check natOps maxDegree honestInitial honestChallenges 8
      honestCertificate = false := by
  decide

/-- Acceptance consumes exactly one challenge per round. A missing challenge is
rejected rather than silently truncating the chain. -/
theorem missingChallenge_rejected :
    check natOps maxDegree honestInitial [] honestTerminal
      honestCertificate = false := by
  decide

/-- A trailing unconsumed challenge is rejected, so extra verifier coins cannot
be appended to an otherwise honest transcript. -/
theorem extraChallenge_rejected :
    check natOps maxDegree honestInitial [3, 3] honestTerminal
      honestCertificate = false := by
  decide

/-! ## Length discipline

The lockstep law is exercised on the honest fixture rather than assumed. -/

theorem honest_lockstep :
    honestCertificate.rounds.length = honestChallenges.length :=
  Chain.messages_length_eq_challenges_length natOps maxDegree honestInitial
    honestTerminal honestCertificate.rounds honestChallenges honest_chain

end NightstreamTests.SumCheckFiniteRejection
