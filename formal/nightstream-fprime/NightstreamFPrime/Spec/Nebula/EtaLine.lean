import NightstreamFPrime.Spec.Nebula.Soundness

/-! What a segment's product test reads from the context (CW-10.8). The segment views, the close
checks and the challenge input read the plan, the hash and the plan digest, but not the
context's challenge function: changing it (`Context.withEta`) keeps them, and the segment's
challenges become the new function at the same input (`eta_withEta`). So a random-oracle model
that reads `η` from the input alone changes only the product test. The input's timestamp is
canonical, because a segment that closes opens below `2 ^ W_ts ≤ 2 ^ 31`
(`ClosesAt.openTs_lt`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Nebula

variable {E Digest : Type}

/-- The context `ctx` with the challenge function `eta`. -/
def Context.withEta (ctx : Context E Digest) (eta : ℕ → EtaInput Digest → E × E) :
    Context E Digest :=
  { ctx with eta := eta }

/-- The segment views of a run do not read the challenge function. -/
theorem segmentView_withEta {σ : Type} (ctx : Context E Digest)
    (eta : ℕ → EtaInput Digest → E × E) (run : List (Invocation σ Digest)) (k : ℕ) :
    segmentView (ctx.withEta eta) run k = segmentView ctx run k := rfl

namespace SegmentView

variable (ctx : Context E Digest) (eta : ℕ → EtaInput Digest → E × E) (η : E × E)
  (v : SegmentView Digest)

theorem etaInput_withEta : v.etaInput (ctx.withEta eta) = v.etaInput ctx := rfl

theorem eta_withEta : v.eta (ctx.withEta eta) = eta v.index (v.etaInput ctx) := rfl

theorem closesAt_withEta [CommRing E] : v.ClosesAt (ctx.withEta eta) η ↔ v.ClosesAt ctx η :=
  ⟨fun ⟨length, rows, opsRoot, initialRoot, finalRoot, products⟩ =>
    ⟨length, rows, opsRoot, initialRoot, finalRoot, products⟩,
   fun ⟨length, rows, opsRoot, initialRoot, finalRoot, products⟩ =>
    ⟨length, rows, opsRoot, initialRoot, finalRoot, products⟩⟩

theorem badChallenge_withEta [CommRing E] :
    BadChallenge (ctx.withEta eta) v ↔
      ¬ (v.multisets ctx.plan.toGeometry).Balanced ∧
        (v.multisets ctx.plan.toGeometry).ProductEq (eta v.index (v.etaInput ctx)) :=
  Iff.rfl

variable {ctx η v}

/-- A segment that closes opens below `2 ^ W_ts`, because its first step's rows bound the
timestamp. -/
theorem ClosesAt.openTs_lt_pow [CommRing E] (valid : ctx.plan.Valid)
    (closes : v.ClosesAt ctx η) :
    v.openTs < 2 ^ ctx.plan.wTs := by
  obtain ⟨z, rest, records⟩ : ∃ z rest, v.records = z :: rest := by
    cases recordsEq : v.records with
    | nil =>
        have := closes.length
        rw [recordsEq, List.length_nil] at this
        exact absurd this.symm (Nat.pos_iff_ne_zero.mp valid.positive.2.2.1)
    | cons z rest => exact ⟨z, rest, rfl⟩
  have rows := closes.rows
  rw [records] at rows
  have boundary := rows.1.boundary
  omega

/-- A segment that closes opens at a canonical timestamp. -/
theorem ClosesAt.openTs_lt [CommRing E] (valid : ctx.plan.Valid) (closes : v.ClosesAt ctx η) :
    v.openTs < fieldModulus :=
  lt_of_lt_of_le (closes.openTs_lt_pow valid)
    ((Nat.pow_le_pow_right (by norm_num) valid.fieldEncoding).trans
      (by norm_num [fieldModulus]))

end SegmentView

end NightstreamFPrime.Spec.Nebula
