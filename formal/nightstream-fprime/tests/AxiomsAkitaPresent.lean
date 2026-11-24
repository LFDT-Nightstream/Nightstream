import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression.Akita.Extract.Present
import NightstreamFPrime.Spec.Compression.Akita.Extract.PresentChecks
import NightstreamFPrime.Spec.Compression.Akita.Extract.PresentDecode
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceLink
import NightstreamFPrime.Spec.Compression.Akita.Extract.LiftSource
import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerCount

/-! Axiom audits for the transcript form of a run of a game (`Extract/Present.lean`) and for the
fold challenges of the code against the decoders of the soundness claims
(`Extract/PresentDecode.lean`), with the checks of the levels (`Extract/PresentChecks.lean`).
The examples at the end show that the hypotheses hold: the fair decoders of the draws `E` and
`Fin 22 → E` for the 22 slots of `Key20.Tower`, a presented segment for each run of
`Game.sumcheck`, the checks of the level 1 of `key20.aks`, `K` from 128 to 131, the families of
`key20.aks` with `2^128` members or more, and the group `(31, 10)` of the degree 64 of the code,
whose `2^128` words all decode. -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Fair.mass_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.sample_surjective
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.pathOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Presents
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.foldl_pure
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.foldl_settle_eq_pure_iff
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.position_eq_pure_iff
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Presents.position_iff
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.position_of_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Segment.game_eq_pure_iff
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Tier.game_eq_pure_iff
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Tier.passed_of_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.unchunk
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.unchunk_surjective
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.chunkReader
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.chunkReader_message
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Presentable
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Presentable.bind
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.exists_pathOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.exists_presents
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.listChunk_onto
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.presentable_sumcheck
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Presentable.ite
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.presentable_eqSumcheck
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.presentable_sumcheckWith
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.presentable_rangeTree
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.presentable_leafStage
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.presentable_normLeafStage
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.presentable_rangeCheck
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.presentable_drawBin
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.presentable_fusedCheck
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.presentable_foldCheck
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.presentable_foldCheckSq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.presentable_setupCheck
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.presentable_offloadCheck
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.presentable_tensorCheck
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.card_lt_words
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.not_injective_words
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.exists_not_decoded
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.exists_word_missed
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.draw_misses
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.draw_misses_field
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Tier.Decodes.challenge_mem
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Tier.Decodes.challenge_ne

namespace AkitaPresentGuards

open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract

/-- The draws `E` and `Fin 22 → E` of the checks have fair decoders for the 22 slots of
`Key20.Tower` (`exists_fair_slot`, `exists_fair_slots`). -/
example {E Coord : Type} [Field E] [Fintype E] [Fintype Coord] [DecidableEq Coord]
    [Nonempty Coord] (slots : 22 ≤ Fintype.card Coord) :
    (∃ decode : (Coord → E) → E, Fair decode) ∧
      ∃ decode : (Coord → E) → Fin Key20.rootVars → E, Fair decode :=
  ⟨exists_fair_slot, exists_fair_slots slots⟩

/-- Each run of `Game.sumcheck` has a presented segment with the wire messages `ℕ → List E`, so
`Presents` and `Presentable` hold together. -/
example {E Coord : Type} [Field E] [Fintype E] [DecidableEq E] [Fintype Coord]
    [DecidableEq Coord] [Nonempty Coord] (d rounds : ℕ) (round : List E → Polynomial E)
    (point : List E) (claim : E)
    {path : List Game.Move} {result : Option (List E × E)}
    (runs : (Game.sumcheck d round rounds point claim).Runs path result) :
    ∃ (first : ℕ → List E) (moves : List ((Coord → E) × (ℕ → List E))),
      Presents (chunkReader (List E)) first moves path (Game.sumcheck d round rounds point claim) ∧
        position (chunkReader (List E)) (Game.sumcheck d round rounds point claim) first moves =
          .pure result :=
  exists_presents runs (presentable_sumcheck d round rounds point claim)

/-- The checks of a packing level with the tensor reduction (`Tier.tensored` of `liftFold`, the
level 1 of `key20.aks` with `μ = 21` and `s_rel = 4`) are presentable when an answer has a slot
for each coordinate of a draw. -/
example {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E Wire Coord : Type} [Field E]
    [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E] [Fintype Coord] [DecidableEq Coord]
    [Nonempty Coord] (β : Module.Basis (Fin O.planes) Zq E) (key : Key S) (openKey : OpeningKey O)
    (claim : Wire → Claim O E) (μ : ℕ) (fits : O.cells fold ≤ 2 ^ μ) (digits : ℕ)
    (room : O.quotientRoom fold digits ≤ 2 ^ μ) (batch : ℕ) (rowsFit : O.rowCount + 1 ≤ 2 ^ batch)
    (block : Fin S.blocks → Coord) (decode : E → Option (Rq O.carrierDegree)) (read : Reader Wire)
    (weight start ν : ℕ) (vars : μ ≤ Fintype.card Coord) (rows : batch ≤ Fintype.card Coord)
    (wire : Wire) (challenge : Fin S.blocks → Rq O.carrierDegree) (table : Table) :
    Presentable (Coord → E) (List E)
      ((Tier.tensored (liftFold β key openKey claim μ fits digits room batch rowsFit block decode
        read weight start) ν).check wire challenge table) :=
  Presentable.bind (presentable_foldCheck _ _ _ _ _ vars rows) fun result => by
    rcases result with _ | ⟨r2, v⟩
    · trivial
    · exact presentable_tensorCheck _ _ _ _

/-- The counts `μ` and `s_rel` of the draws of each level of `key20.aks` are at most 22. -/
example : [Key20.rootVars, Key20.rootBatch, Key20.level₁.vars, Key20.level₁.batch,
    Key20.level₂.vars, Key20.level₃.vars, Key20.level₄.vars, Key20.level₅.vars].all (· ≤ 22) := by
  decide

/-- The words of the groups of `key20.aks` and `key30.aks` have `K` from 128 to 131 bits. -/
example {E : Type} [Field E] [Fintype E] [Algebra Zq E] [Degree2 E] :
    Fintype.card E < 2 ^ 128 ∧ Fintype.card E < 2 ^ 131 :=
  ⟨card_lt_words le_rfl, card_lt_words (by decide)⟩

/-- `draw` from one slot of `E` misses a member of the family `(31, 10)` of the degree 64, and,
with `Key20.Kept`, a member of the filtered family `(31, 11)` that passes the filter. -/
example {E : Type} [Field E] [Fintype E] [Algebra Zq E] [Degree2 E] (kept : Key20.Kept) :
    (∃ c, c ∈ SparseFamily 64 31 10 ∧ True ∧
      ∀ slot : E, draw (SparseFamily 64 31 10) (fun _ => True) slot ≠ some c) ∧
    ∃ c, c ∈ SparseFamily 64 31 11 ∧ TrigTable.codeAccepts 64 18 c ∧
      ∀ slot : E, draw (SparseFamily 64 31 11) (TrigTable.codeAccepts 64 18) slot ≠ some c :=
  ⟨draw_misses_field (by simpa using Key20.families_large.1), draw_misses_field kept.1⟩

/-- The group `(31, 10)` of the degree 64 of the code has words of 128 bits that all decode, so
a word of the code decodes to a fold challenge that no decoder from one slot of `E` gives. -/
example {E : Type} [Field E] [Fintype E] [Algebra Zq E] [Degree2 E]
    (decode : E → Option (Rq 64)) :
    ∃ w c, Replay.slotDecode ⟨1, 64, 64, 31, 10, false⟩ w = some c ∧
      ∀ slot, decode slot ≠ some c := by
  have wide : Replay.FoldGroup.candidateBits ⟨1, 64, 64, 31, 10, false⟩ = 128 := by
    rw [Replay.FoldGroup.candidateBits, ite_eq_right (by decide),
      Nat.log_eq_of_pow_le_of_lt_pow (m := 87) (by decide) (by decide)]
    rfl
  have all : Nat.card {w // (Replay.slotDecode ⟨1, 64, 64, 31, 10, false⟩ w).isSome} = 2 ^ 128 := by
    rw [Nat.card_congr (Equiv.subtypeUnivEquiv fun w =>
      Replay.slotDecode_isSome rfl (by decide) w), Nat.card_eq_fintype_card, Fintype.card_fin, wide]
  exact exists_word_missed (Replay.slotDecode ⟨1, 64, 64, 31, 10, false⟩)
    (Replay.slotDecode_injective _) decode
    (by rw [all]; exact card_lt_words le_rfl)

end AkitaPresentGuards
