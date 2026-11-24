import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression.Akita.Extract.Present
import NightstreamFPrime.Spec.Compression.Akita.Extract.PresentChecks
import NightstreamFPrime.Spec.Compression.Akita.Extract.PresentDecode
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceLink
import NightstreamFPrime.Spec.Compression.Akita.Extract.LiftSource
import NightstreamFPrime.Spec.Compression.Akita.Replay.CodeSlot

/-! Axiom audits for the transcript form of a run of a game (`Extract/Present.lean`) and for the
fold challenges of a transcript (`Extract/PresentDecode.lean`), with the checks of the levels
(`Extract/PresentChecks.lean`). The examples at the end show that the hypotheses hold for the
slots `Replay.CodeSlot E` of the claims: the fair decoders of the draws `E` and `Fin 22 → E`
for the 22 slots of `Key20.Tower`, a presented segment for each run of `Game.sumcheck`, and the
checks of the level 1 of `key20.aks`. -/

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
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Extract.Tier.Decodes.challenge_mem

namespace AkitaPresentGuards

open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract

/-- The draws `E` and `Fin 22 → E` of the checks have fair decoders for the 22 slots
`Replay.CodeSlot E` of `Key20.Tower` (`exists_fair_field`, `exists_fair_fields`). -/
example {E Coord : Type} [Field E] [Fintype E] [Fintype Coord] [DecidableEq Coord]
    [Nonempty Coord] (slots : 22 ≤ Fintype.card Coord) :
    (∃ decode : (Coord → Replay.CodeSlot E) → E, Fair decode) ∧
      ∃ decode : (Coord → Replay.CodeSlot E) → Fin Key20.rootVars → E, Fair decode :=
  ⟨exists_fair_field, exists_fair_fields slots⟩

/-- Each run of `Game.sumcheck` has a presented segment with the wire messages `ℕ → List E`, so
`Presents` and `Presentable` hold together. -/
example {E Coord : Type} [Field E] [Fintype E] [DecidableEq E] [Fintype Coord]
    [DecidableEq Coord] [Nonempty Coord] (d rounds : ℕ) (round : List E → Polynomial E)
    (point : List E) (claim : E)
    {path : List Game.Move} {result : Option (List E × E)}
    (runs : (Game.sumcheck d round rounds point claim).Runs path result) :
    ∃ (first : ℕ → List E) (moves : List ((Coord → Replay.CodeSlot E) × (ℕ → List E))),
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
    (block : Fin S.blocks → Coord) (decode : Replay.CodeSlot E → Option (Rq O.carrierDegree))
    (read : Reader Wire)
    (weight start ν : ℕ) (vars : μ ≤ Fintype.card Coord) (rows : batch ≤ Fintype.card Coord)
    (wire : Wire) (challenge : Fin S.blocks → Rq O.carrierDegree) (table : Table) :
    Presentable (Coord → Replay.CodeSlot E) (List E)
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

end AkitaPresentGuards
