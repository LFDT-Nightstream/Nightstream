import NightstreamFPrime.Export.BoundedDepth.FinalKey20Challenge
import NightstreamFPrime.Export.BoundedDepth.FinalKey20Instances
import NightstreamFPrime.Spec.Compression.Akita.Key20Accept
import NightstreamFPrime.Spec.Compression.Akita.FitChance
import NightstreamFPrime.Spec.Compression.Akita.KeptTail
import NightstreamFPrime.Spec.Compression.Akita.Key20Opening

/-! # Claims: the Akita openings for `key20.aks`

The results for the Akita openings of the compressed proof under the schedule file
`crates/ns-compress/artifacts/key20.aks` of the test key of `2^9` blocks (Akita, ePrint 2026/1983,
revision of 2026-11-23; `SPEC.md` Algorithm 10 and section 9; `SECURITY_MODEL.md` adversary 6 and
trust item 7). The results of Akita that hold for each schedule are in `Claims.Akita`.

- Model: the answers of the stages are one uniform table on the keys of the stage inputs (their
  messages) with independent slots, and an answer of one stage has one slot for each fold
  coordinate. A slot is `Spec.Compression.Akita.Replay.CodeSlot E`, and the fold decoders are
  `slotDecode` of the pinned Akita code on the word of a slot. The verifier is the Lean verifier of the opening. It is not the verifier of the pinned Akita code (`SECURITY_MODEL.md`, trust item 7).
- Scope: no claim of this file is for `crates/ns-compress/artifacts/key30.aks`, the file of the
  key of `2^19` blocks: the claims for that file are in `Claims.AkitaKey30`.
- Bounds: Nightstream results, not Akita Theorem 10.46. One query of the model returns all the
  slots of a key, and the paper counts a query for each indexed coordinate stream (Akita Section
  10.3). The file has no `(λ_FS, Q_max)`. A solution of Module-SIS is an output of the extractor,
  so a bound with no term `breakChance` is not the knowledge error of the paper.
- Assumptions: `[MSIS]` and `[KECCAK]`. Each bound has the term `breakChance`, or it bounds only
  the event that the extractor returns no output. Lean gives no bound on the probability that an
  output is a solution of Module-SIS. `key20_instances` and `key20_instance_list` give the
  instances of Module-SIS that `breakChance` is about, and Lean does not prove their hardness.
  `key20_instances_shared` gives a solution for the view of one setup vector of an instance of
  the 43 views, when the matrices are these views.
- The filters of the challenge families are the predicate of the pinned Akita code
  (`Spec.Compression.Akita.TrigTable.codeAccepts`). Lean proves the error bound of its integer
  tables (`Spec.Compression.Akita.TrigTable.code_accurate`), so the operator bound `Γ` is a
  theorem (`Spec.Compression.Akita.TrigTable.filtered_code`). `key20_kept` reduces the count
  `Spec.Compression.Akita.Key20.Kept` of these filters to a count at one frequency.
- For key20, `Spec.Compression.Akita.Replay.verifyProof_accepts20` gives an accepted opening
  transcript for each proof that the Lean byte verifier `verifyProof` accepts under its hypotheses.
  `Replay.verifyProof_bound20` gives the opening extraction bound under `carries`: for each oracle
  table, the prover transcript must present the bytes that the verifier accepts. -/

namespace NightstreamFPrime.Claims.AkitaKey20

open Spec Spec.Compression Lifecycle Lifecycle.BoundedDepth Export.BoundedDepth
open scoped BigOperators Spec.FieldExtensionRing Spec.Compression.Production FinalKey20Algebra
attribute [local instance low] Classical.propDecidable


/-- The output of the extractor for the opening of `crates/ns-compress/artifacts/key20.aks`, the
schedule of the test key of `2^9` blocks (Akita Theorem 10.17 with Akita Proposition 10.44 (i)).
It does not cover `key30.aks`. The verifier is the Lean verifier `Key20.Tower.accepts` with the
fold decoders of the pinned Akita code. `Replay.verifyProof_accepts20` gives an accepted opening
transcript for each accepted proof under its hypotheses. Each extractor output for an accepted
transcript is a certified decommitment of the commitment of each group of
the root to a table with the claimed value (`OpensAt`), or a solution of Module-SIS for a matrix
of a group, of a level, or for `A_term`. The filters of the families are the predicate of the
pinned Akita code, and Lean proves the operator bound `Γ` for them (Akita Theorem H.1). -/
theorem key20_opening_sound
    {E : Type}
    [fieldE : Field E]
    [fintypeE : Fintype E]
    [decidableEqE : DecidableEq E]
    [algebraZqE : Algebra Akita.Zq E]
    [degree2EAlgebraZqE : Akita.Degree2 E]
    {Wire : Type}
    {Coord : Type}
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [nonemptyCoord : Nonempty Coord]
    {total : ℕ}
    {o : Akita.Key20.Tower E Wire Coord total}
    (drawn : o.Drawn)
    (slots : 22 ≤ Fintype.card Coord)
    (prover : RandomOracle.OracleComp (Akita.Extract.Address Wire total) (Option Coord → Akita.Replay.CodeSlot E) (ℕ → Wire))
    (budget : ℕ)
    (table : Akita.Extract.Address Wire total → Option Coord → Akita.Replay.CodeSlot E)
    (tape : Checkpoint.Tape (Akita.Extract.Address Wire total) (Option Coord) (Akita.Replay.CodeSlot E) budget
          o.body.protocol.stages)
    (out : Akita.Extract.Opened Akita.Key20.root.Weak
            (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₁.opening)
              (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₂.opening)
                (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₃.opening)
                  (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₄.opening)
                    (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₅.opening)
                      (Akita.Extract.Prefix Wire (Option Coord → Akita.Replay.CodeSlot E) total ×
                        Option (Akita.TerminalFork Akita.Key20.terminal))))))) ⊕
          (Akita.Batch.LevelBreak o.key o.openKey fun g =>
              2 * (2 * Akita.Key20.rootMass g) *
                (Akita.Key20.root.fold g).sourceBound (Akita.Key20.root.shape g)) ⊕
            Akita.Extract.LevelBreak o.key₁ o.openKey₁
                (2 * (2 * 51) * Akita.Key20.level₁.fold.sourceBound Akita.Key20.level₁.shape) ⊕
              Akita.Extract.Route.Break o.key₂ o.openKey₂ 31
                  (Akita.Key20.level₂.fold.sourceBound Akita.Key20.level₂.shape)
                  Akita.Extract.Route.coefficient ⊕
                Akita.Extract.Route.Break o.key₃ o.openKey₃ 31
                    (Akita.Key20.level₃.fold.sourceBound Akita.Key20.level₃.shape)
                    (Akita.Extract.Route.euclidean Akita.Key20.norm₃.operator Akita.Key20.norm₃.limit) ⊕
                  Akita.Extract.Route.Break o.key₄ o.openKey₄ 53
                      (Akita.Key20.level₄.fold.sourceBound Akita.Key20.level₄.shape)
                      (Akita.Extract.Route.euclidean Akita.Key20.norm₄.operator Akita.Key20.norm₄.limit) ⊕
                    Akita.Extract.Route.Break o.key₅ o.openKey₅ 53
                        (Akita.Key20.level₅.fold.sourceBound Akita.Key20.level₅.shape)
                        (Akita.Extract.Route.euclidean Akita.Key20.norm₅.operator Akita.Key20.norm₅.limit) ⊕
                      Option
                        ((Akita.Extract.Route.euclidean Akita.Key20.normTerminal.operator
                              Akita.Key20.normTerminal.limit).Kernel
                          o.terminal.matrix o.terminal.weight o.terminal.bound))
    (result : out ∈ Checkpoint.extract o.body prover budget table tape) :
    o.accepts (o.body.protocol.final prover table) ∧
      Sum.elim
        (fun certificate =>
          Akita.Extract.OpensAt 0 o.commit o.value o.place o.key Akita.Key20.rootMass
            (o.body.protocol.input prover table 0) (o.body.witness certificate))
        o.body.breaks out :=
  @NightstreamFPrime.Spec.Compression.Akita.Key20.Tower.extract_sound E fieldE fintypeE decidableEqE
    algebraZqE degree2EAlgebraZqE Wire Coord fintypeCoord decidableEqCoord nonemptyCoord total o
    drawn slots prover budget table tape out result

/-- Akita Theorem 10.17 in the form (161) for the opening of
`crates/ns-compress/artifacts/key20.aks`: a Nightstream bound in the role of Akita Theorem 10.46,
not that theorem (Bounds, in the file header). It does not cover `key30.aks`. The model is one
uniform table of the oracle on keys with independent slots, and the verifier is the Lean verifier
`Key20.Tower.accepts` with the fold decoders of the code. For the byte verifier, the corresponding
bound is `Replay.verifyProof_bound20` under its hypothesis `carries`. For each prover with at most
`Q` queries, the probability that the verifier
accepts and the extractor returns no output is at most
`(Q + 1)·(317479/(1024·q²) + 135/q²) + (Q + 1)·142/budget`: `317479/(1024·q²)` for the
draws of the checks, `135/q²` for the 135 fold coordinates (the slot probability of each decoder
of the code is at most `2^-128 ≤ 1/q²`), and 142 retry loops. The draw of
each check has the factor `2^{−g}` for the work bits `g` of its site in the Akita code (Akita
Remark 10.18, `Extract.workBits`), and the fold terms have no work credit. The bound does not
contain the probability that the output is a solution of Module-SIS (`[MSIS]`). The operator
bound of the filtered families is a theorem for the filter of the code. With `drawn`, each slot
gives a fold challenge at each level (`Spec.Compression.Akita.Key20.Tower.Drawn.decodes`, from
the count `drawn.kept` of Akita Table 12). -/
theorem key20_opening_bound
    {E : Type}
    [fieldE : Field E]
    [fintypeE : Fintype E]
    [decidableEqE : DecidableEq E]
    [algebraZqE : Algebra Akita.Zq E]
    [degree2EAlgebraZqE : Akita.Degree2 E]
    {Wire : Type}
    {Coord : Type}
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [nonemptyCoord : Nonempty Coord]
    {total : ℕ}
    {o : Akita.Key20.Tower E Wire Coord total}
    [finiteWire : Finite Wire]
    (drawn : o.Drawn)
    (slots : 22 ≤ Fintype.card Coord)
    (prover : RandomOracle.OracleComp (Akita.Extract.Address Wire total) (Option Coord → Akita.Replay.CodeSlot E) (ℕ → Wire))
    {queries : ℕ}
    (bounded : prover.QueryBound queries)
    {budget : ℕ}
    (positive : 0 < budget) :
    (𝔼 table,
        𝔼 tape,
          if
              o.accepts (o.body.protocol.final prover table) ∧
                Checkpoint.extract o.body prover budget table tape = none then
            1
          else 0) ≤
      ((↑queries : ℝ) + 1) * (317479 / (1024 * (↑(Fintype.card E) : ℝ)) + 135 * (1 / (↑fieldModulus : ℝ) ^ 2)) +
        ((↑queries : ℝ) + 1) * (↑142 : ℝ) / (↑budget : ℝ) :=
  @NightstreamFPrime.Spec.Compression.Akita.Key20.Tower.extraction_le E fieldE fintypeE decidableEqE
    algebraZqE degree2EAlgebraZqE Wire Coord fintypeCoord decidableEqCoord nonemptyCoord total o
    finiteWire drawn slots prover queries bounded budget positive

/-- The Lean verifier of the opening of `crates/ns-compress/artifacts/key20.aks` accepts the proof
of the honest prover (Akita Theorem 10.9 and Akita Section 8.2) for each family of tables with the
fit conditions of the honest prover. `tables`, `places`: the tables of the key polynomial, `W` and
`V` and their opening points (`live`, `lengths`: their sizes). `fit`: the fit conditions
`Spec.Compression.Akita.Key20.Fit` (Akita Definition 10.5): for each opening point and each fold
challenge vector that the decoders give, the response of each level satisfies `Enc` and the norm
limit of its route. For each table of the oracle on keys whose slots decode to fold challenges and
whose work slots have 57 zero work bits (`worked`: 57 is the largest count of work bits of a site,
and Lean does not model the search of the honest prover for the work bits of Akita Remark 10.18),
a proof exists whose first message has the commitments of the tables, the points and the values,
and whose transcript the verifier `towerLinks` accepts. The proof is a sequence of messages of the
type `Key20.Wire E`, which is finite (`Spec.Compression.Akita.Key20.Note.finite`): the points are
vectors of 20, 16 and 16 coordinates, and `Key20.Note.place` reads them as lists. For the
decoders of the code of `Key20.Tower` and the readers of `Key20.Note`, the proof is a prover with no
queries of the game of `key20_opening_bound` (`key20_opening_played`). With the count
`Spec.Compression.Akita.Key20.Kept`, each slot decodes, so `decodes` to `decodesL` hold for each
table (`Spec.Compression.Akita.Key20.Tower.Drawn.decodes`).
The matrices are free: the claim is for each setup. `fit` holds in two cases: for the zero tables
with each decoder (`Spec.Compression.Akita.Key20.fit_zero`), and for each family of
tables with decoders that give the fold challenge 0 (`Spec.Compression.Akita.Key20.fit_quiet`).
For the challenge families of the schedule and tables that are not 0, `fit` is not a theorem: a
response can be larger than the limit of `Enc`, and the honest prover of the Akita code has a
search over a nonce for a response that fits. The claim is an existence result for each table,
not a completeness bound: Lean has no bound on the probability that a response fits, and the Lean
verifier has no rejection at a zero tensor factor. The claim for `key30.aks` is
`Claims.AkitaKey30.key30_opening_accepts`. -/
theorem key20_opening_accepts
    {E : Type}
    [fieldE : Field E]
    [fintypeE : Fintype E]
    [decidableEqE : DecidableEq E]
    [algebraZqE : Algebra Akita.Zq E]
    [degree2EAlgebraZqE : Akita.Degree2 E]
    {Coord : Type}
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    (β : (g : Fin 3) → Module.Basis (Fin (Akita.Key20.root.planes g)) Akita.Zq E)
    (key : (g : Fin 3) → Akita.Key (Akita.Key20.root.shape g))
    (openKey : Akita.Key20.root.OpenKey)
    (block : Akita.Key20.root.Block → Coord)
    (decode : (g : Fin 3) → Akita.Replay.CodeSlot E → Option (Akita.Rq (Akita.Key20.root.carrierDegree g)))
    (mass : Fin 3 → ℕ)
    (β₁ : Module.Basis (Fin Akita.Key20.level₁.opening.planes) Akita.Zq E)
    (key₁ : Akita.Key Akita.Key20.level₁.shape)
    (openKey₁ : Akita.OpeningKey Akita.Key20.level₁.opening)
    (block₁ : Fin Akita.Key20.level₁.shape.blocks → Coord)
    (decode₁ : Akita.Replay.CodeSlot E → Option (Akita.Rq Akita.Key20.level₁.opening.carrierDegree))
    (weight₁ : ℕ)
    (key₂ : Akita.Key Akita.Key20.level₂.shape)
    (openKey₂ : Akita.OpeningKey Akita.Key20.level₂.opening)
    (block₂ : Fin Akita.Key20.level₂.shape.blocks → Coord)
    (decode₂ : Akita.Replay.CodeSlot E → Option (Akita.Rq Akita.Key20.level₂.shape.innerDegree))
    (weight₂ : ℕ)
    (key₃ : Akita.Key Akita.Key20.level₃.shape)
    (openKey₃ : Akita.OpeningKey Akita.Key20.level₃.opening)
    (block₃ : Fin Akita.Key20.level₃.shape.blocks → Coord)
    (decode₃ : Akita.Replay.CodeSlot E → Option (Akita.Rq Akita.Key20.level₃.shape.innerDegree))
    (weight₃ : ℕ)
    (key₄ : Akita.Key Akita.Key20.level₄.shape)
    (openKey₄ : Akita.OpeningKey Akita.Key20.level₄.opening)
    (block₄ : Fin Akita.Key20.level₄.shape.blocks → Coord)
    (decode₄ : Akita.Replay.CodeSlot E → Option (Akita.Rq Akita.Key20.level₄.shape.innerDegree))
    (weight₄ : ℕ)
    (key₅ : Akita.Key Akita.Key20.level₅.shape)
    (openKey₅ : Akita.OpeningKey Akita.Key20.level₅.opening)
    (block₅ : Fin Akita.Key20.level₅.shape.blocks → Coord)
    (decode₅ : Akita.Replay.CodeSlot E → Option (Akita.Rq Akita.Key20.level₅.shape.innerDegree))
    (weight₅ : ℕ)
    (matrix : Matrix (Fin Akita.Key20.terminal.innerRank)
          (Fin Akita.Key20.terminal.blockLength × Fin Akita.Key20.terminal.sourceDigits)
          (Akita.Rq Akita.Key20.terminal.innerDegree))
    (blockL : Fin Akita.Key20.terminal.blocks → Coord)
    (decodeL : Akita.Replay.CodeSlot E → Option (Akita.Rq Akita.Key20.terminal.innerDegree))
    {total : ℕ}
    (inside : ((Akita.Extract.committedTraceSq key₅ openKey₅
                  (Akita.Key20.Note.claim₅ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.level₅.vars
                  Akita.Key20.sized₅.left Akita.Key20.level₅.batch Akita.Key20.sized₅.right.left
                  block₅ decode₅ Akita.Extract.Page.read weight₅
                  Akita.Key20.norm₅.operator Akita.Key20.norm₅.limit
                  ((Akita.Extract.committedTraceSq key₄ openKey₄
                          (Akita.Key20.Note.claim₄ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.level₄.vars
                          Akita.Key20.sized₄.left Akita.Key20.level₄.batch Akita.Key20.sized₄.right.left
                          block₄ decode₄ Akita.Extract.Page.read weight₄
                          Akita.Key20.norm₄.operator Akita.Key20.norm₄.limit
                          ((Akita.Extract.committedTraceSq key₃ openKey₃
                                  (Akita.Key20.Note.claim₃ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                  Akita.Key20.level₃.vars Akita.Key20.sized₃.left Akita.Key20.level₃.batch
                                  Akita.Key20.sized₃.right.left block₃ decode₃
                                  Akita.Extract.Page.read weight₃ Akita.Key20.norm₃.operator
                                  Akita.Key20.norm₃.limit
                                  ((Akita.Extract.committedTrace key₂ openKey₂
                                          (Akita.Key20.Note.claim₂ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                          Akita.Key20.level₂.vars Akita.Key20.sized₂.left
                                          Akita.Key20.level₂.batch Akita.Key20.sized₂.right.left block₂
                                          decode₂ Akita.Extract.Page.read weight₂
                                          ((Akita.Extract.liftFold β₁ key₁ openKey₁
                                                  (Akita.Key20.Note.claim₁ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                                  Akita.Key20.level₁.vars Akita.Key20.level₁_sized.left
                                                  Akita.Key20.quotientDigits
                                                  Akita.Key20.level₁_sized.right.left
                                                  Akita.Key20.level₁.batch
                                                  Akita.Key20.level₁_sized.right.right.left block₁ decode₁
                                                  Akita.Extract.Page.read weight₁
                                                  (Akita.Extract.liftBatch β key openKey
                                                      (Akita.Key20.Note.claim ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                                      Akita.Key20.rootVars Akita.Key20.root_sized.left 8
                                                      Akita.Key20.quotientDigits Akita.Key20.root_room
                                                      Akita.Key20.rootBatch
                                                      Akita.Key20.root_sized.right.left block decode
                                                      Akita.Extract.Page.read mass (0 + 1)).finish).tensored
                                              Akita.Key20.level₁.tail).finish).tensored
                                      Akita.Key20.level₂.tail).finish).tensored
                              Akita.Key20.level₃.tail).finish).tensored
                      Akita.Key20.level₄.tail).finish).tensored
              Akita.Key20.level₅.tail).finish <
          total)
    (accept : Akita.Extract.Prefix (Akita.Key20.Wire E) (Option Coord → Akita.Replay.CodeSlot E) total → Prop)
    (table : Akita.Extract.Address (Akita.Key20.Wire E) total → Option Coord → Akita.Replay.CodeSlot E)
    (decodes : ∀ (input : Akita.Extract.Address (Akita.Key20.Wire E) total)
          (b : Akita.Key20.root.Block), (decode b.fst (table input (some (block b)))).isSome = true)
    (decodes₁ : ∀ (input : Akita.Extract.Address (Akita.Key20.Wire E) total)
          (i : Fin Akita.Key20.level₁.shape.blocks), (decode₁ (table input (some (block₁ i)))).isSome = true)
    (decodes₂ : ∀ (input : Akita.Extract.Address (Akita.Key20.Wire E) total)
          (i : Fin Akita.Key20.level₂.shape.blocks), (decode₂ (table input (some (block₂ i)))).isSome = true)
    (decodes₃ : ∀ (input : Akita.Extract.Address (Akita.Key20.Wire E) total)
          (i : Fin Akita.Key20.level₃.shape.blocks), (decode₃ (table input (some (block₃ i)))).isSome = true)
    (decodes₄ : ∀ (input : Akita.Extract.Address (Akita.Key20.Wire E) total)
          (i : Fin Akita.Key20.level₄.shape.blocks), (decode₄ (table input (some (block₄ i)))).isSome = true)
    (decodes₅ : ∀ (input : Akita.Extract.Address (Akita.Key20.Wire E) total)
          (i : Fin Akita.Key20.level₅.shape.blocks), (decode₅ (table input (some (block₅ i)))).isSome = true)
    (decodesL : ∀ (input : Akita.Extract.Address (Akita.Key20.Wire E) total)
          (i : Fin Akita.Key20.terminal.blocks), (decodeL (table input (some (blockL i)))).isSome = true)
    (worked : ∀ (input : Akita.Extract.Address (Akita.Key20.Wire E) total),
          table input none ∈ Akita.Extract.workZeros (Akita.Replay.CodeSlot E) 57)
    {tables : Fin 3 → Akita.Table}
    (live : ∀ (g : Fin 3) (cell : ℕ), 2 ^ ![20, 16, 16] g ≤ cell → tables g cell = 0)
    {places : Fin 3 → List E}
    (lengths : ∀ (g : Fin 3), (places g).length = ![20, 16, 16] g)
    (fit : Akita.Key20.Fit β key openKey decode β₁ key₁ openKey₁ decode₁ key₂ openKey₂ decode₂ key₃
        openKey₃ decode₃ key₄ openKey₄ decode₄ key₅ openKey₅ decode₅ decodeL tables) :
    ∃ proof : ℕ → Akita.Key20.Wire E,
      ((proof 0).data.commit = fun g => (Akita.commit (key g) (tables g)).1) ∧
      (proof 0).data.place = places ∧
      ((proof 0).data.value = fun g => Lib.mle (places g) (Akita.lifted E (tables g))) ∧
      Akita.Extract.towerLinks (weight₃ := weight₃) (weight₄ := weight₄) (weight₅ := weight₅) Akita.Extract.Route.coefficient
        (Akita.Extract.Route.euclidean Akita.Key20.norm₃.operator Akita.Key20.norm₃.limit)
        (Akita.Extract.Route.euclidean Akita.Key20.norm₄.operator Akita.Key20.norm₄.limit)
        (Akita.Extract.Route.euclidean Akita.Key20.norm₅.operator Akita.Key20.norm₅.limit)
        (Akita.Extract.liftBatch β key openKey (Akita.Key20.Note.claim ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
          Akita.Key20.rootVars Akita.Key20.root_sized.left 8 Akita.Key20.quotientDigits
          Akita.Key20.root_room Akita.Key20.rootBatch Akita.Key20.root_sized.right.left block decode
          Akita.Extract.Page.read mass (0 + 1))
        ((Akita.Extract.liftFold β₁ key₁ openKey₁ (Akita.Key20.Note.claim₁ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
              Akita.Key20.level₁.vars Akita.Key20.level₁_sized.left Akita.Key20.quotientDigits
              Akita.Key20.level₁_sized.right.left Akita.Key20.level₁.batch
              Akita.Key20.level₁_sized.right.right.left block₁ decode₁ Akita.Extract.Page.read weight₁
              (Akita.Extract.liftBatch β key openKey (Akita.Key20.Note.claim ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                  Akita.Key20.rootVars Akita.Key20.root_sized.left 8 Akita.Key20.quotientDigits
                  Akita.Key20.root_room Akita.Key20.rootBatch Akita.Key20.root_sized.right.left block decode
                  Akita.Extract.Page.read mass (0 + 1)).finish).tensored
          Akita.Key20.level₁.tail)
        ((Akita.Extract.committedTrace key₂ openKey₂ (Akita.Key20.Note.claim₂ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
              Akita.Key20.level₂.vars Akita.Key20.sized₂.left Akita.Key20.level₂.batch
              Akita.Key20.sized₂.right.left block₂ decode₂ Akita.Extract.Page.read weight₂
              ((Akita.Extract.liftFold β₁ key₁ openKey₁ (Akita.Key20.Note.claim₁ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                      Akita.Key20.level₁.vars Akita.Key20.level₁_sized.left Akita.Key20.quotientDigits
                      Akita.Key20.level₁_sized.right.left Akita.Key20.level₁.batch
                      Akita.Key20.level₁_sized.right.right.left block₁ decode₁ Akita.Extract.Page.read
                      weight₁
                      (Akita.Extract.liftBatch β key openKey
                          (Akita.Key20.Note.claim ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.rootVars
                          Akita.Key20.root_sized.left 8 Akita.Key20.quotientDigits Akita.Key20.root_room
                          Akita.Key20.rootBatch Akita.Key20.root_sized.right.left block decode
                          Akita.Extract.Page.read mass (0 + 1)).finish).tensored
                  Akita.Key20.level₁.tail).finish).tensored
          Akita.Key20.level₂.tail)
        ((Akita.Extract.committedTraceSq key₃ openKey₃ (Akita.Key20.Note.claim₃ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
              Akita.Key20.level₃.vars Akita.Key20.sized₃.left Akita.Key20.level₃.batch
              Akita.Key20.sized₃.right.left block₃ decode₃ Akita.Extract.Page.read
              weight₃ Akita.Key20.norm₃.operator Akita.Key20.norm₃.limit
              ((Akita.Extract.committedTrace key₂ openKey₂
                      (Akita.Key20.Note.claim₂ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.level₂.vars
                      Akita.Key20.sized₂.left Akita.Key20.level₂.batch Akita.Key20.sized₂.right.left block₂
                      decode₂ Akita.Extract.Page.read weight₂
                      ((Akita.Extract.liftFold β₁ key₁ openKey₁
                              (Akita.Key20.Note.claim₁ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.level₁.vars
                              Akita.Key20.level₁_sized.left Akita.Key20.quotientDigits
                              Akita.Key20.level₁_sized.right.left Akita.Key20.level₁.batch
                              Akita.Key20.level₁_sized.right.right.left block₁ decode₁
                              Akita.Extract.Page.read weight₁
                              (Akita.Extract.liftBatch β key openKey
                                  (Akita.Key20.Note.claim ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.rootVars
                                  Akita.Key20.root_sized.left 8 Akita.Key20.quotientDigits
                                  Akita.Key20.root_room Akita.Key20.rootBatch
                                  Akita.Key20.root_sized.right.left block decode Akita.Extract.Page.read
                                  mass (0 + 1)).finish).tensored
                          Akita.Key20.level₁.tail).finish).tensored
                  Akita.Key20.level₂.tail).finish).tensored
          Akita.Key20.level₃.tail)
        ((Akita.Extract.committedTraceSq key₄ openKey₄ (Akita.Key20.Note.claim₄ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
              Akita.Key20.level₄.vars Akita.Key20.sized₄.left Akita.Key20.level₄.batch
              Akita.Key20.sized₄.right.left block₄ decode₄ Akita.Extract.Page.read
              weight₄ Akita.Key20.norm₄.operator Akita.Key20.norm₄.limit
              ((Akita.Extract.committedTraceSq key₃ openKey₃
                      (Akita.Key20.Note.claim₃ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.level₃.vars
                      Akita.Key20.sized₃.left Akita.Key20.level₃.batch Akita.Key20.sized₃.right.left
                      block₃ decode₃ Akita.Extract.Page.read weight₃
                      Akita.Key20.norm₃.operator Akita.Key20.norm₃.limit
                      ((Akita.Extract.committedTrace key₂ openKey₂
                              (Akita.Key20.Note.claim₂ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.level₂.vars
                              Akita.Key20.sized₂.left Akita.Key20.level₂.batch Akita.Key20.sized₂.right.left
                              block₂ decode₂ Akita.Extract.Page.read weight₂
                              ((Akita.Extract.liftFold β₁ key₁ openKey₁
                                      (Akita.Key20.Note.claim₁ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                      Akita.Key20.level₁.vars Akita.Key20.level₁_sized.left
                                      Akita.Key20.quotientDigits Akita.Key20.level₁_sized.right.left
                                      Akita.Key20.level₁.batch Akita.Key20.level₁_sized.right.right.left
                                      block₁ decode₁ Akita.Extract.Page.read weight₁
                                      (Akita.Extract.liftBatch β key openKey
                                          (Akita.Key20.Note.claim ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                          Akita.Key20.rootVars Akita.Key20.root_sized.left 8
                                          Akita.Key20.quotientDigits Akita.Key20.root_room
                                          Akita.Key20.rootBatch Akita.Key20.root_sized.right.left block
                                          decode Akita.Extract.Page.read mass (0 + 1)).finish).tensored
                                  Akita.Key20.level₁.tail).finish).tensored
                          Akita.Key20.level₂.tail).finish).tensored
                  Akita.Key20.level₃.tail).finish).tensored
          Akita.Key20.level₄.tail)
        ((Akita.Extract.committedTraceSq key₅ openKey₅ (Akita.Key20.Note.claim₅ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
              Akita.Key20.level₅.vars Akita.Key20.sized₅.left Akita.Key20.level₅.batch
              Akita.Key20.sized₅.right.left block₅ decode₅ Akita.Extract.Page.read
              weight₅ Akita.Key20.norm₅.operator Akita.Key20.norm₅.limit
              ((Akita.Extract.committedTraceSq key₄ openKey₄
                      (Akita.Key20.Note.claim₄ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.level₄.vars
                      Akita.Key20.sized₄.left Akita.Key20.level₄.batch Akita.Key20.sized₄.right.left
                      block₄ decode₄ Akita.Extract.Page.read weight₄
                      Akita.Key20.norm₄.operator Akita.Key20.norm₄.limit
                      ((Akita.Extract.committedTraceSq key₃ openKey₃
                              (Akita.Key20.Note.claim₃ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.level₃.vars
                              Akita.Key20.sized₃.left Akita.Key20.level₃.batch Akita.Key20.sized₃.right.left
                              block₃ decode₃ Akita.Extract.Page.read weight₃
                              Akita.Key20.norm₃.operator Akita.Key20.norm₃.limit
                              ((Akita.Extract.committedTrace key₂ openKey₂
                                      (Akita.Key20.Note.claim₂ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                      Akita.Key20.level₂.vars Akita.Key20.sized₂.left
                                      Akita.Key20.level₂.batch Akita.Key20.sized₂.right.left block₂ decode₂
                                      Akita.Extract.Page.read weight₂
                                      ((Akita.Extract.liftFold β₁ key₁ openKey₁
                                              (Akita.Key20.Note.claim₁ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                              Akita.Key20.level₁.vars Akita.Key20.level₁_sized.left
                                              Akita.Key20.quotientDigits Akita.Key20.level₁_sized.right.left
                                              Akita.Key20.level₁.batch
                                              Akita.Key20.level₁_sized.right.right.left block₁ decode₁
                                              Akita.Extract.Page.read weight₁
                                              (Akita.Extract.liftBatch β key openKey
                                                  (Akita.Key20.Note.claim ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                                  Akita.Key20.rootVars Akita.Key20.root_sized.left 8
                                                  Akita.Key20.quotientDigits Akita.Key20.root_room
                                                  Akita.Key20.rootBatch Akita.Key20.root_sized.right.left
                                                  block decode Akita.Extract.Page.read mass
                                                  (0 + 1)).finish).tensored
                                          Akita.Key20.level₁.tail).finish).tensored
                                  Akita.Key20.level₂.tail).finish).tensored
                          Akita.Key20.level₃.tail).finish).tensored
                  Akita.Key20.level₄.tail).finish).tensored
          Akita.Key20.level₅.tail)
        (Akita.Key20.terminalOf total matrix blockL decodeL
          ((Akita.Extract.committedTraceSq key₅ openKey₅ (Akita.Key20.Note.claim₅ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                  Akita.Key20.level₅.vars Akita.Key20.sized₅.left Akita.Key20.level₅.batch
                  Akita.Key20.sized₅.right.left block₅ decode₅
                  Akita.Extract.Page.read weight₅ Akita.Key20.norm₅.operator Akita.Key20.norm₅.limit
                  ((Akita.Extract.committedTraceSq key₄ openKey₄
                          (Akita.Key20.Note.claim₄ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.level₄.vars
                          Akita.Key20.sized₄.left Akita.Key20.level₄.batch Akita.Key20.sized₄.right.left
                          block₄ decode₄ Akita.Extract.Page.read weight₄
                          Akita.Key20.norm₄.operator Akita.Key20.norm₄.limit
                          ((Akita.Extract.committedTraceSq key₃ openKey₃
                                  (Akita.Key20.Note.claim₃ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                  Akita.Key20.level₃.vars Akita.Key20.sized₃.left Akita.Key20.level₃.batch
                                  Akita.Key20.sized₃.right.left block₃ decode₃
                                  Akita.Extract.Page.read weight₃ Akita.Key20.norm₃.operator
                                  Akita.Key20.norm₃.limit
                                  ((Akita.Extract.committedTrace key₂ openKey₂
                                          (Akita.Key20.Note.claim₂ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                          Akita.Key20.level₂.vars Akita.Key20.sized₂.left
                                          Akita.Key20.level₂.batch Akita.Key20.sized₂.right.left block₂
                                          decode₂ Akita.Extract.Page.read weight₂
                                          ((Akita.Extract.liftFold β₁ key₁ openKey₁
                                                  (Akita.Key20.Note.claim₁ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                                  Akita.Key20.level₁.vars Akita.Key20.level₁_sized.left
                                                  Akita.Key20.quotientDigits
                                                  Akita.Key20.level₁_sized.right.left
                                                  Akita.Key20.level₁.batch
                                                  Akita.Key20.level₁_sized.right.right.left block₁ decode₁
                                                  Akita.Extract.Page.read weight₁
                                                  (Akita.Extract.liftBatch β key openKey
                                                      (Akita.Key20.Note.claim ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
                                                      Akita.Key20.rootVars Akita.Key20.root_sized.left 8
                                                      Akita.Key20.quotientDigits Akita.Key20.root_room
                                                      Akita.Key20.rootBatch
                                                      Akita.Key20.root_sized.right.left block decode
                                                      Akita.Extract.Page.read mass (0 + 1)).finish).tensored
                                              Akita.Key20.level₁.tail).finish).tensored
                                      Akita.Key20.level₂.tail).finish).tensored
                              Akita.Key20.level₃.tail).finish).tensored
                      Akita.Key20.level₄.tail).finish).tensored
              Akita.Key20.level₅.tail).finish)
        0 Akita.Extract.Page.read (Akita.Key20.Note.commit ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
        (Akita.Key20.Note.value ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
        (Akita.Key20.Note.place ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _) Akita.Key20.quarters.left
        Akita.Key20.quarters.right.left Akita.Key20.quarters.right.right.left
        Akita.Key20.quarters.right.right.right.left Akita.Key20.quarters.right.right.right.right
        (Akita.Key20.Note.commit₁ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
        (Akita.Key20.Note.commit₂ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
        (Akita.Key20.Note.commit₃ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
        (Akita.Key20.Note.commit₄ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
        (Akita.Key20.Note.commit₅ ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
        (Akita.Key20.Note.state ∘ Akita.Extract.Page.data : Akita.Key20.Wire E → _)
        ((Akita.Extract.transcript total accept).final (RandomOracle.OracleComp.done proof) table) :=
  @NightstreamFPrime.Spec.Compression.Akita.Key20.accepts_code E fieldE fintypeE decidableEqE algebraZqE
    degree2EAlgebraZqE Coord fintypeCoord decidableEqCoord β key openKey block decode mass β₁ key₁ openKey₁
    block₁ decode₁ weight₁ key₂ openKey₂ block₂ decode₂ weight₂ key₃ openKey₃ block₃ decode₃ weight₃ key₄
    openKey₄ block₄ decode₄ weight₄ key₅ openKey₅ block₅ decode₅ weight₅ matrix blockL decodeL total inside
    accept table decodes decodes₁ decodes₂ decodes₃ decodes₄ decodes₅ decodesL worked tables live places lengths
    fit

/-- The proof of the honest prover of the opening of `crates/ns-compress/artifacts/key20.aks` is
a prover of the game of `key20_opening_bound` (Akita Theorem 10.9 and Theorem 10.17). The data
`o` of that claim is `Spec.Compression.Akita.Key20.noteTower`: the wire messages `Key20.Wire E`
with the readers of `Key20.Note`, the decoders of the code of `Key20.Tower`, and free bases, matrices
and coordinates of the slots. `drawn` is the hypothesis `drawn` of `key20_opening_bound` for this
data, and it fixes the stage `stage` of the terminal. Lean proves that the wire type is finite, so
`key20_opening_bound` holds for `o`, and that for each table of the oracle on keys whose work slots
have 57 zero work bits (`worked`) and each family of tables with the fit conditions `fit` for
these decoders, a proof exists whose first message has the commitments of the tables, the points
and the values, such that the prover `.done proof` has no queries and the verifier `o.accepts`
accepts its transcript. So the accept event of `key20_opening_bound` is not empty when `fit`
holds. For the zero tables `fit` holds with no hypothesis
(`Spec.Compression.Akita.Key20.noteTower_zero`), and `drawn` alone gives a table and a prover in
the accept event (`Spec.Compression.Akita.Key20.noteTower_played`). The proof is chosen for each
table: the claim is an existence result for each table, not a completeness bound. -/
theorem key20_opening_played
    {E : Type}
    [fieldE : Field E]
    [fintypeE : Fintype E]
    [decidableEqE : DecidableEq E]
    [algebraZqE : Algebra Akita.Zq E]
    [degree2EAlgebraZqE : Akita.Degree2 E]
    {Coord : Type}
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    (β : (g : Fin 3) → Module.Basis (Fin (Akita.Key20.root.planes g)) Akita.Zq E)
    (key : (g : Fin 3) → Akita.Key (Akita.Key20.root.shape g))
    (openKey : Akita.Key20.root.OpenKey)
    (block : Akita.Key20.root.Block → Coord)
    (β₁ : Module.Basis (Fin Akita.Key20.level₁.opening.planes) Akita.Zq E)
    (key₁ : Akita.Key Akita.Key20.level₁.shape)
    (openKey₁ : Akita.OpeningKey Akita.Key20.level₁.opening)
    (block₁ : Fin Akita.Key20.level₁.shape.blocks → Coord)
    (key₂ : Akita.Key Akita.Key20.level₂.shape)
    (openKey₂ : Akita.OpeningKey Akita.Key20.level₂.opening)
    (block₂ : Fin Akita.Key20.level₂.shape.blocks → Coord)
    (key₃ : Akita.Key Akita.Key20.level₃.shape)
    (openKey₃ : Akita.OpeningKey Akita.Key20.level₃.opening)
    (block₃ : Fin Akita.Key20.level₃.shape.blocks → Coord)
    (key₄ : Akita.Key Akita.Key20.level₄.shape)
    (openKey₄ : Akita.OpeningKey Akita.Key20.level₄.opening)
    (block₄ : Fin Akita.Key20.level₄.shape.blocks → Coord)
    (key₅ : Akita.Key Akita.Key20.level₅.shape)
    (openKey₅ : Akita.OpeningKey Akita.Key20.level₅.opening)
    (block₅ : Fin Akita.Key20.level₅.shape.blocks → Coord)
    (matrix : Matrix (Fin Akita.Key20.terminal.innerRank)
          (Fin Akita.Key20.terminal.blockLength × Fin Akita.Key20.terminal.sourceDigits)
          (Akita.Rq Akita.Key20.terminal.innerDegree))
    (blockL : Fin Akita.Key20.terminal.blocks → Coord)
    {total stage : ℕ}
    [nonemptyCoord : Nonempty Coord]
    (drawn : (Akita.Key20.noteTower β key openKey block β₁ key₁ openKey₁ block₁ key₂ openKey₂ block₂
          key₃ openKey₃ block₃ key₄ openKey₄ block₄ key₅ openKey₅ block₅ matrix blockL total stage).Drawn)
    (table : Akita.Extract.Address (Akita.Key20.Wire E) total → Option Coord → Akita.Replay.CodeSlot E)
    (worked : ∀ (input : Akita.Extract.Address (Akita.Key20.Wire E) total),
          table input none ∈ Akita.Extract.workZeros (Akita.Replay.CodeSlot E) 57)
    {tables : Fin 3 → Akita.Table}
    (live : ∀ (g : Fin 3) (cell : ℕ), 2 ^ ![20, 16, 16] g ≤ cell → tables g cell = 0)
    {places : Fin 3 → List E}
    (lengths : ∀ (g : Fin 3), (places g).length = ![20, 16, 16] g)
    (fit : Akita.Key20.Fit β key openKey (Akita.Key20.rootDecode E) β₁ key₁ openKey₁
        (Akita.Replay.CodeSlot.open Akita.Key20.open64 Akita.Key20.open64_fits)
        key₂ openKey₂
        (Akita.Replay.CodeSlot.open Akita.Key20.open128 Akita.Key20.open128_fits)
        key₃ openKey₃
        Akita.Replay.CodeSlot.kept128
        key₄ openKey₄
        Akita.Replay.CodeSlot.kept64
        key₅ openKey₅
        Akita.Replay.CodeSlot.kept64
        Akita.Replay.CodeSlot.kept64
        tables) :
    Finite (Akita.Key20.Wire E) ∧
      ∃ proof : ℕ → Akita.Key20.Wire E,
        (RandomOracle.OracleComp.done proof :
            RandomOracle.OracleComp (Akita.Extract.Address (Akita.Key20.Wire E) total) (Option Coord → Akita.Replay.CodeSlot E)
              (ℕ → Akita.Key20.Wire E)).QueryBound 0 ∧
        ((proof 0).data.commit = fun g => (Akita.commit (key g) (tables g)).1) ∧
        (proof 0).data.place = places ∧
        ((proof 0).data.value = fun g => Lib.mle (places g) (Akita.lifted E (tables g))) ∧
        (Akita.Key20.noteTower β key openKey block β₁ key₁ openKey₁ block₁ key₂ openKey₂ block₂ key₃
              openKey₃ block₃ key₄ openKey₄ block₄ key₅ openKey₅ block₅ matrix blockL total stage).accepts
          ((Akita.Key20.noteTower β key openKey block β₁ key₁ openKey₁ block₁ key₂ openKey₂ block₂ key₃
                openKey₃ block₃ key₄ openKey₄ block₄ key₅ openKey₅ block₅ matrix blockL total
                stage).body.protocol.final
            (RandomOracle.OracleComp.done proof) table) :=
  @NightstreamFPrime.Spec.Compression.Akita.Key20.noteTower_accepts E fieldE fintypeE decidableEqE algebraZqE
    degree2EAlgebraZqE Coord fintypeCoord decidableEqCoord β key openKey block β₁ key₁ openKey₁ block₁ key₂
    openKey₂ block₂ key₃ openKey₃ block₃ key₄ openKey₄ block₄ key₅ openKey₅ block₅ matrix blockL total stage
    nonemptyCoord drawn table worked tables live places lengths fit

/-- Soundness of the final verifier of `SPEC.md` Algorithm 10 with the Akita openings of
`crates/ns-compress/artifacts/key20.aks` (Akita Theorem 10.17 for the compression argument over
the opening). It does not cover `key30.aks`. The model is one uniform table of the oracle on keys
with independent slots, and the verifier is the Lean verifier `FinalKey20.finalAccepts` with the
fold decoders of the pinned Akita code. The byte replay theorem `Replay.verifyProof_accepts20`
covers the opening component. For each prover with at most `Q` queries, the probability that
the final verifier accepts and no proof with witnesses satisfies
`Verifies` for the statement is at most the Nightstream bound `FinalKey20.shapesBound` (Bounds, in
the file header) at the slot probability `1/q²`:
for each claim shape, `(Q + 1)·(runCost + 317479/(1024·q²) + 135/q²) + (Q + 1)·144/budget + breakChance`.
Lean gives no bound on `breakChance`, the probability of a solution of Module-SIS (`[MSIS]`).
`drawn`: the conditions `FinalKey20.Opening.Drawn` of each opening. -/
theorem key20_verifier_sound
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
          Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (step : AppState → AppWitness → AppState)
    (control : AppState → AppWitness → Fin 2)
    {Wire : Type}
    {Coord : Type}
    {total : ℕ}
    (read : Akita.Extract.Reader Wire)
    (head : Wire → FinalAkita.First logicalWidth publicFits)
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [nonemptyCoord : Nonempty Coord]
    {ι : Type}
    [fintypeι : Fintype ι]
    (kind : Wire → Option ι)
    (family : ι → FinalKey20.Opening logicalWidth publicFits Wire Coord total)
    (prover : RandomOracle.OracleComp (Akita.Extract.Address Wire total) (Option Coord → Akita.Replay.CodeSlot K) (ℕ → Wire))
    [finiteWire : Finite Wire]
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    (complete : StepExtend.Complete exports)
    (rows : ∀ (circuit : Fin 2),
          (setup.rows (↑circuit : ℕ)).matrices = (FinalRows.packageRows exports complete circuit).matrices)
    (blocks : setup.keyBlocks = 512)
    (coords : 30 ≤ Fintype.card Coord)
    (drawn : ∀ (index : ι),
          FinalKey20.Opening.Drawn setup complete.packageWidth step control read head (family index))
    {queries : ℕ}
    (bounded : prover.QueryBound queries)
    {budget : ℕ}
    (positive : 0 < budget) :
    (𝔼 table,
        if
            FinalKey20.finalAccepts setup complete.packageWidth step control read head kind family
                (FinalKey20.played prover table) ∧
              ¬∃ proof,
                  RunCompleteness.Verifies setup.ajtai setup.circuitKeys step complete.packageWidth
                    control (head (FinalKey20.played prover table).first).statement proof then
          1
        else 0) ≤
      FinalKey20.shapesBound setup complete.packageWidth step control read head family prover
        (1 / (↑fieldModulus : ℝ) ^ 2) queries budget :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey20.verifier_sound_drawn logicalWidth publicFits mode setup
    step control Wire Coord total read head fintypeCoord decidableEqCoord nonemptyCoord ι fintypeι kind
    family prover finiteWire relation cubeFits exports complete rows blocks coords
    drawn queries bounded budget positive

/-- The final verifier with the Akita openings of `crates/ns-compress/artifacts/key20.aks` against a
false history: `Claims.Final.history_sound` in the game with Akita openings (Akita Theorem
10.17). It does not cover `key30.aks`. The model and the verifier are those of
`key20_verifier_sound`. For each prover with at most `Q` queries, the probability that the
verifier accepts `(i, z0, zi)` and no `i` valid steps lead from `z0` to `zi` is at most
`FinalKey20.shapesBound` at the slot probability `1/q²` plus `IvcSecurity.walkBound` under
`[HN-A1]`. `depthBound`: the prover controls the iteration of its statement, and the claim is for
provers whose accepted statement has an iteration of at most `depth` for each table, as in
`Claims.Final.history_sound`. Lean gives no bound on the term `breakChance` of `shapesBound`
(`[MSIS]`). -/
theorem key20_history_sound
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
          Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (step : AppState → AppWitness → AppState)
    (control : AppState → AppWitness → Fin 2)
    {Wire : Type}
    {Coord : Type}
    {total : ℕ}
    (read : Akita.Extract.Reader Wire)
    (head : Wire → FinalAkita.First logicalWidth publicFits)
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [nonemptyCoord : Nonempty Coord]
    {ι : Type}
    [fintypeι : Fintype ι]
    (kind : Wire → Option ι)
    (family : ι → FinalKey20.Opening logicalWidth publicFits Wire Coord total)
    (prover : RandomOracle.OracleComp (Akita.Extract.Address Wire total) (Option Coord → Akita.Replay.CodeSlot K) (ℕ → Wire))
    [finiteWire : Finite Wire]
    (budget : ℕ)
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    {Admitted : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → Prop}
    {StageAdmitted : HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
            (IvcWalk.Answer logicalWidth publicFits) →
          Prop}
    {Efficient : (adversary :
            HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits)) →
          HyperNova.ReverseStages.NifsExtractor (IvcWalk.Answer logicalWidth publicFits) adversary → Prop}
    {error : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → ℝ}
    (assumption : HyperNova.ReverseStages.Assumption1 (IvcWalk.WalkSucceeds setup.ajtai)
          (IvcWalk.WalkExtracted setup.ajtai) Admitted Efficient error)
    (closed : HyperNova.ReverseStages.Closed (IvcWalk.attemptOf setup.ajtai setup.circuitKeys exports)
          (IvcWalk.back setup.ajtai setup.circuitKeys exports) Admitted StageAdmitted Efficient)
    (complete : StepExtend.Complete exports)
    (rows : ∀ (circuit : Fin 2),
          (setup.rows (↑circuit : ℕ)).matrices = (FinalRows.packageRows exports complete circuit).matrices)
    (keysShaped : ∀ (key : Fin 2), List.length (setup.circuitKeys key).digest = 4)
    (blocks : setup.keyBlocks = 512)
    (coords : 30 ≤ Fintype.card Coord)
    (drawn : ∀ (index : ι),
          FinalKey20.Opening.Drawn setup complete.packageWidth step control read head (family index))
    {queries : ℕ}
    (bounded : prover.QueryBound queries)
    (positive : 0 < budget)
    (admitted : StageAdmitted
          (HyperNova.ReverseStages.Stage.start
            (IvcSecurity.walkAdversary
              (FinalKey20.adversary setup complete.packageWidth step control read head kind family
                prover budget))))
    (depth : ℕ)
    (depthBound : ∀ (table : Akita.Extract.Address Wire total → Option Coord → Akita.Replay.CodeSlot K),
          FinalKey20.finalAccepts setup complete.packageWidth step control read head kind family
              (FinalKey20.played prover table) →
            (head (FinalKey20.played prover table).first).statement.iteration ≤ depth) :
    (((FinalKey20.adversary setup complete.packageWidth step control read head kind family prover
                  budget).tape.toOuterMeasure :
            Set
                (FinalKey20.adversary setup complete.packageWidth step control read head kind family
                    prover budget).Tape →
              ENNReal)
          {sample |
            FinalKey20.FalseHistory setup complete.packageWidth step control read head kind family
              prover exports.program sample.1}).toReal ≤
      FinalKey20.shapesBound setup complete.packageWidth step control read head family prover
          (1 / (↑fieldModulus : ℝ) ^ 2) queries budget +
        IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed
          (FinalKey20.adversary setup complete.packageWidth step control read head kind family prover
            budget)
          admitted depth :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey20.history_sound_drawn logicalWidth publicFits mode setup
    step control Wire Coord total read head fintypeCoord decidableEqCoord nonemptyCoord ι fintypeι kind
    family prover finiteWire budget relation cubeFits exports Admitted StageAdmitted Efficient error
    assumption closed complete rows keysShaped blocks coords drawn queries bounded
    positive admitted depth depthBound

/-- The knowledge form of `key20_history_sound`: `Claims.Final.history_knowledge` in the game with
the Akita openings of `crates/ns-compress/artifacts/key20.aks` (Akita Theorem 10.17). It does
not cover `key30.aks`. The probability that the verifier accepts and the witnesses of the
reverse extractor are not a history of the accepted statement is at most `FinalKey20.shapesBound`
at the slot probability `1/q²` plus `IvcSecurity.walkBound`. The model, the verifier,
`depthBound` and the open terms are those of `key20_history_sound`. -/
theorem key20_history_knowledge
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
          Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (step : AppState → AppWitness → AppState)
    (control : AppState → AppWitness → Fin 2)
    {Wire : Type}
    {Coord : Type}
    {total : ℕ}
    (read : Akita.Extract.Reader Wire)
    (head : Wire → FinalAkita.First logicalWidth publicFits)
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [nonemptyCoord : Nonempty Coord]
    {ι : Type}
    [fintypeι : Fintype ι]
    (kind : Wire → Option ι)
    (family : ι → FinalKey20.Opening logicalWidth publicFits Wire Coord total)
    (prover : RandomOracle.OracleComp (Akita.Extract.Address Wire total) (Option Coord → Akita.Replay.CodeSlot K) (ℕ → Wire))
    [finiteWire : Finite Wire]
    (budget : ℕ)
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    {Admitted : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → Prop}
    {StageAdmitted : HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
            (IvcWalk.Answer logicalWidth publicFits) →
          Prop}
    {Efficient : (adversary :
            HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits)) →
          HyperNova.ReverseStages.NifsExtractor (IvcWalk.Answer logicalWidth publicFits) adversary → Prop}
    {error : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) → ℝ}
    (assumption : HyperNova.ReverseStages.Assumption1 (IvcWalk.WalkSucceeds setup.ajtai)
          (IvcWalk.WalkExtracted setup.ajtai) Admitted Efficient error)
    (closed : HyperNova.ReverseStages.Closed (IvcWalk.attemptOf setup.ajtai setup.circuitKeys exports)
          (IvcWalk.back setup.ajtai setup.circuitKeys exports) Admitted StageAdmitted Efficient)
    (complete : StepExtend.Complete exports)
    (rows : ∀ (circuit : Fin 2),
          (setup.rows (↑circuit : ℕ)).matrices = (FinalRows.packageRows exports complete circuit).matrices)
    (keysShaped : ∀ (key : Fin 2), List.length (setup.circuitKeys key).digest = 4)
    (blocks : setup.keyBlocks = 512)
    (coords : 30 ≤ Fintype.card Coord)
    (drawn : ∀ (index : ι),
          FinalKey20.Opening.Drawn setup complete.packageWidth step control read head (family index))
    {queries : ℕ}
    (bounded : prover.QueryBound queries)
    (positive : 0 < budget)
    (admitted : StageAdmitted
          (HyperNova.ReverseStages.Stage.start
            (IvcSecurity.walkAdversary
              (FinalKey20.adversary setup complete.packageWidth step control read head kind family
                prover budget))))
    (depth : ℕ)
    (depthBound : ∀ (table : Akita.Extract.Address Wire total → Option Coord → Akita.Replay.CodeSlot K),
          FinalKey20.finalAccepts setup complete.packageWidth step control read head kind family
              (FinalKey20.played prover table) →
            (head (FinalKey20.played prover table).first).statement.iteration ≤ depth) :
    (((↑(HyperNova.ReverseStages.reverseStages assumption closed
                      (IvcSecurity.walkAdversary
                        (FinalKey20.adversary setup complete.packageWidth step control read head kind
                          family prover budget))
                      admitted depth) :
                  HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                    (IvcWalk.Answer logicalWidth publicFits)).tape.toOuterMeasure :
            Set
                (↑(HyperNova.ReverseStages.reverseStages assumption closed
                        (IvcSecurity.walkAdversary
                          (FinalKey20.adversary setup complete.packageWidth step control read head kind
                            family prover budget))
                        admitted depth) :
                    HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                      (IvcWalk.Answer logicalWidth publicFits)).Tape →
              ENNReal)
          {tape |
            FinalKey20.AdviceFailure setup step control read head kind family prover budget exports
              assumption closed complete admitted depth tape}).toReal ≤
      FinalKey20.shapesBound setup complete.packageWidth step control read head family prover
          (1 / (↑fieldModulus : ℝ) ^ 2) queries budget +
        IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed
          (FinalKey20.adversary setup complete.packageWidth step control read head kind family prover
            budget)
          admitted depth :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey20.history_knowledge_drawn logicalWidth publicFits mode
    setup step control Wire Coord total read head fintypeCoord decidableEqCoord nonemptyCoord ι
    fintypeι kind family prover finiteWire budget relation cubeFits exports Admitted StageAdmitted
    Efficient error assumption closed complete rows keysShaped blocks coords drawn
    queries bounded positive admitted depth depthBound

/-- The memory verifier with the Akita openings of `crates/ns-compress/artifacts/key20.aks`
against a false memory statement: `Claims.Final.memory_sound` in the game with Akita openings
(Akita Theorem 10.17). It does not cover `key30.aks`. The model and the verifier are those of
`key20_verifier_sound`. The prover names a memory statement with the law `claim` for each table
of the oracle. Over the walk's `S_max · N` stages, the probability that the verifier accepts
`(i, z0, zi)`, the checks of SPEC.md Algorithm 8 hold, and the decoded history does not attest
the memory statement is at most `FinalKey20.shapesBound` at the slot probability `1/q²` plus the
terms of `Claims.Nebula.memory_sound`: `IvcSecurity.walkBound` under `[HN-A1]`, the state-link
and chain collision masses, and the memory-round error. The checks of Algorithm 8 bound the
iteration by `S_max · N`, so no premise bounds it. Lean gives no bound on the term `breakChance`
of `shapesBound` (`[MSIS]`). -/
theorem key20_memory_sound
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
          Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (step : AppState → AppWitness → AppState)
    (control : AppState → AppWitness → Fin 2)
    {Wire : Type}
    {Coord : Type}
    {total : ℕ}
    (read : Akita.Extract.Reader Wire)
    (head : Wire → FinalAkita.First logicalWidth publicFits)
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [nonemptyCoord : Nonempty Coord]
    {ι : Type}
    [fintypeι : Fintype ι]
    (kind : Wire → Option ι)
    (family : ι → FinalKey20.Opening logicalWidth publicFits Wire Coord total)
    (prover : RandomOracle.OracleComp (Akita.Extract.Address Wire total)
          (Option Coord → Akita.Replay.CodeSlot K) (ℕ → Wire))
    [finiteWire : Finite Wire]
    (budget : ℕ)
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤
          2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    {Admitted : HyperNova.ReverseStages.NifsAdversary
            (IvcWalk.WalkAttempt logicalWidth publicFits) →
          Prop}
    {StageAdmitted : HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
            (IvcWalk.Answer logicalWidth publicFits) →
          Prop}
    {Efficient : (adversary :
            HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits)) →
          HyperNova.ReverseStages.NifsExtractor (IvcWalk.Answer logicalWidth publicFits)
              adversary →
            Prop}
    {error : HyperNova.ReverseStages.NifsAdversary (IvcWalk.WalkAttempt logicalWidth publicFits) →
          ℝ}
    (assumption : HyperNova.ReverseStages.Assumption1 (IvcWalk.WalkSucceeds setup.ajtai)
          (IvcWalk.WalkExtracted setup.ajtai) Admitted Efficient error)
    (closed : HyperNova.ReverseStages.Closed
          (IvcWalk.attemptOf setup.ajtai setup.circuitKeys exports)
          (IvcWalk.back setup.ajtai setup.circuitKeys exports) Admitted StageAdmitted Efficient)
    (complete : StepExtend.Complete exports)
    {p : Nebula.Plan}
    (app : Nebula.PortApp p.toGeometry)
    (claim : (Akita.Extract.Address Wire total → Option Coord → Akita.Replay.CodeSlot K) →
          PMF (Nebula.EndToEnd.MemoryStatement app.width))
    (rows : ∀ (circuit : Fin 2),
          (setup.rows (↑circuit : ℕ)).matrices =
            (FinalRows.packageRows exports complete circuit).matrices)
    (keysShaped : ∀ (key : Fin 2), List.length (setup.circuitKeys key).digest = 4)
    (blocks : setup.keyBlocks = 512)
    (coords : 30 ≤ Fintype.card Coord)
    (drawn : ∀ (index : ι),
          FinalKey20.Opening.Drawn setup complete.packageWidth step control read head
            (family index))
    {queries : ℕ}
    (bounded : prover.QueryBound queries)
    (positive : 0 < budget)
    (admitted : StageAdmitted
          (HyperNova.ReverseStages.Stage.start
            (IvcSecurity.walkAdversary
              (FinalMemory.adversary head kind
                (fun index =>
                  FinalKey20.Opening.accepts setup complete.packageWidth step control read
                    head (family index))
                (fun index =>
                  FinalKey20.Opening.proof setup complete.packageWidth step control read head
                    (family index) prover)
                prover budget app claim))))
    (valid : p.Valid)
    (programEq : exports.program = Nebula.MemoryApp.program p.toGeometry app)
    {memoryError : NebulaSecurity.ClaimedStage (logicalWidth := logicalWidth)
          (publicFits := publicFits) app → ℝ}
    (round : NebulaSecurity.MemoryRound app setup.ajtai setup.circuitKeys exports StageAdmitted
          memoryError) :
    (((↑(HyperNova.ReverseStages.reverseStages assumption closed
                      (IvcSecurity.walkAdversary
                        (FinalMemory.adversary head kind
                          (fun index =>
                            FinalKey20.Opening.accepts setup complete.packageWidth step
                              control read head (family index))
                          (fun index =>
                            FinalKey20.Opening.proof setup complete.packageWidth step
                              control read head (family index) prover)
                          prover budget app claim))
                      admitted (p.sMax * p.n)) :
                  HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                    (IvcWalk.Answer logicalWidth publicFits)).tape.toOuterMeasure :
            Set
                (↑(HyperNova.ReverseStages.reverseStages assumption closed
                        (IvcSecurity.walkAdversary
                          (FinalMemory.adversary head kind
                            (fun index =>
                              FinalKey20.Opening.accepts setup complete.packageWidth step
                                control read head (family index))
                            (fun index =>
                              FinalKey20.Opening.proof setup complete.packageWidth step
                                control read head (family index) prover)
                            prover budget app claim))
                        admitted (p.sMax * p.n)) :
                    HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                      (IvcWalk.Answer logicalWidth publicFits)).Tape →
              ENNReal)
          {tape |
            FinalMemory.FalseMemory setup step head kind
              (fun index =>
                FinalKey20.Opening.accepts setup complete.packageWidth step control read
                  head (family index))
              prover app exports
              (FinalMemory.tapeOf setup step head kind
                    (fun index =>
                      FinalKey20.Opening.accepts setup complete.packageWidth step control
                        read head (family index))
                    (fun index =>
                      FinalKey20.Opening.proof setup complete.packageWidth step control
                        read head (family index) prover)
                    prover budget app exports assumption closed claim admitted (p.sMax * p.n)
                    tape).1.1
              (FinalMemory.sampleOf setup step head kind
                (fun index =>
                  FinalKey20.Opening.accepts setup complete.packageWidth step control read
                    head (family index))
                (fun index =>
                  FinalKey20.Opening.proof setup complete.packageWidth step control read
                    head (family index) prover)
                prover budget app exports assumption closed claim admitted (p.sMax * p.n)
                tape)}).toReal ≤
      FinalKey20.shapesBound setup complete.packageWidth step control read head family
          prover (1 / (↑fieldModulus : ℝ) ^ 2) queries budget +
        (IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed
                (FinalMemory.adversary head kind
                  (fun index =>
                    FinalKey20.Opening.accepts setup complete.packageWidth step control
                      read head (family index))
                  (fun index =>
                    FinalKey20.Opening.proof setup complete.packageWidth step control read
                      head (family index) prover)
                  prover budget app claim)
                admitted (p.sMax * p.n) +
              (((NebulaSecurity.ClaimedStage.law app
                        (FinalMemory.claimed setup step head kind
                          (fun index =>
                            FinalKey20.Opening.accepts setup complete.packageWidth step
                              control read head (family index))
                          (fun index =>
                            FinalKey20.Opening.proof setup complete.packageWidth step
                              control read head (family index) prover)
                          prover budget app exports assumption closed claim admitted
                          (p.sMax * p.n))).toOuterMeasure :
                    Set (NebulaSecurity.Sample app.width logicalWidth publicFits) → ENNReal)
                  {sample |
                    Nebula.EndToEnd.Experiment.StateLinkCollision app
                      (NebulaSecurity.experiment app setup.ajtai setup.circuitKeys exports)
                      sample}).toReal +
            (((NebulaSecurity.ClaimedStage.law app
                      (FinalMemory.claimed setup step head kind
                        (fun index =>
                          FinalKey20.Opening.accepts setup complete.packageWidth step
                            control read head (family index))
                        (fun index =>
                          FinalKey20.Opening.proof setup complete.packageWidth step
                            control read head (family index) prover)
                        prover budget app exports assumption closed claim admitted
                        (p.sMax * p.n))).toOuterMeasure :
                  Set (NebulaSecurity.Sample app.width logicalWidth publicFits) → ENNReal)
                {sample |
                  Nebula.EndToEnd.Experiment.ChainCollision app
                    (NebulaSecurity.experiment app setup.ajtai setup.circuitKeys exports)
                    sample}).toReal +
          memoryError
            (FinalMemory.claimed setup step head kind
              (fun index =>
                FinalKey20.Opening.accepts setup complete.packageWidth step control read
                  head (family index))
              (fun index =>
                FinalKey20.Opening.proof setup complete.packageWidth step control read
                  head (family index) prover)
              prover budget app exports assumption closed claim admitted (p.sMax * p.n))) :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey20.memory_sound_drawn logicalWidth publicFits
    mode setup step control Wire Coord total read head fintypeCoord decidableEqCoord
    nonemptyCoord ι fintypeι kind family prover finiteWire budget relation cubeFits exports
    Admitted StageAdmitted Efficient error assumption closed complete p app claim rows
    keysShaped blocks coords drawn queries bounded positive admitted valid programEq memoryError
    round

/-- The numbers of the three opening claims of the compression argument under the key of 512
blocks, which is the key of `crates/ns-compress/artifacts/key20.aks`. For a setup whose claims
pass the layout of `SPEC.md` Algorithm 10, step 5 (`use`), the point of the key polynomial has
20 coordinates, the point of `W` has 16, and the point of `V` has 16. These are the table sizes
`2^20`, `2^16` and `2^16` of the three groups of the Lean schedule `Key20.root`. -/
theorem key20_points
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
          Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (shape : FinalVerifier.Claims logicalWidth publicFits)
    (use : FinalKey20.Use setup shape) :
    (setup.params shape).laneVars + (setup.params shape).keyBlockVars + (setup.params shape).rowVars = 20 ∧
      (setup.params shape).witnessVars = 16 ∧
        (setup.params shape).spanVars + (setup.verifier shape).zeros =
            16 ∧
          (setup.params shape).spanVars ≤ 16 :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey20.key20_points logicalWidth publicFits mode setup shape
    use

/-- The block counts of the Lean schedule of `crates/ns-compress/artifacts/key20.aks` agree with
the points of its links (Akita Section 4.5, (40) to (42)). The live blocks of the levels 1 to 5
and of the terminal are at most `2^{r_blk}`, and the segments of the point of each link have the
sum of the coordinates of the final claim of its sender. `tests/AkitaSchedules.lean` compares
the numbers of the Lean schedule with the file. -/
theorem key20_blocks :
    (Akita.Key20.level₁.shape.blocks ≤ 2 ^ 5 ∧ 4 + (6 + (7 + 5)) = Akita.Key20.rootVars) ∧
      (Akita.Key20.level₂.shape.blocks ≤ 2 ^ 5 ∧ 6 + (9 + 5) = Akita.Key20.level₁.tail) ∧
        (Akita.Key20.level₃.shape.blocks ≤ 2 ^ 4 ∧ 6 + (8 + 4) = Akita.Key20.level₂.tail) ∧
          (Akita.Key20.level₄.shape.blocks ≤ 2 ^ 3 ∧ 5 + (9 + 3) = Akita.Key20.level₃.tail) ∧
            (Akita.Key20.level₅.shape.blocks ≤ 2 ^ 3 ∧ 5 + (8 + 3) = Akita.Key20.level₄.tail) ∧
              Akita.Key20.terminal.blocks ≤ 2 ^ 3 ∧ 5 + (8 + 3) = Akita.Key20.level₅.tail :=
  @NightstreamFPrime.Spec.Compression.Akita.Key20.blocks_live

/-- The instances of Module-SIS of the Akita opening of `crates/ns-compress/artifacts/key20.aks`
(Akita Lemma 10.20 and Appendix G). `Key20.instances` has 13 instances `(d, n, m, p, η)`: the ring
degree, the rank, the width, the norm (`true` for `‖·‖_2`, with `η` a bound on the square) and
the bound that the file prices. `Key20.views` has the instance of each of the 43 matrices of the
Lean tower at the bound of its witness of `R_bind`. Each view is below an instance of the list:
one degree, one rank, one norm, a width that is not larger and a bound that is not larger. Each
instance has the width of a view below it, with the bound of this view in the norm `‖·‖_∞` and
with a bound below two times the bound of this view in the norm `‖·‖_2`. No instance is below
another instance. `tests/AkitaSchedules.lean` compares the list with the file. Lean does not
prove the hardness of an instance, and the list is not for `key30.aks`. -/
theorem key20_instance_list :
    Akita.Covered Akita.Key20.instances Akita.Key20.views ∧
      Akita.Key20.instances.map Akita.Setup.Instance.key =
          [(32, 1, 128, false, 1), (64, 1, 1024, false, 1), (64, 2, 77, false, 63),
            (64, 3, 616, false, 63), (64, 8, 256, true, 2 ^ 46), (64, 8, 512, true, 2 ^ 45),
            (128, 1, 1936, false, 7), (128, 1, 1092, false, 31), (128, 4, 512, false, 126852),
            (128, 4, 256, true, 2 ^ 44), (512, 2, 56, false, 53477172),
            (1024, 1, 224, false, 32505732), (1024, 1, 56, false, 53477172)] ∧
        Akita.Key20.views.length = 43 ∧
          (∀ I ∈ Akita.Key20.instances,
              ∃ v ∈ Akita.Key20.views,
                v.Below I ∧ v.cols = I.cols ∧ cond I.euclidean (I.bound < 2 * v.bound) (v.bound = I.bound)) ∧
            List.Pairwise (fun I J => ¬I.Below J ∧ ¬J.Below I) Akita.Key20.instances :=
  @NightstreamFPrime.Spec.Compression.Akita.Key20.instances_exact

/-- The list `Key20.instances` is complete for the term `breakChance` of the claims
`key20_verifier_sound`, `key20_history_sound` and `key20_history_knowledge` (Akita Lemma 10.20).
`solved`: `out` is an output of the extractor that is a solution of Module-SIS, which is the
event of `FinalAkita.breakChance`: a solution for the key commitment, or a witness of `R_bind`
for a matrix of the root, of a level or of the terminal. Then a matrix `M` of the opening has a
solution that gives a solution of an instance of the list (`Key20.Reduced`, `Reduces`). The
instance has the degree, the rank and the norm of `M`. Each matrix of the instance that has the
columns of `M` at some positions has a solution at the bound of the instance, which is the
vector for `M` with zeros in the other columns. So these 13 instances are the Module-SIS
instances that the `key20_` claims rest on. `mass`, `weight₁` and `weight₂`: the opening has the
norms `‖c‖_1` of the fold challenges that `FinalKey20.Opening.Drawn` gives, 31, 51 and 51 for the
groups of the root, 51 for the level 1 and 31 for the level 2. The radius of the reduction for the
key commitment is `η_A` of the key group, 32505732. Lean does not prove the hardness of an
instance, and it has no bound on `breakChance`: the matrices of the claims are fixed.
`key20_instances_shared` joins the list with the views of one setup vector, which is the form of
the sum of Akita (168). -/
theorem key20_instances
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
          Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    {setup : FinalVerifier.Setup mode logicalWidth publicFits}
    {packageWidths : Fin 2 → ℕ}
    {step : AppState → AppWitness → AppState}
    {control : AppState → AppWitness → Fin 2}
    {Wire : Type}
    {Coord : Type}
    {total : ℕ}
    {read : Akita.Extract.Reader Wire}
    {head : Wire → FinalAkita.First logicalWidth publicFits}
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    {o : FinalKey20.Opening logicalWidth publicFits Wire Coord total}
    (mass : o.mass = Akita.Key20.rootMass)
    (weight₁ : o.weight₁ = 51)
    (weight₂ : o.weight₂ = 31)
    {out : Akita.Extract.Opened Akita.Key20.root.Weak
            (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₁.opening)
              (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₂.opening)
                (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₃.opening)
                  (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₄.opening)
                    (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₅.opening)
                      (Akita.Extract.Prefix Wire (Option Coord → Akita.Replay.CodeSlot K) total ×
                        Option (Akita.TerminalFork Akita.Key20.terminal))))))) ⊕
          (Akita.Batch.LevelBreak o.key o.openKey fun g =>
              2 * (2 * o.mass g) * (Akita.Key20.root.fold g).sourceBound (Akita.Key20.root.shape g)) ⊕
            Akita.Extract.LevelBreak o.key₁ o.openKey₁
                (2 * (2 * o.weight₁) * Akita.Key20.level₁.fold.sourceBound Akita.Key20.level₁.shape) ⊕
              Akita.Extract.Route.Break o.key₂ o.openKey₂ o.weight₂
                  (Akita.Key20.level₂.fold.sourceBound Akita.Key20.level₂.shape)
                  Akita.Extract.Route.coefficient ⊕
                Akita.Extract.Route.Break o.key₃ o.openKey₃ o.weight₃
                    (Akita.Key20.level₃.fold.sourceBound Akita.Key20.level₃.shape)
                    (Akita.Extract.Route.euclidean Akita.Key20.norm₃.operator Akita.Key20.norm₃.limit) ⊕
                  Akita.Extract.Route.Break o.key₄ o.openKey₄ o.weight₄
                      (Akita.Key20.level₄.fold.sourceBound Akita.Key20.level₄.shape)
                      (Akita.Extract.Route.euclidean Akita.Key20.norm₄.operator Akita.Key20.norm₄.limit) ⊕
                    Akita.Extract.Route.Break o.key₅ o.openKey₅ o.weight₅
                        (Akita.Key20.level₅.fold.sourceBound Akita.Key20.level₅.shape)
                        (Akita.Extract.Route.euclidean Akita.Key20.norm₅.operator Akita.Key20.norm₅.limit) ⊕
                      Option
                        ((Akita.Extract.Route.euclidean Akita.Key20.normTerminal.operator
                              Akita.Key20.normTerminal.limit).Kernel
                          o.terminal.matrix o.terminal.weight o.terminal.bound)}
    (solved : FinalAkita.Solved (FinalKey20.Opening.tower setup packageWidths step control read head o)
          o.key o.honest ((Akita.Key20.root.fold 0).innerBound (Akita.Key20.root.shape 0)) out) :
    Akita.Key20.Reduced Akita.Key20.instances o.key o.openKey o.key₁ o.openKey₁ o.key₂ o.openKey₂ o.key₃
      o.openKey₃ o.key₄ o.openKey₄ o.key₅ o.openKey₅ o.terminal.matrix :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey20.Opening.reduced logicalWidth publicFits mode setup
    packageWidths step control Wire Coord total read head fintypeCoord decidableEqCoord o mass weight₁
    weight₂ out solved


/-- The join of the list `Key20.views` with the shared setup (Akita Section 4.3 with (35),
Table 3 and Lemma 10.20). `keys`: each matrix of the opening is the view of one flat setup vector
`S` with the degree, the rank and the width of its instance in `views`, as the code reads it from
one stream from the offset 0. `solved`, `mass`, `weight₁` and `weight₂` are those of
`key20_instances`. Then an instance `I` of the 43 views has a solution of Module-SIS for the view
`M_I(S)` of `S` itself, at the bound of the witness of its level (`Setup.Solved`). This is the
event of the algorithm `B` of Akita Lemma 10.20 for the inventory `views`
(`Spec.Compression.Akita.Key20.shared_setup`, with `2^18` cells). Lean does not prove that an
instance is hard (`[MSIS]`). -/
theorem key20_instances_shared
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
          Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    {setup : FinalVerifier.Setup mode logicalWidth publicFits}
    {packageWidths : Fin 2 → ℕ}
    {step : AppState → AppWitness → AppState}
    {control : AppState → AppWitness → Fin 2}
    {Wire : Type}
    {Coord : Type}
    {total : ℕ}
    {read : Akita.Extract.Reader Wire}
    {head : Wire → FinalAkita.First logicalWidth publicFits}
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    {o : FinalKey20.Opening logicalWidth publicFits Wire Coord total}
    (mass : o.mass = Akita.Key20.rootMass)
    (weight₁ : o.weight₁ = 51)
    (weight₂ : o.weight₂ = 31)
    {N : ℕ}
    {S : Fin N → Akita.Zq}
    (keys : Akita.Key20.Shared S o.key o.openKey o.key₁ o.openKey₁ o.key₂ o.openKey₂ o.key₃ o.openKey₃
          o.key₄ o.openKey₄ o.key₅ o.openKey₅ o.terminal.matrix)
    {out : Akita.Extract.Opened Akita.Key20.root.Weak
            (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₁.opening)
              (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₂.opening)
                (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₃.opening)
                  (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₄.opening)
                    (Akita.Extract.Opened (Akita.Weak Akita.Key20.level₅.opening)
                      (Akita.Extract.Prefix Wire (Option Coord → Akita.Replay.CodeSlot K) total ×
                        Option (Akita.TerminalFork Akita.Key20.terminal))))))) ⊕
          (Akita.Batch.LevelBreak o.key o.openKey fun g =>
              2 * (2 * o.mass g) * (Akita.Key20.root.fold g).sourceBound (Akita.Key20.root.shape g)) ⊕
            Akita.Extract.LevelBreak o.key₁ o.openKey₁
                (2 * (2 * o.weight₁) * Akita.Key20.level₁.fold.sourceBound Akita.Key20.level₁.shape) ⊕
              Akita.Extract.Route.Break o.key₂ o.openKey₂ o.weight₂
                  (Akita.Key20.level₂.fold.sourceBound Akita.Key20.level₂.shape)
                  Akita.Extract.Route.coefficient ⊕
                Akita.Extract.Route.Break o.key₃ o.openKey₃ o.weight₃
                    (Akita.Key20.level₃.fold.sourceBound Akita.Key20.level₃.shape)
                    (Akita.Extract.Route.euclidean Akita.Key20.norm₃.operator Akita.Key20.norm₃.limit) ⊕
                  Akita.Extract.Route.Break o.key₄ o.openKey₄ o.weight₄
                      (Akita.Key20.level₄.fold.sourceBound Akita.Key20.level₄.shape)
                      (Akita.Extract.Route.euclidean Akita.Key20.norm₄.operator Akita.Key20.norm₄.limit) ⊕
                    Akita.Extract.Route.Break o.key₅ o.openKey₅ o.weight₅
                        (Akita.Key20.level₅.fold.sourceBound Akita.Key20.level₅.shape)
                        (Akita.Extract.Route.euclidean Akita.Key20.norm₅.operator Akita.Key20.norm₅.limit) ⊕
                      Option
                        ((Akita.Extract.Route.euclidean Akita.Key20.normTerminal.operator
                              Akita.Key20.normTerminal.limit).Kernel
                          o.terminal.matrix o.terminal.weight o.terminal.bound)}
    (solved : FinalAkita.Solved (FinalKey20.Opening.tower setup packageWidths step control read head o)
          o.key o.honest ((Akita.Key20.root.fold 0).innerBound (Akita.Key20.root.shape 0)) out) :
    Akita.Setup.Solved Akita.Key20.views S :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey20.Opening.reduced_shared logicalWidth publicFits mode
    setup packageWidths step control Wire Coord total read head fintypeCoord decidableEqCoord o mass
    weight₁ weight₂ N S keys out solved

/-- The response search of the root fold of `crates/ns-compress/artifacts/key20.aks` (Akita
Lemma 10.7 with (157), Definition 10.5, Lemma K.3 and (313); `FOLD_RESPONSE_ATTEMPTS = 2^12` of
the pinned Akita code). The table of the oracle on keys is uniform, and `input n` is the key with
the nonce `n` after one transcript prefix. A nonce fails when a fold challenge of a group does not
decode, or the response of the key polynomial, of `W` or of `V` does not satisfy `Enc`. The
verifier needs each decode (`Spec.Compression.Akita.Extract.DecodesAt`), so a decoder that gives
no challenge makes each nonce fail. For source blocks with digits of the base `2^10`, each nonce
below `2^12` fails with probability at most `(ε + 2^−12)^4096`. `2^−12` is the Hoeffding bound of
Akita (303) for each group and the union bound over the groups. `sampler`: a fold challenge of
one answer does not decode with probability at most `ε`, the sampler term of Akita (157).
`fresh`: the keys of the nonces are different. `apart`: the fold coordinates have different
slots. `fair`: the sign flips of a fold challenge act on the slots of each decoder
(`Spec.Compression.Akita.SignFair`), which `Spec.Compression.Akita.signFair_signed` gives for one
decoder, and `Spec.Compression.Akita.Key20.rootDecode_signFair` gives for the decoders of the
code of the soundness claims. `energy`:
`‖c‖_2²` is at most 31 and 71, the values of the families `(31, 0)` and `(31, 10)`
(`Spec.Compression.Akita.normSq_of_mem`). `digits`: `‖s_i‖_∞ ≤ 512`. The claim is for one prefix.
It is not a bound for a complete proof: the event has no work bits of `FoldChallengeWork`, and
the Lean model has no nonce for the proof of work. -/
theorem key20_root_search
    {Slot Coord Point : Type}
    [fintypeSlot : Fintype Slot]
    [nonemptySlot : Nonempty Slot]
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [fintypePoint : Fintype Point]
    [decidableEqPoint : DecidableEq Point]
    (input : Fin Akita.Extract.responseTries → Point)
    (fresh : Function.Injective input)
    (block : Akita.Key20.root.Block → Coord)
    (apart : Function.Injective block)
    (decode : (g : Fin 3) → Fin (Akita.Key20.root.shape g).blocks → Slot →
      Option (Akita.Rq (Akita.Key20.root.carrierDegree g)))
    (fair : ∀ g i, Akita.SignFair (decode g i))
    (energy : ∀ g i slot c, decode g i slot = some c → Akita.normSq c ≤ ![31, 71, 71] g)
    (source : (g : Fin 3) → Fin (Akita.Key20.root.shape g).blocks →
      (Akita.Key20.root.shape g).Source)
    (digits : ∀ g i col, Akita.normInf (source g i col) ≤ 512)
    {ε : ℝ}
    (sampler : RandomOracle.mass {look : Coord → Slot |
      ∃ g i, decode g i (look (block ⟨g, i⟩)) = none} ≤ ε) :
    RandomOracle.mass {table : Point → Option Coord → Slot | ∀ n,
      (∃ g i, decode g i (Akita.Extract.view (table (input n)) (block ⟨g, i⟩)) = none) ∨
        ∃ g, ¬ Akita.Encodable (Akita.Key20.root.fold g).responseBase
          (Akita.Key20.root.fold g).responseDigits
          (Akita.response (Akita.Key20.root.opening g)
            (fun i => (decode g i (Akita.Extract.view (table (input n)) (block ⟨g, i⟩))).getD 0)
            (source g))} ≤
      (ε + 1 / 2 ^ 12) ^ Akita.Extract.responseTries :=
  @NightstreamFPrime.Spec.Compression.Akita.Key20.root_search Slot Coord fintypeSlot nonemptySlot
    fintypeCoord decidableEqCoord Point fintypePoint decidableEqPoint input fresh block apart
    decode fair energy source digits ε sampler

/-- `key20_root_search` for decoders that decode each slot (`total`). The sampler term is 0, and
each nonce below `2^12` fails with probability at most `(2^−12)^4096`. The decoders
`Spec.Compression.Akita.Key20.rootDecode` of the soundness claims decode each slot
(`Spec.Compression.Akita.Key20.rootDecode_isSome`: the root has no filter), and they have `fair`
(`Spec.Compression.Akita.Key20.rootDecode_signFair`). The decoder of
`Spec.Compression.Akita.signFair_signed` has `total` and `fair`. -/
theorem key20_root_search_total
    {Slot Coord Point : Type}
    [fintypeSlot : Fintype Slot]
    [nonemptySlot : Nonempty Slot]
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [fintypePoint : Fintype Point]
    [decidableEqPoint : DecidableEq Point]
    (input : Fin Akita.Extract.responseTries → Point)
    (fresh : Function.Injective input)
    (block : Akita.Key20.root.Block → Coord)
    (apart : Function.Injective block)
    (decode : (g : Fin 3) → Fin (Akita.Key20.root.shape g).blocks → Slot →
      Option (Akita.Rq (Akita.Key20.root.carrierDegree g)))
    (total : ∀ g i slot, (decode g i slot).isSome)
    (fair : ∀ g i, Akita.SignFair (decode g i))
    (energy : ∀ g i slot c, decode g i slot = some c → Akita.normSq c ≤ ![31, 71, 71] g)
    (source : (g : Fin 3) → Fin (Akita.Key20.root.shape g).blocks →
      (Akita.Key20.root.shape g).Source)
    (digits : ∀ g i col, Akita.normInf (source g i col) ≤ 512) :
    RandomOracle.mass {table : Point → Option Coord → Slot | ∀ n,
      (∃ g i, decode g i (Akita.Extract.view (table (input n)) (block ⟨g, i⟩)) = none) ∨
        ∃ g, ¬ Akita.Encodable (Akita.Key20.root.fold g).responseBase
          (Akita.Key20.root.fold g).responseDigits
          (Akita.response (Akita.Key20.root.opening g)
            (fun i => (decode g i (Akita.Extract.view (table (input n)) (block ⟨g, i⟩))).getD 0)
            (source g))} ≤
      (1 / 2 ^ 12) ^ Akita.Extract.responseTries :=
  @NightstreamFPrime.Spec.Compression.Akita.Key20.root_search_total Slot Coord fintypeSlot nonemptySlot
    fintypeCoord decidableEqCoord Point fintypePoint decidableEqPoint input fresh block apart
    decode total fair energy source digits

/-- The response search of level 1 of `crates/ns-compress/artifacts/key20.aks` (Akita Lemma 10.7
with (157), Definition 10.5, Lemma K.3 and (313)). A nonce fails when a fold challenge does not
decode or the response does not satisfy `Enc`. For 28 source blocks with digits of the base 8 and
fold challenges with `‖c‖_2² ≤ 71`, each nonce below `2^12` fails with probability at most
`(ε + 2^−49)^4096`. The hypotheses are those of `key20_root_search`, with `‖s_i‖_∞ ≤ 4` in
`digits`. -/
theorem key20_level1_search
    {Slot Coord Point : Type}
    [fintypeSlot : Fintype Slot]
    [nonemptySlot : Nonempty Slot]
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [fintypePoint : Fintype Point]
    [decidableEqPoint : DecidableEq Point]
    (input : Fin Akita.Extract.responseTries → Point)
    (fresh : Function.Injective input)
    (block : Fin Akita.Key20.level₁.shape.blocks → Coord)
    (apart : Function.Injective block)
    (decode : Fin Akita.Key20.level₁.shape.blocks → Slot →
      Option (Akita.Rq Akita.Key20.level₁.opening.carrierDegree))
    (fair : ∀ i, Akita.SignFair (decode i))
    (energy : ∀ i slot c, decode i slot = some c → Akita.normSq c ≤ 71)
    (source : Fin Akita.Key20.level₁.shape.blocks → Akita.Key20.level₁.shape.Source)
    (digits : ∀ i col, Akita.normInf (source i col) ≤ 4)
    {ε : ℝ}
    (sampler : RandomOracle.mass {look : Coord → Slot | ∃ i, decode i (look (block i)) = none} ≤
      ε) :
    RandomOracle.mass {table : Point → Option Coord → Slot | ∀ n,
      (∃ i, decode i (Akita.Extract.view (table (input n)) (block i)) = none) ∨
        ¬ Akita.Encodable Akita.Key20.level₁.fold.responseBase
          Akita.Key20.level₁.fold.responseDigits
          (Akita.response Akita.Key20.level₁.opening
            (fun i => (decode i (Akita.Extract.view (table (input n)) (block i))).getD 0)
            source)} ≤
      (ε + 1 / 2 ^ 49) ^ Akita.Extract.responseTries :=
  @NightstreamFPrime.Spec.Compression.Akita.Key20.level₁_search Slot Coord fintypeSlot nonemptySlot
    fintypeCoord decidableEqCoord Point fintypePoint decidableEqPoint input fresh block apart
    decode fair energy source digits ε sampler

/-- `key20_level1_search` for decoders that decode each slot (`total`). The sampler term is 0, and
each nonce below `2^12` fails with probability at most `(2^−49)^4096`. Level 1 has no filter, so
the decoder of the code of the family `(31, 10)` decodes each slot, and the sign flips act on its
slots (`Spec.Compression.Akita.Replay.open_isSome`, `Spec.Compression.Akita.Replay.signFair_open`). -/
theorem key20_level1_search_total
    {Slot Coord Point : Type}
    [fintypeSlot : Fintype Slot]
    [nonemptySlot : Nonempty Slot]
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [fintypePoint : Fintype Point]
    [decidableEqPoint : DecidableEq Point]
    (input : Fin Akita.Extract.responseTries → Point)
    (fresh : Function.Injective input)
    (block : Fin Akita.Key20.level₁.shape.blocks → Coord)
    (apart : Function.Injective block)
    (decode : Fin Akita.Key20.level₁.shape.blocks → Slot →
      Option (Akita.Rq Akita.Key20.level₁.opening.carrierDegree))
    (total : ∀ i slot, (decode i slot).isSome)
    (fair : ∀ i, Akita.SignFair (decode i))
    (energy : ∀ i slot c, decode i slot = some c → Akita.normSq c ≤ 71)
    (source : Fin Akita.Key20.level₁.shape.blocks → Akita.Key20.level₁.shape.Source)
    (digits : ∀ i col, Akita.normInf (source i col) ≤ 4) :
    RandomOracle.mass {table : Point → Option Coord → Slot | ∀ n,
      (∃ i, decode i (Akita.Extract.view (table (input n)) (block i)) = none) ∨
        ¬ Akita.Encodable Akita.Key20.level₁.fold.responseBase
          Akita.Key20.level₁.fold.responseDigits
          (Akita.response Akita.Key20.level₁.opening
            (fun i => (decode i (Akita.Extract.view (table (input n)) (block i))).getD 0)
            source)} ≤
      (1 / 2 ^ 49) ^ Akita.Extract.responseTries :=
  @NightstreamFPrime.Spec.Compression.Akita.Key20.level₁_search_total Slot Coord fintypeSlot nonemptySlot
    fintypeCoord decidableEqCoord Point fintypePoint decidableEqPoint input fresh block apart
    decode total fair energy source digits

/-- The count `Spec.Compression.Akita.Key20.Kept` from the count
`Spec.Compression.Akita.Key20.Tail` at one frequency: the union bound over the `d/2` frequencies
and the equal-marginal step of Akita Theorem H.2 (ePrint 2026/1983, revision of 2026-11-23), for
the integer predicate of the code. `Tail` states `(d/2)·#{c ∈ C_0 | U_0(c) > T²·Δ²} + 2^128 ≤
|C_0|` for the family `(31, 11)` of the degree 64 with `T = 18` and the family `(31, 0)` of the
degree 128 with `T = 13`. `U_0` is the test (288) at the frequency 0 with the tables of the code
at the scale `2^48` and the error bound `ε + 1`, not `ε`: the tables of the code have
`|C[t + d] + C[t]| ≤ 1` and `|S[t + d] + S[t]| ≤ 1` for `t < d`, and not 0
(`Spec.Compression.Akita.TrigTable.code_antisymmetric`,
`Spec.Compression.Akita.TrigTable.code_cos_one`), so `c ↦ c(X^{2k+1})` moves the test at the
frequency `k` to the test at the frequency 0 only with the radius `(ε + 1)·‖c‖_1`
(`Spec.Compression.Akita.TrigTable.card_reject_le`). Lean does not prove `Tail`: it needs the
exact moments of Akita (292), the tail polynomial of (293) and the left inclusion of Akita
Theorem H.1 at the frequency 0. The certificate of Akita Table 12 bounds the tail fraction by
`q⋆_1 = 0.023909042053` for the degree 64 and `0.008120720198` for the degree 128, and `Tail`
allows the fractions `0.024219975628` and `0.010546520952`
(`Spec.Compression.Akita.Key20.tail_iff`). -/
theorem key20_kept (tail : Akita.Key20.Tail) : Akita.Key20.Kept :=
  @NightstreamFPrime.Spec.Compression.Akita.Key20.kept_of_tail tail

end NightstreamFPrime.Claims.AkitaKey20
