import NightstreamFPrime.Export.BoundedDepth.FinalKey30All
import NightstreamFPrime.Spec.Compression.Akita.Key30Accept

/-! # Claims: the Akita openings of the schedule `key30.aks`

The results for the Akita openings of the compressed proof under the schedule file
`crates/ns-compress/artifacts/key30.aks`, the file of the key of `2^19` blocks (Akita, ePrint
2026/1983, revision of 2026-11-23, Section 9; `SPEC.md` Algorithm 10 and section 9;
`SECURITY_MODEL.md` adversary 6 and trust item 7). The claims for `key20.aks` and the claims for
each schedule are in `Claims.Akita`.

- Model: the model of `Claims.Akita`. The answers of the stages are one uniform table on the keys
  of the stage inputs, with independent slots. The verifier is the Lean verifier of the opening
  with an offloaded root. It is not the verifier of the pinned Akita code.
- Bounds: Nightstream results, not Akita Theorem 10.46. One query returns all the slots of a key,
  and the paper counts a query for each indexed coordinate stream (Akita Section 10.3). The file
  has no `(λ_FS, Q_max)`. Each bound has the term `breakChance`, or it bounds only the event that
  the extractor returns no output, and a solution of Module-SIS is an output.
- Assumptions: `[MSIS]` and `[KECCAK]`. `key30_instances` and `key30_instance_list` give the
  instances of Module-SIS that `breakChance` is about, and Lean does not prove their hardness.
  `key30_instances_shared` gives a solution for the view of one setup vector of an instance of
  the 54 views of the row, when the matrices are these views.
- Setup: the hypothesis `shared` states that the matrices `A`, `B` and `D` of the root are the
  views of the setup prefix (Akita (35)). From it Lean proves the split of the public weight of
  the root into the setup part (Akita (132)) in the form of the pinned Akita code, with the
  factor `powerScale` (`Batch.codeScan`).
- The filters of the challenge families are the predicate of the pinned Akita code
  (`Spec.Compression.Akita.TrigTable.codeAccepts`), and the operator bound `Γ` is a theorem for
  it (`Spec.Compression.Akita.TrigTable.filtered_code`).
- Open: `drawn.kept`, `drawn.registered` (the commitment of the setup prefix in the verifier setup
  is the output of the Lean commitment algorithm, which the offloaded root adds), and the terminal
  decoder and the key commitment of `drawn` (`SECURITY_MODEL.md`, Premises). `key30_opening_accepts`
  states that the Lean verifier of the opening accepts a proof, and `key30_opening_played` that
  this proof is a prover of the game of `key30_opening_bound`, with no hypothesis on the fit for
  the zero tables and the zero setup prefix only. For another setup prefix and other tables, the
  fit conditions of the honest prover are hypotheses (`Spec.Compression.Akita.Key30.tower_complete`).
  No Lean result compares the Lean decoders or the Lean verifier with the code. -/

namespace NightstreamFPrime.Claims.AkitaKey30

open Spec Spec.Compression Lifecycle Lifecycle.BoundedDepth Export.BoundedDepth
open scoped BigOperators Spec.FieldExtensionRing Spec.Compression.Production
attribute [local instance low] Classical.propDecidable
attribute [local instance] NightstreamFPrime.Export.BoundedDepth.FinalKey20.instAlgebraZqK

/-- The output of the extractor for the opening of a row of
`crates/ns-compress/artifacts/key30.aks`, the file of the key of `2^19` blocks (Akita Proposition
10.44 (i) and Lemma 9.1 for this schedule). The verifier is the Lean verifier
`Key30.Tower.accepts` of the opening with an offloaded root, with the decoders
`Akita.Extract.draw` of the sparse families. It is not the verifier of the pinned Akita code.
Each output of the extractor is a certified decommitment of the commitment of each group of the
root to a table that has the claimed value at the claimed point, or a solution of Module-SIS for
a matrix of a group, of a level or for `A_term` (`[MSIS]`). `shared`: the matrices `A`, `B` and
`D` of the root are the views of the setup prefix (Akita (35)), so Lean proves the split (132) of
the public weight of the root. `drawn.registered`: the commitment of the setup prefix in the
verifier setup is the output of the Lean commitment algorithm for the table of the setup prefix
(Akita Lemma 9.1, "correct preprocessing"). -/
theorem key30_opening_sound
    {E : Type}
    [fieldE : Field E]
    [fintypeE : Fintype E]
    [decidableEqE : DecidableEq E]
    [algebraZqE : Algebra Akita.Zq E]
    [degree2E : Akita.Degree2 E]
    {Wire : Type}
    {Coord : Type}
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [nonemptyCoord : Nonempty Coord]
    {total : ℕ}
    {r : Akita.Key30.Row}
    {member : r ∈ Akita.Key30.rows}
    {o : Akita.Key30.Tower r member E Wire Coord total}
    {shared : o.Shared}
    {read : Akita.Extract.Reader Wire}
    (drawn : o.Drawn shared read 0)
    (slots : 28 ≤ Fintype.card Coord)
    (prover : RandomOracle.OracleComp (Akita.Extract.Address Wire total) (Option Coord → E) (ℕ → Wire))
    (budget : ℕ)
    (table : Akita.Extract.Address Wire total → Option Coord → E)
    (tape : Checkpoint.Tape (Akita.Extract.Address Wire total) (Option Coord) E budget
          (o.body shared read 0 (o.accepts shared read)).protocol.stages)
    (out : Akita.Extract.Opened (Akita.Key30.root r).Weak
            (Akita.Extract.Opened (Akita.Key30.level₁ r).Weak
              (Akita.Extract.Opened (Akita.Weak (Akita.Key30.level₂ r).opening)
                (Akita.Extract.Opened (Akita.Weak Akita.Key30.level₃.opening)
                  (Akita.Extract.Opened (Akita.Weak Akita.Key30.level₄.opening)
                    (Akita.Extract.Opened (Akita.Weak Akita.Key30.level₅.opening)
                      (Akita.Extract.Opened (Akita.Weak Akita.Key30.level₆.opening)
                        (Akita.Extract.Prefix Wire (Option Coord → E) total ×
                          Option (Akita.TerminalFork Akita.Key30.terminal)))))))) ⊕
          (Akita.Batch.LevelBreak o.key o.openKey fun g =>
              2 * (2 * Akita.Key30.mass g) *
                ((Akita.Key30.root r).fold g).sourceBound ((Akita.Key30.root r).shape g)) ⊕
            (Akita.Batch.LevelBreak o.key₁ o.openKey₁ fun g =>
                2 * (2 * Akita.Key30.mass₁ g) *
                  ((Akita.Key30.level₁ r).fold g).sourceBound ((Akita.Key30.level₁ r).shape g)) ⊕
              Akita.Extract.LevelBreak o.key₂ o.openKey₂
                  (2 * (2 * 51) * (Akita.Key30.level₂ r).fold.sourceBound (Akita.Key30.level₂ r).shape) ⊕
                Akita.Extract.LevelBreak o.key₃ o.openKey₃
                    (2 * (2 * 51) * Akita.Key30.level₃.fold.sourceBound Akita.Key30.level₃.shape) ⊕
                  Akita.Extract.Route.Break o.key₄ o.openKey₄ 31
                      (Akita.Key30.level₄.fold.sourceBound Akita.Key30.level₄.shape)
                      Akita.Extract.Route.coefficient ⊕
                    Akita.Extract.Route.Break o.key₅ o.openKey₅ 53
                        (Akita.Key30.level₅.fold.sourceBound Akita.Key30.level₅.shape)
                        (Akita.Extract.Route.euclidean Akita.Key30.norm₅.operator Akita.Key30.norm₅.limit) ⊕
                      Akita.Extract.Route.Break o.key₆ o.openKey₆ 53
                          (Akita.Key30.level₆.fold.sourceBound Akita.Key30.level₆.shape)
                          (Akita.Extract.Route.euclidean Akita.Key30.norm₆.operator Akita.Key30.norm₆.limit) ⊕
                        Option
                          ((Akita.Extract.Route.euclidean Akita.Key30.normTerminal.operator
                                Akita.Key30.normTerminal.limit).Kernel
                            o.terminal.matrix o.terminal.weight o.terminal.bound))
    (result : out ∈ Checkpoint.extract (o.body shared read 0 (o.accepts shared read)) prover budget table tape) :
    o.accepts shared read ((o.body shared read 0 (o.accepts shared read)).protocol.final prover table) ∧
      Sum.elim
        (fun certificate =>
          Akita.Extract.OpensAt 0 o.commit o.value o.place o.key Akita.Key30.mass
            ((o.body shared read 0 (o.accepts shared read)).protocol.input prover table 0)
            ((o.body shared read 0 (o.accepts shared read)).witness certificate))
        (o.body shared read 0 (o.accepts shared read)).breaks out :=
  @NightstreamFPrime.Spec.Compression.Akita.Key30.Tower.extract_sound E fieldE fintypeE decidableEqE
    algebraZqE degree2E Wire Coord fintypeCoord decidableEqCoord nonemptyCoord total r member o shared read
    drawn slots prover budget table tape out result

/-- The Nightstream extraction bound (Bounds above), not Akita Theorem 10.46, for the opening of
a row of `crates/ns-compress/artifacts/key30.aks`, with Akita Lemma 9.1 for the offloaded root.
The model is that of `Claims.Akita.checkpoint_extraction`, and the verifier is the Lean verifier
`Key30.Tower.accepts`, which differs from the verifier of the pinned Akita code. For each prover
with at most `Q` queries, the probability that the verifier accepts and the extractor returns no
output is at most `(Q + 1)·(1623435/(4096·q²) + (1458 + B_1 + B_2)/q²) +
(Q + 1)·(1467 + B_1 + B_2)/budget`. `B_1` and `B_2` are the live blocks of the witness group of
level 1 and of level 2 of the row: 167 and 44 for `W` of 27 variables, and 158 and 43 for `W` of
26 variables. The draw of each check has the factor `2^{−g}` for the work bits `g` of its site in
the Akita code (Akita Remark 10.18, `Extract.workBits`), and the fold terms have no work credit.
A solution of Module-SIS (`[MSIS]`) is an output, and the bound has no term for it. With `drawn`,
each slot gives a fold challenge at each level (`Spec.Compression.Akita.Key30.Tower.Drawn.decodes`,
from the count `drawn.kept` of Akita Table 12). -/
theorem key30_opening_bound
    {E : Type}
    [fieldE : Field E]
    [fintypeE : Fintype E]
    [decidableEqE : DecidableEq E]
    [algebraZqE : Algebra Akita.Zq E]
    [degree2E : Akita.Degree2 E]
    {Wire : Type}
    {Coord : Type}
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    [nonemptyCoord : Nonempty Coord]
    {total : ℕ}
    {r : Akita.Key30.Row}
    {member : r ∈ Akita.Key30.rows}
    {o : Akita.Key30.Tower r member E Wire Coord total}
    {shared : o.Shared}
    {read : Akita.Extract.Reader Wire}
    [finiteWire : Finite Wire]
    (drawn : o.Drawn shared read 0)
    (slots : 28 ≤ Fintype.card Coord)
    (prover : RandomOracle.OracleComp (Akita.Extract.Address Wire total) (Option Coord → E) (ℕ → Wire))
    {queries : ℕ}
    (bounded : prover.QueryBound queries)
    {budget : ℕ}
    (positive : 0 < budget) :
    (𝔼 table,
        𝔼 tape,
          if
              o.accepts shared read
                  ((o.body shared read 0 (o.accepts shared read)).protocol.final prover table) ∧
                Checkpoint.extract (o.body shared read 0 (o.accepts shared read)) prover budget table tape =
                  none then
            1
          else 0) ≤
      ((↑queries : ℝ) + 1) *
          (1623435 / (4096 * (↑(Fintype.card E) : ℝ)) +
            (↑(1458 + r.blocks₁ + r.blocks₂) : ℝ) * (1 / (↑fieldModulus : ℝ) ^ 2)) +
        ((↑queries : ℝ) + 1) * (↑(1467 + r.blocks₁ + r.blocks₂) : ℝ) / (↑budget : ℝ) :=
  @NightstreamFPrime.Spec.Compression.Akita.Key30.Tower.extraction_le E fieldE fintypeE decidableEqE
    algebraZqE degree2E Wire Coord fintypeCoord decidableEqCoord nonemptyCoord total r member o shared read
    finiteWire drawn slots prover queries bounded budget positive

/-- The Lean verifier of the opening of a row of `crates/ns-compress/artifacts/key30.aks` accepts
the proof of the honest prover (Akita Theorem 10.9, Section 9 with (136) and Akita Section 8.2)
for each setup prefix and each family of tables with the fit conditions of the honest prover.
`flat`: the setup prefix `S^♭`, with the commitment and the reference decommitment of the Lean
commitment algorithm. `tables`, `places`: the tables of the key polynomial, `W` and `V` and their
opening points (`live`, `lengths`: their sizes). `fit`: the fit conditions
`Spec.Compression.Akita.Key30.Fit` (Akita Definition 10.5): for each opening point and each fold
challenge vector that the decoders give, the response of each level satisfies `Enc` and the norm
limit of its route. For each split `scan` of the public weight of the root (Akita (132)) and each
table of the oracle on keys whose slots decode to fold challenges and whose work slots have 57 zero
work bits (`worked`: 57 is the largest count of work bits of a site, and Lean does not model the
search of the honest prover for the work bits of Akita Remark 10.18), a proof exists whose first
message has the commitments of the tables, the points and the values, and whose transcript the
verifier `offLinks` accepts. The proof is a sequence of messages of the type `Key30.Wire E`,
which is finite (`Spec.Compression.Akita.Key30.Note.finite`): the points are vectors, and
`Key30.Note.place` and `Key30.Note.places` read them as lists. For the decoders `draw` of
`Key30.Tower`, the split `Key30.Tower.scan` and the readers of `Key30.Note`, the proof is a
prover with no queries of the game of `key30_opening_bound` (`key30_opening_played`). With the
count `Spec.Compression.Akita.Key20.Kept`, each slot decodes, so `decodes` to `decodesL` hold for
each table (`Spec.Compression.Akita.Key30.Tower.Drawn.decodes`). `fit` holds in two cases: for the zero setup prefix and the zero tables with each decoder
(`Spec.Compression.Akita.Key30.fit_zero`), and for each setup prefix and each family of tables
with decoders that give the fold challenge 0 (`Spec.Compression.Akita.Key30.fit_quiet`). For the
challenge families of the schedule and tables that are not 0, `fit` is not a theorem: a response
can be larger than the limit of `Enc`, and the honest prover of the Akita code has a search over
a nonce for a response that fits. The claim is an existence result for each table, not a
completeness bound: Lean has no bound on the probability that a response fits, and the Lean
verifier has no rejection at a zero tensor factor. -/
theorem key30_opening_accepts
    {E : Type}
    [fieldE : Field E]
    [fintypeE : Fintype E]
    [decidableEqE : DecidableEq E]
    [algebraZqE : Algebra Akita.Zq E]
    [degree2E : Akita.Degree2 E]
    {Coord : Type}
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    {r : Akita.Key30.Row}
    (member : r ∈ Akita.Key30.rows)
    (flat : Akita.Table)
    (β : (g : Fin 3) → Module.Basis (Fin ((Akita.Key30.root r).planes g)) Akita.Zq E)
    (key : (g : Fin 3) → Akita.Key ((Akita.Key30.root r).shape g))
    (openKey : (Akita.Key30.root r).OpenKey)
    (block : (Akita.Key30.root r).Block → Coord)
    (decode : (g : Fin 3) → E → Option (Akita.Rq ((Akita.Key30.root r).carrierDegree g)))
    (mass : Fin 3 → ℕ)
    (scan : (wire : Akita.Key30.Wire r E) →
          (challenge : (Akita.Key30.root r).Challenge) →
            Akita.SetupScan
              (Akita.Batch.liftedReduction β (Akita.Batch.Claim.statement key openKey wire.data.claim)
                challenge Akita.Key30.rootVars (Akita.Key30.root_sized r member).left 8
                Akita.Key30.quotientDigits₀ (Akita.Key30.root_sized r member).right.left Akita.Key30.rootBatch
                (Akita.Key30.root_sized r member).right.right.left)
              Akita.Key30.setupVars (((Akita.Key30.level₁ r).shape 0).pad flat))
    (β₁ : (g : Fin 2) → Module.Basis (Fin ((Akita.Key30.level₁ r).planes g)) Akita.Zq E)
    (key₁ : (g : Fin 2) → Akita.Key ((Akita.Key30.level₁ r).shape g))
    (openKey₁ : (Akita.Key30.level₁ r).OpenKey)
    (block₁ : (Akita.Key30.level₁ r).Block → Coord)
    (decode₁ : (g : Fin 2) → E → Option (Akita.Rq ((Akita.Key30.level₁ r).carrierDegree g)))
    (mass₁ : Fin 2 → ℕ)
    (β₂ : Module.Basis (Fin (Akita.Key30.level₂ r).opening.planes) Akita.Zq E)
    (key₂ : Akita.Key (Akita.Key30.level₂ r).shape)
    (openKey₂ : Akita.OpeningKey (Akita.Key30.level₂ r).opening)
    (block₂ : Fin (Akita.Key30.level₂ r).shape.blocks → Coord)
    (decode₂ : E → Option (Akita.Rq (Akita.Key30.level₂ r).opening.carrierDegree))
    (weight₂ : ℕ)
    (β₃ : Module.Basis (Fin Akita.Key30.level₃.opening.planes) Akita.Zq E)
    (key₃ : Akita.Key Akita.Key30.level₃.shape)
    (openKey₃ : Akita.OpeningKey Akita.Key30.level₃.opening)
    (block₃ : Fin Akita.Key30.level₃.shape.blocks → Coord)
    (decode₃ : E → Option (Akita.Rq Akita.Key30.level₃.opening.carrierDegree))
    (weight₃ : ℕ)
    (key₄ : Akita.Key Akita.Key30.level₄.shape)
    (openKey₄ : Akita.OpeningKey Akita.Key30.level₄.opening)
    (block₄ : Fin Akita.Key30.level₄.shape.blocks → Coord)
    (decode₄ : E → Option (Akita.Rq Akita.Key30.level₄.shape.innerDegree))
    (weight₄ : ℕ)
    (key₅ : Akita.Key Akita.Key30.level₅.shape)
    (openKey₅ : Akita.OpeningKey Akita.Key30.level₅.opening)
    (block₅ : Fin Akita.Key30.level₅.shape.blocks → Coord)
    (decode₅ : E → Option (Akita.Rq Akita.Key30.level₅.shape.innerDegree))
    (weight₅ : ℕ)
    (key₆ : Akita.Key Akita.Key30.level₆.shape)
    (openKey₆ : Akita.OpeningKey Akita.Key30.level₆.opening)
    (block₆ : Fin Akita.Key30.level₆.shape.blocks → Coord)
    (decode₆ : E → Option (Akita.Rq Akita.Key30.level₆.shape.innerDegree))
    (weight₆ : ℕ)
    (matrix : Matrix (Fin Akita.Key30.terminal.innerRank)
          (Fin Akita.Key30.terminal.blockLength × Fin Akita.Key30.terminal.sourceDigits)
          (Akita.Rq Akita.Key30.terminal.innerDegree))
    (blockL : Fin Akita.Key30.terminal.blocks → Coord)
    (decodeL : E → Option (Akita.Rq Akita.Key30.terminal.innerDegree))
    {total : ℕ}
    (inside : ((Akita.Extract.committedTraceSq key₆ openKey₆
                  (Akita.Key30.Note.claim₆ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _) Akita.Key30.level₆.vars
                  (Akita.Key30.tails_sized Akita.Key30.level₆
                      (List.mem_cons_of_mem Akita.Key30.level₅ List.mem_cons_self)).left
                  Akita.Key30.level₆.batch
                  (Akita.Key30.tails_sized Akita.Key30.level₆
                        (List.mem_cons_of_mem Akita.Key30.level₅ List.mem_cons_self)).right.left
                  block₆ decode₆ Akita.Extract.Page.read weight₆ Akita.Key30.norm₆.operator
                  Akita.Key30.norm₆.limit
                  ((Akita.Extract.committedTraceSq key₅ openKey₅
                          (Akita.Key30.Note.claim₅ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _) Akita.Key30.level₅.vars
                          (Akita.Key30.tails_sized Akita.Key30.level₅ List.mem_cons_self).left
                          Akita.Key30.level₅.batch
                          (Akita.Key30.tails_sized Akita.Key30.level₅ List.mem_cons_self).right.left block₅
                          decode₅ Akita.Extract.Page.read weight₅ Akita.Key30.norm₅.operator
                          Akita.Key30.norm₅.limit
                          ((Akita.Extract.committedTrace key₄ openKey₄
                                  (Akita.Key30.Note.claim₄ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _) Akita.Key30.level₄.vars
                                  Akita.Key30.level₄_sized.left Akita.Key30.level₄.batch
                                  Akita.Key30.level₄_sized.right.left block₄ decode₄ Akita.Extract.Page.read
                                  weight₄
                                  ((Akita.Extract.liftFold β₃ key₃ openKey₃
                                          (Akita.Key30.Note.claim₃ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
                                          Akita.Key30.level₃.vars Akita.Key30.level₃_sized.left
                                          Akita.Key30.quotientDigits Akita.Key30.level₃_sized.right.left
                                          Akita.Key30.level₃.batch Akita.Key30.level₃_sized.right.right.left
                                          block₃ decode₃ Akita.Extract.Page.read weight₃
                                          (Akita.Extract.liftFold β₂ key₂ openKey₂
                                              (Akita.Key30.Note.claim₂ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
                                              (Akita.Key30.level₂ r).vars
                                              (Akita.Key30.level₂_sized r member).left
                                              Akita.Key30.quotientDigits
                                              (Akita.Key30.level₂_sized r member).right.left
                                              (Akita.Key30.level₂ r).batch
                                              (Akita.Key30.level₂_sized r member).right.right.left block₂
                                              decode₂ Akita.Extract.Page.read weight₂
                                              ((Akita.Extract.liftBatch β₁ key₁ openKey₁
                                                      (Akita.Key30.Note.claim₁ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
                                                      Akita.Key30.vars₁
                                                      (Akita.Key30.level₁_sized r member).left 16
                                                      Akita.Key30.quotientDigits
                                                      (Akita.Key30.level₁_sized r member).right.left
                                                      Akita.Key30.batch₁
                                                      (Akita.Key30.level₁_sized r member).right.right.left
                                                      block₁ decode₁ Akita.Extract.Page.read mass₁
                                                      ((Akita.Extract.offBatch β key openKey
                                                            (Akita.Key30.Note.claim ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
                                                            Akita.Key30.rootVars
                                                            (Akita.Key30.root_sized r member).left 8
                                                            Akita.Key30.quotientDigits₀
                                                            (Akita.Key30.root_sized r member).right.left
                                                            Akita.Key30.rootBatch
                                                            (Akita.Key30.root_sized r member).right.right.left
                                                            block decode Akita.Extract.Page.read mass (0 + 1)
                                                            Akita.Key30.setupVars (((Akita.Key30.level₁ r).shape 0).pad flat) scan).finish +
                                                        1)).pinned
                                                  (Akita.Batch.Pinned 0 (Akita.commit (key₁ 0) flat).1 (((Akita.Key30.level₁ r).shape 0).pad flat))
                                                  fun ω _ =>
                                                  Akita.Batch.pinBreak key₁ openKey₁
                                                    (fun g =>
                                                      2 * (2 * mass₁ g) *
                                                        ((Akita.Key30.level₁ r).fold g).sourceBound
                                                          ((Akita.Key30.level₁ r).shape g))
                                                    0 (Akita.commit (key₁ 0) flat).2 ω).finish).finish).tensored
                                      Akita.Key30.level₃.tail).finish).tensored
                              Akita.Key30.level₄.tail).finish).tensored
                      Akita.Key30.level₅.tail).finish).tensored
              Akita.Key30.level₆.tail).finish <
          total)
    (accept : Akita.Extract.Prefix (Akita.Key30.Wire r E) (Option Coord → E) total → Prop)
    (table : Akita.Extract.Address (Akita.Key30.Wire r E) total → Option Coord → E)
    (decodes : ∀ (input : Akita.Extract.Address (Akita.Key30.Wire r E) total)
          (b : (Akita.Key30.root r).Block), (decode b.fst (table input (some (block b)))).isSome = true)
    (decodes₁ : ∀ (input : Akita.Extract.Address (Akita.Key30.Wire r E) total)
          (b : (Akita.Key30.level₁ r).Block), (decode₁ b.fst (table input (some (block₁ b)))).isSome = true)
    (decodes₂ : ∀ (input : Akita.Extract.Address (Akita.Key30.Wire r E) total)
          (i : Fin (Akita.Key30.level₂ r).shape.blocks),
          (decode₂ (table input (some (block₂ i)))).isSome = true)
    (decodes₃ : ∀ (input : Akita.Extract.Address (Akita.Key30.Wire r E) total)
          (i : Fin Akita.Key30.level₃.shape.blocks), (decode₃ (table input (some (block₃ i)))).isSome = true)
    (decodes₄ : ∀ (input : Akita.Extract.Address (Akita.Key30.Wire r E) total)
          (i : Fin Akita.Key30.level₄.shape.blocks), (decode₄ (table input (some (block₄ i)))).isSome = true)
    (decodes₅ : ∀ (input : Akita.Extract.Address (Akita.Key30.Wire r E) total)
          (i : Fin Akita.Key30.level₅.shape.blocks), (decode₅ (table input (some (block₅ i)))).isSome = true)
    (decodes₆ : ∀ (input : Akita.Extract.Address (Akita.Key30.Wire r E) total)
          (i : Fin Akita.Key30.level₆.shape.blocks), (decode₆ (table input (some (block₆ i)))).isSome = true)
    (decodesL : ∀ (input : Akita.Extract.Address (Akita.Key30.Wire r E) total)
          (i : Fin Akita.Key30.terminal.blocks), (decodeL (table input (some (blockL i)))).isSome = true)
    (worked : ∀ (input : Akita.Extract.Address (Akita.Key30.Wire r E) total),
          table input none ∈ Akita.Extract.workZeros E 57)
    {tables : Fin 3 → Akita.Table}
    (live : ∀ (g : Fin 3) (cell : ℕ), 2 ^ ![30, r.vars, 28] g ≤ cell → tables g cell = 0)
    {places : Fin 3 → List E}
    (lengths : ∀ (g : Fin 3), (places g).length = ![30, r.vars, 28] g)
    (fit : Akita.Key30.Fit member flat β key openKey decode β₁ key₁ openKey₁ decode₁ β₂ key₂ openKey₂
        decode₂ β₃ key₃ openKey₃ decode₃ key₄ openKey₄ decode₄ key₅ openKey₅ decode₅ key₆ openKey₆
        decode₆ decodeL tables) :
    ∃ proof : ℕ → Akita.Key30.Wire r E,
      ((proof 0).data.commit = fun g => (Akita.commit (key g) (tables g)).1) ∧
      (proof 0).data.place = places ∧
      ((proof 0).data.value = fun g => Lib.mle (places g) (Akita.lifted E (tables g))) ∧
      Akita.Extract.offLinks (weight₅ := weight₅) (weight₆ := weight₆)
        (route₄ := Akita.Extract.Route.coefficient)
        (route₅ := Akita.Extract.Route.euclidean Akita.Key30.norm₅.operator Akita.Key30.norm₅.limit)
        (route₆ := Akita.Extract.Route.euclidean Akita.Key30.norm₆.operator Akita.Key30.norm₆.limit)
        (Akita.Extract.offBatch β key openKey (Akita.Key30.Note.claim ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
          Akita.Key30.rootVars (Akita.Key30.root_sized r member).left 8 Akita.Key30.quotientDigits₀
          (Akita.Key30.root_sized r member).right.left Akita.Key30.rootBatch
          (Akita.Key30.root_sized r member).right.right.left block decode Akita.Extract.Page.read mass
          (0 + 1) Akita.Key30.setupVars (((Akita.Key30.level₁ r).shape 0).pad flat) scan)
        ((Akita.Extract.liftBatch β₁ key₁ openKey₁ (Akita.Key30.Note.claim₁ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
              Akita.Key30.vars₁ (Akita.Key30.level₁_sized r member).left 16 Akita.Key30.quotientDigits
              (Akita.Key30.level₁_sized r member).right.left Akita.Key30.batch₁
              (Akita.Key30.level₁_sized r member).right.right.left block₁ decode₁ Akita.Extract.Page.read
              mass₁
              ((Akita.Extract.offBatch β key openKey (Akita.Key30.Note.claim ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
                    Akita.Key30.rootVars (Akita.Key30.root_sized r member).left 8
                    Akita.Key30.quotientDigits₀ (Akita.Key30.root_sized r member).right.left
                    Akita.Key30.rootBatch (Akita.Key30.root_sized r member).right.right.left block decode
                    Akita.Extract.Page.read mass (0 + 1) Akita.Key30.setupVars (((Akita.Key30.level₁ r).shape 0).pad flat) scan).finish +
                1)).pinned
          (Akita.Batch.Pinned 0 (Akita.commit (key₁ 0) flat).1 (((Akita.Key30.level₁ r).shape 0).pad flat)) fun ω _ =>
          Akita.Batch.pinBreak key₁ openKey₁
            (fun g =>
              2 * (2 * mass₁ g) *
                ((Akita.Key30.level₁ r).fold g).sourceBound ((Akita.Key30.level₁ r).shape g))
            0 (Akita.commit (key₁ 0) flat).2 ω)
        (Akita.Extract.liftFold β₂ key₂ openKey₂ (Akita.Key30.Note.claim₂ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
          (Akita.Key30.level₂ r).vars (Akita.Key30.level₂_sized r member).left Akita.Key30.quotientDigits
          (Akita.Key30.level₂_sized r member).right.left (Akita.Key30.level₂ r).batch
          (Akita.Key30.level₂_sized r member).right.right.left block₂ decode₂ Akita.Extract.Page.read
          weight₂
          ((Akita.Extract.liftBatch β₁ key₁ openKey₁ (Akita.Key30.Note.claim₁ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
                  Akita.Key30.vars₁ (Akita.Key30.level₁_sized r member).left 16 Akita.Key30.quotientDigits
                  (Akita.Key30.level₁_sized r member).right.left Akita.Key30.batch₁
                  (Akita.Key30.level₁_sized r member).right.right.left block₁ decode₁
                  Akita.Extract.Page.read mass₁
                  ((Akita.Extract.offBatch β key openKey
                        (Akita.Key30.Note.claim ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _) Akita.Key30.rootVars
                        (Akita.Key30.root_sized r member).left 8 Akita.Key30.quotientDigits₀
                        (Akita.Key30.root_sized r member).right.left Akita.Key30.rootBatch
                        (Akita.Key30.root_sized r member).right.right.left block decode
                        Akita.Extract.Page.read mass (0 + 1) Akita.Key30.setupVars (((Akita.Key30.level₁ r).shape 0).pad flat) scan).finish +
                    1)).pinned
              (Akita.Batch.Pinned 0 (Akita.commit (key₁ 0) flat).1 (((Akita.Key30.level₁ r).shape 0).pad flat)) fun ω _ =>
              Akita.Batch.pinBreak key₁ openKey₁
                (fun g =>
                  2 * (2 * mass₁ g) *
                    ((Akita.Key30.level₁ r).fold g).sourceBound ((Akita.Key30.level₁ r).shape g))
                0 (Akita.commit (key₁ 0) flat).2 ω).finish)
        ((Akita.Extract.liftFold β₃ key₃ openKey₃ (Akita.Key30.Note.claim₃ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
              Akita.Key30.level₃.vars Akita.Key30.level₃_sized.left Akita.Key30.quotientDigits
              Akita.Key30.level₃_sized.right.left Akita.Key30.level₃.batch
              Akita.Key30.level₃_sized.right.right.left block₃ decode₃ Akita.Extract.Page.read weight₃
              (Akita.Extract.liftFold β₂ key₂ openKey₂ (Akita.Key30.Note.claim₂ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
                  (Akita.Key30.level₂ r).vars (Akita.Key30.level₂_sized r member).left
                  Akita.Key30.quotientDigits (Akita.Key30.level₂_sized r member).right.left
                  (Akita.Key30.level₂ r).batch (Akita.Key30.level₂_sized r member).right.right.left block₂
                  decode₂ Akita.Extract.Page.read weight₂
                  ((Akita.Extract.liftBatch β₁ key₁ openKey₁
                          (Akita.Key30.Note.claim₁ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _) Akita.Key30.vars₁
                          (Akita.Key30.level₁_sized r member).left 16 Akita.Key30.quotientDigits
                          (Akita.Key30.level₁_sized r member).right.left Akita.Key30.batch₁
                          (Akita.Key30.level₁_sized r member).right.right.left block₁ decode₁
                          Akita.Extract.Page.read mass₁
                          ((Akita.Extract.offBatch β key openKey
                                (Akita.Key30.Note.claim ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _) Akita.Key30.rootVars
                                (Akita.Key30.root_sized r member).left 8 Akita.Key30.quotientDigits₀
                                (Akita.Key30.root_sized r member).right.left Akita.Key30.rootBatch
                                (Akita.Key30.root_sized r member).right.right.left block decode
                                Akita.Extract.Page.read mass (0 + 1) Akita.Key30.setupVars (((Akita.Key30.level₁ r).shape 0).pad flat)
                                scan).finish +
                            1)).pinned
                      (Akita.Batch.Pinned 0 (Akita.commit (key₁ 0) flat).1 (((Akita.Key30.level₁ r).shape 0).pad flat)) fun ω _ =>
                      Akita.Batch.pinBreak key₁ openKey₁
                        (fun g =>
                          2 * (2 * mass₁ g) *
                            ((Akita.Key30.level₁ r).fold g).sourceBound ((Akita.Key30.level₁ r).shape g))
                        0 (Akita.commit (key₁ 0) flat).2 ω).finish).finish).tensored
          Akita.Key30.level₃.tail)
        ((Akita.Extract.committedTrace key₄ openKey₄ (Akita.Key30.Note.claim₄ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
              Akita.Key30.level₄.vars Akita.Key30.level₄_sized.left Akita.Key30.level₄.batch
              Akita.Key30.level₄_sized.right.left block₄ decode₄ Akita.Extract.Page.read weight₄
              ((Akita.Extract.liftFold β₃ key₃ openKey₃
                      (Akita.Key30.Note.claim₃ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _) Akita.Key30.level₃.vars
                      Akita.Key30.level₃_sized.left Akita.Key30.quotientDigits
                      Akita.Key30.level₃_sized.right.left Akita.Key30.level₃.batch
                      Akita.Key30.level₃_sized.right.right.left block₃ decode₃ Akita.Extract.Page.read
                      weight₃
                      (Akita.Extract.liftFold β₂ key₂ openKey₂
                          (Akita.Key30.Note.claim₂ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _) (Akita.Key30.level₂ r).vars
                          (Akita.Key30.level₂_sized r member).left Akita.Key30.quotientDigits
                          (Akita.Key30.level₂_sized r member).right.left (Akita.Key30.level₂ r).batch
                          (Akita.Key30.level₂_sized r member).right.right.left block₂ decode₂
                          Akita.Extract.Page.read weight₂
                          ((Akita.Extract.liftBatch β₁ key₁ openKey₁
                                  (Akita.Key30.Note.claim₁ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _) Akita.Key30.vars₁
                                  (Akita.Key30.level₁_sized r member).left 16 Akita.Key30.quotientDigits
                                  (Akita.Key30.level₁_sized r member).right.left Akita.Key30.batch₁
                                  (Akita.Key30.level₁_sized r member).right.right.left block₁ decode₁
                                  Akita.Extract.Page.read mass₁
                                  ((Akita.Extract.offBatch β key openKey
                                        (Akita.Key30.Note.claim ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
                                        Akita.Key30.rootVars (Akita.Key30.root_sized r member).left 8
                                        Akita.Key30.quotientDigits₀
                                        (Akita.Key30.root_sized r member).right.left Akita.Key30.rootBatch
                                        (Akita.Key30.root_sized r member).right.right.left block decode
                                        Akita.Extract.Page.read mass (0 + 1) Akita.Key30.setupVars (((Akita.Key30.level₁ r).shape 0).pad flat)
                                        scan).finish +
                                    1)).pinned
                              (Akita.Batch.Pinned 0 (Akita.commit (key₁ 0) flat).1 (((Akita.Key30.level₁ r).shape 0).pad flat)) fun ω _ =>
                              Akita.Batch.pinBreak key₁ openKey₁
                                (fun g =>
                                  2 * (2 * mass₁ g) *
                                    ((Akita.Key30.level₁ r).fold g).sourceBound
                                      ((Akita.Key30.level₁ r).shape g))
                                0 (Akita.commit (key₁ 0) flat).2 ω).finish).finish).tensored
                  Akita.Key30.level₃.tail).finish).tensored
          Akita.Key30.level₄.tail)
        ((Akita.Extract.committedTraceSq key₅ openKey₅ (Akita.Key30.Note.claim₅ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
              Akita.Key30.level₅.vars (Akita.Key30.tails_sized Akita.Key30.level₅ List.mem_cons_self).left
              Akita.Key30.level₅.batch
              (Akita.Key30.tails_sized Akita.Key30.level₅ List.mem_cons_self).right.left block₅ decode₅
              Akita.Extract.Page.read weight₅ Akita.Key30.norm₅.operator Akita.Key30.norm₅.limit
              ((Akita.Extract.committedTrace key₄ openKey₄
                      (Akita.Key30.Note.claim₄ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _) Akita.Key30.level₄.vars
                      Akita.Key30.level₄_sized.left Akita.Key30.level₄.batch
                      Akita.Key30.level₄_sized.right.left block₄ decode₄ Akita.Extract.Page.read weight₄
                      ((Akita.Extract.liftFold β₃ key₃ openKey₃
                              (Akita.Key30.Note.claim₃ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _) Akita.Key30.level₃.vars
                              Akita.Key30.level₃_sized.left Akita.Key30.quotientDigits
                              Akita.Key30.level₃_sized.right.left Akita.Key30.level₃.batch
                              Akita.Key30.level₃_sized.right.right.left block₃ decode₃
                              Akita.Extract.Page.read weight₃
                              (Akita.Extract.liftFold β₂ key₂ openKey₂
                                  (Akita.Key30.Note.claim₂ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
                                  (Akita.Key30.level₂ r).vars (Akita.Key30.level₂_sized r member).left
                                  Akita.Key30.quotientDigits
                                  (Akita.Key30.level₂_sized r member).right.left
                                  (Akita.Key30.level₂ r).batch
                                  (Akita.Key30.level₂_sized r member).right.right.left block₂ decode₂
                                  Akita.Extract.Page.read weight₂
                                  ((Akita.Extract.liftBatch β₁ key₁ openKey₁
                                          (Akita.Key30.Note.claim₁ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
                                          Akita.Key30.vars₁ (Akita.Key30.level₁_sized r member).left 16
                                          Akita.Key30.quotientDigits
                                          (Akita.Key30.level₁_sized r member).right.left
                                          Akita.Key30.batch₁
                                          (Akita.Key30.level₁_sized r member).right.right.left block₁
                                          decode₁ Akita.Extract.Page.read mass₁
                                          ((Akita.Extract.offBatch β key openKey
                                                (Akita.Key30.Note.claim ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
                                                Akita.Key30.rootVars
                                                (Akita.Key30.root_sized r member).left 8
                                                Akita.Key30.quotientDigits₀
                                                (Akita.Key30.root_sized r member).right.left
                                                Akita.Key30.rootBatch
                                                (Akita.Key30.root_sized r member).right.right.left block
                                                decode Akita.Extract.Page.read mass (0 + 1)
                                                Akita.Key30.setupVars (((Akita.Key30.level₁ r).shape 0).pad flat) scan).finish +
                                            1)).pinned
                                      (Akita.Batch.Pinned 0 (Akita.commit (key₁ 0) flat).1 (((Akita.Key30.level₁ r).shape 0).pad flat)) fun ω _ =>
                                      Akita.Batch.pinBreak key₁ openKey₁
                                        (fun g =>
                                          2 * (2 * mass₁ g) *
                                            ((Akita.Key30.level₁ r).fold g).sourceBound
                                              ((Akita.Key30.level₁ r).shape g))
                                        0 (Akita.commit (key₁ 0) flat).2 ω).finish).finish).tensored
                          Akita.Key30.level₃.tail).finish).tensored
                  Akita.Key30.level₄.tail).finish).tensored
          Akita.Key30.level₅.tail)
        ((Akita.Extract.committedTraceSq key₆ openKey₆ (Akita.Key30.Note.claim₆ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
              Akita.Key30.level₆.vars
              (Akita.Key30.tails_sized Akita.Key30.level₆
                  (List.mem_cons_of_mem Akita.Key30.level₅ List.mem_cons_self)).left
              Akita.Key30.level₆.batch
              (Akita.Key30.tails_sized Akita.Key30.level₆
                    (List.mem_cons_of_mem Akita.Key30.level₅ List.mem_cons_self)).right.left
              block₆ decode₆ Akita.Extract.Page.read weight₆ Akita.Key30.norm₆.operator
              Akita.Key30.norm₆.limit
              ((Akita.Extract.committedTraceSq key₅ openKey₅
                      (Akita.Key30.Note.claim₅ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _) Akita.Key30.level₅.vars
                      (Akita.Key30.tails_sized Akita.Key30.level₅ List.mem_cons_self).left
                      Akita.Key30.level₅.batch
                      (Akita.Key30.tails_sized Akita.Key30.level₅ List.mem_cons_self).right.left block₅
                      decode₅ Akita.Extract.Page.read weight₅ Akita.Key30.norm₅.operator
                      Akita.Key30.norm₅.limit
                      ((Akita.Extract.committedTrace key₄ openKey₄
                              (Akita.Key30.Note.claim₄ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _) Akita.Key30.level₄.vars
                              Akita.Key30.level₄_sized.left Akita.Key30.level₄.batch
                              Akita.Key30.level₄_sized.right.left block₄ decode₄ Akita.Extract.Page.read
                              weight₄
                              ((Akita.Extract.liftFold β₃ key₃ openKey₃
                                      (Akita.Key30.Note.claim₃ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
                                      Akita.Key30.level₃.vars Akita.Key30.level₃_sized.left
                                      Akita.Key30.quotientDigits Akita.Key30.level₃_sized.right.left
                                      Akita.Key30.level₃.batch Akita.Key30.level₃_sized.right.right.left
                                      block₃ decode₃ Akita.Extract.Page.read weight₃
                                      (Akita.Extract.liftFold β₂ key₂ openKey₂
                                          (Akita.Key30.Note.claim₂ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
                                          (Akita.Key30.level₂ r).vars
                                          (Akita.Key30.level₂_sized r member).left
                                          Akita.Key30.quotientDigits
                                          (Akita.Key30.level₂_sized r member).right.left
                                          (Akita.Key30.level₂ r).batch
                                          (Akita.Key30.level₂_sized r member).right.right.left block₂
                                          decode₂ Akita.Extract.Page.read weight₂
                                          ((Akita.Extract.liftBatch β₁ key₁ openKey₁
                                                  (Akita.Key30.Note.claim₁ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
                                                  Akita.Key30.vars₁
                                                  (Akita.Key30.level₁_sized r member).left 16
                                                  Akita.Key30.quotientDigits
                                                  (Akita.Key30.level₁_sized r member).right.left
                                                  Akita.Key30.batch₁
                                                  (Akita.Key30.level₁_sized r member).right.right.left
                                                  block₁ decode₁ Akita.Extract.Page.read mass₁
                                                  ((Akita.Extract.offBatch β key openKey
                                                        (Akita.Key30.Note.claim ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
                                                        Akita.Key30.rootVars
                                                        (Akita.Key30.root_sized r member).left 8
                                                        Akita.Key30.quotientDigits₀
                                                        (Akita.Key30.root_sized r member).right.left
                                                        Akita.Key30.rootBatch
                                                        (Akita.Key30.root_sized r member).right.right.left
                                                        block decode Akita.Extract.Page.read mass (0 + 1)
                                                        Akita.Key30.setupVars (((Akita.Key30.level₁ r).shape 0).pad flat) scan).finish +
                                                    1)).pinned
                                              (Akita.Batch.Pinned 0 (Akita.commit (key₁ 0) flat).1 (((Akita.Key30.level₁ r).shape 0).pad flat))
                                              fun ω _ =>
                                              Akita.Batch.pinBreak key₁ openKey₁
                                                (fun g =>
                                                  2 * (2 * mass₁ g) *
                                                    ((Akita.Key30.level₁ r).fold g).sourceBound
                                                      ((Akita.Key30.level₁ r).shape g))
                                                0 (Akita.commit (key₁ 0) flat).2 ω).finish).finish).tensored
                                  Akita.Key30.level₃.tail).finish).tensored
                          Akita.Key30.level₄.tail).finish).tensored
                  Akita.Key30.level₅.tail).finish).tensored
          Akita.Key30.level₆.tail)
        (Akita.Key30.terminalOf r total matrix blockL decodeL
          ((Akita.Extract.committedTraceSq key₆ openKey₆
                  (Akita.Key30.Note.claim₆ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _) Akita.Key30.level₆.vars
                  (Akita.Key30.tails_sized Akita.Key30.level₆
                      (List.mem_cons_of_mem Akita.Key30.level₅ List.mem_cons_self)).left
                  Akita.Key30.level₆.batch
                  (Akita.Key30.tails_sized Akita.Key30.level₆
                        (List.mem_cons_of_mem Akita.Key30.level₅ List.mem_cons_self)).right.left
                  block₆ decode₆ Akita.Extract.Page.read weight₆ Akita.Key30.norm₆.operator
                  Akita.Key30.norm₆.limit
                  ((Akita.Extract.committedTraceSq key₅ openKey₅
                          (Akita.Key30.Note.claim₅ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _) Akita.Key30.level₅.vars
                          (Akita.Key30.tails_sized Akita.Key30.level₅ List.mem_cons_self).left
                          Akita.Key30.level₅.batch
                          (Akita.Key30.tails_sized Akita.Key30.level₅ List.mem_cons_self).right.left
                          block₅ decode₅ Akita.Extract.Page.read weight₅ Akita.Key30.norm₅.operator
                          Akita.Key30.norm₅.limit
                          ((Akita.Extract.committedTrace key₄ openKey₄
                                  (Akita.Key30.Note.claim₄ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
                                  Akita.Key30.level₄.vars Akita.Key30.level₄_sized.left
                                  Akita.Key30.level₄.batch Akita.Key30.level₄_sized.right.left block₄
                                  decode₄ Akita.Extract.Page.read weight₄
                                  ((Akita.Extract.liftFold β₃ key₃ openKey₃
                                          (Akita.Key30.Note.claim₃ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
                                          Akita.Key30.level₃.vars Akita.Key30.level₃_sized.left
                                          Akita.Key30.quotientDigits Akita.Key30.level₃_sized.right.left
                                          Akita.Key30.level₃.batch
                                          Akita.Key30.level₃_sized.right.right.left block₃ decode₃
                                          Akita.Extract.Page.read weight₃
                                          (Akita.Extract.liftFold β₂ key₂ openKey₂
                                              (Akita.Key30.Note.claim₂ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
                                              (Akita.Key30.level₂ r).vars
                                              (Akita.Key30.level₂_sized r member).left
                                              Akita.Key30.quotientDigits
                                              (Akita.Key30.level₂_sized r member).right.left
                                              (Akita.Key30.level₂ r).batch
                                              (Akita.Key30.level₂_sized r member).right.right.left block₂
                                              decode₂ Akita.Extract.Page.read weight₂
                                              ((Akita.Extract.liftBatch β₁ key₁ openKey₁
                                                      (Akita.Key30.Note.claim₁ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
                                                      Akita.Key30.vars₁
                                                      (Akita.Key30.level₁_sized r member).left 16
                                                      Akita.Key30.quotientDigits
                                                      (Akita.Key30.level₁_sized r member).right.left
                                                      Akita.Key30.batch₁
                                                      (Akita.Key30.level₁_sized r member).right.right.left
                                                      block₁ decode₁ Akita.Extract.Page.read mass₁
                                                      ((Akita.Extract.offBatch β key openKey
                                                            (Akita.Key30.Note.claim ∘
                                                              Akita.Extract.Page.data)
                                                            Akita.Key30.rootVars
                                                            (Akita.Key30.root_sized r member).left 8
                                                            Akita.Key30.quotientDigits₀
                                                            (Akita.Key30.root_sized r member).right.left
                                                            Akita.Key30.rootBatch
                                                            (Akita.Key30.root_sized r
                                                                    member).right.right.left
                                                            block decode Akita.Extract.Page.read mass
                                                            (0 + 1) Akita.Key30.setupVars (((Akita.Key30.level₁ r).shape 0).pad flat) scan).finish +
                                                        1)).pinned
                                                  (Akita.Batch.Pinned 0 (Akita.commit (key₁ 0) flat).1 (((Akita.Key30.level₁ r).shape 0).pad flat))
                                                  fun ω _ =>
                                                  Akita.Batch.pinBreak key₁ openKey₁
                                                    (fun g =>
                                                      2 * (2 * mass₁ g) *
                                                        ((Akita.Key30.level₁ r).fold g).sourceBound
                                                          ((Akita.Key30.level₁ r).shape g))
                                                    0 (Akita.commit (key₁ 0) flat).2
                                                    ω).finish).finish).tensored
                                      Akita.Key30.level₃.tail).finish).tensored
                              Akita.Key30.level₄.tail).finish).tensored
                      Akita.Key30.level₅.tail).finish).tensored
              Akita.Key30.level₆.tail).finish)
        0 Akita.Extract.Page.read (Akita.Key30.Note.commit ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
        (Akita.Key30.Note.value ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
        (Akita.Key30.Note.place ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
        (Akita.Key30.Note.commits ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
        (Akita.Key30.Note.values ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
        (Akita.Key30.Note.places ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _) (Akita.commit (key₁ 0) flat).1
        (Akita.Key30.Note.commit₁ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
        (Akita.Key30.Note.commit₂ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
        (Akita.Key30.Note.commit₃ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _) Akita.Key30.quarters.left
        Akita.Key30.quarters.right.left Akita.Key30.quarters.right.right.left
        Akita.Key30.quarters.right.right.right (Akita.Key30.Note.commit₄ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
        (Akita.Key30.Note.commit₅ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
        (Akita.Key30.Note.commit₆ ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
        (Akita.Key30.Note.state ∘ Akita.Extract.Page.data : Akita.Key30.Wire r E → _)
        ((Akita.Extract.transcript total accept).final (RandomOracle.OracleComp.done proof) table) :=
  @NightstreamFPrime.Spec.Compression.Akita.Key30.accepts E fieldE fintypeE decidableEqE algebraZqE
    degree2E Coord fintypeCoord decidableEqCoord r member flat β key openKey block decode mass scan β₁ key₁
    openKey₁ block₁ decode₁ mass₁ β₂ key₂ openKey₂ block₂ decode₂ weight₂ β₃ key₃ openKey₃ block₃ decode₃
    weight₃ key₄ openKey₄ block₄ decode₄ weight₄ key₅ openKey₅ block₅ decode₅ weight₅ key₆ openKey₆ block₆
    decode₆ weight₆ matrix blockL decodeL total inside accept table decodes decodes₁ decodes₂ decodes₃
    decodes₄ decodes₅ decodes₆ decodesL worked tables live places lengths fit

/-- The proof of the honest prover of the opening of a row of
`crates/ns-compress/artifacts/key30.aks` is a prover of the game of `key30_opening_bound` (Akita
Theorem 10.9, Section 9 with (136), and Theorem 10.17). The data `o` of that claim is
`Spec.Compression.Akita.Key30.noteTower`: the wire messages `Key30.Wire E` with the readers of
`Key30.Note`, the setup prefix `flat` with the commitment and the reference decommitment of the
Lean commitment algorithm, the decoders `draw` of `Key30.Tower`, and free bases, matrices and
coordinates of the slots. `shared` and `drawn` are the hypotheses of `key30_opening_bound` for
this data, and `drawn` fixes the stage `stage` of the terminal. Lean proves that the wire type is
finite, so `key30_opening_bound` holds for `o`, and that for each table of the oracle on keys whose
work slots have 57 zero work bits (`worked`) and each family of tables with the fit conditions
`fit` for these decoders, a proof exists whose first message has the commitments of the tables,
the points and the values, such that the prover `.done proof` has no queries and the verifier
`o.accepts` accepts its transcript. So the accept event of `key30_opening_bound` is not empty
when `fit` holds. For the zero setup prefix and the zero tables `fit` holds with no hypothesis
(`Spec.Compression.Akita.Key30.noteTower_zero`, `Spec.Compression.Akita.Key30.noteTower_played`).
The proof is chosen for each table: the claim is an existence result for each table, not a
completeness bound. -/
theorem key30_opening_played
    {E : Type}
    [fieldE : Field E]
    [fintypeE : Fintype E]
    [decidableEqE : DecidableEq E]
    [algebraZqE : Algebra Akita.Zq E]
    [degree2E : Akita.Degree2 E]
    {Coord : Type}
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    {r : Akita.Key30.Row}
    (member : r ∈ Akita.Key30.rows)
    (flat : Akita.Table)
    (β : (g : Fin 3) → Module.Basis (Fin ((Akita.Key30.root r).planes g)) Akita.Zq E)
    (key : (g : Fin 3) → Akita.Key ((Akita.Key30.root r).shape g))
    (openKey : (Akita.Key30.root r).OpenKey)
    (block : (Akita.Key30.root r).Block → Coord)
    (β₁ : (g : Fin 2) → Module.Basis (Fin ((Akita.Key30.level₁ r).planes g)) Akita.Zq E)
    (key₁ : (g : Fin 2) → Akita.Key ((Akita.Key30.level₁ r).shape g))
    (openKey₁ : (Akita.Key30.level₁ r).OpenKey)
    (block₁ : (Akita.Key30.level₁ r).Block → Coord)
    (β₂ : Module.Basis (Fin (Akita.Key30.level₂ r).opening.planes) Akita.Zq E)
    (key₂ : Akita.Key (Akita.Key30.level₂ r).shape)
    (openKey₂ : Akita.OpeningKey (Akita.Key30.level₂ r).opening)
    (block₂ : Fin (Akita.Key30.level₂ r).shape.blocks → Coord)
    (β₃ : Module.Basis (Fin Akita.Key30.level₃.opening.planes) Akita.Zq E)
    (key₃ : Akita.Key Akita.Key30.level₃.shape)
    (openKey₃ : Akita.OpeningKey Akita.Key30.level₃.opening)
    (block₃ : Fin Akita.Key30.level₃.shape.blocks → Coord)
    (key₄ : Akita.Key Akita.Key30.level₄.shape)
    (openKey₄ : Akita.OpeningKey Akita.Key30.level₄.opening)
    (block₄ : Fin Akita.Key30.level₄.shape.blocks → Coord)
    (key₅ : Akita.Key Akita.Key30.level₅.shape)
    (openKey₅ : Akita.OpeningKey Akita.Key30.level₅.opening)
    (block₅ : Fin Akita.Key30.level₅.shape.blocks → Coord)
    (key₆ : Akita.Key Akita.Key30.level₆.shape)
    (openKey₆ : Akita.OpeningKey Akita.Key30.level₆.opening)
    (block₆ : Fin Akita.Key30.level₆.shape.blocks → Coord)
    (matrix : Matrix (Fin Akita.Key30.terminal.innerRank)
          (Fin Akita.Key30.terminal.blockLength × Fin Akita.Key30.terminal.sourceDigits)
          (Akita.Rq Akita.Key30.terminal.innerDegree))
    (blockL : Fin Akita.Key30.terminal.blocks → Coord)
    {total stage : ℕ}
    [nonemptyCoord : Nonempty Coord]
    (shared : (Akita.Key30.noteTower member flat β key openKey block β₁ key₁ openKey₁ block₁ β₂ key₂ openKey₂
          block₂ β₃ key₃ openKey₃ block₃ key₄ openKey₄ block₄ key₅ openKey₅ block₅ key₆ openKey₆ block₆
          matrix blockL total stage).Shared)
    (drawn : (Akita.Key30.noteTower member flat β key openKey block β₁ key₁ openKey₁ block₁ β₂ key₂ openKey₂
          block₂ β₃ key₃ openKey₃ block₃ key₄ openKey₄ block₄ key₅ openKey₅ block₅ key₆ openKey₆ block₆
          matrix blockL total stage).Drawn shared Akita.Extract.Page.read 0)
    (table : Akita.Extract.Address (Akita.Key30.Wire r E) total → Option Coord → E)
    (worked : ∀ (input : Akita.Extract.Address (Akita.Key30.Wire r E) total),
          table input none ∈ Akita.Extract.workZeros E 57)
    {tables : Fin 3 → Akita.Table}
    (live : ∀ (g : Fin 3) (cell : ℕ), 2 ^ ![30, r.vars, 28] g ≤ cell → tables g cell = 0)
    {places : Fin 3 → List E}
    (lengths : ∀ (g : Fin 3), (places g).length = ![30, r.vars, 28] g)
    (fit : Akita.Key30.Fit member flat β key openKey (Akita.Key30.rootDecode E r) β₁ key₁ openKey₁
        (Akita.Key30.decode₁ E r) β₂ key₂ openKey₂
        (Akita.Extract.draw (Akita.SparseFamily (Akita.Key30.level₂ r).opening.carrierDegree 31 10) fun _ => True)
        β₃ key₃ openKey₃
        (Akita.Extract.draw (Akita.SparseFamily Akita.Key30.level₃.opening.carrierDegree 31 10) fun _ => True)
        key₄ openKey₄
        (Akita.Extract.draw (Akita.SparseFamily Akita.Key30.level₄.shape.innerDegree 31 0) fun _ => True)
        key₅ openKey₅
        (Akita.Extract.draw (Akita.SparseFamily Akita.Key30.level₅.shape.innerDegree 31 11)
          (Akita.TrigTable.codeAccepts Akita.Key30.level₅.shape.innerDegree Akita.Key30.norm₅.operator))
        key₆ openKey₆
        (Akita.Extract.draw (Akita.SparseFamily Akita.Key30.level₆.shape.innerDegree 31 11)
          (Akita.TrigTable.codeAccepts Akita.Key30.level₆.shape.innerDegree Akita.Key30.norm₆.operator))
        (Akita.Extract.draw (Akita.SparseFamily Akita.Key30.terminal.innerDegree 31 11)
          (Akita.TrigTable.codeAccepts Akita.Key30.terminal.innerDegree Akita.Key30.normTerminal.operator))
        tables) :
    Finite (Akita.Key30.Wire r E) ∧
      ∃ proof : ℕ → Akita.Key30.Wire r E,
        (RandomOracle.OracleComp.done proof :
            RandomOracle.OracleComp (Akita.Extract.Address (Akita.Key30.Wire r E) total) (Option Coord → E)
              (ℕ → Akita.Key30.Wire r E)).QueryBound 0 ∧
        ((proof 0).data.commit = fun g => (Akita.commit (key g) (tables g)).1) ∧
        (proof 0).data.place = places ∧
        ((proof 0).data.value = fun g => Lib.mle (places g) (Akita.lifted E (tables g))) ∧
        (Akita.Key30.noteTower member flat β key openKey block β₁ key₁ openKey₁ block₁ β₂ key₂ openKey₂
          block₂ β₃ key₃ openKey₃ block₃ key₄ openKey₄ block₄ key₅ openKey₅ block₅ key₆ openKey₆ block₆
          matrix blockL total stage).accepts shared Akita.Extract.Page.read
          (((Akita.Key30.noteTower member flat β key openKey block β₁ key₁ openKey₁ block₁ β₂ key₂ openKey₂
            block₂ β₃ key₃ openKey₃ block₃ key₄ openKey₄ block₄ key₅ openKey₅ block₅ key₆ openKey₆ block₆
            matrix blockL total stage).body shared Akita.Extract.Page.read 0
              ((Akita.Key30.noteTower member flat β key openKey block β₁ key₁ openKey₁ block₁ β₂ key₂ openKey₂
              block₂ β₃ key₃ openKey₃ block₃ key₄ openKey₄ block₄ key₅ openKey₅ block₅ key₆ openKey₆ block₆
              matrix blockL total stage).accepts shared Akita.Extract.Page.read)).protocol.final
            (RandomOracle.OracleComp.done proof) table) :=
  @NightstreamFPrime.Spec.Compression.Akita.Key30.noteTower_accepts E fieldE fintypeE decidableEqE algebraZqE
    degree2E Coord fintypeCoord decidableEqCoord r member flat β key openKey block β₁ key₁ openKey₁ block₁ β₂
    key₂ openKey₂ block₂ β₃ key₃ openKey₃ block₃ key₄ openKey₄ block₄ key₅ openKey₅ block₅ key₆ openKey₆ block₆
    matrix blockL total stage nonemptyCoord shared drawn table worked tables live places lengths fit

/-- Soundness of the final verifier with the Akita openings of
`crates/ns-compress/artifacts/key30.aks`, the file of the key of `2^19` blocks:
`Claims.Final.verifier_sound` in the game with Akita openings (Akita Theorem 10.17 with
Section 9). The claim is for each claim shape of the family and for the proof of zero steps, and
for the two rows of the file. The verifier is the Lean verifier `FinalKey30.finalAccepts`, not
the verifier of the pinned Akita code. For each prover with at most `Q` queries to the oracle,
the probability that the verifier accepts and no proof with witnesses satisfies `Verifies` is at
most the Nightstream bound `FinalKey30.shapesBound` (Bounds above): for each shape,
`(Q + 1)·(runCost + 1623435/(4096·q²) + (1458 + B_1 + B_2)/q²) +
(Q + 1)·(1469 + B_1 + B_2)/budget + breakChance`. Lean gives no bound on `breakChance`, the
probability that the extractor returns a solution of Module-SIS (`[MSIS]`). `drawn`: the
hypotheses `FinalKey30.Opening.Drawn` on the opening of each shape. `shared`: for each shape,
the matrices `A`, `B` and `D` of the root are the views of the setup prefix (Akita (35)), so Lean
proves the split (132) of the public weight of the root. -/
theorem key30_verifier_sound
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
          Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (step : AppState → AppWitness → AppState)
    (cap : ℕ)
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
    (family : ι → FinalKey30.Opening logicalWidth publicFits Wire Coord total)
    (shared : ∀ (index : ι), (family index).tower.Shared)
    (prover : RandomOracle.OracleComp (Akita.Extract.Address Wire total) (Option Coord → K) (ℕ → Wire))
    [finiteWire : Finite Wire]
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    (complete : StepExtend.Complete exports)
    (rows : ∀ (circuit : Fin 2),
          (setup.rows (↑circuit : ℕ)).matrices = (FinalRows.packageRows exports complete circuit).matrices)
    (blocks : setup.keyBlocks = 524288)
    (coords : 40 ≤ Fintype.card Coord)
    (drawn : ∀ (index : ι),
          FinalKey30.Opening.Drawn setup complete.packageWidth step cap control read head (family index)
            (shared index))
    {queries : ℕ}
    (bounded : prover.QueryBound queries)
    {budget : ℕ}
    (positive : 0 < budget) :
    (𝔼 table,
        if
            FinalKey30.finalAccepts setup complete.packageWidth step cap control read head kind family
                shared (FinalKey20.played prover table) ∧
              ¬∃ proof,
                  RunCompleteness.Verifies setup.ajtai setup.circuitKeys step complete.packageWidth cap
                    control (head (FinalKey20.played prover table).first).statement proof then
          1
        else 0) ≤
      FinalKey30.shapesBound setup complete.packageWidth step cap control read head family shared prover
        queries budget :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey30.verifier_sound_drawn logicalWidth publicFits mode setup
    step cap control Wire Coord total read head fintypeCoord decidableEqCoord nonemptyCoord ι fintypeι kind
    family shared prover finiteWire relation cubeFits exports complete rows blocks coords drawn queries
    bounded budget positive

/-- The final verifier with the Akita openings of `crates/ns-compress/artifacts/key30.aks`
against a false history: `Claims.Final.history_sound` in the game with Akita openings (Akita
Theorem 10.17 with Section 9). The model and the verifier are those of `key30_verifier_sound`.
For each prover with at most `Q` queries, the probability that the verifier accepts
`(i, z0, zi)` and no `i` valid steps lead from `z0` to `zi` is at most `FinalKey30.shapesBound`
plus `IvcSecurity.walkBound` under `[HN-A1]`. `depthBound`: the prover controls the iteration of
its statement, and the claim is for provers whose accepted statement has an iteration of at most
`depth` for each table, as in `Claims.Final.history_sound`. Lean gives no bound on the term
`breakChance` of `shapesBound` (`[MSIS]`). -/
theorem key30_history_sound
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
          Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (step : AppState → AppWitness → AppState)
    (cap : ℕ)
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
    (family : ι → FinalKey30.Opening logicalWidth publicFits Wire Coord total)
    (shared : ∀ (index : ι), (family index).tower.Shared)
    (prover : RandomOracle.OracleComp (Akita.Extract.Address Wire total) (Option Coord → K) (ℕ → Wire))
    [finiteWire : Finite Wire]
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    (complete : StepExtend.Complete exports)
    (budget : ℕ)
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
    (rows : ∀ (circuit : Fin 2),
          (setup.rows (↑circuit : ℕ)).matrices = (FinalRows.packageRows exports complete circuit).matrices)
    (keysShaped : ∀ (key : Fin 2), List.length (setup.circuitKeys key).digest = 4)
    (blocks : setup.keyBlocks = 524288)
    (coords : 40 ≤ Fintype.card Coord)
    (drawn : ∀ (index : ι),
          FinalKey30.Opening.Drawn setup complete.packageWidth step cap control read head (family index)
            (shared index))
    {queries : ℕ}
    (bounded : prover.QueryBound queries)
    (positive : 0 < budget)
    (admitted : StageAdmitted
          (HyperNova.ReverseStages.Stage.start
            (IvcSecurity.walkAdversary
              (FinalKey30.adversary setup complete.packageWidth step cap control read head kind family shared
                prover budget))))
    (depth : ℕ)
    (depthBound : ∀ (table : Akita.Extract.Address Wire total → Option Coord → K),
          FinalKey30.finalAccepts setup complete.packageWidth step cap control read head kind family shared
              (FinalKey20.played prover table) →
            (head (FinalKey20.played prover table).first).statement.iteration ≤ depth) :
    (((FinalKey30.adversary setup complete.packageWidth step cap control read head kind family shared prover
                  budget).tape.toOuterMeasure :
            Set
                (FinalKey30.adversary setup complete.packageWidth step cap control read head kind family
                    shared prover budget).Tape →
              ENNReal)
          {sample |
            FinalKey30.FalseHistory setup complete.packageWidth step cap control read head kind family
              shared prover exports.program sample.1}).toReal ≤
      FinalKey30.shapesBound setup complete.packageWidth step cap control read head family shared prover
          queries budget +
        IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed
          (FinalKey30.adversary setup complete.packageWidth step cap control read head kind family shared
            prover budget)
          admitted depth :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey30.history_sound_drawn logicalWidth publicFits mode setup
    step cap control Wire Coord total read head fintypeCoord decidableEqCoord nonemptyCoord ι fintypeι kind
    family shared prover finiteWire relation cubeFits exports complete budget Admitted StageAdmitted
    Efficient error assumption closed rows keysShaped blocks coords drawn queries bounded positive
    admitted depth depthBound

/-- The knowledge form of `key30_history_sound`: `Claims.Final.history_knowledge` in the game
with the Akita openings of `crates/ns-compress/artifacts/key30.aks`. The probability that the
final verifier accepts and the witnesses that the reverse extractor returns are not a history of
the accepted statement is at most `FinalKey30.shapesBound` plus `IvcSecurity.walkBound` under
`[HN-A1]`. `depthBound` is that of `key30_history_sound`. Lean gives no bound on the term
`breakChance` of `shapesBound` (`[MSIS]`). -/
theorem key30_history_knowledge
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
          Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (step : AppState → AppWitness → AppState)
    (cap : ℕ)
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
    (family : ι → FinalKey30.Opening logicalWidth publicFits Wire Coord total)
    (shared : ∀ (index : ι), (family index).tower.Shared)
    (prover : RandomOracle.OracleComp (Akita.Extract.Address Wire total) (Option Coord → K) (ℕ → Wire))
    [finiteWire : Finite Wire]
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    (complete : StepExtend.Complete exports)
    (budget : ℕ)
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
    (rows : ∀ (circuit : Fin 2),
          (setup.rows (↑circuit : ℕ)).matrices = (FinalRows.packageRows exports complete circuit).matrices)
    (keysShaped : ∀ (key : Fin 2), List.length (setup.circuitKeys key).digest = 4)
    (blocks : setup.keyBlocks = 524288)
    (coords : 40 ≤ Fintype.card Coord)
    (drawn : ∀ (index : ι),
          FinalKey30.Opening.Drawn setup complete.packageWidth step cap control read head (family index)
            (shared index))
    {queries : ℕ}
    (bounded : prover.QueryBound queries)
    (positive : 0 < budget)
    (admitted : StageAdmitted
          (HyperNova.ReverseStages.Stage.start
            (IvcSecurity.walkAdversary
              (FinalKey30.adversary setup complete.packageWidth step cap control read head kind family shared
                prover budget))))
    (depth : ℕ)
    (depthBound : ∀ (table : Akita.Extract.Address Wire total → Option Coord → K),
          FinalKey30.finalAccepts setup complete.packageWidth step cap control read head kind family shared
              (FinalKey20.played prover table) →
            (head (FinalKey20.played prover table).first).statement.iteration ≤ depth) :
    (((↑(HyperNova.ReverseStages.reverseStages assumption closed
                      (IvcSecurity.walkAdversary
                        (FinalKey30.adversary setup complete.packageWidth step cap control read head kind
                          family shared prover budget))
                      admitted depth) :
                  HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                    (IvcWalk.Answer logicalWidth publicFits)).tape.toOuterMeasure :
            Set
                (↑(HyperNova.ReverseStages.reverseStages assumption closed
                        (IvcSecurity.walkAdversary
                          (FinalKey30.adversary setup complete.packageWidth step cap control read head
                            kind family shared prover budget))
                        admitted depth) :
                    HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                      (IvcWalk.Answer logicalWidth publicFits)).Tape →
              ENNReal)
          {tape |
            FinalKey30.AdviceFailure setup step cap control read head kind family shared prover exports
              complete budget assumption closed admitted depth tape}).toReal ≤
      FinalKey30.shapesBound setup complete.packageWidth step cap control read head family shared prover
          queries budget +
        IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed
          (FinalKey30.adversary setup complete.packageWidth step cap control read head kind family shared
            prover budget)
          admitted depth :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey30.history_knowledge_drawn logicalWidth publicFits mode setup
    step cap control Wire Coord total read head fintypeCoord decidableEqCoord nonemptyCoord ι fintypeι kind
    family shared prover finiteWire relation cubeFits exports complete budget Admitted StageAdmitted
    Efficient error assumption closed rows keysShaped blocks coords drawn queries bounded positive
    admitted depth depthBound

/-- The memory verifier with the Akita openings of `crates/ns-compress/artifacts/key30.aks`
against a false memory statement: `Claims.Final.memory_sound` in the game with Akita openings
(Akita Theorem 10.17 with Section 9). The model and the verifier are those of
`key30_verifier_sound`. The prover names a memory statement with the law `claim` for each table
of the oracle. Over the walk's `S_max · N` stages, the probability that the verifier accepts
`(i, z0, zi)`, the checks of SPEC.md Algorithm 8 hold, and the decoded history does not attest
the memory statement is at most `FinalKey30.shapesBound` plus the terms of
`Claims.Nebula.memory_sound`: `IvcSecurity.walkBound` under `[HN-A1]`, the state-link and chain
collision masses, and the memory-round error. The checks of Algorithm 8 bound the iteration by
`S_max · N`, so no premise bounds it. Lean gives no bound on the term `breakChance` of
`shapesBound` (`[MSIS]`). -/
theorem key30_memory_sound
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
          Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (step : AppState → AppWitness → AppState)
    (cap : ℕ)
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
    (family : ι → FinalKey30.Opening logicalWidth publicFits Wire Coord total)
    (shared : ∀ (index : ι), (family index).tower.Shared)
    (prover : RandomOracle.OracleComp (Akita.Extract.Address Wire total)
          (Option Coord → K) (ℕ → Wire))
    [finiteWire : Finite Wire]
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {cubeFits : Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth ≤
          2 ^ cubeVariables}
    (exports : TerminalBridge.Exports relation setup.ajtai setup.circuitKeys step cubeFits)
    (complete : StepExtend.Complete exports)
    (budget : ℕ)
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
    {p : Nebula.Plan}
    (app : Nebula.PortApp p.toGeometry)
    (claim : (Akita.Extract.Address Wire total → Option Coord → K) →
          PMF (Nebula.EndToEnd.MemoryStatement app.width))
    (rows : ∀ (circuit : Fin 2),
          (setup.rows (↑circuit : ℕ)).matrices =
            (FinalRows.packageRows exports complete circuit).matrices)
    (keysShaped : ∀ (key : Fin 2), List.length (setup.circuitKeys key).digest = 4)
    (blocks : setup.keyBlocks = 524288)
    (coords : 40 ≤ Fintype.card Coord)
    (drawn : ∀ (index : ι),
          FinalKey30.Opening.Drawn setup complete.packageWidth step cap control read head
            (family index) (shared index))
    {queries : ℕ}
    (bounded : prover.QueryBound queries)
    (positive : 0 < budget)
    (admitted : StageAdmitted
          (HyperNova.ReverseStages.Stage.start
            (IvcSecurity.walkAdversary
              (FinalMemory.adversary head kind
                (fun index =>
                  FinalKey30.Opening.accepts setup complete.packageWidth step cap control read
                    head (family index) (shared index))
                (fun index =>
                  FinalKey30.Opening.proof setup complete.packageWidth step cap control read head
                    (family index) (shared index) prover)
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
                            FinalKey30.Opening.accepts setup complete.packageWidth step cap
                              control read head (family index) (shared index))
                          (fun index =>
                            FinalKey30.Opening.proof setup complete.packageWidth step cap
                              control read head (family index) (shared index) prover)
                          prover budget app claim))
                      admitted (p.sMax * p.n)) :
                  HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                    (IvcWalk.Answer logicalWidth publicFits)).tape.toOuterMeasure :
            Set
                (↑(HyperNova.ReverseStages.reverseStages assumption closed
                        (IvcSecurity.walkAdversary
                          (FinalMemory.adversary head kind
                            (fun index =>
                              FinalKey30.Opening.accepts setup complete.packageWidth step cap
                                control read head (family index) (shared index))
                            (fun index =>
                              FinalKey30.Opening.proof setup complete.packageWidth step cap
                                control read head (family index) (shared index) prover)
                            prover budget app claim))
                        admitted (p.sMax * p.n)) :
                    HyperNova.ReverseStages.Stage (IvcWalk.Visit logicalWidth publicFits)
                      (IvcWalk.Answer logicalWidth publicFits)).Tape →
              ENNReal)
          {tape |
            FinalMemory.FalseMemory setup step head kind
              (fun index =>
                FinalKey30.Opening.accepts setup complete.packageWidth step cap control read
                  head (family index) (shared index))
              prover app exports
              (FinalMemory.tapeOf setup step head kind
                    (fun index =>
                      FinalKey30.Opening.accepts setup complete.packageWidth step cap control
                        read head (family index) (shared index))
                    (fun index =>
                      FinalKey30.Opening.proof setup complete.packageWidth step cap control
                        read head (family index) (shared index) prover)
                    prover budget app exports assumption closed claim admitted (p.sMax * p.n)
                    tape).1.1
              (FinalMemory.sampleOf setup step head kind
                (fun index =>
                  FinalKey30.Opening.accepts setup complete.packageWidth step cap control read
                    head (family index) (shared index))
                (fun index =>
                  FinalKey30.Opening.proof setup complete.packageWidth step cap control read
                    head (family index) (shared index) prover)
                prover budget app exports assumption closed claim admitted (p.sMax * p.n)
                tape)}).toReal ≤
      FinalKey30.shapesBound setup complete.packageWidth step cap control read head family
          shared prover queries budget +
        (IvcSecurity.walkBound setup.ajtai setup.circuitKeys exports assumption closed
                (FinalMemory.adversary head kind
                  (fun index =>
                    FinalKey30.Opening.accepts setup complete.packageWidth step cap control
                      read head (family index) (shared index))
                  (fun index =>
                    FinalKey30.Opening.proof setup complete.packageWidth step cap control read
                      head (family index) (shared index) prover)
                  prover budget app claim)
                admitted (p.sMax * p.n) +
              (((NebulaSecurity.ClaimedStage.law app
                        (FinalMemory.claimed setup step head kind
                          (fun index =>
                            FinalKey30.Opening.accepts setup complete.packageWidth step cap
                              control read head (family index) (shared index))
                          (fun index =>
                            FinalKey30.Opening.proof setup complete.packageWidth step cap
                              control read head (family index) (shared index) prover)
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
                          FinalKey30.Opening.accepts setup complete.packageWidth step cap
                            control read head (family index) (shared index))
                        (fun index =>
                          FinalKey30.Opening.proof setup complete.packageWidth step cap
                            control read head (family index) (shared index) prover)
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
                FinalKey30.Opening.accepts setup complete.packageWidth step cap control read
                  head (family index) (shared index))
              (fun index =>
                FinalKey30.Opening.proof setup complete.packageWidth step cap control read
                  head (family index) (shared index) prover)
              prover budget app exports assumption closed claim admitted (p.sMax * p.n))) :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey30.memory_sound_drawn logicalWidth publicFits
    mode setup step cap control Wire Coord total read head fintypeCoord decidableEqCoord
    nonemptyCoord ι fintypeι kind family shared prover finiteWire relation cubeFits exports
    complete budget Admitted StageAdmitted Efficient error assumption closed p app claim rows
    keysShaped blocks coords drawn queries bounded positive admitted valid programEq memoryError
    round

/-- The numbers of the three opening claims of the final verifier for the key of
`crates/ns-compress/artifacts/key30.aks` and the row `r` of the file. For a setup with
`FinalKey30.Use`, the point of the key polynomial has 30 coordinates, the point of `V` has 28
coordinates, the span has at most 26 variables, and the point of `W` has the count of variables
of the row, 27 or 26. These are the sizes of the tables of the groups of `Key30.root`. -/
theorem key30_points
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
          Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (shape : FinalVerifier.Claims logicalWidth publicFits)
    {r : Akita.Key30.Row}
    (use : FinalKey30.Use setup shape r) :
    (setup.params shape).laneVars + (setup.params shape).keyBlockVars + (setup.params shape).rowVars = 30 ∧
      (setup.params shape).spanVars + (setup.verifier shape).columnVars + (setup.verifier shape).zeros =
          28 ∧
        (setup.params shape).spanVars ≤ 26 ∧ (setup.params shape).witnessVars = r.vars :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey30.key30_points logicalWidth publicFits mode setup shape r
    use

/-- The instances of Module-SIS of the opening of the two rows of
`crates/ns-compress/artifacts/key30.aks` (Akita Table 3 and Appendix G). Each of the 54 matrix
views of each row is below one of the 18 instances `(d, n, m, p, η)` of the list: the same
degree, rank and norm, and a width and a bound that are not larger. The list has no instance
without a view, and no instance below another instance. A Euclidean instance has the power of
two that the file prices. `tests/AkitaSchedules.lean` compares the list with the file. Lean does
not prove that an instance is hard (`[MSIS]`). -/
theorem key30_instance_list :
    (∀ r ∈ Akita.Key30.rows, Akita.Covered Akita.Key30.instances (Akita.Key30.views r)) ∧
      List.map Akita.Setup.Instance.key Akita.Key30.instances =
          [(32, 1, 128, false, 1), (64, 1, 1024, false, 1), (64, 2, 77, false, 63),
            (64, 3, 616, false, 63), (64, 8, 256, true, 2 ^ 46), (64, 8, 512, true, 2 ^ 45),
            (128, 1, 416, false, 31), (128, 1, 4224, false, 15), (128, 1, 11264, false, 7),
            (128, 4, 512, false, 126852), (128, 5, 512, false, 835380), (128, 6, 2048, false, 835380),
            (256, 1, 8192, false, 15), (256, 1, 28160, false, 7), (512, 2, 1024, false, 8126340),
            (512, 2, 2048, false, 507780), (512, 3, 5120, false, 2080374660),
            (2048, 1, 4096, false, 16642998148)] ∧
        (∀ r ∈ Akita.Key30.rows, (Akita.Key30.views r).length = 54) ∧
          (∀ I ∈ Akita.Key30.instances,
              ∃ r ∈ Akita.Key30.rows,
                ∃ v ∈ Akita.Key30.views r,
                  v.Below I ∧
                    v.cols = I.cols ∧ bif I.euclidean then I.bound < 2 * v.bound else v.bound = I.bound) ∧
            List.Pairwise (fun I J => ¬I.Below J ∧ ¬J.Below I) Akita.Key30.instances :=
  @NightstreamFPrime.Spec.Compression.Akita.Key30.instances_exact

/-- The list `Key30.instances` is complete for the term `breakChance` of
`key30_verifier_sound` (Akita Lemma 10.20, Table 3 and Lemma 9.1). `out` is an output of the
extractor that is a solution of Module-SIS (`solved`): for the key commitment, or a witness of
`R_bind` of the opening. A witness of `R_bind` of the level 1 has also the solution of Akita
Lemma 9.1 for the commitment of the setup prefix. Then a matrix of the opening has a solution
that gives a solution of an instance of the list, at the bound that the file prices. The radius
of the reduction for the key commitment is `η_A = 16642998148` of the key group. Lean does not
prove that an instance is hard (`[MSIS]`). -/
theorem key30_instances
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
          Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (packageWidths : Fin 2 → ℕ)
    (step : AppState → AppWitness → AppState)
    (cap : ℕ)
    (control : AppState → AppWitness → Fin 2)
    {Wire : Type}
    {Coord : Type}
    {total : ℕ}
    (read : Akita.Extract.Reader Wire)
    (head : Wire → FinalAkita.First logicalWidth publicFits)
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    {o : FinalKey30.Opening logicalWidth publicFits Wire Coord total}
    {shared : o.tower.Shared}
    {out : Akita.Extract.Opened (Akita.Key30.root o.row).Weak
            (Akita.Extract.Opened (Akita.Key30.level₁ o.row).Weak
              (Akita.Extract.Opened (Akita.Weak (Akita.Key30.level₂ o.row).opening)
                (Akita.Extract.Opened (Akita.Weak Akita.Key30.level₃.opening)
                  (Akita.Extract.Opened (Akita.Weak Akita.Key30.level₄.opening)
                    (Akita.Extract.Opened (Akita.Weak Akita.Key30.level₅.opening)
                      (Akita.Extract.Opened (Akita.Weak Akita.Key30.level₆.opening)
                        (Akita.Extract.Prefix Wire (Option Coord → K) total ×
                          Option (Akita.TerminalFork Akita.Key30.terminal)))))))) ⊕
          (Akita.Batch.LevelBreak o.tower.key o.tower.openKey fun g =>
              2 * (2 * Akita.Key30.mass g) *
                ((Akita.Key30.root o.row).fold g).sourceBound ((Akita.Key30.root o.row).shape g)) ⊕
            (Akita.Batch.LevelBreak o.tower.key₁ o.tower.openKey₁ fun g =>
                2 * (2 * Akita.Key30.mass₁ g) *
                  ((Akita.Key30.level₁ o.row).fold g).sourceBound ((Akita.Key30.level₁ o.row).shape g)) ⊕
              Akita.Extract.LevelBreak o.tower.key₂ o.tower.openKey₂
                  (2 * (2 * 51) *
                    (Akita.Key30.level₂ o.row).fold.sourceBound (Akita.Key30.level₂ o.row).shape) ⊕
                Akita.Extract.LevelBreak o.tower.key₃ o.tower.openKey₃
                    (2 * (2 * 51) * Akita.Key30.level₃.fold.sourceBound Akita.Key30.level₃.shape) ⊕
                  Akita.Extract.Route.Break o.tower.key₄ o.tower.openKey₄ 31
                      (Akita.Key30.level₄.fold.sourceBound Akita.Key30.level₄.shape)
                      Akita.Extract.Route.coefficient ⊕
                    Akita.Extract.Route.Break o.tower.key₅ o.tower.openKey₅ 53
                        (Akita.Key30.level₅.fold.sourceBound Akita.Key30.level₅.shape)
                        (Akita.Extract.Route.euclidean Akita.Key30.norm₅.operator Akita.Key30.norm₅.limit) ⊕
                      Akita.Extract.Route.Break o.tower.key₆ o.tower.openKey₆ 53
                          (Akita.Key30.level₆.fold.sourceBound Akita.Key30.level₆.shape)
                          (Akita.Extract.Route.euclidean Akita.Key30.norm₆.operator Akita.Key30.norm₆.limit) ⊕
                        Option
                          ((Akita.Extract.Route.euclidean Akita.Key30.normTerminal.operator
                                Akita.Key30.normTerminal.limit).Kernel
                            o.tower.terminal.matrix o.tower.terminal.weight o.tower.terminal.bound)}
    (solved : FinalAkita.Solved
          (FinalKey30.Opening.body setup packageWidths step cap control read head o shared) o.tower.key
          o.honest (((Akita.Key30.root o.row).fold 0).innerBound ((Akita.Key30.root o.row).shape 0)) out) :
    Akita.Key30.Reduced Akita.Key30.instances o.tower.key o.tower.openKey o.tower.key₁ o.tower.openKey₁
      o.tower.key₂ o.tower.openKey₂ o.tower.key₃ o.tower.openKey₃ o.tower.key₄ o.tower.openKey₄
      o.tower.key₅ o.tower.openKey₅ o.tower.key₆ o.tower.openKey₆ o.tower.terminal.matrix :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey30.Opening.reduced logicalWidth publicFits mode setup
    packageWidths step cap control Wire Coord total read head fintypeCoord decidableEqCoord o shared out
    solved


/-- The join of the list `Key30.views` with the shared setup (Akita Section 4.3 with (35),
Table 3, Lemma 10.20 and Lemma 9.1). `keys`: each matrix of the opening is the view of one flat
setup vector `S` with the degree, the rank and the width of its instance in `views`, as the code
reads it from one stream from the offset 0. `shared` and `solved` are those of
`key30_instances`; at `S` the setup prefix `flat`, `keys` gives `shared`
(`Spec.Compression.Akita.Key30.Tower.shared_of_views`). Then an instance `I` of the 54 views of
the row has a solution of Module-SIS for the view `M_I(S)` of `S` itself, at the bound of the
witness of its level (`Setup.Solved`). This is the event of the algorithm `B` of Akita
Lemma 10.20 for the inventory `views` (`Spec.Compression.Akita.Key30.shared_setup`, with the
`2^23` cells of the setup prefix). Lean does not prove that an instance is hard (`[MSIS]`). -/
theorem key30_instances_shared
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
          Folding.PiCCS.PaperJoint.RingCarrierLayout.carrierWidth logicalWidth}
    {mode : FinalVerifier.Mode}
    (setup : FinalVerifier.Setup mode logicalWidth publicFits)
    (packageWidths : Fin 2 → ℕ)
    (step : AppState → AppWitness → AppState)
    (cap : ℕ)
    (control : AppState → AppWitness → Fin 2)
    {Wire : Type}
    {Coord : Type}
    {total : ℕ}
    (read : Akita.Extract.Reader Wire)
    (head : Wire → FinalAkita.First logicalWidth publicFits)
    [fintypeCoord : Fintype Coord]
    [decidableEqCoord : DecidableEq Coord]
    {o : FinalKey30.Opening logicalWidth publicFits Wire Coord total}
    {shared : o.tower.Shared}
    {N : ℕ}
    {S : Fin N → Akita.Zq}
    (keys : Akita.Key30.Shared S o.tower.key o.tower.openKey o.tower.key₁ o.tower.openKey₁
          o.tower.key₂ o.tower.openKey₂ o.tower.key₃ o.tower.openKey₃ o.tower.key₄ o.tower.openKey₄
          o.tower.key₅ o.tower.openKey₅ o.tower.key₆ o.tower.openKey₆ o.tower.terminal.matrix)
    {out : Akita.Extract.Opened (Akita.Key30.root o.row).Weak
            (Akita.Extract.Opened (Akita.Key30.level₁ o.row).Weak
              (Akita.Extract.Opened (Akita.Weak (Akita.Key30.level₂ o.row).opening)
                (Akita.Extract.Opened (Akita.Weak Akita.Key30.level₃.opening)
                  (Akita.Extract.Opened (Akita.Weak Akita.Key30.level₄.opening)
                    (Akita.Extract.Opened (Akita.Weak Akita.Key30.level₅.opening)
                      (Akita.Extract.Opened (Akita.Weak Akita.Key30.level₆.opening)
                        (Akita.Extract.Prefix Wire (Option Coord → K) total ×
                          Option (Akita.TerminalFork Akita.Key30.terminal)))))))) ⊕
          (Akita.Batch.LevelBreak o.tower.key o.tower.openKey fun g =>
              2 * (2 * Akita.Key30.mass g) *
                ((Akita.Key30.root o.row).fold g).sourceBound ((Akita.Key30.root o.row).shape g)) ⊕
            (Akita.Batch.LevelBreak o.tower.key₁ o.tower.openKey₁ fun g =>
                2 * (2 * Akita.Key30.mass₁ g) *
                  ((Akita.Key30.level₁ o.row).fold g).sourceBound ((Akita.Key30.level₁ o.row).shape g)) ⊕
              Akita.Extract.LevelBreak o.tower.key₂ o.tower.openKey₂
                  (2 * (2 * 51) *
                    (Akita.Key30.level₂ o.row).fold.sourceBound (Akita.Key30.level₂ o.row).shape) ⊕
                Akita.Extract.LevelBreak o.tower.key₃ o.tower.openKey₃
                    (2 * (2 * 51) * Akita.Key30.level₃.fold.sourceBound Akita.Key30.level₃.shape) ⊕
                  Akita.Extract.Route.Break o.tower.key₄ o.tower.openKey₄ 31
                      (Akita.Key30.level₄.fold.sourceBound Akita.Key30.level₄.shape)
                      Akita.Extract.Route.coefficient ⊕
                    Akita.Extract.Route.Break o.tower.key₅ o.tower.openKey₅ 53
                        (Akita.Key30.level₅.fold.sourceBound Akita.Key30.level₅.shape)
                        (Akita.Extract.Route.euclidean Akita.Key30.norm₅.operator Akita.Key30.norm₅.limit) ⊕
                      Akita.Extract.Route.Break o.tower.key₆ o.tower.openKey₆ 53
                          (Akita.Key30.level₆.fold.sourceBound Akita.Key30.level₆.shape)
                          (Akita.Extract.Route.euclidean Akita.Key30.norm₆.operator Akita.Key30.norm₆.limit) ⊕
                        Option
                          ((Akita.Extract.Route.euclidean Akita.Key30.normTerminal.operator
                                Akita.Key30.normTerminal.limit).Kernel
                            o.tower.terminal.matrix o.tower.terminal.weight o.tower.terminal.bound)}
    (solved : FinalAkita.Solved
          (FinalKey30.Opening.body setup packageWidths step cap control read head o shared) o.tower.key
          o.honest (((Akita.Key30.root o.row).fold 0).innerBound ((Akita.Key30.root o.row).shape 0)) out) :
    Akita.Setup.Solved (Akita.Key30.views o.row) S :=
  @NightstreamFPrime.Export.BoundedDepth.FinalKey30.Opening.reduced_shared logicalWidth publicFits mode
    setup packageWidths step cap control Wire Coord total read head fintypeCoord decidableEqCoord o shared N
    S keys out solved

end NightstreamFPrime.Claims.AkitaKey30
