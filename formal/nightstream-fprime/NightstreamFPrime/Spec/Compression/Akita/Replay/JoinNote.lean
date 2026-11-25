import NightstreamFPrime.Spec.Compression.Akita.Key20Accept
import NightstreamFPrime.Spec.Compression.Akita.Replay.LinkRoot
import NightstreamFPrime.Spec.Compression.Akita.Replay.LinkTerminal
import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeKey20

/-! The wire messages and the tower data of the model for the replay of `key20.aks` (Akita
revision `d50f700`; ePrint 2026/1983, revision of 2026-11-23, Figure 4 and Section 8.2;
`Key20.noteTower` of `Key20Accept.lean`).

- `rootClaimIn`, `claimIn₁`, `traceClaimIn`, `terminalStatementIn` and `responseIn` are the claims
  of the levels 0 to 5, the statement and the response of the terminal from the recorded values of
  a state of the replay, and `noteOf` is the `Key20.Note` of these values for one state.
- `rootClaim20`, `claim20₁` and `traceClaim20` are the claims of the levels 0 to 5 from the
  values of the replay: the claim of `RootLinked` (`verifyProof_key20_rootLink`), of `Tier.Linked`
  (`verifyProof_key20_packLink`) and of `Tier.TraceLinked` (`verifyProof_key20_traceLink`), each
  on the state before the `fused.end` of its level.
- `terminalStatement20` is `TerminalStatement.ofPoint` of `verifyProof_key20_terminal` on the state
  before the terminal, for the last blocks `tLast` of `t`; `response20` is the response `z` of the
  payload of the terminal.
- `note20` is the `Key20.Note` of these values, the commitments, values and points of the public
  statement, and the clear partials with the last block `eLast` of `e`.
- `basis20` is the basis `(1, u)` of `E` over `Z_q`, the basis `extBasis` of the replay
  (`basis20_apply`). `setup20` and `full20` are the setup prefixes of the folds and of the
  terminal of the replay.
- `tower20` is `Key20.noteTower` with the basis `basis20` and the matrices of the setup prefixes
  of the replay: `Key.ofSetup`, `OpeningKey.ofSetup` and `matrixOf`. The coordinates of the fold
  challenges are free. `tower20_drawn` gives `Tower.Drawn` from injective coordinates, the stage
  and `Kept`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib
open NightstreamFPrime.Spec.Compression.Akita

open Replayer

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]

section Note

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- The run of the relation of the root group `g`: the run at `(groupOrder 3).symm g`. -/
abbrev rootRun20 (g : Fin 3) : RhsRun :=
  groupRun (fun k => Key20.root.shape (groupOrder 3 k)) Key20.root.openRank Key20.root.chain
    (fun k => .root (groupOrder 3 k).val) ((groupOrder 3).symm g)

/-- The point of the public group `g`. -/
def rootPoint20 (g : Fin 3) : List Ext := ((pub.groups[g.val]?.map (·.point)).getD #[]).toList

/-- The claim of the root fold of `RootLinked` on the state `s`: the payloads of the commitments
of the public groups, the opening payload, the point of each group from the segments
`(r_pack, r_tail, r_pos, r_blk)` of the point of its public group, the weights `row_coefficients`
and the target `Σ_g ϑ_g·v_g`. -/
noncomputable def rootClaimIn (s : Replayer) : Key20.root.Claim E :=
  let ϑ := fun g : Fin 3 =>
    (((recordedIn s ⟨0, "row_coefficients", none⟩).getD g Ext.zero).into : E)
  let v := fun g : Fin 3 => (((pub.groups.flatMap (·.value)).getD g Ext.zero).into : E)
  let r := rootPoint20 pub
  let a := fun g => Nat.log 2 (Key20.root.opening g).stride
  let b := fun g => Nat.log 2 (Key20.root.opening g).carrierDegree
  let m := fun g => Nat.log 2 (Key20.root.shape g).blockLength
  let pack := fun g => ((r g).take (a g)).map (Ext.into (E := E))
  let tail := fun g => (((r g).drop (a g)).take (b g)).map (Ext.into (E := E))
  let position := fun g => (((r g).drop (a g + b g)).take (m g)).map (Ext.into (E := E))
  let block := fun g => ((r g).drop (a g + b g + m g)).map (Ext.into (E := E))
  ⟨fun g => payloadOf (Key20.root.shape g).chain _ _
      (commitmentRows ((pub.groups[g.val]?.map (·.commitment)).getD .empty)
        ((rootRun20 g).degree * (rootRun20 g).count)),
    payloadOf Key20.root.chain _ _ ((s.fields.get? ⟨0, "opening_payload", none⟩).getD #[]),
    fun g => Point.ofSegments (Key20.root.opening g) (block g) (position g) (pack g) (tail g),
    ϑ, ∑ g, ϑ g * v g⟩

/-- The claim of the root fold of `RootLinked`: `rootClaimIn` on the state before the `fused.end`
of the level 0. -/
noncomputable def rootClaim20 : Key20.root.Claim E :=
  rootClaimIn pub (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 0))

/-- The claim of the level 1 of `Tier.Linked` on the states `s` and `s'`: the payload of the words
`next_witness` of the level 0 and the opening payload of the level 1 of `s'`, the point of the
segments `(r_pack, r_tail, r_pos, r_blk)` of `fused.point` of the level 0 of `s` and its
`witness_evaluation`. -/
noncomputable def claimIn₁ (s s' : Replayer) : Extract.Claim Key20.level₁.opening E :=
  let r2 := (recordedIn s ⟨0, "fused.point", none⟩).toList
  let pack := (r2.take 4).map (Ext.into (E := E))
  let tail := ((r2.drop 4).take 6).map (Ext.into (E := E))
  let position := ((r2.drop (4 + 6)).take 7).map (Ext.into (E := E))
  let block := (r2.drop (4 + 6 + 7)).map (Ext.into (E := E))
  ⟨payloadOf Key20.level₁.shape.chain _ _ ((s'.fields.get? ⟨0, "next_witness", none⟩).getD #[]),
    payloadOf Key20.level₁.opening.chain _ _
      ((s'.fields.get? ⟨1, "opening_payload", none⟩).getD #[]),
    Point.ofSegments Key20.level₁.opening block position pack tail,
    (firstIn s ⟨0, "witness_evaluation", none⟩).into⟩

/-- The claim of the level 1 of `Tier.Linked`: `claimIn₁` on the states before the `fused.end` of
the levels 0 and 1. -/
noncomputable def claim20₁ : Extract.Claim Key20.level₁.opening E :=
  claimIn₁ (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 0))
    (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 1))

/-- The claim of the level `level + 1` (the level `L'`) of `Tier.TraceLinked` on the state `s`:
`TraceClaim.ofPoint` for the payload of the words `next_witness` of the level `level`, the opening
payload, the segments `(ρ_pack, ρ_pos, ρ_blk)` of the tensor point `ρ`, the factor `θ = A_η(ρ)`
and the final claim `ṽ`. -/
noncomputable def traceClaimIn (s : Replayer) (level : ℕ) (L' : Key20.Level) {m : ℕ}
    (quarter : 4 * m = L'.shape.innerDegree) : Extract.TraceClaim L'.opening E :=
  let r2 := (recordedIn s ⟨level, "fused.point", none⟩).toList
  let ρ := (recordedIn s ⟨level + 1, "tensor.sumcheck.point", none⟩).toList
  let η := firstIn s ⟨level + 1, "tensor.eta", none⟩
  Extract.TraceClaim.ofPoint quarter
    (payloadOf L'.shape.chain _ _ ((s.fields.get? ⟨level, "next_witness", none⟩).getD #[]))
    (payloadOf L'.opening.chain _ _
      ((s.fields.get? ⟨level + 1, "opening_payload", none⟩).getD #[]))
    ((blockPoint L'.shape ρ).map (Ext.into (E := E)))
    ((positionPoint L'.shape ρ).map (Ext.into (E := E)))
    ((packPoint L'.shape ρ).map (Ext.into (E := E)))
    (mle (ρ.map Ext.into) (tensorWeight (r2.tail.map Ext.into) η.into))
    (firstIn s ⟨level + 1, "tensor.final_claims", none⟩).into

/-- The claim of the level `level + 1` (the level `L'`) of `Tier.TraceLinked`: `traceClaimIn` on
the state before its `fused.end`. -/
noncomputable def traceClaim20 (level : ℕ) (L' : Key20.Level) {m : ℕ}
    (quarter : 4 * m = L'.shape.innerDegree) : Extract.TraceClaim L'.opening E :=
  traceClaimIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout (level + 1))) level L'
    quarter

/-- The state before the terminal: the last operation of the script. -/
abbrev terminalState20 : Replayer := stateBefore pub proof messages pub.ops (pub.ops.size - 1)

/-- The setup prefix of the fold of the level `ℓ` of `layout20`. -/
def setup20 (ℓ : Fin 6) : Array Fq := (pub.setupFor (fold20 ℓ).setupCount).getD #[]

/-- The setup prefix of the terminal. -/
def full20 : Array Fq := (pub.setupFor pub.layout.terminal.setupCount).getD #[]

/-- The statement of the terminal of `verifyProof_key20_terminal` on the state `s`, for the last
block `tLast` of `t`: `TerminalStatement.ofPoint` for `A_term` of the setup prefix, the segments
of the tensor point `ρ`, `θ = A_η(ρ)` and the final claim `ṽ`. -/
noncomputable def terminalStatementIn (s : Replayer) (tLast : ℕ → Rq Key20.terminal.innerDegree) :
    TerminalStatement Key20.terminal E :=
  let r2 := (recordedIn s ⟨5, "fused.point", none⟩).toList
  let ρ := (recordedIn s ⟨6, "tensor.sumcheck.point", none⟩).toList
  let η := firstIn s ⟨6, "tensor.eta", none⟩
  Extract.TerminalStatement.ofPoint (And.right (And.right (And.right (And.right Key20.quarters))))
    (matrixOf Key20.terminal (full20 pub))
    (stateOf Key20.terminal (terminalMessagesIn s 6) tLast)
    ((blockPoint Key20.terminal ρ).map (Ext.into (E := E)))
    ((positionPoint Key20.terminal ρ).map (Ext.into (E := E)))
    ((packPoint Key20.terminal ρ).map (Ext.into (E := E)))
    (mle (ρ.map Ext.into) (tensorWeight (r2.tail.map Ext.into) η.into))
    (firstIn s ⟨6, "tensor.final_claims", none⟩).into

/-- The statement of the terminal: `terminalStatementIn` on the state before the terminal. -/
noncomputable def terminalStatement20 (tLast : ℕ → Rq Key20.terminal.innerDegree) :
    TerminalStatement Key20.terminal E :=
  terminalStatementIn pub (terminalState20 pub proof messages) tLast

/-- The response `z` of the payload of the terminal on the state `s`, and 0 when the payload does
not decode. -/
def responseIn (s : Replayer) : Array ℤ :=
  (Golomb.decode pub.layout.terminal.params (terminalMessagesIn s 6).payload).getD #[]

/-- The response `z` of the payload of the terminal: `responseIn` on the state before the
terminal. -/
def response20 : Array ℤ := responseIn pub (terminalState20 pub proof messages)

/-- The data of a wire message of the model from one state `s` of the replay: the commitments,
values and points of the public statement, the claims of the levels 0 to 5 with their
commitments, and the state with the last block `tLast`, the statement, the clear partials with the
last block `eLast` and the response of the terminal, each on the recorded values of `s`. -/
noncomputable def noteOf (s : Replayer) (tLast : ℕ → Rq Key20.terminal.innerDegree)
    (eLast : Rq Key20.terminal.innerDegree) : Key20.Note E where
  commit := (rootClaimIn (E := E) pub s).payload
  value g := (((pub.groups.flatMap (·.value)).getD g Ext.zero).into : E)
  point g i := (((rootPoint20 pub g).getD i Ext.zero).into : E)
  claim := rootClaimIn pub s
  commit₁ := (claimIn₁ (E := E) s s).payload
  claim₁ := claimIn₁ s s
  commit₂ := (traceClaimIn (E := E) s 1 Key20.level₂ Key20.quarters.1).payload
  claim₂ := traceClaimIn s 1 Key20.level₂ Key20.quarters.1
  commit₃ := (traceClaimIn (E := E) s 2 Key20.level₃ Key20.quarters.2.1).payload
  claim₃ := traceClaimIn s 2 Key20.level₃ Key20.quarters.2.1
  commit₄ := (traceClaimIn (E := E) s 3 Key20.level₄ Key20.quarters.2.2.1).payload
  claim₄ := traceClaimIn s 3 Key20.level₄ Key20.quarters.2.2.1
  commit₅ := (traceClaimIn (E := E) s 4 Key20.level₅ Key20.quarters.2.2.2.1).payload
  claim₅ := traceClaimIn s 4 Key20.level₅ Key20.quarters.2.2.2.1
  state := stateOf Key20.terminal (terminalMessagesIn s 6) tLast
  statement := terminalStatementIn pub s tLast
  partials := partialsOf Key20.terminal (terminalMessagesIn s 6) eLast
  response := zSource Key20.terminal (responseIn pub s)
  trace := Extract.Trace.empty 5 16384

/-- The data of a wire message of the model for the replay of a proof of `key20.aks`: the
commitments, values and points of the public statement, the claims of the levels 0 to 5 with
their commitments, and the state with the last block `tLast`, the statement, the clear partials
with the last block `eLast` and the response of the terminal. -/
noncomputable def note20 (tLast : ℕ → Rq Key20.terminal.innerDegree)
    (eLast : Rq Key20.terminal.innerDegree) : Key20.Note E where
  commit := (rootClaim20 (E := E) pub proof messages).payload
  value g := (((pub.groups.flatMap (·.value)).getD g Ext.zero).into : E)
  point g i := (((rootPoint20 pub g).getD i Ext.zero).into : E)
  claim := rootClaim20 pub proof messages
  commit₁ := (claim20₁ (E := E) pub proof messages).payload
  claim₁ := claim20₁ pub proof messages
  commit₂ := (traceClaim20 (E := E) pub proof messages 1 Key20.level₂ Key20.quarters.1).payload
  claim₂ := traceClaim20 pub proof messages 1 Key20.level₂ Key20.quarters.1
  commit₃ := (traceClaim20 (E := E) pub proof messages 2 Key20.level₃
    Key20.quarters.2.1).payload
  claim₃ := traceClaim20 pub proof messages 2 Key20.level₃ Key20.quarters.2.1
  commit₄ := (traceClaim20 (E := E) pub proof messages 3 Key20.level₄
    Key20.quarters.2.2.1).payload
  claim₄ := traceClaim20 pub proof messages 3 Key20.level₄ Key20.quarters.2.2.1
  commit₅ := (traceClaim20 (E := E) pub proof messages 4 Key20.level₅
    Key20.quarters.2.2.2.1).payload
  claim₅ := traceClaim20 pub proof messages 4 Key20.level₅ Key20.quarters.2.2.2.1
  state := stateOf Key20.terminal (terminalMessagesIn (terminalState20 pub proof messages) 6) tLast
  statement := terminalStatement20 pub proof messages tLast
  partials :=
    partialsOf Key20.terminal (terminalMessagesIn (terminalState20 pub proof messages) 6) eLast
  response := zSource Key20.terminal (response20 pub proof messages)
  trace := Extract.Trace.empty 5 16384

end Note

section Tower

variable (E) in
/-- The basis `(1, u)` of `E` over `Z_q`. -/
noncomputable def basis20 : Module.Basis (Fin 2) Zq E :=
  (QuadraticAlgebra.basis (2 : Zq) 0).map (Degree2.coords (E := E)).symm.toLinearEquiv

/-- The basis `basis20` is the basis `extBasis` of the replay. -/
theorem basis20_apply (t : Fin 2) : basis20 E t = (extBasis.getD t Ext.zero).into := by
  show (Degree2.coords (E := E)).symm (QuadraticAlgebra.basis (2 : Zq) 0 t) =
    (Degree2.coords (E := E)).symm (extBasis.getD t Ext.zero).toE
  congr 1
  fin_cases t
  · exact (QuadraticAlgebra.basis_apply_zero _ _).trans Ext.toE_one.symm
  · refine QuadraticAlgebra.basis_apply_one.trans ?_
    ext <;> simp [extBasis, Ext.toE, Fq.toZq_zero, Fq.toZq_one]

variable (pub : Public) {Coord : Type}

/-- The tower data `Key20.noteTower` of the replay of `pub`: the basis `basis20`, the matrices
`Key.ofSetup`, `OpeningKey.ofSetup` of the setup prefix of each fold of `layout20` and `matrixOf`
of the setup prefix of the terminal, with the coordinates `block` to `blockL` of the fold
challenges, `total` stages and the fold challenge of the terminal at `stage`. -/
noncomputable def tower20 (block : Key20.root.Block → Coord)
    (block₁ : Fin Key20.level₁.shape.blocks → Coord) (block₂ : Fin Key20.level₂.shape.blocks → Coord)
    (block₃ : Fin Key20.level₃.shape.blocks → Coord) (block₄ : Fin Key20.level₄.shape.blocks → Coord)
    (block₅ : Fin Key20.level₅.shape.blocks → Coord) (blockL : Fin Key20.terminal.blocks → Coord)
    (total stage : ℕ) : Key20.Tower E (Key20.Wire E) Coord total :=
  Key20.noteTower (fun _ => basis20 E)
    (fun g => Key.ofSetup (flatOf (setup20 pub 0)) (Key20.root.shape g))
    ⟨Setup.shared (flatOf (setup20 pub 0)) Key20.root.openDegree Key20.root.openRank _
      Key20.root.openSetupCode, PathKey.ofSetup (flatOf (setup20 pub 0)) Key20.root.chain⟩
    block (basis20 E) (Key.ofSetup (flatOf (setup20 pub 1)) Key20.level₁.shape)
    (OpeningKey.ofSetup (flatOf (setup20 pub 1)) Key20.level₁.opening) block₁
    (Key.ofSetup (flatOf (setup20 pub 2)) Key20.level₂.shape)
    (OpeningKey.ofSetup (flatOf (setup20 pub 2)) Key20.level₂.opening) block₂
    (Key.ofSetup (flatOf (setup20 pub 3)) Key20.level₃.shape)
    (OpeningKey.ofSetup (flatOf (setup20 pub 3)) Key20.level₃.opening) block₃
    (Key.ofSetup (flatOf (setup20 pub 4)) Key20.level₄.shape)
    (OpeningKey.ofSetup (flatOf (setup20 pub 4)) Key20.level₄.opening) block₄
    (Key.ofSetup (flatOf (setup20 pub 5)) Key20.level₅.shape)
    (OpeningKey.ofSetup (flatOf (setup20 pub 5)) Key20.level₅.opening) block₅
    (matrixOf Key20.terminal (full20 pub)) blockL total stage

/-- `Tower.Drawn` for `tower20`: the coordinates of the fold challenges of each level and of the
terminal are injective, the fold challenge of the terminal is the stage 472 after the 472 stages
of the root and the levels 0 to 5, a stage of the transcript, and `Kept` holds. -/
theorem tower20_drawn [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] [Fintype E]
    [DecidableEq E] {block : Key20.root.Block → Coord}
    {block₁ : Fin Key20.level₁.shape.blocks → Coord} {block₂ : Fin Key20.level₂.shape.blocks → Coord}
    {block₃ : Fin Key20.level₃.shape.blocks → Coord} {block₄ : Fin Key20.level₄.shape.blocks → Coord}
    {block₅ : Fin Key20.level₅.shape.blocks → Coord} {blockL : Fin Key20.terminal.blocks → Coord}
    (injective : Function.Injective block) (injective₁ : Function.Injective block₁)
    (injective₂ : Function.Injective block₂) (injective₃ : Function.Injective block₃)
    (injective₄ : Function.Injective block₄) (injective₅ : Function.Injective block₅)
    (injectiveL : Function.Injective blockL) {total : ℕ} (inside : 472 < total)
    (kept : Key20.Kept) :
    (tower20 (E := E) pub block block₁ block₂ block₃ block₄ block₅ blockL total 472).Drawn :=
  ⟨injective, injective₁, injective₂, injective₃, injective₄, injective₅, by
    simp only [tower20, Key20.noteTower, Key20.terminalOf, Extract.Tier.finish,
      Extract.Tier.tensored, Extract.committedTraceSq, Extract.committedTrace, Extract.liftFold,
      Extract.liftBatch, List.length_append, Extract.foldCosts_length, Extract.foldCostsSq_length,
      Extract.tensorCosts, List.length_cons, List.length_replicate]
    decide, inside, injectiveL, rfl, kept⟩

end Tower

end NightstreamFPrime.Spec.Compression.Akita.Replay
