import CompressionLean.Terminal.Soundness
import CompressionLean.Layer1.Soundness
import CompressionLean.Game.Restoration
import CompressionLean.Shrink.Layout

/-!
Owns the claim of a final proof and the encoding premise that links the
three layers.

- `claimOf`: the layer-1 claim that the final program reads from the
  terminal output columns (`Package::apply`, `terminal/package.rs:215-236`).
- `TerminalFor`: a terminal assignment for the statement `x`.
- `TerminalClaim`: what a final proof for `x` claims. Some terminal
  assignment for `x` has a parent claim with a CE(B) witness.
- `ProgramEncodes`: the encoding that links the layers. A witness of the
  final program's CCS holds a terminal assignment and accepting layer-1
  transcripts for it. `programEncodes` (`Final/Encodes.lean`) proves it for
  the production final program.

Does not own the layers' games or their errors.
-/

namespace CompressionLean.Final

open CompressionLean.Game
open CompressionLean NightstreamFPrime Polynomial

variable {F K : Type} [Field F]

/-- Word `k` of an output list on `env`, read into `F`; zero past the end. -/
def word (list : List Circuit.Expr) (env : Circuit.Env) (k : ℕ) : F :=
  (((list.map fun expression : Circuit.Expr => expression.eval env).getD k 0).val : F)

/-- The layer-1 claim of a terminal assignment, read from its output columns
as `Package::apply` reads it (`terminal/package.rs:215-236`): `D`-word rows
for the commitment and the public input, word pairs for the point, and `D`
pairs for `Eval_K` and for each `Eval_A`. -/
def claimOf (env : Circuit.Env) : Layer1.Statement F where
  commitment ρ l :=
    if l < Spec.ringDegree then word Terminal.outputs.commitment env (Spec.ringDegree * ρ + l) else 0
  publicInput j l :=
    if l < Spec.ringDegree then word Terminal.outputs.publicInput env (Spec.ringDegree * j + l) else 0
  r := (List.range (Terminal.outputs.point.length / 2)).map fun k =>
    (word Terminal.outputs.point env (2 * k), word Terminal.outputs.point env (2 * k + 1))
  evalK l :=
    if l < Spec.ringDegree then
      (word Terminal.outputs.eval_K env (2 * l), word Terminal.outputs.eval_K env (2 * l + 1))
    else 0
  evalA j l :=
    if l < Spec.ringDegree then
      (word Terminal.outputs.eval_A env (2 * Spec.ringDegree * j + 2 * l),
        word Terminal.outputs.eval_A env (2 * Spec.ringDegree * j + 2 * l + 1))
    else 0

/-- The statement of a final proof: the iteration, `z0` and the current
state, the nine words that the final program makes public
(`lifecycle/terminal/mod.rs:46-52`). -/
abbrev StateWords := Fin 9 → Spec.F

/-- The statement words of the prior state preimage, after the four context
words (`STATE_CONTEXT_START`, `lifecycle/inputs.rs:41-43`). -/
def priorStatement (env : Circuit.Env) : StateWords := fun k =>
  (Layout.Stage1.PiCCSInputs.priorStateWord
    (Lifecycle.PiCCS.v1_2.StateBinding.contextWordStart + 4 + k.val)).eval
    (Layout.Stage1.Spartan.pullback env)

/-- A terminal assignment for the statement `x`, as the final program wires
it: every terminal row holds, the public context columns hold the circuit's
context digest, the statement words of the prior state preimage are `x`
(`terminal/package.rs:186-213`), and the iteration is not zero
(`terminal/mod.rs:60-68`). -/
def TerminalFor (authority : Lifecycle.VerifierContext.Authority) (x : StateWords)
    (env : Circuit.Env) : Prop :=
  (Terminal.circuitPackage ()).RowsHold env ∧
    Export.Stage1.Package.SelectedVerifierContext authority env ∧
    priorStatement env = x ∧ x 0 ≠ 0

/-- What a final proof for statement `x` claims: some terminal assignment
for `x` has a parent claim with a CE(B) witness. `circuitPackage_implies_terminal`
gives the assignment's F′ meaning. -/
def TerminalClaim (s : Layer1.Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Layer1.Setup F)
    (authority : Lifecycle.VerifierContext.Authority) (x : StateWords) : Prop :=
  ∃ env, TerminalFor authority x env ∧ ∃ z, Layer1.Relation s Φ bar S (claimOf env) z

/-- **The encoding.** A witness `w` of the CCS of the final program for
statement `x` holds a terminal assignment for `x`, and every reading of its
layer-1 part is an accepting path of the layer-1 game `g` for that
assignment's parent claim.

`Reads x w p` says that the layer-1 path `p` is what program `x` reads from
`w`. Its statement, proof words and challenges are the words that the
in-circuit verifier computes from `w`. At each commitment, its candidate list
holds a table that agrees with every value `w` opens there. The model's
commitments are candidate lists and the program's are Merkle roots, so a
reading is a relation, not a function of `w`.

It stands for `FinalProgram::run` (`lifecycle/terminal/mod.rs:54-80`):
`Package::apply` (`terminal/package.rs:179-237`) and `verify_with`
(`neo-spartan/src/lib.rs:692-703`, `verifier.rs:136-316`), recorded as rows
by `circuit/record.rs` and placed by `shrink/layout.rs`. `programEncodes`
(`Final/Encodes.lean`) proves it for the production final program. -/
def ProgramEncodes {t L : ℕ} (T : Shrink.Template F) (s : Shrink.Counts) (prog : StateWords → Shrink.Run F)
    (authority : Lifecycle.VerifierContext.Authority) (g : Game (Option (Layer1.Transcript F K t L)))
    (Reads : StateWords → (ℕ → F) → g.Pos → Prop) : Prop :=
  ∀ x w, ((prog x).ccs T s).Relation w → ∃ env, TerminalFor authority x env ∧
    ∀ p, Reads x w p → ∃ tr, Game.Pos.node g p = .pure (some tr) ∧ tr.statement = claimOf env

/-- One link of the chain. A witness of program `x`'s CCS and a reading of it
whose leaf extracts a CE(B) witness give the terminal claim for `x`. -/
theorem terminalClaim_of {t L : ℕ} {T : Shrink.Template F} {s : Shrink.Counts}
    {prog : StateWords → Shrink.Run F} {authority : Lifecycle.VerifierContext.Authority}
    {g : Game (Option (Layer1.Transcript F K t L))} {Reads : StateWords → (ℕ → F) → g.Pos → Prop}
    (s1 : Layer1.Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Layer1.Setup F)
    (encodes : ProgramEncodes T s prog authority g Reads) {x : StateWords} {w : ℕ → F}
    (holds : ((prog x).ccs T s).Relation w) {p : g.Pos} (reads : Reads x w p)
    (extracts : ∀ r, Game.Pos.node g p = .pure r →
      ∀ tr, r = some tr → ∃ v ∈ tr.early.candidates, Layer1.Relation s1 Φ bar S tr.statement v.z) :
    TerminalClaim s1 Φ bar S authority x := by
  obtain ⟨env, terminal, leaves⟩ := encodes x w holds
  obtain ⟨tr, leaf, statement⟩ := leaves p reads
  obtain ⟨v, -, relation⟩ := extracts _ leaf tr rfl
  exact ⟨env, terminal, v.z, statement ▸ relation⟩

end CompressionLean.Final
