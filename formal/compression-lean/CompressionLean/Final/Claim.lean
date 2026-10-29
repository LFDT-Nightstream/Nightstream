import CompressionLean.Terminal.Soundness
import CompressionLean.Layer1.Soundness
import CompressionLean.Game.Restoration
import CompressionLean.Shrink.Layout
import CompressionLean.Spec.Parent

namespace CompressionLean.Final

open CompressionLean.Spec
open CompressionLean NightstreamFPrime Polynomial

variable {F K : Type} [Field F]

def word (list : List Circuit.Expr) (env : Circuit.Env) (k : ℕ) : F :=
  (((list.map fun expression : Circuit.Expr => expression.eval env).getD k 0).val : F)

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

def TerminalClaim (s : Layer1.Shape) (Φ : F[X]) (bar : ℕ → ℕ → F) (S : Layer1.Setup F)
    (authority : Lifecycle.VerifierContext.Authority) (x : StateWords) : Prop :=
  ∃ env, TerminalFor authority x env ∧ ∃ z, Layer1.Relation s Φ bar S (claimOf env) z

def ProgramEncodes {t L : ℕ} (T : Shrink.Template F) (s : Shrink.Counts) (prog : StateWords → Shrink.Run F)
    (authority : Lifecycle.VerifierContext.Authority) (g : Game (Option (Layer1.Transcript F K t L)))
    (Reads : StateWords → (ℕ → F) → g.Pos → Prop) : Prop :=
  ∀ x w, ((prog x).ccs T s).Relation w → ∃ env, TerminalFor authority x env ∧
    ∀ p, Reads x w p → ∃ tr, Game.Pos.node g p = .pure (some tr) ∧ tr.statement = claimOf env

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
