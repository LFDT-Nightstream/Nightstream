import NightstreamFPrime.Spec.Compression.Fresh
import NightstreamFPrime.Spec.Compression.Akita.Extract.Uniform

/-! Π_CCS on one source as the stages of a Fiat–Shamir transcript (Akita, ePrint 2026/1983,
revision of 2026-11-23, Definition 10.16 (vii): the bad sets of the stages of a checkpoint
segment; SPEC.md Algorithm 10, step 7).

- `costs` has the bound of each draw of `Fresh.game` in its order: `α`, `γ` and the round
  challenges. `costs_sum` is their sum.
- `game_staged`: each draw has its bound of `costs`, and each path has all the draws.
- `game_bare`: the verifier does not read the witness. It is in the bad sets only. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Fresh

open NightstreamFPrime.Spec NightstreamFPrime.Spec.FieldExtensionRing
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression.Akita.Extract

/-- The bounds of the draws of `game` in their order: `α`, `γ` and the round challenges. -/
noncomputable def costs (shape : Shape) (width : ℕ) : List ℝ :=
  (shape.cubeVariables : ℝ) / Fintype.card K ::
    ((shape.jointCoefficientCount - 1 : ℕ) : ℝ) / Fintype.card K ::
      List.replicate shape.cubeVariables ((width : ℝ) / Fintype.card K)

theorem costs_sum (shape : Shape) (width : ℕ) :
    (costs shape width).sum =
      (shape.cubeVariables + (shape.jointCoefficientCount - 1 : ℕ) +
        shape.cubeVariables * width : ℝ) / Fintype.card K := by
  simp only [costs, List.sum_cons, List.sum_replicate]
  ring

theorem costs_nonnegative (shape : Shape) (width : ℕ) : ∀ cost ∈ costs shape width, 0 ≤ cost := by
  intro cost member
  simp only [costs, List.mem_cons, List.mem_replicate] at member
  rcases member with rfl | rfl | ⟨_, rfl⟩ <;> positivity

variable {Commitment PublicInput : Type} {shape : Shape} {columns blockCount : ℕ}
  (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)
  (witness : OutputWitness shape columns) (width : ℕ) {Reply : Type} [Fintype Reply]

theorem roundGame_staged (scalar : ∃ decode : Reply → K, Fair decode)
    (degreeCovers : (statement.verifierInput K.embed).sumcheckDegreeBound ≤ width)
    (a : Fin shape.cubeVariables → K) (g : K) (more : List ℝ) :
    ∀ (n : ℕ) (points : List K) (messages : List (FixedPolynomial K width)),
      points.length + n = shape.cubeVariables →
      StagedTo Reply (roundGame statement witness width a g n points messages)
        (List.replicate n ((width : ℝ) / Fintype.card K) ++ more) fun _ rest => rest = more
  | 0, _, _, _ => rfl
  | n + 1, points, messages, length => fun message =>
      stagedTo_draw scalar
        (round_density statement witness width degreeCovers a g points message n (by omega))
        fun r => roundGame_staged scalar degreeCovers a g more n _ _ (by simp; omega)

/-- Each draw of `game` has its bound of `costs`, and each path has all the draws: the bounds
`more` of later draws remain. `vector` and `scalar`: an answer of the oracle gives a uniform
`α` and a uniform element of `K`. -/
theorem game_staged (vector : ∃ decode : Reply → Fin shape.cubeVariables → K, Fair decode)
    (scalar : ∃ decode : Reply → K, Fair decode)
    (degreeCovers : (statement.verifierInput K.embed).sumcheckDegreeBound ≤ width)
    (more : List ℝ) :
    StagedTo Reply (game statement witness width) (costs shape width ++ more)
      fun _ rest => rest = more :=
  stagedTo_draw vector (alpha_density statement witness) fun a =>
    stagedTo_draw scalar (gamma_density statement witness a) fun g =>
      (stagedTo_bind _ _ _ _).mpr
        ((roundGame_staged statement witness width scalar degreeCovers a g more _ [] []
          (by simp)).mono fun _ _ same _ => same)

omit [Fintype Reply] in
theorem roundGame_bare (other : OutputWitness shape columns) (a : Fin shape.cubeVariables → K)
    (g : K) : ∀ (n : ℕ) (points : List K) (messages : List (FixedPolynomial K width)),
      bare (roundGame statement witness width a g n points messages) =
        bare (roundGame statement other width a g n points messages)
  | 0, _, _ => rfl
  | n + 1, _, _ => bare_recv_congr fun _ => bare_draw_congr fun _ =>
      roundGame_bare other a g n _ _

omit [Fintype Reply] in
/-- The verifier of `game` is the same for each witness. -/
theorem game_bare (other : OutputWitness shape columns) :
    bare (game statement witness width) = bare (game statement other width) :=
  bare_draw_congr fun a => bare_draw_congr fun g =>
    bare_bind_congr (roundGame_bare statement witness width other a g _ [] []) fun _ => rfl

end NightstreamFPrime.Spec.Compression.Fresh
