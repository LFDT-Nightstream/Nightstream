import CompressionLean.Final.Parent
import CompressionLean.Final.Soundness
import CompressionLean.Lib.List

/-!
Owns the claim of a final proof in F′'s terms and its soundness over the
concrete fields.

- `claimOf_parent`: the layer-1 claim that the final program reads from the
  terminal output columns is the statement of the PiRLC parent there.
- `ParentClaim`: some terminal assignment for `x` holds a PiCCS proof that
  F′'s verifier accepts, and the PiRLC parent that the verifier computes
  has a witness of F′'s CE(B) relation.
- `parentClaim_of_terminalClaim`: under the honest setup, the terminal claim
  gives the F′ claim.
- `final_sound_parent`, `final_restore_budget_parent`: `final_sound` and
  `final_restore_budget` for the F′ claim, over `Gl` and `Cubic`.
-/

namespace CompressionLean.Final

open CompressionLean.Lib
open Polynomial NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel

variable {lw : ℕ} {pf : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}

/-! ## Reading the output columns -/

/-- The word that `claimOf` reads at a position of a list of words. -/
theorem word_eq (list : List Circuit.Expr) (env : Circuit.Env) (k : ℕ) (values : List F)
    (eval : list.map (fun expression => expression.eval env) = values) :
    (word list env k : Gl) = gl (values.getD k 0) := by
  unfold word
  rw [eval]
  exact ZMod.natCast_zmod_val (n := goldilocksModulus) _

/-- **The claim names the parent.** The layer-1 claim that the final program
reads from the terminal output columns (`claimOf`) is the statement of the
PiRLC parent in those columns. -/
theorem claimOf_parent (relation : ProductionKey.LogicalRelation lw pf)
    (env : Circuit.Env) :
    (claimOf env : Layer1.Statement Gl) = statementOf (Terminal.parent relation env) := by
  have commitment := Terminal.commitment_eval relation env
  have publicInput := Terminal.publicInput_eval relation env
  have point := Terminal.point_eval relation env
  have evalK := Terminal.eval_K_eval env
  have evalA := Terminal.eval_A_eval env
  have family : family (Terminal.parent relation env) = Terminal.parentEvaluation env := by
    rw [Final.family, Terminal.parent_evaluations]
    rfl
  have pairs (n : ℕ) (g h : Fin n → F) : ∀ x ∈ List.ofFn (fun c => [g c, h c]), x.length = 2 := by
    simp
  have rows (m n : ℕ) (f : Fin m → Fin n → F) : ∀ x ∈ List.ofFn (fun i => List.ofFn (f i)), x.length = n := by
    simp
  have coordinates : ∀ x ∈ (Terminal.parent relation env).point.coordinates.map
      (fun value => [value.c0, value.c1]), x.length = 2 := by
    simp
  have blocks : ∀ x ∈ List.ofFn (fun matrix => (List.ofFn fun coefficient =>
      [((Terminal.parentEvaluation env).matrix matrix coefficient).c0,
        ((Terminal.parentEvaluation env).matrix matrix coefficient).c1]).flatten), x.length = 2 * ringDegree := by
    intro x member
    obtain ⟨matrix, rfl⟩ := List.mem_ofFn.mp member
    rw [length_flatten_uniform 2 _ (pairs _ _ _), List.length_ofFn]
    rfl
  show Layer1.Statement.mk _ _ _ _ _ = Layer1.Statement.mk _ _ _ _ _
  congr 1
  · funext ρ l
    split_ifs with hl h h
    · rw [word_eq _ env _ _ commitment, getD_flatten ringDegree 0 _ (rows _ _ _) ρ l hl,
        dif_pos (by simpa using h.1), List.getElem_ofFn, getD_ofFn, dif_pos hl]
    · rw [word_eq _ env _ _ commitment, getD_flatten ringDegree 0 _ (rows _ _ _) ρ l hl,
        dif_neg (by simpa using fun lt => h ⟨lt, hl⟩)]
      rfl
    · exact absurd h.2 hl
    · rfl
  · funext j l
    split_ifs with hl h h
    · rw [word_eq _ env _ _ publicInput, getD_ofFn, dif_pos (show ringDegree * j + l <
        (PaperAlgebra.FullShape lw pf).publicWidth from h.2)]
    · rw [word_eq _ env _ _ publicInput, getD_ofFn, dif_neg (show ¬ (ringDegree * j + l <
        (PaperAlgebra.FullShape lw pf).publicWidth) from fun lt => h ⟨hl, lt⟩)]
      rfl
    · exact absurd h.1 hl
    · rfl
  · have length : Terminal.outputs.point.length / 2 = (Terminal.parent relation env).point.coordinates.length := by
      have := congrArg List.length point
      rw [List.length_map, length_flatten_uniform 2 _ coordinates, List.length_map] at this
      omega
    apply List.ext_getElem
    · simp [length]
    · intro k inside _
      have small : k < (Terminal.parent relation env).point.coordinates.length := by simpa [length] using inside
      rw [List.getElem_map, List.getElem_map, List.getElem_range, word_eq _ env _ _ point,
        word_eq _ env _ _ point, show 2 * k = 2 * k + 0 from rfl, getD_flatten 2 0 _ coordinates k 0 (by decide),
        getD_flatten 2 0 _ coordinates k 1 (by decide), dif_pos (by simpa using small),
        dif_pos (by simpa using small)]
      simp
  · funext l
    split_ifs with hl
    · rw [word_eq _ env _ _ evalK, word_eq _ env _ _ evalK, family, show 2 * l = 2 * l + 0 from rfl,
        getD_flatten 2 0 _ (pairs _ _ _) l 0 (by decide), getD_flatten 2 0 _ (pairs _ _ _) l 1 (by decide),
        dif_pos (by rw [List.length_ofFn]; exact hl), dif_pos (by rw [List.length_ofFn]; exact hl)]
      simp
    · rfl
  · funext j l
    split_ifs with hl h h
    · have lane : 2 * l < 2 * ringDegree := by omega
      have lane' : 2 * l + 1 < 2 * ringDegree := by omega
      rw [word_eq _ env _ _ evalA, word_eq _ env _ _ evalA, family,
        show 2 * ringDegree * j + 2 * l + 1 = 2 * ringDegree * j + (2 * l + 1) by ring,
        getD_flatten (2 * ringDegree) 0 _ blocks j (2 * l) lane, getD_flatten (2 * ringDegree) 0 _ blocks j _ lane',
        dif_pos (by rw [List.length_ofFn]; exact h.1), dif_pos (by rw [List.length_ofFn]; exact h.1),
        List.getElem_ofFn, show 2 * l = 2 * l + 0 from rfl, getD_flatten 2 0 _ (pairs _ _ _) l 0 (by decide),
        getD_flatten 2 0 _ (pairs _ _ _) l 1 (by decide), dif_pos (by rw [List.length_ofFn]; exact hl),
        dif_pos (by rw [List.length_ofFn]; exact hl)]
      simp
    · have lane : 2 * l < 2 * ringDegree := by omega
      have lane' : 2 * l + 1 < 2 * ringDegree := by omega
      rw [word_eq _ env _ _ evalA, word_eq _ env _ _ evalA,
        show 2 * ringDegree * j + 2 * l + 1 = 2 * ringDegree * j + (2 * l + 1) by ring,
        getD_flatten (2 * ringDegree) 0 _ blocks j (2 * l) lane, getD_flatten (2 * ringDegree) 0 _ blocks j _ lane',
        dif_neg (by rw [List.length_ofFn]; exact fun lt => h ⟨lt, hl⟩),
        dif_neg (by rw [List.length_ofFn]; exact fun lt => h ⟨lt, hl⟩)]
      rfl
    · exact absurd h.2 hl
    · rfl

/-! ## The F′ claim of a final proof -/

/-- The running claims of a terminal assignment, as F′'s accumulator reads
them from the package columns. -/
abbrev runningOf (env : Circuit.Env) :=
  Layout.Stage1.AccumulatorInputs.running lw pf (Layout.Stage1.Spartan.pullback env)

/-- The fresh claim of a terminal assignment. -/
abbrev freshOf (env : Circuit.Env) :=
  Layout.Stage1.AccumulatorInputs.fresh lw pf (Layout.Stage1.Spartan.pullback env)

/-- The PiCCS proof of a terminal assignment. -/
abbrev proofOf (relation : ProductionKey.LogicalRelation lw pf) (env : Circuit.Env) :=
  Layout.Stage1.AccumulatorInputs.proof relation (Layout.Stage1.Spartan.pullback env)

/-- **What a final proof for `x` claims, in F′'s terms.** Some terminal
assignment for `x` (`TerminalFor`) holds a PiCCS proof that F′'s verifier
accepts for the key of `relation` and `ajtai`, and the PiRLC parent that the
verifier computes from it has a witness of F′'s CE(B) relation (SuperNeo
v1.2 Definition 21). -/
def ParentClaim (relation : ProductionKey.LogicalRelation lw pf)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
    (authority : Lifecycle.VerifierContext.Authority) (x : StateWords) : Prop :=
  ∃ env, TerminalFor authority x env ∧
    Spec.Folding.Nifs.PaperNonInteractive.piCcsCheck (ProductionKey.key relation ajtai) (runningOf env) (freshOf env)
      (proofOf relation env) = true ∧
    (ProductionKey.key relation ajtai).parent (runningOf env) (freshOf env) (proofOf relation env) =
      some (Terminal.parent relation env) ∧
    ∃ a, Spec.CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams (Terminal.parent relation env) a

/-- Under the honest setup, the terminal claim is the F′ claim: the layer-1
witness of the parent claim is a witness of F′'s relation for the PiRLC
parent (`parent_holds`). -/
theorem parentClaim_of_terminalClaim (s1 : Layer1.Shape) (bar : ℕ → ℕ → Gl) (S : Layer1.Setup Gl)
    (relation : ProductionKey.LogicalRelation lw pf)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
    (shape : ParentShape s1 lw) (blocks_le : s1.blocks ≤ 2 ^ s1.blockVars)
    (barEq : ∀ p l : Fin ringDegree, bar p l = gl (nativeBarEntry p l))
    (honest : HonestSetup s1 S relation ajtai) (authority : Lifecycle.VerifierContext.Authority) (x : StateWords)
    (claim : TerminalClaim s1 modulus bar S authority x) : ParentClaim relation ajtai authority x := by
  obtain ⟨env, terminal, z, holds⟩ := claim
  rw [claimOf_parent relation env] at holds
  obtain ⟨check, parent, -, -⟩ := Terminal.circuitPackage_implies_terminal relation ajtai env terminal.1
  exact ⟨env, terminal, check, parent, _,
    parent_holds s1 bar S relation ajtai _ shape blocks_le barEq honest rfl rfl rfl holds⟩

/-! ## End-to-end soundness for F′'s relation -/

theorem modulus_natDegree : modulus.natDegree = ringDegree :=
  natDegree_eq_of_degree_eq_some degree_modulus

/-- **End-to-end soundness for F′'s relation, in sequence.** Over the
Goldilocks field and its cubic extension, and under the honest setup of the
F′ key, a prover makes the final verifier accept a statement whose F′ claim
(`ParentClaim`) is false with probability at most the sum of the layer
errors at `p³` challenges. The premises are those of `final_sound`, with `Φ`
the Φ81 modulus, `bar` the Φ81 bar transform and `S` the honest setup. -/
theorem final_sound_parent (s1 : Layer1.Shape) (bar : ℕ → ℕ → Gl) (S : Layer1.Setup Gl)
    (relation : ProductionKey.LogicalRelation lw pf)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
    (shape : ParentShape s1 lw) (barEq : ∀ p l : Fin ringDegree, bar p l = gl (nativeBarEntry p l))
    (honest : HonestSetup s1 S relation ajtai) (fits : S.Fits s1)
    (L0 L1 L P : ℕ) [NeZero L] [NeZero P] (q : ℕ → Cubic) (N : ℕ) (sizes : s1.Sizes N)
    (small : N ≤ goldilocksModulus) (distinct : ∀ i j, i < L → j < L → q i = q j → i = j)
    (T : Shrink.Template Gl) (s : Shrink.Counts) (prog : StateWords → Shrink.Run Gl) (Ls : ℕ)
    (wf : T.WellFormed) (valid : ∀ x, (prog x).Valid T)
    (authority : Lifecycle.VerifierContext.Authority)
    (Reads : StateWords → (ℕ → Gl) → (Layer1.layer1Game s1 modulus bar S L0 L1 L P q).Pos → Prop)
    (encodes : ProgramEncodes T s prog authority (Layer1.layer1Game s1 modulus bar S L0 L1 L P q) Reads)
    (prover : Layer1.Game.Strategy (Outcome Gl Cubic s1.queries L)) :
    Layer1.Game.prob prover
        (pair (Layer1.layer1Game s1 modulus bar S L0 L1 L P q) (Shrink.shrinkGame T s prog Ls))
        (FinalBreak (ParentClaim relation ajtai authority) T s prog (Layer1.layer1Game s1 modulus bar S L0 L1 L P q)
          Reads) ≤
      Layer1.layer1Error s1 L0 L1 L P (goldilocksModulus ^ 3) +
        Shrink.shrinkError (Shrink.Layout.rowVars T s) (Shrink.Layout.cellVars T s) Ls (goldilocksModulus ^ 3) := by
  rw [← card_cubic]
  exact (prob_mono prover _ (finalBreak_mono
      (parentClaim_of_terminalClaim s1 bar S relation ajtai shape sizes.blocks_le barEq honest authority) T s prog _
        Reads)).trans
    (final_sound s1 modulus bar S L0 L1 L P q N sizes (cubic_natCast_injective N small) distinct fits modulus_monic
      (by rw [modulus_natDegree, shape.D]) (by rw [shape.D]; decide) T s prog Ls wf valid
      (cubic_natCast_injective 9 (by norm_num [goldilocksModulus])) authority Reads encodes prover)

/-- **The production bound.** Under Fiat–Shamir, over the Goldilocks field
and its cubic extension, and with the charged terms of each commitment
within Rust's split (as in `final_restore_budget`), a prover with `calls`
queries makes the final verifier accept a statement whose F′ claim is false
with probability at most `calls / 2^t`. -/
theorem final_restore_budget_parent (s1 : Layer1.Shape) (bar : ℕ → ℕ → Gl) (S : Layer1.Setup Gl)
    (relation : ProductionKey.LogicalRelation lw pf)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
    (shape : ParentShape s1 lw) (barEq : ∀ p l : Fin ringDegree, bar p l = gl (nativeBarEntry p l))
    (honest : HonestSetup s1 S relation ajtai) (fits : S.Fits s1)
    (L0 L1 : ℕ) (h0 : 1 ≤ L0) (h1 : 1 ≤ L1) (index : s1.setupVars ≤ 35) (t : ℕ)
    (T : Shrink.Template Gl) (s : Shrink.Counts) (prog : StateWords → Shrink.Run Gl) (Ls : ℕ)
    (p0 : (L0 : ℝ) * L1 * Layer1.rustDegrees s1 / 2 ^ 191 ≤ 1 / 2 ^ (t + 1))
    (p1 : (L1 : ℝ) * (1 / 2) ^ (s1.queries - 1) ≤ 1 / 2 ^ (t + 1))
    (rust : (Ls : ℝ) * Shrink.rustDegrees (Shrink.Layout.rowVars T s) (Shrink.Layout.cellVars T s) / 2 ^ 191 ≤
      1 / 2 ^ t)
    (q : ℕ → Cubic) (N : ℕ) (sizes : s1.Sizes N) (small : N ≤ goldilocksModulus)
    (distinct : ∀ i j, i < 2 ^ (s1.setupVars - 2) → j < 2 ^ (s1.setupVars - 2) → q i = q j → i = j)
    (wf : T.WellFormed) (valid : ∀ x, (prog x).Valid T)
    (authority : Lifecycle.VerifierContext.Authority)
    (Reads : StateWords → (ℕ → Gl) →
      (Layer1.layer1Game s1 modulus bar S L0 L1 (2 ^ (s1.setupVars - 2)) Layer1.goldilocks q).Pos → Prop)
    (encodes : ProgramEncodes T s prog authority
      (Layer1.layer1Game s1 modulus bar S L0 L1 (2 ^ (s1.setupVars - 2)) Layer1.goldilocks q) Reads)
    (calls : ℕ)
    (prover : Layer1.Game.Strategy
      (RestoreOutcome Gl Cubic (Layer1.layer1Game s1 modulus bar S L0 L1 (2 ^ (s1.setupVars - 2)) Layer1.goldilocks q))) :
    Layer1.Game.prob prover
        (restorePair (Shrink.shrinkGame T s prog Ls)
          (Layer1.layer1Game s1 modulus bar S L0 L1 (2 ^ (s1.setupVars - 2)) Layer1.goldilocks q) calls [] [])
        (FinalRestoreBreak (ParentClaim relation ajtai authority) T s prog
          (Layer1.layer1Game s1 modulus bar S L0 L1 (2 ^ (s1.setupVars - 2)) Layer1.goldilocks q) Reads) ≤
      calls / 2 ^ t :=
  (prob_mono prover _ (finalRestoreBreak_mono
      (parentClaim_of_terminalClaim s1 bar S relation ajtai shape sizes.blocks_le barEq honest authority) T s prog _
        Reads)).trans
    (final_restore_budget s1 modulus bar S L0 L1 h0 h1 cubic_big index t T s prog Ls p0 p1 rust q N sizes
      (cubic_natCast_injective N small) distinct fits modulus_monic (by rw [modulus_natDegree, shape.D])
      (by rw [shape.D]; decide) wf valid (cubic_natCast_injective 9 (by norm_num [goldilocksModulus])) authority Reads
      encodes calls prover)

end CompressionLean.Final
