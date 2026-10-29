import CompressionLean.Final.Parent
import CompressionLean.Final.Soundness

namespace CompressionLean.Final

open Polynomial NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel

theorem getD_flatten {α : Type} (n : ℕ) (d : α) :
    ∀ (ls : List (List α)), (∀ x ∈ ls, x.length = n) → ∀ j i, i < n →
      ls.flatten.getD (n * j + i) d = if h : j < ls.length then ls[j].getD i d else d
  | [], _, j, i, _ => by simp
  | x :: rest, len, j, i, hi => by
      have head : x.length = n := len x (by simp)
      rw [List.flatten_cons]
      cases j with
      | zero =>
          rw [Nat.mul_zero, Nat.zero_add, List.getD_append _ _ _ _ (by rw [head]; exact hi)]
          simp
      | succ j =>
          rw [List.getD_append_right _ _ _ _ (by rw [head, Nat.mul_succ]; omega),
            show n * (j + 1) + i - x.length = n * j + i by rw [head, Nat.mul_succ]; omega,
            getD_flatten n d rest (fun y member => len y (by simp [member])) j i hi]
          simp

theorem length_flatten_uniform {α : Type} (n : ℕ) :
    ∀ (ls : List (List α)), (∀ x ∈ ls, x.length = n) → ls.flatten.length = n * ls.length
  | [], _ => by simp
  | x :: rest, len => by
      rw [List.flatten_cons, List.length_append, len x (by simp),
        length_flatten_uniform n rest (fun y member => len y (by simp [member])), List.length_cons, Nat.mul_succ]
      ring

theorem getD_ofFn {α : Type} {n : ℕ} (f : Fin n → α) (d : α) (k : ℕ) :
    (List.ofFn f).getD k d = if h : k < n then f ⟨k, h⟩ else d := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_ofFn]
  split <;> rfl

theorem word_eq (list : List Circuit.Expr) (env : Circuit.Env) (k : ℕ) (values : List F)
    (eval : list.map (fun expression => expression.eval env) = values) :
    (word list env k : Gl) = gl (values.getD k 0) := by
  unfold word
  rw [eval]
  exact ZMod.natCast_zmod_val (n := goldilocksModulus) _

theorem claimOf_parent (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
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
        dite_eq_left (by simpa using h.1), List.getElem_ofFn, getD_ofFn, dite_eq_left hl]
    · rw [word_eq _ env _ _ commitment, getD_flatten ringDegree 0 _ (rows _ _ _) ρ l hl,
        dite_eq_right (by simpa using fun lt => h ⟨lt, hl⟩)]
      rfl
    · exact absurd h.2 hl
    · rfl
  · funext j l
    split_ifs with hl h h
    · rw [word_eq _ env _ _ publicInput, getD_ofFn, dite_eq_left (show ringDegree * j + l <
        (PaperAlgebra.FullShape Data.logicalWidth Data.publicFits).publicWidth from h.2)]
    · rw [word_eq _ env _ _ publicInput, getD_ofFn, dite_eq_right (show ¬ (ringDegree * j + l <
        (PaperAlgebra.FullShape Data.logicalWidth Data.publicFits).publicWidth) from fun lt => h ⟨hl, lt⟩)]
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
        getD_flatten 2 0 _ coordinates k 1 (by decide), dite_eq_left (by simpa using small),
        dite_eq_left (by simpa using small)]
      simp
  · funext l
    split_ifs with hl
    · rw [word_eq _ env _ _ evalK, word_eq _ env _ _ evalK, family, show 2 * l = 2 * l + 0 from rfl,
        getD_flatten 2 0 _ (pairs _ _ _) l 0 (by decide), getD_flatten 2 0 _ (pairs _ _ _) l 1 (by decide),
        dite_eq_left (by rw [List.length_ofFn]; exact hl), dite_eq_left (by rw [List.length_ofFn]; exact hl)]
      simp
    · rfl
  · funext j l
    split_ifs with hl h h
    · have lane : 2 * l < 2 * ringDegree := by omega
      have lane' : 2 * l + 1 < 2 * ringDegree := by omega
      rw [word_eq _ env _ _ evalA, word_eq _ env _ _ evalA, family,
        show 2 * ringDegree * j + 2 * l + 1 = 2 * ringDegree * j + (2 * l + 1) by ring,
        getD_flatten (2 * ringDegree) 0 _ blocks j (2 * l) lane, getD_flatten (2 * ringDegree) 0 _ blocks j _ lane',
        dite_eq_left (by rw [List.length_ofFn]; exact h.1), dite_eq_left (by rw [List.length_ofFn]; exact h.1),
        List.getElem_ofFn, show 2 * l = 2 * l + 0 from rfl, getD_flatten 2 0 _ (pairs _ _ _) l 0 (by decide),
        getD_flatten 2 0 _ (pairs _ _ _) l 1 (by decide), dite_eq_left (by rw [List.length_ofFn]; exact hl),
        dite_eq_left (by rw [List.length_ofFn]; exact hl)]
      simp
    · have lane : 2 * l < 2 * ringDegree := by omega
      have lane' : 2 * l + 1 < 2 * ringDegree := by omega
      rw [word_eq _ env _ _ evalA, word_eq _ env _ _ evalA,
        show 2 * ringDegree * j + 2 * l + 1 = 2 * ringDegree * j + (2 * l + 1) by ring,
        getD_flatten (2 * ringDegree) 0 _ blocks j (2 * l) lane, getD_flatten (2 * ringDegree) 0 _ blocks j _ lane',
        dite_eq_right (by rw [List.length_ofFn]; exact fun lt => h ⟨lt, hl⟩),
        dite_eq_right (by rw [List.length_ofFn]; exact fun lt => h ⟨lt, hl⟩)]
      rfl
    · exact absurd h.2 hl
    · rfl

abbrev runningOf (env : Circuit.Env) :=
  Layout.Stage1.AccumulatorInputs.running Data.logicalWidth Data.publicFits (Layout.Stage1.Spartan.pullback env)

abbrev freshOf (env : Circuit.Env) :=
  Layout.Stage1.AccumulatorInputs.fresh Data.logicalWidth Data.publicFits (Layout.Stage1.Spartan.pullback env)

abbrev proofOf (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits) (env : Circuit.Env) :=
  Layout.Stage1.AccumulatorInputs.proof relation (Layout.Stage1.Spartan.pullback env)

/-- SuperNeo v1.2 Definition 21. -/
def ParentClaim (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (authority : Lifecycle.VerifierContext.Authority) (x : StateWords) : Prop :=
  ∃ env, TerminalFor authority x env ∧
    Spec.Folding.Nifs.PaperNonInteractive.piCcsCheck (ProductionKey.key relation ajtai) (runningOf env) (freshOf env)
      (proofOf relation env) = true ∧
    (ProductionKey.key relation ajtai).parent (runningOf env) (freshOf env) (proofOf relation env) =
      some (Terminal.parent relation env) ∧
    ∃ a, Spec.CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams (Terminal.parent relation env) a

theorem parentClaim_of_terminalClaim (s1 : Layer1.Shape) (bar : ℕ → ℕ → Gl) (S : Layer1.Setup Gl)
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (shape : ParentShape s1 Data.logicalWidth) (blocks_le : s1.blocks ≤ 2 ^ s1.blockVars)
    (barEq : ∀ p l : Fin ringDegree, bar p l = gl (nativeBarEntry p l))
    (honest : HonestSetup s1 S relation ajtai) (authority : Lifecycle.VerifierContext.Authority) (x : StateWords)
    (claim : TerminalClaim s1 modulus bar S authority x) : ParentClaim relation ajtai authority x := by
  obtain ⟨env, terminal, z, holds⟩ := claim
  rw [claimOf_parent relation env] at holds
  obtain ⟨check, parent, -, -⟩ := Terminal.circuitPackage_implies_terminal relation ajtai env terminal.1
  exact ⟨env, terminal, check, parent, _,
    parent_holds s1 bar S relation ajtai _ shape blocks_le barEq honest rfl rfl rfl holds⟩

theorem modulus_natDegree : modulus.natDegree = ringDegree :=
  natDegree_eq_of_degree_eq_some degree_modulus

theorem final_sound_parent (s1 : Layer1.Shape) (bar : ℕ → ℕ → Gl) (S : Layer1.Setup Gl)
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (shape : ParentShape s1 Data.logicalWidth) (barEq : ∀ p l : Fin ringDegree, bar p l = gl (nativeBarEntry p l))
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
    (final_sound s1 modulus bar S L0 L1 L P q N sizes (natCast_injective N small) distinct fits modulus_monic
      (by rw [modulus_natDegree, shape.D]) (by rw [shape.D]; decide) T s prog Ls wf valid
      (natCast_injective 9 (by norm_num [goldilocksModulus])) authority Reads encodes prover)

theorem final_restore_budget_parent (s1 : Layer1.Shape) (bar : ℕ → ℕ → Gl) (S : Layer1.Setup Gl)
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (shape : ParentShape s1 Data.logicalWidth) (barEq : ∀ p l : Fin ringDegree, bar p l = gl (nativeBarEntry p l))
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
      (natCast_injective N small) distinct fits modulus_monic (by rw [modulus_natDegree, shape.D])
      (by rw [shape.D]; decide) wf valid (natCast_injective 9 (by norm_num [goldilocksModulus])) authority Reads
      encodes calls prover)

end CompressionLean.Final
