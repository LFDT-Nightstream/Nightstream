import CompressionLean.Final.Parent

/-! The converse of `Final.parent_holds` (CW-10.6): a parent claim that holds with a witness that is
within the key's norm bound and zero past the key's blocks has its statement in its layer-1
relation, with layer 1's witness padded back to the same assignment (`parent_complete`).

Layer 1's witness of an assignment puts lane `l` of block `b` at `2^laneVars · b + l` and is zero
elsewhere (`witnessOf`). Each row of the statement is a ring sum over the carrier's blocks
(`Final.residual_eq`, with the row identities of `Final/Parent.lean`). The CE claim gives each ring
sum: the commitment rows from the commitment, the evaluation rows from the evaluation family
(`Final.pad_identity`, `Final.evaluate_coefficient`), and the public rows from the public input. -/

namespace CompressionLean.Final

open CompressionLean.Spec
open Polynomial NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial

/-- The converse of `centeredMagnitude_slotValue`: a value of centered magnitude at most `H` is a
slot value of the norm table. -/
theorem exists_slotValue (H : ℕ) (y : Gl) (small : centeredMagnitude y ≤ H) :
    ∃ t, t < 2 * H + 1 ∧ y = Layer1.slotValue H t := by
  have lt : ZMod.val y < goldilocksModulus := ZMod.val_lt y
  change min (ZMod.val y) (goldilocksModulus - ZMod.val y) ≤ H at small
  unfold Layer1.slotValue
  rcases le_or_gt (ZMod.val y) H with low | high
  · refine ⟨H + ZMod.val y, by omega, ?_⟩
    rw [Nat.cast_add, add_sub_cancel_left, ZMod.natCast_zmod_val]
  · have up : goldilocksModulus - ZMod.val y ≤ H := by
      rcases min_le_iff.mp small with left | right
      · omega
      · exact right
    refine ⟨H - (goldilocksModulus - ZMod.val y), by omega, ?_⟩
    rw [Nat.cast_sub up, Nat.cast_sub lt.le, ZMod.natCast_self, ZMod.natCast_zmod_val]
    ring

variable {lw : ℕ} {pf : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}

/-- Layer 1's witness of a carrier assignment: lane `l < D` of block `b` at `2^laneVars · b + l`,
zero on the other lanes and past the carrier. `assignmentOf` reads it back
(`assignmentOf_witnessOf`). -/
def witnessOf (s : Layer1.Shape) (w : PaperAlgebra.Assignment (logicalWidth := lw) (publicFits := pf)) :
    ℕ → Gl := fun x =>
  if h : x % 2 ^ s.laneVars < ringDegree ∧
      ringDegree * (x / 2 ^ s.laneVars) + x % 2 ^ s.laneVars < Phi81CarrierLayout.carrierWidth lw then
    gl (w ⟨_, h.2⟩)
  else 0

theorem assignmentOf_witnessOf {s : Layer1.Shape} (shape : ParentShape s lw)
    {w : PaperAlgebra.Assignment (logicalWidth := lw) (publicFits := pf)}
    (support : ∀ c : Fin (Phi81CarrierLayout.carrierWidth lw), s.blocks ≤ c.val / ringDegree → w c = 0) :
    assignmentOf (lw := lw) (pf := pf) s (witnessOf s w) = w := by
  funext c
  by_cases inside : c.val / ringDegree < s.blocks
  · have kept : assignmentOf (lw := lw) (pf := pf) s (witnessOf s w) c =
        witnessOf s w (2 ^ s.laneVars * (c.val / ringDegree) + c.val % ringDegree) := by
      unfold assignmentOf
      exact ite_eq_left inside
    have lane : c.val % ringDegree < ringDegree := Nat.mod_lt _ (by decide)
    have div : (2 ^ s.laneVars * (c.val / ringDegree) + c.val % ringDegree) / 2 ^ s.laneVars =
        c.val / ringDegree := by
      rw [shape.laneVars]
      change (64 * (c.val / 54) + c.val % 54) / 64 = c.val / 54
      omega
    have mod : (2 ^ s.laneVars * (c.val / ringDegree) + c.val % ringDegree) % 2 ^ s.laneVars =
        c.val % ringDegree := by
      rw [shape.laneVars]
      change (64 * (c.val / 54) + c.val % 54) % 64 = c.val % 54
      omega
    have back : ringDegree * (c.val / ringDegree) + c.val % ringDegree = c.val := Nat.div_add_mod _ _
    rw [kept]
    unfold witnessOf
    rw [dite_eq_left ⟨by rw [mod]; exact lane, by rw [div, mod, back]; exact c.isLt⟩]
    exact congrArg w (Fin.ext (show ringDegree * _ + _ = c.val by rw [div, mod, back]))
  · have zero : assignmentOf (lw := lw) (pf := pf) s (witnessOf s w) c = 0 := by
      unfold assignmentOf
      exact ite_eq_right inside
    rw [zero]
    exact (support c (by omega)).symm

theorem witnessOf_norm {s : Layer1.Shape}
    {w : PaperAlgebra.Assignment (logicalWidth := lw) (publicFits := pf)}
    (norm : ∀ c, centeredMagnitude (w c) ≤ s.normBound) (x : ℕ) :
    ∃ t, t < 2 * s.normBound + 1 ∧ witnessOf s w x = Layer1.slotValue s.normBound t := by
  unfold witnessOf
  split_ifs with h
  · exact exists_slotValue _ _ (norm _)
  · exact ⟨s.normBound, by omega, by unfold Layer1.slotValue; rw [sub_self]⟩

section Rows

variable {s : Layer1.Shape} {bar : ℕ → ℕ → Gl} {S : Layer1.Setup Gl}
  {relation : ProductionKey.LogicalRelation lw pf}
  {ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf)}
  {P : PiRLC.v1_2.InputBinding.InputInstance lw pf} {z : ℕ → Gl}
  {w : PaperAlgebra.Assignment (logicalWidth := lw) (publicFits := pf)}

theorem key_row (shape : ParentShape s lw) (honest : HonestSetup s S relation ajtai)
    (assign : assignmentOf (lw := lw) (pf := pf) s z = w)
    (commit : (PaperAlgebra.semantics ajtai).commit w = P.commitment) (ρ : Fin productionProfile.commitmentWidth) :
    modulus ∣ (Layer1.relationRows s bar S (statementOf P)).residual (Layer1.witnessBlock s z) ρ := by
  rw [residual_eq (pf := pf) z shape ρ (fun b => ajtai ρ b) (P.commitment ρ) (key_weight_eq shape honest ρ)
    (key_target_eq shape P ρ), assign]
  exact dvd_of_ringSum_eq _ _ _ (congrFun commit ρ)

/-- A row whose weights are `barRing (u b)` and whose target's coefficients are the ring sum. -/
theorem bar_row (shape : ParentShape s lw) (assign : assignmentOf (lw := lw) (pf := pf) s z = w) {ρ : ℕ}
    (u : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw)) → RingF) (y : RingF)
    (weight : ∀ b : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw)),
      (Layer1.relationRows s bar S (statementOf P)).weight ρ b = toPolynomial (barRing (u b)))
    (target : (Layer1.relationRows s bar S (statementOf P)).target ρ = toPolynomial y)
    (sum : ∀ p, gl (y p) = ∑ b, ∑ l, ∑ e, gl (u b l) *
      gl (Phi81Relation.EvaluationHomomorphism.CarrierAction.assignmentBlock w b e) *
        gl (phi81Kernel.weight p l e)) :
    modulus ∣ (Layer1.relationRows s bar S (statementOf P)).residual (Layer1.witnessBlock s z) ρ := by
  rw [residual_eq (pf := pf) z shape ρ (fun b => barRing (u b)) y (fun b _ => weight b) target, assign]
  apply dvd_of_ringSum_eq
  funext p
  change gl _ = gl (y p)
  rw [ringFSum_apply, sum p]
  exact Finset.sum_congr rfl fun b _ => ringFMul_barRing _ _ p

/-- The evaluation family of a CE claim that holds. -/
theorem family_eq (source : P.constraintSystem = PiRLC.v1_2.InputBinding.relationSource relation)
    (evals : (PaperAlgebra.semantics ajtai).evaluations P.constraintSystem w P.point = P.evaluations) :
    family P = PaperAlgebra.evaluationFamily (PiRLC.v1_2.InputBinding.relationSource relation) w P.point := by
  rw [family, ← evals, source]
  rfl

theorem evalK_row (shape : ParentShape s lw) (barEq : ∀ p l : Fin ringDegree, bar p l = gl (nativeBarEntry p l))
    (assign : assignmentOf (lw := lw) (pf := pf) s z = w)
    (source : P.constraintSystem = PiRLC.v1_2.InputBinding.relationSource relation)
    (evals : (PaperAlgebra.semantics ajtai).evaluations P.constraintSystem w P.point = P.evaluations)
    (part : ℕ) (small : part < 2) (π : Layer1.Quad Gl →+ Gl) (hπ : ∀ q, π q = sel part (q.re, q.im))
    (scale : ∀ q x, π (q * ⟨x, 0⟩) = π q * x) :
    modulus ∣ (Layer1.relationRows s bar S (statementOf P)).residual (Layer1.witnessBlock s z) (s.evalKRow part) := by
  apply bar_row shape assign _ _ (evalK_weight_eq shape barEq P part small π hπ) (evalK_target_eq shape P part small)
  intro p
  have value : sel part ((statementOf P).evalK p) = π (toQuad ((family P).pad p)) := by
    rw [hπ]
    simp only [sel, statementOf]
    rw [dite_eq_left p.isLt]
    rfl
  change sel part ((statementOf P).evalK p) = _
  rw [value, family_eq source evals, show PiRLC.v1_2.InputBinding.relationSource relation =
      PaperAlgebra.relationSource (Spec.Folding.PiCCS.CanonicalRowLayout.layout cubeVariables
        (Phi81CarrierLayout.carrierWidth lw) relation.cubeFits) relation.system from rfl,
    PaperAlgebra.evaluationFamily_eq_paper]
  exact pad_identity relation w P.point π scale p

theorem evalA_row (shape : ParentShape s lw) (barEq : ∀ p l : Fin ringDegree, bar p l = gl (nativeBarEntry p l))
    (honest : HonestSetup s S relation ajtai) (assign : assignmentOf (lw := lw) (pf := pf) s z = w)
    (source : P.constraintSystem = PiRLC.v1_2.InputBinding.relationSource relation)
    (evals : (PaperAlgebra.semantics ajtai).evaluations P.constraintSystem w P.point = P.evaluations)
    (part : ℕ) (small : part < 2) (π : Layer1.Quad Gl →+ Gl) (hπ : ∀ q, π q = sel part (q.re, q.im))
    (scale : ∀ q x, π (q * ⟨x, 0⟩) = π q * x) (j : Fin productionProfile.ccsMatrices) :
    modulus ∣ (Layer1.relationRows s bar S (statementOf P)).residual (Layer1.witnessBlock s z)
      (s.evalARow j part) := by
  apply bar_row shape assign _ _ (evalA_weight_eq shape barEq honest P part small π hπ j)
    (evalA_target_eq shape P part small j)
  intro p
  have value : sel part ((statementOf P).evalA j p) = π (toQuad ((family P).matrix j p)) := by
    rw [hπ]
    simp only [sel, statementOf]
    rw [dite_eq_left (show j.val < productionProfile.ccsMatrices ∧ p.val < ringDegree from ⟨j.isLt, p.isLt⟩)]
    rfl
  change sel part ((statementOf P).evalA j p) = _
  rw [value, family_eq source evals, show PiRLC.v1_2.InputBinding.relationSource relation =
      PaperAlgebra.relationSource (Spec.Folding.PiCCS.CanonicalRowLayout.layout cubeVariables
        (Phi81CarrierLayout.carrierWidth lw) relation.cubeFits) relation.system from rfl,
    PaperAlgebra.evaluationFamily_eq_paper]
  exact evaluate_coefficient relation _ w P.point p π scale

theorem public_row (shape : ParentShape s lw) (assign : assignmentOf (lw := lw) (pf := pf) s z = w)
    (pub : (PaperAlgebra.semantics ajtai).projectPublicInput w = P.publicInput) {j : ℕ}
    (hj : j < PaperAlgebra.publicRingColumns) :
    modulus ∣ (Layer1.relationRows s bar S (statementOf P)).residual (Layer1.witnessBlock s z) (s.publicRow j) := by
  have jlt : j < Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw) := by
    have fits := pf
    rw [Phi81CarrierLayout.blockCount_carrierWidth]
    change 54 * 5 ≤ Phi81ColumnLayout.blockCount lw * 54 at fits
    change j < 5 at hj
    omega
  let block : Fin (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw)) := ⟨j, jlt⟩
  let y : RingF := fun l => P.publicInput ⟨ringDegree * j + l, public_lt hj l.isLt⟩
  rw [show s.publicRow j = s.publicRow block from rfl,
    residual_eq (pf := pf) z shape _ (fun b => if b = block then ringFOne else ringFZero) y
      (fun b _ => public_weight_eq block b) (public_target_eq shape P hj), assign,
    Finset.sum_eq_single block (fun b _ other => by rw [ite_eq_right other, toPolynomial_zero, zero_mul])
      (fun outside => absurd (Finset.mem_univ block) outside), ite_eq_left rfl, toPolynomial_one, one_mul]
  have same : Phi81Relation.EvaluationHomomorphism.CarrierAction.assignmentBlock w block = y := by
    funext l
    refine Eq.trans ?_ (congrFun pub ⟨ringDegree * j + l, public_lt hj l.isLt⟩)
    exact congrArg w (Fin.ext (show j * ringDegree + l.val = ringDegree * j + l.val by ring))
  rw [same, sub_self]
  exact dvd_zero _

end Rows

/-- SuperNeo v1.2 Definition 21: the converse of `parent_holds`. A parent claim that holds with a
witness within the key's norm bound and zero on the blocks past the key's has its statement in its
layer-1 relation, with a witness that `assignmentOf` reads back as the same assignment. -/
theorem parent_complete (s : Layer1.Shape) (bar : ℕ → ℕ → Gl) (S : Layer1.Setup Gl)
    (relation : ProductionKey.LogicalRelation lw pf)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
    (P : PiRLC.v1_2.InputBinding.InputInstance lw pf)
    (shape : ParentShape s lw) (barEq : ∀ p l : Fin ringDegree, bar p l = gl (nativeBarEntry p l))
    (honest : HonestSetup s S relation ajtai)
    (source : P.constraintSystem = PiRLC.v1_2.InputBinding.relationSource relation)
    {w : PaperAlgebra.Assignment (logicalWidth := lw) (publicFits := pf)}
    (holds : Spec.CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams P w)
    (norm : ∀ c, centeredMagnitude (w c) ≤ s.normBound)
    (support : ∀ c : Fin (Phi81CarrierLayout.carrierWidth lw), s.blocks ≤ c.val / ringDegree → w c = 0) :
    ∃ z, Layer1.Relation s modulus bar S (statementOf P) z ∧ assignmentOf (lw := lw) (pf := pf) s z = w := by
  obtain ⟨⟨commit, pub, -⟩, -, evals⟩ := holds
  have assign := assignmentOf_witnessOf shape support
  refine ⟨witnessOf s w, ⟨fun x _ => witnessOf_norm norm x, fun ρ inside => ?_⟩, assign⟩
  have rows : ρ < 37 := by rw [← rows_eq shape]; exact inside
  have kappa : s.kappa = 22 := shape.kappa
  have matrices : s.matrices = 4 := shape.matrices
  by_cases key : ρ < 22
  · exact key_row (ρ := ⟨ρ, key⟩) shape honest assign commit
  by_cases eval : ρ < 24
  · rcases (show ρ = s.evalKRow 0 ∨ ρ = s.evalKRow 1 by unfold Layer1.Shape.evalKRow; omega) with rfl | rfl
    · exact evalK_row shape barEq assign source evals 0 (by decide) reHom (fun _ => rfl) reHom_scale
    · exact evalK_row shape barEq assign source evals 1 (by decide) imHom (fun _ => rfl) imHom_scale
  by_cases family : ρ < 32
  · have j_lt : (ρ - 24) / 2 < productionProfile.ccsMatrices := by change (ρ - 24) / 2 < 4; omega
    rcases (show ρ = s.evalARow ((ρ - 24) / 2) 0 ∨ ρ = s.evalARow ((ρ - 24) / 2) 1 by
        unfold Layer1.Shape.evalARow; omega) with same | same
    · rw [same]
      exact evalA_row (j := ⟨(ρ - 24) / 2, j_lt⟩) shape barEq honest assign source evals 0 (by decide) reHom
        (fun _ => rfl) reHom_scale
    · rw [same]
      exact evalA_row (j := ⟨(ρ - 24) / 2, j_lt⟩) shape barEq honest assign source evals 1 (by decide) imHom
        (fun _ => rfl) imHom_scale
  · rw [show ρ = s.publicRow (ρ - 32) by unfold Layer1.Shape.publicRow; omega]
    exact public_row shape assign pub (show ρ - 32 < 5 by omega)

end CompressionLean.Final
