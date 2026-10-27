import CompressionLean.Final.Agrees
import CompressionLean.Circuit.MirrorRead

/-!
Owns the reduction of `VerifyAgrees` to the verifier proper
(`verifyAgrees_of_core`): what `verifyWith` reads before it verifies, and
the key's static facts that the replay checks and the circuit does not.

- `KeyFits`: the named static premise on the key. Each field is a fact the
  replay checks and the circuit does not; it holds for a consistent key.
- `readProof_ok`: a proof view with the sizes of `ProofView.read` decodes
  to a proof that `readProof` accepts, under `KeyFits`.
- `VerifyCore` (open): on a run of `verify` whose checks hold, from a
  builder state that holds the seed, the claim and the proof, and whose
  decoded statement and proof pass the replay's static checks, the replay
  accepts.
-/

namespace CompressionLean.Final

open CompressionLean.Replay (TreeShape layerCount layerFactors layerDegree activeTrees partFactors)

/-- Every tree of `shapes` has a leaf layer and at most two numerator factors. -/
def TreesOk (shapes : Array TreeShape) : Prop := ∀ s ∈ shapes, s.depth ≠ 0 ∧ s.factors ≤ 2

theorem readGkr_ok {shapes : Array TreeShape} (trees : TreesOk shapes) {g : Circuit.GkrView}
    (shaped : g.Shaped shapes) (v : Circuit.Var → Replay.Gl) : Replay.readGkr (Decode.gkr v g) shapes = .ok () := by
  obtain ⟨roots, layers, each⟩ := shaped
  have any : shapes.any (fun s => s.depth == 0 || s.factors > 2) = false := by
    rw [Array.any_eq_false]
    intro i hi
    have := trees _ (Array.getElem_mem hi)
    simp [this.1]
    omega
  unfold Replay.readGkr
  simp only [any, Decode.gkr, Decode.pairs, Array.size_map, roots, layers, bne_self_eq_false, Bool.or_false,
    Bool.false_eq_true, ite_false]
  have ok : ∀ i ∈ List.range (layerCount shapes), Replay.layerShapeOk shapes (i + 1)
      ((Decode.gkr v g).layers[i]!) = true := by
    intro i hi
    have hi : i < g.layers.size := by rw [layers]; simpa using hi
    obtain ⟨rs, rsize, cs, csize⟩ := each i hi
    rw [getElem!_pos _ i (by simpa [Decode.gkr] using hi)]
    simp only [Decode.gkr, Array.getElem_map]
    unfold Replay.layerShapeOk
    have h1 : ((g.layers[i].rounds.map (Decode.exts v)).size == i + 1 - 1) = true := by simp [rs]
    have h2 : (g.layers[i].rounds.map (Decode.exts v)).all
        (fun r => r.size == layerDegree (layerFactors shapes (i + 1))) = true :=
      Array.all_eq_true'.mpr fun x hx => by
        obtain ⟨r, hr, rfl⟩ := Array.mem_map.mp hx
        simp [Decode.exts, rsize r hr]
    have h3 : ((g.layers[i].children.map (Decode.exts v)).size == (activeTrees shapes (i + 1)).size) = true := by
      simp [cs, Replay.layerFactors]
    have h4 : ((activeTrees shapes (i + 1)).zip (g.layers[i].children.map (Decode.exts v))).all
        (fun pair => pair.2.size == 2 * (partFactors shapes (i + 1) pair.1 + 1)) = true := by
      rw [Array.all_eq_true]
      intro j hj
      simp only [Array.size_zip, Array.size_map] at hj
      have hj' : j < g.layers[i].children.size := by omega
      simp only [Array.getElem_zip, Array.getElem_map, Decode.exts, Array.size_map]
      rw [csize j hj', getElem!_pos (layerFactors shapes (i + 1)) j (by rw [← cs]; exact hj')]
      simp [Replay.layerFactors]
    simp only [h1, h2, h3, h4, Bool.and_self]
  have all := List.all_eq_true.mpr ok
  simp only [Decode.gkr] at all
  rw [all]
  rfl

/-- A check loop whose checks pass on every entry passes. -/
theorem forIn_checks_ok {α : Type} (l : List α) (body : α → Unit → Except String (ForInStep Unit))
    (h : ∀ x ∈ l, body x () = .ok (.yield ())) : forIn l () body = .ok () := by
  induction l with
  | nil => rfl
  | cons x rest ih =>
      rw [List.forIn_cons, h x (by simp)]
      exact ih fun y hy => h y (by simp [hy])

theorem ensure_ok {ok : Bool} {what : String} (h : ok = true) :
    (Replay.ensure ok what : Except String Unit) = .ok () := by
  subst h
  rfl

theorem readOpening_ok {shape : Replay.WhirShape} {o : Circuit.OpeningView} (shaped : o.Shaped shape)
    (v : Circuit.Var → Replay.Gl) : Replay.readOpening shape (Decode.opening v o) = .ok () := by
  obtain ⟨rounds, evals, ood, evalSizes, initial, roundSizes, final, finalSum⟩ := shaped
  unfold Replay.readOpening
  have e1 : (Decode.opening v o).rounds.size == shape.rounds.size && (Decode.opening v o).evals.size ==
      shape.batches.size && (Decode.opening v o).initialOod.size == shape.oodSamples &&
      (shape.batches.zip (Decode.opening v o).evals).all (fun pair => pair.2.size == pair.1.size) := by
    simp only [Decode.opening, Decode.exts, Array.size_map, rounds, evals, ood, beq_self_eq_true, Bool.true_and]
    rw [Array.all_eq_true]
    intro i hi
    simp only [Array.size_zip, Array.size_map] at hi
    simp [Array.getElem_zip, Decode.exts, evalSizes i (by omega) (by omega)]
  rw [ensure_ok e1]
  rw [← Array.forIn_toList, forIn_checks_ok]
  · have e2 : (Decode.opening v o).initialSumcheck.size = shape.initialSumcheck.rounds := by
      simp [Decode.opening, Decode.pairs, initial]
    have e3 : (Decode.opening v o).finalPoly.size = shape.finalPolyLen := by
      simp [Decode.opening, Decode.exts, final]
    have e4 : (Decode.opening v o).finalSumcheck.size = shape.finalSumcheck.rounds := by
      simp [Decode.opening, Decode.pairs, finalSum]
    simp only [e2, e3, e4, beq_self_eq_true, ensure_ok rfl]
    rfl
  · intro pair hp
    obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.mp hp
    simp only [Array.length_toList, Array.size_zip, Decode.opening, Array.size_map] at hi
    obtain ⟨a, b⟩ := roundSizes i (by omega) (by omega)
    simp [Decode.opening, Decode.exts, Decode.pairs, a, b, ensure_ok]
    rfl

/-- **The decoded proof has the read shapes.** A proof view with the sizes
of `ProofView.read` decodes to a proof that `readProof` accepts, when the
key's trees are well formed. -/
theorem readProof_ok {rel : Replay.Relation} (early : TreesOk rel.earlyShapes) (late : TreesOk rel.lateShapes)
    {p : Circuit.ProofView} (shaped : p.Shaped rel) (v : Circuit.Var → Replay.Gl) :
    Replay.readProof rel (Decode.proof v p) = .ok () := by
  obtain ⟨hist, earlyShape, -, pairs, quotient, linear, linearAll, finals, partials, partialsAll, leaves, leafAll,
    lateShape, lateValues, p1, p0⟩ := shaped
  unfold Replay.readProof
  have la : (Decode.proof v p).linear.all (·.size == 2) = true := Array.all_eq_true'.mpr fun x hx => by
    obtain ⟨r, hr, rfl⟩ := Array.mem_map.mp hx
    simp [Decode.exts, linearAll r hr]
  have pa : (Decode.proof v p).partials.all (·.size == 2 ^ Replay.foldVars) = true :=
    Array.all_eq_true'.mpr fun x hx => by
      obtain ⟨r, hr, rfl⟩ := Array.mem_map.mp hx
      simp [Decode.exts, partialsAll r hr]
  have e1 : ((Decode.proof v p).histogram.size == 2 * rel.normBound + 1 &&
      (Decode.proof v p).earlyValues.pairs.size == rel.matrices && (Decode.proof v p).quotient.size == Replay.D - 1 &&
      (Decode.proof v p).linear.size == rel.cubeVars && (Decode.proof v p).linear.all (·.size == 2) &&
      (Decode.proof v p).leaves.size == rel.queries && (Decode.proof v p).lateValues.size == rel.matrices &&
      (Decode.proof v p).p0.size == 4 && (Decode.proof v p).p1.size == 4 && (Decode.proof v p).finals.size == 3 &&
      (Decode.proof v p).partials.size == 2 &&
      (Decode.proof v p).partials.all (·.size == 2 ^ Replay.foldVars)) = true := by
    rw [la, pa]
    simp [Decode.proof, Decode.pairs, Decode.exts, Decode.digest, hist, pairs, quotient, linear, leaves, lateValues,
      finals, partials]
  have e2 : (Decode.proof v p).leaves.all (fun leaf => leaf.values.size == (rel.kappa + Replay.runOracles) *
      2 ^ Replay.foldVars && leaf.path.size == rel.setupVars + 1 - Replay.foldVars) = true :=
    Array.all_eq_true'.mpr fun x hx => by
      obtain ⟨leaf, hl, rfl⟩ := Array.mem_map.mp hx
      simp [leafAll leaf hl]
  rw [ensure_ok e1]
  simp only [bind, Except.bind]
  rw [show (Decode.proof v p).early = Decode.gkr v p.early from rfl, readGkr_ok early earlyShape, ensure_ok e2,
    show (Decode.proof v p).late = Decode.gkr v p.late from rfl, readGkr_ok late lateShape,
    show (Decode.proof v p).p1Opening = Decode.opening v p.p1Opening from rfl, readOpening_ok p1,
    show (Decode.proof v p).p0Opening = Decode.opening v p.p0Opening from rfl, readOpening_ok p0]

/-! ## The key -/

/-- **The key fits the replay's static checks.** Each field is a fact of
the key that the replay checks and the circuit does not. -/
structure KeyFits (key : Circuit.Layer1Input) : Prop where
  /-- `readGkr`: every early tree has a leaf layer and at most two factors. -/
  early : TreesOk key.relation.earlyShapes
  /-- `readGkr`: every late tree has a leaf layer and at most two factors. -/
  late : TreesOk key.relation.lateShapes

/-! ## The reduction -/

/-- **The verifier proper agrees with the replay.** Open: the stage lemmas
will prove it. On a run of `verify` whose checks hold, from a builder
state with `n` values that holds the seed, the claim and the proof, and
whose decoded statement and proof pass the replay's static checks, the
replay accepts from `start`. -/
def VerifyCore (key : Circuit.Layer1Input) : Prop :=
  ∀ (c : Circuit.Ctx Gl) (v : Circuit.Var → Replay.Gl) (n : ℕ) (seed : Circuit.DigestVar)
    (claim : Circuit.ClaimView) (proof : Circuit.ProofView),
    Circuit.Accepts c n
      (Circuit.Permutes c.perm ∧ (∀ h, c.val h = Circuit.word (v h)) ∧ Circuit.DigestVar.Below n seed ∧
        (∀ h ∈ claim.words, h < n) ∧ ClaimFits claim key.relation ∧ proof.Below n ∧
        proof.Shaped key.relation ∧ Replay.readProof key.relation (Decode.proof v proof) = .ok ())
      (Circuit.verify key.relation key.p1Key key.p0Key seed claim proof) fun _ _ =>
        ∃ points t', Replay.Runs (Replay.verify key.relation (Decode.digest v seed) (Decode.statement v claim)
          (Decode.proof v proof)) Shipped.start points t'

/-- **`VerifyAgrees` from the verifier proper.** The handoff gives a seed
among the run's values (`handoff_below`), `ProofView.read` a proof among
them with the read sizes (`proofRead`), and `KeyFits` makes the decoded
proof pass `readProof`. -/
theorem verifyAgrees_of_core {key : Circuit.Layer1Input} (fits : KeyFits key) (core : VerifyCore key) :
    VerifyAgrees key := by
  intro c v n state claim
  unfold Circuit.verifyWith
  refine (Circuit.handoff_below state fun h => h.2.2.1).bind fun seed m₁ l₁ => ?_
  refine (Circuit.Accepts.proofRead _ _ _).bind fun proof m₂ l₂ => ?_
  refine ((core c v m₂ seed claim proof).pre fun h => ?_).bind fun _ m₃ _ => Circuit.Accepts.pure fun h => h.2
  obtain ⟨⟨⟨perm, words, -, below, fit⟩, hseed⟩, hproof, shaped⟩ := h
  exact ⟨perm, words, fun lane => lt_of_lt_of_le (hseed lane) l₂,
    fun h hh => lt_of_lt_of_le (below h hh) (l₁.trans l₂), fit, hproof, shaped,
    readProof_ok fits.early fits.late shaped v⟩

namespace Shipped

variable (sp : Shipped)

/-- **The circuit-replay agreement from the verifier proper.** -/
theorem circuitAgrees_of_core (fits : KeyFits sp.key) (core : VerifyCore sp.key)
    (fit : OutputsFit sp.package sp.key.relation) : sp.CircuitAgrees :=
  sp.circuitAgrees_of_verify (verifyAgrees_of_core fits core) fit

end Shipped

end CompressionLean.Final
