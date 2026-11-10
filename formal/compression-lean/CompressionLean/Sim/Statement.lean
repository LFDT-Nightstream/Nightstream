import CompressionLean.Sim.EvalK
import CompressionLean.Sim.Views
import CompressionLean.Spec.Program
import CompressionLean.Spec.Relation
import CompressionLean.Bridge.Relation
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.NoZeroDivisors
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetConvention
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.BaseLinear

/-! The statement words that a final program reads first (`statementHandles`), and the two
constants that its layer-1 calls share with the native relation (`phi_eq_modulus`, `barOf_eq`). -/

namespace CompressionLean.Sim

open CompressionLean.Spec
open CompressionLean.Circuit
open CompressionLean.Terminal.Program (statementWords)
open CompressionLean.Replay (Word)
open CompressionLean.Bridge
open NightstreamFPrime
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial (modulus)

def statementOf (c : Ctx NightstreamFPrime.Spec.F) : Spec.StateWords := fun k => c.stmt.getD k 0

def ReadsStatement (c : Ctx NightstreamFPrime.Spec.F) : Prop := ∀ k : Fin 9, c.val k = statementOf c k

private theorem pubs_run {δ : Type} : ∀ (xs : List δ) (p : Builder), (xs.mapM fun _ => Build.pub).run p =
    (List.range' p.values xs.length,
      { instrs := p.instrs ++ Array.replicate xs.length Instr.pub, values := p.values + xs.length })
  | [], p => by simp; rfl
  | x :: xs, p => by
      rw [List.mapM_cons]
      show ((xs.mapM fun _ => Build.pub).run _ >>= fun r => pure (p.values :: r.1, r.2)) = _
      rw [pubs_run xs]
      show (p.values :: List.range' (p.values + 1) xs.length, _) = _
      simp only [List.length_cons, List.range'_succ, Prod.mk.injEq, true_and, Builder.mk.injEq, Instr.outputs]
      refine ⟨Array.toList_inj.mp ?_, by omega⟩
      simp [List.replicate_succ]

theorem statementHandles_run (p : Builder) : statementHandles.run p =
    ((List.range' p.values 9).toArray,
      { instrs := p.instrs ++ Array.replicate 9 Instr.pub, values := p.values + 9 }) := by
  rw [statementHandles, Array.mapM_eq_mapM_toList]
  show (fun r => (r.1.toArray, r.2)) ((List.mapM (fun _ => Build.pub) (Array.range statementWords).toList).run p) = _
  rw [pubs_run]
  simp [statementWords]

instance statementHandles_appends : Appends statementHandles := by unfold statementHandles; appends

theorem statementHandles_ensures {c : Ctx NightstreamFPrime.Spec.F} {n : ℕ} :
    Ensures c n statementHandles fun s m => s = (List.range' n 9).toArray ∧ m = n + 9 :=
  fun p hp _ _ _ _ _ _ => by
    rw [statementHandles_run, hp]
    exact ⟨rfl, rfl⟩

section Decode

variable {F : Type} [CommRing F] {c : Ctx F} {w : Var → Word} {m : ℕ}

private theorem isKs_map (hw : Decodes c w) {xs : Array KVar} (h : ∀ p ∈ xs, p.1 < m ∧ p.2 < m) :
    IsKs c m xs (xs.map (Prod.map w w)) := by
  refine ⟨(Array.size_map ..).symm, fun i hi => ?_⟩
  rw [getElem!_pos xs i hi, getElem!_pos _ i (by simpa using hi), Array.getElem_map]
  have := h _ (Array.getElem_mem hi)
  exact ⟨⟨this.1, hw _⟩, ⟨this.2, hw _⟩⟩

theorem isClaim_decode (hw : Decodes c w) {v : ClaimView} (below : ∀ h ∈ v.words, h < m) :
    IsClaim c m v (v.decode w) := by
  have pair : ∀ p ∈ v.point ++ v.evalK ++ v.evalA.flatten, p.1 < m ∧ p.2 < m := fun p hp =>
    ⟨below _ (Array.mem_append_right _ (Array.mem_flatMap.mpr ⟨p, hp, by simp⟩)),
      below _ (Array.mem_append_right _ (Array.mem_flatMap.mpr ⟨p, hp, by simp⟩))⟩
  have row : ∀ {xss : Array (Array Var)}, (∀ h ∈ xss.flatten, h < m) →
      xss.size = (xss.map (·.map w)).size ∧ ∀ i < xss.size, IsWords c m xss[i]! (xss.map (·.map w))[i]! :=
    fun h => ⟨(Array.size_map ..).symm, fun i hi => by
      rw [getElem!_pos _ i hi, getElem!_pos _ i (by simpa using hi), Array.getElem_map]
      exact isWords_map hw fun j hj => h _ (Array.mem_flatten.mpr ⟨_, Array.getElem_mem hi, Array.getElem_mem hj⟩)⟩
  refine ⟨row fun h hh => below h (Array.mem_append_left _ (Array.mem_append_left _ hh)),
    row fun h hh => below h (Array.mem_append_left _ (Array.mem_append_right _ hh)),
    isKs_map hw fun p hp => pair p (by simp [hp]), isKs_map hw fun p hp => pair p (by simp [hp]),
    ⟨(Array.size_map ..).symm, fun j hj => ?_⟩⟩
  rw [getElem!_pos _ j hj, getElem!_pos _ j (by simpa [ClaimView.decode] using hj)]
  simp only [ClaimView.decode, Array.getElem_map]
  exact isKs_map hw fun p hp =>
    pair p (Array.mem_append_right _ (Array.mem_flatten.mpr ⟨_, Array.getElem_mem hj, hp⟩))

end Decode

theorem phi_eq_modulus : phi = modulus := by
  unfold phi modulus
  rw [add_assoc]
  rfl

theorem barOf_eq (p l : Fin Spec.ringDegree) :
    barOf p l = Spec.gl (Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel.nativeBarEntry p l) := by
  unfold barOf Replay.bar
  rw [dite_eq_left ⟨p.isLt, l.isLt⟩, Replay.Word.toF_ofNat]
  exact ZMod.natCast_zmod_val (show Gl from _)

end CompressionLean.Sim
