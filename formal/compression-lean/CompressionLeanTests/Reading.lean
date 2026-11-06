import CompressionLean

/-! Non-vacuity gate for each slot's `Spec.OptionB.Reads` (`reads_of_witness`). No theorem here joins
it with `Bridge.merkleBinds_of_root`. -/

namespace CompressionLeanTests.Reading

open CompressionLean CompressionLean.Spec CompressionLean.Spec.OptionB CompressionLean.Sim.OptionB
open CompressionLean.Lowering CompressionLean.Circuit CompressionLean.Bridge CompressionLean.Layer1
open NightstreamFPrime.Export.OptionB.StepPackage (Package)

theorem reads_of_witness {finish : Package} {stateKeys : List Replay.Word} {keys : Fin 2 → Layer1Key}
    (S : Fin 2 → Setup Gl) (facts : ∀ slot, KeyFacts (keys slot))
    (fits : ∀ slot, (S slot).Fits (shapeOf (keys slot).relation))
    (describes : ∀ slot, (structureOf (keys slot).relation.key.layout).Describes (shapeOf (keys slot).relation) (S slot))
    (L0 L1 : ℕ) (fit : OptionB.fits finish = true) (keysLength : stateKeys.length ≤ 8)
    (x : StateWords) (wit : ℕ → Gl)
    (holds : ((shrinkRun finish stateKeys keys x).ccs Block.poseidonTemplate (counts finish stateKeys keys)).Relation
      wit)
    (slot : Fin 2) (cands : List (P0Tables Gl)) (short0 : cands.length ≤ L0) (p1 : List (P1Tables Cubic))
    (short1 : p1.length ≤ L1)
    (whir : ∀ points t', Runs (Replay.verify (keys slot).relation
        (seedWith finish stateKeys keys slot (wordsOf finish stateKeys keys x wit))
        (statementWith finish stateKeys slot (wordsOf finish stateKeys keys x wit))
        (proofWith finish stateKeys keys slot (wordsOf finish stateKeys keys x wit))) start points t' →
      WhirOpens (shapeOf (keys slot).relation)
        (proofWith finish stateKeys keys slot (wordsOf finish stateKeys keys x wit)) points cands p1)
    (merkle : MerkleBinds (keys slot).relation (S slot)
      (proofWith finish stateKeys keys slot (wordsOf finish stateKeys keys x wit)).leaves) :
    ∃ p, Reads finish stateKeys keys slot (S slot) L0 L1 x wit p := by
  rw [counts, lower_counts _ [] (List.ofFn x)] at holds
  obtain ⟨_env, -, slots⟩ := program_gives fit keysLength facts
    (fun slot => named_of (facts slot) (fits slot) (describes slot)) x wit holds
  obtain ⟨-, points, t', run⟩ := slots slot
  obtain ⟨tr, ds, rest, -, ⟨t₄, run₄, draws₄⟩, -, -, reach⟩ :=
    replay_accepts_implies_path (keys slot).relation (S slot) (facts slot).relation (fits slot) (describes slot)
      _ _ _ merkle L0 L1 cands short0 p1 short1 run (whir _ _ run)
  rw [Sim.phi_eq_modulus] at reach
  obtain ⟨p, -, pathDraws, pathMsgs⟩ := reach
  refine ⟨p, cands, short0, p1, short1, pathMsgs, ?_, whir, merkle⟩
  rw [show Spec.eval _ start = _ from run₄, draws₄]
  exact pathDraws

end CompressionLeanTests.Reading
