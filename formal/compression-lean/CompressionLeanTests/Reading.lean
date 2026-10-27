import CompressionLean

/-!
The reading gate. `lake build` runs it.

`production_bound` bounds an event in which every valid program witness has a
reading (`Reads`). If no witness could have a reading, the bound would say
nothing about a valid witness. `reads_of_witness` shows that the reading asks
for no more than its two named assumptions: a valid witness has a reading when
WHIR opens its candidates and its setup leaves bind (`MerkleBinds`). That
honest leaves bind is `Bridge.honest_leaves_bind`: under the honest key root
they meet `MerkleBinds` unless Poseidon2 collides at inputs computed from a
leaf.
-/

namespace CompressionLeanTests.Reading

open CompressionLean CompressionLean.Spec CompressionLean.Sim CompressionLean.Lowering CompressionLean.Checks
open CompressionLean.Circuit CompressionLean.Terminal.Package CompressionLean.Bridge CompressionLean.Layer1
open NightstreamFPrime

/-- A witness of the production final program's CCS has a reading, given only
WHIR's openings at the points of its replay and the binding of its setup
leaves. -/
theorem reads_of_witness {key : Layer1Key} {loaded : Loaded} {context : Array Replay.Word}
    {authority : Lifecycle.VerifierContext.Authority} (S : Setup Gl) (facts : KeyFacts key)
    (fits : S.Fits (shapeOf key.relation))
    (describes : (structureOf key.relation.key.layout).Describes (shapeOf key.relation) S) (L0 L1 : ℕ)
    (loads : load (CompressionLean.Terminal.circuitPackage ()) CompressionLean.Terminal.outputs = .ok loaded)
    (valid : (CompressionLean.Terminal.circuitPackage ()).CompactTemplatesValid)
    (hctx : ∀ lane : Fin 4,
      Replay.Word.cast (context.getD lane 0) = (Lifecycle.VerifierContext.digest authority).getD lane 0)
    (x : StateWords) (wit : ℕ → Gl)
    (holds : ((shrinkRun key loaded context x).ccs Block.poseidonTemplate (counts key loaded context)).Relation wit)
    (cands : List (P0Tables Gl)) (short0 : cands.length ≤ L0) (p1 : List (P1Tables Cubic))
    (short1 : p1.length ≤ L1)
    (whir : ∀ points t', Runs (Replay.verify key.relation (seedWith loaded context (wordsOf key loaded context x wit))
        (statementWith loaded context (wordsOf key loaded context x wit))
        (proofWith key loaded context (wordsOf key loaded context x wit))) start points t' →
      WhirOpens (shapeOf key.relation) (proofWith key loaded context (wordsOf key loaded context x wit)) points cands p1)
    (merkle : MerkleBinds key.relation S (proofWith key loaded context (wordsOf key loaded context x wit)).leaves) :
    ∃ p, Reads key loaded context S L0 L1 x wit p := by
  rw [counts, lower_counts _ [] (List.ofFn x)] at holds
  have named : ∀ s ∈ key.relation.key.layout.segments,
      s.classIndex < key.relation.key.layout.classes.size ∧ s.lane < Replay.D := fun s member => by
    obtain ⟨i, inside, rfl⟩ := Array.mem_iff_getElem.mp member
    have lane := Bridge.segment_lane key.relation S fits facts.relation.checked describes inside
    have named := facts.relation.checked.class_named i inside
    simp only [structureOf] at named
    rw [getElem!_pos key.relation.key.layout.segments i inside] at lane named
    exact ⟨named, lane⟩
  obtain ⟨env, -, -, points, t', run⟩ := program_gives loads valid hctx facts named x wit holds
  obtain ⟨tr, ds, rest, -, ⟨t₄, run₄, draws₄⟩, -, -, reach⟩ :=
    replay_accepts_implies_path key.relation S facts.relation fits describes _ _ _ merkle L0 L1 cands short0 p1
      short1 run (whir _ _ run)
  rw [phi_eq_modulus] at reach
  obtain ⟨p, -, pathDraws, pathMsgs⟩ := reach
  refine ⟨p, cands, short0, p1, short1, pathMsgs, ?_, whir, merkle⟩
  rw [show Spec.eval _ start = _ from run₄, draws₄]
  exact pathDraws

end CompressionLeanTests.Reading
