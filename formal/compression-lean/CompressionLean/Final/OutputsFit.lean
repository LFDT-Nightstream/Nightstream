import CompressionLean.Final.Agrees

/-!
Owns `outputsFit_of_shape`: the terminal package's outputs fit every key
whose relation has the parent shape (`ParentShape`) and the parent point's
size. The shape is a premise of the production bound, so `OutputsFit` adds
only `rel.pointVariables = productionShape.cubeVariables`, which the layer-1
model does not read.

`load` keeps each output list's length (`load_outputs`), and the lengths of
`Terminal.outputs` are products of the parent's sizes.
-/

namespace CompressionLean.Final

open CompressionLean.Replay (D)
open NightstreamFPrime.Lifecycle (productionShape)

theorem mapM_ok_length {α β ε : Type} {f : α → Except ε β} :
    ∀ {l : List α} {r : List β}, l.mapM f = .ok r → r.length = l.length
  | [], r, h => by simp [List.mapM_nil, pure, Except.pure] at h; subst h; rfl
  | a :: l, r, h => by
      rw [List.mapM_cons] at h
      simp only [bind, Except.bind] at h
      split at h
      · cases h
      · split at h
        · cases h
        · rename_i bs hbs
          cases h
          simp [mapM_ok_length hbs]

theorem outputColumns_size {layout : NightstreamFPrime.Export.Package.PhysicalLayout} {index : Array (Option ℕ)}
    {list : List NightstreamFPrime.Circuit.Expr} {columns : Array ℕ}
    (h : Circuit.Package.outputColumns layout index list = .ok columns) : columns.size = list.length := by
  unfold Circuit.Package.outputColumns at h
  rw [Array.mapM_eq_mapM_toList] at h
  simp only [Functor.map, Except.map] at h
  split at h
  · cases h
  · rename_i bs hbs
    cases h
    simp [mapM_ok_length hbs]

/-- `load` keeps the length of each claim list of the output map. -/
theorem load_outputs {pkg : NightstreamFPrime.Export.Package.CircuitPackage} {map : Terminal.Outputs}
    {loaded : Circuit.Package.Loaded} (h : Circuit.Package.load pkg map = .ok loaded) :
    loaded.outputs.commitment.size = map.commitment.length ∧
      loaded.outputs.publicInput.size = map.publicInput.length ∧ loaded.outputs.point.size = map.point.length ∧
      loaded.outputs.evalK.size = map.eval_K.length ∧ loaded.outputs.evalA.size = map.eval_A.length := by
  unfold Circuit.Package.load at h
  simp only [bind, Except.bind] at h
  split at h
  · repeat' (split at h; · cases h)
    cases h
    refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> (apply outputColumns_size; assumption)
  · cases h

/-- **The shipped outputs fit.** A relation with the parent shape and the
parent point's size fits the claim that the terminal package outputs. -/
theorem outputsFit_of_shape {package : Circuit.Package.Loaded} {rel : Replay.Relation} {lw : ℕ}
    (loads : Circuit.Package.load (Terminal.circuitPackage ()) Terminal.outputs = .ok package)
    (shape : ParentShape (Bridge.shapeOf rel) lw) (point : rel.pointVariables = productionShape.cubeVariables) :
    OutputsFit package rel := by
  obtain ⟨commitment, publicInput, points, evalK, evalA⟩ := load_outputs loads
  have kappa : rel.kappa = _ := shape.kappa
  have matrices : rel.matrices = _ := shape.matrices
  have blocks : rel.publicBlocks = _ := shape.publicBlocks
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rw [commitment, kappa]
    simp [Terminal.outputs, Function.comp_def]
    decide
  · rw [publicInput, blocks]
    simp [Terminal.outputs]
    decide
  · rw [points, point]
    simp [Terminal.outputs, Terminal.extension, Function.comp_def]
  · rw [evalK]
    simp [Terminal.outputs, Terminal.extension, Function.comp_def]
    decide
  · rw [evalA, matrices]
    simp [Terminal.outputs, Terminal.extension, Function.comp_def]
    decide

end CompressionLean.Final
