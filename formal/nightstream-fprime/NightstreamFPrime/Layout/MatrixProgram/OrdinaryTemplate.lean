import NightstreamFPrime.Layout.MatrixProgram.Affine

/-! An explicit ordinary-row template for direct CCS gadgets. Its rows are
matrix-program data, not extra physical R1CS constraints or witness values. -/

namespace NightstreamFPrime.Layout.MatrixProgram.OrdinaryTemplate

private def combination (row : R1CS.Row) (port : Fin 3) : R1CS.LinearCombination :=
  if port.val = 0 then row.a else if port.val = 1 then row.b else row.c

def ofSemantic {count : Nat} (rows : Fin count → R1CS.Row) : Affine.Table :=
  Affine.Table.ofSemantic fun index : Fin (count * 3) =>
    let decoded : Fin count × Fin 3 := Fin.decodeProd index
    combination (rows decoded.1) decoded.2

def row? (table : Affine.Table) (index : Nat) : Option R1CS.Row := do
  let a ← table.combination? (3 * index)
  let b ← table.combination? (3 * index + 1)
  let c ← table.combination? (3 * index + 2)
  pure ⟨a, b, c⟩

private theorem cell {count : Nat} (rows : Fin count → R1CS.Row) (index : Fin count) (port : Fin 3) :
    (ofSemantic rows).combination? (3 * index.val + port.val) = some (combination (rows index) port) := by
  have exact := Affine.Table.combination?_ofSemantic
    (fun entry : Fin (count * 3) =>
      let decoded : Fin count × Fin 3 := Fin.decodeProd entry
      combination (rows decoded.1) decoded.2) (Fin.encodeProd (index, port))
  simp only [Fin.decodeProd_encodeProd] at exact
  simpa [ofSemantic, Fin.encodeProd] using exact

/-- The template decoder reconstructs the three exact affine row ports. -/
theorem row?_ofSemantic {count : Nat} (rows : Fin count → R1CS.Row) (index : Fin count) :
    row? (ofSemantic rows) index.val = some (rows index) := by
  have a := cell rows index (0 : Fin 3)
  have b := cell rows index (1 : Fin 3)
  have c := cell rows index (2 : Fin 3)
  simp [combination] at a b c
  simp only [row?, a, b, c]
  rfl

end NightstreamFPrime.Layout.MatrixProgram.OrdinaryTemplate
