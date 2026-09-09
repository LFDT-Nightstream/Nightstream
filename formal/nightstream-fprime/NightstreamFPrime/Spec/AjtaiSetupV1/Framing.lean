import NightstreamFPrime.Spec.AjtaiSetupV1

/-! The actual setup descriptor uniquely determines its dimensions and seed.
The dimension bounds are necessary because descriptor words are field values.
This is an encoding theorem; it makes no pseudorandomness assumption. -/

namespace NightstreamFPrime.Spec.AjtaiSetupV1.Setup

private theorem authorityNats_canonical {rows columns : Nat}
    (setup : Setup rows columns)
    (rowsBound : rows < goldilocksModulus)
    (columnsBound : columns < goldilocksModulus) :
    ∀ word ∈ setup.authorityNats, word < goldilocksModulus := by
  intro word member
  unfold authorityNats at member
  simp only [List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with ((rfl | inId) | (rfl | rfl | rfl)) | inSeed
  · rw [setupIdBytes_length]
    decide
  · have tagCanonical : ∀ byte ∈ setupIdBytes, byte < 256 := by decide
    exact Nat.lt_trans (tagCanonical word inId) (by decide)
  · exact rowsBound
  · exact columnsBound
  · rw [setup.seed.length_eq]
    decide
  · exact Nat.lt_trans (setup.seed.canonical word inSeed) (by decide)

private theorem recover_canonical_words (values : List Nat)
    (canonical : ∀ word ∈ values, word < goldilocksModulus) :
    (values.map Poseidon2.ofNat).map Fin.val = values := by
  have recovered : (values.map Poseidon2.ofNat).map Fin.val = values.map id := by
    rw [List.map_map]
    apply List.map_congr_left
    intro word member
    exact Nat.mod_eq_of_lt (canonical word member)
  simpa using recovered

private theorem authorityNats_eq_iff
    {leftRows leftColumns rightRows rightColumns : Nat}
    (left : Setup leftRows leftColumns) (right : Setup rightRows rightColumns) :
    left.authorityNats = right.authorityNats ↔
      leftRows = rightRows ∧ leftColumns = rightColumns ∧
        left.seed.bytes = right.seed.bytes := by
  simp [authorityNats, List.append_assoc, left.seed.length_eq, right.seed.length_eq]

/-- Equal canonical setup words force equal key dimensions and identical seed
bytes. In particular, neither seed framing nor field reduction can alias two
in-range setup descriptors. -/
theorem authorityWords_eq_iff
    {leftRows leftColumns rightRows rightColumns : Nat}
    (left : Setup leftRows leftColumns) (right : Setup rightRows rightColumns)
    (leftRowsBound : leftRows < goldilocksModulus)
    (leftColumnsBound : leftColumns < goldilocksModulus)
    (rightRowsBound : rightRows < goldilocksModulus)
    (rightColumnsBound : rightColumns < goldilocksModulus) :
    left.authorityWords = right.authorityWords ↔
      leftRows = rightRows ∧ leftColumns = rightColumns ∧
        left.seed.bytes = right.seed.bytes := by
  constructor
  · intro sameWords
    have sameNats := congrArg (List.map Fin.val) sameWords
    unfold authorityWords at sameNats
    rw [recover_canonical_words _
      (authorityNats_canonical left leftRowsBound leftColumnsBound),
      recover_canonical_words _
        (authorityNats_canonical right rightRowsBound rightColumnsBound)] at sameNats
    exact (authorityNats_eq_iff left right).mp sameNats
  · intro same
    exact congrArg (List.map Poseidon2.ofNat)
      ((authorityNats_eq_iff left right).mpr same)

/-- The canonical descriptor determines every coefficient of the semantic key,
not only its dimensions or its field range. No key matrix is expanded. -/
theorem verifierKey_eq_of_authorityWords {rows columns : Nat}
    (left right : Setup rows columns)
    (rowsBound : rows < goldilocksModulus)
    (columnsBound : columns < goldilocksModulus)
    (sameWords : left.authorityWords = right.authorityWords) :
    left.verifierKey = right.verifierKey := by
  have sameSeed := ((authorityWords_eq_iff left right
    rowsBound columnsBound rowsBound columnsBound).mp sameWords).2.2
  funext row block lane
  apply Fin.ext
  change wideCoefficientNat left.seed.bytes row.val block.val lane.val =
    wideCoefficientNat right.seed.bytes row.val block.val lane.val
  rw [sameSeed]

end NightstreamFPrime.Spec.AjtaiSetupV1.Setup
