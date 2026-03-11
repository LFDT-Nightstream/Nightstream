import SuperNeo.Parameters
import SuperNeo.ProofSystem.Lattice
import SuperNeo.ProofSystem.LatticeReductionsDerived

namespace SuperNeo.ProofSystem

/--
Goldilocks Appendix B.2 lattice-parameter family for the active SuperNeo path.

The paper fixes `κ = 18`, `B = 2^14`, and `T = 216`. The Ajtai message length
remains an explicit parameter here because the current repo does not yet encode
one canonical in-repo choice for that dimension on the final theorem path.
-/
def goldilocksPaperAjtaiParams (messageLength : Nat) : AjtaiParams where
  ringDim := SuperNeo.Parameters.Goldilocks.kappa
  messageLength := messageLength
  bindingNormBound := SuperNeo.Parameters.Goldilocks.B
  relaxedExpansion := SuperNeo.Parameters.Goldilocks.T

theorem goldilocksPaperAjtaiParams_kappa
  (messageLength : Nat) :
  (goldilocksPaperAjtaiParams messageLength).kappa =
    SuperNeo.Parameters.Goldilocks.kappa := by
  rfl

theorem goldilocksPaperAjtaiParams_msgLen
  (messageLength : Nat) :
  (goldilocksPaperAjtaiParams messageLength).msgLen = messageLength := by
  rfl

theorem goldilocksPaperAjtaiParams_bindingNormBound
  (messageLength : Nat) :
  (goldilocksPaperAjtaiParams messageLength).bindingNormBound =
    SuperNeo.Parameters.Goldilocks.B := by
  rfl

theorem goldilocksPaperAjtaiParams_relaxedExpansion
  (messageLength : Nat) :
  (goldilocksPaperAjtaiParams messageLength).relaxedExpansion =
    SuperNeo.Parameters.Goldilocks.T := by
  rfl

theorem goldilocksPaperAjtaiParams_relaxedExpansion_pos
  (messageLength : Nat) :
  0 < (goldilocksPaperAjtaiParams messageLength).relaxedExpansion := by
  simpa [goldilocksPaperAjtaiParams_relaxedExpansion] using
    SuperNeo.Parameters.Goldilocks.T_pos

theorem goldilocksPaperAjtaiParams_three_d_le
  (messageLength : Nat) :
  3 * d ≤ (goldilocksPaperAjtaiParams messageLength).relaxedExpansion := by
  rw [goldilocksPaperAjtaiParams_relaxedExpansion]
  rw [SuperNeo.Parameters.Goldilocks.T_eq_216]
  decide

theorem goldilocksPaperAjtaiParams_sideConditions
  {messageLength : Nat}
  (hMsg : 0 < messageLength) :
  (goldilocksPaperAjtaiParams messageLength).SideConditions := by
  refine ⟨?_, hMsg, ?_, ?_⟩
  · simpa [goldilocksPaperAjtaiParams_kappa] using
      SuperNeo.Parameters.Goldilocks.kappa_pos
  · simpa [goldilocksPaperAjtaiParams_bindingNormBound] using
      SuperNeo.Parameters.Goldilocks.B_pos
  · exact goldilocksPaperAjtaiParams_relaxedExpansion_pos messageLength

namespace MSISToAjtaiReductions

/--
Canonical MSIS-to-Ajtai reduction package on the active Goldilocks Appendix B.2
paper-carrier path.
-/
def ofGoldilocksPaperCarrierAndMSISBoundary
  (messageLength : Nat)
  (hMsis : MSISHardnessBoundary (goldilocksPaperAjtaiParams messageLength)) :
  MSISToAjtaiReductions (goldilocksPaperAjtaiParams messageLength) :=
  ofPaperCarrierFromThreeDLeAndMSISBoundary
    (params := goldilocksPaperAjtaiParams messageLength)
    (goldilocksPaperAjtaiParams_three_d_le messageLength)
    (goldilocksPaperAjtaiParams_relaxedExpansion_pos messageLength)
    hMsis

/--
Canonical MSIS-to-Ajtai reduction package on the active Goldilocks Appendix B.2
paper-carrier path, reconstructed directly from the theorem-level MSIS hardness
assumption.
-/
noncomputable def ofGoldilocksPaperCarrierAndMSISHardness
  (messageLength : Nat)
  (hMsis : MSISHardnessAssumption (goldilocksPaperAjtaiParams messageLength)) :
  MSISToAjtaiReductions (goldilocksPaperAjtaiParams messageLength) :=
  ofGoldilocksPaperCarrierAndMSISBoundary
    messageLength
    (MSISHardnessBoundary.ofHardness hMsis)

end MSISToAjtaiReductions

end SuperNeo.ProofSystem
