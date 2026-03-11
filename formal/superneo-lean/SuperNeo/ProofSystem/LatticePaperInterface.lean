import SuperNeo.ProofSystem.LatticePaper

/-!
Contract interface for `SuperNeo.ProofSystem.LatticePaper`.

Spec: `specs/ProofSystem/LatticePaper.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
Anchors:
- Appendix B.2 concrete lattice parameters.
- Theorem 2 concrete Goldilocks / `paperCarrier` specialization.
-/

namespace SuperNeo
namespace ProofSystem.LatticePaperInterface

noncomputable section

abbrev goldilocksPaperAjtaiParams := SuperNeo.ProofSystem.goldilocksPaperAjtaiParams

theorem goldilocksPaperAjtaiParams_kappa
  (messageLength : Nat) :
  (goldilocksPaperAjtaiParams messageLength).kappa =
    SuperNeo.Parameters.Goldilocks.kappa :=
  SuperNeo.ProofSystem.goldilocksPaperAjtaiParams_kappa messageLength

theorem goldilocksPaperAjtaiParams_msgLen
  (messageLength : Nat) :
  (goldilocksPaperAjtaiParams messageLength).msgLen = messageLength :=
  SuperNeo.ProofSystem.goldilocksPaperAjtaiParams_msgLen messageLength

theorem goldilocksPaperAjtaiParams_bindingNormBound
  (messageLength : Nat) :
  (goldilocksPaperAjtaiParams messageLength).bindingNormBound =
    SuperNeo.Parameters.Goldilocks.B :=
  SuperNeo.ProofSystem.goldilocksPaperAjtaiParams_bindingNormBound messageLength

theorem goldilocksPaperAjtaiParams_relaxedExpansion
  (messageLength : Nat) :
  (goldilocksPaperAjtaiParams messageLength).relaxedExpansion =
    SuperNeo.Parameters.Goldilocks.T :=
  SuperNeo.ProofSystem.goldilocksPaperAjtaiParams_relaxedExpansion messageLength

theorem goldilocksPaperAjtaiParams_relaxedExpansion_pos
  (messageLength : Nat) :
  0 < (goldilocksPaperAjtaiParams messageLength).relaxedExpansion :=
  SuperNeo.ProofSystem.goldilocksPaperAjtaiParams_relaxedExpansion_pos messageLength

theorem goldilocksPaperAjtaiParams_three_d_le
  (messageLength : Nat) :
  3 * d ≤ (goldilocksPaperAjtaiParams messageLength).relaxedExpansion :=
  SuperNeo.ProofSystem.goldilocksPaperAjtaiParams_three_d_le messageLength

theorem goldilocksPaperAjtaiParams_sideConditions
  {messageLength : Nat}
  (hMsg : 0 < messageLength) :
  (goldilocksPaperAjtaiParams messageLength).SideConditions :=
  SuperNeo.ProofSystem.goldilocksPaperAjtaiParams_sideConditions hMsg

def MSISToAjtaiReductions_ofGoldilocksPaperCarrierAndMSISBoundary
  (messageLength : Nat)
  (hMsis :
    SuperNeo.ProofSystem.MSISHardnessBoundary
      (goldilocksPaperAjtaiParams messageLength)) :
  SuperNeo.ProofSystem.MSISToAjtaiReductions
    (goldilocksPaperAjtaiParams messageLength) :=
  SuperNeo.ProofSystem.MSISToAjtaiReductions.ofGoldilocksPaperCarrierAndMSISBoundary
    messageLength
    hMsis

noncomputable def MSISToAjtaiReductions_ofGoldilocksPaperCarrierAndMSISHardness
  (messageLength : Nat)
  (hMsis :
    SuperNeo.ProofSystem.MSISHardnessAssumption
      (goldilocksPaperAjtaiParams messageLength)) :
  SuperNeo.ProofSystem.MSISToAjtaiReductions
    (goldilocksPaperAjtaiParams messageLength) :=
  SuperNeo.ProofSystem.MSISToAjtaiReductions.ofGoldilocksPaperCarrierAndMSISHardness
    messageLength
    hMsis

end
end ProofSystem.LatticePaperInterface
end SuperNeo
