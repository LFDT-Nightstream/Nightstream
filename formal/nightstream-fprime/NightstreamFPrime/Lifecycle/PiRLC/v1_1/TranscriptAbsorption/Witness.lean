import NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption

/-! The scalar-domain leaf executes one canonical permutation after its
two domain words. These identities expose its witness program to transport. -/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption

open NightstreamFPrime.Circuit NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex

def permutationInput (interface : Interface) (source start : Nat) : EState :=
  Hash.absorbE (interface.initialState start) (constantWords (frameWords source))

private theorem frame_chunks (source : Nat) :
    Hash.inputChunks (constantWords (frameWords source)) = [constantWords (frameWords source)] := by
  norm_num [Hash.inputChunks, constantWords, frameWords, NightstreamFPrime.Spec.Poseidon2.rate]

theorem witnessConstraints (interface : Interface) (source start : Nat) :
    flatConstraints (Circuit.ops (circuit interface source).main start) =
      recipeConstraints start (Permutation.compile start (permutationInput interface source start)
        Permutation.schedule).recipes := by
  change flatConstraints (Formal.Owned.opsAt (ownedInterface interface source) start) = _
  rw [Formal.Owned.flatConstraints_opsAt]
  simp only [Formal.Owned.allAssertions, Formal.Owned.program, ownedInterface, actions, Formal.compile,
    frame_chunks, Hash.compileAbsorptions, List.append_nil]
  rfl

theorem output_eq_permutation (interface : Interface) (source start : Nat) :
    output interface source start = Permutation.scheduleOutput start := by
  simp only [output, Formal.Owned.output, Formal.Owned.program, ownedInterface, actions, Formal.compile,
    frame_chunks, Hash.compileAbsorptions]
  exact (Permutation.scheduleOutput_eq_compile start (permutationInput interface source start)).symm

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption
