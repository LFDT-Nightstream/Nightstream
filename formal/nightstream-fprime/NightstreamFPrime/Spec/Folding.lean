import NightstreamFPrime.Spec.Folding.BatchArity
import NightstreamFPrime.Spec.Folding.PiRLC
import NightstreamFPrime.Spec.Folding.PiRLC.v1_1.InputBinding
import NightstreamFPrime.Spec.Folding.PiDEC
import NightstreamFPrime.Spec.Folding.PiDEC.BindingCollision
import NightstreamFPrime.Spec.Folding.PiDEC.PaperVerifier
import NightstreamFPrime.Spec.Folding.PiCCS
import NightstreamFPrime.Spec.Folding.Nifs
import NightstreamFPrime.Spec.Folding.Nifs.PaperStrongCompleteness
import NightstreamFPrime.Spec.Folding.Nifs.PaperCausalReplay
import NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Completeness
import NightstreamFPrime.Spec.Folding.Nifs.PaperWeakAlgorithm
import NightstreamFPrime.Spec.Folding.Nifs.PaperWeakCompleteness
import NightstreamFPrime.Spec.Folding.Nifs.PaperAlignedExtraction
import NightstreamFPrime.Spec.Folding.Nifs.PaperCompositionProbability
import NightstreamFPrime.Spec.Folding.Nifs.PaperCompositionWork
import NightstreamFPrime.Spec.Folding.Nifs.PaperCompositionAgreement

/-! Folding-path verifier semantics: batch arity, Π_RLC combination, Π_DEC
decomposition and its paper verifier. -/
