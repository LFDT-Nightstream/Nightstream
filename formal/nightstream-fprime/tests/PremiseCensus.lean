import tests.PremiseAudit
import NightstreamFPrime

open NightstreamFPrime in
#premise_census "artifacts/nightstream-fprime-premises-v1.tsv" "NightstreamFPrime"
  Export.BoundedDepth.StepRelation.light_stepHolds
  Export.BoundedDepth.StepRelation.refresh_stepHolds
  Export.BoundedDepth.TerminalBridge.bridge
  Lifecycle.BoundedDepth.RandomOracle.FoldGate.light_statisticalError_le_bits
  Lifecycle.BoundedDepth.RandomOracle.FoldGate.refresh_statisticalError_le_bits
  Export.BoundedDepth.FoldSponge.light_knowledge_le
  Export.BoundedDepth.FoldSponge.refresh_knowledge_le
  Export.BoundedDepth.IvcSecurity.extraction_bound
  Export.BoundedDepth.IvcSecurity.probability_bound
  Export.BoundedDepth.VerifierSecurity.soundness
  Export.BoundedDepth.NebulaSecurity.bad_le
  Export.BoundedDepth.NebulaSecurity.romMemoryRound
  Export.BoundedDepth.EtaSponge.unbalanced_le
  Spec.Compression.Verifier.soundness
  Spec.Compression.Production.params_soundness
  Spec.Compression.Fresh.soundness
  Export.BoundedDepth.FinalVerifier.soundness
  Export.BoundedDepth.FinalVerifier.real_soundness
  Export.BoundedDepth.FinalHistory.soundness
  Export.BoundedDepth.FinalHistory.real_soundness
  Export.BoundedDepth.NebulaFinal.soundness
  Export.BoundedDepth.NebulaFinal.real_soundness
