//! Terminal compression entrypoints for direct-CCS state.

use super::*;

impl DirectCcsIvcState {
    pub fn compress_snark_with_trace(
        &self,
        emit: &mut dyn FnMut(&str),
    ) -> Result<
        (
            DirectCcsIvcSnark,
            DirectCcsIvcSnarkVerifierKey,
            DirectCcsFPrimeSnarkPerf,
        ),
        DirectCcsFPrimeSnarkError,
    > {
        self.ensure_terminal_compression_is_proof_complete()?;
        emit("direct_ccs_ivc.phase=latest_relation_and_advice.start");
        let circuit = self.latest_circuit()?;
        emit("direct_ccs_ivc.phase=latest_relation_and_advice.done");
        emit("direct_ccs_ivc.phase=spartan_terminal_prove.start");
        let proved = prove_direct_ccs_terminal_snark(circuit, emit);
        emit("direct_ccs_ivc.phase=spartan_terminal_prove.done");
        let proved = proved?;
        Ok((proved.snark, proved.verifier_key, proved.perf))
    }

    pub fn compress_snark(
        &self,
    ) -> Result<
        (
            DirectCcsIvcSnark,
            DirectCcsIvcSnarkVerifierKey,
            DirectCcsFPrimeSnarkPerf,
        ),
        DirectCcsFPrimeSnarkError,
    > {
        let mut emit = |_message: &str| {};
        self.compress_snark_with_trace(&mut emit)
    }

    fn ensure_terminal_compression_is_proof_complete(&self) -> Result<(), DirectCcsFPrimeSnarkError> {
        let last = self.last_step.as_ref().ok_or_else(|| {
            DirectCcsFPrimeSnarkError::Input(
                "direct CCS folded compression requires at least one appended SuperNeo relation".into(),
            )
        })?;
        if self.state.chunk_count > 1 && last.construction2_fold.is_none() {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "plain direct CCS terminal compression is latest-only and disabled for multi-step runs".into(),
            ));
        }
        Ok(())
    }
}
