use super::*;

pub(super) fn encode_rv64im_nightstream_proof_fields(
    out: &mut Vec<BridgeFieldWord>,
    proof: &Rv64imNightstreamProof,
) -> Result<(), Rv64imBridgeError> {
    encode_digest32_field_words(out, rv64im_main_nightstream_proof_digest(proof.main_proof()));
    encode_digest32_field_words(out, proof.side_proof().expected_digest());
    encode_digest32_field_words(out, proof.main_proof().published_statement().expected_digest());
    let proof_bytes = bincode::serialize(proof).map_err(|err| Rv64imBridgeError::WitnessEncode(err.to_string()))?;
    out.extend(encode_bytes_field_words(&proof_bytes));
    Ok(())
}

pub(super) fn decode_rv64im_nightstream_proof_fields(
    words: &[BridgeFieldWord],
    cursor: &mut usize,
) -> Result<Rv64imNightstreamProof, Rv64imBridgeError> {
    let main_proof_digest = decode_digest32_field_words(words, cursor, "main proof digest")?;
    let side_proof_digest = decode_digest32_field_words(words, cursor, "side proof digest")?;
    let published_statement_digest = decode_digest32_field_words(words, cursor, "published statement digest")?;
    let proof_bytes = decode_bytes_field_words(words, cursor, "nightstream proof bytes")?;
    let proof = bincode::deserialize::<Rv64imNightstreamProof>(&proof_bytes)
        .map_err(|err| Rv64imBridgeError::WitnessDecode(err.to_string()))?;
    if rv64im_main_nightstream_proof_digest(proof.main_proof()) != main_proof_digest {
        return Err(Rv64imBridgeError::WitnessDecode(
            "nightstream proof bytes do not match the carried main proof digest".into(),
        ));
    }
    if proof.side_proof().expected_digest() != side_proof_digest {
        return Err(Rv64imBridgeError::WitnessDecode(
            "nightstream proof bytes do not match the carried side proof digest".into(),
        ));
    }
    if proof.main_proof().published_statement().expected_digest() != published_statement_digest {
        return Err(Rv64imBridgeError::WitnessDecode(
            "nightstream proof bytes do not match the carried published statement digest".into(),
        ));
    }
    Ok(proof)
}
