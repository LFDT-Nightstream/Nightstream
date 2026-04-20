use std::fs;

const IVC_RS: &str =
    "/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/crates/neo-fold-next/src/rv64im/ivc.rs";
const IVC_SNARK_RS: &str =
    "/Users/nicolasarqueros/starstream/develop/nightstream-clean-up/crates/neo-fold-next/src/rv64im/ivc_snark.rs";

#[test]
fn rv64im_ivc_native_module_does_not_reference_spartan2() {
    let source = fs::read_to_string(IVC_RS).expect("read native IVC module");
    assert!(
        !source.contains("spartan2"),
        "native RV64IM IVC ownership must stay Spartan-free"
    );
}

#[test]
fn rv64im_ivc_compression_module_owns_explicit_compress_boundary() {
    let source = fs::read_to_string(IVC_SNARK_RS).expect("read IVC compression module");
    assert!(
        source.contains("impl Rv64imIvcState")
            && source.contains("pub fn compress(&self)")
            && source.contains("prove_rv64im_chunk_step_ivc_spartan"),
        "RV64IM IVC compression must stay explicitly owned by ivc_snark.rs"
    );
}
