use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| "failed to resolve repository root from CARGO_MANIFEST_DIR".to_string())
}

fn guest_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("riscv-tests")
        .join("guests")
        .join(name)
}

pub fn build_note_spend_rv64im_elf() -> Result<Vec<u8>, String> {
    build_guest_rv64im_elf("circuit-l2-transfer", "circuit_l2_transfer")
}

pub fn build_note_deposit_rv64im_elf() -> Result<Vec<u8>, String> {
    build_guest_rv64im_elf("note-deposit", "note_deposit")
}

fn build_guest_rv64im_elf(guest_name: &str, binary_name: &str) -> Result<Vec<u8>, String> {
    let repo_root = repo_root()?;
    let guest_dir = guest_dir(guest_name);
    let manifest = guest_dir.join("Cargo.toml");
    let guest_support_dir = repo_root
        .join("crates")
        .join("nightstream-sdk")
        .join("guest");
    let linker = guest_support_dir.join("riscv64im-unknown-none-elf.ld");
    let target_json = guest_support_dir.join("riscv64im-unknown-none-elf.json");
    let target_dir = repo_root
        .join("target")
        .join("rv64-guests")
        .join(guest_name);

    let output = Command::new("cargo")
        .arg("+nightly")
        .arg("build")
        .arg("-Z")
        .arg("build-std=core,alloc")
        .arg("-Z")
        .arg("build-std-features=compiler-builtins-mem")
        .arg("--manifest-path")
        .arg(&manifest)
        .arg("--release")
        .arg("--target")
        .arg(&target_json)
        .env("CARGO_TARGET_DIR", &target_dir)
        .env("RUSTFLAGS", format!("-C link-arg=-T{}", linker.display()))
        .output()
        .map_err(|e| format!("failed to invoke cargo for RV64 guest build: {e}"))?;

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "RV64 guest build failed\nstdout:\n{}\nstderr:\n{}",
            stdout, stderr
        ));
    }

    let elf_path = target_dir
        .join("riscv64im-unknown-none-elf")
        .join("release")
        .join(binary_name);
    fs::read(&elf_path).map_err(|e| format!("failed to read {}: {e}", elf_path.display()))
}
