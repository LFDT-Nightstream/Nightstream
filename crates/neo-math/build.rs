use std::path::PathBuf;
use std::process::Command;

fn xcrun(tool: &str) -> Command {
    let mut command = Command::new("xcrun");
    command.args(["--sdk", "macosx", tool]);
    command
}

fn main() {
    const SOURCE: &str = "src/signed_sums/sme2.c";
    println!("cargo:rustc-check-cfg=cfg(nightstream_sme2)");
    println!("cargo:rerun-if-changed={SOURCE}");
    if std::env::var("TARGET").as_deref() != Ok("aarch64-apple-darwin") {
        return;
    }
    for name in ["DEVELOPER_DIR", "TOOLCHAINS", "SDKROOT"] {
        println!("cargo:rerun-if-env-changed={name}");
    }

    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    let object = out.join("sme2.o");
    // A relocatable Clang link includes the Arm streaming ABI routines from compiler-rt.
    let compile = xcrun("clang")
        .args([
            "-target",
            "arm64-apple-macos11",
            "-march=armv8-a+sme2",
            "-O3",
            "-r",
            SOURCE,
            "-o",
        ])
        .arg(&object)
        .output();
    match compile {
        Ok(output) if output.status.success() => {}
        Ok(output) => {
            println!(
                "cargo:warning=using Accelerate: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
            return;
        }
        Err(error) => {
            println!("cargo:warning=using Accelerate: xcrun: {error}");
            return;
        }
    }

    let runtime_linked = xcrun("nm")
        .arg("-uj")
        .arg(&object)
        .output()
        .is_ok_and(|imports| {
            imports.status.success()
                && !String::from_utf8_lossy(&imports.stdout)
                    .lines()
                    .any(|name| name.starts_with("___arm_"))
        });
    if !runtime_linked {
        println!("cargo:warning=using Accelerate: compiler-rt lacks static SME support");
        return;
    }
    let archived = xcrun("ar")
        .arg("crs")
        .arg(out.join("libnightstream_sme2.a"))
        .arg(&object)
        .status()
        .is_ok_and(|status| status.success());
    if !archived {
        println!("cargo:warning=using Accelerate: could not archive the SME2 kernel");
        return;
    }
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=nightstream_sme2");
    println!("cargo:rustc-cfg=nightstream_sme2");
}
