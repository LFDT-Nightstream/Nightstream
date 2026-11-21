use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(nightstream_sme)");
    println!("cargo:rerun-if-changed=src/signed_blocks.c");
    if env::var("TARGET").as_deref() != Ok("aarch64-apple-darwin") {
        return;
    }
    let mut build = cc::Build::new();
    build.file("src/signed_blocks.c").opt_level(3);
    let Ok(compiler) = build.try_get_compiler() else {
        return;
    };
    let Ok(resources) = compiler.to_command().arg("-print-resource-dir").output() else {
        return;
    };
    if !resources.status.success() {
        return;
    }
    let path = String::from_utf8_lossy(&resources.stdout);
    let runtime = PathBuf::from(path.trim()).join("lib/darwin");
    if runtime.join("libclang_rt.osx.a").is_file() && build.try_compile("nightstream_signed_blocks").is_ok() {
        println!("cargo:rustc-link-search=native={}", runtime.display());
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
        println!("cargo:rustc-cfg=nightstream_sme");
    }
}
