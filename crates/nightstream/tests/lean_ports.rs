//! Checks `formal/compression-lean/ports.json`: each Lean port names a Rust
//! item and the hash of its tokens, without comments or formatting. A path is
//! `<file>::<item>` or `<file>::<Type>::<fn>` (a method of `impl Type`,
//! `impl Trait for Type` or `trait Type`); a name that two items share is an error.

use proc_macro2::{Delimiter, Group, TokenStream, TokenTree};
use quote::ToTokens;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;
use syn::{File, ImplItem, Item, TraitItem, Type};

#[derive(Deserialize)]
struct Port {
    lean: String,
    rust: String,
    hash: String,
}

#[test]
fn lean_ports_match_their_rust_items() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let manifest = std::fs::read_to_string(root.join("formal/compression-lean/ports.json")).expect("read ports.json");
    let ports: Vec<Port> = serde_json::from_str(&manifest).expect("parse ports.json");
    let mut files: BTreeMap<&str, Result<File, String>> = BTreeMap::new();
    let mut failures = Vec::new();
    for port in &ports {
        let Some((file, item)) = port.rust.split_once(".rs::") else {
            failures.push(format!("{}: not a `<file>.rs::<item>` path", port.rust));
            continue;
        };
        let parsed = files.entry(file).or_insert_with(|| {
            let source = std::fs::read_to_string(root.join(format!("{file}.rs"))).map_err(|e| e.to_string())?;
            syn::parse_file(&source).map_err(|e| e.to_string())
        });
        match parsed
            .as_ref()
            .map_err(String::clone)
            .and_then(|f| tokens(f, item))
        {
            Ok(tokens) => {
                let hash = format!("{:016x}", fnv1a(&without_docs(tokens).to_string()));
                if hash != port.hash {
                    failures.push(format!(
                        "{}: changed; re-read {}; new hash {hash}",
                        port.rust, port.lean
                    ));
                }
            }
            Err(error) => failures.push(format!("{}: {error}; re-read {}", port.rust, port.lean)),
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} ports need review:\n{}",
        failures.len(),
        ports.len(),
        failures.join("\n")
    );
}

fn tokens(file: &File, item: &str) -> Result<TokenStream, String> {
    let found: Vec<TokenStream> = match item.split_once("::") {
        None => file
            .items
            .iter()
            .filter(|i| top_level_name(i).is_some_and(|name| name == item))
            .map(ToTokens::to_token_stream)
            .collect(),
        Some((ty, method)) => file
            .items
            .iter()
            .flat_map(|i| match i {
                Item::Impl(block) if self_name(&block.self_ty).is_some_and(|name| name == ty) => block
                    .items
                    .iter()
                    .filter_map(|i| match i {
                        ImplItem::Fn(f) if f.sig.ident == method => Some(f.to_token_stream()),
                        _ => None,
                    })
                    .collect(),
                Item::Trait(t) if t.ident == ty => t
                    .items
                    .iter()
                    .filter_map(|i| match i {
                        TraitItem::Fn(f) if f.sig.ident == method => Some(f.to_token_stream()),
                        _ => None,
                    })
                    .collect(),
                _ => Vec::new(),
            })
            .collect(),
    };
    match <[_; 1]>::try_from(found) {
        Ok([tokens]) => Ok(tokens),
        Err(found) if found.is_empty() => Err("not found".into()),
        Err(found) => Err(format!("{} items share this name", found.len())),
    }
}

fn top_level_name(item: &Item) -> Option<String> {
    let ident = match item {
        Item::Fn(i) => &i.sig.ident,
        Item::Const(i) => &i.ident,
        Item::Static(i) => &i.ident,
        Item::Struct(i) => &i.ident,
        Item::Enum(i) => &i.ident,
        Item::Trait(i) => &i.ident,
        Item::Type(i) => &i.ident,
        _ => return None,
    };
    Some(ident.to_string())
}

fn self_name(ty: &Type) -> Option<String> {
    match ty {
        Type::Path(path) => path.path.segments.last().map(|s| s.ident.to_string()),
        _ => None,
    }
}

fn without_docs(tokens: TokenStream) -> TokenStream {
    let trees: Vec<TokenTree> = tokens.into_iter().collect();
    let mut out = TokenStream::new();
    let mut i = 0;
    while i < trees.len() {
        let inner = matches!(trees.get(i + 1), Some(TokenTree::Punct(p)) if p.as_char() == '!');
        let attribute = i + 1 + usize::from(inner);
        if matches!(&trees[i], TokenTree::Punct(p) if p.as_char() == '#') && is_doc(trees.get(attribute)) {
            i = attribute + 1;
            continue;
        }
        out.extend([match &trees[i] {
            TokenTree::Group(g) => TokenTree::Group(Group::new(g.delimiter(), without_docs(g.stream()))),
            tree => tree.clone(),
        }]);
        i += 1;
    }
    out
}

fn is_doc(tree: Option<&TokenTree>) -> bool {
    matches!(tree, Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Bracket
        && matches!(g.stream().into_iter().next(), Some(TokenTree::Ident(i)) if i == "doc"))
}

/// 64-bit FNV-1a: a change detector, not a protocol digest.
fn fnv1a(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}
