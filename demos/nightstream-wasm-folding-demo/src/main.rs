//! Minimal local HTTP boundary for the Nightstream WASM folding demo.

use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use nightstream_wasm_folding_demo::{
    preprocess_program, prove_prepared, trace_program, PreparationRequest, PreparationResponse, PreparedProof,
    ProofRequest, TraceRequest, TraceResponse, TracedProgram, BR_TABLE_WAT, COUNTER_WAT, MUTABLE_GLOBAL_WAT,
    TABLE_DISPATCH_WAT, TRAPPING_DIVISION_WAT,
};
use serde::Serialize;

const ADDRESS: &str = "127.0.0.1:3000";
const MAX_REQUEST_BYTES: usize = 512 * 1024;
const INDEX_HTML: &str = include_str!("../static/index.html");
const APP_JS: &str = include_str!("../static/app.js");
const STYLE_CSS: &str = include_str!("../static/style.css");

fn main() -> io::Result<()> {
    let listener = TcpListener::bind(ADDRESS)?;
    let state = Arc::new(Mutex::new(DemoState::default()));
    println!("Nightstream WASM folding demo: http://{ADDRESS}");
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let state = Arc::clone(&state);
                thread::spawn(move || {
                    if let Err(error) = handle_connection(stream, &state) {
                        eprintln!("request failed: {error}");
                    }
                });
            }
            Err(error) => eprintln!("connection failed: {error}"),
        }
    }
    Ok(())
}

#[derive(Default)]
struct DemoState {
    next_trace_id: u64,
    traced: Option<TracedEntry>,
    next_preparation_id: u64,
    prepared: Option<PreparedEntry>,
}

struct TracedEntry {
    id: u64,
    program: Arc<TracedProgram>,
}

struct PreparedEntry {
    id: u64,
    proof: PreparedProof,
}

#[derive(Serialize)]
struct TraceEnvelope {
    trace_id: u64,
    #[serde(flatten)]
    trace: TraceResponse,
}

#[derive(Serialize)]
struct PreparationEnvelope {
    preparation_id: u64,
    #[serde(flatten)]
    preparation: PreparationResponse,
}

fn handle_connection(mut stream: TcpStream, state: &Mutex<DemoState>) -> io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    let request = read_request(&mut stream)?;
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/") => write_response(&mut stream, 200, "text/html; charset=utf-8", INDEX_HTML.as_bytes()),
        ("GET", "/app.js") => write_response(&mut stream, 200, "text/javascript; charset=utf-8", APP_JS.as_bytes()),
        ("GET", "/style.css") => write_response(&mut stream, 200, "text/css; charset=utf-8", STYLE_CSS.as_bytes()),
        ("GET", "/api/program") => write_json(
            &mut stream,
            200,
            &serde_json::json!({
                "default_id": "counter",
                "programs": [
                    { "id": "counter", "label": "Loop and direct call", "source": COUNTER_WAT },
                    { "id": "table_dispatch", "label": "Mutable function table", "source": TABLE_DISPATCH_WAT },
                    { "id": "mutable_global", "label": "Mutable global state", "source": MUTABLE_GLOBAL_WAT },
                    { "id": "trapping_division", "label": "Division by zero trap", "source": TRAPPING_DIVISION_WAT },
                    { "id": "br_table", "label": "Multi-way branch table", "source": BR_TABLE_WAT },
                ],
            }),
        ),
        ("POST", "/api/trace") => match serde_json::from_slice::<TraceRequest>(&request.body) {
            Ok(input) => match trace_program(input) {
                Ok((trace, program)) => {
                    let trace_id = store_trace(state, program)?;
                    write_json(&mut stream, 200, &TraceEnvelope { trace_id, trace })
                }
                Err(message) => write_json(&mut stream, 422, &serde_json::json!({ "error": message })),
            },
            Err(error) => write_json(
                &mut stream,
                400,
                &serde_json::json!({ "error": format!("invalid JSON request: {error}") }),
            ),
        },
        ("POST", "/api/prepare") => match serde_json::from_slice::<PreparationRequest>(&request.body) {
            Ok(input) => preprocess_cached(&mut stream, state, input),
            Err(error) => write_json(
                &mut stream,
                400,
                &serde_json::json!({ "error": format!("invalid JSON request: {error}") }),
            ),
        },
        ("POST", "/api/prove") => match serde_json::from_slice::<ProofRequest>(&request.body) {
            Ok(input) => prove_cached(&mut stream, state, input),
            Err(error) => write_json(
                &mut stream,
                400,
                &serde_json::json!({ "error": format!("invalid JSON request: {error}") }),
            ),
        },
        _ => write_json(&mut stream, 404, &serde_json::json!({ "error": "not found" })),
    }
}

fn store_trace(state: &Mutex<DemoState>, program: Arc<TracedProgram>) -> io::Result<u64> {
    let mut state = state
        .lock()
        .map_err(|_| io::Error::other("demo state lock was poisoned"))?;
    state.next_trace_id += 1;
    let id = state.next_trace_id;
    state.traced = Some(TracedEntry { id, program });
    state.prepared = None;
    Ok(id)
}

fn preprocess_cached(stream: &mut TcpStream, state: &Mutex<DemoState>, input: PreparationRequest) -> io::Result<()> {
    let program = {
        let state = state
            .lock()
            .map_err(|_| io::Error::other("demo state lock was poisoned"))?;
        state
            .traced
            .as_ref()
            .filter(|traced| traced.id == input.trace_id)
            .map(|traced| Arc::clone(&traced.program))
    };
    let Some(program) = program else {
        return write_json(
            stream,
            409,
            &serde_json::json!({ "error": "trace is no longer available; trace the program again" }),
        );
    };

    match preprocess_program(program, input.mode) {
        Ok((preparation, proof)) => match store_preparation(state, input.trace_id, proof)? {
            Some(preparation_id) => write_json(
                stream,
                200,
                &PreparationEnvelope {
                    preparation_id,
                    preparation,
                },
            ),
            None => write_json(
                stream,
                409,
                &serde_json::json!({ "error": "program changed during preprocessing; trace it again" }),
            ),
        },
        Err(message) => write_json(stream, 422, &serde_json::json!({ "error": message })),
    }
}

fn store_preparation(state: &Mutex<DemoState>, trace_id: u64, proof: PreparedProof) -> io::Result<Option<u64>> {
    let mut state = state
        .lock()
        .map_err(|_| io::Error::other("demo state lock was poisoned"))?;
    if !state
        .traced
        .as_ref()
        .is_some_and(|traced| traced.id == trace_id)
    {
        return Ok(None);
    }
    state.next_preparation_id += 1;
    let id = state.next_preparation_id;
    state.prepared = Some(PreparedEntry { id, proof });
    Ok(Some(id))
}

fn prove_cached(stream: &mut TcpStream, state: &Mutex<DemoState>, input: ProofRequest) -> io::Result<()> {
    let state = state
        .lock()
        .map_err(|_| io::Error::other("demo state lock was poisoned"))?;
    let Some(prepared) = state
        .prepared
        .as_ref()
        .filter(|prepared| prepared.id == input.preparation_id)
    else {
        return write_json(
            stream,
            409,
            &serde_json::json!({ "error": "preparation is no longer available; preprocess the program again" }),
        );
    };
    match prove_prepared(&prepared.proof) {
        Ok(response) => write_json(stream, 200, &response),
        Err(message) => write_json(stream, 422, &serde_json::json!({ "error": message })),
    }
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(stream: &mut TcpStream) -> io::Result<Request> {
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 8192];
    let header_end = loop {
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "request ended before headers",
            ));
        }
        bytes.extend_from_slice(&chunk[..read]);
        if bytes.len() > MAX_REQUEST_BYTES {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "request is too large"));
        }
        if let Some(position) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            break position + 4;
        }
    };

    let headers = std::str::from_utf8(&bytes[..header_end])
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "request headers are not UTF-8"))?;
    let mut lines = headers.split("\r\n");
    let mut request_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing request line"))?
        .split_whitespace();
    let method = request_line.next().unwrap_or_default().to_string();
    let path = request_line.next().unwrap_or_default().to_string();
    let content_length = lines
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    if header_end + content_length > MAX_REQUEST_BYTES {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "request body is too large"));
    }
    while bytes.len() < header_end + content_length {
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "request body ended early"));
        }
        bytes.extend_from_slice(&chunk[..read]);
    }

    Ok(Request {
        method,
        path,
        body: bytes[header_end..header_end + content_length].to_vec(),
    })
}

fn write_json<T: Serialize>(stream: &mut TcpStream, status: u16, value: &T) -> io::Result<()> {
    let body = serde_json::to_vec(value).map_err(io::Error::other)?;
    write_response(stream, status, "application/json; charset=utf-8", &body)
}

fn write_response(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) -> io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        409 => "Conflict",
        404 => "Not Found",
        422 => "Unprocessable Content",
        _ => "Error",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\nX-Content-Type-Options: nosniff\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)
}
