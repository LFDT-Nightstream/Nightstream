//! Command-line entry point for bounded cvc5 redundancy checks.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use recursive_constraint_minimizer::{
    render_query, run_cvc5, Conclusion, Evidence, Problem, Selection, SolverConfig, SolverMode,
};

const USAGE: &str = "\
recursive-constraint-minimizer

USAGE:
  recursive-constraint-minimizer emit --input FILE (--remove-row ID | --remove-family NAME) --output FILE
  recursive-constraint-minimizer check --input FILE (--remove-row ID | --remove-family NAME) --evidence FILE [OPTIONS]

CHECK OPTIONS:
  --solver PATH          cvc5 executable (default: cvc5)
  --ff-solver MODE       gb or split (default: gb)
  --timeout-ms N         cvc5 query limit, at most 300000 (default: 60000)
";

enum CommandLine {
    Emit {
        input: PathBuf,
        selection: Selection,
        output: PathBuf,
    },
    Check {
        input: PathBuf,
        selection: Selection,
        evidence: PathBuf,
        solver: SolverConfig,
    },
}

#[derive(Default)]
struct ParsedOptions {
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    evidence: Option<PathBuf>,
    remove_row: Option<String>,
    remove_family: Option<String>,
    solver: Option<PathBuf>,
    solver_mode: Option<String>,
    timeout_ms: Option<String>,
}

fn main() {
    match run() {
        Ok(code) => process::exit(code),
        Err(message) => {
            eprintln!("error: {message}\n\n{USAGE}");
            process::exit(1);
        }
    }
}

fn run() -> Result<i32, String> {
    let command = parse_command(env::args().skip(1).collect())?;
    match command {
        CommandLine::Emit {
            input,
            selection,
            output,
        } => {
            let problem = read_problem(&input)?;
            let query = render_query(&problem, &selection).map_err(|error| error.to_string())?;
            write_file(&output, query.smt2.as_bytes())?;
            println!(
                "wrote {} retained rows and {} candidate rows to {}",
                query.retained_rows.len(),
                query.removed_rows.len(),
                output.display()
            );
            Ok(0)
        }
        CommandLine::Check {
            input,
            selection,
            evidence,
            solver,
        } => {
            let problem = read_problem(&input)?;
            let query = render_query(&problem, &selection).map_err(|error| error.to_string())?;
            let run = run_cvc5(&query, &solver).map_err(|error| error.to_string())?;
            let conclusion = run.conclusion;
            let record = Evidence::new(&problem, selection, query, &solver, run);
            let bytes = serde_json::to_vec_pretty(&record).map_err(|error| error.to_string())?;
            write_file(&evidence, &bytes)?;
            println!("{:?}; evidence: {}", conclusion, evidence.display());
            Ok(if conclusion == Conclusion::Inconclusive { 2 } else { 0 })
        }
    }
}

fn parse_command(arguments: Vec<String>) -> Result<CommandLine, String> {
    if arguments.is_empty() || arguments[0] == "--help" || arguments[0] == "-h" {
        println!("{USAGE}");
        process::exit(0);
    }
    let command = arguments[0].as_str();
    if command != "emit" && command != "check" {
        return Err(format!("unknown command {command:?}"));
    }
    let options = parse_options(&arguments[1..])?;
    let input = options.input.ok_or_else(|| "missing --input".to_owned())?;
    let selection = match (options.remove_row, options.remove_family) {
        (Some(row), None) => Selection::Row(row),
        (None, Some(family)) => Selection::Family(family),
        (None, None) => return Err("select one --remove-row or --remove-family".to_owned()),
        (Some(_), Some(_)) => return Err("--remove-row and --remove-family are mutually exclusive".to_owned()),
    };
    if command == "emit" {
        if options.evidence.is_some()
            || options.solver.is_some()
            || options.solver_mode.is_some()
            || options.timeout_ms.is_some()
        {
            return Err("solver options are valid only for check".to_owned());
        }
        return Ok(CommandLine::Emit {
            input,
            selection,
            output: options
                .output
                .ok_or_else(|| "missing --output".to_owned())?,
        });
    }
    if options.output.is_some() {
        return Err("--output is valid only for emit".to_owned());
    }
    let mut solver = SolverConfig::default();
    if let Some(executable) = options.solver {
        solver.executable = executable;
    }
    if let Some(mode) = options.solver_mode {
        solver.mode = SolverMode::parse(&mode).map_err(|error| error.to_string())?;
    }
    if let Some(timeout) = options.timeout_ms {
        solver.timeout_ms = timeout
            .parse::<u64>()
            .map_err(|_| format!("invalid --timeout-ms value {timeout:?}"))?;
    }
    Ok(CommandLine::Check {
        input,
        selection,
        evidence: options
            .evidence
            .ok_or_else(|| "missing --evidence".to_owned())?,
        solver,
    })
}

fn parse_options(arguments: &[String]) -> Result<ParsedOptions, String> {
    let mut parsed = ParsedOptions::default();
    let mut index = 0;
    while index < arguments.len() {
        let flag = arguments[index].as_str();
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?
            .clone();
        match flag {
            "--input" => set_once(&mut parsed.input, PathBuf::from(value), flag)?,
            "--output" => set_once(&mut parsed.output, PathBuf::from(value), flag)?,
            "--evidence" => set_once(&mut parsed.evidence, PathBuf::from(value), flag)?,
            "--remove-row" => set_once(&mut parsed.remove_row, value, flag)?,
            "--remove-family" => set_once(&mut parsed.remove_family, value, flag)?,
            "--solver" => set_once(&mut parsed.solver, PathBuf::from(value), flag)?,
            "--ff-solver" => set_once(&mut parsed.solver_mode, value, flag)?,
            "--timeout-ms" => set_once(&mut parsed.timeout_ms, value, flag)?,
            _ => return Err(format!("unknown option {flag:?}")),
        }
        index += 2;
    }
    Ok(parsed)
}

fn set_once<T>(slot: &mut Option<T>, value: T, flag: &str) -> Result<(), String> {
    if slot.replace(value).is_some() {
        return Err(format!("duplicate option {flag}"));
    }
    Ok(())
}

fn read_problem(path: &Path) -> Result<Problem, String> {
    let bytes = fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let problem: Problem =
        serde_json::from_slice(&bytes).map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
    problem.validate().map_err(|error| error.to_string())?;
    Ok(problem)
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    fs::write(path, bytes).map_err(|error| format!("failed to write {}: {error}", path.display()))
}
