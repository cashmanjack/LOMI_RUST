use dmd_analyzer_rust::{analyze_token_trace, write_rd_histogram, write_token_frequency};
use polars::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use tokenizers::{Tokenizer, Encoding};
use rand::Rng;

#[derive(Debug, Deserialize)]
struct Record {
    model: String,
    window_size: usize,
    percent_used: f64,
    dmd_sequence: String,
    gram: String,
}

#[derive(Deserialize, Serialize, Debug)]
struct DataRecord {
    source_id: usize,
    human: String,
    chatgpt: String,
}

fn read_original_jsonl(filepath: &str) -> Result<DataFrame, Box<dyn std::error::Error>> {
    println!("Reading data from {}...", filepath);
    let file = File::open(filepath)?;
    let reader = BufReader::new(file);
    let mut records: Vec<DataRecord> = Vec::new();

    for line in reader.lines() {
        let line: String = line?;
        // print!("\n\n{line}\n\n");
        if !line.contains("[]") {
            let data: DataRecord = serde_json::from_str(&line)?;
            records.push(data);
        }
    }

    let df: DataFrame = DataFrame::new(vec![
        Series::new("source_id", records.iter().map(|r| r.source_id as u64).collect::<Vec<_>>()),
        Series::new("human", records.iter().map(|r| r.human.clone()).collect::<Vec<_>>()),
        Series::new("chatgpt", records.iter().map(|r| r.chatgpt.clone()).collect::<Vec<_>>()),
    ])?;
    Ok(df)
}

fn clean_string(input_string: &str) -> String {
    let re = Regex::new(r"[^\x20-\x7E]+").unwrap();
    let mut cleaned = re.replace_all(input_string, "").to_string();
    
    cleaned = cleaned.replace("\\n", "");
    cleaned = cleaned.replace("\\", "");
    cleaned = cleaned.replace("`", "");
    cleaned = cleaned.replace("\"", "\"");

    let re = Regex::new(r"\s+").unwrap();
    cleaned = re.replace_all(&cleaned, " ").to_string();
    
    cleaned.trim_matches(&['\'', '"'][..]).to_string()
}

fn tokenize_input_n_gram(input_string: &str, n: usize, tokenizer: &Tokenizer) -> Vec<String> {
    let cleaned_string = clean_string(input_string);
    let encoding = tokenizer.encode(&*cleaned_string, true).unwrap();
    let tokens: Vec<&str> = encoding.get_tokens().iter().map(|s| s.as_str()).collect();
    
    (0..tokens.len() - n + 1)
        .map(|i| tokens[i..i + n].join(" "))
        .collect()
}

fn tokenize_answers_n_gram(answers: &[String], n: usize, tokenizer: &Tokenizer) -> Vec<Vec<String>> {
    answers.iter()
        .map(|ans| tokenize_input_n_gram(ans, n, tokenizer))
        .collect()
}

fn tokenize_df(df: &mut DataFrame, n: usize, tokenizer: &Tokenizer) -> Result<(), Box<dyn std::error::Error>> {
    println!("Tokenizing input with n_gram={}...", n);
    let mut new_columns: Vec<Series> = Vec::new();

    for col in df.get_column_names() {
        if col != "source_id" {
            //print!("\n\nup to here now\n\n\n");

            let tokenized_values: Vec<Vec<String>> = df.column(col)?
                .str()?
                .iter()
                .filter_map(|opt_value| {
                    opt_value.map(|value| tokenize_input_n_gram(value, n, tokenizer))
                })
                .collect();

            let collected_tokenized: Vec<String> = tokenized_values
                .into_iter()
                .map(|inner_vec| inner_vec.join(" "))
                .collect();

            // let limit = 10;
            // for (i, value) in collected_tokenized.iter().take(limit).enumerate() {
            //     println!("Value {}: {}", i + 1, value);
            // }

            let tokens_series: Series = Series::new(&format!("{}_tokenized", col), collected_tokenized);
        
            new_columns.push(tokens_series);
        }
    }

    let mut df = df.clone();
    df.hstack_mut(&new_columns)?;

    /*for col in df.get_column_names() {
        if col.ends_with("_tokenized") {
            if !df.column(col).is_ok() {
                println!("Warning: {} contains null values.", col);
            }
            if !df.column(col)?.u8()?.into_iter().all(|opt_value| opt_value.map(|v| v.is_empty() || v.is_string()).unwrap_or(false)) {
                println!("Warning: {} contains non-list or non-string entries.", col);
            }
        }
    }*/
    let mut file = std::fs::File::create("./tokenized.csv").unwrap();
    CsvWriter::new(&mut file).finish(&mut df).unwrap();
    //df.write_csv("./dataset/tokenized.csv")?;
    Ok(())
}

fn sample_windows(tokens: &[String], window_size: usize) -> Vec<String> {
    if tokens.len() <= window_size {
        return tokens.to_vec();
    }

    let max_start_index = tokens.len() - window_size;
    let start_index = rand::thread_rng().gen_range(0..=max_start_index);

    tokens[start_index..start_index + window_size].to_vec()
}

fn calculate_dmd(tokens: Vec<String>) -> (f64, f64, f64, usize) {
    let token_frequency = write_token_frequency(tokens.clone());
    let token_count = token_frequency.len();

    let approx: f64 = (1..=token_count).map(|i| (i as f64).sqrt()).sum();

    let rd_histogram = write_rd_histogram(tokens.clone());
    let dmd: f64 = rd_histogram.iter()
        .map(|(rd, count)| (rd.parse::<f64>().unwrap_or(0.0).sqrt() * count.parse::<f64>().unwrap_or(0.0)))
        .sum();

    let reuse_count = analyze_token_trace(tokens.clone())
        .iter()
        .filter(|(_, rd)| rd != "inf")
        .count();

    let per_reuse = if reuse_count == 0 { 0.0 } else { dmd / reuse_count as f64 };

    (per_reuse, dmd, approx, reuse_count)
}

fn generate_dmd_sequence(df: &DataFrame, window_size: usize, n_gram: usize) -> Result<DataFrame, Box<dyn Error>> {
    let model_columns: Vec<&str> = df.get_column_names()
        .iter()
        .filter(|&&col| col.ends_with("_tokenized"))
        .map(|&col| col)
        .collect();
    
    let mut dmd_sequence_df = DataFrame::new(vec![
        Series::new("model", Vec::<String>::new()),
        Series::new("window_size", Vec::<u32>::new()),
        Series::new("percent_used", Vec::<f64>::new()),
        Series::new("dmd_sequence", Vec::<String>::new()),
        Series::new("gram", Vec::<u32>::new())
    ])?;

    for model_col in model_columns {
        let model_name = model_col.replace("_tokenized", "");
        let mut dmd_sequence = Vec::new();
        let mut total_answers = 0;
        let mut skipped_answers = 0;

        println!("Processing model: {}, window_size: {}, gram: {}", model_name, window_size, n_gram);

        if let Ok(column) = df.column(model_col) {
            for opt_value in column.str()? {
                match opt_value {
                    Some(tokenized_answer) => {
                        let tokenized_answer = tokenized_answer.to_string();
        
                        total_answers += 1;
                        if tokenized_answer.is_empty() || tokenized_answer.len() < window_size {
                            skipped_answers += 1;
                            continue;
                        }
        
                        let tokens: Vec<String> = tokenized_answer.split_whitespace().map(String::from).collect();
                        
                        let sampled_window = sample_windows(&tokens, window_size);
                        let (per_reuse, dmd, approx, reuse_count) = calculate_dmd(sampled_window);
                        dmd_sequence.push(dmd.to_string());
                    }
                    None => {
                        skipped_answers += 1;
                    }
                }
            }
        }

        let percent_used = if total_answers > 0 {
            (total_answers - skipped_answers) as f64 / total_answers as f64
        } else {
            0.0
        };
        println!("Percent used: {}", percent_used);

        let new_data = DataFrame::new(vec![
            Series::new("model", vec![model_name]),
            Series::new("window_size", vec![window_size as u32]),
            Series::new("percent_used", vec![percent_used]),
            Series::new("dmd_sequence", vec![dmd_sequence.join(",")]),
            Series::new("gram", vec![n_gram as u32])
        ])?;

        dmd_sequence_df.vstack_mut(&new_data)?;
    }
    
    Ok(dmd_sequence_df)
}

fn run_with_step(df: DataFrame, step: usize, n_gram: usize) -> Result<DataFrame, Box<dyn Error>> {
    let mut final_dmd_sequence_df = DataFrame::default();

    for window_size in (50..=500).step_by(step) {
        let dmd_sequence_df = generate_dmd_sequence(&df.clone(), window_size, n_gram)?;
        final_dmd_sequence_df = final_dmd_sequence_df.vstack(&dmd_sequence_df)?;
    }

    Ok(final_dmd_sequence_df)
}

fn run_all(path: &str, n_gram: usize, step: usize) -> Result<DataFrame, Box<dyn Error>> {
    let tokenizer: Tokenizer = Tokenizer::from_file("/Users/jackcashman/Documents/Summer_2024/Code/LOMI_RUST/dmd_analyzer_rust/tokenizer.json").map_err(|e| e.to_string())?;
    
    let mut df: DataFrame = read_original_jsonl(path)?;
    tokenize_df(&mut df, n_gram, &tokenizer)?;
    //print!("\n\nup to here\n\n\n");
    let dmd_sequence_df = run_with_step(df, step, n_gram)?;
    Ok(dmd_sequence_df)
}

fn main() -> Result<(), Box<dyn Error>> {
    let path: &str = "/Users/jackcashman/Documents/Summer_2024/Code/LOMI_RUST/dataset/HC3_reddit_transformed.jsonl";
    let step: usize = 50;
    let mut dmd_sequence_df: DataFrame = DataFrame::default();
    for n_gram in 1..=3 {
        let result: DataFrame = run_all(path, n_gram, step)?;
        dmd_sequence_df.vstack_mut(&result)?;
    }

    let file = File::create("./dataset/HC3_reddit_all.csv")?;
    CsvWriter::new(file).finish(&mut dmd_sequence_df)?;
    
    Ok(())
}