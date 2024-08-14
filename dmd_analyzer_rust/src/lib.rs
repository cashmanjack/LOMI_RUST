use lru::LruCache;
use std::{collections::{BTreeMap, HashMap}, num::NonZero};

pub fn analyze_token_trace(tokens: Vec<String>) -> Vec<(String, String)> {
    let mut token_rd_pairs = Vec::new();
    if tokens.is_empty() {
        println!("ERROR NO VALUES FOUND");
        return token_rd_pairs;
    }

    let length: usize = tokens.len();
    let mut lru_cache = LruCache::new(NonZero::new(length).unwrap());

    for token in tokens.iter() {
        let rd = match lru_cache.get(token) {
            Some(_) => lru_cache.len(),
            None => 0,
        };

        lru_cache.put(token.clone(), true);
        let rd_str = if rd == 0 { "inf".to_string() } else { rd.to_string() };
        token_rd_pairs.push((token.clone(), rd_str));
    }

    token_rd_pairs
}

pub fn write_rd_histogram(tokens: Vec<String>) -> Vec<(String, String)> {
    let length: usize = tokens.len();
    let mut lru_cache = LruCache::new(NonZero::new(length).unwrap());
    let mut rd_histogram = HashMap::new();

    for token in tokens.iter() {
        let rd = match lru_cache.get(token) {
            Some(_) => lru_cache.len(),
            None => 0,
        };

        if rd > 0 {
            *rd_histogram.entry(rd).or_insert(0) += 1;
        }

        lru_cache.put(token.clone(), true);
    }

    let mut sorted_histogram: BTreeMap<usize, usize> = BTreeMap::new();
    sorted_histogram.extend(rd_histogram.into_iter());

    let mut result = Vec::new();
    for (rd, count) in sorted_histogram {
        result.push((rd.to_string(), count.to_string()));
    }

    result
}

pub fn write_token_frequency(tokens: Vec<String>) -> Vec<(String, String)> {
    let mut token_frequency = HashMap::new();
    for token in tokens.iter() {
        *token_frequency.entry(token.clone()).or_insert(0) += 1;
    }

    let mut sorted_frequency: Vec<(String, usize)> = token_frequency.into_iter().collect();
    sorted_frequency.sort_by(|a, b| b.1.cmp(&a.1));

    let mut result = Vec::new();
    for (token, count) in sorted_frequency {
        result.push((token, count.to_string()));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_token_trace() {
        let tokens = vec!["a".to_string(), "b".to_string(), "a".to_string(), "c".to_string(), "b".to_string(), "a".to_string()];
        let expected = vec![
            ("a".to_string(), "inf".to_string()),
            ("b".to_string(), "inf".to_string()),
            ("a".to_string(), "2".to_string()),
            ("c".to_string(), "inf".to_string()),
            ("b".to_string(), "3".to_string()),
            ("a".to_string(), "3".to_string()),
        ];
        let result = analyze_token_trace(tokens);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_write_rd_histogram() {
        let tokens = vec!["a".to_string(), "b".to_string(), "a".to_string(), "c".to_string(), "b".to_string(), "a".to_string()];
        let expected = vec![
            ("2".to_string(), "1".to_string()),
            ("3".to_string(), "2".to_string()),
        ];
        let result = write_rd_histogram(tokens);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_write_token_frequency() {
        let tokens = vec!["a".to_string(), "b".to_string(), "a".to_string(), "c".to_string(), "b".to_string(), "a".to_string()];
        let expected = vec![
            ("a".to_string(), "3".to_string()),
            ("b".to_string(), "2".to_string()),
            ("c".to_string(), "1".to_string()),
        ];
        let result = write_token_frequency(tokens);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_empty_input() {
        let tokens: Vec<String> = vec![];

        let expected_trace: Vec<(String, String)> = vec![];
        let result_trace = analyze_token_trace(tokens.clone());
        assert_eq!(result_trace, expected_trace);

        let expected_histogram: Vec<(String, String)> = vec![];
        let result_histogram = write_rd_histogram(tokens.clone());
        assert_eq!(result_histogram, expected_histogram);

        let expected_frequency: Vec<(String, String)> = vec![];
        let result_frequency = write_token_frequency(tokens);
        assert_eq!(result_frequency, expected_frequency);
    }

    #[test]
    fn test_single_token() {
        let tokens = vec!["a".to_string()];

        let expected_trace = vec![("a".to_string(), "inf".to_string())];
        let result_trace = analyze_token_trace(tokens.clone());
        assert_eq!(result_trace, expected_trace);

        let expected_histogram: Vec<(String, String)> = vec![];
        let result_histogram = write_rd_histogram(tokens.clone());
        assert_eq!(result_histogram, expected_histogram);

        let expected_frequency = vec![("a".to_string(), "1".to_string())];
        let result_frequency = write_token_frequency(tokens);
        assert_eq!(result_frequency, expected_frequency);
    }
}