use std::collections::HashMap;

use anyhow::{Context, Result};

#[derive(Default, Clone)]
pub struct EngineConfig {
    values: HashMap<String, String>,
}

impl EngineConfig {
    pub fn from_ini(input: &str) -> Self {
        let mut values = HashMap::new();
        for line in input.lines() {
            let line = line.trim();
            if line.is_empty()
                || line.starts_with(';')
                || line.starts_with('#')
                || line.starts_with('[')
            {
                continue;
            }
            if let Some(eq) = line.find('=') {
                let key = line[..eq].trim().to_uppercase();
                let value = line[eq + 1..].trim().to_string();
                values.insert(key, value);
            }
        }
        Self { values }
    }

    pub fn set(&mut self, key: &str, value: i64) {
        self.values.insert(key.to_uppercase(), value.to_string());
    }

    pub fn get_int(&self, key: &str) -> Result<i64> {
        let s = self
            .values
            .get(&key.to_uppercase())
            .with_context(|| format!("{key} not found in config"))?;
        s.parse::<i64>()
            .with_context(|| format!("Config key {key} is not a valid integer: {s}"))
    }

    pub fn get_bool(&self, key: &str) -> Result<bool> {
        let v = self
            .values
            .get(&key.to_uppercase())
            .with_context(|| format!("{key} not found in config"))?;
        match v.to_lowercase().as_str() {
            "true" | "1" | "yes" => Ok(true),
            "false" | "0" | "no" => Ok(false),
            _ => Err(anyhow::anyhow!("Config key {key} is not a valid bool: {v}")),
        }
    }
}
