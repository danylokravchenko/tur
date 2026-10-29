//! Gemma's typed tool argument grammar, including escaped strings and object keys.
use super::{ToolCall, ToolDefinition};
use serde_json::{Map, Value};

const QUOTE: &str = "<|\"|>";
struct Parser<'a> {
    rest: &'a str,
}
impl Parser<'_> {
    fn consume(&mut self, token: &str) -> Option<()> {
        self.rest = self.rest.trim_start().strip_prefix(token)?;
        Some(())
    }
    fn string(&mut self) -> Option<String> {
        self.consume(QUOTE)?;
        let (text, rest) = self.rest.split_once(QUOTE)?;
        self.rest = rest;
        Some(text.into())
    }
    fn key(&mut self) -> Option<String> {
        self.rest = self.rest.trim_start();
        if self.rest.starts_with(QUOTE) {
            return self.string();
        }
        let end = self.rest.find(':')?;
        let key = self.rest.get(..end)?.trim();
        if key.is_empty()
            || key
                .chars()
                .any(|c| c.is_whitespace() || "{},[]".contains(c))
        {
            return None;
        }
        self.rest = self.rest.get(end..)?;
        Some(key.into())
    }
    fn object(&mut self, depth: usize) -> Option<Value> {
        self.consume("{")?;
        let mut object = Map::new();
        if self.consume("}").is_some() {
            return Some(object.into());
        }
        loop {
            let key = self.key()?;
            self.consume(":")?;
            if object.insert(key, self.value(depth + 1)?).is_some() {
                return None;
            }
            if self.consume("}").is_some() {
                return Some(object.into());
            }
            self.consume(",")?;
        }
    }
    fn array(&mut self, depth: usize) -> Option<Value> {
        self.consume("[")?;
        let mut values = Vec::new();
        if self.consume("]").is_some() {
            return Some(values.into());
        }
        loop {
            values.push(self.value(depth + 1)?);
            if self.consume("]").is_some() {
                return Some(values.into());
            }
            self.consume(",")?;
        }
    }
    fn value(&mut self, depth: usize) -> Option<Value> {
        if depth > 64 {
            return None;
        }
        self.rest = self.rest.trim_start();
        if self.rest.starts_with(QUOTE) {
            return self.string().map(Value::String);
        }
        if self.rest.starts_with('{') {
            return self.object(depth);
        }
        if self.rest.starts_with('[') {
            return self.array(depth);
        }
        let end = self.rest.find([',', ']', '}']).unwrap_or(self.rest.len());
        let token = self.rest.get(..end)?.trim();
        let value = serde_json::from_str::<Value>(token).ok()?;
        if !matches!(value, Value::Bool(_) | Value::Number(_) | Value::Null) {
            return None;
        }
        self.rest = self.rest.get(end..)?;
        Some(value)
    }
}

pub(super) fn parse(text: &str, tools: &[ToolDefinition]) -> Option<ToolCall> {
    let (name, body) = text.strip_prefix("call:")?.split_once('{')?;
    let name = name.trim();
    if !tools.iter().any(|t| t.name == name) {
        return None;
    }
    let input = format!("{{{body}");
    let mut parser = Parser { rest: &input };
    let arguments = parser.object(0)?;
    if !parser.rest.trim().is_empty() {
        return None;
    }
    Some(ToolCall {
        name: name.into(),
        arguments,
    })
}

#[cfg(test)]
#[allow(
    clippy::panic_in_result_fn,
    reason = "tests assert complete tool argument values"
)]
mod tests {
    use super::*;
    use crate::Result;
    #[test]
    fn test_gemma_typed_arguments_and_nested_values_preserve_strings_and_reject_malformed_calls()
    -> Result<()> {
        let tools = [ToolDefinition::new("lookup", "", serde_json::json!({}))];
        let text = r#"call:lookup{city:<|"|>Kyiv: {✓}<|"|>,days:2,filters:{<|"|>active<|"|>:true,tags:[<|"|>a,b<|"|>,null,1.5]}}"#;
        let call = parse(text, &tools)
            .ok_or_else(|| crate::TurError::Other("missing Gemma call".into()))?;
        assert_eq!(call.name, "lookup");
        assert_eq!(
            call.arguments,
            serde_json::json!({"city":"Kyiv: {✓}","days":2,"filters":{"active":true,"tags":["a,b",null,1.5]}})
        );
        for text in [
            "call:missing{}",
            "call:lookup{a:1,a:2}",
            "call:lookup{a:word}",
            "call:lookup{a:[1,]}",
            "call:lookup{} trailing",
            r#"call:lookup{a:<|"|>unclosed}"#,
        ] {
            assert!(parse(text, &tools).is_none(), "{text}");
        }
        let output = format!(
            "before <|tool_call>{text}<tool_call|> after <|tool_call>call:lookup{{days:3}}<tool_call|>"
        );
        let calls = ToolCall::parse_from_output(&output, &tools);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments, call.arguments);
        assert_eq!(calls[1].arguments, serde_json::json!({"days":3}));
        Ok(())
    }
}
