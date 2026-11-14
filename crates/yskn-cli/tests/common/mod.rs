use serde_json::{Value, json};
pub fn bad_requests() -> Vec<(Value, &'static str)> {
    vec![
        (
            json!({"stream_options":true}),
            "stream_options must be an object",
        ),
        (json!({"messages":[]}), "messages must be a non-empty list"),
        (json!({"messages":[5]}), "role must"),
        (json!({"messages":[{"role":"function"}]}), "role must"),
        (
            json!({"messages":[{"role":"user","content":5}]}),
            "content must be",
        ),
        (
            json!({"messages":[{"role":"user","content":[{"type":"image_url"}]}]}),
            "only text content",
        ),
        (
            json!({"messages":[{"role":"assistant","tool_calls":"x"}]}),
            "tool_calls must",
        ),
        (
            json!({"messages":[{"role":"assistant","tool_calls":[{}]}]}),
            "a function call with a name",
        ),
        (
            json!({"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":"{bad"}}]}]}),
            "arguments is not valid JSON",
        ),
        (
            json!({"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":"[]"}}]}]}),
            "arguments must be a JSON object",
        ),
        (json!({"tools":"x"}), "tools must be a list"),
        (json!({"tools":[{"type":"function"}]}), "only tools of type"),
        (
            json!({"tools":[{"function":{}}]}),
            "the function needs a name",
        ),
        (
            json!({"tools":[{"function":{"name":"a<b"}}]}),
            "the function needs a name",
        ),
        (
            json!({"tools":[{"function":{"name":"f","parameters":[]}}]}),
            "parameters must be",
        ),
        (json!({"functions":[{}]}), "functions is not supported"),
        (
            json!({"function_call":"auto"}),
            "function_call is not supported",
        ),
        (json!({"tool_choice":"required"}), "tool_choice"),
        (json!({"tool_choice":{}}), "tool_choice"),
        (json!({"logprobs":true}), "logprobs is not supported"),
        (json!({"n":2}), "n must be 1"),
        (
            json!({"response_format":{"type":"json_object"}}),
            "response_format",
        ),
        (json!({"max_tokens":0}), "v must be between 1 and None"),
        (json!({"max_tokens":"x"}), "v must be a number"),
        (
            json!({"temperature":5}),
            "temperature must be between 0.0 and 2.0",
        ),
        (json!({"temperature":{}}), "temperature must be a number"),
        (json!({"top_p":0}), "top_p must be between 1e-06 and 1.0"),
        (json!({"top_p":"x"}), "top_p must be a number"),
        (json!({"top_k":[]}), "top_k must be a number"),
        (json!({"min_p":2}), "min_p must be between 0.0 and 1.0"),
        (json!({"min_p":"x"}), "min_p must be a number"),
        (json!({"seed":{}}), "seed must be a number"),
        (json!({"stop":["a","a","a","a","a"]}), "stop must be"),
        (json!({"stop":[5]}), "stop must be"),
        (json!({"stop":true}), "stop must be"),
        (
            json!({"chat_template_kwargs":"bad"}),
            "chat_template_kwargs must be an object",
        ),
        (
            json!({"reasoning_effort":"bad"}),
            "reasoning_effort must be",
        ),
        (json!({"reasoning_effort":[]}), "reasoning_effort must be"),
        (
            json!({"thinking_budget":-1}),
            "thinking_budget must be between 0 and None",
        ),
        (
            json!({"thinking_budget":"x"}),
            "thinking_budget must be a number",
        ),
        (json!({"temperature":"NaN"}), "temperature must be a number"),
        (json!({"seed":1e100}), "seed must be a number"),
    ]
}
