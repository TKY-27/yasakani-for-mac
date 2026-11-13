#!/usr/bin/env python3
"""Generate text-only oracle vectors with upstream Python; never load weights."""
import argparse
import hashlib
import json
from pathlib import Path

from yasakani.textio import ChatFormatter, OutputStream, StreamDecoder, ToolCallFilter, parse_tool_call, tool_schemas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_folder', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    fmt = ChatFormatter(args.model_folder)
    data = {k: [] for k in ('selfcheck', 'chat', 'tools', 'errors', 'raw', 'stream', 'output', 'tool_filter', 'decode', 'parse')}

    def rendered(category, name, messages, options=None, tools=None, prompt_sha256=None):
        options = options or {}
        kw = dict(options)
        if tools is not None:
            kw['tools'] = tools
        case = dict(name=name, messages=messages, tools=tools, options=options)
        try:
            text = fmt.render(messages, **kw)
            ids = fmt.encode(text).tolist()
            case.update(rendered=text, ids=ids, decoded=fmt.decode(ids))
            if options.get('add_generation_prompt', True):
                prompt_ids, inside = fmt.prompt(messages, **kw)
                assert prompt_ids.tolist() == ids
                case['starts_inside_thinking'] = inside
            if prompt_sha256:
                assert hashlib.sha256(b''.join(i.to_bytes(4, 'little', signed=True) for i in ids)).hexdigest() == prompt_sha256
                case['prompt_sha256'] = prompt_sha256
        except Exception as exc:
            if prompt_sha256:
                raise
            case['error'] = str(exc)
        data[category].append(case)

    def raw(category, name, text, prompt_sha256=None):
        ids = fmt.encode(text).tolist()
        case = dict(name=name, text=text, rendered=text, ids=ids, decoded=fmt.decode(ids))
        if prompt_sha256:
            assert hashlib.sha256(b''.join(i.to_bytes(4, 'little', signed=True) for i in ids)).hexdigest() == prompt_sha256
            case['prompt_sha256'] = prompt_sha256
        data[category].append(case)
        return ids

    refs = {c['name']: c for c in json.loads((args.model_folder / 'selfcheck.json').read_text())['cases']}
    rendered('selfcheck', 'short-answer', [{'role': 'user', 'content': 'Say hello in one short sentence.'}], {'enable_thinking': False}, prompt_sha256=refs['short-answer']['prompt_sha256'])
    rendered('selfcheck', 'thinking', [{'role': 'user', 'content': 'What is 17 times 23? Think it through step by step.'}], prompt_sha256=refs['thinking']['prompt_sha256'])
    raw('selfcheck', 'plain-text', 'The capital of France is', refs['plain-text']['prompt_sha256'])
    user = [{'role': 'user', 'content': 'こんにちは。Explain 17 × 23.'}]
    history = [{'role': 'system', 'content': 'Be brief.'}, {'role': 'user', 'content': 'q1'}, {'role': 'assistant', 'content': 'a1', 'reasoning_content': ' r1 日本語 '}, {'role': 'user', 'content': 'q2'}]
    for name, options in [('unset', {}), ('on', {'enable_thinking': True}), ('off', {'enable_thinking': False}), ('medium', {'reasoning_effort': 'medium'}), ('low', {'reasoning_effort': 'low'}), ('no-generation', {'add_generation_prompt': False})]:
        rendered('chat', name, user, options)
    rendered('chat', 'system-user', [{'role': 'system', 'content': ' Be helpful. '}] + user, {'enable_thinking': False})
    for preserve in (None, True, False):
        for enable in (True, False):
            options = {'enable_thinking': enable}
            if preserve is not None:
                options['preserve_thinking'] = preserve
            rendered('chat', f'history-{preserve}-{enable}', history, options)
    rendered('chat', 'missing-reasoning', [{'role': 'user', 'content': 'q1'}, {'role': 'assistant', 'content': 'a1'}] + user)
    rendered('chat', 'content-parts', [{'role': 'user', 'content': [{'type': 'text', 'text': ' hello '}, {'type': 'image_url', 'image_url': {'url': 'placeholder'}}, {'type': 'video', 'video': 'placeholder'}]}], {'add_vision_id': True})
    rendered('chat', 'null-content', user + [{'role': 'assistant', 'content': None}])
    tools = [{'type': 'function', 'function': {'name': 'weather', 'description': '天気 <>&', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}, 'options': {'type': 'object', 'properties': {'days': {'type': 'integer'}, 'units': {'type': 'array', 'items': {'type': 'string'}}}}}, 'required': ['city']}}}]
    rendered('tools', 'nested-schema', user, {'enable_thinking': False}, tools)
    assistant = {'role': 'assistant', 'content': 'Checking.', 'reasoning_content': 'Need weather.', 'tool_calls': [{'type': 'function', 'function': {'name': 'weather', 'arguments': {'city': '大阪', 'options': {'days': 3, 'units': ['C', 'F']}}}}]}
    rendered('tools', 'object-arguments', user + [assistant], tools=tools)
    string_call = json.loads(json.dumps(assistant))
    string_call['tool_calls'][0]['function']['arguments'] = '{"city": "大阪"}'
    # ChatFormatter passes through unchanged: Jinja2 rejects |items on a string.
    rendered('tools', 'string-arguments-error', user + [string_call], tools=tools)
    empty_call = json.loads(json.dumps(assistant))
    empty_call['tool_calls'][0]['function']['arguments'] = ''
    rendered('tools', 'empty-string-arguments', user + [empty_call], tools=tools)
    for count in (1, 3):
        results = [{'role': 'tool', 'tool_call_id': f'call_{i}', 'content': f'result {i}: 晴れ'} for i in range(count)]
        rendered('tools', f'tool-results-{count}', user + [assistant] + results, tools=tools)
        rendered('tools', f'tool-results-followup-{count}', user + [assistant] + results + [{'role': 'assistant', 'content': 'Sunny.'}] + user, {'preserve_thinking': False}, tools)
    for name, messages, options in [
        ('empty', [], {}), ('system-not-first', user + [{'role': 'system', 'content': 'late'}], {}),
        ('unexpected-role', user + [{'role': 'alien', 'content': 'hi'}], {}), ('no-user', [{'role': 'system', 'content': 'only system'}], {}),
        ('bad-effort', user, {'reasoning_effort': 'ultra'}),
        ('system-image', [{'role': 'system', 'content': [{'type': 'image', 'image': 'x'}]}] + user, {}),
        ('system-video', [{'role': 'system', 'content': [{'type': 'video', 'video': 'x'}]}] + user, {}),
        ('bad-part', [{'role': 'user', 'content': [{'type': 'audio'}]}], {}),
        ('bad-content', [{'role': 'user', 'content': 42}], {}),
    ]:
        rendered('errors', name, messages, options)
    texts = [
        ('ascii', 'Hello, world! A plain ASCII sentence.\n'),
        ('code', 'def f(x):\n    if x:\n\treturn x ** 2\n    # spaces   \n'),
        ('japanese', '日本語、ひらがな、カタカナ。𠮷野家'), ('chinese', '中文：简体汉字和繁體漢字。'),
        ('korean', '한국어 문장과 한글입니다.'), ('emoji', '👍🏽 👨‍👩‍👧‍👦 👩🏿‍💻 🇯🇵 😀'),
        ('nfd', 'Cafe\u0301 nai\u0308ve A\u030angstro\u0308m'), ('newlines', 'a\r\nb\nc\rd\r\n\n'),
        ('spaces', '  start' + ' ' * 80 + '\t\tend\n   '),
        ('special', '<|im_start|>user<|im_end|><think>thought</think><tool_call><function=f><parameter=x>1</parameter></function></tool_call><|endoftext|>'),
        ('contractions', "don't DON'T I'LL I'll we're WE'RE can't CAN'T"),
        ('numbers', '0 12345678901234567890 -3.14159 1e-10 1_000 ２０２６'),
        ('document-20000', ('A document 日本語 line 123.\n' * 1000)[:20000]),
    ]
    assert len(texts[-1][1]) == 20000
    for name, text in texts:
        ids = raw('raw', name, text)
        if name in ('ascii', 'japanese', 'emoji', 'nfd', 'special', 'document-20000'):
            stream = StreamDecoder(fmt.tok)
            pieces = [stream.push([i]) for i in ids]
            tail = stream.flush()
            assert ''.join(pieces) + tail == fmt.decode(ids)
            data['stream'].append(dict(name=name, text=text, ids=ids, pieces=pieces, tail=tail, decoded=fmt.decode(ids)))
    for name, reasoning, text, inside in [('thinking', '考えます 👍🏽', '\n\nAnswer 日本語.', True), ('content', '', '\n\nHello 👩🏿‍💻', False), ('unfinished', 'still reasoning', '', True)]:
        ids = fmt.encode(reasoning).tolist()
        if inside and name != 'unfinished':
            ids += [fmt.think_end]
        ids += fmt.encode(text).tolist() + [248046, 248044]
        stream = OutputStream(fmt, inside)
        pieces = [stream.feed(ids[:i]) for i in range(1, len(ids) + 1)]
        tail = stream.feed(ids, final=True)
        data['output'].append(dict(name=name, ids=ids, in_reasoning=inside, pieces=pieces, tail=tail, think_end_at=stream.think_end_at))
    call = '<tool_call>\n<function=weather>\n<parameter=city>\n大阪\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>'
    for name, text in [('plain', 'Hello 日本語  <tool, done. '), ('call', 'Checking.\n\n' + call), ('two', call + '\n' + call + '\n'), ('unfinished', 'Text\n' + call[:-20]), ('invalid', 'x\n<tool_call>bad</tool_call>'), ('json', '<tool_call>{"name":"weather","arguments":"{\\"city\\":\\"大阪\\"}"}</tool_call>')]:
        for step in (1, 7, 500):
            chunks = [text[i:i + step] for i in range(0, len(text), step)]
            filt = ToolCallFilter(tools)
            pieces = [filt.push(c) for c in chunks]
            tail = filt.flush()
            data['tool_filter'].append(dict(name=f'{name}-{step}', tools=tools, chunks=chunks, pieces=pieces, tail=tail, calls=filt.calls))
    # Review regressions: every expected result below is produced by upstream.
    whitespace = '\u001c\u001d\u001e\u001f\u00a0\u3000\u2028\t\n'
    for index, char in enumerate(whitespace):
        rendered('chat', f'review-1-content-{index}', [{'role': 'user', 'content': char + 'Hello' + char}])
        rendered('chat', f'review-1-reasoning-{index}', user + [{'role': 'assistant', 'content': 'answer', 'reasoning_content': char + 'reason' + char}], {'preserve_thinking': True})
    rendered('chat', 'review-1-mixed', [{'role': 'user', 'content': whitespace + 'Hello' + whitespace}])
    values = {'rounded': 51.248178375505404, 'one': 1.0, 'large_float': 1e20,
              'small_float': 1e-7, 'tenth': 0.1, 'large_int': 18446744073709551615,
              'negative_int': -9223372036854775808, 'yes': True, 'no': False,
              'nil': None, 'escaped': '"\\\n\u0000\u001f/\u2028日本語'}
    rendered('tools', 'review-2-schema-values', user, tools=[{'type': 'function', 'function': {'name': 'values', 'parameters': values}}])
    rendered('tools', 'review-2-argument-values', user + [{'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'values', 'arguments': values}}]}])

    def stream_case(name, ids):
        stream = StreamDecoder(fmt.tok)
        pieces = [stream.push([i]) for i in ids]
        tail = stream.flush()
        data['stream'].append(dict(name=name, ids=ids, pieces=pieces, tail=tail, decoded=fmt.decode(ids)))

    def output_case(name, ids, inside, hidden=None):
        stream = OutputStream(fmt, inside) if hidden is None else OutputStream(fmt, inside, hidden)
        pieces = [stream.feed(ids[:i]) for i in range(1, len(ids) + 1)]
        tail = stream.feed(ids, final=True)
        case = dict(name=name, ids=ids, in_reasoning=inside, pieces=pieces, tail=tail, think_end_at=stream.think_end_at)
        if hidden is not None:
            case['hidden_ids'] = hidden
        data['output'].append(case)

    emoji_ids = fmt.encode('A 👩🏿‍💻').tolist()
    partial = next(emoji_ids[:end] for end in range(len(emoji_ids) - 1, 0, -1) if fmt.decode(emoji_ids[:end]).endswith('�'))
    normal = fmt.encode('Hello 日本語.').tolist()
    unknown = [248077, 248100, 248319]
    mixed_ids = normal[:1] + unknown[:1] + normal[1:2] + unknown[1:] + normal[2:]
    data['decode'].append(dict(name='review-3-unknown', ids=mixed_ids, decoded=fmt.decode(mixed_ids)))
    stream_case('review-3-unknown', mixed_ids)
    output_case('review-3-unknown-content', mixed_ids, False)
    output_case('review-3-unknown-reasoning', mixed_ids + [fmt.think_end] + normal, True)
    output_case('review-3-unknown-hidden', mixed_ids, False, unknown + [248046, 248044])
    unknown_first = [248077] + normal
    unknown_inside_emoji = partial[:2] + [248077] + partial[2:]
    unknown_last = normal + [248319]
    for name, ids in (
        ('review-3-unknown-first', unknown_first),
        ('review-3-unknown-inside-emoji', unknown_inside_emoji),
        ('review-3-unknown-last', unknown_last),
    ):
        stream_case(name, ids)
        output_case(name + '-content', ids, False)
    scalars = [1e20, 1e-7, 1.0, 0.1, 51.248178375505404,
               18446744073709551615, -9223372036854775808, True, False, None]
    rendered('chat', 'review-4-scalars', [{'role': 'user', 'content': [{'type': 'text', 'text': value} for value in scalars]}])
    malformed = [
        ('function-newline', '<function=bad\nname><parameter=x>1</parameter>'),
        ('parameter-newline', '<function=f><parameter=bad\nname>1</parameter><parameter=ok>2</parameter>'),
        ('unterminated-newline', '<function=\n<function=valid><parameter=\n<parameter=ok>3</parameter>'),
        ('adversarial', '<function=\n' * 10000 + '<function=valid><parameter=\n' * 10000 + '<parameter=ok>3</parameter>'),
    ]
    for name, block in malformed:
        name = 'review-5-' + name
        data['parse'].append(dict(name=name, block=block, tools=tools, result=parse_tool_call(block, tool_schemas(tools))))
        text = '<tool_call>' + block + '</tool_call>'
        chunks = [text]  # Exercise the entire long tag scan without quadratic chunk construction.
        filt = ToolCallFilter(tools)
        pieces = [filt.push(c) for c in chunks]
        data['tool_filter'].append(dict(name=name, tools=tools, chunks=chunks, pieces=pieces, tail=filt.flush(), calls=filt.calls))

    for effort in (None, 'xhigh', 'medium', 'low', 'high'):
        category = 'errors' if effort in (None, 'high') else 'chat'
        rendered(category, f'review-6-effort-{effort}', user, {'reasoning_effort': effort})
    for value in (1, 0, 'yes', None):
        rendered('chat', f'review-6-thinking-{value}', user, {'enable_thinking': value})
    rendered('chat', 'review-6-system-parts', [{'role': 'system', 'content': [{'type': 'text', 'text': ' Be '}, {'type': 'text', 'text': 'brief. '}]}] + user)
    calls = [{'function': {'name': 'first', 'arguments': {'x': 1}}}, {'function': {'name': 'second', 'arguments': {'y': False}}}]
    rendered('tools', 'review-6-multiple-calls', user + [{'role': 'assistant', 'content': 'Checking.', 'tool_calls': calls}])
    rendered('tools', 'review-6-empty-content', user + [{'role': 'assistant', 'content': '', 'tool_calls': calls[:1]}])
    rendered('tools', 'review-6-flat-call', user + [{'role': 'assistant', 'content': '', 'tool_calls': [calls[0]['function']]}])
    stream_case('review-6-partial-utf8', partial)
    output_case('review-6-hidden-middle', normal[:1] + [248046] + normal[1:], False)
    output_case('review-6-second-think-end', normal + [fmt.think_end] + normal + [fmt.think_end] + normal, True)
    output_case('review-6-first-think-end', [fmt.think_end] + fmt.encode('\n\nAnswer').tolist(), True)

    unicode_inputs = [
        ('cjk-extension-i', '\U0002ebf0'), ('ol-onal-letter', '\U0001e5d0'),
        ('ol-onal-mark', '\U0001e5ee'), ('garay-digit', '\U00010d40'),
        ('cyrillic', '\u1c89'), ('unassigned', '\u0378\U00010ffff'),
        ('max-unicode-scalar', '\U0010ffff'),
        ('private-use', '\ue000\U000f0000'), ('combining', '\u0301\u20dd'),
    ]
    # Exclude only independently confirmed regex-engine differences; retain exact
    # inputs, both id sequences and library versions here if any are found.
    known_divergences = []
    for name, char in unicode_inputs:
        text = f'a{char}b A{char}B {char}abc123{char}45!{char}\n'
        if name not in {case['name'] for case in known_divergences}:
            raw('raw', 'review-7-' + name, text)
    for size in (1, 2, 3, 100, 10000):
        raw('raw', f'review-7-spaces-{size}', 'a' + ' ' * size + 'b')
        raw('raw', f'review-7-mixed-whitespace-{size}', 'a' + ' \t\n' * size + 'b')
    assert all('error' in c for c in data['errors'])
    assert all('error' not in c for c in data['chat'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(', '.join(f'{k}={len(v)}' for k, v in data.items()))


if __name__ == '__main__':
    main()
