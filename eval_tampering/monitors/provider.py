"""Shared bounded Responses transport; each caller owns its prompt and result schema."""

from datetime import date
from decimal import Decimal
import time

from ..messages import InputError, artifact_ref, atomic_bytes, atomic_json, decode_json, fields, require


def _decimal(value, name):
    require(type(value) is str and len(value) <= 32, f'{name} must be a decimal string')
    try:
        result = Decimal(value)
    except ArithmeticError as exc:
        raise InputError('invalid_input', f'Invalid {name}') from exc
    require(result.is_finite() and result >= 0, f'Invalid {name}')
    return result


def _integer(value, low, high, name):
    require(type(value) is int and low <= value <= high, f'Invalid {name}')


def validate_provider(provider):
    if provider is not None:
        fields(provider, {'model', 'reasoning_effort', 'max_input_tokens', 'max_output_tokens', 'max_calls',
                          'max_seconds', 'timeout_seconds', 'max_cost_usd', 'rates'}, 'provider config')
        require(type(provider['model']) is str and 1 <= len(provider['model']) <= 128 and
                not any(char.isspace() for char in provider['model']), 'Explicit provider model ID required')
        require(provider['reasoning_effort'] in ('none', 'minimal', 'low', 'medium', 'high'), 'Invalid reasoning effort')
        for key, low, high in [('max_input_tokens', 1, 131072), ('max_output_tokens', 16, 8192),
                               ('max_calls', 1, 4096), ('max_seconds', 1, 86400), ('timeout_seconds', 1, 600)]:
            _integer(provider[key], low, high, key)
        require(_decimal(provider['max_cost_usd'], 'max_cost_usd') > 0, 'A positive declared per-batch API cap is required')
        fields(provider['rates'], {'input', 'cached_input', 'output', 'verified_at', 'source'}, 'rates')
        for name in ('input', 'cached_input', 'output'):
            _decimal(provider['rates'][name], name)
        require(_decimal(provider['rates']['input'], 'input') > 0 and _decimal(provider['rates']['output'], 'output') > 0, 'Input/output rates must be positive')
        try:
            date.fromisoformat(provider['rates']['verified_at'])
        except (ValueError, TypeError) as exc:
            raise InputError('invalid_input', 'Record the rate verification date') from exc
        require(type(provider['rates']['source']) is str and provider['rates']['source'].startswith('https://'), 'Record the rate source URL')


def response_json(response, expected_model):
    require(type(response) is dict and response.get('model') == expected_model, 'Provider model changed', 'provider_model_mismatch')
    require(response.get('status') == 'completed', 'Provider response did not complete', 'provider_incomplete')
    require(type(response.get('output')) is list, 'Missing provider output', 'provider_malformed')
    texts = []
    for item in response['output']:
        require(type(item) is dict, 'Invalid provider output item', 'provider_malformed')
        if item.get('type') == 'reasoning':
            continue
        require(item.get('type') == 'message' and item.get('role') == 'assistant' and item.get('status') == 'completed' and
                type(item.get('content')) is list, 'Unexpected provider output item', 'provider_malformed')
        for content in item['content']:
            require(type(content) is dict, 'Invalid provider content', 'provider_malformed')
            require(content.get('type') != 'refusal', 'Provider refused to score', 'provider_refusal')
            require(content.get('type') == 'output_text' and type(content.get('text')) is str, 'Invalid provider text', 'provider_malformed')
            texts.append(content['text'])
    require(len(texts) == 1, 'Expected exactly one structured provider answer', 'provider_malformed')
    answer = decode_json(texts[0])
    return answer


def call_json(client, provider, request, directory, ledger, started, parse_answer):
    """One synchronous counted request, no automatic retries or replacement scores."""
    import openai
    remaining = provider['max_seconds'] - (time.monotonic() - started)
    require(client is not None, 'Supply an explicit OpenAI client', 'provider_unavailable')
    require(not ledger['blocked'], 'Earlier external-call charge is unknown', 'unknown_charge')
    require(ledger['calls'] < provider['max_calls'] and remaining > 0, 'Provider call/time limit reached', 'provider_limit')
    prices = {name: _decimal(provider['rates'][name], name) for name in ('input', 'cached_input', 'output')}
    maximum = _decimal(provider['max_cost_usd'], 'max_cost_usd')
    spent = _decimal(ledger['accounted_usd'], 'accounted_usd')
    require(spent + provider['max_output_tokens'] * prices['output'] / 1000000 <= maximum, 'Insufficient remaining API allowance', 'cost_limit')
    client = client.with_options(max_retries=0, timeout=min(provider['timeout_seconds'], remaining))
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / 'request.json', request)
    row = {'status': 'incomplete', 'reservation_usd': '0', 'cost_usd': None,
           'request': artifact_ref(directory / 'request.json', 'json')}
    atomic_json(directory / 'record.json', row)
    try:
        ledger['count_requests'] += 1
        count = client.responses.input_tokens.with_raw_response.count(**request)
        atomic_bytes(directory / 'count-response.json', count.content)
        count_data = decode_json(count.content)
        require(type(count_data) is dict, 'Invalid token count response', 'provider_malformed')
        tokens = count_data.get('input_tokens')
        _integer(tokens, 0, provider['max_input_tokens'], 'counted input tokens')
        reserved = (tokens * max(prices['input'], prices['cached_input']) + provider['max_output_tokens'] * prices['output']) / 1000000
        require(spent + reserved <= maximum, 'Insufficient remaining API allowance', 'cost_limit')
        remaining = provider['max_seconds'] - (time.monotonic() - started)
        require(remaining > 0, 'Provider time limit reached before generation', 'provider_limit')
        ledger['calls'] += 1
        ledger['accounted_usd'] = str(spent + reserved)
        ledger['blocked'] = True  # A sent request stays reserved until usage is verified.
        row.update(counted_input_tokens=tokens, reservation_usd=str(reserved), charge_status='unknown')
        atomic_json(directory / 'record.json', row)
        atomic_json(directory.parent / 'ledger.json', ledger)
        response = client.responses.with_raw_response.create(**request, max_output_tokens=provider['max_output_tokens'],
            service_tier='default', store=False, background=False, stream=False,
            timeout=min(provider['timeout_seconds'], remaining))
        atomic_bytes(directory / 'response.json', response.content)
        row['response'] = artifact_ref(directory / 'response.json', 'json')
        row['request_id'] = response.headers.get('x-request-id')
        require(len(response.content) <= 4194304, 'Oversized provider response', 'provider_malformed')
        data = decode_json(response.content)
        require(type(data) is dict and type(data.get('usage')) is dict, 'Provider usage unavailable', 'unknown_charge')
        require(data.get('model') == provider['model'], 'Provider model changed', 'provider_model_mismatch')
        usage = data['usage']
        for name in ('input_tokens', 'output_tokens', 'total_tokens'):
            _integer(usage.get(name), 0, 262144, name)
        details = usage.get('input_tokens_details')
        require(type(details) is dict, 'Missing input token accounting', 'unknown_charge')
        cached = details.get('cached_tokens')
        _integer(cached, 0, usage['input_tokens'], 'cached input tokens')
        require(details.get('cache_write_tokens', 0) == 0 and usage['input_tokens'] <= tokens and
                usage['output_tokens'] <= provider['max_output_tokens'] and
                usage['total_tokens'] == usage['input_tokens'] + usage['output_tokens'] and
                data.get('service_tier') == 'default', 'Unexpected provider billing/token limits', 'unknown_charge')
        cost = ((usage['input_tokens'] - cached) * prices['input'] + cached * prices['cached_input'] + usage['output_tokens'] * prices['output']) / 1000000
        require(cost <= reserved, 'Actual cost exceeds reservation', 'unknown_charge')
        row.update(cost_usd=str(cost), usage=usage, charge_status='known')
        ledger.update(accounted_usd=str(spent + cost), blocked=False)
        answer = parse_answer(data, provider['model'])
        row.update(**answer)
    except openai.APIError as exc:
        if getattr(exc, 'response', None) is not None:
            atomic_bytes(directory / 'provider-error.txt', exc.response.content)
        row.update(status='provider_error', error_type=type(exc).__name__, status_code=getattr(exc, 'status_code', None))
        ledger['blocked'] = True
    except InputError as exc:
        row.update(status=exc.code, error=str(exc))
    finally:
        atomic_json(directory / 'record.json', row)
        atomic_json(directory.parent / 'ledger.json', ledger)
    return row | {'record': artifact_ref(directory / 'record.json', 'json')}
