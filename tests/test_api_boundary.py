"""Adversarial serialization/snapshot regressions; offline and unittest runnable."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from pydantic import ValidationError

from gig.service import snapshot
from gig.service.contracts import Contract, EnvelopeSchema, State
from gig.service.serialization import Limits, SerializationError, dumps, loads, normalize, public_error


class SerializationBoundaryTests(unittest.TestCase):
    def rejected(self, code, call, *args, **kwargs):
        with self.assertRaises(SerializationError) as caught:
            call(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)

    def test_nonfinite_numbers_rejected_everywhere(self):
        for value in (math.nan, math.inf, -math.inf, np.float32('nan'), Decimal('NaN'), Decimal('Infinity')):
            with self.subTest(value=str(value)):
                self.rejected('NONFINITE_NUMBER', dumps, {'nested': [value]})
        for token in ('NaN', 'Infinity', '-Infinity', '1e309'):
            with self.subTest(token=token):
                self.rejected('NONFINITE_NUMBER', loads, '{"value":' + token + '}')

    def test_finite_extremes_subnormal_signed_zero_roundtrip(self):
        values = [sys.float_info.max, -sys.float_info.max, 5e-324, -5e-324, -0.0, 0.0]
        result = loads(dumps(values))
        self.assertEqual(result, values)
        self.assertEqual(math.copysign(1, result[4]), -1)
        self.assertEqual(math.copysign(1, result[5]), 1)

    def test_numeric_underflow_is_not_zero(self):
        for token in ('1e-400', '-1e-400'):
            self.rejected('NUMERIC_UNDERFLOW', loads, token)
        self.assertEqual(math.copysign(1, loads('-0.0')), -1)

    def test_integer_identity_survives_numpy_and_nested_wire(self):
        number = 9007199254740993
        self.assertEqual(loads('{"id":9007199254740993}')['id'], number)
        result = loads(dumps({'id': np.int64(number), 'other': [-(2**63)]}))
        self.assertEqual(result['id'], {'$integer': str(number)})
        self.assertEqual(result['other'][0], {'$integer': str(-(2**63))})
        self.assertEqual(normalize(np.int64(42)), 42)

    def test_decimal_exact_lexeme_and_exponent_preserved(self):
        for value in ('1.2300000000000000000000001', '-0.000', '1E+900'):
            result = loads(dumps(Decimal(value)))
            self.assertEqual(result, {'$decimal': value})

    def test_unsupported_objects_and_bytes_fail_without_repr(self):
        class SecretObject:
            def __repr__(self):
                raise AssertionError('object repr must never be called')
        for value in (b'synthetic-secret', bytearray(b'x'), SecretObject(), {1, 2}, complex(1, 2)):
            self.rejected('UNSUPPORTED_VALUE', dumps, value)
        self.rejected('NONSTRING_KEY', dumps, {1: 'value'})

    def test_control_characters_escaped_without_source_changes(self):
        source = ' NaN prose\n\t\x00"\\é\r\n '
        self.assertEqual(loads(dumps(source)), source)
        self.rejected('INVALID_JSON', loads, '"raw\x01control"')

    def test_malformed_json_utf8_and_duplicate_keys(self):
        for source in ('{', '{"a":}', '[1,]', '{"a":1}trailing'):
            self.rejected('INVALID_JSON', loads, source)
        self.rejected('INVALID_UTF8', loads, b'"\xff"')
        self.rejected('INVALID_UTF8', loads, '"\\ud800"')
        self.rejected('INVALID_UTF8', dumps, '\ud800')
        self.rejected('DUPLICATE_KEY', loads, '{"x":{"a":1,"a":2}}')
        self.rejected('DUPLICATE_KEY', loads, '{"a":1,"\\u0061":2}')

    def test_depth_limit_inside_and_outside_declared_bounds(self):
        limit = Limits(max_depth=4)
        value = [[[[1]]]]
        self.assertEqual(loads(dumps(value, limits=limit), limits=limit), value)
        self.rejected('TOO_DEEP', dumps, [value], limits=limit)
        self.rejected('TOO_DEEP', loads, '['*5+'1'+']'*5, limits=limit)
        self.assertEqual(loads('"[[[[[[[[[["', limits=limit), '[[[[[[[[[[')

    def test_tagged_scalar_expansion_obeys_wire_depth_limits(self):
        for value in (Decimal('1.0'), 2**63):
            self.rejected('TOO_DEEP', dumps, [[value]], limits=Limits(max_depth=2))
            wire = dumps([value], limits=Limits(max_depth=2))
            self.assertEqual(dumps(loads(wire, limits=Limits(max_depth=2)), limits=Limits(max_depth=2)), wire)

    def test_collection_node_and_string_bounds(self):
        limit = Limits(max_items=2, max_nodes=20, max_string=4)
        self.rejected('TOO_MANY_ITEMS', dumps, [1, 2, 3], limits=limit)
        self.rejected('TOO_MANY_ITEMS', loads, '{"a":1,"b":2,"c":3}', limits=limit)
        self.rejected('STRING_TOO_LONG', dumps, '12345', limits=limit)
        self.rejected('STRING_TOO_LONG', loads, '"12345"', limits=limit)
        self.rejected('STRING_TOO_LONG', dumps, Decimal('12345'), limits=limit)
        self.rejected('STRING_TOO_LONG', dumps, 2**63, limits=limit)
        self.rejected('TOO_MANY_NODES', dumps, [1, 2], limits=Limits(max_nodes=2))

    def test_utf8_byte_limits_and_replay(self):
        self.rejected('PAYLOAD_TOO_LARGE', dumps, 'éé', limits=Limits(max_bytes=5))
        self.rejected('PAYLOAD_TOO_LARGE', loads, '"éé"', limits=Limits(max_bytes=5))
        source = {'b': [Decimal('1.0'), np.int64(2**62)], 'a': -0.0}
        first = dumps(source)
        self.assertEqual(dumps(loads(first)), first)
        self.assertEqual(first, dumps({'a': -0.0, 'b': source['b']}))

    def test_cycles_in_containers_and_object_arrays(self):
        value = []; value.append(value)
        self.rejected('CYCLIC_VALUE', dumps, value)
        array = np.empty(1, dtype=object); array[0] = array
        self.rejected('CYCLIC_VALUE', dumps, array)
        self.assertEqual(normalize(np.array([[1, 2], [3, 4]])), [[1, 2], [3, 4]])

    def test_extended_numpy_precision_never_silently_narrows(self):
        if np.dtype(np.longdouble).itemsize > 8:
            self.rejected('UNSUPPORTED_FLOAT_PRECISION', dumps, np.longdouble('1.000000000000000001'))

    def test_giant_integer_interpreter_limit_sanitized(self):
        self.rejected('INTEGER_TOO_LARGE', dumps, 10**10000)

    def test_aware_timestamp_microseconds_and_nanoseconds(self):
        micros = datetime(2026, 1, 2, 3, 4, 5, 123456, tzinfo=timezone(timedelta(hours=5, minutes=30)))
        nanos = pd.Timestamp('2026-01-02T03:04:05.123456789+05:30')
        self.assertEqual(normalize(micros), '2026-01-01T21:34:05.123456+00:00')
        self.assertEqual(normalize(nanos), '2026-01-01T21:34:05.123456789+00:00')
        self.assertEqual(dumps(nanos), dumps(nanos.tz_convert('UTC')))
        self.rejected('AMBIGUOUS_TIMESTAMP', dumps, datetime(2026, 1, 1))
        self.rejected('AMBIGUOUS_TIMESTAMP', dumps, pd.NaT)

    def test_legacy_reader_does_not_rewrite_research_text(self):
        source = '{"source":"NaN Analytics","text":"NaN Infinity -Infinity","value":NaN}'
        result = loads(source, legacy_nonfinite=True)
        self.assertEqual(result['source'], 'NaN Analytics')
        self.assertEqual(result['text'], 'NaN Infinity -Infinity')
        self.assertIsNone(result['value'])

    def test_invalid_server_limits_rejected(self):
        for kwargs in ({'max_depth': 129}, {'max_items': -1}, {'max_bytes': True}, {'max_string': 1.5}):
            with self.assertRaises(ValueError): Limits(**kwargs)


class ContractBoundaryTests(unittest.TestCase):
    def available(self, data=None, **kwargs):
        return Contract.admit(State.AVAILABLE, {'value': 1} if data is None else data,
                              source='fixture', classification='SYNTHETIC', **kwargs)

    def test_all_five_truth_states_and_null_semantics(self):
        available = self.available({'value': 0}).to_dict()
        self.assertEqual(available['data']['value'], 0)
        for state in (State.UNAVAILABLE, State.UNKNOWN, State.ERROR):
            contract = Contract.admit(state, None, source='local',
                                      error=public_error() if state == State.ERROR else None)
            self.assertIsNone(contract.to_dict()['data'])
        stale = Contract.admit(State.STALE, {'value': None}, source='fixture', classification='HISTORICAL',
                               as_of='2026-01-01T00:00:00.000000001Z')
        self.assertEqual(stale.to_dict()['state'], 'STALE')
        self.assertEqual(stale.to_dict()['provenance']['as_of'], '2026-01-01T00:00:00.000000001+00:00')

    def test_inconsistent_states_and_errors_rejected(self):
        invalid = [(State.AVAILABLE, None, {}), (State.UNKNOWN, {}, {}),
                   (State.ERROR, None, {}), (State.STALE, {}, {'classification': 'HISTORICAL'}),
                   (State.AVAILABLE, {}, {'error': public_error()})]
        for state, data, kwargs in invalid:
            with self.subTest(state=state, kwargs=kwargs), self.assertRaises(ValueError):
                Contract.admit(state, data, source='fixture', **kwargs)

    def test_provenance_identity_and_equivalent_instants(self):
        one = self.available(as_of='2026-01-01T05:30:00.123456789+05:30',
                             instrument_identity='EXCHANGE:ISIN:9007199254740993',
                             source_identity='provider-feed', model_identity='model-sha', model_version='v1')
        two = self.available(as_of='2026-01-01T00:00:00.123456789Z',
                             instrument_identity='EXCHANGE:ISIN:9007199254740993',
                             source_identity='provider-feed', model_identity='model-sha', model_version='v1')
        self.assertEqual(one.wire, two.wire)
        self.assertEqual(one.to_dict()['provenance']['model_identity'], 'model-sha')
        self.assertEqual(one.to_dict()['provenance']['readiness'], 'RESEARCH_ONLY')
        with self.assertRaises(ValueError): self.available(as_of='2026-01-01')
        with self.assertRaises(ValueError): self.available(as_of='2026-01-01T00:00:00.1234567890Z')

    def test_mutation_after_admission_and_replay(self):
        source = {'rows': [{'id': 9007199254740993, 'text': ' NaN original\n'}]}
        contract = self.available(source)
        admitted = contract.wire
        source['rows'][0]['text'] = 'mutated'
        returned = contract.to_dict(); returned['data']['rows'].clear()
        self.assertEqual(contract.wire, admitted)
        self.assertEqual(Contract(admitted).wire, admitted)
        self.assertEqual(contract.to_dict()['data']['rows'][0]['text'], ' NaN original\n')

    def test_direct_schema_cannot_admit_unserializable_data_or_false_states(self):
        value = self.available().to_dict()
        value['state'] = 'ERROR'
        with self.assertRaises(ValidationError): EnvelopeSchema.model_validate(value)
        value = self.available().to_dict(); value['data'] = {'x': b'secret'}
        with self.assertRaises(ValueError): EnvelopeSchema.model_validate(value)

    def test_content_tampering_and_noncanonical_wire_rejected(self):
        value = self.available().to_dict(); value['data']['value'] = 2
        with self.assertRaises(ValueError): Contract(dumps(value).encode())
        with self.assertRaises(ValueError): Contract(json.dumps(self.available().to_dict()).encode())
        with self.assertRaises(ValueError): Contract(bytearray(self.available().wire))

    def test_external_error_text_must_be_canonical(self):
        with self.assertRaises(ValueError):
            Contract.admit(State.ERROR, None, source='local',
                           error={'code': 'INTERNAL_ERROR', 'message': 'SYNTHETIC_FAKE_PROVIDER_SECRET'})
        self.assertEqual(public_error('unsupported'), public_error())


class SnapshotBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = SimpleNamespace(results_dir=self.root)
        self.patch = patch.object(snapshot, 'get_settings', return_value=self.settings)
        self.patch.start(); self.addCleanup(self.patch.stop)
        snapshot.clear_cache(); self.addCleanup(snapshot.clear_cache)

    def write(self, value):
        path = self.root/'equity_ls_last.json'
        path.write_bytes(value if isinstance(value, bytes) else value.encode())
        return path

    def test_research_source_text_identity_and_source_bytes_preserved(self):
        original = b'{"source":"NaN Analytics","source_text":"  NaN prose\\n","config_hash":"000001","metrics":{"sharpe":NaN},"factor_ic_n":{"factor":9007199254740993}}'
        path = self.write(original)
        result = snapshot.research_snapshot()
        self.assertTrue(result['available'])
        self.assertEqual(result['source'], 'NaN Analytics')
        self.assertEqual(result['source_text'], '  NaN prose\n')
        self.assertEqual(result['config_hash'], '000001')
        self.assertIsNone(result['metrics']['sharpe'])
        self.assertEqual(result['factor_ic_n']['factor'], 9007199254740993)
        self.assertEqual(result['evidence']['source_sha256'], hashlib.sha256(original).hexdigest())
        self.assertEqual(path.read_bytes(), original)
        self.assertTrue(result['modified'].endswith('+00:00'))
        self.assertEqual(loads(dumps(result))['factor_ic_n']['factor'], {'$integer':'9007199254740993'})

    def test_research_bad_shapes_unicode_duplicate_keys_and_size_sanitized(self):
        for raw in (b'{"metrics":[]}', b'{"metrics":"SYNTHETIC_FAKE_PROVIDER_SECRET"}',
                    b'{"source_text":{}}', b'{"a":1,"a":2}', b'\xff', b'['+b'0,'*600000+b'0]'):
            with self.subTest(raw_size=len(raw)):
                self.write(raw); result = snapshot.research_snapshot()
                self.assertFalse(result['available']); self.assertEqual(result['state'], 'ERROR')
                self.assertEqual(result['error'], public_error()['message'])

    def test_research_no_data_is_unavailable(self):
        self.assertEqual(snapshot.research_snapshot()['state'], 'UNAVAILABLE')

    def test_research_read_error_never_emits_provider_diagnostics(self):
        self.write('{}')
        with patch.object(Path, 'open', side_effect=RuntimeError('SYNTHETIC_FAKE_PROVIDER_SECRET')):
            self.assertNotIn('SYNTHETIC_FAKE_PROVIDER_SECRET', dumps(snapshot.research_snapshot()))

    def test_cache_admits_bytes_and_detaches_builder_and_callers(self):
        original = {'rows': [{'value': 1}]}
        first = snapshot._cached('boundary', 60, lambda: original)
        original['rows'][0]['value'] = 2; first['rows'][0]['value'] = 3
        second = snapshot._cached('boundary', 60, lambda: self.fail('cache should hit'))
        self.assertEqual(second['rows'][0]['value'], 1)
        self.assertIsNot(first, second)
        self.assertIsInstance(snapshot._CACHE['boundary'][1], bytes)

    def test_cache_rejects_bad_wire_values_before_storage(self):
        with self.assertRaises(SerializationError): snapshot._cached('bad', 60, lambda: {'value': math.nan})
        self.assertNotIn('bad', snapshot._CACHE)
        with self.assertRaises(ValueError): snapshot._cached('bad-ttl', -1, lambda: {})

    def test_cache_preserves_identity_timestamps_and_monotonic_expiry(self):
        stamp = pd.Timestamp('2026-01-01T00:00:00.123456789Z')
        with patch.object(snapshot.time, 'monotonic', return_value=10):
            first = snapshot._cached('identity', 5, lambda: {'id':np.int64(2**62), 'at':stamp})
        self.assertEqual(first['id'], {'$integer': str(2**62)})
        self.assertEqual(first['at'], '2026-01-01T00:00:00.123456789+00:00')
        with patch.object(snapshot.time, 'monotonic', return_value=16):
            self.assertEqual(snapshot._cached('identity', 5, lambda: {'new':1}), {'new':1})

    def test_cache_entry_count_bounded(self):
        for i in range(snapshot._MAX_CACHE_ENTRIES+3):
            snapshot._cached(f'entry-{i}', 60, lambda: {'value':1})
        self.assertLessEqual(len(snapshot._CACHE), snapshot._MAX_CACHE_ENTRIES)

    def test_experiment_text_preserved_invalid_rows_counted(self):
        raw = '{"name":"NaN experiment","config":{"text":"NaN prose"},"metrics":{"sharpe":NaN}}\nnot-json\n{"metrics":[]}\n'
        (self.root/'experiments.jsonl').write_text(raw)
        result = snapshot.experiment_history()
        self.assertEqual(result['rejected_lines'], 2)
        self.assertEqual(result['runs'][0]['name'], 'NaN experiment')
        self.assertEqual(result['runs'][0]['config']['text'], 'NaN prose')
        self.assertEqual(result['evidence']['source_sha256'], hashlib.sha256(raw.encode()).hexdigest())
        self.assertIsNone(result['runs'][0]['sharpe'])

    def test_history_all_invalid_is_error_not_empty_success(self):
        (self.root/'experiments.jsonl').write_text('not-json\n')
        result = snapshot.experiment_history()
        self.assertEqual(result['state'], 'ERROR'); self.assertFalse(result['available'])

    def test_clean_never_rounds_numpy_identity(self):
        self.assertEqual(snapshot._clean({'id':np.int64(9007199254740993)}),
                         {'id': {'$integer':'9007199254740993'}})
        with self.assertRaises(SerializationError): snapshot._clean({'x':b'x'})


class ProviderErrorBoundaryTests(unittest.TestCase):
    def service_fixture(self, root):
        return SimpleNamespace(data_source='synthetic', db_path=root/'lake.duckdb', universe_file=root/'universe.yaml',
                               keys_status=lambda:{'alpaca':False}, max_gross_leverage=2., max_net_exposure=.1,
                               max_name_weight=.05, max_sector_net=.1, spread_bps=4., commission_bps=1., impact_eta=.1,
                               use_optimizer=True, target_vol=.1)

    def test_busy_database_cannot_invent_row_count_from_file_size(self):
        from gig.nlp import ollama
        from gig.ops import status
        @contextlib.contextmanager
        def busy_lock(**kwargs):
            raise TimeoutError('SYNTHETIC_FAKE_PROVIDER_SECRET')
            yield
        fake = types.ModuleType('gig.data.lake'); fake.db_lock = busy_lock
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); settings = self.service_fixture(root)
            settings.db_path.write_bytes(b'0'*1_000_001)
            with patch.object(snapshot, 'get_settings', return_value=settings), \
                 patch.object(snapshot, '_STATUS_CACHE', {'ts':0.,'tables':{},'asof':None}), \
                 patch.object(ollama, 'ollama_status', return_value={'available':False}), \
                 patch.object(status, 'ops_snapshot', return_value={'freshness':{'asof':'2026-01-01'}}), \
                 patch.dict(sys.modules, {'gig.data.lake':fake}):
                result = snapshot.service_status()
        self.assertEqual(result['state'], 'UNKNOWN')
        self.assertEqual(result['tables'], {})
        self.assertFalse(result['db_ready'])

    def test_failed_database_returns_explicit_stale_cached_observation(self):
        from gig.nlp import ollama
        from gig.ops import status
        @contextlib.contextmanager
        def broken_lock(**kwargs):
            raise RuntimeError('SYNTHETIC_FAKE_PROVIDER_SECRET')
            yield
        fake = types.ModuleType('gig.data.lake'); fake.db_lock = broken_lock
        with tempfile.TemporaryDirectory() as directory:
            settings = self.service_fixture(Path(directory))
            cache = {'ts':0.,'tables':{'bars':7},'asof':'2026-01-01T00:00:00.123456789Z'}
            with patch.object(snapshot, 'get_settings', return_value=settings), patch.object(snapshot, '_STATUS_CACHE', cache), \
                 patch.object(ollama, 'ollama_status', return_value={'available':False,'host':'secret-host','error':'SYNTHETIC_FAKE_PROVIDER_SECRET'}), \
                 patch.object(status, 'ops_snapshot', return_value={}), patch.dict(sys.modules, {'gig.data.lake':fake}):
                result = snapshot.service_status()
        self.assertEqual(result['state'], 'STALE')
        self.assertEqual(result['tables']['bars'], 7)
        self.assertEqual(result['asof'], cache['asof'])
        self.assertEqual(result['data_freshness_state'], 'UNKNOWN')
        self.assertNotIn('SYNTHETIC_FAKE_PROVIDER_SECRET', dumps(result))
        self.assertNotIn('secret-host', dumps(result))

    def test_new_database_observation_can_decrease_count_and_preserves_nanoseconds(self):
        from gig.nlp import ollama
        from gig.ops import status
        fake = types.ModuleType('gig.data.lake'); fake.db_lock = lambda **kwargs:contextlib.nullcontext()
        stamp = pd.Timestamp('2026-01-01T00:00:00.123456789Z')
        store = SimpleNamespace(stats=lambda:{'bars':3}, con=SimpleNamespace(execute=lambda query:SimpleNamespace(fetchone=lambda:(stamp,))))
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(snapshot, 'get_settings', return_value=self.service_fixture(Path(directory))), \
                 patch.object(snapshot, '_STATUS_CACHE', {'ts':0.,'tables':{'bars':99},'asof':None}), \
                 patch.object(snapshot, '_open_store', return_value=store), \
                 patch.object(ollama, 'ollama_status', return_value={'available':False}), \
                 patch.object(status, 'ops_snapshot', return_value={}), patch.dict(sys.modules, {'gig.data.lake':fake}):
                result = snapshot.service_status()
        self.assertEqual(result['tables']['bars'], 3)
        self.assertEqual(result['asof'], stamp.isoformat())
        self.assertEqual(result['state'], 'AVAILABLE')

    def test_ollama_status_sanitizes_error_and_endpoint_credentials(self):
        from gig.nlp import ollama
        with patch.object(ollama, '_get', side_effect=RuntimeError('SYNTHETIC_FAKE_PROVIDER_SECRET')), \
             patch.object(ollama, '_host', return_value='http://secret-user:secret-pass@localhost:11434/path?token=secret-query'):
            result = ollama.ollama_status()
        self.assertEqual(result['error'], public_error()['message'])
        self.assertEqual(result['host'], 'REDACTED_CONFIGURED_ENDPOINT')
        self.assertNotIn('SYNTHETIC_FAKE_PROVIDER_SECRET', dumps(result))
        self.assertNotIn('secret-pass', dumps(result))

    def test_ollama_brief_preserves_original_question_in_actual_prompt(self):
        from gig.nlp import ollama
        question = '  NaN is quoted evidence\n\t'
        posted = []
        def post(path, payload, **kwargs):
            posted.append(payload); return {'response':'answer'}
        with patch.object(ollama, 'ollama_status', return_value={'available':True}), \
             patch.object(ollama, '_snapshot_context', return_value={}), \
             patch.object(ollama, 'structured_brief', return_value='fallback'), patch.object(ollama, '_post', side_effect=post):
            result = ollama.research_brief(question)
        self.assertEqual(result['question'], question)
        self.assertIn('OPERATOR QUESTION:\n'+question+'\n\n', posted[0]['prompt'])

    def test_ollama_brief_failure_is_sanitized(self):
        from gig.nlp import ollama
        with patch.object(ollama, 'ollama_status', return_value={'available':True}), \
             patch.object(ollama, '_snapshot_context', return_value={}), \
             patch.object(ollama, 'structured_brief', return_value='fallback'), \
             patch.object(ollama, '_post', side_effect=RuntimeError('SYNTHETIC_FAKE_PROVIDER_SECRET')):
            result = ollama.research_brief('original')
        self.assertEqual(result['answer'], 'fallback')
        self.assertEqual(result['error'], public_error()['message'])

    def test_ops_and_attribution_database_errors_sanitized(self):
        from gig.ops import status
        from gig.research import attribution
        @contextlib.contextmanager
        def broken_lock(**kwargs):
            raise RuntimeError('SYNTHETIC_FAKE_PROVIDER_SECRET')
            yield
        fake = types.ModuleType('gig.data.lake'); fake.db_lock = broken_lock; fake.open_store = lambda **kwargs:None
        with patch.dict(sys.modules, {'gig.data.lake':fake}):
            ops = status._last_trade_run(); trail = attribution.paper_nav_trail()
        self.assertEqual(ops['error'], public_error()['message'])
        self.assertEqual(trail['error'], public_error()['message'])
        self.assertNotIn('SYNTHETIC_FAKE_PROVIDER_SECRET', dumps([ops,trail]))


if __name__ == '__main__':
    unittest.main()
