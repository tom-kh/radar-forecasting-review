"""Hand-derived fixtures for cache geometry QA; no models or sensor readers."""
import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from doppler_jepa.inspect_real import (
    quantiles, range_bin_counts, validate_window, summarize_windows,
    select_windows, window_geometry,
)


@pytest.fixture
def example():
    # A 90 degree ego turn maps world [10, 22, 0] to current [2, 0, 0].
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    frames = {}
    for i, token in enumerate(['a', 'b', 'c', 'd', 'e', 'f', 'g']):
        points = np.array([[10., 22., 0., 0., 1., 0., 2., 10., -.1]])
        frames[token] = {
            'timestamp': np.array(1_000_000 + i * 500_000),
            'single_min_us': np.array(900_000 + i * 500_000),
            'rotation': rotation.copy(), 'translation': np.array([10., 20., 0.]),
            'points': points.copy(), 'single': points.copy(),
            'polygons': np.array([[[9., 21., 0.], [11., 21., 0.],
                                   [11., 23., 0.], [9., 23., 0.]]]),
            'moving': np.array([False]),
        }
    window = {'scene': 'scene', 'anchor': 'd', 'history': ['a', 'b', 'c', 'd'],
              'future': ['e', 'f', 'g']}
    metadata = {'synthetic': False, 'history': 4, 'future': 3,
                'radars': ['RADAR_FRONT'], 'prepare_config': {'max_age': .45},
                'preprocess_skips': {'train': {}, 'dev': {}, 'test': {}}}
    return window, frames, metadata


def test_quantiles_fixed_vector_and_empty():
    assert quantiles([0., 100.]) == {
        'p0': 0., 'p25': 25., 'p50': 50., 'p75': 75.,
        'p90': 90., 'p95': 95., 'p99': 99.}
    assert all(value is None for value in quantiles([]).values())
    with pytest.raises(ValueError, match='finite'):
        quantiles([1., np.nan])


def test_range_boundaries_and_out_of_range_exclusion():
    xy = np.array([[0, 0], [19.9, 0], [0, 20], [39.9, 0],
                   [40, 0], [0, 64], [64.01, 0], [np.nan, 0]])
    assert range_bin_counts(xy) == {'[0,20)': 2, '[20,40)': 2, '[40,64]': 2}


def test_history_offsets_future_offsets_and_transform(example):
    window, frames, metadata = example
    result = validate_window(window, frames, metadata)
    assert result['status'] == 'PASS'
    assert result['history_dt_seconds'] == [-1.5, -1., -.5, 0.]
    assert result['future_dt_seconds'] == [.5, 1., 1.5]
    geometry = window_geometry(window, frames)
    np.testing.assert_allclose(geometry['history_xy'], [[2., 0.]] * 4, atol=1e-12)
    np.testing.assert_allclose(geometry['history_age_seconds'], [-1.6, -1.1, -.6, -.1])
    np.testing.assert_allclose(geometry['latest_velocity_xy'], [[2., 0.]], atol=1e-12)
    np.testing.assert_allclose(geometry['current_polygons'][0, 0], [1., 1., 0.])
    # Rendering registration must never mutate cached radar ages.
    assert frames['a']['points'][0, 8] == -.1


@pytest.mark.parametrize('change,check', [
    ('repeated_timestamp', 'strict_timestamp_order'),
    ('future_packet', 'causal_radar'),
    ('future_overlap', 'future_measurements_after_anchor'),
    ('future_single_overlap', 'future_measurements_after_anchor'),
    ('nan_points', 'finite_radar'), ('nan_polygon', 'finite_polygons'),
    ('too_old', 'max_age'), ('missing', 'frame_files_present'),
    ('invalid', 'frame_validity'), ('skipped', 'frame_validity'),
    ('anchor', 'anchor_matches_history'), ('long_gap', 'timestamp_gap'),
    ('nan_pose', 'finite_pose'), ('bad_shape', 'frame_shapes'),
])
def test_invalid_windows_report_specific_failed_check(example, change, check):
    window, frames, metadata = example
    if change == 'repeated_timestamp':
        frames['b']['timestamp'] = frames['a']['timestamp']
    elif change == 'future_packet':
        frames['a']['points'][0, 8] = .01
    elif change == 'future_overlap':
        frames['e']['single_min_us'] = frames['d']['timestamp']
    elif change == 'future_single_overlap':
        frames['e']['single'][0, 8] = -.5
    elif change == 'nan_points':
        frames['c']['single'][0, 4] = np.nan
    elif change == 'nan_polygon':
        frames['g']['polygons'][0, 0, 0] = np.inf
    elif change == 'too_old':
        frames['a']['points'][0, 8] = -.45001
    elif change == 'missing':
        del frames['b']
    elif change == 'invalid':
        frames['b']['valid'] = np.array(False)
    elif change == 'skipped':
        metadata['preprocess_skips']['test']['other_scene'] = {
            'no_causal_radar_frames': 1, 'tokens': ['b']}
    elif change == 'anchor':
        window['anchor'] = 'c'
    elif change == 'long_gap':
        frames['g']['timestamp'] += 2_000_000
    elif change == 'nan_pose':
        frames['d']['rotation'][0, 0] = np.nan
    elif change == 'bad_shape':
        frames['a']['points'] = np.zeros((2, 8))
    result = validate_window(window, frames, metadata)
    assert result['status'] == 'FAIL'
    assert result['checks'][check] is False
    assert result['errors']


def test_age_roundoff_tolerance_is_explicit(example):
    window, frames, metadata = example
    frames['a']['points'][0, 8] = -.4500005
    result = validate_window(window, frames, metadata)
    assert result['status'] == 'PASS'
    assert result['age_tolerance_seconds'] == 1e-6


def test_summary_counts_ages_empty_latest_and_skips(example):
    window, frames, metadata = example
    frames['d']['points'] = np.empty((0, 9))
    frames['d']['single'] = np.empty((0, 9))
    metadata['preprocess_skips']['test']['elsewhere'] = {
        'no_causal_radar_frames': 2, 'tokens': ['missing1', 'missing2']}
    result = summarize_windows([window], frames, metadata)
    assert result['status'] == 'PASS'
    assert result['empty_latest_fraction'] == 1.
    assert result['points_per_frame']['p0'] == 0.
    assert result['age_seconds']['p0'] == pytest.approx(-1.6)
    assert result['future_dt_seconds']['horizon_3']['p50'] == 1.5
    assert result['range_bins'] == {'[0,20)': 3, '[20,40)': 0, '[40,64]': 0}
    assert result['polygon_count'] == 4
    assert result['skipped_no_causal_counts'] == {'train': 0, 'dev': 0, 'test': 2}
    assert result['synthetic'] is False
    assert summarize_windows([], frames, metadata)['status'] == 'FAIL'


def test_selection_keeps_manifest_order_and_count():
    records = [{'anchor': x} for x in ['z', 'b', 'a', 'c']]
    assert select_windows(records, count=2, offset=1) == [(1, records[1]), (2, records[2])]
    assert select_windows(records, count=4, offset=3) == [(3, records[3])]
    with pytest.raises(ValueError):
        select_windows(records, count=0, offset=0)
    with pytest.raises(ValueError):
        select_windows(records, count=1, offset=-1)


def test_cli_generates_deterministic_index_and_reports_missing_file(tmp_path, example):
    window, frames, metadata = example
    cache = tmp_path / 'cache'
    (cache / 'frames').mkdir(parents=True)
    (cache / 'metadata.json').write_text(json.dumps(metadata))
    rows = [copy.deepcopy(window) for _ in range(3)]
    (cache / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    for token, frame in frames.items():
        np.savez(cache / 'frames' / f'{token}.npz', **frame)
    out = tmp_path / 'out'
    cmd = [sys.executable, '-m', 'doppler_jepa.inspect_real', '--cache', str(cache),
           '--split', 'train', '--out', str(out), '--count', '1', '--offset', '1']
    first = subprocess.run(cmd, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    qa = json.loads((out / 'qa.json').read_text())
    assert qa['selection'] == {'split': 'train', 'offset': 1, 'count': 1,
                               'selected_count': 1, 'row_indices': [1], 'tokens': ['d']}
    assert (out / '00001_d.png').stat().st_size > 1000
    html = (out / 'index.html').read_text()
    assert '00001_d.png' in html and 'src="00001_d.png"' in html
    assert subprocess.run(cmd, capture_output=True).returncode == 0
    assert (out / 'index.html').read_text() == html
    (cache / 'frames' / 'b.npz').unlink()
    assert subprocess.run(cmd, capture_output=True).returncode == 1
    qa = json.loads((out / 'qa.json').read_text())
    assert qa['status'] == 'FAIL'
    assert not qa['checks']['frame_files_present']
    assert '00001_d.png' not in (out / 'index.html').read_text()
