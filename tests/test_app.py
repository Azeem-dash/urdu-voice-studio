import csv
import itertools
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import app


class TopicTests(unittest.TestCase):
    def test_counts_and_topic_identity(self):
        for counts in itertools.product((1, 2, 5, 6), repeat=3):
            requested = list(zip(['Cricket', 'Travel', 'کراچی'], counts))
            def fake_json(client, system, prompt):
                count = int(prompt.split()[1])
                return {'notes': [{'topic': 'Renamed by model', 'urdu_text': 'یہ ایک کہانی ہے۔',
                                   'roman_urdu': 'Yeh aik kahani hai.'} for _ in range(count)]}
            with patch.object(app, 'deepseek_client', return_value=object()), \
                 patch.object(app, 'deepseek_json', side_effect=fake_json):
                rows, _ = app.write_topic_notes(app.topic_plan(requested, 3), 'Male')
            self.assertEqual(Counter(row['topic'] for row in rows), dict(requested))

    def test_invalid_counts_and_batch_limit(self):
        for value in (0, -1, 1.5, 101, None, '', float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(app.gr.Error):
                app.topic_plan([['Cricket', value]], 3)
        with self.assertRaises(app.gr.Error):
            app.topic_plan([['Cricket', 60], ['Travel', 41]], 3)
        self.assertEqual(app.topic_plan([['Cricket', 100]], 3), [('Cricket', 100)])
        self.assertEqual(app.topic_table('Cricket\n\n • Travel\nکراچی', 2),
                         [['Cricket', 2], ['Travel', 2], ['کراچی', 2]])
        self.assertEqual(app.topic_plan([], 3), [('', 3)])

    def test_no_unrelated_fallback_or_missing_notes(self):
        with patch.object(app, 'deepseek_client', return_value=None):
            with self.assertRaises(RuntimeError):
                app.write_notes(['Cricket'], 1, 'Male')
            self.assertEqual(len(app.write_notes([], 3, 'Male')[0]), 3)
        with patch.object(app, 'deepseek_client', return_value=object()), \
             patch.object(app, 'deepseek_json', return_value={'notes': []}):
            with self.assertRaises(RuntimeError):
                app.write_notes(['Cricket'], 1, 'Male')

    def test_generation_downloads_are_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav = root / 'voice.wav'
            app.sf.write(wav, np.zeros(1600), 16000)
            class TTS:
                def voice(self, *args): return object()
            def fake_write(topics, count, gender):
                return [dict(topic=topic, urdu_text='کہانی۔', roman_urdu='Kahani.') for topic in topics], 'OK'
            durations = []
            def fake_audio(*args):
                target = args[4]
                durations.append(target)
                return ['کہانی۔'], np.zeros(round(target * 16000)), [(0, target)], [None]
            with patch.object(app, 'DATASETS_DIR', root), \
                 patch.object(app, 'voice_files', return_value=(wav, root / 'voice.txt')), \
                 patch.object(app, 'get_tts', return_value=TTS()), \
                 patch.object(app, 'current_voice_status', return_value='Saved voice'), \
                 patch.object(app, 'write_notes', side_effect=fake_write), \
                 patch.object(app, 'make_note', side_effect=fake_audio):
                for expected in (5, 1, 3):
                    topics = [["Cricket", 3], ["Travel", 2]] if expected == 5 else [["Travel", expected]]
                    events = list(app.run(None, topics, 3,
                                          'Male', 45, 50, False, 1234, 'test',
                                          progress=lambda *args, **kwargs: None))
                    path = events[-1][3]['value']
                    self.assertTrue(events[-1][3]['interactive'])
                    self.assertEqual(len(events[-1][2]), expected)
                    with zipfile.ZipFile(path) as archive:
                        self.assertIsNone(archive.testzip())
                        self.assertEqual(sum(n.endswith('.wav') for n in archive.namelist()), expected)
                        self.assertTrue({'metadata.csv', 'segments.csv', 'README.md'} <= set(archive.namelist()))
                    self.assertEqual(sum(bool(e[3].get('value')) for e in events), expected + 1)
                    self.assertTrue(events[0][2].empty)
                    self.assertIsNone(events[0][3]['value'])
                    with zipfile.ZipFile(path) as archive:
                        per_note = [n for n in archive.namelist() if n.startswith('metadata/') and n.endswith('.csv')]
                        self.assertEqual(len(per_note), expected)
                        for name in per_note:
                            rows = list(csv.DictReader(archive.read(name).decode('utf-8-sig').splitlines()))
                            self.assertEqual(len(rows), 1)
                            self.assertGreater(float(rows[0]['duration_sec']), 45)
                            self.assertLessEqual(float(rows[0]['duration_sec']), 50)
                self.assertEqual(len(set(round(d, 1) for d in durations[:5])), 5)
                self.assertEqual(len(app.saved_datasets()), 3)
                self.assertTrue(app.download_saved(app.saved_datasets()[0])['interactive'])
                for name in ('../escape', '.', '', 'missing'):
                    with self.assertRaises(app.gr.Error):
                        app.archive_dataset(name)

    def test_multiple_voice_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = []
            for i in range(3):
                path = root / f"source_{i}.wav"
                app.sf.write(path, np.full(24000 * 10, (i + 1) / 10), 24000)
                sources.append(str(path))
            with patch.object(app, 'VOICES_DIR', root), \
                 patch.object(app, 'transcribe_reference', return_value='یہ میری آواز ہے۔'):
                app.save_voice(sources)
                wav, txt = app.voice_files()
                audio, sr = app.sf.read(wav)
                self.assertEqual(sr, 24000)
                self.assertLessEqual(len(audio) / sr, 20)
                for level in (.1, .2, .3):
                    self.assertGreater(np.sum(np.isclose(audio, level, atol=.001)), 24000)
                self.assertIn('3 recording(s)', app.current_voice_status())
                before = wav.read_bytes()
                with self.assertRaises(app.gr.Error):
                    app.save_voice(sources * 2)
                with patch.object(app, 'transcribe_reference', return_value=''):
                    with self.assertRaises(app.gr.Error):
                        app.save_voice(sources)
                self.assertEqual(wav.read_bytes(), before)

    def test_duration_and_timeline(self):
        sample_rate = 16000
        for seconds in (40, 44.99, 45, 47.5, 50, 50.01, 55):
            samples = np.arange(round(seconds * sample_rate))
            audio = np.sin(2 * np.pi * 220 * samples / sample_rate).astype(np.float32)
            fitted, spans = app.fit_duration(audio, [(0, seconds / 2), (seconds / 2, seconds)], sample_rate)
            self.assertGreaterEqual(len(fitted) / sample_rate, 45)
            self.assertLessEqual(len(fitted) / sample_rate, 50)
            self.assertAlmostEqual(spans[-1][1], len(fitted) / sample_rate, places=4)
            self.assertEqual(spans[0][1], spans[1][0])
            if 45 < seconds <= 50:
                self.assertIs(fitted, audio)
        for target in (46, 48, 49):
            audio = np.zeros(47 * sample_rate, dtype=np.float32)
            fitted, spans = app.fit_duration(audio, [(0, 47)], sample_rate, target_sec=target)
            self.assertAlmostEqual(len(fitted) / sample_rate, target, places=4)
            self.assertAlmostEqual(spans[0][1], target, places=4)
        for seconds in (0, 10, 70):
            with self.assertRaises(ValueError):
                app.fit_duration(np.zeros(seconds * sample_rate), [], sample_rate)



if __name__ == '__main__':
    unittest.main()
