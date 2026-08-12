"""Smoke tests for scene-change frame extraction."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from frames import extract_scene_change, extract  # noqa: E402


def _make_test_video(out: Path, seconds: int = 6) -> Path:
    """Generate a synthetic test video with 3 distinct scenes (color changes)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"color=c=red:size=320x240:duration=2",
        "-f", "lavfi", "-i", f"color=c=green:size=320x240:duration=2",
        "-f", "lavfi", "-i", f"color=c=blue:size=320x240:duration=2",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
        "-map", "[v]",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def _make_clustered_video(out: Path) -> Path:
    """3 fast cuts up front, then one long still tail — the screencast shape."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=red:size=320x240:duration=0.5",
        "-f", "lavfi", "-i", "color=c=green:size=320x240:duration=0.5",
        "-f", "lavfi", "-i", "color=c=blue:size=320x240:duration=0.5",
        "-f", "lavfi", "-i", "color=c=black:size=320x240:duration=9",
        "-filter_complex", "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0[v]",
        "-map", "[v]",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def _make_test_video_n_scenes(out: Path, n: int = 6, each: int = 2) -> Path:
    """n evenly-spaced scenes of `each` seconds — well-distributed cuts."""
    out.parent.mkdir(parents=True, exist_ok=True)
    colors = ["red", "green", "blue", "yellow", "magenta", "cyan", "white", "gray"]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for i in range(n):
        cmd += ["-f", "lavfi", "-i",
                f"color=c={colors[i % len(colors)]}:size=320x240:duration={each}"]
    streams = "".join(f"[{i}:v]" for i in range(n))
    cmd += [
        "-filter_complex", f"{streams}concat=n={n}:v=1:a=0[v]",
        "-map", "[v]",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


class TestSceneChange(unittest.TestCase):

    def setUp(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not available")
        self.tmp = Path(tempfile.mkdtemp(prefix="watch-test-"))
        self.video = _make_test_video(self.tmp / "input.mp4", seconds=6)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_detects_scene_boundaries(self):
        out_dir = self.tmp / "frames"
        frames = extract_scene_change(
            str(self.video), out_dir,
            scene_threshold=0.3, resolution=128, max_frames=10,
        )
        self.assertGreaterEqual(len(frames), 2, "expected >=2 scene frames")
        self.assertLessEqual(len(frames), 10, "respect max_frames cap")
        for f in frames:
            self.assertTrue(Path(f["path"]).exists())
            self.assertIn("timestamp_seconds", f)

    def test_falls_back_to_uniform_when_no_scenes(self):
        static = self.tmp / "static.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=red:size=320x240:duration=4",
            "-pix_fmt", "yuv420p",
            str(static),
        ], check=True, capture_output=True)

        out_dir = self.tmp / "static_frames"
        frames = extract_scene_change(
            str(static), out_dir,
            scene_threshold=0.3, resolution=128,
            max_frames=10, uniform_fallback_min=5,
        )
        self.assertGreaterEqual(len(frames), 5, "fallback should produce >=5 frames")

    def test_falls_back_when_scenes_are_clustered(self):
        """Enough frames to clear the count bar, but bunched into the opening.

        The screencast failure mode: hard cuts between UI states early, then
        minutes of near-still screen. Count says "fine", coverage says otherwise.
        """
        clustered = _make_clustered_video(self.tmp / "clustered.mp4")

        out_dir = self.tmp / "clustered_frames"
        frames = extract_scene_change(
            str(clustered), out_dir,
            scene_threshold=0.3, resolution=128,
            max_frames=20, uniform_fallback_min=3,
        )

        self.assertGreater(len(frames), 4, "should resample, not keep the cluster")
        self.assertNotIn(
            "source", frames[0],
            "uniform fallback frames carry no 'source' key; scene frames do",
        )
        # The point of falling back: no blind spot wider than a quarter of the video.
        times = [f["timestamp_seconds"] for f in frames]
        marks = [0.0] + sorted(times) + [10.5]
        largest_gap = max(b - a for a, b in zip(marks, marks[1:]))
        self.assertLess(largest_gap, 0.25 * 10.5)

    def test_keeps_scene_frames_when_well_distributed(self):
        """Regression guard: evenly-cut video must NOT trip the coverage check."""
        # Dense enough that ffmpeg missing the odd boundary still leaves the
        # widest hole well under the threshold (~0.17 of the range).
        even = _make_test_video_n_scenes(self.tmp / "even.mp4", n=12, each=1)

        out_dir = self.tmp / "even_frames"
        frames = extract_scene_change(
            str(even), out_dir,
            scene_threshold=0.3, resolution=128,
            max_frames=20, uniform_fallback_min=3,
        )

        self.assertEqual(
            frames[0].get("source"), "scene-change",
            "evenly-distributed scenes should survive both fallbacks",
        )


if __name__ == "__main__":
    unittest.main()
