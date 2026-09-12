import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DispatchResourceTests(unittest.TestCase):
    def test_invalid_gpu_counts_have_no_filesystem_or_provider_side_effects(self):
        for mode in ([], ["--stage-only"], ["--submit"]):
            for count in ("-1", "0", "9", "2147483647"):
                with (
                    self.subTest(mode=mode, count=count),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    root = Path(temporary)
                    lane = root / "lane"
                    lane.mkdir()
                    (lane / "run.py").write_text("raise RuntimeError('must not run')\n")
                    config = root / "config.json"
                    config.write_text('{"seed": 17}\n')
                    binary = root / "bin"
                    binary.mkdir()
                    kubectl = binary / "kubectl"
                    kubectl.write_text(
                        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$DISPATCH_TEST_CALLS"\nexit 91\n'
                    )
                    kubectl.chmod(0o755)
                    before = {
                        path.relative_to(root): path.read_bytes()
                        if path.is_file()
                        else None
                        for path in root.rglob("*")
                    }
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).with_name("dispatch.py")),
                            "--lane-dir",
                            str(lane),
                            "--config",
                            str(config),
                            "--id",
                            "resource-probe",
                            "--gpus",
                            count,
                            "--control-pod",
                            "must-not-contact",
                            *mode,
                        ],
                        cwd=root,
                        env={
                            **os.environ,
                            "PATH": str(binary)
                            + os.pathsep
                            + os.environ.get("PATH", ""),
                            "DISPATCH_TEST_CALLS": str(root / "provider-calls"),
                            "PYTHONDONTWRITEBYTECODE": "1",
                        },
                        capture_output=True,
                        timeout=10,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr.decode())
                    self.assertIn(b"INVALID_GPU_COUNT", result.stderr)
                    self.assertEqual(result.stdout, b"")
                    self.assertEqual(
                        before,
                        {
                            path.relative_to(root): path.read_bytes()
                            if path.is_file()
                            else None
                            for path in root.rglob("*")
                        },
                    )

    def test_valid_gpu_boundaries_create_only_dry_run_manifests(self):
        for count in (1, 8):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                lane = root / "lane"
                lane.mkdir()
                (lane / "run.py").write_text("raise RuntimeError('dry run only')\n")
                config = root / "config.json"
                config.write_text('{"seed": 17}\n')
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("dispatch.py")),
                        "--lane-dir",
                        str(lane),
                        "--config",
                        str(config),
                        "--id",
                        "resource-probe",
                        "--gpus",
                        str(count),
                        "--control-pod",
                        "must-not-contact",
                    ],
                    cwd=root,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                response = json.loads(result.stdout)
                self.assertEqual(response["source_files"], 1)
                manifest = json.loads(Path(response["manifest"]).read_text())
                self.assertEqual(manifest["gpus"], count)
                self.assertEqual(manifest["config"], {"seed": 17})
                self.assertFalse((root / "runs").exists())


if __name__ == "__main__":
    unittest.main()
