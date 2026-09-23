#!/usr/bin/env python3
"""
HTTP transcription server.

    python serve.py --port 8000
    curl -X POST --data-binary @audio.wav http://127.0.0.1:8000/transcribe
"""

import os
import sys

from asr.server import main

if __name__ == "__main__":
    # The VAD loads models/whisper_vad.onnx by relative path, so the process must
    # run from the directory holding models/.
    if getattr(sys, "frozen", False):
        os.chdir(os.path.dirname(sys.executable))
    else:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
