#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python3 python_imu_segment_demo_student.py "$@"
