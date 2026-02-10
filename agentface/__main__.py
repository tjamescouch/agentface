"""CLI entry point: reads stdin line by line, emits MocapFrame JSON to stdout."""

import sys
import time
import argparse

from agentface.pipeline import AgentFacePipeline


def main():
    parser = argparse.ArgumentParser(description="agentface — text → mocap frames")
    parser.add_argument("--fps", type=int, default=30, help="Target frames per second")
    args = parser.parse_args()

    pipeline = AgentFacePipeline(fps=args.fps)
    last_time = time.monotonic()

    # Also emit frames during silence (idle behaviors need ticking)
    # Read stdin non-blocking style: process available lines, then tick
    import select

    try:
        while True:
            now = time.monotonic()
            dt = now - last_time
            last_time = now

            # Check for available input (non-blocking)
            if select.select([sys.stdin], [], [], 0.0)[0]:
                line = sys.stdin.readline()
                if not line:
                    break  # EOF
                pipeline.feed(line, time.time())

            # Always step (drives idle behaviors + decay)
            pipeline.step(dt, time.time())

            # Sleep to target FPS
            time.sleep(1.0 / args.fps)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
