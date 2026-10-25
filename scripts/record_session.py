"""
Synchronized EMG + controller recording.

Streams EMG from an OpenBCI Cyton while polling a game controller, stamping
both against the same wall clock so they can be aligned offline.

Outputs two CSVs sharing a timebase:
    <out>_emg.csv         timestamp, ch1..ch8
    <out>_control.csv     timestamp, axis0..axisN, btn0..btnM

Sync check: at the start of each session, squeeze the electrode site hard
while slamming the trigger. That co-occurring spike in both streams lets you
measure your actual alignment offset instead of assuming it's zero.

Usage:
    python record_session.py --port /dev/ttyUSB0 --seconds 120 --out data/run_001
"""

import argparse
import os
import time

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

CONTROL_HZ = 120.0          # controller poll rate


def parse_args():
    p = argparse.ArgumentParser(description="Synchronized EMG + controller capture")
    p.add_argument("--port", help="Cyton dongle serial port (omit with --synthetic)")
    p.add_argument("--synthetic", action="store_true",
                   help="Use BrainFlow's synthetic board instead of real hardware. "
                        "Same API, fake data -- for dry runs of the pipeline.")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--out", required=True,
                   help="Output prefix, e.g. data/run_001")
    p.add_argument("--no-controller", action="store_true",
                   help="EMG only (useful for noise-floor runs)")
    a = p.parse_args()
    if not a.synthetic and not a.port:
        p.error("--port is required unless --synthetic is set")
    return a


def init_controller():
    import pygame
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No controller detected. Is it paired/plugged in?")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Controller: {js.get_name()} "
          f"({js.get_numaxes()} axes, {js.get_numbuttons()} buttons)")
    return pygame, js


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    pygame = js = None
    n_axes = n_btns = 0
    if not args.no_controller:
        pygame, js = init_controller()
        # capture these now -- pygame.quit() invalidates the joystick object
        n_axes = js.get_numaxes()
        n_btns = js.get_numbuttons()

    board_id = (BoardIds.SYNTHETIC_BOARD.value if args.synthetic
                else BoardIds.CYTON_BOARD.value)

    params = BrainFlowInputParams()
    if args.port:
        params.serial_port = args.port
    board = BoardShim(board_id, params)

    fs = BoardShim.get_sampling_rate(board_id)
    exg_rows = BoardShim.get_exg_channels(board_id)
    ts_row = BoardShim.get_timestamp_channel(board_id)

    board.prepare_session()
    board.start_stream()

    if args.synthetic:
        print("\n*** SYNTHETIC BOARD -- fake EMG, real controller ***")
    print(f"\nRecording {args.seconds:.0f}s at {fs} Hz.")
    print(">>> SYNC MARKER: tap the electrode site and mash the trigger NOW <<<\n")

    control_log = []
    t_start = time.time()
    next_poll = t_start

    try:
        while time.time() - t_start < args.seconds:
            now = time.time()
            if js is not None and now >= next_poll:
                pygame.event.pump()
                axes = [js.get_axis(i) for i in range(n_axes)]
                btns = [js.get_button(i) for i in range(n_btns)]
                control_log.append([now] + axes + btns)
                next_poll += 1.0 / CONTROL_HZ
                if next_poll < now:          # fell behind; resync
                    next_poll = now + 1.0 / CONTROL_HZ

            elapsed = now - t_start
            if int(elapsed) % 10 == 0 and abs(elapsed - int(elapsed)) < 0.002:
                print(f"  {int(elapsed):>4}s / {int(args.seconds)}s")

            time.sleep(0.001)

        data = board.get_board_data()
    finally:
        board.stop_stream()
        board.release_session()
        if pygame is not None:
            pygame.quit()

    # --- EMG ---
    emg_path = f"{args.out}_emg.csv"
    emg_rows = np.vstack([data[ts_row]] + [data[r] for r in exg_rows]).T
    emg_header = "timestamp," + ",".join(f"ch{i}" for i in range(1, len(exg_rows) + 1))
    np.savetxt(emg_path, emg_rows, delimiter=",", header=emg_header, comments="")
    print(f"\nEMG:        {emg_rows.shape[0]:>7} samples -> {emg_path}")

    # --- Controller ---
    if control_log:
        ctl_path = f"{args.out}_control.csv"
        ctl = np.array(control_log)
        ctl_header = ("timestamp,"
                      + ",".join(f"axis{i}" for i in range(n_axes))
                      + ("," if n_btns else "")
                      + ",".join(f"btn{i}" for i in range(n_btns)))
        np.savetxt(ctl_path, ctl, delimiter=",", header=ctl_header, comments="")
        print(f"Controller: {ctl.shape[0]:>7} samples -> {ctl_path}")

        # effective poll rate -- if this is far below CONTROL_HZ, the loop
        # is starved and your labels are coarser than you think
        dt = np.diff(ctl[:, 0])
        print(f"            effective rate {1.0/np.median(dt):.1f} Hz "
              f"(jitter p95 {np.percentile(dt, 95)*1000:.1f} ms)")

    # --- Coverage check ---
    if control_log:
        emg_span = (emg_rows[0, 0], emg_rows[-1, 0])
        ctl_span = (ctl[0, 0], ctl[-1, 0])
        overlap = min(emg_span[1], ctl_span[1]) - max(emg_span[0], ctl_span[0])
        print(f"\nOverlapping window: {overlap:.1f}s "
              f"(EMG starts {ctl_span[0]-emg_span[0]:+.2f}s vs controller)")


if __name__ == "__main__":
    main()
