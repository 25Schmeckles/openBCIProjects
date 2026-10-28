# openBCIProjects

Four bipolar sEMG channels on an OpenBCI Cyton are recorded alongside gamepad input.
The near-term goal is a working classifier. 
The end goal is to implement the trained decision function as a physical circuit. (LDA weights as conductances in a resistor crossbar; fed by analog feature extraction). 
Power consumption and computation minimization using biomemetic/neuromorphic inspired techniques.

## Scripts

**`scripts/read_raw.py`** — hardware check. Streams the Cyton for N seconds and prints per-channel statistics, flagging railed or flat channels.

```
python read_raw.py --port /dev/ttyUSB0 --seconds 10 --out data/check.csv
```

**`scripts/record_session.py`** — data collection. Streams EMG while polling a gamepad at 120 Hz, stamping both against the same wall clock so they can be aligned offline.

```
python record_session.py --port /dev/ttyUSB0 --seconds 120 --out data/run_001
```

**`scripts/live_loop.py`** — real-time path. Runs a decoder into a Linux uinput virtual gamepad. Decoders are pluggable; `passthrough` and `envelope` need no trained model and exist to validate latency and plumbing.

```
python live_loop.py --decoder envelope --port /dev/ttyUSB0
```

**`Trial_Prompter.html`** — browser cue display for recording sessions. Shows the action to perform, a lead-in countdown, and the upcoming queue; logs cue timestamps as CSV.

## Data format

`<run>_emg.csv` — `timestamp, ch1..ch8`. Unix epoch seconds, values in µV. Channels 1–4 are in use; the Cyton reports all eight.

`<run>_control.csv` — `timestamp, axis0..axisN, btn0..btnM`. Same clock as the EMG file.

Electrode placement is a circumferential montage over the forearm — four bipolar pairs spanning the flexor and extensor compartments.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`live_loop.py` needs uinput access for the virtual gamepad; setup steps are in the comments at the top of that file.
