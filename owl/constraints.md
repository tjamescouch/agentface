# Constraints

## package boundary

- agentface and visage are separate packages with separate repositories, dependencies, and release cycles.
- the sole interface between them is the MocapFrame JSON format.
- agentface has no rendering dependencies.
- either package can be replaced independently as long as the mocap contract is honored.

## runtime

- Python 3.10+.
- NumPy for vector math.
- ONNX Runtime or PyTorch for the expression net (CPU only, no GPU requirement).
- the full pipeline (parse + sentiment + merge + net + idle + emit) must complete in under 10ms per frame at 30 FPS.
- the expression net must infer in under 1ms on CPU.
- background threads for token stream input only; the processing pipeline is single-threaded.
- thread-safe queues for all cross-thread communication.

## marker protocol

- markers use the format `<af:EXPRESSION>` or `<af:EXPRESSION:INTENSITY>`.
- the `af` prefix is reserved — no collision with HTML or other markup.
- markers are always stripped before text reaches the user. agentface guarantees clean output.
- unknown expression names in markers are silently dropped.
- intensity is clamped to [0.0, 1.0]. missing intensity defaults to 1.0.
- markers embedded mid-word are still extracted: `hel<af:happy>lo` → `hello` + marker.

## sentiment

- the keyword lexicon is the baseline. it works without any model cooperation.
- recency weighting ensures the signal tracks the latest tokens, not ancient context.
- decay toward neutral is mandatory. no expression persists without reinforcement.
- the talking flag is based on token arrival timing, not content analysis.

## expression net

- the net is a small MLP: input ~10 floats, output 18 floats, 2-3 hidden layers of 64-128 units.
- until a trained net is available, a hand-crafted linear mapping serves as the fallback.
- the net and the fallback must produce output in the same MocapFrame format.
- the net is loaded from a file at startup; it is not trained at runtime.

## mocap format

- MocapFrame is a JSON object with `t` (timestamp float) and `pts` (dict of named floats).
- 18 control points in v1 (see product.md for the full list).
- all values are deltas from neutral rest pose unless otherwise noted.
- the format is extensible: new points can be added. consumers ignore points they don't recognize.
- frames are self-contained — no delta encoding, no dependency on previous frames.
- transport options: stdout JSON lines (one frame per line), WebSocket (JSON messages), or direct function call.

## idle behaviors

- breathing, blinking, eye drift, and talking oscillation are applied as post-processing on the expression net output.
- they are always active. they modulate the mocap output, never replace it.
- blink timing is randomized within [2s, 6s]. blink duration is 150ms.
- breathing oscillates face_scale at ~1.5 Hz, amplitude ±0.008.
- eye drift uses slow sine waves at different frequencies per eye for natural asymmetry.
- talking oscillation modulates mouth_open at ~5 Hz when the talking flag is set.
