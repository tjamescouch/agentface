# Components

## marker parser

Extracts hidden expression markers from the LLM token stream and produces clean text.

### state

- **pending buffer**: incomplete tag accumulator for tokens that arrive mid-tag.

### capabilities

- scans text for `<af:expression>` and `<af:expression:intensity>` patterns.
- extracts expression name and intensity from each tag.
- strips all matched tags from the text, producing clean output for user display.
- handles tags split across multiple text chunks (partial tag buffering).
- validates expression names against the known vocabulary; unknown names are dropped.
- clamps intensity to [0.0, 1.0]; missing intensity defaults to 1.0.

### interfaces

- **parse(text) -> (clean_text, markers[])**: accepts raw text, returns stripped text and a list of extracted markers.

### invariants

- output text never contains `<af:` sequences.
- marker extraction is lossless for non-marker text — no characters are added or removed except the tags themselves.

---

## sentiment brain

Converts a stream of text into a continuous emotion signal. Operates on the clean text (after marker stripping).

### state

- **emotion**: valence [-1, +1], arousal [0, 1], talking flag.
- **text window**: bounded sliding window of recent tokens (200 token limit).
- **silence timer**: elapsed time since the last token.

### capabilities

- accepts text chunks with timestamps.
- scores text against keyword lexicons (positive, negative, thinking, surprise).
- applies exponential recency weighting — recent words dominate the signal.
- blends new signals into running emotion proportional to signal strength.
- decays toward neutral during silence, faster after prolonged silence.
- detects talking state from token recency.

### interfaces

- **feed(text, timestamp)**: ingests a chunk of clean text.
- **step(dt, timestamp)**: advances emotion state by one time step.
- **emotion**: current valence, arousal, and talking state.

### invariants

- valence clamped to [-1, +1], arousal clamped to [0, 1].
- emotion always decays toward neutral in silence; never drifts unbounded.
- the talking flag reflects token recency, not sentiment content.

---

## signal merger

Combines marker signals and sentiment signals into a single expression vector.

### state

- **active markers**: list of currently active marker-driven expressions with intensity and decay timer.
- **expression vector**: array of weights, one per expression class (happy, sad, thinking, surprised, confused, angry, neutral, talking).
- **arousal**: current arousal level.
- **talking**: current talking flag.

### capabilities

- accepts marker events (expression, intensity) and pushes them as high-priority layers.
- accepts sentiment updates (valence, arousal, talking) and maps them to expression weights.
- markers override sentiment for the dimensions they specify; sentiment fills uncovered dimensions.
- all active signals decay toward neutral over time (configurable decay rates).
- produces a single expression vector each frame by blending all active signals.

### interfaces

- **push_marker(expression, intensity)**: injects a marker-driven expression signal.
- **push_sentiment(valence, arousal, talking)**: updates the ambient sentiment signal.
- **step(dt) -> expression_vector**: advances decay and returns the current blended expression vector.

### invariants

- expression weights are non-negative and sum-normalized when multiple expressions are active.
- markers take priority: if a marker sets "happy" to 0.8, sentiment cannot reduce it below 0.8 while the marker is active.
- all signals decay; nothing persists forever without reinforcement.

---

## expression net

Small neural network that maps expression vectors to mocap points.

### state

- **model weights**: loaded from ONNX file or PyTorch checkpoint.
- **idle state**: internal timers for breathing, blink scheduling, eye drift.

### capabilities

- accepts an expression vector (7-10 floats) and produces a mocap point array (18 floats).
- runs inference in <1ms on CPU.
- learns natural co-activations from training data (e.g., smiling narrows eyes).
- applies idle overlay post-inference: breathing oscillation, periodic blinks, eye drift, talking mouth oscillation.
- can be replaced with a hand-crafted mapping for bootstrapping before a trained model is available.

### interfaces

- **forward(expression_vector, dt) -> MocapFrame**: produces one frame of mocap output.
- **load(model_path)**: loads model weights.

### invariants

- output mocap points are always within valid ranges for each control point.
- idle behaviors are always active — they modulate the net output, never replace it.
- inference time is bounded; the net never blocks the frame pipeline.

---

## frame emitter

Packages mocap output and delivers it to consumers at a steady frame rate.

### state

- **target FPS**: default 30.
- **frame clock**: timing accumulator for steady emission.
- **transport**: configured output channel.

### capabilities

- wraps mocap point arrays into timestamped MocapFrame objects.
- emits frames at the target rate, interpolating if the processing pipeline runs faster or slower.
- supports multiple transport modes: stdout JSON lines, WebSocket, callback function, or shared memory buffer.
- drops frames rather than buffering if the consumer falls behind.

### interfaces

- **emit(mocap_points)**: accepts raw points, packages and delivers a MocapFrame.
- **configure(fps, transport)**: sets frame rate and output channel.

### invariants

- frames are emitted at a steady rate; jitter is smoothed by the timing accumulator.
- each emitted frame is self-contained — no dependency on previous frames.
- transport failures are logged but do not crash the pipeline.
