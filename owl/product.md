# agentface

Turns LLM token streams into facial mocap points. Extracts sentiment and hidden expression markers, blends them through a small neural network, and emits a steady stream of MocapFrame data.

agentface knows nothing about rendering. It produces motion data; a separate renderer (such as [visage](https://github.com/tjamescouch/visage)) turns that into pixels.

## Inputs

1. **Token stream** — Raw LLM output. Analyzed for sentiment via keyword lexicon scoring (valence, arousal, talking state). This is the ambient signal — always running, no model cooperation needed.

2. **Hidden markers** — Expression directives embedded in the token stream by the LLM. Format: `<af:expression>` or `<af:expression:intensity>`. Examples:
   - `<af:happy:0.8>` — trigger happy at 80% intensity
   - `<af:thinking>` — trigger thinking at default intensity (1.0)
   - `<af:neutral>` — explicitly return to neutral
   - Markers are stripped from text before user display. The LLM does not need to use them — sentiment analysis covers the baseline. Markers are for when the model wants explicit control.

3. **Explicit API calls** — Programmatic overrides for integration: `agentface.push("surprised", 0.6)`. Same effect as a marker but injected externally.

## Processing Pipeline

```
tokens ──▶ marker parser ──▶ markers[] + clean text
                                │            │
                                ▼            ▼
                          marker signals   sentiment brain
                          (expression,     (valence, arousal,
                           intensity)       talking flag)
                                │            │
                                └──────┬─────┘
                                       ▼
                                signal merger
                                (markers override sentiment,
                                 both decay toward neutral)
                                       │
                                       ▼
                                expression vector
                                (blended weights for
                                 each expression class)
                                       │
                                       ▼
                                 small net (MLP)
                                 expression vector ──▶ mocap points
                                       │
                                       ▼
                                 idle overlay
                                 (breathing, blink, drift)
                                       │
                                       ▼
                                  MocapFrame output
```

## Small Net

- Architecture: MLP (2-3 hidden layers, ~64-128 units) or single ONNX model
- Input: expression vector (7-10 floats: one weight per expression class + arousal + talking flag)
- Output: mocap point array (see Mocap Format below)
- Inference: <1ms on CPU. No GPU required at runtime.
- Training data: facial expression datasets mapped to mocap point targets. Can start with hand-crafted mapping and replace with learned net later.
- The net learns natural co-activation — e.g., a smile naturally narrows the eyes, surprise raises brows AND opens mouth. This replaces hand-coded expression parameter overrides.

## Output

- Stream of `MocapFrame` at 30 FPS (configurable)
- Transport: stdout JSON lines, WebSocket, callback, or shared memory
- Each frame is self-contained — no dependency on previous frames

## Hidden Markers — Detailed Spec

### Syntax

```
<af:EXPRESSION>           intensity defaults to 1.0
<af:EXPRESSION:INTENSITY>  intensity is a float 0.0 to 1.0
```

- EXPRESSION: one of the known expression names (happy, sad, thinking, surprised, confused, angry, neutral, talking)
- INTENSITY: optional float, clamped to [0.0, 1.0]
- Tags are case-insensitive
- Multiple tags can appear in one text chunk

### Stripping

agentface outputs clean text with all `<af:…>` tags removed. The caller is responsible for displaying this clean text to the user, not the raw token stream.

### Priority

When markers and sentiment conflict, markers win. A marker sets the expression at the specified intensity; sentiment fills in the gaps for dimensions the marker doesn't address. When no markers are active, sentiment drives everything.

## MocapFrame Format

The `MocapFrame` is the output contract. Any renderer that consumes this format is compatible.

```json
{
  "t": 1234567890.123,
  "pts": {
    "left_eye_open": 0.85,
    "right_eye_open": 0.85,
    "left_pupil_x": 0.02,
    "left_pupil_y": -0.01,
    "right_pupil_x": 0.02,
    "right_pupil_y": -0.01,
    "left_brow_height": 0.03,
    "left_brow_angle": 0.0,
    "right_brow_height": 0.03,
    "right_brow_angle": 0.0,
    "mouth_open": 0.0,
    "mouth_wide": 0.0,
    "mouth_smile": 0.1,
    "jaw_open": 0.0,
    "face_scale": 1.0,
    "head_pitch": 0.0,
    "head_yaw": 0.0,
    "head_roll": 0.0
  }
}
```

- **18 control points** — named floats, each normalized to a natural range
- All values are relative deltas from a neutral rest pose (0.0 = neutral for most, 1.0 for eye_open/face_scale)
- Points are inspired by ARKit blendshapes but simplified for 2D/2.5D faces
- The set is extensible — consumers ignore points they don't recognize

## Non-Goals

- agentface does NOT render anything
- No heavy ML models in the real-time path
- No audio input or lip-sync in v1
- No code-level dependency on any renderer
