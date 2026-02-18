# agentface

**LLM token stream → facial mocap data. Extracts sentiment and expression markers, blends them, emits real-time animation frames.**

Turns raw LLM output into facial motion capture coordinates. Analyzes token sentiment, detects embedded expression markers, and produces MocapFrame data for a renderer (like [visage](https://github.com/tjamescouch/visage)) to display.

## Overview

Agentface is the **data layer**. It produces motion data, not pixels. It knows nothing about rendering — that's the renderer's job.

```
LLM token stream
    ↓
[marker extraction + sentiment analysis]
    ↓
[expression blending + decay]
    ↓
MocapFrame stream (quaternions, blend weights)
    ↓
Renderer (visage) → pixels on screen
```

## Inputs

### 1. Token Stream

Raw LLM output from a provider (Claude, GPT, etc.). Agentface continuously:
- Scans for embedded expression markers (see below)
- Analyzes sentiment via lexicon scoring (valence, arousal, talking state)
- Produces a steady stream of motion frames

The sentiment analysis is **always on** and requires no LLM cooperation. It's the baseline.

### 2. Expression Markers

The LLM can optionally embed expression directives in its output:

```
<af:happy:0.8>      — trigger "happy" at 80% intensity
<af:thinking>       — trigger "thinking" at default intensity (1.0)
<af:surprised:0.6>
<af:neutral>        — explicitly return to neutral
```

Markers are **stripped from output** before the user sees it. The LLM doesn't need to use them — sentiment is the fallback. Markers are for when the model wants explicit, fine-grained control.

Supported expressions: `happy`, `sad`, `angry`, `surprised`, `thinking`, `calm`, `neutral` (and extensible).

### 3. Programmatic API

Override expressions at runtime:

```javascript
agentface.push('happy', 0.8);
agentface.push('thinking');
agentface.reset();  // Return to neutral
```

## Processing Pipeline

```
Token Stream
    ↓
┌───────────────────────────────────────┐
│ Marker Extraction                     │
│ - Parse <af:*> markers               │
│ - Strip from text                    │
│ - Queue signals                      │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Sentiment Analysis                    │
│ - Keyword lexicon (valence/arousal)  │
│ - Detect talking (punctuation, ...)  │
│ - Smooth scores (moving avg)         │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Signal Merger                         │
│ - Marker signals override sentiment  │
│ - Both decay toward neutral          │
│ - Blended weight vector              │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Neural Blend Layer                    │
│ - Small neural network               │
│ - Learns smooth transitions          │
│ - Prevents jittery expressions       │
└───────────────────────────────────────┘
    ↓
MocapFrame
  - head: { q: Quaternion, pos: [x, y, z] }
  - jaw: { rotation: Quaternion, openness: 0-1 }
  - leftEye: { gaze: Quaternion, blink: 0-1 }
  - rightEye: { gaze: Quaternion, blink: 0-1 }
  - blends: { happy: 0.2, sad: 0.1, ... }
  - timestamp: ms
```

## Output Format

```typescript
interface MocapFrame {
  timestamp: number;            // Unix milliseconds
  head: { q: [x, y, z, w]; };  // Head rotation (quaternion)
  jaw: {
    rotation: [x, y, z, w];
    openness: 0;               // 0 = closed, 1 = fully open
  };
  leftEye: {
    gaze: [x, y, z, w];       // Eye direction
    blink: 0;                  // 0 = open, 1 = closed
  };
  rightEye: {
    gaze: [x, y, z, w];
    blink: 0;
  };
  blends: {
    happy: 0.2;
    sad: 0.1;
    angry: 0;
    surprised: 0;
    thinking: 0.5;
    calm: 0;
    neutral: 0.2;
  };
}
```

## API

### `new Agentface(config?)`

Create an agentface instance.

**Config:**
- `decayRate` — How fast expressions fade (0-1, default 0.1)
- `sentimentSmoothing` — Moving average window (default 5 frames)
- `blendNetwork` — Optional pre-trained blend network weights

### `agentface.processTokens(text: string): MocapFrame[]`

Process a chunk of tokens. Returns array of motion frames since last call.

### `agentface.push(expression: string, intensity?: number)`

Override with a specific expression (programmatic control).

### `agentface.reset()`

Return to neutral.

### `agentface.onFrame(callback: (frame: MocapFrame) => void)`

Register a callback for each frame (streaming mode).

## Integration with Visage

Visage consumes the MocapFrame stream:

```javascript
const face = new Agentface();
const visage = new Visage(renderer);

// Stream frames to visage
face.onFrame(frame => {
  visage.render(frame);
});

// Pipe LLM tokens
llm.onToken(token => {
  face.processTokens(token);
});
```

## Design Philosophy

1. **Separation of concerns** — Motion data production (agentface) vs rendering (visage)
2. **No rendering** — Agentface doesn't know about WebGL, Three.js, or pixels
3. **Real-time** — Processes tokens as they arrive, no batching required
4. **Graceful degradation** — If markers are absent, sentiment analysis alone drives expressions
5. **Extensible** — Add new expressions, change lexicon, retrain blend network

## Performance

- **Token processing** — O(token length) for marker extraction and sentiment scoring
- **Frame generation** — One frame per processing call (configurable cadence)
- **Memory** — Minimal state (current blend weights, decay timers)
- **Latency** — <5ms per token chunk on modern hardware

## Security

- Expression markers are local (no external API)
- Sentiment lexicon is static (no model inference)
- Blend network is optional and local

## See Also

- [visage](https://github.com/tjamescouch/visage) — Renderer that consumes MocapFrame output
- [visage3d](https://github.com/tjamescouch/visage3d) — 3D face toolkit
- [agent-face](https://github.com/tjamescouch/agent-face) — Similar sentiment pipeline (different architecture)
- [Product spec](owl/product.md) — Complete technical specification

## License

MIT
