# OpenAI Interrogator

The OpenAI Interrogator is an optional paid semantic interpolation layer for WebP Animator. The existing Python alignment, RIFE interpolation and WebP encoder remain usable without OpenAI.

## Architecture

```text
browser
  -> app_openai.py (Python, authoritative job owner)
     -> durable ephemeral job state in .openai-jobs/
     -> localhost OpenAI bridge
        -> vendored openai-schema for planner/auditor structured output
        -> OpenAI Images edit endpoint for one generated frame
  -> accepted semantic frames can be inserted into the browser animation
  -> existing RIFE can then add free dense smoothing
```

The bridge binds only to `127.0.0.1`. The OpenAI API key is never sent to the browser.

## Setup

Initialise/build the pinned `openai-schema` submodule:

```bash
git submodule update --init --recursive
python setup_openai.py
```

Create `.env` from `.env.example` and set:

```text
OPENAI_API_KEY=...
```

`.env` is ignored by Git.

Run the enhanced application:

```bash
python app_openai.py
```

It uses the same `0.0.0.0:18743` application port as `app.py`, so run one entry point or the other, not both.

The local OpenAI bridge defaults to `127.0.0.1:18744` and is started on demand by Python. It uses Bun when available and otherwise Node.js. The bridge itself has no OpenAI SDK dependency.

## Interpolation contract

The permanent planner prompt is generic. It does not assume a character, person, sprite, object or any particular subject. It receives temporal anchor frames, optional whole-animation context, sequence topology and optional user guidance. It refines the user's guidance into a structured interpolation plan.

Each paid image-generation call creates exactly one missing frame. The generated frame is then audited with another structured-output call. Rejected images, their plan, the audit diagnosis and later user feedback are retained as evidence for replanning. A rejected image never becomes a temporal anchor. An accepted image can become an anchor for later missing frames.

Supported modes:

- **Single**: generate one missing frame.
- **Fixed**: generate N evenly spaced missing states. Targets are generated centre-first so later calls use the nearest accepted anchors.
- **Auto**: recursively inspect the largest remaining temporal interval and stop when the planner says another frame would not materially improve continuity or the configured benefit threshold/max-frame limit is reached.

The job may be marked **open** or **looping**. Loop closure is represented as the temporal gap from the current final frame back to frame 1. Auto mode may add closure states until another state is judged not worthwhile, subject to hard limits.

## Whole-animation context

For sequence-aware planning, Python builds a labelled contact sheet from the currently known animation. Immediate anchor images are still supplied separately at full detail. This lets planning and auditing understand direction, cadence, invariants and loop phase without repeatedly sending every full-resolution frame in every request.

## Cost control

The UI estimates the cost before starting and the server estimates each next paid attempt before it is made. The estimate includes planning, image generation/editing and auditing. It is explicitly approximate because image token usage depends on the actual API request and model processing.

Each completed attempt stores observed API usage when available and replaces the estimate with an actual calculated cost. Jobs support:

- automatic retry limit per target frame;
- maximum generated frames in Auto mode;
- minimum Auto-mode benefit score;
- maximum total job spend in USD.

If the next estimated attempt would exceed the configured budget, the job enters `budget_wait` instead of spending more.

## Background and recovery

Once `/openai/interpolate` returns a Job ID, the browser is not part of execution. Source frames are staged into `.openai-jobs/<job-id>/` so closing Safari, changing tabs, locking the phone or losing the client connection does not cancel the worker.

The browser keeps recent Job IDs in localStorage and automatically reconnects. A Job ID can also be pasted manually into **Restore job**.

Job manifests and all attempts are written atomically. If the Python server itself restarts during an active job, that job is marked `interrupted` rather than discarded. It can be resumed from the UI; accepted frames and rejected-attempt evidence remain available.

Default cleanup policy in the manager:

- active jobs are never cleaned;
- completed/error jobs: 24 hours;
- review/budget/interrupted jobs: 7 days.

## Job states

```text
queued
  -> planning / replanning
  -> generating
  -> auditing
     -> accepted -> next target
     -> rejected -> automatic retry or needs_review
  -> done

budget_wait   -> resume after raising budget
interrupted   -> resume after server restart
needs_review  -> add user feedback and retry
```

## Hybrid workflow

When an OpenAI job completes, **Insert accepted frames into animation** fetches the accepted PNGs back into the browser's existing ordered frame list. From there the ordinary WebP Animator workflow is unchanged. You can select RIFE 2x/4x/8x to create inexpensive dense in-betweens around the semantically corrected OpenAI keyframes, then run the normal lossless WebP optimiser.

This intentionally keeps OpenAI optional: the tool still works as a local/free RIFE animator when no API key is configured.
