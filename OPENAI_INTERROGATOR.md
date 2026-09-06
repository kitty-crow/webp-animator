# OpenAI Interrogator

The OpenAI Interrogator is an optional paid semantic interpolation layer for WebP Animator. The existing Python alignment, RIFE interpolation and WebP encoder remain usable without OpenAI.

OpenAI interpolation is intentionally placed before RIFE in the UI and workflow. The intended hybrid path is:

```text
source animation
  -> OpenAI semantic/key frames where needed
  -> insert accepted frames into the sequence
  -> local RIFE dense smoothing
  -> WebP encoding/optimisation
```

## Architecture

```text
browser
  -> app_openai.py (Python, authoritative job owner)
     -> durable ephemeral single-gap state in .openai-jobs/
     -> durable multi-gap batch state in .openai-batches/
     -> localhost OpenAI bridge
        -> vendored openai-schema for planner/auditor structured output
        -> OpenAI Images edit endpoint for one generated frame
  -> accepted semantic frames are inserted into the browser animation
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

Per-gap generation modes:

- **Single**: generate one missing frame.
- **Fixed**: generate N evenly spaced missing states. Targets are generated centre-first so later calls use the nearest accepted anchors.
- **Auto**: recursively inspect the largest remaining temporal interval and stop when the planner says another frame would not materially improve continuity or the configured benefit threshold/max-frame limit is reached.

Interpolation scope is independent of the per-gap mode:

- **One selected gap**: interpolate between any two selected anchors, including an explicitly incomplete gap.
- **Selected range**: run the chosen mode between every adjacent pair from the selected first frame through the selected last frame.
- **Whole animation**: run the chosen mode between every adjacent pair in the current ordered animation.

A looping whole-animation job can additionally include the final-frame → first-frame closure gap. Multi-gap jobs are executed sequentially behind one recoverable batch Job ID rather than launching every paid gap at once.

## Whole-animation context

The **Give the full animation to the planner and auditor as sequence context** option is explicit and enabled by default.

When enabled, every known frame is represented in the labelled sequence timeline supplied to planning and auditing. Immediate temporal anchors are also supplied separately at full detail. This lets the semantic stages reason about direction, cadence, invariants, incomplete sequences and loop phase while the image generator remains focused on the immediate pair and the planner's structured instructions.

This context costs additional image-input tokens, so it can be disabled for simple isolated pair interpolation.

## Image dimensions

The animation's original canvas remains authoritative. If the image-generation endpoint requires an API output size different from the source canvas, the bridge chooses a nearby valid generation size while preserving orientation and aspect as closely as possible. The Python job worker then restores the candidate to the exact animation canvas before auditing/insertion.

In particular, dimensions that are not divisible by 16 are normalised before the paid image-generation request rather than being sent to the API and rejected.

## Cost control

The UI estimates the cost before starting and the server estimates each next paid attempt before it is made. Range/whole-animation estimates include the number of selected gaps. The estimate includes planning, image generation/editing and auditing. It is explicitly approximate because image token usage depends on the actual API request and model processing.

Each completed attempt stores observed API usage when available and replaces the estimate with an actual calculated cost. Jobs support:

- automatic retry limit per target frame;
- maximum generated frames per gap in Auto mode;
- minimum Auto-mode benefit score;
- maximum total job spend in USD.

If the next estimated attempt would exceed the configured budget, the job enters `budget_wait` instead of spending more.

## Background and recovery

Once `/openai/interpolate` returns a Job ID, the browser is not part of execution. Source frames are staged server-side so closing Safari, changing tabs, locking the phone or losing the client connection does not cancel the worker.

Single-gap jobs live under `.openai-jobs/<job-id>/`. Range and whole-animation jobs have a parent manifest under `.openai-batches/<job-id>/` and use ordinary child jobs internally for each temporal gap.

The browser keeps recent Job IDs in localStorage and automatically reconnects. A Job ID can also be pasted manually into **Restore job**.

Job manifests and all attempts are written atomically. If the Python server itself restarts during an active job, the job is marked `interrupted` rather than discarded and can be resumed. Accepted frames and rejected-attempt evidence remain available.

Default cleanup policy:

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

A multi-gap parent additionally tracks which adjacent gap is currently running and aggregates accepted frames, attempts, spend and progress from its child jobs.

## Hybrid workflow

When an OpenAI job completes, **Insert accepted frames into animation** fetches the accepted PNGs back into the browser's ordered frame list at their correct gaps. For range and whole-animation jobs, insertion is performed from later source gaps backwards so earlier source indexes are not shifted while applying results.

The RIFE controls below can then create inexpensive dense in-betweens around the semantically corrected OpenAI keyframes before the normal lossless WebP optimiser runs.

This intentionally keeps OpenAI optional: the tool still works as a local/free RIFE animator when no API key is configured.
