export interface DdimStepParameters {
  readonly alpha: number;
  readonly alphaPrevious: number;
  readonly eta: number;
}

export interface DdimLoopStep {
  readonly trainingTimestep: number;
  readonly alpha: number;
  readonly alphaPrevious: number;
}

export type NoisePredictor = (
  sample: Float32Array,
  trainingTimestep: number,
  stepIndex: number,
) => Promise<Float32Array>;

export type NoiseSource = (length: number, stepIndex: number) => Float32Array;

function requireUnitInterval(value: number, label: string): void {
  if (!Number.isFinite(value) || value <= 0 || value > 1) {
    throw new Error(`${label} must be in (0, 1].`);
  }
}

export function makeUniformDdimSchedule(
  alphasCumprod: readonly number[],
  inferenceSteps: number,
): readonly DdimLoopStep[] {
  if (alphasCumprod.length < 2) throw new Error('DDIM requires at least two training alpha values.');
  if (!Number.isInteger(inferenceSteps) || inferenceSteps < 1) {
    throw new Error('DDIM inferenceSteps must be a positive integer.');
  }
  if (inferenceSteps > alphasCumprod.length) {
    throw new Error('DDIM inferenceSteps cannot exceed the training schedule length.');
  }

  const indices: number[] = [];
  if (inferenceSteps === 1) {
    indices.push(alphasCumprod.length - 1);
  } else {
    for (let index = 0; index < inferenceSteps; index += 1) {
      const fraction = index / (inferenceSteps - 1);
      indices.push(Math.round(fraction * (alphasCumprod.length - 1)));
    }
  }

  const deduplicated = [...new Set(indices)];
  if (deduplicated.length !== inferenceSteps) {
    throw new Error('DDIM inference schedule collapsed duplicate timesteps.');
  }

  const descending = deduplicated.reverse();
  return descending.map((trainingTimestep, index) => {
    const alpha = alphasCumprod[trainingTimestep];
    if (alpha === undefined) throw new Error('DDIM alpha lookup failed.');
    const nextTrainingTimestep = descending[index + 1];
    const alphaPrevious = nextTrainingTimestep === undefined
      ? 1
      : alphasCumprod[nextTrainingTimestep];
    if (alphaPrevious === undefined) throw new Error('DDIM previous-alpha lookup failed.');
    requireUnitInterval(alpha, 'DDIM alpha');
    requireUnitInterval(alphaPrevious, 'DDIM previous alpha');
    return { trainingTimestep, alpha, alphaPrevious };
  });
}

export function ddimStep(
  sample: Float32Array,
  predictedNoise: Float32Array,
  parameters: DdimStepParameters,
  stochasticNoise: Float32Array | null = null,
): Float32Array {
  if (sample.length !== predictedNoise.length) {
    throw new Error('DDIM sample and predicted-noise shapes differ.');
  }
  if (stochasticNoise !== null && stochasticNoise.length !== sample.length) {
    throw new Error('DDIM stochastic-noise shape differs from the sample.');
  }
  requireUnitInterval(parameters.alpha, 'DDIM alpha');
  requireUnitInterval(parameters.alphaPrevious, 'DDIM previous alpha');
  if (!Number.isFinite(parameters.eta) || parameters.eta < 0) {
    throw new Error('DDIM eta must be a finite non-negative number.');
  }

  const alpha = parameters.alpha;
  const alphaPrevious = parameters.alphaPrevious;
  const sigmaSquared = parameters.eta * parameters.eta
    * Math.max(0, (1 - alphaPrevious) / Math.max(1e-12, 1 - alpha))
    * Math.max(0, 1 - alpha / alphaPrevious);
  const sigma = Math.sqrt(Math.max(0, sigmaSquared));
  const predictedOriginalScale = Math.sqrt(alpha);
  const noiseScale = Math.sqrt(Math.max(0, 1 - alpha));
  const directionScale = Math.sqrt(Math.max(0, 1 - alphaPrevious - sigmaSquared));
  const previousOriginalScale = Math.sqrt(alphaPrevious);
  const output = new Float32Array(sample.length);

  for (let index = 0; index < sample.length; index += 1) {
    const current = sample[index] ?? 0;
    const epsilon = predictedNoise[index] ?? 0;
    const predictedOriginal = (current - noiseScale * epsilon) / Math.max(1e-12, predictedOriginalScale);
    const random = stochasticNoise?.[index] ?? 0;
    output[index] = previousOriginalScale * predictedOriginal + directionScale * epsilon + sigma * random;
  }
  return output;
}

export async function runDdimLoop(
  initialSample: Float32Array,
  schedule: readonly DdimLoopStep[],
  predictNoise: NoisePredictor,
  eta = 0,
  noiseSource: NoiseSource | null = null,
  onStep: ((completed: number, total: number) => void) | null = null,
): Promise<Float32Array> {
  let sample: Float32Array<ArrayBufferLike> = initialSample.slice();
  for (let index = 0; index < schedule.length; index += 1) {
    const step = schedule[index];
    if (!step) throw new Error('DDIM schedule contains an empty step.');
    const predictedNoise = await predictNoise(sample, step.trainingTimestep, index);
    const stochasticNoise = eta > 0
      ? noiseSource?.(sample.length, index) ?? null
      : null;
    if (eta > 0 && stochasticNoise === null) {
      throw new Error('DDIM eta is non-zero but no stochastic noise source was provided.');
    }
    sample = ddimStep(sample, predictedNoise, {
      alpha: step.alpha,
      alphaPrevious: step.alphaPrevious,
      eta,
    }, stochasticNoise);
    onStep?.(index + 1, schedule.length);
  }
  return sample;
}
