export interface DdimStepParameters {
  readonly alpha: number;
  readonly alphaPrevious: number;
  readonly eta: number;
}

export interface DdimVelocityStepParameters extends DdimStepParameters {
  readonly dynamicScale?: number;
  readonly dynamicScalePrevious?: number;
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

function sigmaSquared(parameters: DdimStepParameters): number {
  return parameters.eta * parameters.eta
    * Math.max(0, (1 - parameters.alphaPrevious) / Math.max(1e-12, 1 - parameters.alpha))
    * Math.max(0, 1 - parameters.alpha / parameters.alphaPrevious);
}

function validateStepInputs(
  sample: Float32Array,
  prediction: Float32Array,
  parameters: DdimStepParameters,
  stochasticNoise: Float32Array | null,
): void {
  if (sample.length !== prediction.length) {
    throw new Error('DDIM sample and model-output shapes differ.');
  }
  if (stochasticNoise !== null && stochasticNoise.length !== sample.length) {
    throw new Error('DDIM stochastic-noise shape differs from the sample.');
  }
  requireUnitInterval(parameters.alpha, 'DDIM alpha');
  requireUnitInterval(parameters.alphaPrevious, 'DDIM previous alpha');
  if (!Number.isFinite(parameters.eta) || parameters.eta < 0) {
    throw new Error('DDIM eta must be a finite non-negative number.');
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

/**
 * Reproduces the original latent-diffusion make_ddim_timesteps("uniform")
 * convention used by MoG/ToonCrafter: floor-stride timesteps are offset by
 * +1, and the first previous alpha is alphas_cumprod[0] rather than 1.0.
 */
export function makeLegacyLdmUniformDdimSchedule(
  alphasCumprod: readonly number[],
  inferenceSteps: number,
): readonly DdimLoopStep[] {
  if (alphasCumprod.length < 2) throw new Error('Legacy DDIM requires at least two training alpha values.');
  if (!Number.isInteger(inferenceSteps) || inferenceSteps < 1 || inferenceSteps > alphasCumprod.length) {
    throw new Error('Legacy DDIM inferenceSteps are invalid.');
  }
  const stride = Math.floor(alphasCumprod.length / inferenceSteps);
  if (stride < 1) throw new Error('Legacy DDIM stride collapsed to zero.');

  const ascending: number[] = [];
  for (let raw = 0; raw < alphasCumprod.length; raw += stride) {
    const trainingTimestep = raw + 1;
    if (trainingTimestep >= alphasCumprod.length) break;
    ascending.push(trainingTimestep);
  }
  if (ascending.length === 0) throw new Error('Legacy DDIM produced no timesteps.');

  const alphaPreviousAscending = ascending.map((_, index): number => {
    if (index === 0) {
      const first = alphasCumprod[0];
      if (first === undefined) throw new Error('Legacy DDIM initial alpha is missing.');
      return first;
    }
    const previousTimestep = ascending[index - 1];
    if (previousTimestep === undefined) throw new Error('Legacy DDIM previous timestep is missing.');
    const previous = alphasCumprod[previousTimestep];
    if (previous === undefined) throw new Error('Legacy DDIM previous alpha is missing.');
    return previous;
  });

  const output: DdimLoopStep[] = [];
  for (let index = ascending.length - 1; index >= 0; index -= 1) {
    const trainingTimestep = ascending[index];
    const alphaPrevious = alphaPreviousAscending[index];
    if (trainingTimestep === undefined || alphaPrevious === undefined) {
      throw new Error('Legacy DDIM schedule entry is missing.');
    }
    const alpha = alphasCumprod[trainingTimestep];
    if (alpha === undefined) throw new Error('Legacy DDIM alpha lookup failed.');
    requireUnitInterval(alpha, 'Legacy DDIM alpha');
    requireUnitInterval(alphaPrevious, 'Legacy DDIM previous alpha');
    output.push({ trainingTimestep, alpha, alphaPrevious });
  }
  return output;
}

export function ddimStep(
  sample: Float32Array,
  predictedNoise: Float32Array,
  parameters: DdimStepParameters,
  stochasticNoise: Float32Array | null = null,
): Float32Array {
  validateStepInputs(sample, predictedNoise, parameters, stochasticNoise);

  const sigma2 = sigmaSquared(parameters);
  const sigma = Math.sqrt(Math.max(0, sigma2));
  const predictedOriginalScale = Math.sqrt(parameters.alpha);
  const noiseScale = Math.sqrt(Math.max(0, 1 - parameters.alpha));
  const directionScale = Math.sqrt(Math.max(0, 1 - parameters.alphaPrevious - sigma2));
  const previousOriginalScale = Math.sqrt(parameters.alphaPrevious);
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

/**
 * DDIM step for Stable-Diffusion-style v-prediction. This mirrors MoG's
 * predict_eps_from_z_and_v / predict_start_from_z_and_v equations and its
 * optional dynamic latent rescale.
 */
export function ddimVelocityStep(
  sample: Float32Array,
  predictedVelocity: Float32Array,
  parameters: DdimVelocityStepParameters,
  stochasticNoise: Float32Array | null = null,
): Float32Array {
  validateStepInputs(sample, predictedVelocity, parameters, stochasticNoise);
  const scale = parameters.dynamicScale ?? 1;
  const previousScale = parameters.dynamicScalePrevious ?? scale;
  if (!Number.isFinite(scale) || scale <= 0 || !Number.isFinite(previousScale) || previousScale <= 0) {
    throw new Error('DDIM dynamic scales must be finite positive numbers.');
  }

  const sigma2 = sigmaSquared(parameters);
  const sigma = Math.sqrt(Math.max(0, sigma2));
  const sqrtAlpha = Math.sqrt(parameters.alpha);
  const sqrtOneMinusAlpha = Math.sqrt(Math.max(0, 1 - parameters.alpha));
  const directionScale = Math.sqrt(Math.max(0, 1 - parameters.alphaPrevious - sigma2));
  const previousOriginalScale = Math.sqrt(parameters.alphaPrevious);
  const rescale = previousScale / scale;
  const output = new Float32Array(sample.length);

  for (let index = 0; index < sample.length; index += 1) {
    const current = sample[index] ?? 0;
    const velocity = predictedVelocity[index] ?? 0;
    const epsilon = sqrtAlpha * velocity + sqrtOneMinusAlpha * current;
    const predictedOriginal = (sqrtAlpha * current - sqrtOneMinusAlpha * velocity) * rescale;
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

export async function runDdimVelocityLoop(
  initialSample: Float32Array,
  schedule: readonly DdimLoopStep[],
  predictVelocity: NoisePredictor,
  eta: number,
  dynamicScaleForTimestep: ((trainingTimestep: number) => number) | null = null,
  noiseSource: NoiseSource | null = null,
  onStep: ((completed: number, total: number) => void) | null = null,
): Promise<Float32Array> {
  let sample: Float32Array<ArrayBufferLike> = initialSample.slice();
  for (let index = 0; index < schedule.length; index += 1) {
    const step = schedule[index];
    if (!step) throw new Error('DDIM velocity schedule contains an empty step.');
    const velocity = await predictVelocity(sample, step.trainingTimestep, index);
    const stochasticNoise = eta > 0
      ? noiseSource?.(sample.length, index) ?? null
      : null;
    if (eta > 0 && stochasticNoise === null) {
      throw new Error('DDIM eta is non-zero but no stochastic noise source was provided.');
    }

    const ascendingPreviousTimestep = index + 1 < schedule.length
      ? schedule[index + 1]?.trainingTimestep ?? step.trainingTimestep
      : step.trainingTimestep;
    const dynamicScale = dynamicScaleForTimestep?.(step.trainingTimestep) ?? 1;
    const dynamicScalePrevious = dynamicScaleForTimestep?.(ascendingPreviousTimestep) ?? dynamicScale;
    sample = ddimVelocityStep(sample, velocity, {
      alpha: step.alpha,
      alphaPrevious: step.alphaPrevious,
      eta,
      dynamicScale,
      dynamicScalePrevious,
    }, stochasticNoise);
    onStep?.(index + 1, schedule.length);
  }
  return sample;
}
