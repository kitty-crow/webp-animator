export interface MogScheduleConfig {
  readonly trainingTimesteps: number;
  readonly inferenceSteps: number;
  readonly linearStart: number;
  readonly linearEnd: number;
  readonly eta: number;
  readonly zeroTerminalSnr: boolean;
  readonly spacing: 'uniform' | 'uniform_trailing';
}

export interface MogDdimStep {
  readonly trainingTimestep: number;
  readonly alpha: number;
  readonly alphaPrevious: number;
  readonly sigma: number;
}

export interface MogDdimPrediction {
  readonly predictedX0: Float32Array;
  readonly epsilon: Float32Array;
}

export const DEFAULT_MOG_SCHEDULE: MogScheduleConfig = {
  trainingTimesteps: 1000,
  inferenceSteps: 50,
  linearStart: 0.00085,
  linearEnd: 0.012,
  eta: 1,
  zeroTerminalSnr: true,
  spacing: 'uniform_trailing',
};

function requirePositive(value: number, label: string): void {
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${label} must be positive.`);
}

function linearBetas(config: MogScheduleConfig): Float64Array {
  const count = config.trainingTimesteps;
  if (!Number.isInteger(count) || count < 2) throw new Error('MoG training timesteps must be an integer of at least two.');
  requirePositive(config.linearStart, 'MoG linearStart');
  requirePositive(config.linearEnd, 'MoG linearEnd');
  const start = Math.sqrt(config.linearStart);
  const end = Math.sqrt(config.linearEnd);
  const output = new Float64Array(count);
  for (let index = 0; index < count; index += 1) {
    const ratio = index / (count - 1);
    const value = start + (end - start) * ratio;
    output[index] = value * value;
  }
  return output;
}

export function rescaleZeroTerminalSnr(betas: Float64Array): Float64Array {
  if (betas.length < 2) throw new Error('Zero-terminal-SNR rescaling requires at least two betas.');
  const sqrtAlphaBar = new Float64Array(betas.length);
  let cumulative = 1;
  for (let index = 0; index < betas.length; index += 1) {
    const beta = betas[index];
    if (beta === undefined || !Number.isFinite(beta) || beta < 0 || beta >= 1) throw new Error('MoG beta schedule contains an invalid value.');
    cumulative *= 1 - beta;
    sqrtAlphaBar[index] = Math.sqrt(cumulative);
  }
  const first = sqrtAlphaBar[0];
  const last = sqrtAlphaBar[sqrtAlphaBar.length - 1];
  if (first === undefined || last === undefined || first === last) throw new Error('MoG alpha schedule cannot be rescaled.');
  const alphaBars = new Float64Array(betas.length);
  const scale = first / (first - last);
  for (let index = 0; index < sqrtAlphaBar.length; index += 1) {
    const shifted = ((sqrtAlphaBar[index] ?? 0) - last) * scale;
    alphaBars[index] = shifted * shifted;
  }
  const output = new Float64Array(betas.length);
  for (let index = 0; index < output.length; index += 1) {
    const alpha = index === 0
      ? alphaBars[0]
      : (alphaBars[index] ?? 0) / Math.max(alphaBars[index - 1] ?? 0, Number.MIN_VALUE);
    if (alpha === undefined) throw new Error('MoG rescaled alpha is missing.');
    output[index] = 1 - alpha;
  }
  return output;
}

export function alphaCumprodForMog(config: MogScheduleConfig = DEFAULT_MOG_SCHEDULE): Float64Array {
  const betas = config.zeroTerminalSnr ? rescaleZeroTerminalSnr(linearBetas(config)) : linearBetas(config);
  const output = new Float64Array(betas.length);
  let cumulative = 1;
  for (let index = 0; index < betas.length; index += 1) {
    cumulative *= 1 - (betas[index] ?? 0);
    output[index] = cumulative;
  }
  return output;
}

export function mogTimesteps(config: MogScheduleConfig = DEFAULT_MOG_SCHEDULE): Int32Array {
  const training = config.trainingTimesteps;
  const inference = config.inferenceSteps;
  if (!Number.isInteger(inference) || inference <= 0 || inference > training) throw new Error('MoG inference steps are invalid.');
  const selected: number[] = [];
  if (config.spacing === 'uniform') {
    const step = Math.floor(training / inference);
    for (let value = 0; value < training && selected.length < inference; value += step) selected.push(value + 1);
  } else {
    const step = training / inference;
    for (let value = training; value > 0 && selected.length < inference; value -= step) selected.push(Math.round(value) - 1);
    selected.reverse();
  }
  if (selected.length !== inference) throw new Error(`MoG DDIM timestep selection produced ${selected.length}, expected ${inference}.`);
  for (const value of selected) {
    if (!Number.isInteger(value) || value < 0 || value >= training) throw new Error(`MoG DDIM timestep ${value} is outside the training schedule.`);
  }
  return new Int32Array(selected);
}

export function makeMogDdimSchedule(config: MogScheduleConfig = DEFAULT_MOG_SCHEDULE): readonly MogDdimStep[] {
  requirePositive(config.eta, 'MoG DDIM eta');
  const alphaCumprod = alphaCumprodForMog(config);
  const timesteps = mogTimesteps(config);
  const steps: MogDdimStep[] = [];
  for (let index = 0; index < timesteps.length; index += 1) {
    const timestep = timesteps[index];
    if (timestep === undefined) throw new Error('MoG selected timestep is missing.');
    const alpha = alphaCumprod[timestep];
    const previousTimestep = index === 0 ? null : timesteps[index - 1];
    const alphaPrevious = previousTimestep === null
      ? alphaCumprod[0]
      : previousTimestep === undefined
        ? undefined
        : alphaCumprod[previousTimestep];
    if (alpha === undefined || alphaPrevious === undefined) throw new Error('MoG DDIM alpha lookup failed.');
    const sigma = config.eta * Math.sqrt(
      Math.max(0, (1 - alphaPrevious) / Math.max(1e-20, 1 - alpha))
      * Math.max(0, 1 - alpha / Math.max(alphaPrevious, 1e-20)),
    );
    steps.push({ trainingTimestep: timestep, alpha, alphaPrevious, sigma });
  }
  return steps;
}

export function mogVPrediction(
  sample: Float32Array,
  velocity: Float32Array,
  alpha: number,
): MogDdimPrediction {
  if (sample.length !== velocity.length) throw new Error('MoG sample and velocity shapes differ.');
  if (!Number.isFinite(alpha) || alpha < 0 || alpha > 1) throw new Error('MoG v-prediction alpha is invalid.');
  const sqrtAlpha = Math.sqrt(alpha);
  const sqrtOneMinusAlpha = Math.sqrt(Math.max(0, 1 - alpha));
  const predictedX0 = new Float32Array(sample.length);
  const epsilon = new Float32Array(sample.length);
  for (let index = 0; index < sample.length; index += 1) {
    const x = sample[index] ?? 0;
    const v = velocity[index] ?? 0;
    predictedX0[index] = sqrtAlpha * x - sqrtOneMinusAlpha * v;
    epsilon[index] = sqrtAlpha * v + sqrtOneMinusAlpha * x;
  }
  return { predictedX0, epsilon };
}

export function mogDdimStep(
  sample: Float32Array,
  velocity: Float32Array,
  step: MogDdimStep,
  noise: Float32Array,
): Float32Array {
  if (sample.length !== noise.length) throw new Error('MoG DDIM noise shape differs from the sample.');
  const prediction = mogVPrediction(sample, velocity, step.alpha);
  const directionScale = Math.sqrt(Math.max(0, 1 - step.alphaPrevious - step.sigma * step.sigma));
  const previousOriginalScale = Math.sqrt(step.alphaPrevious);
  const output = new Float32Array(sample.length);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = previousOriginalScale * (prediction.predictedX0[index] ?? 0)
      + directionScale * (prediction.epsilon[index] ?? 0)
      + step.sigma * (noise[index] ?? 0);
  }
  return output;
}
