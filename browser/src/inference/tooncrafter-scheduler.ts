import {
  DEFAULT_MOG_SCHEDULE,
  makeMogDdimSchedule,
  mogVPrediction,
  type MogDdimStep,
  type MogScheduleConfig,
} from './mog-scheduler.js';

export interface ToonCrafterScheduleConfig extends MogScheduleConfig {
  readonly baseScale: number;
  readonly turningStep: number;
}

export const DEFAULT_TOONCRAFTER_SCHEDULE: ToonCrafterScheduleConfig = {
  ...DEFAULT_MOG_SCHEDULE,
  baseScale: 0.7,
  turningStep: 400,
};

export interface ToonCrafterDdimStep extends MogDdimStep {
  readonly dynamicScale: number;
  readonly previousDynamicScale: number;
}

function dynamicScaleAt(timestep: number, config: ToonCrafterScheduleConfig): number {
  if (!Number.isInteger(timestep) || timestep < 0 || timestep >= config.trainingTimesteps) throw new Error('ToonCrafter dynamic-scale timestep is invalid.');
  if (!Number.isFinite(config.baseScale) || config.baseScale <= 0) throw new Error('ToonCrafter baseScale must be positive.');
  if (!Number.isInteger(config.turningStep) || config.turningStep < 2) throw new Error('ToonCrafter turningStep must be at least two.');
  if (timestep >= config.turningStep) return config.baseScale;
  const ratio = timestep / (config.turningStep - 1);
  return 1 + (config.baseScale - 1) * ratio;
}

export function makeToonCrafterDdimSchedule(
  config: ToonCrafterScheduleConfig = DEFAULT_TOONCRAFTER_SCHEDULE,
): readonly ToonCrafterDdimStep[] {
  const base = makeMogDdimSchedule(config);
  return base.map((step, index) => {
    const dynamicScale = dynamicScaleAt(step.trainingTimestep, config);
    const previous = index === 0 ? step : base[index - 1];
    if (!previous) throw new Error('ToonCrafter previous DDIM step is missing.');
    const previousDynamicScale = index === 0
      ? dynamicScale
      : dynamicScaleAt(previous.trainingTimestep, config);
    return { ...step, dynamicScale, previousDynamicScale };
  });
}

export function toonCrafterDdimStep(
  sample: Float32Array,
  velocity: Float32Array,
  step: ToonCrafterDdimStep,
  noise: Float32Array,
): Float32Array {
  if (sample.length !== velocity.length || sample.length !== noise.length) throw new Error('ToonCrafter DDIM tensor lengths differ.');
  const prediction = mogVPrediction(sample, velocity, step.alpha);
  const rescale = step.previousDynamicScale / step.dynamicScale;
  const directionScale = Math.sqrt(Math.max(0, 1 - step.alphaPrevious - step.sigma * step.sigma));
  const previousOriginalScale = Math.sqrt(step.alphaPrevious);
  const output = new Float32Array(sample.length);
  for (let index = 0; index < output.length; index += 1) {
    const predictedX0 = (prediction.predictedX0[index] ?? 0) * rescale;
    output[index] = previousOriginalScale * predictedX0
      + directionScale * (prediction.epsilon[index] ?? 0)
      + step.sigma * (noise[index] ?? 0);
  }
  return output;
}

export function toonCrafterScheduleForLatentWidth(latentWidth: number): ToonCrafterScheduleConfig {
  if (!Number.isInteger(latentWidth) || latentWidth <= 0) throw new Error('ToonCrafter latent width must be a positive integer.');
  return {
    ...DEFAULT_TOONCRAFTER_SCHEDULE,
    spacing: latentWidth === 32 ? 'uniform' : 'uniform_trailing',
  };
}
