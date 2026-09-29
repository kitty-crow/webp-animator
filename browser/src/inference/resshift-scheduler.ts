export interface ResShiftScheduleConfig {
  readonly timesteps: number;
  readonly kappa: number;
  readonly p: number;
  readonly minNoiseLevel: number;
  readonly etasEnd: number;
}

export interface ResShiftSchedule {
  readonly sqrtSumEta: Float64Array;
  readonly sumEta: Float64Array;
  readonly sumPreviousEta: Float64Array;
  readonly backwardMeanC1: Float64Array;
  readonly backwardMeanC2: Float64Array;
  readonly backwardStd: Float64Array;
}

export interface ResShiftEndpoints {
  readonly first: Float32Array;
  readonly second: Float32Array;
  readonly tau: number;
}

function requireEndpoints(endpoints: ResShiftEndpoints): void {
  if (endpoints.first.length !== endpoints.second.length) {
    throw new Error('ResShift endpoint tensors must have the same shape.');
  }
  if (!Number.isFinite(endpoints.tau) || endpoints.tau < 0 || endpoints.tau > 1) {
    throw new Error('ResShift tau must be between 0 and 1.');
  }
}

function scheduleValue(array: Float64Array, index: number, label: string): number {
  const value = array[index];
  if (value === undefined || !Number.isFinite(value)) {
    throw new Error(`ResShift ${label} is missing at step ${index}.`);
  }
  return value;
}

export function makeResShiftSchedule(config: ResShiftScheduleConfig): ResShiftSchedule {
  const timesteps = Math.floor(config.timesteps);
  if (timesteps < 2) throw new Error('ResShift requires at least two timesteps.');
  if (!Number.isFinite(config.kappa) || config.kappa <= 0) throw new Error('ResShift kappa must be positive.');
  if (!Number.isFinite(config.p) || config.p <= 0) throw new Error('ResShift p must be positive.');
  if (!Number.isFinite(config.minNoiseLevel) || config.minNoiseLevel <= 0) {
    throw new Error('ResShift minNoiseLevel must be positive.');
  }
  if (!Number.isFinite(config.etasEnd) || config.etasEnd <= 0 || config.etasEnd >= 1) {
    throw new Error('ResShift etasEnd must be between 0 and 1.');
  }

  const sqrtEtaOne = Math.min(
    config.minNoiseLevel / config.kappa,
    config.minNoiseLevel,
    Math.sqrt(0.001),
  );
  const b0 = Math.exp(Math.log(config.etasEnd / sqrtEtaOne) / (timesteps - 1));
  const sqrtSumEta = new Float64Array(timesteps);
  const sumEta = new Float64Array(timesteps);
  const sumPreviousEta = new Float64Array(timesteps);
  const backwardMeanC1 = new Float64Array(timesteps);
  const backwardMeanC2 = new Float64Array(timesteps);
  const backwardStd = new Float64Array(timesteps);

  for (let index = 0; index < timesteps; index += 1) {
    const fraction = index / (timesteps - 1);
    const beta = Math.pow(fraction, config.p) * (timesteps - 1);
    const sqrtEta = Math.pow(b0, beta) * sqrtEtaOne;
    sqrtSumEta[index] = sqrtEta;
    sumEta[index] = sqrtEta * sqrtEta;
  }

  for (let index = 0; index < timesteps; index += 1) {
    const eta = scheduleValue(sumEta, index, 'sum eta');
    const previous = index === 0 ? 0 : scheduleValue(sumEta, index - 1, 'previous sum eta');
    const alpha = eta - previous;
    sumPreviousEta[index] = previous;
    backwardMeanC1[index] = eta === 0 ? 0 : previous / eta;
    backwardMeanC2[index] = eta === 0 ? 0 : alpha / eta;
    backwardStd[index] = config.kappa * Math.sqrt(Math.max(0, previous * alpha / Math.max(eta, 1e-18)));
  }

  return { sqrtSumEta, sumEta, sumPreviousEta, backwardMeanC1, backwardMeanC2, backwardStd };
}

export function initialiseResShiftSample(
  endpoints: ResShiftEndpoints,
  schedule: ResShiftSchedule,
  kappa: number,
  stochasticNoise: Float32Array,
): Float32Array {
  requireEndpoints(endpoints);
  if (stochasticNoise.length !== endpoints.first.length) {
    throw new Error('ResShift initial noise shape differs from the endpoints.');
  }
  const finalIndex = schedule.sumEta.length - 1;
  const eta = scheduleValue(schedule.sumEta, finalIndex, 'sum eta');
  const sqrtEta = scheduleValue(schedule.sqrtSumEta, finalIndex, 'sqrt sum eta');
  const tauFirst = endpoints.tau;
  const tauSecond = 1 - endpoints.tau;
  const output = new Float32Array(endpoints.first.length);
  for (let index = 0; index < output.length; index += 1) {
    const first = endpoints.first[index] ?? 0;
    const second = endpoints.second[index] ?? 0;
    const mean = eta * (tauFirst * first + tauSecond * second);
    output[index] = mean + kappa * sqrtEta * (stochasticNoise[index] ?? 0);
  }
  return output;
}

export function resShiftReverseStep(
  sample: Float32Array,
  predictedX0: Float32Array,
  endpoints: ResShiftEndpoints,
  schedule: ResShiftSchedule,
  timestep: number,
  stochasticNoise: Float32Array,
): Float32Array {
  requireEndpoints(endpoints);
  if (sample.length !== predictedX0.length || sample.length !== endpoints.first.length) {
    throw new Error('ResShift reverse-step tensor shapes differ.');
  }
  if (stochasticNoise.length !== sample.length) {
    throw new Error('ResShift reverse-step noise shape differs from the sample.');
  }
  if (!Number.isInteger(timestep) || timestep < 0 || timestep >= schedule.sumEta.length) {
    throw new Error('ResShift timestep is outside the schedule.');
  }

  const meanC1 = scheduleValue(schedule.backwardMeanC1, timestep, 'backward mean c1');
  const meanC2 = scheduleValue(schedule.backwardMeanC2, timestep, 'backward mean c2');
  const std = scheduleValue(schedule.backwardStd, timestep, 'backward std');
  const eta = scheduleValue(schedule.sumEta, timestep, 'sum eta');
  const previousEta = scheduleValue(schedule.sumPreviousEta, timestep, 'previous sum eta');
  const tauFirst = endpoints.tau;
  const tauSecond = 1 - endpoints.tau;
  const output = new Float32Array(sample.length);

  for (let index = 0; index < sample.length; index += 1) {
    const predicted = predictedX0[index] ?? 0;
    const firstError = (endpoints.first[index] ?? 0) - predicted;
    const secondError = (endpoints.second[index] ?? 0) - predicted;
    const etaError = eta * (tauFirst * firstError + tauSecond * secondError);
    const previousEtaError = previousEta * (tauFirst * firstError + tauSecond * secondError);
    const mean = meanC1 * ((sample[index] ?? 0) + etaError)
      + meanC2 * predicted
      - previousEtaError;
    output[index] = mean + std * (stochasticNoise[index] ?? 0);
  }
  return output;
}
