import type { AmtAssetBundle, AmtScaleAsset } from './amt.js';
import type { BrowserExternalDataAsset, BrowserModelAsset } from './catalog.js';
import type { EdenAssetBundle, SpeedAssetBundle } from './frame-generator.js';
import type { MogAssetBundle } from './mog.js';
import type { ResShiftAssetBundle } from './resshift.js';
import type { ToonCrafterAssetBundle } from './tooncrafter.js';

export type ManifestBackedFamily = 'amt' | 'resshift' | 'mog' | 'tooncrafter' | 'eden' | 'speed';

export type LoadedModelManifest =
  | { readonly family: 'amt'; readonly bundle: AmtAssetBundle }
  | { readonly family: 'resshift'; readonly bundle: ResShiftAssetBundle }
  | { readonly family: 'mog'; readonly bundle: MogAssetBundle }
  | { readonly family: 'tooncrafter'; readonly bundle: ToonCrafterAssetBundle }
  | { readonly family: 'eden'; readonly bundle: EdenAssetBundle }
  | { readonly family: 'speed'; readonly bundle: SpeedAssetBundle };

interface ManifestAssetRecord {
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
}

function recordOf(value: unknown, label: string): Readonly<Record<string, unknown>> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) throw new Error(`${label} must be an object.`);
  return value as Readonly<Record<string, unknown>>;
}

function stringField(record: Readonly<Record<string, unknown>>, key: string, label: string): string {
  const value = record[key];
  if (typeof value !== 'string' || value.trim().length === 0) throw new Error(`${label}.${key} must be a non-empty string.`);
  return value;
}

function finiteNumber(record: Readonly<Record<string, unknown>>, key: string, label: string): number {
  const value = record[key];
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${label}.${key} must be finite.`);
  return value;
}

function positiveInteger(record: Readonly<Record<string, unknown>>, key: string, label: string): number {
  const value = finiteNumber(record, key, label);
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${label}.${key} must be a positive integer.`);
  return value;
}

function integerField(record: Readonly<Record<string, unknown>>, key: string, label: string): number {
  const value = finiteNumber(record, key, label);
  if (!Number.isInteger(value)) throw new Error(`${label}.${key} must be an integer.`);
  return value;
}

function optionalString(record: Readonly<Record<string, unknown>>, key: string): string | null {
  const value = record[key];
  return typeof value === 'string' && value.trim().length > 0 ? value : null;
}

function parseAssets(value: unknown): readonly ManifestAssetRecord[] {
  if (!Array.isArray(value) || value.length === 0) throw new Error('Model manifest assets must be a non-empty array.');
  return value.map((entry: unknown, index: number): ManifestAssetRecord => {
    const record = recordOf(entry, `assets[${index}]`);
    const path = stringField(record, 'path', `assets[${index}]`);
    const bytes = positiveInteger(record, 'bytes', `assets[${index}]`);
    const sha256 = stringField(record, 'sha256', `assets[${index}]`).toLowerCase();
    if (!/^[0-9a-f]{64}$/.test(sha256)) throw new Error(`assets[${index}].sha256 must be a 64-character hexadecimal digest.`);
    return { path, bytes, sha256 };
  });
}

function componentPath(components: Readonly<Record<string, unknown>>, key: string): string {
  return stringField(components, key, 'components');
}

function assetFor(
  manifestUrl: URL,
  records: readonly ManifestAssetRecord[],
  graphPath: string,
  family: ManifestBackedFamily,
  source: string,
  licence: string,
): BrowserModelAsset {
  const graph = records.find((candidate) => candidate.path === graphPath);
  if (!graph) throw new Error(`Model manifest is missing asset metadata for ${graphPath}.`);
  const stem = graphPath.endsWith('.onnx') ? graphPath.slice(0, -'.onnx'.length) : graphPath;
  const externalRecords = records.filter((candidate) => candidate.path !== graphPath && candidate.path.startsWith(`${stem}.`));
  const externalData: BrowserExternalDataAsset[] = externalRecords.map((entry): BrowserExternalDataAsset => ({
    id: `${family}:${entry.path}`,
    path: entry.path,
    url: new URL(entry.path, manifestUrl).href,
    sha256: entry.sha256,
    bytes: entry.bytes,
  }));
  return {
    id: `${family}:${graphPath}`,
    url: new URL(graphPath, manifestUrl).href,
    sha256: graph.sha256,
    bytes: graph.bytes,
    licence: licence || `Upstream model terms (${source})`,
    ...(externalData.length > 0 ? { externalData } : {}),
  };
}

function parseMogSchedule(value: unknown): NonNullable<MogAssetBundle['schedule']> {
  const record = recordOf(value, 'ddim');
  const spacing = stringField(record, 'spacing', 'ddim');
  if (spacing !== 'uniform' && spacing !== 'uniform_trailing') throw new Error('ddim.spacing is invalid.');
  const zeroTerminalSnr = record['zeroTerminalSnr'];
  if (typeof zeroTerminalSnr !== 'boolean') throw new Error('ddim.zeroTerminalSnr must be boolean.');
  const dynamicRescale = record['dynamicRescale'];
  const baseScale = record['baseScale'];
  const turningStep = record['turningStep'];
  return {
    trainingTimesteps: positiveInteger(record, 'trainingTimesteps', 'ddim'),
    inferenceSteps: positiveInteger(record, 'inferenceSteps', 'ddim'),
    linearStart: finiteNumber(record, 'linearStart', 'ddim'),
    linearEnd: finiteNumber(record, 'linearEnd', 'ddim'),
    eta: finiteNumber(record, 'eta', 'ddim'),
    zeroTerminalSnr,
    spacing,
    ...(typeof dynamicRescale === 'boolean' ? { dynamicRescale } : {}),
    ...(typeof baseScale === 'number' && Number.isFinite(baseScale) ? { baseScale } : {}),
    ...(typeof turningStep === 'number' && Number.isInteger(turningStep) ? { turningStep } : {}),
  };
}

function parseToonSchedule(value: unknown): NonNullable<ToonCrafterAssetBundle['schedule']> {
  const record = recordOf(value, 'ddim');
  const zeroTerminalSnr = record['zeroTerminalSnr'];
  if (typeof zeroTerminalSnr !== 'boolean') throw new Error('ddim.zeroTerminalSnr must be boolean.');
  return {
    trainingTimesteps: positiveInteger(record, 'trainingTimesteps', 'ddim'),
    inferenceSteps: positiveInteger(record, 'inferenceSteps', 'ddim'),
    linearStart: finiteNumber(record, 'linearStart', 'ddim'),
    linearEnd: finiteNumber(record, 'linearEnd', 'ddim'),
    eta: finiteNumber(record, 'eta', 'ddim'),
    zeroTerminalSnr,
    dynamicRescale: true,
    baseScale: finiteNumber(record, 'baseScale', 'ddim'),
    turningStep: positiveInteger(record, 'turningStep', 'ddim'),
  };
}

function parseAmtScales(
  value: unknown,
  components: Readonly<Record<string, unknown>>,
  make: (path: string) => BrowserModelAsset,
): readonly AmtScaleAsset[] {
  if (!Array.isArray(value) || value.length === 0) throw new Error('AMT manifest scales must be a non-empty array.');
  const scales = value.map((entry: unknown, index: number): AmtScaleAsset => {
    const record = recordOf(entry, `scales[${index}]`);
    const component = stringField(record, 'component', `scales[${index}]`);
    const scaleFactor = finiteNumber(record, 'scaleFactor', `scales[${index}]`);
    const divisor = positiveInteger(record, 'divisor', `scales[${index}]`);
    if (scaleFactor <= 0 || scaleFactor > 1) throw new Error(`scales[${index}].scaleFactor must be in (0, 1].`);
    return {
      scaleFactor,
      divisor,
      asset: make(componentPath(components, component)),
    };
  });
  const ordered = [...scales].sort((left, right) => right.scaleFactor - left.scaleFactor);
  if (ordered[0]?.scaleFactor !== 1) throw new Error('AMT manifest must include a full-scale graph.');
  return ordered;
}

function parseSpeedScales(value: unknown): readonly number[] {
  if (!Array.isArray(value) || value.length === 0) throw new Error('SPEED manifest scales must be a non-empty array.');
  const scales = value.map((entry: unknown, index: number): number => {
    if (typeof entry !== 'number' || !Number.isFinite(entry) || entry <= 0 || entry > 1) {
      throw new Error(`scales[${index}] must be in (0, 1].`);
    }
    return entry;
  });
  const ordered = [...scales].sort((left, right) => right - left);
  if (ordered[0] !== 1) throw new Error('SPEED manifest must include full scale 1.0.');
  return ordered;
}

export async function loadModelManifest(
  manifestUrlText: string,
  expectedFamily: ManifestBackedFamily,
): Promise<LoadedModelManifest> {
  const manifestUrl = new URL(manifestUrlText, globalThis.location?.href ?? 'https://localhost/');
  const response = await fetch(manifestUrl, { cache: 'no-cache' });
  if (!response.ok) throw new Error(`Model manifest fetch failed: HTTP ${response.status}.`);
  const root = recordOf(await response.json() as unknown, 'manifest');
  if (finiteNumber(root, 'format', 'manifest') !== 1) throw new Error('Unsupported model manifest format.');
  const family = stringField(root, 'family', 'manifest');
  if (family !== expectedFamily) throw new Error(`Expected ${expectedFamily} manifest, received ${family}.`);
  const source = stringField(root, 'source', 'manifest');
  const licence = optionalString(root, 'licence') ?? `Upstream model terms (${source})`;
  const components = recordOf(root['components'], 'components');
  const assets = parseAssets(root['assets']);
  const make = (path: string): BrowserModelAsset => assetFor(manifestUrl, assets, path, expectedFamily, source, licence);

  if (expectedFamily === 'eden') {
    const cosSimStd = finiteNumber(root, 'cosSimStd', 'manifest');
    if (cosSimStd === 0) throw new Error('manifest.cosSimStd must be non-zero.');
    return {
      family: 'eden',
      bundle: {
        generator: make(componentPath(components, 'generator')),
        internalWidth: positiveInteger(root, 'internalWidth', 'manifest'),
        internalHeight: positiveInteger(root, 'internalHeight', 'manifest'),
        latentDim: positiveInteger(root, 'latentDim', 'manifest'),
        cosSimMean: finiteNumber(root, 'cosSimMean', 'manifest'),
        cosSimStd,
        seed: integerField(root, 'seed', 'manifest'),
      },
    };
  }

  if (expectedFamily === 'speed') {
    return {
      family: 'speed',
      bundle: {
        generator: make(componentPath(components, 'generator')),
        scales: parseSpeedScales(root['scales']),
        seed: integerField(root, 'seed', 'manifest'),
      },
    };
  }

  if (expectedFamily === 'amt') {
    return {
      family: 'amt',
      bundle: { scales: parseAmtScales(root['scales'], components, make) },
    };
  }

  if (expectedFamily === 'resshift') {
    return {
      family: 'resshift',
      bundle: {
        flow: make(componentPath(components, 'flow')),
        extractor: make(componentPath(components, 'extractor')),
        synthesis: make(componentPath(components, 'synthesis')),
      },
    };
  }

  const frames = positiveInteger(root, 'frames', 'manifest');
  const posteriorScaleFactor = finiteNumber(root, 'posteriorScaleFactor', 'manifest');
  const referenceHiddenCount = positiveInteger(root, 'referenceHiddenCount', 'manifest');
  const fps = positiveInteger(root, 'fps', 'manifest');
  if (posteriorScaleFactor <= 0) throw new Error('manifest.posteriorScaleFactor must be positive.');

  if (expectedFamily === 'mog') {
    return {
      family: 'mog',
      bundle: {
        motion: make(componentPath(components, 'motion')),
        vaeEncoder: make(componentPath(components, 'vaeEncoder')),
        imageCondition: make(componentPath(components, 'imageCondition')),
        denoiser: make(componentPath(components, 'denoiser')),
        vaeDecoder: make(componentPath(components, 'vaeDecoder')),
        frames,
        posteriorScaleFactor,
        referenceHiddenCount,
        fps,
        schedule: parseMogSchedule(root['ddim']),
      },
    };
  }

  return {
    family: 'tooncrafter',
    bundle: {
      vaeEncoder: make(componentPath(components, 'vaeEncoder')),
      imageCondition: make(componentPath(components, 'imageCondition')),
      denoiser: make(componentPath(components, 'denoiser')),
      vaeDecoder: make(componentPath(components, 'vaeDecoder')),
      frames,
      posteriorScaleFactor,
      referenceHiddenCount,
      fps,
      schedule: parseToonSchedule(root['ddim']),
    },
  };
}
