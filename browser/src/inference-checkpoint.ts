import type { ModelExecutionProvider } from './inference/ort-runtime.js';

const DATABASE_NAME = 'webp-animator-inference-v1';
const DATABASE_VERSION = 1;
const STORE_NAME = 'jobs';

export type InferenceCheckpointStatus = 'running' | 'completed' | 'cancelled' | 'failed';

export interface InferenceCheckpoint {
  readonly jobId: string;
  readonly model: 'rife';
  readonly status: InferenceCheckpointStatus;
  readonly current: number;
  readonly total: number;
  readonly provider: ModelExecutionProvider | null;
  readonly updatedAt: number;
  readonly error: string | null;
}

function indexedDb(): IDBFactory | null {
  return typeof indexedDB === 'undefined' ? null : indexedDB;
}

function requestResult<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    request.addEventListener('success', () => resolve(request.result), { once: true });
    request.addEventListener('error', () => reject(request.error ?? new Error('IndexedDB request failed.')), { once: true });
  });
}

async function openDatabase(): Promise<IDBDatabase | null> {
  const factory = indexedDb();
  if (!factory) return null;
  const request = factory.open(DATABASE_NAME, DATABASE_VERSION);
  request.addEventListener('upgradeneeded', () => {
    const database = request.result;
    if (!database.objectStoreNames.contains(STORE_NAME)) database.createObjectStore(STORE_NAME, { keyPath: 'jobId' });
  });
  try {
    return await requestResult(request);
  } catch (error: unknown) {
    console.warn('Inference checkpoint database is unavailable.', error);
    return null;
  }
}

function transactionDone(transaction: IDBTransaction): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    transaction.addEventListener('complete', () => resolve(), { once: true });
    transaction.addEventListener('abort', () => reject(transaction.error ?? new Error('IndexedDB transaction aborted.')), { once: true });
    transaction.addEventListener('error', () => reject(transaction.error ?? new Error('IndexedDB transaction failed.')), { once: true });
  });
}

export async function writeInferenceCheckpoint(checkpoint: InferenceCheckpoint): Promise<void> {
  const database = await openDatabase();
  if (!database) return;
  try {
    const transaction = database.transaction(STORE_NAME, 'readwrite');
    transaction.objectStore(STORE_NAME).put(checkpoint);
    await transactionDone(transaction);
  } catch (error: unknown) {
    console.warn('Could not persist inference checkpoint.', error);
  } finally {
    database.close();
  }
}

export async function readInferenceCheckpoint(jobId: string): Promise<InferenceCheckpoint | null> {
  const database = await openDatabase();
  if (!database) return null;
  try {
    const transaction = database.transaction(STORE_NAME, 'readonly');
    const result: unknown = await requestResult(transaction.objectStore(STORE_NAME).get(jobId));
    await transactionDone(transaction);
    if (typeof result !== 'object' || result === null) return null;
    const record = result as Partial<InferenceCheckpoint>;
    return record.jobId === jobId && record.model === 'rife' ? record as InferenceCheckpoint : null;
  } catch (error: unknown) {
    console.warn('Could not read inference checkpoint.', error);
    return null;
  } finally {
    database.close();
  }
}

export async function deleteInferenceCheckpoint(jobId: string): Promise<void> {
  const database = await openDatabase();
  if (!database) return;
  try {
    const transaction = database.transaction(STORE_NAME, 'readwrite');
    transaction.objectStore(STORE_NAME).delete(jobId);
    await transactionDone(transaction);
  } catch (error: unknown) {
    console.warn('Could not delete inference checkpoint.', error);
  } finally {
    database.close();
  }
}
