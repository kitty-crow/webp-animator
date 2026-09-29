import type { BrowserModelAsset } from './catalog.js';

const MODEL_CACHE_NAME = 'webp-animator-models-v1';

function bytesToHex(bytes: Uint8Array): string {
  let result = '';
  for (const value of bytes) result += value.toString(16).padStart(2, '0');
  return result;
}

async function sha256Hex(buffer: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  return bytesToHex(new Uint8Array(digest));
}

async function verifyAsset(asset: BrowserModelAsset, buffer: ArrayBuffer): Promise<void> {
  if (buffer.byteLength !== asset.bytes) {
    throw new Error(
      `${asset.id} model size mismatch: expected ${asset.bytes} bytes, received ${buffer.byteLength}.`,
    );
  }
  const actual = await sha256Hex(buffer);
  if (actual.toLowerCase() !== asset.sha256.toLowerCase()) {
    throw new Error(`${asset.id} SHA-256 mismatch. Expected ${asset.sha256}, received ${actual}.`);
  }
}

async function cacheStorage(): Promise<Cache | null> {
  if (!('caches' in globalThis)) return null;
  try {
    return await caches.open(MODEL_CACHE_NAME);
  } catch {
    return null;
  }
}

async function readCached(asset: BrowserModelAsset, cache: Cache): Promise<Uint8Array | null> {
  const response = await cache.match(asset.url);
  if (!response) return null;
  const buffer = await response.arrayBuffer();
  try {
    await verifyAsset(asset, buffer);
    return new Uint8Array(buffer);
  } catch {
    await cache.delete(asset.url);
    return null;
  }
}

async function fetchVerified(asset: BrowserModelAsset): Promise<{ readonly bytes: Uint8Array; readonly response: Response }> {
  const response = await fetch(asset.url, { mode: 'cors', credentials: 'omit', cache: 'no-cache' });
  if (!response.ok) throw new Error(`Failed to download ${asset.id}: HTTP ${response.status}.`);
  const buffer = await response.arrayBuffer();
  await verifyAsset(asset, buffer);
  return { bytes: new Uint8Array(buffer), response };
}

export async function loadModelBytes(asset: BrowserModelAsset): Promise<Uint8Array> {
  const cache = await cacheStorage();
  if (cache) {
    const cached = await readCached(asset, cache);
    if (cached) return cached;
  }

  const downloaded = await fetchVerified(asset);
  if (cache) {
    try {
      await cache.put(
        asset.url,
        new Response(downloaded.bytes.slice().buffer, {
          headers: {
            'content-type': 'application/octet-stream',
            'x-webp-animator-sha256': asset.sha256,
          },
        }),
      );
    } catch {
      // Cache quota or private-browsing restrictions must never make inference fail.
    }
  }
  return downloaded.bytes;
}

export async function clearBrowserModelCache(): Promise<boolean> {
  if (!('caches' in globalThis)) return false;
  return caches.delete(MODEL_CACHE_NAME);
}
