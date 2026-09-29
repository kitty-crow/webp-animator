interface ExtendableEventLike extends Event {
  waitUntil(promise: Promise<unknown>): void;
}

interface FetchEventLike extends Event {
  readonly request: Request;
  respondWith(response: Promise<Response>): void;
}

interface ClientCollectionLike {
  claim(): Promise<void>;
}

interface ServiceWorkerScopeLike {
  readonly clients: ClientCollectionLike;
  skipWaiting(): Promise<void>;
  addEventListener(type: 'install', listener: (event: ExtendableEventLike) => void): void;
  addEventListener(type: 'activate', listener: (event: ExtendableEventLike) => void): void;
  addEventListener(type: 'fetch', listener: (event: FetchEventLike) => void): void;
}

const scope = globalThis as unknown as ServiceWorkerScopeLike;

scope.addEventListener('install', (event: ExtendableEventLike) => {
  event.waitUntil(scope.skipWaiting());
});

scope.addEventListener('activate', (event: ExtendableEventLike) => {
  event.waitUntil(scope.clients.claim());
});

scope.addEventListener('fetch', (event: FetchEventLike) => {
  const request = event.request;
  if (request.cache === 'only-if-cached' && request.mode !== 'same-origin') return;
  event.respondWith((async (): Promise<Response> => {
    const response = await fetch(request);
    if (response.type === 'opaque') return response;
    const headers = new Headers(response.headers);
    headers.set('Cross-Origin-Opener-Policy', 'same-origin');
    headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
    headers.set('Cross-Origin-Resource-Policy', 'same-origin');
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers,
    });
  })());
});
