/* Gatherion v2.0 Mobil Kumanda — PWA Service Worker (ui/sw.js) */
'use strict';

const CACHE_NAME = 'gatherion-v2-pwa-cache-v1';
const SHELL_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/sw.js'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(SHELL_ASSETS).catch(() => {
        return Promise.resolve();
      });
    }).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            return caches.delete(key);
          }
          return Promise.resolve();
        })
      );
    }).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (!req || req.method !== 'GET') {
    return;
  }

  const url = new URL(req.url);

  // Canlı telemetri, SSE akışı, WebSocket ve kamera kareleri asla önbelleğe alınmaz
  if (
    url.pathname.startsWith('/api/') ||
    url.pathname === '/ws' ||
    url.pathname === '/api/ws' ||
    url.pathname === '/api/stream'
  ) {
    event.respondWith(
      fetch(req).catch(() => {
        return new Response(
          JSON.stringify({
            ok: false,
            offline: true,
            error: 'Çevrimdışı mod: Gatherion sunucusuna şu anda ulaşılamıyor.'
          }),
          {
            status: 503,
            headers: {
              'Content-Type': 'application/json; charset=utf-8',
              'Cache-Control': 'no-store'
            }
          }
        );
      })
    );
    return;
  }

  // Kabuk (App Shell) dosyaları için Network-First, çevrimdışıysa Cache-Fallback
  event.respondWith(
    fetch(req)
      .then((networkResp) => {
        if (networkResp && networkResp.status === 200) {
          const respClone = networkResp.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(req, respClone);
          });
        }
        return networkResp;
      })
      .catch(() => {
        return caches.match(req).then((cachedResp) => {
          if (cachedResp) {
            return cachedResp;
          }
          if (req.mode === 'navigate') {
            return caches.match('/');
          }
          return new Response('Gatherion v2.0 Çevrimdışı', {
            status: 503,
            headers: { 'Content-Type': 'text/plain; charset=utf-8' }
          });
        });
      })
  );
});

self.addEventListener('message', (event) => {
  if (!event.data) return;
  if (event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  } else if (event.data.type === 'CLEAR_CACHE') {
    caches.delete(CACHE_NAME);
  }
});

self.addEventListener('push', (event) => {
  let payload = {
    title: 'Gatherion v2.0 Mobil Kumanda',
    body: 'Bot durum bildirimi alındı.',
    tag: 'gatherion-alert'
  };
  if (event.data) {
    try {
      const parsed = event.data.json();
      payload = Object.assign(payload, parsed);
    } catch (e) {
      payload.body = event.data.text() || payload.body;
    }
  }
  event.waitUntil(
    self.registration.showNotification(payload.title, {
      body: payload.body,
      tag: payload.tag || 'gatherion-alert',
      icon: '/icon-192.png',
      badge: '/icon-192.png',
      vibrate: [180, 80, 180]
    })
  );
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clientList) => {
      for (const client of clientList) {
        if ('focus' in client) {
          return client.focus();
        }
      }
      if (self.clients.openWindow) {
        return self.clients.openWindow('/');
      }
      return Promise.resolve();
    })
  );
});
