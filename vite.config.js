// File: vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      workbox: {
        runtimeCaching: [
          {
            urlPattern: /^\/api\/(chat|gemini|rag).*$/,
            handler: 'NetworkOnly',
          },
        ],
      },
      manifest: {
        name: 'Misophonia Companion',
        short_name: 'Companion',
        description: 'A therapeutic and research PWA for misophonia.',
        start_url: '.',
        display: 'standalone',
        background_color: '#f8f6ff',
        theme_color: '#b2d8d8',
        icons: [
          {
            src: 'icon-192.png',
            sizes: '192x192',
            type: 'image/png'
          },
          {
            src: 'icon-512.png',
            sizes: '512x512',
            type: 'image/png'
          }
        ]
      }
    })
  ],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:3001',
        changeOrigin: true,
      }
    }
  }
})
