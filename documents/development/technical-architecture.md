<!-- File: documents/development/technical-architecture.md -->
################################################################################
# File: documents/development/technical-architecture.md
################################################################################
# 🏗️ Technical Architecture

## Overview
- Frontend: React + Vite
- Backend: Node.js + Express
- AI: OpenAI API (secure, server-side only)
- Deployment: Netlify (frontend), custom server (backend)

## Security
- All API keys and secrets are stored server-side
- No secrets exposed to the frontend or version control ([see .gitignore](../../.gitignore))

## References
- [Product Vision](./product-vision.md)
- [Feature Roadmap](./feature-roadmap.md)
