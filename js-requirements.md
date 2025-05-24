<!-- File: js-requirements.md -->
################################################################################
# File: js-requirements.md
################################################################################
# JavaScript Dependencies

## Core Dependencies

For frontend:
```json
"react": "^18.2.0",
"react-dom": "^18.2.0",
"vite": "^4.0.0",
"@vitejs/plugin-react": "^3.0.0"
```

For backend (Node.js):
```json
"express": "^4.18.0",
"cors": "^2.8.5",
"dotenv": "^16.0.0",
"openai": "^4.0.0"
```

## AI/ML Dependencies

```json
"openai": "^4.0.0",
"groq-sdk": "^0.21.0"
```

## Vector Database Dependencies

```json
"@supabase/supabase-js": "^2.39.0"
```

## Development Dependencies

```json
"@types/node": "^18.0.0",
"@types/react": "^18.0.0",
"@types/react-dom": "^18.0.0",
"eslint": "^8.0.0",
"typescript": "^4.9.0"
```

## Installation Commands

1. For the main project:
```bash
npm install
```

2. If using a separate server directory:
```bash
cd server && npm install
```

## Notes

- All dependencies are focused on Supabase integration for vector storage
- The OpenAI SDK handles embeddings and chat completions
- Groq SDK provides alternative LLM options
