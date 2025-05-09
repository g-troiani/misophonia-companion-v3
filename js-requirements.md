<!-- File: js-requirements.md -->
################################################################################
# File: js-requirements.md
################################################################################
# JavaScript Dependencies for Misophonia Research RAG System

## Core Dependencies

```json
{
  "dependencies": {
    "dotenv": "^16.0.3",
    "fast-glob": "^3.3.3",
    "firebase": "^10.0.0",
    "firebase-admin": "^11.8.0",
    "mammoth": "^1.9.0",
    "node-fetch": "^3.3.1",
    "openai": "^4.9.0",
    "pdfjs-dist": "^3.7.107",
    "express": "^4.18.2",
    "cors": "^2.8.5",
    "axios": "^1.4.0",
    "body-parser": "^1.20.2"
  }
}
```

## Development Dependencies

```json
{
  "devDependencies": {
    "@eslint/js": "^9.22.0",
    "@types/react": "^19.0.10",
    "@types/react-dom": "^19.0.4",
    "@vitejs/plugin-react": "^4.3.4",
    "eslint": "^9.22.0",
    "eslint-plugin-react-hooks": "^5.2.0",
    "eslint-plugin-react-refresh": "^0.4.19",
    "globals": "^16.0.0",
    "netlify-cli": "^13.2.2",
    "vite": "^6.3.1",
    "vite-plugin-pwa": "^1.0.0"
  }
}
```

## Firebase-specific Dependencies

```json
{
  "dependencies": {
    "firebase/app": "included in firebase package",
    "firebase/firestore": "included in firebase package",
    "firebase/functions": "included in firebase package",
    "firebase-admin/app": "included in firebase-admin package",
    "firebase-admin/firestore": "included in firebase-admin package",
    "firebase-functions": "^4.3.0"
  }
}
```

## Installation Instructions

1. These dependencies are already defined in your project's `package.json` file.
2. To install all dependencies, run:
   ```
   npm install
   ```
3. For Firebase Cloud Functions, navigate to the functions directory and run:
   ```
   cd functions
   npm install
   ```

## Notes

- The Firebase client and admin SDKs are separate packages with different use cases:
  - `firebase` is for client-side applications
  - `firebase-admin` is for server-side applications and has elevated privileges

- For the RAG system, the key dependencies are:
  - `openai`: For generating embeddings and AI responses
  - `firebase-admin`: For accessing Firestore vector database
  - `pdfjs-dist`: For processing PDF documents
  - `express`: For the web server interface
