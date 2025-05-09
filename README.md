<!-- File: README.md -->
################################################################################
# File: README.md
################################################################################

<!-- PROJECT LOGO -->
<p align="center">
  <img src="public/vite.svg" alt="Logo" width="120" height="120">

</p>

<h1 align="center">Misophonia Companion</h1>

<p align="center">
  <b>The modern, AI-powered guide and support tool for those living with misophonia.</b><br>
  <i>Built with React, Vite, Node.js, and OpenAI</i>
  <br><br>
  <a href="https://flourishing-sprite-c819cb.netlify.app/"><img src="https://img.shields.io/badge/Live%20Demo-Online-brightgreen?style=for-the-badge" alt="Live Demo"></a>
  <a href="https://github.com/mannino49/Misophonia-companion-v2"><img src="https://img.shields.io/github/stars/mannino49/Misophonia-companion-v2?style=for-the-badge" alt="GitHub Stars"></a>
</p>

---

## 🚀 Features

- **Conversational AI Chatbot:** Powered by OpenAI, get real-time support and information.
- **Soundscape Player:** Customizable soundscapes to help manage triggers.
- **Modern UI:** Responsive, accessible, and visually appealing interface.
- **Progressive Web App:** Installable and works offline.
- **Secure Backend:** All API keys and secrets are kept on the server, never exposed to the client.

---

## 🖥️ Tech Stack

<div align="center">
  <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" />
  <img src="https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=FFD62E" />
  <img src="https://img.shields.io/badge/Node.js-339933?style=for-the-badge&logo=nodedotjs&logoColor=white" />
  <img src="https://img.shields.io/badge/Express-000000?style=for-the-badge&logo=express&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/Netlify-00C7B7?style=for-the-badge&logo=netlify&logoColor=white" />
</div>

---

## 📦 Project Structure

```shell
Misophonia Guide/
├── public/                # Static assets (icons, manifest)
├── src/                   # React frontend source
│   ├── App.jsx            # Main app logic
│   ├── main.jsx           # React entry point
│   └── ...
├── server/                # Node.js/Express backend
│   ├── index.js           # API server entry
│   └── ...
├── netlify.toml           # Netlify deployment config
├── package.json           # Frontend config
└── ...
```

---

## ⚡ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/g-troiani/misophonia-companion-v3
cd Misophonia-companion-v3
```

### 2. Install dependencies
```bash
npm install
cd server && npm install
```

### 3. Set up environment variables
- Copy `.env.example` to `.env` in the `server/` directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_key_here
```

### 4. Run the backend server
```bash
cd server
npm start
```

### 5. Run the frontend (in a new terminal)
```bash
npm run dev
```

- Frontend: [http://localhost:5173](http://localhost:5173)
- Backend API: [http://localhost:3001](http://localhost:3001)

---

## 🌐 Deployment

- Deployed on Netlify: [Live Demo](https://flourishing-sprite-c819cb.netlify.app/)
- Backend runs as a separate Node.js server (see `server/`)
- All secrets are stored in environment variables and never exposed to the frontend.

---

## 🛡️ Security & Best Practices

- **No secrets or API keys are stored in the frontend.**
- **.env files and private keys are gitignored.**
- **Backend validates API key presence and never exposes it to the client.**

---

## 🤝 Contributing

Contributions are welcome! Please open issues or submit pull requests.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Made with ❤️ by Mannino49</b>
</p>
