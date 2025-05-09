// File: src/App.jsx
import './App.css'
import React, { useState, useEffect, useRef } from 'react';
import botAvatar from './assets/bot-avatar.png';
import userAvatar from './assets/user-avatar.png';
import TermsModal from './TermsModal';
import RagAssistant from './RagAssistant.jsx';


function NavBar({ setSection, section }) {
  return (
    <nav className="navbar">
      <button className={section === 'home' ? 'active' : ''} onClick={() => setSection('home')}>
        <span role="img" aria-label="home" style={{marginRight: 6}}>🏠</span> Home
      </button>
      <button className={section === 'chatbot' ? 'active' : ''} onClick={() => setSection('chatbot')}>
        <span role="img" aria-label="chat" style={{marginRight: 6}}>💬</span> Let's Talk
      </button>
      <button className={section === 'tools' ? 'active' : ''} onClick={() => setSection('tools')}>
        <span role="img" aria-label="tools" style={{marginRight: 6}}>🧰</span> Therapeutic Tools
      </button>
      <button className={section === 'research' ? 'active' : ''} onClick={() => setSection('research')}>
        <span role="img" aria-label="research" style={{marginRight: 6}}>🔬</span> Research Assistant
      </button>
    </nav>
  );
}

const AFFIRMATIONS = [
  "You are safe here.",
  "It's okay to take a break.",
  "Your feelings are valid.",
  "Breathe in calm, breathe out stress.",
  "You are not alone."
];
const SOUNDS = [
  { label: 'Rain', src: 'https://cdn.pixabay.com/audio/2022/07/26/audio_124bfae45e.mp3' },
  { label: 'Forest', src: 'https://cdn.pixabay.com/audio/2022/03/15/audio_115b9d7bfa.mp3' },
  { label: 'White Noise', src: 'https://cdn.pixabay.com/audio/2022/03/15/audio_115b9d7bfa.mp3' }
];

function AffirmationBanner() {
  const [idx] = useState(() => Math.floor(Math.random() * AFFIRMATIONS.length));
  return (
    <div style={{
      background: 'linear-gradient(90deg, #e0e7ef 60%, #f8f6ff 100%)',
      color: '#4b6073',
      borderRadius: '16px',
      margin: '0 0 1.1rem 0',
      padding: '0.7rem 1.2rem',
      fontSize: '1.12rem',
      fontWeight: 500,
      textAlign: 'center',
      boxShadow: '0 1px 6px 0 rgba(31, 38, 135, 0.04)',
      letterSpacing: '0.01em',
      opacity: 0.97
    }}>
      {AFFIRMATIONS[idx]}
    </div>
  );
}

function SoundscapePlayer() {
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(false);
  const [soundIdx, setSoundIdx] = useState(0);
  const audioRef = useRef(null);

  const handlePlayPause = () => {
    setPlaying(p => !p);
  };
  const handleMute = () => {
    setMuted(m => !m);
  };
  const handleSoundChange = (e) => {
    setSoundIdx(Number(e.target.value));
    setPlaying(false);
    setTimeout(() => setPlaying(true), 50);
  };

  useEffect(() => {
    if (!audioRef.current) return;
    audioRef.current.muted = muted;
    if (playing) {
      audioRef.current.play();
    } else {
      audioRef.current.pause();
    }
  }, [playing, muted, soundIdx]);

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '0.7rem',
      background: 'rgba(255,255,255,0.88)',
      borderRadius: '13px',
      padding: '0.25rem 0.8rem',
      marginBottom: '1.1rem',
      boxShadow: '0 1px 4px 0 rgba(31, 38, 135, 0.03)',
      fontSize: '1.01rem',
      maxWidth: 320
    }}>
      <span style={{color: '#b2d8d8', fontWeight: 700, fontSize: '1.1rem'}}>Soundscape:</span>
      <select value={soundIdx} onChange={handleSoundChange} style={{borderRadius: 7, border: '1px solid #e0e7ef', background: '#f8f6ff', color: '#4b6073', padding: '0.2rem 0.5rem'}}>
        {SOUNDS.map((s, i) => <option value={i} key={s.label}>{s.label}</option>)}
      </select>
      <button onClick={handlePlayPause} style={{border: 'none', background: 'none', cursor: 'pointer', color: playing ? '#81b0b0' : '#aaa', fontWeight: 700, fontSize: '1.05rem'}}>{playing ? 'Pause' : 'Play'}</button>
      <button onClick={handleMute} style={{border: 'none', background: 'none', cursor: 'pointer', color: muted ? '#aaa' : '#b2d8d8', fontWeight: 700, fontSize: '1.05rem'}}>{muted ? 'Unmute' : 'Mute'}</button>
      <audio ref={audioRef} src={SOUNDS[soundIdx].src} loop preload="auto" style={{display: 'none'}} />
    </div>
  );
}

function App() {
  const [section, setSection] = useState('home');
  const [termsAccepted, setTermsAccepted] = useState(localStorage.getItem('termsAccepted') === 'true');
  if (!termsAccepted) return <TermsModal onAccept={() => setTermsAccepted(true)} />;

  return (
    <>
      <div className="animated-bg">
        <div className="bubble bubble1"></div>
        <div className="bubble bubble2"></div>
        <div className="bubble bubble3"></div>
        <div className="bubble bubble4"></div>
        <div className="bubble bubble5"></div>
        <div className="bubble bubble6"></div>
        <div className="bubble bubble7"></div>
        <div className="bubble bubble8"></div>
      </div>

      {/* MAIN CONTENT BOX */}
      <div className="container">
        <AffirmationBanner />
        <SoundscapePlayer />
        <NavBar setSection={setSection} section={section} />
        {section === 'home' && (
          <div className="card">
            <main>
              <h1 className="title">Welcome to Misophonia Companion</h1>
              <p className="subtitle">A soothing space to manage triggers, support healing, and explore research—built for both sufferers and professionals.</p>
            </main>
          </div>
        )}
        {section === 'chatbot' && <Chatbot />}
        {section === 'tools' && (
          <div className="card">
            <main>
              <h2>Therapeutic Tools</h2>
              <p>Coming soon: Sound therapy, coping strategies, and relaxation exercises.</p>
            </main>
          </div>
        )}
        {section === 'research' && <RagAssistant />}
      </div>

      {/* NEW: footer lives here, outside .container */}
      <div className="disclaimer-footer">
        Misophonia Companion is not a clinical tool or a substitute for
        professional psychological or medical treatment. It does not provide
        diagnosis, therapy, or crisis intervention. If you are experiencing a
        mental-health emergency, please contact a licensed provider or
        emergency services immediately.
      </div>
    </>
  );
}


function Chatbot() {
  const [messages, setMessages] = useState([
    { sender: 'bot', text: 'Hello! I am your Misophonia Companion. How can I support you today?' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSend(e) {
    e.preventDefault();
    if (!input.trim()) return;
    const userMsg = { sender: 'user', text: input };
    setMessages((msgs) => [...msgs, userMsg]);
    setLoading(true);
    setError(null);
    setInput('');
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [
            { role: 'system', content: 'You are a calm, supportive misophonia companion and research assistant.' },
            ...messages.map(m => ({ role: m.sender === 'user' ? 'user' : 'assistant', content: m.text })),
            { role: 'user', content: input }
          ]
        })
      });
      if (!res.ok) throw new Error('API error');
      const data = await res.json();
      setMessages((msgs) => [...msgs, { sender: 'bot', text: data.reply }]);
    } catch (err) {
      setMessages((msgs) => [...msgs, { sender: 'bot', text: 'Sorry, I could not connect to the assistant. Make sure the backend server and your API key are set up.' }]);
      setError('API error');
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h2>Let's Talk</h2>
      <div className="chatbot-box">
        <div className="chat-messages">
          {messages.map((msg, idx) => (
            <div key={idx} className={msg.sender === 'bot' ? 'msg bot' : 'msg user'} style={{display: 'flex', alignItems: 'flex-end', justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start'}}>
              {msg.sender === 'bot' && (
                <img
                  src={botAvatar}
                  alt="Bot"
                  className="chat-avatar"
                  onError={e => { e.target.onerror = null; e.target.src = 'https://ui-avatars.com/api/?name=Bot&background=b2d8d8&color=fff&rounded=true&size=64'; }}
                />
              )}
              <span className="bubble-content">{msg.text}</span>
              {msg.sender === 'user' && (
                <img
                  src={userAvatar}
                  alt="You"
                  className="chat-avatar"
                  onError={e => { e.target.onerror = null; e.target.src = 'https://ui-avatars.com/api/?name=You&background=ffdac1&color=3a3a3a&rounded=true&size=64'; }}
                />
              )}
            </div>
          ))}
          {loading && <div className="msg bot" style={{display: 'flex', alignItems: 'flex-end'}}><img src={botAvatar} alt="Bot" className="chat-avatar" /><span className="bubble-content">Thinking…</span></div>}
        </div>
        <form className="chat-input-row" onSubmit={handleSend}>
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Type your message..."
            className="chat-input"
            autoFocus
            aria-label="Type your message"
            disabled={loading}
          />
          <button type="submit" className="chat-send" disabled={loading || !input.trim()}>Send</button>
        </form>
        {error && <div style={{ color: '#b22222', marginTop: '0.5rem' }}>
          Make sure your backend server is running and your OpenAI API key is set in <code>server/.env</code>.
        </div>}
      </div>
    </main>
  );
}



// GeminiChatbot: Gemini 2.5 Pro chat with topics and structured output
function GeminiChatbot() {
  const TOPICS = [
    { label: 'Neuroscience', value: 'Neuroscience' },
    { label: 'Genetics', value: 'Genetics' },
    { label: 'Therapy', value: 'Therapy' },
    { label: 'Advocacy', value: 'Advocacy' },
    { label: 'News', value: 'Latest News' },
    { label: 'Free Text', value: '' }
  ];
  const [messages, setMessages] = useState([
    { sender: 'bot', text: 'Hi! I am your Gemini-powered Research Assistant. Select a topic or ask anything about misophonia research.' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [mode, setMode] = useState('gemini'); // future: allow OpenAI fallback

  async function handleGeminiSend(e, topicOverride) {
    if (e) e.preventDefault();
    const topicVal = topicOverride !== undefined ? topicOverride : selectedTopic;
    const userText = topicVal && topicVal !== '' && topicVal !== 'Free Text' ? topicVal : input;
    if (!userText.trim()) return;
    const userMsg = { sender: 'user', text: userText };
    setMessages((msgs) => [...msgs, userMsg]);
    setLoading(true);
    setError(null);
    setInput('');
    setSelectedTopic(null);
    try {
      const res = await fetch('/api/gemini', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: messages.map(m => ({ role: m.sender === 'user' ? 'user' : 'assistant', content: m.text })).concat([{ role: 'user', content: userText }]),
          topic: topicVal && topicVal !== 'Free Text' ? topicVal : undefined
        })
      });
      if (!res.ok) throw new Error('API error');
      const data = await res.json();
      setMessages((msgs) => [...msgs, { sender: 'bot', text: data.reply, structured: data.structured }]);
    } catch (err) {
      setMessages((msgs) => [...msgs, { sender: 'bot', text: 'Sorry, Gemini API error. Check your backend and API key.' }]);
      setError('API error');
    } finally {
      setLoading(false);
    }
  }

  function renderStructured(structured) {
    if (!structured) return null;
    // Example structure: { sections: [{ title, bullets, highlights, ... }], ... }
    return (
      <div className="gemini-structured">
        {structured.sections && structured.sections.map((sec, i) => (
          <div key={i} className="g-section">
            {sec.title && <div className="g-title">{sec.title}</div>}
            {sec.highlights && Array.isArray(sec.highlights) && (
              <ul className="g-highlights">{sec.highlights.map((h, j) => <li key={j} className="g-highlight">{h}</li>)}</ul>
            )}
            {sec.bullets && Array.isArray(sec.bullets) && (
              <ul className="g-bullets">{sec.bullets.map((b, j) => <li key={j}>{b}</li>)}</ul>
            )}
            {sec.text && <div className="g-text">{sec.text}</div>}
          </div>
        ))}
        {structured.extra && <div className="g-extra">{structured.extra}</div>}
      </div>
    );
  }

  return (
    <div className="gemini-chatbot">
      <div className="gemini-toggle-row">
        <button
          className={mode === 'gemini' ? 'toggle-active' : ''}
          onClick={() => setMode('gemini')}
        >Gemini 2.5 Pro</button>
        {/* <button
          className={mode === 'openai' ? 'toggle-active' : ''}
          onClick={() => setMode('openai')}
        >OpenAI</button> */}
      </div>
      <div className="gemini-topic-row">
        {TOPICS.map(t => (
          <button
            key={t.label}
            className={selectedTopic === t.value ? 'topic-btn selected' : 'topic-btn'}
            onClick={() => {
              setSelectedTopic(t.value);
              if (t.value && t.value !== 'Free Text') handleGeminiSend(null, t.value);
            }}
            disabled={loading}
          >{t.label}</button>
        ))}
      </div>
      <div className="chat-messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={msg.sender === 'bot' ? 'msg bot' : 'msg user'} style={{display: 'flex', alignItems: 'flex-end', justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start'}}>
            {msg.sender === 'bot' && (
              <img
                src={botAvatar}
                alt="Bot"
                className="chat-avatar"
                onError={e => { e.target.onerror = null; e.target.src = 'https://ui-avatars.com/api/?name=Bot&background=b2d8d8&color=fff&rounded=true&size=64'; }}
              />
            )}
            <span className="bubble-content">
              {msg.structured ? renderStructured(msg.structured) : msg.text}
            </span>
            {msg.sender === 'user' && (
              <img
                src={userAvatar}
                alt="You"
                className="chat-avatar"
                onError={e => { e.target.onerror = null; e.target.src = 'https://ui-avatars.com/api/?name=You&background=ffdac1&color=3a3a3a&rounded=true&size=64'; }}
              />
            )}
          </div>
        ))}
        {loading && <div className="msg bot" style={{display: 'flex', alignItems: 'flex-end'}}><img src={botAvatar} alt="Bot" className="chat-avatar" /><span className="bubble-content">Thinking…</span></div>}
      </div>
      <form className="chat-input-row" onSubmit={e => handleGeminiSend(e)}>
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Type your question or pick a topic..."
          className="chat-input"
          autoFocus
          aria-label="Type your message"
          disabled={loading}
        />
        <button type="submit" className="chat-send" disabled={loading || !input.trim()}>Send</button>
      </form>
      {error && <div style={{ color: '#b22222', marginTop: '0.5rem' }}>
        Gemini backend error. Check your server and API key in <code>server/.env</code>.
      </div>}
    </div>
  );
}

export default App
