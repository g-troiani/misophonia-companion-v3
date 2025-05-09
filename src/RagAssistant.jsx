import React, { useState } from 'react';
import { marked } from 'marked';

// ⬇️  NEW DEFAULT:  falls back to " /api/rag" in production/PWA
const API = import.meta.env.VITE_RAG_ENDPOINT || '/api/rag';

export default function RagAssistant() {
  const [q, setQ] = useState('');
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState('');
  const [results, setResults] = useState([]);

  const ask = async e => {
    e.preventDefault();
    if (!q.trim()) return;
    setLoading(true);

    try {
      const r = await fetch(API, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, limit: 8 })
      });
      if (!r.ok) throw new Error(`Server responded ${r.status}`);
      const d = await r.json();
      setAnswer(d.answer ? marked.parse(d.answer) : 'No answer.');
      setResults(d.results || []);
    } catch (err) {
      setAnswer(`<p style="color:red;">${err.message}</p>`);
      setResults([]);
    } finally {
      setLoading(false);    // always clear spinner
    }
  };

  return (
    <div className="ra-card">
      <h2 style={{marginBottom: '1rem'}}>Research Assistant</h2>
      
      <div className="ra-body">
        {answer && (
          <div
            dangerouslySetInnerHTML={{ __html: answer }}
            style={{ marginBottom: '1.5rem' }}
          />
        )}

        {loading && <p>Searching…</p>}

        {results.map((r, i) => (
          <details key={i} style={{marginBottom: '1rem'}}>
            <summary>
              <strong>[{i + 1}] {r.doc?.title || 'Untitled'}</strong>
              <span style={{float: 'right', fontSize: '.85rem', opacity: .7}}>
                {r.similarity.toFixed(1)} %
              </span>
            </summary>
            <blockquote style={{margin: '.75rem 0'}}>{r.text}</blockquote>
            <p style={{fontSize: '.85rem', opacity: .8}}>
              {(r.doc?.authors || []).join(', ') || 'Unknown authors'} · {r.doc?.journal || ''} ({r.doc?.year || 'n.d.'}) · pp. {r.page_start}-{r.page_end}
            </p>
            {r.doc?.doi && (
              <a href={`https://doi.org/${r.doc.doi}`} target="_blank" rel="noreferrer">doi:{r.doc.doi}</a>
            )}
          </details>
        ))}
      </div>

      <form className="ra-input-row" onSubmit={ask}>
        <input
          className="ra-input"
          value={q}
          onChange={e => setQ(e.target.value)}
          placeholder="Ask a research question…"
        />
        <button className="ra-btn" disabled={loading}>Search</button>
      </form>
    </div>
  );
}
