import { Search, Send } from "lucide-react";
import { useMemo, useState } from "react";
import { documentsApi, friendlyError, ragApi } from "../services/api.js";

function Assistant() {
  const sessionId = useMemo(() => `session-${crypto.randomUUID()}`, []);
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("semantic");
  const [answer, setAnswer] = useState(null);
  const [searchResults, setSearchResults] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const ask = async () => {
    if (!query.trim()) return;
    setBusy(true);
    setError("");
    try {
      const [qa, search] = await Promise.all([
        ragApi.ask({
          query,
          session_id: sessionId,
          search_mode: mode,
          top_k: 5,
        }),
        documentsApi.search(query, mode, 5),
      ]);
      setAnswer(qa);
      setSearchResults(qa.retrieved_context || search.results || []);
    } catch (err) {
      setError(
        friendlyError(err, "The assistant could not answer that question."),
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="page-shell">
      <header className="page-header">
        <div>
          <p className="eyebrow">RAG</p>
          <h2>Grounded Research Answers</h2>
        </div>
      </header>
      {error && <div className="alert">{error}</div>}
      <section className="panel">
        <div className="grid gap-3 md:grid-cols-[1fr_160px_120px]">
          <input
            className="input"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && ask()}
            placeholder="Ask a question from uploaded documents"
          />
          <select
            className="input"
            value={mode}
            onChange={(event) => setMode(event.target.value)}
          >
            <option value="semantic">Semantic</option>
            <option value="keyword">Keyword</option>
            <option value="hybrid">Hybrid</option>
          </select>
          <button
            type="button"
            className="primary-button"
            disabled={busy}
            onClick={ask}
          >
            <Send size={17} />
            <span>{busy ? "Thinking..." : "Ask"}</span>
          </button>
        </div>
      </section>
      {answer && (
        <section className="panel mt-6">
          <h3>Answer</h3>
          <p className="answer-block">{answer.answer}</p>
          <div className="mt-4 text-sm text-steel">
            Confidence: {(answer.confidence_score * 100).toFixed(1)}%
          </div>
          <div className="mt-4 grid gap-3">
            {(answer.sources || []).length === 0 ? (
              <div className="empty-state">
                No source citations were available.
              </div>
            ) : (
              (answer.sources || []).map((source) => (
                <div
                  key={`${source.document_id}-${source.page_number}-${source.chunk_content.slice(0, 12)}`}
                  className="source-card"
                >
                  <strong>{source.document_name}</strong>
                  <span>Page {source.page_number}</span>
                  <p>{source.chunk_content}</p>
                </div>
              ))
            )}
          </div>
        </section>
      )}
      <section className="panel mt-6">
        <h3 className="flex items-center gap-2">
          <Search size={18} /> Retrieved Context
        </h3>
        <div className="mt-3 grid gap-3">
          {searchResults.length === 0 && (
            <div className="empty-state">
              Ask a question to retrieve context.
            </div>
          )}
          {searchResults.map((result) => (
            <div
              key={
                result.chunk_id || `${result.document_id}-${result.page_number}`
              }
              className="source-card"
            >
              <strong>{result.document_name || result.document_id}</strong>
              <span>
                Page {result.page_number} | Score{" "}
                {Number(result.similarity_score || 0).toFixed(3)}
              </span>
              <p>{result.content || result.chunk_content}</p>
            </div>
          ))}
        </div>
      </section>
    </section>
  );
}

export default Assistant;
