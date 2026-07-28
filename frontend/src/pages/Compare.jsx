import { GitCompare } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { documentsApi, friendlyError, ragApi } from "../services/api.js";

function Compare() {
  const sessionId = useMemo(() => `compare-${crypto.randomUUID()}`, []);
  const [documents, setDocuments] = useState([]);
  const [selected, setSelected] = useState([]);
  const [comparison, setComparison] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    documentsApi
      .list()
      .then((data) => setDocuments(data.documents || []))
      .catch((err) =>
        setError(friendlyError(err, "Documents could not be loaded.")),
      );
  }, []);

  const toggle = (id) => {
    setSelected((current) =>
      current.includes(id)
        ? current.filter((item) => item !== id)
        : [...current, id].slice(0, 5),
    );
  };

  const compare = async () => {
    if (selected.length < 2) return;
    setBusy(true);
    setError("");
    try {
      const result = await ragApi.compare({
        document_ids: selected,
        session_id: sessionId,
        comparison_aspects: [
          "methodology",
          "advantages",
          "disadvantages",
          "similarities",
          "differences",
          "implementation",
          "conclusion",
        ],
      });
      setComparison(result);
    } catch (err) {
      setError(friendlyError(err, "Comparison could not be generated."));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="page-shell">
      <header className="page-header">
        <div>
          <p className="eyebrow">Comparison</p>
          <h2>Multi-document Analysis</h2>
        </div>
        <button
          type="button"
          className="primary-button"
          disabled={selected.length < 2 || busy}
          onClick={compare}
        >
          <GitCompare size={18} />
          <span>{busy ? "Comparing..." : "Compare"}</span>
        </button>
      </header>
      {error && <div className="alert">{error}</div>}
      <section className="panel">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {documents.map((doc) => (
            <label
              key={doc.document_id}
              className={`select-card ${selected.includes(doc.document_id) ? "select-card-active" : ""}`}
            >
              <input
                type="checkbox"
                checked={selected.includes(doc.document_id)}
                onChange={() => toggle(doc.document_id)}
              />
              <span>{doc.document_name}</span>
              <small>{doc.category}</small>
            </label>
          ))}
          {documents.length === 0 && (
            <div className="empty-state">
              Upload at least two documents to compare.
            </div>
          )}
        </div>
      </section>
      {comparison && (
        <section className="panel mt-6">
          <h3>Comparison</h3>
          <p className="answer-block">{comparison.comparison}</p>
          <div className="mt-5 overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Aspect</th>
                  {comparison.documents_compared.map((name) => (
                    <th key={name}>{name}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {comparison.aspects.map((aspect) => (
                  <tr key={aspect.aspect}>
                    <td>{aspect.aspect}</td>
                    {comparison.documents_compared.map((name) => (
                      <td key={name}>{aspect.documents[name] || ""}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </section>
  );
}

export default Compare;
