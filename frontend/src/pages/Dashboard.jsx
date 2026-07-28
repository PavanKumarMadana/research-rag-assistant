import { Activity, Database, FileStack, SearchCheck } from "lucide-react";
import { useEffect, useState } from "react";
import { analyticsApi, friendlyError } from "../services/api.js";

function Dashboard() {
  const [analytics, setAnalytics] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    analyticsApi
      .full()
      .then(setAnalytics)
      .catch((err) =>
        setError(friendlyError(err, "Analytics could not be loaded.")),
      )
      .finally(() => setLoading(false));
  }, []);

  const overview = analytics?.overview || {};
  const stats = [
    {
      label: "Documents",
      value: overview.total_documents ?? 0,
      icon: FileStack,
    },
    { label: "Chunks", value: overview.total_chunks ?? 0, icon: Database },
    {
      label: "Embeddings",
      value: overview.total_embeddings ?? 0,
      icon: SearchCheck,
    },
    { label: "Queries", value: overview.total_queries ?? 0, icon: Activity },
  ];

  return (
    <section className="page-shell">
      <section className="hero-panel">
        <div>
          <p className="eyebrow">Enterprise Research Assistant</p>
          <h2>
            Upload papers, search semantically, and ask grounded questions with
            citations.
          </h2>
          <p>
            A production-oriented RAG workspace for document processing, vector
            search, TensorFlow classification, comparison, summarization, and
            analytics.
          </p>
        </div>
      </section>
      <header className="page-header">
        <div>
          <p className="eyebrow">Analytics</p>
          <h2>Research Corpus Overview</h2>
        </div>
      </header>
      {error && <div className="alert">{error}</div>}
      {loading && (
        <div className="panel text-sm text-steel">Loading analytics...</div>
      )}
      <div className="grid gap-4 md:grid-cols-4">
        {stats.map(({ label, value, icon: Icon }) => (
          <div key={label} className="metric-card">
            <Icon size={22} />
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
      <div className="mt-6 grid gap-6 xl:grid-cols-2">
        <section className="panel">
          <h3>Category Distribution</h3>
          <div className="space-y-3">
            {!loading &&
              (analytics?.category_distribution || []).length === 0 && (
                <div className="empty-state">No categorized documents yet.</div>
              )}
            {(analytics?.category_distribution || []).map((item) => (
              <div key={item.category}>
                <div className="row">
                  <span>{item.category}</span>
                  <strong>{item.count}</strong>
                </div>
                <div className="bar-track">
                  <div
                    className="bar-fill"
                    style={{
                      width: `${Math.min(100, (item.count / Math.max(overview.total_documents || 1, 1)) * 100)}%`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </section>
        <section className="panel">
          <h3>Most Searched Documents</h3>
          <div className="space-y-3">
            {!loading && (analytics?.top_documents || []).length === 0 && (
              <div className="empty-state">
                No document queries recorded yet.
              </div>
            )}
            {(analytics?.top_documents || []).map((doc) => (
              <div key={doc.document_id} className="row">
                <span>{doc.document_name}</span>
                <strong>{doc.query_count}</strong>
              </div>
            ))}
          </div>
        </section>
      </div>
    </section>
  );
}

export default Dashboard;
