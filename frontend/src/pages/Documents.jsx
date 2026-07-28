import { RefreshCw, Trash2, Upload } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { documentsApi, friendlyError, ragApi } from "../services/api.js";

function Documents() {
  const inputRef = useRef(null);
  const [documents, setDocuments] = useState([]);
  const [selected, setSelected] = useState("");
  const [summaryType, setSummaryType] = useState("technical");
  const [summary, setSummary] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [toast, setToast] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isDragging, setIsDragging] = useState(false);

  const loadDocuments = useCallback(
    () =>
      documentsApi.list().then((data) => setDocuments(data.documents || [])),
    [],
  );

  useEffect(() => {
    loadDocuments().catch((err) =>
      setError(friendlyError(err, "Documents could not be loaded.")),
    );
  }, [loadDocuments]);

  const uploadFiles = async (files) => {
    if (!files?.length) return;
    setBusy(true);
    setError("");
    setToast("");
    setUploadProgress(0);
    try {
      await documentsApi.upload(files, setUploadProgress);
      await loadDocuments();
      setToast("Upload started. Processing continues in the background.");
    } catch (err) {
      setError(
        friendlyError(
          err,
          "Upload failed. Please check the PDF files and retry.",
        ),
      );
    } finally {
      setBusy(false);
      setUploadProgress(0);
    }
  };

  const upload = async (event) => {
    await uploadFiles(event.target.files);
    event.target.value = "";
  };

  const dropUpload = async (event) => {
    event.preventDefault();
    setIsDragging(false);
    const files = Array.from(event.dataTransfer.files).filter(
      (file) => file.type === "application/pdf" || file.name.endsWith(".pdf"),
    );
    if (files.length === 0) {
      setError("Please drop one or more PDF files.");
      return;
    }
    await uploadFiles(files);
  };

  const summarize = async () => {
    if (!selected) return;
    setBusy(true);
    setSummary("");
    try {
      const result = await ragApi.summarize({
        document_id: selected,
        summary_type: summaryType,
        max_length: 500,
      });
      setSummary(result.summary);
      setToast("Summary generated.");
    } catch (err) {
      setError(friendlyError(err, "Summary could not be generated."));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="page-shell">
      <header className="page-header">
        <div>
          <p className="eyebrow">Documents</p>
          <h2>Upload, Process, Summarize</h2>
        </div>
        <button
          type="button"
          className="primary-button"
          disabled={busy}
          onClick={() => inputRef.current?.click()}
        >
          <Upload size={18} />
          <span>Upload PDFs</span>
        </button>
        <input
          ref={inputRef}
          className="hidden"
          type="file"
          accept="application/pdf"
          multiple
          onChange={upload}
        />
      </header>
      {toast && <div className="toast">{toast}</div>}
      {error && <div className="alert">{error}</div>}
      <button
        type="button"
        className={`drop-zone ${isDragging ? "drop-zone-active" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={dropUpload}
      >
        <Upload size={22} />
        <strong>Drag and drop PDF research papers</strong>
        <span>
          Multiple PDF upload is supported. Files are processed automatically.
        </span>
        {busy && uploadProgress > 0 && (
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        )}
      </button>
      <section className="panel overflow-x-auto">
        {documents.length === 0 ? (
          <div className="empty-state">No documents uploaded yet.</div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Status</th>
                <th>Pages</th>
                <th>Chunks</th>
                <th>Category</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {documents.map((doc) => (
                <tr key={doc.document_id}>
                  <td>{doc.document_name}</td>
                  <td>
                    <span className={`status status-${doc.processing_status}`}>
                      {doc.processing_status}
                    </span>
                  </td>
                  <td>{doc.total_pages}</td>
                  <td>{doc.total_chunks}</td>
                  <td>{doc.category}</td>
                  <td>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        className="icon-button"
                        title="Reprocess"
                        onClick={() =>
                          documentsApi
                            .reprocess(doc.document_id)
                            .then(loadDocuments)
                            .then(() => setToast("Reprocessing started."))
                            .catch((err) =>
                              setError(friendlyError(err, "Reprocess failed.")),
                            )
                        }
                      >
                        <RefreshCw size={16} />
                      </button>
                      <button
                        type="button"
                        className="icon-button danger"
                        title="Delete"
                        onClick={() =>
                          documentsApi
                            .remove(doc.document_id)
                            .then(loadDocuments)
                            .then(() => setToast("Document deleted."))
                            .catch((err) =>
                              setError(friendlyError(err, "Delete failed.")),
                            )
                        }
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
      <section className="panel mt-6">
        <h3>Document Summary</h3>
        <div className="mt-3 flex flex-col gap-3 md:flex-row">
          <select
            className="input"
            value={selected}
            onChange={(event) => setSelected(event.target.value)}
          >
            <option value="">Select a processed document</option>
            {documents.map((doc) => (
              <option key={doc.document_id} value={doc.document_id}>
                {doc.document_name}
              </option>
            ))}
          </select>
          <select
            className="input md:max-w-56"
            value={summaryType}
            onChange={(event) => setSummaryType(event.target.value)}
          >
            <option value="executive">Executive</option>
            <option value="technical">Technical</option>
            <option value="bullet">Bullet</option>
            <option value="key_takeaways">Key Takeaways</option>
          </select>
          <button
            type="button"
            className="secondary-button"
            disabled={!selected || busy}
            onClick={summarize}
          >
            {busy ? "Working..." : "Generate"}
          </button>
        </div>
        {summary ? (
          <p className="answer-block">{summary}</p>
        ) : (
          <div className="empty-state mt-4">
            Choose a document to generate a summary.
          </div>
        )}
      </section>
    </section>
  );
}

export default Documents;
