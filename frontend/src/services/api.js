import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
  timeout: 120000,
});

export const documentsApi = {
  list: () => api.get("/api/documents/").then((res) => res.data),
  upload: (files, onProgress) => {
    const formData = new FormData();
    Array.from(files).forEach((file) => {
      formData.append("files", file);
    });
    return api
      .post("/api/documents/upload", formData, {
        onUploadProgress: (event) => {
          if (!onProgress || !event.total) return;
          onProgress(Math.round((event.loaded * 100) / event.total));
        },
      })
      .then((res) => res.data);
  },
  remove: (documentId) =>
    api.delete(`/api/documents/${documentId}`).then((res) => res.data),
  reprocess: (documentId) =>
    api.post(`/api/documents/reprocess/${documentId}`).then((res) => res.data),
  search: (query, searchMode = "semantic", topK = 5) =>
    api
      .get(`/api/documents/search/${encodeURIComponent(query)}`, {
        params: { search_mode: searchMode, top_k: topK },
      })
      .then((res) => res.data),
};

export function friendlyError(
  error,
  fallback = "Something went wrong. Please try again.",
) {
  const detail = error?.response?.data?.detail;
  if (typeof detail === "string") {
    if (detail.toLowerCase().includes("llm")) {
      return "The AI provider is not configured yet. Document search still works, but generated answers need an API key.";
    }
    if (detail.toLowerCase().includes("embedding")) {
      return "Embedding generation failed. Please retry after checking the backend model setup.";
    }
    if (detail.toLowerCase().includes("classif")) {
      return "Document classification failed. The document remains available and can be reprocessed.";
    }
    return detail;
  }
  if (error?.code === "ERR_NETWORK") {
    return "Cannot reach the backend API. Confirm the FastAPI server is running on port 8000.";
  }
  return error?.message || fallback;
}

export const ragApi = {
  ask: (payload) => api.post("/api/rag/ask", payload).then((res) => res.data),
  summarize: (payload) =>
    api.post("/api/rag/summarize", payload).then((res) => res.data),
  compare: (payload) =>
    api.post("/api/rag/compare", payload).then((res) => res.data),
};

export const analyticsApi = {
  full: () => api.get("/api/analytics/full").then((res) => res.data),
  health: () => api.get("/api/analytics/health").then((res) => res.data),
};

export default api;
