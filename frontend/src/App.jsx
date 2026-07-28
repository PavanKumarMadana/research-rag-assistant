import {
  BarChart3,
  FileText,
  GitCompare,
  MessageSquareText,
} from "lucide-react";
import { useState } from "react";
import Assistant from "./pages/Assistant.jsx";
import Compare from "./pages/Compare.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Documents from "./pages/Documents.jsx";

const navItems = [
  { id: "analytics", label: "Analytics", icon: BarChart3 },
  { id: "documents", label: "Documents", icon: FileText },
  { id: "assistant", label: "Ask", icon: MessageSquareText },
  { id: "compare", label: "Compare", icon: GitCompare },
];

const pages = {
  analytics: Dashboard,
  documents: Documents,
  assistant: Assistant,
  compare: Compare,
};

function App() {
  const [activePage, setActivePage] = useState("analytics");
  const ActivePage = pages[activePage];

  return (
    <div className="min-h-screen bg-paper text-ink">
      <aside className="fixed inset-y-0 left-0 hidden w-64 border-r border-slate-200 bg-white px-4 py-5 lg:block">
        <div className="mb-8">
          <p className="text-sm font-semibold uppercase tracking-wide text-mint">
            Research RAG
          </p>
          <h1 className="mt-1 text-2xl font-bold">Knowledge Assistant</h1>
        </div>
        <nav className="space-y-1">
          {navItems.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              type="button"
              onClick={() => setActivePage(id)}
              className={`nav-link w-full ${activePage === id ? "nav-link-active" : ""}`}
            >
              <Icon size={18} />
              <span>{label}</span>
            </button>
          ))}
        </nav>
      </aside>
      <main className="lg:pl-64">
        <div className="border-b border-slate-200 bg-white px-4 py-3 lg:hidden">
          <div className="flex items-center gap-2 overflow-x-auto">
            {navItems.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                type="button"
                onClick={() => setActivePage(id)}
                className={`mobile-link ${activePage === id ? "mobile-link-active" : ""}`}
              >
                <Icon size={16} />
                <span>{label}</span>
              </button>
            ))}
          </div>
        </div>
        <ActivePage />
      </main>
    </div>
  );
}

export default App;
