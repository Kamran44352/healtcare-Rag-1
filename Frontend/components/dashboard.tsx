"use client";

import { Navbar } from "@/components/navbar";
import { ChatPanel } from "@/components/chat-panel";
import { useLocalStorage } from "@/hooks/use-local-storage";

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

export function Dashboard() {
  const [apiBase] = useLocalStorage("rag_api_base", DEFAULT_API_BASE);

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <Navbar />
      <main className="flex min-h-0 flex-1 flex-col">
        <ChatPanel baseUrl={apiBase} />
      </main>
    </div>
  );
}
