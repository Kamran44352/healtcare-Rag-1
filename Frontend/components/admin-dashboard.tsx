"use client";

import { useEffect, useState } from "react";
import { Activity } from "lucide-react";
import { toast } from "sonner";

import { CrawlForm, CrawlList, useCrawls } from "@/components/crawl-panel";
import { IngestionPanel } from "@/components/ingestion-panel";
import { Navbar } from "@/components/navbar";
import { RetrievalPanel } from "@/components/retrieval-panel";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { checkHealth } from "@/lib/api";
import { useLocalStorage } from "@/hooks/use-local-storage";

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

export function AdminDashboard() {
  const [apiBase, setApiBase] = useLocalStorage("rag_api_base", DEFAULT_API_BASE);
  const [health, setHealth] = useState<"unknown" | "healthy" | "down">("unknown");

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      if (!apiBase) return;
      try {
        await checkHealth(apiBase);
        if (!cancelled) setHealth("healthy");
      } catch {
        if (!cancelled) setHealth("down");
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [apiBase]);

  return (
    <div className="flex min-h-screen flex-col overflow-x-hidden">
      <Navbar />
      <main className="min-w-0 flex-1 px-4 py-6 sm:px-6 lg:px-10">
        <div className="mx-auto w-full max-w-screen-xl space-y-6">
          <div className="space-y-1">
            <h1 className="text-2xl font-bold tracking-tight text-primary">Admin Panel</h1>
            <p className="text-sm text-muted-foreground">Manage documents and monitor ingestion jobs.</p>
          </div>

          <div className="flex min-w-0 flex-col gap-3 rounded-xl border border-border bg-card p-4 shadow-sm sm:flex-row sm:items-end">
            <div className="flex-1 space-y-1.5">
              <Label htmlFor="apiBase" className="text-sm font-medium">
                Backend API URL
              </Label>
              <Input
                id="apiBase"
                value={apiBase}
                onChange={(e) => setApiBase(e.target.value)}
                placeholder="http://127.0.0.1:8000"
                className="font-mono text-sm"
              />
            </div>
            <div className="flex items-center gap-3">
              <Button
                variant="secondary"
                onClick={async () => {
                  try {
                    await checkHealth(apiBase);
                    setHealth("healthy");
                    toast.success("Backend reachable");
                  } catch (error) {
                    setHealth("down");
                    toast.error(`Backend unreachable: ${(error as Error).message}`);
                  }
                }}
              >
                Check Health
              </Button>
              <Badge
                variant={health === "healthy" ? "success" : health === "down" ? "destructive" : "secondary"}
                className="gap-1.5 px-3 py-1"
              >
                <Activity className="h-3.5 w-3.5" />
                {health}
              </Badge>
            </div>
          </div>

          <RetrievalPanel baseUrl={apiBase} />
          <IngestionWithCrawls baseUrl={apiBase} />
        </div>
      </main>
    </div>
  );
}

// Renders the document pipeline with the website-crawl cards interleaved:
// Upload → Crawl Website → Documents Library → Website Crawls → Ingestion Jobs.
function IngestionWithCrawls({ baseUrl }: { baseUrl: string }) {
  const crawl = useCrawls(baseUrl);
  return (
    <IngestionPanel
      baseUrl={baseUrl}
      slotAfterUpload={<CrawlForm crawl={crawl} />}
      slotAfterLibrary={<CrawlList crawl={crawl} />}
    />
  );
}
