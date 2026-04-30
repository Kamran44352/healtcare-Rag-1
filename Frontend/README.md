# Frontend Next (Modern UI)

Modern Next.js frontend for the Healthcare RAG backend.

## Stack
- Next.js (App Router, TypeScript)
- Tailwind CSS
- shadcn-style component architecture
- Magic UI style motion/background components (`BlurFade`, `DotPattern`)
- AI-elements style chat building blocks (`Conversation`, `Message`, `PromptInput`, `Response`)

## Features
- Chat UI with SSE streaming support (`/v1/chat/query/stream`) + fallback HTTP mode (`/v1/chat/query`)
- Session controls (`new`, `load history`)
- Retrieval debug/stage timeline
- Ingestion UI for PDF upload + metadata (`category`, `reference`, `guideline_title`)
- Live ingestion job polling (`/v1/ingestions/{id}`)

## Run
```bash
cd frontend-next
npm install
npm run dev
```

Open `http://localhost:3000`.

## Environment
Optional:
```bash
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000
```

If not set, you can still edit API base URL in-app.
