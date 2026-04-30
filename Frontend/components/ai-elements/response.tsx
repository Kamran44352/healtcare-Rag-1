import { Streamdown } from "streamdown";

import { cn } from "@/lib/utils";

type ResponseProps = {
  markdown: string;
  className?: string;
  streaming?: boolean;
};

function normalizeMarkdown(markdown: string) {
  const normalized = String(markdown || "")
    .replace(/\r\n/g, "\n")
    .replace(/\\n/g, "\n")
    // Convert [SOURCE n] inline citation tags to styled inline code spans e.g. `[1]`
    .replace(/\[SOURCE\s+(\d+)\]/gi, (_, n) => `\`[${n}]\``);

  return normalizeGfmTables(unwrapFencedTables(normalized));
}

function isTableDelimiterLine(line: string) {
  return /^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$/.test(line);
}

function looksLikeTableRow(line: string) {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith("```")) return false;
  return trimmed.includes("|") && (trimmed.match(/\|/g) || []).length >= 2;
}

function containsMarkdownTable(text: string) {
  const lines = text.split("\n");
  for (let idx = 0; idx < lines.length - 1; idx += 1) {
    if (looksLikeTableRow(lines[idx]) && isTableDelimiterLine(lines[idx + 1])) {
      return true;
    }
  }
  return false;
}

function unwrapFencedTables(markdown: string) {
  const wrapped = markdown.replace(
    /(^|\n)```(?:markdown|md|gfm|table|text|txt)?[^\n]*\n([\s\S]*?)\n```(?=\n|$)/gi,
    (fullMatch, prefix: string, body: string) => {
      if (!containsMarkdownTable(body)) {
        return fullMatch;
      }
      return `${prefix}${body}`;
    }
  );

  return wrapped.replace(
    /(^|\n)```(?:markdown|md|gfm|table|text|txt)?[^\n]*\n([\s\S]*)$/i,
    (fullMatch, prefix: string, body: string) => {
      if (!containsMarkdownTable(body)) {
        return fullMatch;
      }
      return `${prefix}${body}`;
    }
  );
}

function normalizeGfmTables(markdown: string) {
  const lines = markdown.split("\n");
  const out: string[] = [];

  let idx = 0;
  while (idx < lines.length) {
    const line = lines[idx];
    const next = idx + 1 < lines.length ? lines[idx + 1] : "";
    const tableStart = looksLikeTableRow(line) && isTableDelimiterLine(next);

    if (!tableStart) {
      out.push(line);
      idx += 1;
      continue;
    }

    if (out.length > 0 && out[out.length - 1].trim() !== "") {
      out.push("");
    }

    out.push(line.trimEnd());
    out.push(next.trimEnd());
    idx += 2;

    while (idx < lines.length && looksLikeTableRow(lines[idx])) {
      out.push(lines[idx].trimEnd());
      idx += 1;
    }

    if (idx < lines.length && lines[idx].trim() !== "") {
      out.push("");
    }
  }

  return out.join("\n");
}

function splitStableStreamingMarkdown(markdown: string) {
  if (!markdown) return { stable: "", tail: "" };
  const lines = markdown.split("\n");
  const endsWithNewline = markdown.endsWith("\n");
  let stableLineCount = endsWithNewline ? lines.length : Math.max(0, lines.length - 1);
  let fenceOpen = false;
  let lastFenceStart = -1;

  for (let idx = 0; idx < lines.length; idx += 1) {
    if (/^\s*```/.test(lines[idx])) {
      fenceOpen = !fenceOpen;
      if (fenceOpen) lastFenceStart = idx;
    }
  }
  if (fenceOpen && lastFenceStart >= 0) {
    stableLineCount = Math.min(stableLineCount, lastFenceStart);
  }

  return {
    stable: lines.slice(0, stableLineCount).join("\n"),
    tail: lines.slice(stableLineCount).join("\n")
  };
}

function closeUnterminatedCodeFence(markdown: string) {
  const fenceCount = (markdown.match(/(^|\n)\s*```/g) || []).length;
  if (fenceCount % 2 === 0) return markdown;
  return `${markdown}\n\`\`\``;
}

function suppressDanglingMarkdownArtifacts(markdown: string) {
  let value = markdown;

  value = value.replace(/(^|\n)\s*[-*+]\s*$/g, "$1");
  value = value.replace(/(^|\n)\s*\d+\.\s*$/g, "$1");
  value = value.replace(/(^|\n)```[a-z0-9_-]*\s*$/gim, "$1");
  value = value.replace(/[`*_~|]+\s*$/g, "");

  return value;
}

function sanitizeStreamingTail(tail: string) {
  return tail
    .replace(/```/g, "")
    .replace(/\*\*/g, "")
    .replace(/__/g, "")
    .replace(/[*_`~]/g, "")
    .replace(/\n{3,}/g, "\n\n");
}

export function Response({ markdown, className, streaming = false }: ResponseProps) {
  const normalized = normalizeMarkdown(markdown);
  const stableParts = streaming ? splitStableStreamingMarkdown(normalized) : { stable: normalized, tail: "" };
  const renderable = closeUnterminatedCodeFence(
    streaming ? suppressDanglingMarkdownArtifacts(stableParts.stable) : stableParts.stable
  );

  return (
    <div className={cn("max-w-none text-sm leading-relaxed text-foreground", className)}>
      <Streamdown
        mode={streaming ? "streaming" : "static"}
        parseIncompleteMarkdown
        components={{
          h1: ({ ...props }) => (
            <h1
              className="mb-3 mt-5 border-b border-primary/20 pb-1.5 text-xl font-bold tracking-tight text-primary first:mt-0"
              {...props}
            />
          ),
          h2: ({ ...props }) => (
            <h2
              className="mb-2.5 mt-4 text-lg font-semibold tracking-tight text-primary first:mt-0"
              {...props}
            />
          ),
          h3: ({ ...props }) => (
            <h3
              className="mb-2 mt-3 border-l-2 border-accent pl-3 text-base font-semibold text-foreground first:mt-0"
              {...props}
            />
          ),
          h4: ({ ...props }) => (
            <h4 className="mb-1.5 mt-2.5 text-sm font-semibold text-foreground/90 first:mt-0" {...props} />
          ),
          p: ({ ...props }) => <p className="my-2 leading-relaxed" {...props} />,
          ul: ({ ...props }) => <ul className="my-2 list-disc space-y-1.5 pl-5" {...props} />,
          ol: ({ ...props }) => <ol className="my-2 list-decimal space-y-1.5 pl-5" {...props} />,
          li: ({ ...props }) => <li className="pl-0.5 leading-relaxed" {...props} />,
          blockquote: ({ ...props }) => (
            <blockquote
              className="my-3 rounded-r-md border-l-2 border-accent bg-accent/5 py-2 pl-3 pr-2 text-foreground/80"
              {...props}
            />
          ),
          strong: ({ ...props }) => <strong className="font-semibold text-primary/90" {...props} />,
          table: ({ ...props }) => (
            <div className="my-3 overflow-x-auto rounded-lg border border-border">
              <table className="w-full table-auto border-collapse text-xs" {...props} />
            </div>
          ),
          thead: ({ ...props }) => <thead className="bg-primary/8" {...props} />,
          th: ({ ...props }) => (
            <th
              className="border-b border-border bg-primary/10 px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-primary whitespace-normal break-words"
              {...props}
            />
          ),
          td: ({ ...props }) => (
            <td className="border-b border-border/50 px-3 py-2 align-top whitespace-normal break-words" {...props} />
          ),
          code: ({ className, children, ...props }) => {
            const textContent = Array.isArray(children) ? children.join("") : String(children ?? "");
            const isBlock = Boolean(className?.includes("language-")) || textContent.includes("\n");
            if (isBlock) {
              return (
                <code
                  className={cn(
                    "block overflow-x-auto rounded-lg border border-border/60 bg-secondary/60 p-3 font-mono text-xs",
                    className
                  )}
                  {...props}
                >
                  {children}
                </code>
              );
            }
            return (
              <code
                className={cn("rounded-md bg-primary/8 px-1.5 py-0.5 font-mono text-[0.82em] text-primary/90", className)}
                {...props}
              >
                {children}
              </code>
            );
          },
          inlineCode: ({ className, ...props }) => (
            <code
              className={cn("rounded-md bg-primary/8 px-1.5 py-0.5 font-mono text-[0.82em] text-primary/90", className)}
              {...props}
            />
          ),
          pre: ({ ...props }) => <pre className="my-3" {...props} />
        }}
      >
        {renderable}
      </Streamdown>
      {streaming && stableParts.tail ? (
        <p className="my-2 whitespace-pre-wrap text-foreground/85">{sanitizeStreamingTail(stableParts.tail)}</p>
      ) : null}
    </div>
  );
}
