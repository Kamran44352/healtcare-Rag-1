"use client";

import type { FormEvent } from "react";
import { Send } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

type PromptInputProps = {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => Promise<void> | void;
  disabled?: boolean;
  placeholder?: string;
};

export function PromptInput({ value, onChange, onSubmit, disabled, placeholder }: PromptInputProps) {
  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (disabled || !value.trim()) return;
    await onSubmit();
  };

  return (
    <form onSubmit={handleSubmit} className="relative flex items-end gap-2">
      <Textarea
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder ?? "Ask a healthcare guideline question..."}
        className="min-h-[52px] max-h-[180px] flex-1 resize-none rounded-xl border-border/70 bg-white pr-14 text-sm shadow-sm transition-shadow focus:shadow-md focus-visible:ring-1 focus-visible:ring-accent"
        onKeyDown={async (event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            if (!disabled && value.trim()) {
              await onSubmit();
            }
          }
        }}
      />
      <Button
        type="submit"
        disabled={disabled || !value.trim()}
        size="icon"
        className="absolute bottom-2 right-2 h-8 w-8 shrink-0 rounded-lg bg-accent text-white shadow-sm hover:bg-accent/90 disabled:opacity-40"
      >
        <Send className="h-3.5 w-3.5" />
      </Button>
    </form>
  );
}
