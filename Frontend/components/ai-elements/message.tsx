import type { PropsWithChildren } from "react";

import { cn } from "@/lib/utils";

type MessageProps = PropsWithChildren<{
  role: "user" | "assistant";
  className?: string;
}>;

export function Message({ role, children, className }: MessageProps) {
  if (role === "user") {
    return (
      <div className={cn("flex w-full justify-end gap-3", className)}>
        <div className="max-w-[78%] rounded-2xl rounded-tr-sm bg-primary px-4 py-3 text-sm leading-relaxed text-white shadow-sm [&_*]:text-white">
          {children}
        </div>
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/15 text-xs font-bold text-primary">
          You
        </div>
      </div>
    );
  }

  return (
    <div className={cn("flex w-full justify-start gap-3", className)}>
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-primary to-primary/70 shadow-sm">
        <span className="text-xs font-bold text-white">C</span>
      </div>
      <div className="max-w-[85%] rounded-2xl rounded-tl-sm border border-border/60 bg-card px-4 py-3 text-sm leading-relaxed shadow-sm">
        {children}
      </div>
    </div>
  );
}
