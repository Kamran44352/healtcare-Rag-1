import type { PropsWithChildren } from "react";

import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

type ConversationProps = PropsWithChildren<{
  className?: string;
}>;

export function Conversation({ children, className }: ConversationProps) {
  return (
    <ScrollArea className={cn("h-[56vh] rounded-lg border border-border bg-card/85 p-4", className)}>
      <div className="space-y-4 pr-3">{children}</div>
    </ScrollArea>
  );
}
