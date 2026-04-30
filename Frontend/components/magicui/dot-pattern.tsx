import { cn } from "@/lib/utils";

type DotPatternProps = {
  className?: string;
};

export function DotPattern({ className }: DotPatternProps) {
  return (
    <div
      className={cn("pointer-events-none absolute inset-0 -z-10 opacity-45", className)}
      style={{
        backgroundImage:
          "radial-gradient(circle at 1px 1px, rgba(22,52,112,.22) 1px, transparent 0)",
        backgroundSize: "22px 22px"
      }}
      aria-hidden
    />
  );
}
