"use client";

import { motion } from "framer-motion";

type ThinkingProps = {
  label?: string;
};

export function Thinking({ label = "Thinking" }: ThinkingProps) {
  return (
    <div className="space-y-2">
      <p className="text-xs font-medium text-muted-foreground">{label}</p>
      <div className="flex items-center gap-1.5">
        {[0, 1, 2].map((idx) => (
          <motion.span
            key={idx}
            className="h-2 w-2 rounded-full bg-primary/70"
            animate={{ y: [0, -4, 0], opacity: [0.45, 1, 0.45] }}
            transition={{
              duration: 0.9,
              repeat: Infinity,
              delay: idx * 0.12,
              ease: "easeInOut"
            }}
          />
        ))}
      </div>
    </div>
  );
}
