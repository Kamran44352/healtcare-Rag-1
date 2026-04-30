"use client";

import { motion } from "framer-motion";
import type { ReactNode } from "react";

type BlurFadeProps = {
  children: ReactNode;
  delay?: number;
  className?: string;
};

export function BlurFade({ children, delay = 0, className }: BlurFadeProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8, filter: "blur(8px)" }}
      animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      transition={{ duration: 0.4, ease: "easeOut", delay }}
      className={className}
    >
      {children}
    </motion.div>
  );
}
