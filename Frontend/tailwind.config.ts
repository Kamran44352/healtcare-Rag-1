import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        border: "hsl(217 28% 78%)",
        input: "hsl(217 28% 78%)",
        ring: "hsl(24 88% 42%)",
        background: "hsl(220 30% 98%)",
        foreground: "hsl(220 58% 14%)",
        primary: {
          DEFAULT: "hsl(220 74% 31%)",
          foreground: "hsl(0 0% 100%)"
        },
        secondary: {
          DEFAULT: "hsl(216 38% 92%)",
          foreground: "hsl(220 58% 14%)"
        },
        muted: {
          DEFAULT: "hsl(216 30% 88%)",
          foreground: "hsl(220 23% 40%)"
        },
        accent: {
          DEFAULT: "hsl(24 88% 42%)",
          foreground: "hsl(0 0% 100%)"
        },
        card: {
          DEFAULT: "hsl(0 0% 100%)",
          foreground: "hsl(220 58% 14%)"
        },
        destructive: {
          DEFAULT: "hsl(0 72% 51%)",
          foreground: "hsl(0 0% 100%)"
        }
      },
      borderRadius: {
        lg: "1rem",
        md: "0.75rem",
        sm: "0.5rem"
      },
      boxShadow: {
        soft: "0 14px 36px rgba(16, 36, 77, 0.12)"
      },
      fontFamily: {
        sans: ["var(--font-sora)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"]
      },
      keyframes: {
        "fade-up": {
          from: { opacity: "0", transform: "translateY(8px)" },
          to: { opacity: "1", transform: "translateY(0)" }
        }
      },
      animation: {
        "fade-up": "fade-up 0.28s ease-out"
      }
    }
  },
  plugins: []
};

export default config;
