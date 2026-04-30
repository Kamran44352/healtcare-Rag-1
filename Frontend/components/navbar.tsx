"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { Menu, X, Settings, Mail, FileText, ChevronRight } from "lucide-react";

const menuItems = [
  { label: "Admin", href: "/admin", icon: Settings, description: "Manage documents & settings" },
  { label: "Contact", href: "/contact", icon: Mail, description: "Get in touch with us" },
  { label: "Terms", href: "/terms", icon: FileText, description: "Terms & conditions" },
];

export function Navbar() {
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  return (
    <header className="sticky top-0 z-50 border-b border-border/60 bg-white/95 shadow-sm backdrop-blur supports-[backdrop-filter]:bg-white/85">
      <div className="mx-auto flex h-16 max-w-screen-xl items-center justify-between px-4 sm:px-6 lg:px-10">
        {/* Logo */}
        <Link href="/" className="flex select-none items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-primary/80 shadow-sm">
            <span className="text-sm font-bold text-white">C</span>
          </span>
          <span className="text-xl font-bold tracking-tight">
            <span className="text-primary">Clin</span>
            <span className="text-accent">tel</span>
          </span>
        </Link>

        {/* Burger */}
        <div ref={menuRef} className="relative">
          <button
            type="button"
            onClick={() => setOpen((prev) => !prev)}
            aria-label="Open menu"
            className="flex h-9 w-9 items-center justify-center rounded-lg text-foreground/70 transition-colors hover:bg-primary/8 hover:text-primary"
          >
            {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>

          {open && (
            <div className="absolute right-0 top-full mt-3 w-64 rounded-2xl border border-border/50 bg-white shadow-2xl ring-1 ring-black/5">
              {/* Items */}
              <div className="p-2">
                {menuItems.map(({ label, href, icon: Icon, description }) => (
                  <Link
                    key={href}
                    href={href}
                    onClick={() => setOpen(false)}
                    className="group flex items-center gap-3 rounded-xl px-3 py-2.5 transition-colors hover:bg-primary/6"
                  >
                    <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-primary/15 bg-primary/6 text-primary transition-colors group-hover:bg-primary group-hover:text-white">
                      <Icon className="h-4 w-4" />
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-medium text-foreground">{label}</p>
                      <p className="truncate text-[11px] text-muted-foreground">{description}</p>
                    </div>
                    <ChevronRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground/40 transition-transform group-hover:translate-x-0.5 group-hover:text-primary" />
                  </Link>
                ))}
              </div>

              {/* Footer */}
              <div className="rounded-b-2xl border-t border-border/50 px-4 py-2.5">
                <p className="text-center text-[10px] text-muted-foreground/60">
                  NICE guidelines at your fingertips
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
