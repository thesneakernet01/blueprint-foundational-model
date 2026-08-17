/** @type {import('tailwindcss').Config} */
// Palette matched to the Cloudera brand deck (demo-design-template repo,
// 2025_Cloudera_new_brand_TOOLKIT_PPT.pptx theme) — LIGHT, like the body
// slides: white/#F5F5FA surfaces, near-black #1A1A26 text, orange #FF550C
// accent, blue #5555F9/#8789FB secondary, indigo reserved for emphasis.
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'source-code-pro', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
      colors: {
        'surface-0': '#F5F5FA',   // page bg — the deck's SOFT_BG
        'surface-1': '#FFFFFF',   // header / dialogs / chrome
        'surface-2': '#FFFFFF',   // cards (border separates from page)
        'surface-3': '#EBEBF3',   // hover bg / secondary buttons / dividers
        'surface-4': '#D9D9E8',   // input borders / faint rules
        accent: '#FF550C',        // Cloudera orange
        'accent-dim': '#B93D05',
        'status-red': '#ef4444',
        'status-red-dim': '#991b1b',
        'status-amber': '#f59e0b',
        'status-amber-dim': '#92400e',
        'status-green': '#10b981',
        'status-green-dim': '#065f46',
        'status-purple': '#5555F9',      // Cloudera blue (nexus chips, on light)
        'status-purple-dim': '#26177B',
        'brand-blue': '#5555F9',
        'brand-blue-soft': '#8789FB',
        'brand-gray': '#A8AFB9',
        ink: '#1A1A26',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-slide-in': 'fadeSlideIn 0.3s ease-out',
        'pulse-glow': 'pulseGlow 2s ease-in-out infinite',
        'pulse-glow-accent': 'pulseGlowAccent 2s ease-in-out infinite',
      },
      keyframes: {
        fadeSlideIn: {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        pulseGlow: {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(239, 68, 68, 0.3)' },
          '50%': { boxShadow: '0 0 12px 4px rgba(239, 68, 68, 0.15)' },
        },
        pulseGlowAccent: {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(255, 85, 12, 0.35)' },
          '50%': { boxShadow: '0 0 12px 4px rgba(255, 85, 12, 0.18)' },
        },
      },
    },
  },
  plugins: [],
};
