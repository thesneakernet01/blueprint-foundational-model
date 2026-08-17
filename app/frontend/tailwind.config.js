/** @type {import('tailwindcss').Config} */
// Palette matched to the Cloudera brand deck (demo-design-template repo,
// 2025_Cloudera_new_brand_TOOLKIT_PPT.pptx theme): orange #FF550C accent on
// indigo #100045/#110046 surfaces, blue #5555F9/#8789FB secondary.
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'source-code-pro', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
      colors: {
        'surface-0': '#0A0328',   // page bg — indigo-hued near-black
        'surface-1': '#110046',   // header / dialogs — exact brand indigo
        'surface-2': '#1B1157',   // cards
        'surface-3': '#271C68',   // borders / buttons
        'surface-4': '#332979',   // hover / faint rules
        accent: '#FF550C',        // Cloudera orange
        'accent-dim': '#B93D05',
        'status-red': '#ef4444',
        'status-red-dim': '#991b1b',
        'status-amber': '#f59e0b',
        'status-amber-dim': '#92400e',
        'status-green': '#10b981',
        'status-green-dim': '#065f46',
        'status-purple': '#8789FB',      // Cloudera light blue (nexus chips)
        'status-purple-dim': '#3F41B8',
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
