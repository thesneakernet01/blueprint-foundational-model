/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'source-code-pro', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
      colors: {
        'surface-0': '#0a0a0f',
        'surface-1': '#12121a',
        'surface-2': '#1a1a25',
        'surface-3': '#242430',
        'surface-4': '#2e2e3a',
        accent: '#6366f1',
        'accent-dim': '#4338ca',
        'status-red': '#ef4444',
        'status-red-dim': '#991b1b',
        'status-amber': '#f59e0b',
        'status-amber-dim': '#92400e',
        'status-green': '#10b981',
        'status-green-dim': '#065f46',
        'status-purple': '#a78bfa',
        'status-purple-dim': '#5b21b6',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-slide-in': 'fadeSlideIn 0.3s ease-out',
        'pulse-glow': 'pulseGlow 2s ease-in-out infinite',
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
      },
    },
  },
  plugins: [],
};
