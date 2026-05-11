/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'soft-mint': '#a2d5c6', // Warna medis modern sesuai blueprint Anda
      }
    },
  },
  plugins: [],
}