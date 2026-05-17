/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        lumber: {
          plate: "#C4A265",
          stud: "#DEB887",
          king: "#CD853F",
          jack: "#D2691E",
          header: "#7B5E2A",
          sill: "#B8895A",
          cripple: "#E8C99A",
        },
        panel: {
          bg: "#F7F4EE",
          outline: "#BDBDBD",
          ghost: "#EDEDED",
          "ghost-edge": "#C0C0C0",
        },
        path: {
          clear: "#43A047",
          collide: "#E53935",
        },
        robot: "#1565C0",
      },
    },
  },
  plugins: [],
};
