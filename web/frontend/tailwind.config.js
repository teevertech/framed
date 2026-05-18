import defaultTheme from "tailwindcss/defaultTheme";

/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", ...defaultTheme.fontFamily.sans],
      },
      colors: {
        // ── Semantic UI shell ──────────────────────────────────────────
        // Values resolve through CSS custom properties in index.css.
        // To retheme the entire app, edit only :root / .dark in index.css.
        // No component file needs to change.
        "c-base":         "var(--c-base)",        // page / app shell bg
        "c-surface":      "var(--c-surface)",     // cards, panels, popovers
        "c-subtle":       "var(--c-subtle)",      // inputs, controls bg
        "c-border":       "var(--c-border)",      // card & input borders
        "c-divider":      "var(--c-divider)",     // inner rules, row separators
        "c-text-1":       "var(--c-text-1)",      // headings, primary labels
        "c-text-2":       "var(--c-text-2)",      // body, secondary labels
        "c-text-3":       "var(--c-text-3)",      // muted, placeholders
        "c-action":       "var(--c-action)",      // primary button bg
        "c-action-hover": "var(--c-action-hover)",
        "c-action-fg":    "var(--c-action-fg)",   // primary button text

        // ── Physical / data colors ────────────────────────────────────
        // These represent real materials and robot physics.
        // Intentionally NOT in the theme system — they stay the sam/** @type {import('tailwindcss').Config} */
        export default {
          darkMode: "class",
          content: ["./index.html", "./src/**/*.{ts,tsx}"],
          theme: {
            extend: {
              fontFamily: {
                // Inline fallbacks — no defaultTheme import needed.
                sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
              },
              colors: {
                // ── Semantic UI shell ──────────────────────────────────────────
                // Values resolve through CSS custom properties in index.css.
                // To retheme the entire app, edit only :root / .dark in index.css.
                // No component file needs to change.
                "c-base":         "var(--c-base)",        // page / app shell bg
                "c-surface":      "var(--c-surface)",     // cards, panels, popovers
                "c-subtle":       "var(--c-subtle)",      // inputs, controls bg
                "c-border":       "var(--c-border)",      // card & input borders
                "c-divider":      "var(--c-divider)",     // inner rules, row separators
                "c-text-1":       "var(--c-text-1)",      // headings, primary labels
                "c-text-2":       "var(--c-text-2)",      // body, secondary labels
                "c-text-3":       "var(--c-text-3)",      // muted, placeholders
                "c-action":       "var(--c-action)",      // primary button bg
                "c-action-hover": "var(--c-action-hover)",
                "c-action-fg":    "var(--c-action-fg)",   // primary button text

                // ── Physical / data colors ────────────────────────────────────
                // These represent real materials and robot physics.
                // Intentionally NOT in the theme system — they stay the same
                // in both light and dark mode.
                lumber: {
                  plate:   "#C4A265",
                  stud:    "#DEB887",
                  king:    "#CD853F",
                  jack:    "#D2691E",
                  header:  "#7B5E2A",
                  sill:    "#B8895A",
                  cripple: "#E8C99A",
                },
                panel: {
                  bg:           "#F7F4EE",
                  outline:      "#BDBDBD",
                  ghost:        "#EDEDED",
                  "ghost-edge": "#C0C0C0",
                },
                path: {
                  clear:   "#43A047",
                  collide: "#E53935",
                },
                robot: "#1565C0",
              },
            },
          },
          plugins: [],
        };e
        // in both light and dark mode.
        lumber: {
          plate:   "#C4A265",
          stud:    "#DEB887",
          king:    "#CD853F",
          jack:    "#D2691E",
          header:  "#7B5E2A",
          sill:    "#B8895A",
          cripple: "#E8C99A",
        },
        panel: {
          bg:           "#F7F4EE",
          outline:      "#BDBDBD",
          ghost:        "#EDEDED",
          "ghost-edge": "#C0C0C0",
        },
        path: {
          clear:   "#43A047",
          collide: "#E53935",
        },
        robot: "#1565C0",
      },
    },
  },
  plugins: [],
};
