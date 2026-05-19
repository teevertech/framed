/**
 * Theme tokens — the single source of truth for the app's color palette.
 *
 * Edit values here to retune the visual theme. The keys are CSS custom
 * property names (without the `--` prefix). At runtime, App.tsx writes
 * the appropriate set onto <html> via setProperty(), and components
 * reference them through Tailwind utilities like `bg-c-surface`,
 * `text-c-text-1`, `border-c-border`, etc.
 */

export type ThemeTokens = Record<string, string>;

export const lightTheme: ThemeTokens = {
  // Backgrounds
  "--c-base":         "#f9fafb",  // page / app shell
  "--c-surface":      "#ffffff",  // cards, panels, popovers
  "--c-subtle":       "#f3f4f6",  // inputs, controls
  // Borders
  "--c-border":       "#e5e7eb",  // card and input outlines
  "--c-divider":      "#f3f4f6",  // inner rules, row separators
  // Text
  "--c-text-1":       "#111827",  // headings, primary labels
  "--c-text-2":       "#4b5563",  // body, secondary labels
  "--c-text-3":       "#9ca3af",  // muted, placeholders
  // Primary action button (inverts in dark mode)
  "--c-action":       "#111827",
  "--c-action-hover": "#374151",
  "--c-action-fg":    "#ffffff",
};

export const darkTheme: ThemeTokens = {
  "--c-base":         "#030712",
  "--c-surface":      "#111827",
  "--c-subtle":       "#1f2937",
  "--c-border":       "#374151",
  "--c-divider":      "#1f2937",
  "--c-text-1":       "#f9fafb",
  "--c-text-2":       "#9ca3af",
  "--c-text-3":       "#6b7280",
  "--c-action":       "#f3f4f6",
  "--c-action-hover": "#e5e7eb",
  "--c-action-fg":    "#111827",
};

/** Apply a theme by writing every token onto <html>'s inline style. */
export function applyTheme(tokens: ThemeTokens) {
  const root = document.documentElement;
  for (const [name, value] of Object.entries(tokens)) {
    root.style.setProperty(name, value);
  }
}
