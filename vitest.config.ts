import { resolve } from "node:path";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: {
      "@warlock.js/ai": resolve(__dirname, "../ai/src/index.ts"),
      "@warlock.js/cache": resolve(__dirname, "../cache/src/index.ts"),
      "@warlock.js/logger": resolve(__dirname, "../logger/src/index.ts"),
      "@warlock.js/fs": resolve(__dirname, "../fs/src/index.ts"),
    },
  },
  test: {
    environment: "node",
    globals: false,
    include: ["src/**/*.spec.ts"],
  },
});
