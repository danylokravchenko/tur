import { HtmlBasePlugin } from "@11ty/eleventy";

export default function (eleventyConfig) {
  eleventyConfig.addPlugin(HtmlBasePlugin);
  eleventyConfig.addPassthroughCopy({ "src/assets": "assets" });

  // Fonts ship with the site, so a visit makes no request to a third party.
  const fonts = {
    "@fontsource/bowlby-one/files/bowlby-one-latin-400-normal.woff2": "bowlby-one-400.woff2",
    "@fontsource/ibm-plex-sans/files/ibm-plex-sans-latin-400-normal.woff2": "plex-400.woff2",
    "@fontsource/ibm-plex-sans/files/ibm-plex-sans-latin-400-italic.woff2": "plex-400-italic.woff2",
    "@fontsource/ibm-plex-sans/files/ibm-plex-sans-latin-600-normal.woff2": "plex-600.woff2",
    "@fontsource/jetbrains-mono/files/jetbrains-mono-latin-400-normal.woff2": "mono-400.woff2",
    "@fontsource/jetbrains-mono/files/jetbrains-mono-latin-700-normal.woff2": "mono-700.woff2",
  };
  for (const [from, to] of Object.entries(fonts)) {
    eleventyConfig.addPassthroughCopy({ [`node_modules/${from}`]: `assets/fonts/${to}` });
  }

  // The book and README live outside the input directory.
  eleventyConfig.addWatchTarget("../docs/src/");
  eleventyConfig.addWatchTarget("../README.md");
  eleventyConfig.addWatchTarget("./lib/");
}

export const config = {
  dir: { input: "src", includes: "_includes", data: "_data", output: "_site" },
  htmlTemplateEngine: "njk",
  // GitHub Pages serves a project site under the repository's name.
  pathPrefix: "/tur/",
};
