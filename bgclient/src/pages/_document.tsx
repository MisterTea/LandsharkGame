import * as snippet from "@segment/snippet";
import Document, { Head, Html, Main, NextScript } from "next/document";

const {
  ANALYTICS_WRITE_KEY = "SRtsam0l01B8s01HAqWVNZqCOGY43AzA",
  NODE_ENV = "development",
} = process.env;

export default class MyDocument extends Document {
  renderSnippet() {
    const opts = {
      apiKey: ANALYTICS_WRITE_KEY,
      // note: the page option only covers SSR tracking.
      // Page.js is used to track other events using `window.analytics.page()`
      page: true,
    };

    if (NODE_ENV === "development") {
      return snippet.max(opts);
    }

    return snippet.min(opts);
  }

  render() {
    return (
      <Html>
        <Head>
          {/* Inject the Segment snippet into the <head> of the document  */}
          <script dangerouslySetInnerHTML={{ __html: this.renderSnippet() }} />
        </Head>
        <body>
          <Main />
          <NextScript />
        </body>
      </Html>
    );
  }
}
