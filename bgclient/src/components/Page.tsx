import Router from "next/router";
import React from "react";

// Track client-side page views with Segment
Router.events.on("routeChangeComplete", (url) => {
  (window as any).analytics.page(url);
});

const Page = ({ children }) => <div>{children}</div>;

export default Page;
