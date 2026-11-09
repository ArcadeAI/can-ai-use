// The canaiuse.com Worker: /mcp and /feedback are answered here, everything else is the static site.

import {Catalog} from "./catalog.ts";
import {type FeedbackDb, handleFeedback} from "./feedback.ts";
import {handleMcp} from "./mcp.ts";

export interface Env {
  ASSETS: {fetch(request: Request): Promise<Response>};
  /** D1, private reader feedback (worker/migrations). */
  FEEDBACK?: FeedbackDb;
}

// The deployed site's own data files, read through the assets binding so the server
// always answers from the data that is live, cached per isolate for ten minutes.
let catalog: Catalog | undefined;
const catalogFor = (env: Env): Catalog =>
  (catalog ??= new Catalog(async (path) => {
    const response = await env.ASSETS.fetch(new Request(`https://canaiuse.com/data/${path}`));
    if (!response.ok) throw new Error(`data/${path} is not deployed (${response.status})`);
    return response.json();
  }));

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const {pathname} = new URL(request.url);
    if (pathname === "/mcp" || pathname === "/mcp/") return handleMcp(request, catalogFor(env));
    if (pathname === "/feedback") return handleFeedback(request, env.FEEDBACK);
    // The site routes by hash; a typed or shared /harnesses lands on the agents page.
    if (pathname === "/harnesses" || pathname === "/harnesses/") {
      return Response.redirect(new URL("/#/agents", request.url).toString(), 301);
    }
    return env.ASSETS.fetch(request);
  },
};
