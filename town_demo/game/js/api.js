// Thin JSON client for town_demo/server.py. Non-2xx responses throw Error(body.error).

async function call(method, path, body) {
  const init = { method, headers: {} };
  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const res = await fetch(path, init);
  let data;
  try {
    data = await res.json();
  } catch {
    throw new Error(`${method} ${path}: HTTP ${res.status} (no JSON body)`);
  }
  if (!res.ok) throw new Error(data.error || `${method} ${path}: HTTP ${res.status}`);
  return data;
}

export const state = () => call("GET", "/api/state");
export const newGame = (name, avatar) => call("POST", "/api/new", { name, avatar });
export const step = (player) => call("POST", "/api/step", { player });
export const choose = (i) => call("POST", "/api/choose", { choice: i });
export const talkOptions = (npc) => call("GET", `/api/talk-options?npc=${encodeURIComponent(npc)}`);
export const talk = (npc, topic) => call("POST", "/api/talk", { npc, topic });
export const mind = (npc) => call("GET", `/api/mind?npc=${encodeURIComponent(npc)}`);
export const tools = () => call("GET", "/api/tools");
export const useTool = (tool, helper) => call("POST", "/api/use-tool", { tool, helper });
export const help = (useId) => call("POST", "/api/help", { use_id: useId });
export const site = () => call("GET", "/api/site");
export const build = () => call("POST", "/api/build", {});
