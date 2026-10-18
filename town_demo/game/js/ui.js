// Dialog box, menus, toasts, HUD and full-screen sheets. All dynamic text goes through textContent.
//
// Input model: modal UI pushes a handler onto a stack; the top handler receives every button press
// ("up" | "down" | "left" | "right" | "a" | "b"). An empty stack means the overworld has control, and
// any non-empty stack pauses the town clock.

import { frontPortrait } from "./sprites.js";

const $ = (sel) => document.querySelector(sel);
const stack = [];

export function pushInput(fn) {
  stack.push(fn);
  return () => {
    const i = stack.lastIndexOf(fn);
    if (i >= 0) stack.splice(i, 1);
  };
}

export const hasModal = () => stack.length > 0;

export function dispatch(btn) {
  const top = stack[stack.length - 1];
  if (!top) return false;
  top(btn);
  return true;
}

export function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

// ---------- dialog ----------
const CHARS_PER_SECOND = 40;

export function hideDialog() {
  $("#dialog").classList.add("hidden");
}

// Types `text` into the dialog box; A/B completes it. Resolves once fully shown; the box stays open.
// `sprite`: a speaker portrait shown at the left of the box.
export function show(text, { sprite } = {}) {
  const dialog = $("#dialog");
  dialog.classList.remove("hidden");
  dialog.classList.toggle("with-portrait", Boolean(sprite));
  dialog.replaceChildren();
  if (sprite) {
    const portrait = el("div", "portrait");
    frontPortrait(portrait, sprite, 1.5);
    dialog.append(portrait);
  }
  const span = el("span");
  dialog.append(span);
  const chars = [...text];
  return new Promise((resolve) => {
    let i = 0;
    let timer = null;
    const finish = () => {
      clearInterval(timer);
      span.textContent = text;
      pop();
      resolve();
    };
    const pop = pushInput((btn) => {
      if (btn === "a" || btn === "b") finish();
    });
    if (!chars.length) return finish();
    timer = setInterval(() => {
      i += 1;
      span.textContent = chars.slice(0, i).join("");
      if (i >= chars.length) finish();
    }, 1000 / CHARS_PER_SECOND);
  });
}

export function waitA() {
  return new Promise((resolve) => {
    const pop = pushInput((btn) => {
      if (btn === "a" || btn === "b") {
        pop();
        resolve();
      }
    });
  });
}

// Types `text`, waits for A, closes the box.
export async function say(text, { sprite } = {}) {
  await show(text, { sprite });
  $("#dialog").append(el("span", "more", "▼"));
  await waitA();
  hideDialog();
}

// ---------- menus ----------
// options: strings or {label, cls?}. Resolves with the index, or null on B (unless cancel === false).
// opts.disabledReasons[i]: picking i says the reason and keeps the menu open.
// opts.wide: a full-width menu for long option texts.
export function choose(options, opts = {}) {
  const menu = $("#menu");
  const items = options.map((o) => (typeof o === "string" ? { label: o } : o));
  let cur = Math.min(opts.initial ?? 0, items.length - 1);
  let busy = false;

  const render = () => {
    menu.replaceChildren();
    if (opts.title) menu.append(el("div", "title", opts.title));
    items.forEach((item, i) => {
      const row = el("div", "row");
      if (i === cur) row.classList.add("cur");
      if (opts.disabledReasons?.[i]) row.classList.add("disabled");
      if (item.cls) row.classList.add(item.cls);
      row.textContent = item.label;
      menu.append(row);
    });
  };

  menu.className = `box${opts.wide ? " wide" : ""}`;
  render();
  return new Promise((resolve) => {
    const done = (value) => {
      pop();
      menu.classList.add("hidden");
      resolve(value);
    };
    const pop = pushInput(async (btn) => {
      if (busy) return;
      if (btn === "up") cur = (cur + items.length - 1) % items.length;
      else if (btn === "down") cur = (cur + 1) % items.length;
      else if (btn === "a") {
        const reason = opts.disabledReasons?.[cur];
        if (!reason) return done(cur);
        busy = true;
        await say(reason);
        busy = false;
      } else if (btn === "b" && opts.cancel !== false) return done(null);
      render();
    });
  });
}

// Name entry: a real <input> inside the dialog box; Enter confirms, empty means "Alex".
export function askName(prompt) {
  const dialog = $("#dialog");
  dialog.classList.remove("hidden");
  dialog.replaceChildren(el("span", null, prompt));
  const input = el("input");
  input.maxLength = 12;
  input.autocomplete = "off";
  input.spellcheck = false;
  dialog.append(input);
  const pop = pushInput(() => {});
  const keepFocus = () => setTimeout(() => input.isConnected && input.focus(), 0);
  input.addEventListener("blur", keepFocus);
  input.focus();
  return new Promise((resolve) => {
    input.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key !== "Enter") return;
      e.preventDefault();
      input.removeEventListener("blur", keepFocus);
      pop();
      hideDialog();
      resolve(input.value.trim().replace(/\s+/g, " ") || "Alex");
    });
  });
}

// ---------- toast & HUD ----------
export function toast(text) {
  const box = $("#toast");
  const node = el("div", null, text);
  box.append(node);
  while (box.children.length > 4) box.firstElementChild.remove();
  setTimeout(() => node.remove(), 4000);
}

export function hud({ clock, area, status }) {
  $("#hud").classList.remove("hidden");
  $("#hud-clock").textContent = clock;
  $("#hud-area").textContent = area;
  $("#hud-status").textContent = status;
}

// ---------- sheets ----------
export function openSheet() {
  const sheet = $("#sheet");
  sheet.replaceChildren();
  sheet.classList.remove("hidden");
  return sheet;
}

export function closeSheet() {
  $("#sheet").classList.add("hidden");
}

// Scrollable list on the sheet. items: [{text, cls?}]. onA(i) may be async; B closes and resolves.
export function browse({ title, items, hint, onA, pageSize = 15 }) {
  let cur = 0;
  let busy = false;
  const render = () => {
    const sheet = openSheet();
    sheet.append(el("h2", null, title));
    const list = el("div", "list");
    if (!items.length) list.append(el("div", "muted", "Nothing yet."));
    const start = Math.max(0, Math.min(cur - Math.floor(pageSize / 2), items.length - pageSize));
    items.slice(start, start + pageSize).forEach((item, k) => {
      const row = el("div", `item${item.cls ? ` ${item.cls}` : ""}`, item.text);
      if (onA && start + k === cur) row.classList.add("cur");
      list.append(row);
    });
    sheet.append(list, el("div", "hint", hint));
  };
  render();
  return new Promise((resolve) => {
    const pop = pushInput(async (btn) => {
      if (busy) return;
      if (btn === "b") {
        pop();
        closeSheet();
        return resolve();
      }
      if (btn === "up" && cur > 0) cur -= 1;
      else if (btn === "down" && cur < items.length - 1) cur += 1;
      else if (btn === "a" && onA && items.length) {
        busy = true;
        closeSheet();
        await onA(cur);
        busy = false;
      }
      render();
    });
  });
}
