// 在原网页完成验证与解析，通过轮询返回资源，不向网页暴露安卓调用接口。
(() => {
    if (window.top !== window) return;
    if (window.__xhsSaver) return;

    let generation = 0;
    let state = { kind: "idle", message: "正在准备解析网页。", resources: [], text: "" };
    const downloads = [];
    const originalFetch = window.fetch;

    const secureUrl = (value) => {
        if (typeof value !== "string" || !value.trim()) return "";
        try {
            const url = new URL(String(value || ""), location.href);
            return url.protocol === "https:" && !url.username && !url.password ? url.href : "";
        } catch {
            return "";
        }
    };

    const acceptResult = (payload, requestGeneration) => {
        if (requestGeneration !== generation) return;
        if (payload.code !== 200) {
            const waiting = [401, 403, 601].includes(payload.code);
            state = {
                kind: waiting ? "waiting" : "error",
                message: waiting ? "请在解析网页中完成验证或登录。" :
                    String(payload.msg || "网页未能解析该链接，请在网页中重试。"),
                resources: [], text: ""
            };
            return;
        }

        const data = payload.data || {};
        const proxy = secureUrl(data.webProxyUrl).replace(/\/+$/, "");
        const resources = [];
        for (const [field, kind, label, hashFields] of [
            ["videoList", "video", "视频", ["webVideoHash", "videoHash"]],
            ["imgList", "image", "图片", ["webImgHash", "imgHash"]]
        ]) {
            const entries = Array.isArray(data[field]) ? data[field] : [];
            entries.slice(0, 100).forEach((entry, index) => {
                const raw = secureUrl(entry.url);
                if (!raw) return;
                const hash = entry[hashFields[0]] || entry[hashFields[1]];
                let url = raw;
                if (proxy && hash && data.timestamp != null) {
                    url = `${proxy}/res/dl?url=${encodeURIComponent(raw)}` +
                        `&hash=${encodeURIComponent(hash)}&timestamp=${encodeURIComponent(data.timestamp)}`;
                }
                resources.push({ url, kind, name: `${label} ${index + 1}` });
            });
        }
        const text = (Array.isArray(data.descList) ? data.descList : [])
            .filter(value => typeof value === "string").join("\n").slice(0, 20000);
        state = resources.length || text ? {
            kind: "result", message: "解析完成。", resources, text
        } : {
            kind: "error", message: "网页没有返回可保存的图片或视频，请查看网页提示。",
            resources: [], text: ""
        };
    };

    window.fetch = async function (...args) {
        const address = typeof args[0] === "string" ? args[0] : args[0]?.url;
        let isParse = false;
        try {
            const url = new URL(address, location.href);
            isParse = url.hostname === "api.videotool.cc" && url.pathname === "/tapsave/api/v3/parse";
        } catch { /* 与下载无关的请求保持原行为。 */ }
        const requestGeneration = generation;
        try {
            const response = await Reflect.apply(originalFetch, this, args);
            if (isParse) {
                response.clone().json().then(payload => acceptResult(payload, requestGeneration))
                    .catch(() => {
                        if (generation === requestGeneration) {
                            state = { kind: "error", message: "解析服务返回了无法识别的内容，请在网页中重试。", resources: [], text: "" };
                        }
                    });
            }
            return response;
        } catch (error) {
            if (isParse && generation === requestGeneration) {
                state = { kind: "error", message: "解析请求未能完成，请检查网络后重试。", resources: [], text: "" };
            }
            throw error;
        }
    };

    // 原网页用临时链接发起下载；只接管明确标记为下载的点击。
    document.addEventListener("click", event => {
        const anchor = event.target instanceof Element ? event.target.closest("a[href]") : null;
        if (!anchor) return;
        const url = secureUrl(anchor.href);
        if (!url) return;
        const parsed = new URL(url);
        if (!anchor.hasAttribute("download") && !parsed.searchParams.has("downloadFilename")) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        const filename = anchor.download || parsed.searchParams.get("downloadFilename") || "";
        const kind = /\.(mp4|webm|mov)$/i.test(filename) ? "video" :
            /\.(jpe?g|png|webp|gif|heic)$/i.test(filename) ? "image" : "file";
        downloads.push({ url, kind, name: filename.slice(0, 160) || "网页下载" });
    }, true);

    window.__xhsSaver = {
        submit(text) {
            const current = ++generation;
            state = { kind: "loading", message: "正在通过 TapSave 解析；出现验证时请在网页中完成。", resources: [], text: "" };
            let attempts = 0;
            const fill = () => {
                if (current !== generation) return;
                const workspace = document.querySelector("#download-workspace");
                const input = workspace?.querySelector('input[type="text"]');
                const form = input?.closest("form");
                const mounted = document.querySelector("#__nuxt")?.__vue_app__;
                if (input && form && mounted) {
                    workspace.scrollIntoView({ block: "start" });
                    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(input, text);
                    input.dispatchEvent(new Event("input", { bubbles: true }));
                    input.dispatchEvent(new Event("change", { bubbles: true }));
                    input.focus({ preventScroll: true });
                    form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
                    return;
                }
                if (++attempts < 80) {
                    setTimeout(fill, 250);
                } else {
                    state = { kind: "error", message: "网页未准备好或页面结构已变化，可以在下方网页手动粘贴解析。", resources: [], text: "" };
                }
            };
            fill();
            return "queued";
        },
        drain() {
            // 旧版 WebView 不支持文档起始脚本时，从网页已展示的结果降级读取。
            if (state.kind !== "result") {
                const result = document.querySelector('section[aria-live="polite"]');
                const visible = [];
                result?.querySelectorAll("video[src], img[src]").forEach(element => {
                    const url = secureUrl(element.getAttribute("src"));
                    if (!url || visible.some(item => item.url === url)) return;
                    const kind = element.tagName === "VIDEO" ? "video" : "image";
                    visible.push({ url, kind, name: `${kind === "video" ? "视频" : "图片"} ${visible.length + 1}` });
                });
                if (visible.length) {
                    const text = result.querySelector('[class*="max-h-[420px]"]')?.textContent || "";
                    state = { kind: "result", message: "解析完成。", resources: visible.slice(0, 200), text: text.slice(0, 20000) };
                }
            }
            return { ...state, downloads: downloads.splice(0, 100) };
        }
    };
})();
