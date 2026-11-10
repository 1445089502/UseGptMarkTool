// 管理网页会话与轮询；仅在可信主页面执行适配脚本，关闭时释放 WebView。
package net.local.xhssaver;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Handler;
import android.os.Looper;
import android.view.ViewGroup;
import android.webkit.CookieManager;
import android.webkit.RenderProcessGoneDetail;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.FrameLayout;

import androidx.webkit.ScriptHandler;
import androidx.webkit.WebViewCompat;
import androidx.webkit.WebViewFeature;

import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONTokener;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashSet;

final class TapSaveBrowser {
    static final String PAGE = "https://www.tapsave.net/zh/xiaohongshu/";

    interface Listener {
        void onStatus(String message, boolean terminal);
        void onResult(JSONObject result);
        void onDownload(MediaResource resource);
    }

    private final Activity activity;
    private final Listener listener;
    private final WebView view;
    private final Handler handler = new Handler(Looper.getMainLooper());
    private final String adapter;
    private String link = "";
    private String lastState = "";
    private int session;
    private int injectedSession = -1;
    private boolean paused;
    private boolean closed;
    private boolean resultDelivered;
    private boolean timeoutReported;
    private ScriptHandler documentScript;
    private final Runnable parseTimeout = new Runnable() {
        @Override public void run() {
            if (!closed && !resultDelivered && !timeoutReported) {
                timeoutReported = true;
                listener.onStatus("等待解析超时，可在网页中继续验证，或重新解析。", true);
            }
        }
    };

    private final Runnable poll = new Runnable() {
        @Override public void run() {
            if (paused || closed || !trusted(view.getUrl())) return;
            int expected = session;
            view.evaluateJavascript("window.__xhsSaver ? JSON.stringify(window.__xhsSaver.drain()) : null", raw -> {
                if (closed || paused || expected != session) return;
                handleState(raw);
                handler.postDelayed(this, 800);
            });
        }
    };

    @SuppressLint("SetJavaScriptEnabled")
    TapSaveBrowser(Activity activity, FrameLayout container, Listener listener) throws IOException {
        this.activity = activity;
        this.listener = listener;
        try (InputStream input = activity.getAssets().open("tapsave.js");
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192];
            int count;
            while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
            adapter = output.toString(StandardCharsets.UTF_8.name());
        }
        view = new WebView(activity);
        WebSettings settings = view.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setAllowFileAccess(false);
        settings.setAllowContentAccess(false);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        settings.setMediaPlaybackRequiresUserGesture(true);
        if (WebViewFeature.isFeatureSupported(WebViewFeature.DOCUMENT_START_SCRIPT)) {
            documentScript = WebViewCompat.addDocumentStartJavaScript(view, adapter,
                    new HashSet<>(Arrays.asList("https://www.tapsave.net", "https://tapsave.net")));
        }
        CookieManager.getInstance().setAcceptCookie(true);
        CookieManager.getInstance().setAcceptThirdPartyCookies(view, true);
        view.setWebChromeClient(new WebChromeClient());
        view.setWebViewClient(new WebViewClient() {
            @Override public void onPageFinished(WebView webView, String url) {
                if (closed || !trusted(url) || injectedSession == session) return;
                injectedSession = session;
                int expected = session;
                webView.evaluateJavascript(adapter, ignored -> {
                    if (closed || expected != session) return;
                    webView.evaluateJavascript("window.__xhsSaver.submit(" + JSONObject.quote(link) + ")", unused -> { });
                    resume();
                });
            }

            @Override public boolean shouldOverrideUrlLoading(WebView webView, WebResourceRequest request) {
                if (!request.isForMainFrame()) return false;
                Uri url = request.getUrl();
                if (trusted(url.toString()) || url.toString().equals("about:blank")) return false;
                if (request.hasGesture() && "https".equalsIgnoreCase(url.getScheme())) {
                    try {
                        activity.startActivity(new Intent(Intent.ACTION_VIEW, url));
                    } catch (RuntimeException error) {
                        listener.onStatus("未找到可打开此链接的浏览器。", true);
                    }
                }
                return true;
            }

            @Override public void onReceivedError(WebView webView, WebResourceRequest request, WebResourceError error) {
                if (!closed && request.isForMainFrame()) {
                    timeoutReported = true;
                    listener.onStatus("解析网页加载失败，请检查网络后重新解析。", true);
                }
            }

            @Override public void onReceivedHttpError(WebView webView, WebResourceRequest request,
                                                       WebResourceResponse response) {
                if (!closed && request.isForMainFrame()) {
                    timeoutReported = true;
                    listener.onStatus("解析网页暂不可用（HTTP " + response.getStatusCode() + "），请在网页中检查提示或稍后重试。", true);
                }
            }

            @Override public boolean onRenderProcessGone(WebView webView, RenderProcessGoneDetail detail) {
                close();
                listener.onStatus("系统已关闭网页进程，请重新解析。", true);
                return true;
            }
        });
        view.setDownloadListener((url, userAgent, disposition, mime, length) -> {
            try {
                String kind = mime != null && mime.startsWith("image/") ? "image" :
                        mime != null && mime.startsWith("video/") ? "video" : "file";
                listener.onDownload(MediaResource.fromJson(new JSONObject()
                        .put("url", url).put("kind", kind).put("name", "网页下载")));
            } catch (Exception error) {
                listener.onStatus("网页没有提供可保存的 HTTPS 文件地址。", true);
            }
        });
        container.addView(view, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
    }

    void open(String link) {
        if (closed) throw new IllegalStateException("网页会话已经关闭");
        session++;
        this.link = link;
        lastState = "";
        resultDelivered = false;
        timeoutReported = false;
        handler.removeCallbacks(poll);
        handler.removeCallbacks(parseTimeout);
        handler.postDelayed(parseTimeout, 90000);
        view.stopLoading();
        view.loadUrl(PAGE);
    }

    boolean isClosed() { return closed; }

    String userAgent() { return view.getSettings().getUserAgentString(); }

    void pause() {
        paused = true;
        handler.removeCallbacks(poll);
        if (!closed) view.onPause();
    }

    void resume() {
        if (closed) return;
        paused = false;
        view.onResume();
        handler.removeCallbacks(poll);
        handler.post(poll);
    }

    void close() {
        if (closed) return;
        closed = true;
        session++;
        handler.removeCallbacksAndMessages(null);
        view.stopLoading();
        if (view.getParent() instanceof ViewGroup parent) parent.removeView(view);
        view.setDownloadListener(null);
        if (documentScript != null) documentScript.remove();
        view.destroy();
    }

    private void handleState(String raw) {
        if (raw == null || raw.equals("null")) return;
        try {
            Object decoded = new JSONTokener(raw).nextValue();
            if (!(decoded instanceof String json)) return;
            JSONObject state = new JSONObject(json);
            JSONArray downloads = state.optJSONArray("downloads");
            if (downloads != null) {
                for (MediaResource resource : MediaResource.fromArray(downloads)) listener.onDownload(resource);
            }
            String kind = state.optString("kind");
            String message = state.optString("message");
            if (kind.equals("result") && !resultDelivered) {
                resultDelivered = true;
                handler.removeCallbacks(parseTimeout);
                listener.onResult(state);
            } else if (!resultDelivered && !(kind + message).equals(lastState)) {
                lastState = kind + message;
                if (kind.equals("error")) handler.removeCallbacks(parseTimeout);
                listener.onStatus(message, kind.equals("error"));
            }
        } catch (Exception error) {
            listener.onStatus("网页返回的数据无法识别，可以使用网页内的下载按钮。", true);
        }
    }

    private static boolean trusted(String value) {
        if (value == null) return false;
        Uri url = Uri.parse(value);
        return "https".equalsIgnoreCase(url.getScheme()) &&
                ("www.tapsave.net".equalsIgnoreCase(url.getHost()) || "tapsave.net".equalsIgnoreCase(url.getHost()));
    }
}
