// 在用户发起保存后串行下载；写入完成才向相册公开文件，失败和取消会删除半成品。
package net.local.xhssaver;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.net.Uri;
import android.os.Environment;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.webkit.MimeTypeMap;

import org.json.JSONArray;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public final class DownloadService extends Service {
    static final String ACTION_SAVE = "net.local.xhssaver.SAVE";
    static final String ACTION_CANCEL = "net.local.xhssaver.CANCEL_DOWNLOADS";
    static final String ACTION_PROGRESS = "net.local.xhssaver.DOWNLOAD_PROGRESS";
    private static final String CHANNEL = "downloads";
    private static final int NOTIFICATION_ID = 11;
    private static volatile boolean running;
    private static volatile Status latestStatus;
    private static final ConcurrentHashMap<String, DownloadService> QUEUED = new ConcurrentHashMap<>();
    // 仅表示本进程中确认写入成功的地址，不作为跨进程的相册历史。
    private static final Set<String> SAVED_URLS = ConcurrentHashMap.newKeySet();

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final Handler main = new Handler(Looper.getMainLooper());
    private final AtomicBoolean cancelled = new AtomicBoolean();
    private final AtomicInteger remaining = new AtomicInteger();
    private final AtomicInteger saved = new AtomicInteger();
    private final AtomicInteger failed = new AtomicInteger();
    private volatile HttpURLConnection activeConnection;
    private volatile String lastFailure = "";
    private int latestStartId;

    static boolean isRunning() { return running; }
    static boolean isQueued(String url) { return QUEUED.containsKey(url); }
    static boolean isSaved(String url) { return SAVED_URLS.contains(url); }
    static Status getStatus() { return latestStatus; }

    static final class Status {
        final String message;
        final boolean finished;
        final int percent;

        Status(String message, boolean finished, int percent) {
            this.message = message;
            this.finished = finished;
            this.percent = percent;
        }
    }

    @Override public void onCreate() {
        super.onCreate();
        running = true;
        NotificationChannel channel = new NotificationChannel(CHANNEL,
                getString(R.string.download_channel), NotificationManager.IMPORTANCE_LOW);
        getSystemService(NotificationManager.class).createNotificationChannel(channel);
    }

    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        latestStartId = startId;
        if (intent == null || ACTION_CANCEL.equals(intent.getAction())) {
            cancelDownloads("已取消保存。");
            return START_NOT_STICKY;
        }
        if (cancelled.get()) {
            broadcast("上一次保存任务正在结束，请稍后重试。", true);
            stopSelf();
            return START_NOT_STICKY;
        }
        try {
            startForeground(NOTIFICATION_ID, notification("正在准备保存…"),
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC);
            String payload = intent.getStringExtra("resources");
            if (payload == null || payload.length() > 800000) throw new IOException("保存任务数据无效");
            List<MediaResource> resources = MediaResource.fromArray(new JSONArray(payload));
            List<String> cookies = intent.getStringArrayListExtra("cookies");
            String userAgent = intent.getStringExtra("userAgent");
            for (int index = 0; index < resources.size(); index++) {
                MediaResource resource = resources.get(index);
                if (SAVED_URLS.contains(resource.url) || QUEUED.putIfAbsent(resource.url, this) != null) continue;
                String cookie = cookies != null && index < cookies.size() ? cookies.get(index) : null;
                remaining.incrementAndGet();
                worker.execute(() -> saveOne(resource, cookie, userAgent));
            }
            if (remaining.get() == 0) finishIfIdle();
            else broadcast("已加入保存队列，剩余 " + remaining.get() + " 项。", false);
        } catch (Exception error) {
            broadcast("无法创建保存任务：" + error.getMessage(), remaining.get() == 0);
            if (remaining.get() == 0) {
                stopForeground(STOP_FOREGROUND_REMOVE);
                stopSelfResult(latestStartId);
            }
        }
        return START_NOT_STICKY;
    }

    private void saveOne(MediaResource resource, String cookie, String userAgent) {
        Uri pending = null;
        try {
            checkCancelled();
            HttpURLConnection connection = connect(resource.url, cookie, userAgent);
            long length = connection.getContentLengthLong();
            try (BufferedInputStream input = new BufferedInputStream(connection.getInputStream())) {
                String mime = detectMime(input);
                if (resource.kind.equals("image") && !mime.startsWith("image/")
                        || resource.kind.equals("video") && !mime.startsWith("video/")) {
                    throw new IOException("下载内容与解析出的媒体类型不一致，请重新解析");
                }
                boolean image = mime.startsWith("image/");
                boolean video = mime.startsWith("video/");
                String extension = MimeTypeMap.getSingleton().getExtensionFromMimeType(mime);
                if (extension == null) extension = mime.substring(mime.indexOf('/') + 1);
                String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.ROOT).format(new Date());
                ContentValues values = new ContentValues();
                values.put(MediaStore.MediaColumns.DISPLAY_NAME, "xhs_" + stamp + "." + extension);
                values.put(MediaStore.MediaColumns.MIME_TYPE, mime);
                values.put(MediaStore.MediaColumns.RELATIVE_PATH,
                        (image ? Environment.DIRECTORY_PICTURES : video ? Environment.DIRECTORY_MOVIES :
                                Environment.DIRECTORY_DOWNLOADS) + "/XhsSaver");
                values.put(MediaStore.MediaColumns.IS_PENDING, 1);
                Uri collection = image ? MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY) :
                        video ? MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY) :
                                MediaStore.Downloads.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY);
                ContentResolver resolver = getContentResolver();
                pending = resolver.insert(collection, values);
                if (pending == null) throw new IOException("系统未能创建相册文件，请检查剩余空间");
                try (OutputStream output = resolver.openOutputStream(pending, "w")) {
                    if (output == null) throw new IOException("无法写入相册文件");
                    byte[] buffer = new byte[65536];
                    long received = 0;
                    long lastUpdate = 0;
                    int count;
                    while ((count = input.read(buffer)) != -1) {
                        checkCancelled();
                        output.write(buffer, 0, count);
                        received += count;
                        long now = SystemClock.elapsedRealtime();
                        if (now - lastUpdate > 700) {
                            lastUpdate = now;
                            String progress = length > 0 ? " " + Math.min(100, received * 100 / length) + "%" :
                                    String.format(Locale.ROOT, " %.1f MB", received / 1048576.0);
                            update("正在保存" + resource.name + progress,
                                    length > 0 ? (int) Math.min(100, received * 100 / length) : -1);
                        }
                    }
                    if (received == 0 || length > 0 && received != length) {
                        throw new IOException("文件未完整下载，请重新保存");
                    }
                }
                checkCancelled();
                values.clear();
                values.put(MediaStore.MediaColumns.IS_PENDING, 0);
                if (resolver.update(pending, values, null, null) != 1) throw new IOException("无法完成相册文件保存");
                pending = null;
                SAVED_URLS.add(resource.url);
                saved.incrementAndGet();
                broadcast(resource.name + "已保存" + (image || video ? "到相册。" : "到下载目录。"), false);
            } finally {
                connection.disconnect();
                activeConnection = null;
            }
        } catch (Exception error) {
            if (!cancelled.get()) {
                failed.incrementAndGet();
                lastFailure = resource.name + "：" + friendlyError(error);
                broadcast("保存失败，" + lastFailure, false);
            }
        } finally {
            if (pending != null) {
                try {
                    getContentResolver().delete(pending, null, null);
                } catch (RuntimeException cleanupError) {
                    broadcast("保存失败，系统暂未清理未完成文件；该文件不会出现在相册中。", false);
                }
            }
            QUEUED.remove(resource.url, this);
            remaining.decrementAndGet();
            main.post(() -> {
                if (!cancelled.get() && remaining.get() > 0) broadcast("正在准备剩余 " + remaining.get() + " 项资源。", false);
                finishIfIdle();
            });
        }
    }

    private HttpURLConnection connect(String address, String cookie, String userAgent) throws IOException {
        URL initial = new URL(address);
        URL current = initial;
        for (int redirects = 0; redirects <= 5; redirects++) {
            checkCancelled();
            if (!current.getProtocol().equals("https") || current.getUserInfo() != null) {
                throw new IOException("下载服务返回了不支持的地址，请重新解析");
            }
            HttpURLConnection connection = (HttpURLConnection) current.openConnection();
            activeConnection = connection;
            connection.setInstanceFollowRedirects(false);
            connection.setConnectTimeout(15000);
            connection.setReadTimeout(30000);
            connection.setRequestProperty("Accept-Encoding", "identity");
            connection.setRequestProperty("Referer", TapSaveBrowser.PAGE);
            if (userAgent != null) connection.setRequestProperty("User-Agent", userAgent);
            if (cookie != null && sameOrigin(initial, current)) connection.setRequestProperty("Cookie", cookie);
            int status;
            try {
                status = connection.getResponseCode();
            } catch (IOException error) {
                connection.disconnect();
                activeConnection = null;
                throw error;
            }
            if (status == 200) return connection;
            String location = connection.getHeaderField("Location");
            connection.disconnect();
            activeConnection = null;
            if ((status == 301 || status == 302 || status == 303 || status == 307 || status == 308)
                    && location != null) {
                current = new URL(current, location);
                continue;
            }
            throw new IOException(status == 401 || status == 403 ? "资源访问被拒绝或链接已过期，请重新解析" :
                    "下载服务器返回 HTTP " + status);
        }
        throw new IOException("下载地址重定向过多，请重新解析");
    }

    private static boolean sameOrigin(URL first, URL second) {
        return first.getProtocol().equals(second.getProtocol()) && first.getHost().equalsIgnoreCase(second.getHost())
                && first.getPort() == second.getPort();
    }

    private static String detectMime(BufferedInputStream input) throws IOException {
        input.mark(64);
        byte[] header = new byte[32];
        int count = 0;
        while (count < header.length) {
            int read = input.read(header, count, header.length - count);
            if (read == -1) break;
            count += read;
        }
        input.reset();
        if (count >= 3 && (header[0] & 255) == 255 && (header[1] & 255) == 216 && (header[2] & 255) == 255) return "image/jpeg";
        String signature = new String(header, 0, count, StandardCharsets.ISO_8859_1);
        if (count >= 8 && (header[0] & 255) == 137 && signature.startsWith("PNG", 1)) return "image/png";
        if (signature.startsWith("GIF87a") || signature.startsWith("GIF89a")) return "image/gif";
        if (signature.startsWith("RIFF") && signature.startsWith("WEBP", 8)) return "image/webp";
        if (signature.startsWith("ftyp", 4)) {
            String brand = signature.length() >= 12 ? signature.substring(8, 12) : "";
            if (brand.equals("avif") || brand.equals("avis")) return "image/avif";
            if (brand.equals("heic") || brand.equals("heix") || brand.equals("mif1")) return "image/heif";
            return "video/mp4";
        }
        if (count >= 4 && (header[0] & 255) == 26 && (header[1] & 255) == 69
                && (header[2] & 255) == 223 && (header[3] & 255) == 163) return "video/webm";
        if (count >= 4 && header[0] == 80 && header[1] == 75 &&
                (header[2] == 3 && header[3] == 4 || header[2] == 5 && header[3] == 6)) return "application/zip";
        throw new IOException("服务器返回的内容不是可识别的图片、视频或压缩包，请重新解析");
    }

    private void checkCancelled() throws InterruptedIOException {
        if (cancelled.get() || Thread.currentThread().isInterrupted()) throw new InterruptedIOException("已取消保存");
    }

    private static String friendlyError(Exception error) {
        if (error instanceof java.net.SocketTimeoutException) return "网络连接超时，请重试";
        if (error instanceof java.net.UnknownHostException) return "无法连接下载服务器，请检查网络";
        if (error instanceof javax.net.ssl.SSLException) return "下载服务器的安全连接失败，请稍后重试";
        String message = error.getMessage();
        return message == null ? "无法完成文件写入，请检查网络与剩余空间" : message;
    }

    private void update(String message, int percent) {
        main.post(() -> {
            if (cancelled.get()) return;
            getSystemService(NotificationManager.class).notify(NOTIFICATION_ID, notification(message));
            broadcast(message, false, percent);
        });
    }

    private void broadcast(String message, boolean finished) {
        broadcast(message, finished, -1);
    }

    private void broadcast(String message, boolean finished, int percent) {
        if (cancelled.get() && !finished) return;
        latestStatus = new Status(message, finished, percent);
        sendBroadcast(new Intent(ACTION_PROGRESS).setPackage(getPackageName())
                .putExtra("message", message).putExtra("finished", finished)
                .putExtra("saved", saved.get()).putExtra("failed", failed.get()).putExtra("percent", percent));
    }

    private Notification notification(String message) {
        PendingIntent open = PendingIntent.getActivity(this, 0, new Intent(this, MainActivity.class),
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        PendingIntent cancel = PendingIntent.getService(this, 1,
                new Intent(this, DownloadService.class).setAction(ACTION_CANCEL),
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        return new Notification.Builder(this, CHANNEL).setSmallIcon(R.drawable.ic_download)
                .setContentTitle("笔记保存").setContentText(message).setContentIntent(open)
                .setOngoing(true).setOnlyAlertOnce(true)
                .addAction(new Notification.Action.Builder(null, "取消保存", cancel).build()).build();
    }

    private void finishIfIdle() {
        if (remaining.get() != 0 || cancelled.get()) return;
        broadcast("保存完成：成功 " + saved.get() + " 项" +
                (failed.get() > 0 ? "，失败 " + failed.get() + " 项。最近失败：" + lastFailure : "。"), true);
        stopForeground(STOP_FOREGROUND_REMOVE);
        stopSelfResult(latestStartId);
    }

    private void cancelDownloads(String message) {
        cancelled.set(true);
        QUEUED.entrySet().removeIf(entry -> entry.getValue() == this);
        HttpURLConnection connection = activeConnection;
        if (connection != null) connection.disconnect();
        broadcast(message, true);
        stopForeground(STOP_FOREGROUND_REMOVE);
        stopSelf();
    }

    @Override public void onTimeout(int startId, int fgsType) {
        cancelDownloads("系统已暂停长时间运行的保存任务，请重新保存剩余内容。");
    }

    @Override public void onDestroy() {
        running = false;
        cancelled.set(true);
        HttpURLConnection connection = activeConnection;
        if (connection != null) connection.disconnect();
        worker.shutdownNow();
        QUEUED.entrySet().removeIf(entry -> entry.getValue() == this);
        if (latestStatus != null && !latestStatus.finished) {
            broadcast("保存任务已停止，可重试未完成内容。", true);
        }
        main.removeCallbacksAndMessages(null);
        super.onDestroy();
    }

    @Override public IBinder onBind(Intent intent) { return null; }
}
