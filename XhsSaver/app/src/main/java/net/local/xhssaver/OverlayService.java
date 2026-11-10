// 维护用户主动开启的悬浮球；点击打开前台粘贴，拖动移动，长按关闭。
package net.local.xhssaver;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.provider.Settings;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewConfiguration;
import android.view.WindowManager;
import android.widget.Toast;

public final class OverlayService extends Service {
    static final String ACTION_STOP = "net.local.xhssaver.CLOSE_OVERLAY";
    static final String ACTION_STATE = "net.local.xhssaver.OVERLAY_STATE";
    private static final String CHANNEL = "overlay";
    private static volatile boolean running;
    private final Handler handler = new Handler(Looper.getMainLooper());
    private WindowManager manager;
    private View handle;
    private WindowManager.LayoutParams params;

    static boolean isRunning() { return running; }

    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent == null || ACTION_STOP.equals(intent.getAction()) || !Settings.canDrawOverlays(this)) {
            stopSelf();
            return START_NOT_STICKY;
        }
        NotificationChannel channel = new NotificationChannel(CHANNEL,
                getString(R.string.overlay_channel), NotificationManager.IMPORTANCE_LOW);
        getSystemService(NotificationManager.class).createNotificationChannel(channel);
        PendingIntent open = PendingIntent.getActivity(this, 2, new Intent(this, MainActivity.class),
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        PendingIntent close = PendingIntent.getService(this, 3,
                new Intent(this, OverlayService.class).setAction(ACTION_STOP),
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        Notification notification = new Notification.Builder(this, CHANNEL)
                .setSmallIcon(R.drawable.ic_download).setContentTitle("悬浮球已开启")
                .setContentText("复制链接后点击悬浮球；长按可关闭。")
                .setContentIntent(open).setOnlyAlertOnce(true).setOngoing(true)
                .addAction(new Notification.Action.Builder(null, "关闭悬浮球", close).build()).build();
        try {
            if (Build.VERSION.SDK_INT >= 34) {
                startForeground(12, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE);
            } else {
                startForeground(12, notification);
            }
            if (handle == null) attachHandle();
            running = true;
            sendBroadcast(new Intent(ACTION_STATE).setPackage(getPackageName()));
        } catch (RuntimeException error) {
            Toast.makeText(this, "悬浮球未能开启，请检查悬浮窗权限和系统后台限制。", Toast.LENGTH_LONG).show();
            stopSelf();
        }
        return START_NOT_STICKY;
    }

    private void attachHandle() {
        manager = getSystemService(WindowManager.class);
        int size = dp(56);
        params = new WindowManager.LayoutParams(size, size,
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
                PixelFormat.TRANSLUCENT);
        params.gravity = Gravity.TOP | Gravity.LEFT;
        params.x = Math.max(0, getResources().getDisplayMetrics().widthPixels - size - dp(16));
        params.y = getResources().getDisplayMetrics().heightPixels / 3;
        handle = new SaveHandle();
        handle.setContentDescription("粘贴小红书链接并保存，长按关闭悬浮球");
        handle.setTooltipText("点击粘贴保存，长按关闭");
        handle.setOnClickListener(view -> {
            try {
                startActivity(new Intent(this, MainActivity.class)
                        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_SINGLE_TOP)
                        .putExtra("pasteAndParse", true));
            } catch (RuntimeException error) {
                Toast.makeText(this, "系统阻止了打开保存面板，请从桌面打开笔记保存。", Toast.LENGTH_LONG).show();
            }
        });
        handle.setOnLongClickListener(view -> {
            stopSelf();
            return true;
        });
        handle.setOnTouchListener(new View.OnTouchListener() {
            private float startX;
            private float startY;
            private int windowX;
            private int windowY;
            private boolean moved;
            private boolean longPressed;
            private final int slop = ViewConfiguration.get(OverlayService.this).getScaledTouchSlop();
            private final Runnable hold = () -> {
                longPressed = true;
                stopSelf();
            };

            @Override public boolean onTouch(View view, MotionEvent event) {
                switch (event.getActionMasked()) {
                    case MotionEvent.ACTION_DOWN:
                        startX = event.getRawX();
                        startY = event.getRawY();
                        windowX = params.x;
                        windowY = params.y;
                        moved = false;
                        longPressed = false;
                        handler.postDelayed(hold, ViewConfiguration.getLongPressTimeout());
                        return true;
                    case MotionEvent.ACTION_MOVE:
                        float deltaX = event.getRawX() - startX;
                        float deltaY = event.getRawY() - startY;
                        if (Math.hypot(deltaX, deltaY) > slop) {
                            moved = true;
                            handler.removeCallbacks(hold);
                        }
                        if (moved && running) {
                            params.x = Math.max(0, Math.min(getResources().getDisplayMetrics().widthPixels - size,
                                    windowX + Math.round(deltaX)));
                            params.y = Math.max(0, Math.min(getResources().getDisplayMetrics().heightPixels - size,
                                    windowY + Math.round(deltaY)));
                            try {
                                manager.updateViewLayout(view, params);
                            } catch (RuntimeException error) {
                                stopSelf();
                            }
                        }
                        return true;
                    case MotionEvent.ACTION_UP:
                        handler.removeCallbacks(hold);
                        if (!moved && !longPressed) view.performClick();
                        return true;
                    case MotionEvent.ACTION_CANCEL:
                        handler.removeCallbacks(hold);
                        return true;
                    default:
                        return false;
                }
            }
        });
        manager.addView(handle, params);
    }

    private int dp(int value) { return Math.round(value * getResources().getDisplayMetrics().density); }

    private final class SaveHandle extends View {
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);

        SaveHandle() {
            super(OverlayService.this);
            setClickable(true);
            setFocusable(true);
        }

        @Override protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            float unit = getWidth() / 56f;
            canvas.save();
            canvas.scale(unit, unit);
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(Color.rgb(23, 107, 80));
            canvas.drawCircle(28, 28, 27, paint);
            paint.setColor(Color.WHITE);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(2.6f);
            paint.setStrokeCap(Paint.Cap.ROUND);
            canvas.drawLine(28, 15, 28, 32, paint);
            canvas.drawLine(21, 25, 28, 32, paint);
            canvas.drawLine(35, 25, 28, 32, paint);
            canvas.drawLine(19, 39, 37, 39, paint);
            canvas.restore();
        }
    }

    @Override public void onDestroy() {
        running = false;
        sendBroadcast(new Intent(ACTION_STATE).setPackage(getPackageName()));
        handler.removeCallbacksAndMessages(null);
        if (handle != null && handle.isAttachedToWindow()) {
            manager.removeView(handle);
        }
        handle = null;
        stopForeground(STOP_FOREGROUND_REMOVE);
        super.onDestroy();
    }

    @Override public IBinder onBind(Intent intent) { return null; }
}
