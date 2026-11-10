// 提供分享、粘贴、验证和保存入口，网页与下载服务各自维护资源生命周期。
package net.local.xhssaver;

import android.Manifest;
import android.content.BroadcastReceiver;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Rect;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.text.TextUtils;
import android.view.Gravity;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.webkit.CookieManager;
import android.widget.FrameLayout;
import android.widget.GridLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.MediaController;
import android.widget.TextView;
import android.widget.VideoView;

import androidx.activity.OnBackPressedCallback;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.core.widget.NestedScrollView;

import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestManager;
import com.bumptech.glide.load.DataSource;
import com.bumptech.glide.load.HttpException;
import com.bumptech.glide.load.engine.GlideException;
import com.bumptech.glide.load.model.GlideUrl;
import com.bumptech.glide.load.model.LazyHeaders;
import com.bumptech.glide.request.RequestListener;
import com.bumptech.glide.request.target.Target;
import com.google.android.material.appbar.MaterialToolbar;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.card.MaterialCardView;
import com.google.android.material.checkbox.MaterialCheckBox;
import com.google.android.material.color.MaterialColors;
import com.google.android.material.dialog.MaterialAlertDialogBuilder;
import com.google.android.material.materialswitch.MaterialSwitch;
import com.google.android.material.progressindicator.LinearProgressIndicator;
import com.google.android.material.snackbar.Snackbar;
import com.google.android.material.textfield.TextInputEditText;
import com.google.android.material.textfield.TextInputLayout;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class MainActivity extends AppCompatActivity {
    private TextInputEditText input;
    private TextInputLayout inputLayout;
    private MaterialButton parseButton;
    private MaterialButton saveAll;
    private MaterialButton browserToggle;
    private MaterialSwitch overlaySwitch;
    private LinearProgressIndicator progress;
    private TextView status;
    private TextView downloadStatus;
    private TextView noteText;
    private LinearLayout results;
    private GridLayout resourceList;
    private LinearProgressIndicator downloadProgress;
    private View browserPanel;
    private NestedScrollView scroll;
    private TapSaveBrowser browser;
    private RequestManager imageRequests;
    private final List<PreviewBinding> previewTargets = new ArrayList<>();
    private AlertDialog imageDialog;
    private AlertDialog videoDialog;
    private VideoView videoPreview;
    private MediaController videoController;
    private final Set<String> selectedUrls = new HashSet<>();
    private final Map<MaterialCheckBox, String> selectionControls = new LinkedHashMap<>();
    private final Map<MaterialButton, String> saveControls = new LinkedHashMap<>();
    private boolean updatingSelection;
    private boolean activityResumed;
    private boolean previewRefreshPending;
    private boolean textExpanded;
    private List<MediaResource> resources = new ArrayList<>();
    private boolean pendingPaste;
    private boolean pendingOverlay;
    private boolean updatingSwitch;
    private boolean receiverRegistered;
    private String savedText = "";

    private final BroadcastReceiver downloadReceiver = new BroadcastReceiver() {
        @Override public void onReceive(Context context, Intent intent) {
            if (OverlayService.ACTION_STATE.equals(intent.getAction())) {
                refreshOverlaySwitch();
                return;
            }
            refreshDownloadState();
        }
    };

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        WindowCompat.setDecorFitsSystemWindows(getWindow(), false);
        setContentView(R.layout.activity_main);
        imageRequests = Glide.with(this);
        View root = findViewById(R.id.root);
        ViewCompat.setOnApplyWindowInsetsListener(root, (view, insets) -> {
            Insets safe = insets.getInsets(WindowInsetsCompat.Type.systemBars() |
                    WindowInsetsCompat.Type.displayCutout() | WindowInsetsCompat.Type.ime());
            view.setPadding(safe.left, safe.top, safe.right, safe.bottom);
            return insets;
        });
        input = findViewById(R.id.link_input);
        inputLayout = findViewById(R.id.link_layout);
        parseButton = findViewById(R.id.parse_button);
        saveAll = findViewById(R.id.save_all);
        browserToggle = findViewById(R.id.browser_toggle);
        overlaySwitch = findViewById(R.id.overlay_switch);
        progress = findViewById(R.id.progress);
        status = findViewById(R.id.status);
        downloadStatus = findViewById(R.id.download_status);
        downloadProgress = findViewById(R.id.download_progress);
        noteText = findViewById(R.id.note_text);
        results = findViewById(R.id.results);
        resourceList = findViewById(R.id.resource_list);
        browserPanel = findViewById(R.id.browser_panel);
        scroll = findViewById(R.id.scroll);
        findViewById(R.id.paste_button).setOnClickListener(view -> paste(true));
        parseButton.setOnClickListener(view -> parse());
        findViewById(R.id.cancel_parse).setOnClickListener(view -> {
            if (browser != null) browser.close();
            setParsing(false);
            showBrowser(false);
            browserToggle.setVisibility(View.GONE);
            setParseStatus("已取消解析，可以修改链接后重新开始。", false);
        });
        saveAll.setOnClickListener(view -> {
            List<MediaResource> selected = new ArrayList<>();
            for (MediaResource resource : resources) {
                if (selectedUrls.contains(resource.url)) selected.add(resource);
            }
            save(selected);
        });
        findViewById(R.id.select_all).setOnClickListener(view -> toggleSelection());
        findViewById(R.id.cancel_save).setOnClickListener(view -> {
            try {
                startService(new Intent(this, DownloadService.class).setAction(DownloadService.ACTION_CANCEL));
            } catch (RuntimeException error) {
                message("未能取消保存，请从通知中的取消按钮重试。");
            }
        });
        findViewById(R.id.expand_text).setOnClickListener(view -> {
            textExpanded = !textExpanded;
            noteText.setMaxLines(textExpanded ? Integer.MAX_VALUE : 6);
            noteText.setEllipsize(textExpanded ? null : TextUtils.TruncateAt.END);
            ((MaterialButton) view).setText(textExpanded ? "收起全文" : "展开全文");
        });
        MaterialToolbar toolbar = findViewById(R.id.toolbar);
        toolbar.setOnMenuItemClickListener(item -> {
            if (item.getItemId() == R.id.action_gallery) {
                try {
                    startActivity(new Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_APP_GALLERY));
                } catch (RuntimeException error) {
                    message("请打开系统相册，在 XhsSaver 相册中查看已保存内容。");
                }
            } else if (item.getItemId() == R.id.action_help) {
                new MaterialAlertDialogBuilder(this).setTitle("如何保存")
                        .setMessage("在小红书复制笔记链接或分享给“笔记保存”。\n\n粘贴并解析后，如出现验证，请在网页中完成。选择图片或视频，再点击保存。\n\n图片可点开查看大图，视频可点击播放预览。保存后在系统相册的 XhsSaver 相册中查看。\n\n分享菜单没有入口时，可以使用复制链接或悬浮球。")
                        .setPositiveButton("知道了", null).show();
            }
            return true;
        });
        scroll.setOnScrollChangeListener((NestedScrollView.OnScrollChangeListener)
                (view, x, y, previousX, previousY) -> queuePreviewRefresh());
        scroll.addOnLayoutChangeListener((view, left, top, right, bottom, oldLeft, oldTop, oldRight, oldBottom) -> {
            if (right - left != oldRight - oldLeft || bottom - top != oldBottom - oldTop) applyContentSize();
            queuePreviewRefresh();
        });
        scroll.post(this::applyContentSize);
        browserToggle.setOnClickListener(view -> showBrowser(browserPanel.getVisibility() != View.VISIBLE));
        findViewById(R.id.copy_text).setOnClickListener(view -> {
            getSystemService(ClipboardManager.class).setPrimaryClip(ClipData.newPlainText("笔记文案", savedText));
            message("笔记文案已复制。");
        });
        overlaySwitch.setOnCheckedChangeListener((button, enabled) -> {
            if (updatingSwitch) return;
            if (!enabled) {
                stopService(new Intent(this, OverlayService.class));
            } else if (Settings.canDrawOverlays(this)) {
                startOverlay();
            } else {
                pendingOverlay = true;
                refreshOverlaySwitch();
                try {
                    startActivity(new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                            Uri.parse("package:" + getPackageName())));
                } catch (RuntimeException error) {
                    pendingOverlay = false;
                    message("系统未提供悬浮窗授权入口；仍可使用分享和粘贴保存。");
                }
            }
        });
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) {
            @Override public void handleOnBackPressed() {
                if (browserPanel.getVisibility() == View.VISIBLE) {
                    showBrowser(false);
                } else {
                    setEnabled(false);
                    getOnBackPressedDispatcher().onBackPressed();
                }
            }
        });
        if (state != null) {
            input.setText(state.getString("input", ""));
            pendingOverlay = state.getBoolean("pendingOverlay");
            try {
                resources = MediaResource.fromArray(new JSONArray(state.getString("resources", "[]")));
                savedText = state.getString("text", "");
                ArrayList<Integer> selection = state.getIntegerArrayList("selection");
                if (selection != null) {
                    for (int index : selection) {
                        if (index >= 0 && index < resources.size()) selectedUrls.add(resources.get(index).url);
                    }
                }
                else for (MediaResource resource : resources) selectedUrls.add(resource.url);
                if (!resources.isEmpty() || !savedText.isEmpty()) {
                    renderResults();
                    setParseStatus("已恢复上次结果，可继续预览和保存。", false);
                }
            } catch (Exception error) {
                status.setText("之前的解析结果未能恢复，请重新解析。");
            }
        } else {
            receiveIntent(getIntent());
        }
    }

    @Override protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        receiveIntent(intent);
    }

    private void receiveIntent(Intent intent) {
        if (intent == null) return;
        if (intent.getBooleanExtra("pasteAndParse", false)) {
            pendingPaste = true;
            intent.removeExtra("pasteAndParse");
            if (hasWindowFocus()) input.post(MainActivity.this::consumePendingPaste);
            return;
        }
        if (Intent.ACTION_SEND.equals(intent.getAction())) {
            CharSequence text = intent.getCharSequenceExtra(Intent.EXTRA_TEXT);
            if (text == null && intent.getClipData() != null && intent.getClipData().getItemCount() > 0) {
                text = intent.getClipData().getItemAt(0).getText();
            }
            if (text == null || text.length() > 20000) {
                message("收到的分享文字为空或过长，请复制笔记链接后粘贴。");
                return;
            }
            input.setText(text);
            input.post(this::parse);
        }
    }

    private void paste(boolean autoParse) {
        try {
            ClipboardManager clipboard = getSystemService(ClipboardManager.class);
            ClipData clip = clipboard.getPrimaryClip();
            if (clip == null || clip.getItemCount() == 0) {
                message("剪贴板里没有文字，请先复制小红书笔记链接。");
                return;
            }
            CharSequence value = clip.getItemAt(0).getText();
            String link = value == null ? null : SharedLink.extract(value.toString());
            if (link == null) {
                message("剪贴板里没有有效的小红书链接，请重新复制。");
                return;
            }
            input.setText(link);
            inputLayout.setError(null);
            if (autoParse) parse();
        } catch (RuntimeException error) {
            message("系统暂未允许读取剪贴板，请长按输入框粘贴链接。");
        }
    }

    private void parse() {
        if (isFinishing() || isDestroyed()) return;
        String link = SharedLink.extract(String.valueOf(input.getText()));
        if (link == null) {
            inputLayout.setError("请粘贴小红书笔记链接，支持 xhslink.cn、xhslink.com 和 xiaohongshu.com。");
            return;
        }
        input.setText(link);
        input.setSelection(link.length());
        inputLayout.setError(null);
        getSystemService(InputMethodManager.class).hideSoftInputFromWindow(input.getWindowToken(), 0);
        input.clearFocus();
        clearImagePreviews();
        selectedUrls.clear();
        selectionControls.clear();
        saveControls.clear();
        resources = new ArrayList<>();
        savedText = "";
        results.setVisibility(View.GONE);
        setParsing(true);
        setParseStatus("正在打开解析网页，如有验证请在下方完成。", false);
        browserToggle.setVisibility(View.VISIBLE);
        showBrowser(true);
        try {
            // 新笔记使用独立页面，避免上一笔请求和加载回调写入当前结果。
            if (browser != null) browser.close();
            {
                FrameLayout container = findViewById(R.id.browser_container);
                browser = new TapSaveBrowser(this, container, new TapSaveBrowser.Listener() {
                    @Override public void onStatus(String text, boolean terminal) {
                        if (isFinishing() || isDestroyed()) return;
                        setParseStatus(text, terminal);
                        if (terminal) setParsing(false);
                    }

                    @Override public void onResult(JSONObject data) {
                        if (isFinishing() || isDestroyed()) return;
                        try {
                            resources = MediaResource.fromArray(data.getJSONArray("resources"));
                            selectedUrls.clear();
                            for (MediaResource resource : resources) selectedUrls.add(resource.url);
                            savedText = data.optString("text");
                            setParsing(false);
                            setParseStatus("解析完成，可点击预览并勾选需要保存的内容。", false);
                            renderResults();
                            showBrowser(false);
                            scroll.post(() -> scroll.smoothScrollTo(0, results.getTop()));
                        } catch (Exception error) {
                            setParsing(false);
                            setParseStatus("网页返回的资源格式无法识别，可以使用网页下载按钮。", true);
                        }
                    }

                    @Override public void onDownload(MediaResource resource) {
                        if (!isFinishing() && !isDestroyed()) save(Collections.singletonList(resource));
                    }
                });
            }
            browser.open(link);
        } catch (Exception error) {
            setParsing(false);
            setParseStatus("无法启动解析网页，请更新系统的 Android System WebView 后重试。", true);
        }
    }

    private void renderResults() {
        clearImagePreviews();
        results.setVisibility(View.VISIBLE);
        int images = 0;
        int videos = 0;
        for (MediaResource resource : resources) {
            if (resource.kind.equals("image")) images++;
            if (resource.kind.equals("video")) videos++;
        }
        ((TextView) findViewById(R.id.result_title)).setText(
                resources.isEmpty() ? "提取到笔记文案" : "选择要保存的内容");
        ((TextView) findViewById(R.id.result_summary)).setText(
                images + " 张图片，" + videos + " 个视频" + (savedText.isEmpty() ? "" : "，含笔记文案"));
        saveAll.setVisibility(resources.isEmpty() ? View.GONE : View.VISIBLE);
        findViewById(R.id.selection_actions).setVisibility(resources.isEmpty() ? View.GONE : View.VISIBLE);
        selectionControls.clear();
        saveControls.clear();
        resourceList.removeAllViews();
        for (MediaResource resource : resources) {
            MaterialCardView card = (MaterialCardView) getLayoutInflater()
                    .inflate(R.layout.item_media, resourceList, false);
            resourceList.addView(card, new GridLayout.LayoutParams());
            MaterialCheckBox selected = card.findViewById(R.id.media_selected);
            selected.setText(resource.name);
            selected.setContentDescription("选择" + resource.name);
            selected.setChecked(selectedUrls.contains(resource.url));
            selectionControls.put(selected, resource.url);
            selected.setOnCheckedChangeListener((button, checked) -> {
                if (updatingSelection) return;
                if (checked) selectedUrls.add(resource.url);
                else selectedUrls.remove(resource.url);
                refreshSelection();
            });
            FrameLayout slot = card.findViewById(R.id.media_preview);
            if (resource.kind.equals("image")) {
                slot.addView(createImagePreview(resource, dp(160), false),
                        new FrameLayout.LayoutParams(-1, -1));
            } else if (resource.kind.equals("video")) {
                MaterialButton play = new MaterialButton(this, null,
                        com.google.android.material.R.attr.materialButtonOutlinedStyle);
                play.setText("播放视频");
                play.setIconResource(R.drawable.ic_play);
                play.setIconGravity(MaterialButton.ICON_GRAVITY_TEXT_TOP);
                play.setIconSize(dp(36));
                play.setStrokeWidth(0);
                play.setContentDescription("预览" + resource.name);
                play.setOnClickListener(view -> showVideoPreview(resource));
                slot.addView(play, new FrameLayout.LayoutParams(-1, -1));
            }
            MaterialButton button = card.findViewById(R.id.media_save);
            button.setContentDescription("保存" + resource.name);
            button.setOnClickListener(view -> save(Collections.singletonList(resource)));
            saveControls.put(button, resource.url);
        }
        layoutResourceCards();
        noteText.setText(savedText);
        textExpanded = false;
        noteText.setMaxLines(6);
        noteText.setEllipsize(TextUtils.TruncateAt.END);
        ((MaterialButton) findViewById(R.id.expand_text)).setText("展开全文");
        findViewById(R.id.text_panel).setVisibility(savedText.isEmpty() ? View.GONE : View.VISIBLE);
        findViewById(R.id.expand_text).setVisibility(View.VISIBLE);
        refreshSelection();
        queuePreviewRefresh();
    }

    private void toggleSelection() {
        boolean allSelected = true;
        for (MediaResource resource : resources) {
            if (!DownloadService.isSaved(resource.url) && !DownloadService.isQueued(resource.url)
                    && !selectedUrls.contains(resource.url)) allSelected = false;
        }
        for (MediaResource resource : resources) {
            if (allSelected) selectedUrls.remove(resource.url);
            else if (!DownloadService.isSaved(resource.url)) selectedUrls.add(resource.url);
        }
        refreshSelection();
    }

    private void refreshSelection() {
        selectedUrls.removeIf(DownloadService::isSaved);
        updatingSelection = true;
        for (Map.Entry<MaterialCheckBox, String> entry : selectionControls.entrySet()) {
            boolean saved = DownloadService.isSaved(entry.getValue());
            boolean queued = DownloadService.isQueued(entry.getValue());
            entry.getKey().setChecked(selectedUrls.contains(entry.getValue()));
            entry.getKey().setEnabled(!saved && !queued);
        }
        updatingSelection = false;
        int count = 0;
        int available = 0;
        for (MediaResource resource : resources) {
            if (!DownloadService.isSaved(resource.url) && !DownloadService.isQueued(resource.url)) {
                available++;
                if (selectedUrls.contains(resource.url)) count++;
            }
        }
        ((TextView) findViewById(R.id.selection_count)).setText("已选 " + count + " 项");
        ((MaterialButton) findViewById(R.id.select_all)).setText(
                available > 0 && count == available ? "取消全选" : "全选");
        findViewById(R.id.select_all).setEnabled(available > 0);
        saveAll.setEnabled(count > 0);
        saveAll.setText(count > 0 ? "保存选中内容（" + count + "）" : "请选择要保存的内容");
        for (Map.Entry<MaterialButton, String> entry : saveControls.entrySet()) {
            boolean queued = DownloadService.isQueued(entry.getValue());
            boolean saved = DownloadService.isSaved(entry.getValue());
            entry.getKey().setEnabled(!queued && !saved);
            entry.getKey().setText(saved ? "已保存" : queued ? "保存中…" : "保存");
        }
    }

    private void applyContentSize() {
        View content = findViewById(R.id.content);
        int width = Math.min(scroll.getWidth(), dp(760));
        if (width <= 0) return;
        FrameLayout.LayoutParams params = (FrameLayout.LayoutParams) content.getLayoutParams();
        if (params.width != width) {
            params.width = width;
            params.gravity = Gravity.TOP | Gravity.CENTER_HORIZONTAL;
            content.setLayoutParams(params);
        }
        View container = findViewById(R.id.browser_container);
        int height = Math.max(dp(260), Math.min(dp(440), scroll.getHeight() * 2 / 3));
        if (container.getLayoutParams().height != height) {
            container.getLayoutParams().height = height;
            container.requestLayout();
        }
        layoutResourceCards();
        resizePreviewDialogs();
    }

    private void resizePreviewDialogs() {
        int width = Math.min(Math.max(dp(160), scroll.getWidth() - dp(24)), dp(760));
        for (PreviewBinding binding : previewTargets) {
            if (binding.expanded) {
                binding.frame.getLayoutParams().height = Math.max(dp(160), Math.min(dp(640), scroll.getHeight() * 2 / 3));
                binding.frame.requestLayout();
            }
        }
        if (imageDialog != null && imageDialog.getWindow() != null) imageDialog.getWindow().setLayout(width, -2);
        if (videoDialog != null && videoDialog.getWindow() != null) {
            videoDialog.getWindow().setLayout(width, -2);
            if (videoPreview != null && videoPreview.getParent() instanceof FrameLayout frame) {
                frame.getLayoutParams().height = Math.max(dp(160), Math.min(dp(480), scroll.getHeight() * 2 / 3));
                frame.requestLayout();
            }
        }
    }

    private void layoutResourceCards() {
        int width = findViewById(R.id.content).getWidth();
        int columns = getResources().getConfiguration().fontScale > 1.3f ? 1 : width >= dp(640) ? 3 : 2;
        // 先把旧列位置归零，避免旋转或放大字体时旧列超出新的列数。
        for (int index = 0; index < resourceList.getChildCount(); index++) {
            GridLayout.LayoutParams params = (GridLayout.LayoutParams) resourceList.getChildAt(index).getLayoutParams();
            params.columnSpec = GridLayout.spec(0, 1f);
            resourceList.getChildAt(index).setLayoutParams(params);
        }
        resourceList.setColumnCount(columns);
        for (int index = 0; index < resourceList.getChildCount(); index++) {
            GridLayout.LayoutParams params = (GridLayout.LayoutParams) resourceList.getChildAt(index).getLayoutParams();
            int column = index % columns;
            params.columnSpec = GridLayout.spec(column, GridLayout.FILL, 1f);
            params.rowSpec = GridLayout.spec(index / columns, GridLayout.FILL);
            params.width = 0;
            params.height = LinearLayout.LayoutParams.WRAP_CONTENT;
            params.leftMargin = column == 0 ? 0 : dp(6);
            params.rightMargin = column == columns - 1 ? 0 : dp(6);
            params.bottomMargin = dp(12);
            resourceList.getChildAt(index).setLayoutParams(params);
        }
        queuePreviewRefresh();
    }

    private static final class PreviewBinding {
        final MediaResource resource;
        final FrameLayout frame;
        final ImageView image;
        final TextView hint;
        final boolean expanded;
        boolean requested;
        boolean failed;

        PreviewBinding(MediaResource resource, FrameLayout frame, ImageView image, TextView hint, boolean expanded) {
            this.resource = resource;
            this.frame = frame;
            this.image = image;
            this.hint = hint;
            this.expanded = expanded;
        }
    }

    private final Runnable refreshPreviews = () -> {
        previewRefreshPending = false;
        updateVisiblePreviews();
    };

    private void queuePreviewRefresh() {
        if (scroll == null || previewRefreshPending || isDestroyed()) return;
        previewRefreshPending = true;
        scroll.postDelayed(refreshPreviews, 32);
    }

    private void updateVisiblePreviews() {
        if (!activityResumed || isFinishing() || isDestroyed()) return;
        Rect viewport = new Rect();
        if (!scroll.getGlobalVisibleRect(viewport)) return;
        int[] location = new int[2];
        for (PreviewBinding binding : previewTargets) {
            if (binding.expanded) continue;
            binding.frame.getLocationOnScreen(location);
            boolean near = binding.frame.isShown() && binding.frame.getHeight() > 0
                    && location[1] + binding.frame.getHeight() >= viewport.top - dp(160)
                    && location[1] <= viewport.bottom + dp(160);
            if (near && !binding.requested && !binding.failed) {
                loadImagePreview(binding);
            } else if (!near && binding.requested) {
                binding.requested = false;
                binding.failed = false;
                imageRequests.clear(binding.image);
                binding.hint.setText("滚动到这里加载预览");
                binding.hint.setVisibility(View.VISIBLE);
                binding.frame.setOnClickListener(null);
            }
        }
    }

    private FrameLayout createImagePreview(MediaResource resource, int height, boolean expanded) {
        FrameLayout frame = new FrameLayout(this);
        frame.setLayoutParams(new LinearLayout.LayoutParams(-1, height));
        frame.setBackgroundColor(MaterialColors.getColor(frame,
                com.google.android.material.R.attr.colorSurfaceContainerHigh));
        ImageView image = new ImageView(this);
        image.setScaleType(ImageView.ScaleType.FIT_CENTER);
        image.setContentDescription(resource.name + (expanded ? "大图" : "预览"));
        frame.addView(image, new FrameLayout.LayoutParams(-1, -1));
        TextView hint = new TextView(this);
        hint.setGravity(Gravity.CENTER);
        hint.setPadding(dp(12), dp(12), dp(12), dp(12));
        hint.setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_BodySmall);
        hint.setText("正在准备预览…");
        frame.addView(hint, new FrameLayout.LayoutParams(-1, -1));
        PreviewBinding binding = new PreviewBinding(resource, frame, image, hint, expanded);
        previewTargets.add(binding);
        if (expanded) loadImagePreview(binding);
        else queuePreviewRefresh();
        return frame;
    }

    private void loadImagePreview(PreviewBinding binding) {
        if (!previewTargets.contains(binding) || isFinishing() || isDestroyed()) return;
        binding.requested = true;
        binding.failed = false;
        binding.hint.setText("正在加载图片…");
        binding.hint.setVisibility(View.VISIBLE);
        binding.frame.setOnClickListener(null);
        String userAgent = browser != null && !browser.isClosed() ? browser.userAgent() : "XhsSaver/0.2";
        GlideUrl url = new GlideUrl(binding.resource.url, new LazyHeaders.Builder()
                .setHeader("Referer", TapSaveBrowser.PAGE).setHeader("User-Agent", userAgent).build());
        imageRequests.load(url).override(binding.expanded ? 1440 : 480, binding.expanded ? 1920 : 320)
                .fitCenter().timeout(20000).listener(new RequestListener<Drawable>() {
                    @Override public boolean onLoadFailed(GlideException error, Object model,
                                                           Target<Drawable> target, boolean first) {
                        if (!binding.requested) return true;
                        binding.failed = true;
                        binding.hint.setText(previewError(error));
                        binding.frame.setOnClickListener(view -> loadImagePreview(binding));
                        return false;
                    }

                    @Override public boolean onResourceReady(Drawable drawable, Object model,
                            Target<Drawable> target, DataSource source, boolean first) {
                        if (!binding.requested) return true;
                        binding.hint.setVisibility(View.GONE);
                        if (!binding.expanded) {
                            binding.frame.setContentDescription(binding.resource.name + "，点按查看大图");
                            binding.frame.setOnClickListener(view -> showImagePreview(binding.resource));
                        }
                        return false;
                    }
                }).into(binding.image);
    }

    private static String previewError(GlideException error) {
        if (error != null) {
            for (Throwable cause : error.getRootCauses()) {
                if (cause instanceof HttpException http) {
                    if (http.getStatusCode() == 401 || http.getStatusCode() == 403) {
                        return "预览地址已过期或访问被拒绝，请重新解析。";
                    }
                    if (http.getStatusCode() == 404) return "图片地址已失效，请重新解析。";
                    return "图片服务器暂未返回可用内容，点按重试。";
                }
                if (cause instanceof java.net.SocketTimeoutException) return "图片加载超时，点按重试。";
                if (cause instanceof java.net.UnknownHostException) return "无法连接图片服务器，点按重试。";
                if (cause instanceof javax.net.ssl.SSLException) return "图片安全连接失败，点按重试。";
            }
        }
        return "图片未能加载或解码，点按重试；也可直接保存。";
    }

    private void showImagePreview(MediaResource resource) {
        if (isFinishing() || isDestroyed()) return;
        if (imageDialog != null) imageDialog.dismiss();
        int height = Math.max(dp(160), Math.min(dp(640), scroll.getHeight() * 2 / 3));
        FrameLayout preview = createImagePreview(resource, height, true);
        PreviewBinding binding = previewTargets.get(previewTargets.size() - 1);
        AlertDialog dialog = new MaterialAlertDialogBuilder(this).setTitle(resource.name)
                .setView(preview).setNegativeButton("关闭", null)
                .setPositiveButton("保存图片", (window, which) -> save(Collections.singletonList(resource)))
                .create();
        dialog.setOnDismissListener(window -> {
            binding.requested = false;
            imageRequests.clear(binding.image);
            previewTargets.remove(binding);
            if (imageDialog == dialog) imageDialog = null;
        });
        imageDialog = dialog;
        dialog.show();
        if (dialog.getWindow() != null) {
            dialog.getWindow().setLayout(Math.min(getResources().getDisplayMetrics().widthPixels - dp(24), dp(760)), -2);
        }
    }

    private void showVideoPreview(MediaResource resource) {
        if (isFinishing() || isDestroyed()) return;
        if (videoDialog != null) videoDialog.dismiss();
        FrameLayout frame = new FrameLayout(this);
        frame.setBackgroundColor(Color.BLACK);
        int height = Math.max(dp(160), Math.min(dp(480), scroll.getHeight() * 2 / 3));
        frame.setLayoutParams(new LinearLayout.LayoutParams(-1, height));
        VideoView player = new VideoView(this);
        frame.addView(player, new FrameLayout.LayoutParams(-1, -1, Gravity.CENTER));
        TextView hint = new TextView(this);
        hint.setText("正在准备视频…");
        hint.setTextColor(Color.WHITE);
        hint.setGravity(Gravity.CENTER);
        hint.setPadding(dp(20), dp(16), dp(20), dp(64));
        frame.addView(hint, new FrameLayout.LayoutParams(-1, -1));
        MaterialButton retry = new MaterialButton(this);
        retry.setText("重新加载视频");
        retry.setVisibility(View.GONE);
        FrameLayout.LayoutParams retryParams = new FrameLayout.LayoutParams(-2, -2, Gravity.BOTTOM | Gravity.CENTER_HORIZONTAL);
        retryParams.bottomMargin = dp(8);
        frame.addView(retry, retryParams);
        MediaController controls = new MediaController(this);
        controls.setAnchorView(frame);
        player.setMediaController(controls);
        Map<String, String> headers = new HashMap<>();
        headers.put("Referer", TapSaveBrowser.PAGE);
        headers.put("User-Agent", browser != null && !browser.isClosed() ? browser.userAgent() : "XhsSaver/0.2");
        Runnable load = () -> {
            hint.setText("正在准备视频…");
            hint.setVisibility(View.VISIBLE);
            retry.setVisibility(View.GONE);
            player.setVideoURI(Uri.parse(resource.url), headers);
        };
        player.setOnPreparedListener(media -> {
            if (videoPreview != player) return;
            hint.setVisibility(View.GONE);
            if (activityResumed) {
                player.start();
                controls.show(3000);
            }
        });
        player.setOnErrorListener((media, what, extra) -> {
            if (videoPreview != player) return true;
            hint.setText("视频预览未能加载，请检查网络或重新解析。也可先保存，再用系统播放器打开。");
            hint.setVisibility(View.VISIBLE);
            retry.setVisibility(View.VISIBLE);
            return true;
        });
        retry.setOnClickListener(view -> load.run());
        AlertDialog dialog = new MaterialAlertDialogBuilder(this).setTitle(resource.name)
                .setView(frame).setNegativeButton("关闭", null)
                .setPositiveButton("保存视频", (window, which) -> save(Collections.singletonList(resource)))
                .create();
        dialog.setOnDismissListener(window -> {
            controls.hide();
            player.stopPlayback();
            if (videoDialog == dialog) {
                videoDialog = null;
                videoPreview = null;
                videoController = null;
            }
        });
        videoDialog = dialog;
        videoPreview = player;
        videoController = controls;
        dialog.show();
        if (dialog.getWindow() != null) {
            dialog.getWindow().setLayout(Math.min(getResources().getDisplayMetrics().widthPixels - dp(24), dp(760)), -2);
        }
        load.run();
    }

    private void clearImagePreviews() {
        if (scroll != null) scroll.removeCallbacks(refreshPreviews);
        previewRefreshPending = false;
        if (imageDialog != null) {
            imageDialog.dismiss();
            imageDialog = null;
        }
        if (videoDialog != null) {
            videoDialog.dismiss();
            videoDialog = null;
            videoPreview = null;
            videoController = null;
        }
        for (PreviewBinding binding : new ArrayList<>(previewTargets)) {
            binding.requested = false;
            imageRequests.clear(binding.image);
        }
        previewTargets.clear();
    }

    private void save(List<MediaResource> selection) {
        if (selection.isEmpty()) return;
        List<MediaResource> pending = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        for (MediaResource resource : selection) {
            if (!DownloadService.isSaved(resource.url) && !DownloadService.isQueued(resource.url)
                    && seen.add(resource.url)) pending.add(resource);
        }
        if (pending.isEmpty()) {
            message("所选内容已保存或正在保存。");
            refreshSelection();
            return;
        }
        try {
            String payload = MediaResource.toArray(pending);
            if (payload.length() > 300000) {
                message("本次资源地址过长，请使用每项右侧的保存按钮分开保存。");
                return;
            }
            ArrayList<String> cookies = new ArrayList<>();
            for (MediaResource resource : pending) cookies.add(CookieManager.getInstance().getCookie(resource.url));
            Intent task = new Intent(this, DownloadService.class).setAction(DownloadService.ACTION_SAVE)
                    .putExtra("resources", payload)
                    .putStringArrayListExtra("cookies", cookies)
                    .putExtra("userAgent", browser != null && !browser.isClosed() ? browser.userAgent() : "XhsSaver/0.2");
            startForegroundService(task);
            saveAll.setEnabled(false);
            findViewById(R.id.download_panel).setVisibility(View.VISIBLE);
            downloadProgress.setVisibility(View.VISIBLE);
            findViewById(R.id.cancel_save).setVisibility(View.VISIBLE);
            downloadStatus.setText("已加入保存队列，切换回小红书也会继续保存。");
            requestNotificationPermission();
        } catch (Exception error) {
            message("未能开始保存，请保持应用在前台后重试。");
        }
    }

    private void requestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
                && !getPreferences(MODE_PRIVATE).getBoolean("notificationAsked", false)) {
            getPreferences(MODE_PRIVATE).edit().putBoolean("notificationAsked", true).apply();
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 1);
        }
    }

    private void setParsing(boolean parsing) {
        progress.setVisibility(parsing ? View.VISIBLE : View.GONE);
        inputLayout.setEnabled(!parsing);
        findViewById(R.id.cancel_parse).setVisibility(parsing ? View.VISIBLE : View.GONE);
        parseButton.setEnabled(!parsing);
        findViewById(R.id.paste_button).setEnabled(!parsing);
        parseButton.setText(parsing ? "正在解析…" : "解析输入");
    }

    private void setParseStatus(String message, boolean error) {
        status.setText(message);
        status.setTextColor(MaterialColors.getColor(status,
                error ? com.google.android.material.R.attr.colorError : com.google.android.material.R.attr.colorOnSurfaceVariant));
    }

    private void refreshDownloadState() {
        DownloadService.Status state = DownloadService.getStatus();
        if (state != null) {
            findViewById(R.id.download_panel).setVisibility(View.VISIBLE);
            downloadStatus.setText(state.message);
            boolean active = DownloadService.isRunning() && !state.finished;
            boolean unknown = state.percent < 0;
            if (downloadProgress.isIndeterminate() != unknown) {
                downloadProgress.setVisibility(View.INVISIBLE);
                downloadProgress.setIndeterminate(unknown);
            }
            downloadProgress.setVisibility(active ? View.VISIBLE : View.GONE);
            if (active && !unknown) downloadProgress.setProgressCompat(state.percent, true);
            findViewById(R.id.cancel_save).setVisibility(active ? View.VISIBLE : View.GONE);
        }
        refreshSelection();
    }

    private void showBrowser(boolean visible) {
        browserPanel.setVisibility(visible ? View.VISIBLE : View.GONE);
        browserToggle.setText(visible ? "收起解析网页" : "显示解析网页");
        queuePreviewRefresh();
    }

    private void startOverlay() {
        try {
            startForegroundService(new Intent(this, OverlayService.class));
            requestNotificationPermission();
        } catch (RuntimeException error) {
            message("系统未允许开启悬浮球，仍可使用分享保存。");
            refreshOverlaySwitch();
        }
    }

    private void refreshOverlaySwitch() {
        updatingSwitch = true;
        overlaySwitch.setChecked(Settings.canDrawOverlays(this) && OverlayService.isRunning());
        updatingSwitch = false;
    }

    private void message(String text) { Snackbar.make(findViewById(R.id.root), text, Snackbar.LENGTH_LONG).show(); }

    private int dp(int value) { return Math.round(value * getResources().getDisplayMetrics().density); }

    @Override public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) consumePendingPaste();
    }

    private void consumePendingPaste() {
        if (pendingPaste && input != null && hasWindowFocus() && !isFinishing() && !isDestroyed()) {
            pendingPaste = false;
            input.post(() -> paste(true));
        }
    }

    @Override protected void onResume() {
        super.onResume();
        activityResumed = true;
        if (browser != null) browser.resume();
        if (!receiverRegistered) {
            IntentFilter filter = new IntentFilter(DownloadService.ACTION_PROGRESS);
            filter.addAction(OverlayService.ACTION_STATE);
            if (Build.VERSION.SDK_INT >= 33) registerReceiver(downloadReceiver, filter, Context.RECEIVER_NOT_EXPORTED);
            else registerReceiver(downloadReceiver, filter);
            receiverRegistered = true;
        }
        if (pendingOverlay) {
            pendingOverlay = false;
            if (Settings.canDrawOverlays(this)) startOverlay();
            else message("未开启悬浮窗权限，分享和粘贴保存仍可使用。");
        }
        refreshOverlaySwitch();
        refreshDownloadState();
        queuePreviewRefresh();
        if (videoController != null) videoController.show(3000);
    }

    @Override protected void onPause() {
        activityResumed = false;
        scroll.removeCallbacks(refreshPreviews);
        previewRefreshPending = false;
        if (videoPreview != null) videoPreview.pause();
        if (videoController != null) videoController.hide();
        if (browser != null) browser.pause();
        if (receiverRegistered) {
            unregisterReceiver(downloadReceiver);
            receiverRegistered = false;
        }
        super.onPause();
    }

    @Override protected void onSaveInstanceState(Bundle state) {
        state.putString("input", String.valueOf(input.getText()));
        state.putString("text", savedText);
        state.putBoolean("pendingOverlay", pendingOverlay);
        ArrayList<Integer> selection = new ArrayList<>();
        for (int index = 0; index < resources.size(); index++) {
            if (selectedUrls.contains(resources.get(index).url)) selection.add(index);
        }
        state.putIntegerArrayList("selection", selection);
        try {
            String payload = MediaResource.toArray(resources);
            if (payload.length() <= 300000) state.putString("resources", payload);
        } catch (Exception error) {
            state.putString("resources", "[]");
        }
        super.onSaveInstanceState(state);
    }

    @Override protected void onDestroy() {
        clearImagePreviews();
        if (browser != null) browser.close();
        super.onDestroy();
    }
}
