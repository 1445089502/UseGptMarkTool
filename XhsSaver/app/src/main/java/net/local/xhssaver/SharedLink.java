// 从小红书分享文字中提取链接，避免把整段文字当作下载地址。
package net.local.xhssaver;

import android.net.Uri;

import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

final class SharedLink {
    // 保留链接中的查询参数和百分号编码，以中文、空白及非链接字符分隔分享文案。
    private static final Pattern URL = Pattern.compile(
            "https?://[a-z0-9\\-._~:/?#\\[\\]@!$&'()*+,;=%]+", Pattern.CASE_INSENSITIVE);
    private static final String TRAILING = "，。！？；：、）】》〉」』,.!?;:)]}'";

    private SharedLink() { }

    static String extract(String text) {
        if (text == null || text.length() > 20000) return null;
        Matcher matcher = URL.matcher(text);
        while (matcher.find()) {
            String candidate = matcher.group();
            while (!candidate.isEmpty() && TRAILING.indexOf(candidate.charAt(candidate.length() - 1)) >= 0) {
                candidate = candidate.substring(0, candidate.length() - 1);
            }
            Uri uri = Uri.parse(candidate);
            String host = uri.getHost();
            if (host == null || uri.getUserInfo() != null || candidate.length() > 8192) continue;
            host = host.toLowerCase(Locale.ROOT);
            if (host.equals("xiaohongshu.com") || host.endsWith(".xiaohongshu.com")
                    || host.equals("xhslink.com") || host.endsWith(".xhslink.com")
                    || host.equals("xhslink.cn") || host.endsWith(".xhslink.cn")) {
                return candidate;
            }
        }
        return null;
    }
}
