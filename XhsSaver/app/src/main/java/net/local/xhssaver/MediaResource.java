// 表示网页返回且用户可选择保存的一项 HTTPS 媒体资源。
package net.local.xhssaver;

import android.net.Uri;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

final class MediaResource {
    final String url;
    final String kind;
    final String name;

    private MediaResource(String url, String kind, String name) {
        this.url = url;
        this.kind = kind;
        this.name = name;
    }

    static MediaResource fromJson(JSONObject value) throws JSONException {
        String url = value.optString("url");
        Uri parsed = Uri.parse(url);
        if (url.length() > 32768 || !"https".equalsIgnoreCase(parsed.getScheme())
                || parsed.getHost() == null || parsed.getUserInfo() != null) {
            throw new JSONException("下载地址必须是有效的 HTTPS 链接");
        }
        String kind = value.optString("kind", "file");
        if (!kind.equals("image") && !kind.equals("video") && !kind.equals("file")) {
            throw new JSONException("无法识别媒体类型");
        }
        String name = value.optString("name", "媒体文件");
        return new MediaResource(url, kind, name.substring(0, Math.min(name.length(), 160)));
    }

    JSONObject toJson() throws JSONException {
        return new JSONObject().put("url", url).put("kind", kind).put("name", name);
    }

    static List<MediaResource> fromArray(JSONArray values) throws JSONException {
        if (values.length() > 200) throw new JSONException("单次最多保存 200 项资源");
        List<MediaResource> resources = new ArrayList<>();
        for (int index = 0; index < values.length(); index++) {
            resources.add(fromJson(values.getJSONObject(index)));
        }
        return resources;
    }

    static String toArray(List<MediaResource> resources) throws JSONException {
        JSONArray values = new JSONArray();
        for (MediaResource resource : resources) values.put(resource.toJson());
        return values.toString();
    }
}
