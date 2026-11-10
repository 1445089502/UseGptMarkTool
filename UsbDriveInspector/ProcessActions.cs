// 在同一个进程句柄上核对创建时间并执行操作，防止 PID 重用误伤其他程序。
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace UsbDriveInspector
{
    public sealed class ProcessAction : IDisposable
    {
        private IntPtr handle;
        private readonly int pid;
        private readonly long created;

        public ProcessAction(Occupancy row, bool terminate)
        {
            pid = row.Pid;
            created = row.Created;
            if (pid <= 4 || pid == Process.GetCurrentProcess().Id || created == 0)
                throw new InvalidOperationException("系统进程、工具自身或身份无法核实的进程不能在这里结束。");
            handle = Native.OpenProcess(0x1000 | 0x100000u | (terminate ? 1u : 0u), false, pid);
            if (handle == IntPtr.Zero) throw Native.Error("无法操作此进程，它可能已退出或需要管理员权限");
            try
            {
                EnsureIdentity();
                bool critical;
                if (!Native.IsProcessCritical(handle, out critical)) throw Native.Error("无法确认进程安全状态");
                uint session;
                if (!Native.ProcessIdToSessionId((uint)pid, out session)) throw Native.Error("无法确认进程所属会话");
                string name = Path.GetFileNameWithoutExtension(Native.ImagePath(handle));
                string[] protectedNames = { "System", "Registry", "smss", "csrss", "wininit", "services", "lsass",
                    "winlogon", "svchost", "dwm", "fontdrvhost", "sihost", "explorer", "UsbDriveInspector" };
                if (critical || session == 0 || protectedNames.Contains(name, StringComparer.OrdinalIgnoreCase))
                    throw new InvalidOperationException("该进程属于系统、服务或桌面组件，已禁止结束。请关闭相关窗口，或通过应用 / 服务自身退出。");
            }
            catch { Dispose(); throw; }
        }

        private void EnsureIdentity()
        {
            if (handle == IntPtr.Zero || Native.WaitForSingleObject(handle, 0) == 0 || Native.CreationTime(handle) != created)
                throw new InvalidOperationException("原进程已退出或 PID 已被重新使用，请重新扫描。");
        }

        public string CloseNormally()
        {
            EnsureIdentity();
            using (var process = Process.GetProcessById(pid))
            {
                if (process.StartTime.ToUniversalTime().ToFileTimeUtc() != created)
                    throw new InvalidOperationException("进程身份已变化，请重新扫描。");
                if (!process.CloseMainWindow()) return "该进程没有可正常关闭的主窗口，请先从应用自身退出。";
                return "已请求正常关闭。请处理该应用的保存提示，然后重新扫描。";
            }
        }

        public string Terminate()
        {
            EnsureIdentity();
            if (!Native.TerminateProcess(handle, 1)) throw Native.Error("结束进程失败");
            return Native.WaitForSingleObject(handle, 3000) == 0
                ? "已结束所选进程，正在重新扫描。"
                : "已发送结束请求，进程仍在退出，请稍后重新扫描。";
        }

        public void Dispose()
        {
            if (handle != IntPtr.Zero) { Native.CloseHandle(handle); handle = IntPtr.Zero; }
        }
    }
}
