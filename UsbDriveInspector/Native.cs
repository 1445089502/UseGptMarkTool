// 封装 Windows 句柄、进程身份和权限接口；调用方负责释放所有原生句柄。
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Win32.SafeHandles;

namespace UsbDriveInspector
{
    internal static class Native
    {
        [StructLayout(LayoutKind.Sequential)]
        internal struct HandleEntry
        {
            public IntPtr Object;
            public UIntPtr ProcessId;
            public UIntPtr Handle;
            public uint GrantedAccess;
            public ushort CreatorBackTraceIndex;
            public ushort ObjectTypeIndex;
            public uint Attributes;
            public uint Reserved;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct FileTime { public uint Low; public uint High; }

        [StructLayout(LayoutKind.Sequential)]
        private struct TokenPrivilege
        {
            public uint Count;
            public uint LuidLow;
            public int LuidHigh;
            public uint Attributes;
        }

        [DllImport("ntdll.dll")]
        internal static extern int NtQuerySystemInformation(int infoClass, IntPtr buffer, int size, out int required);
        [DllImport("kernel32.dll", SetLastError = true)]
        internal static extern IntPtr OpenProcess(uint access, bool inherit, int pid);
        [DllImport("kernel32.dll", SetLastError = true)]
        internal static extern bool CloseHandle(IntPtr handle);
        [DllImport("kernel32.dll")]
        internal static extern IntPtr GetCurrentProcess();
        [DllImport("kernel32.dll", SetLastError = true)]
        internal static extern bool DuplicateHandle(IntPtr sourceProcess, IntPtr sourceHandle, IntPtr targetProcess,
            out IntPtr targetHandle, uint access, bool inherit, uint options);
        [DllImport("kernel32.dll", SetLastError = true)]
        internal static extern uint GetFileType(IntPtr file);
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        internal static extern uint GetFinalPathNameByHandle(IntPtr file, StringBuilder path, uint size, uint flags);
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        internal static extern uint QueryDosDevice(string device, StringBuilder target, int size);
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        internal static extern bool QueryFullProcessImageName(IntPtr process, uint flags, StringBuilder path, ref int size);
        [DllImport("kernel32.dll", SetLastError = true)]
        internal static extern bool GetProcessTimes(IntPtr process, out FileTime created, out FileTime exited,
            out FileTime kernel, out FileTime user);
        [DllImport("kernel32.dll", SetLastError = true)]
        internal static extern bool IsProcessCritical(IntPtr process, out bool critical);
        [DllImport("kernel32.dll", SetLastError = true)]
        internal static extern bool TerminateProcess(IntPtr process, uint exitCode);
        [DllImport("kernel32.dll")]
        internal static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);
        [DllImport("kernel32.dll", SetLastError = true)]
        internal static extern bool ProcessIdToSessionId(uint pid, out uint session);
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        internal static extern SafeFileHandle CreateFile(string name, uint access, uint share, IntPtr security,
            uint disposition, uint flags, IntPtr template);
        [DllImport("advapi32.dll", SetLastError = true)]
        private static extern bool OpenProcessToken(IntPtr process, uint access, out IntPtr token);
        [DllImport("advapi32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern bool LookupPrivilegeValue(string system, string name, out long luid);
        [DllImport("advapi32.dll", SetLastError = true)]
        private static extern bool AdjustTokenPrivileges(IntPtr token, bool disableAll, ref TokenPrivilege state,
            int length, IntPtr previous, IntPtr returned);

        internal static long CreationTime(IntPtr process)
        {
            FileTime created, exited, kernel, user;
            if (!GetProcessTimes(process, out created, out exited, out kernel, out user)) return 0;
            return ((long)created.High << 32) | created.Low;
        }

        internal static string ImagePath(IntPtr process)
        {
            var path = new StringBuilder(32768);
            int length = path.Capacity;
            return QueryFullProcessImageName(process, 0, path, ref length) ? path.ToString() : "";
        }

        internal static bool EnableDebugPrivilege()
        {
            IntPtr token;
            if (!OpenProcessToken(GetCurrentProcess(), 0x28, out token)) return false;
            try
            {
                long luid;
                if (!LookupPrivilegeValue(null, "SeDebugPrivilege", out luid)) return false;
                var state = new TokenPrivilege { Count = 1, LuidLow = (uint)luid,
                    LuidHigh = (int)(luid >> 32), Attributes = 2 };
                return AdjustTokenPrivileges(token, false, ref state, 0, IntPtr.Zero, IntPtr.Zero)
                    && Marshal.GetLastWin32Error() == 0;
            }
            finally { CloseHandle(token); }
        }

        internal static Exception Error(string action)
        {
            return new Win32Exception(Marshal.GetLastWin32Error(), action + "：" +
                new Win32Exception(Marshal.GetLastWin32Error()).Message);
        }
    }
}
