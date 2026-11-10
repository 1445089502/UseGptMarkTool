// 枚举文件句柄与已加载程序；隔离扫描进程使异常驱动查询可被取消或限时结束。
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Web.Script.Serialization;

namespace UsbDriveInspector
{
    public sealed class Occupancy
    {
        public int Pid { get; set; }
        public long Created { get; set; }
        public string ProcessName { get; set; }
        public string Image { get; set; }
        public string Path { get; set; }
        public string Kind { get; set; }
        public int Count { get; set; }
    }

    public sealed class ScanResult
    {
        public List<Occupancy> Rows = new List<Occupancy>();
        public List<string> Warnings = new List<string>();
        public bool Completed;
        public bool Cancelled;
        public double Seconds;
    }

    internal sealed class ScanMessage
    {
        public string Type { get; set; }
        public Occupancy Row { get; set; }
        public string Text { get; set; }
        public int Denied { get; set; }
        public int Unresolved { get; set; }
        public int ModuleDenied { get; set; }
    }

    internal static class HandleScanner
    {
        private static readonly JavaScriptSerializer Json = new JavaScriptSerializer();

        private static void Emit(ScanMessage message)
        {
            Console.WriteLine(Json.Serialize(message));
            Console.Out.Flush();
        }

        internal static string MapPath(string path, string device, string drive)
        {
            if (string.IsNullOrEmpty(path)) return null;
            if (path.StartsWith(@"\\?\", StringComparison.Ordinal)) path = path.Substring(4);
            if (path.Equals(device, StringComparison.OrdinalIgnoreCase)) return drive;
            if (path.StartsWith(device + "\\", StringComparison.OrdinalIgnoreCase))
                return drive.TrimEnd('\\') + path.Substring(device.Length);
            if (path.StartsWith(drive, StringComparison.OrdinalIgnoreCase)) return path;
            return null;
        }

        private static void EmitRow(int pid, long created, string image, string path, string kind)
        {
            Emit(new ScanMessage { Type = "row", Row = new Occupancy {
                Pid = pid, Created = created, Image = image,
                ProcessName = string.IsNullOrEmpty(image) ? "进程 " + pid : System.IO.Path.GetFileName(image),
                Path = path, Kind = kind, Count = 1
            }});
        }

        internal static int RunWorker(string requestedDrive, int shard, int shards, string requestedFile = null)
        {
            int denied = 0, unresolved = 0, moduleDenied = 0;
            try
            {
                string drive = Program.NormalizeDrive(requestedDrive);
                string targetFile = null;
                bool targetIsDirectory = false;
                if (requestedFile != null)
                {
                    targetFile = ScanService.NormalizeFilePath(requestedFile);
                    targetIsDirectory = (File.GetAttributes(targetFile) & FileAttributes.Directory) != 0;
                    // 只读取路径信息，不读取正文；尽量解析目录链接与短文件名，且不保留目标句柄。
                    using (var target = Native.CreateFile(targetFile, 0, 7, IntPtr.Zero, 3,
                        targetIsDirectory ? 0x02000000u : 0u, IntPtr.Zero))
                    {
                        if (!target.IsInvalid)
                        {
                            var resolved = new StringBuilder(32768);
                            uint length = Native.GetFinalPathNameByHandle(target.DangerousGetHandle(), resolved,
                                (uint)resolved.Capacity, 0);
                            if (length > 0 && length < resolved.Capacity)
                                targetFile = ScanService.NormalizeFilePath(resolved.ToString());
                        }
                        // 独占或权限限制下仍可按用户给出的路径扫描，不能把打开失败当作没有占用。
                    }
                    drive = Program.NormalizeDrive(Path.GetPathRoot(targetFile));
                }
                var deviceBuffer = new StringBuilder(32768);
                if (Native.QueryDosDevice(drive.Substring(0, 2), deviceBuffer, deviceBuffer.Capacity) == 0)
                    throw Native.Error("无法读取磁盘映射，磁盘可能已移除");
                string device = deviceBuffer.ToString();
                Native.EnableDebugPrivilege();
                int self = Process.GetCurrentProcess().Id;

                // 运行中的 EXE / DLL 可能没有普通文件句柄，另行检查已加载模块。
                foreach (var process in Process.GetProcesses())
                {
                    using (process)
                    {
                        int pid = process.Id;
                        if (pid == 0 || pid == self || pid % shards != shard) continue;
                        IntPtr handle = Native.OpenProcess(0x1000, false, pid);
                        if (handle == IntPtr.Zero) { moduleDenied++; continue; }
                        try
                        {
                            long created = Native.CreationTime(handle);
                            string image = Native.ImagePath(handle);
                            string imageMatch = MapPath(image, device, drive);
                            if (MatchesTarget(imageMatch, targetFile, targetIsDirectory)) EmitRow(pid, created, image, imageMatch, "运行中的程序");
                            try
                            {
                                foreach (ProcessModule module in process.Modules)
                                {
                                    string match = MapPath(module.FileName, device, drive);
                                    if (MatchesTarget(match, targetFile, targetIsDirectory) && !string.Equals(match, imageMatch, StringComparison.OrdinalIgnoreCase))
                                        EmitRow(pid, created, image, match, "加载的模块");
                                }
                            }
                            catch (Win32Exception) { moduleDenied++; }
                            catch (InvalidOperationException) { moduleDenied++; }
                        }
                        finally { Native.CloseHandle(handle); }
                    }
                }

                // 用本进程的已知文件句柄确定 File 类型编号，避免依赖不同 Windows 版本的固定值。
                using (var probe = File.Open(Program.Executable, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete))
                {
                    int size = 1024 * 1024;
                    IntPtr buffer = IntPtr.Zero;
                    try
                    {
                        int required;
                        while (true)
                        {
                            buffer = Marshal.AllocHGlobal(size);
                            int status = Native.NtQuerySystemInformation(64, buffer, size, out required);
                            if (status == 0) break;
                            Marshal.FreeHGlobal(buffer);
                            buffer = IntPtr.Zero;
                            if (status != unchecked((int)0xC0000004))
                                throw new InvalidOperationException("Windows 句柄快照失败，状态 0x" + status.ToString("X8"));
                            size = Math.Max(size * 2, required + 65536);
                            if (size > 256 * 1024 * 1024) throw new InvalidOperationException("系统句柄数量超过扫描上限。");
                        }
                        long count = Marshal.ReadIntPtr(buffer).ToInt64();
                        int stride = Marshal.SizeOf(typeof(Native.HandleEntry));
                        if (count < 0 || count > (size - IntPtr.Size * 2) / stride)
                            throw new InvalidOperationException("Windows 返回了不兼容的句柄快照。");
                        ushort fileType = 0;
                        long ownHandle = probe.SafeFileHandle.DangerousGetHandle().ToInt64();
                        for (long i = 0; i < count; i++)
                        {
                            var entry = (Native.HandleEntry)Marshal.PtrToStructure(
                                IntPtr.Add(buffer, checked(IntPtr.Size * 2 + (int)i * stride)), typeof(Native.HandleEntry));
                            if ((long)entry.ProcessId.ToUInt64() == self && (long)entry.Handle.ToUInt64() == ownHandle)
                            { fileType = entry.ObjectTypeIndex; break; }
                        }
                        if (fileType == 0) throw new InvalidOperationException("未能识别 Windows 文件句柄类型。");
                        var groups = new Dictionary<int, List<Native.HandleEntry>>();
                        for (long i = 0; i < count; i++)
                        {
                            var entry = (Native.HandleEntry)Marshal.PtrToStructure(
                                IntPtr.Add(buffer, checked(IntPtr.Size * 2 + (int)i * stride)), typeof(Native.HandleEntry));
                            ulong rawPid = entry.ProcessId.ToUInt64();
                            if (entry.ObjectTypeIndex != fileType || rawPid > int.MaxValue) continue;
                            int pid = (int)rawPid;
                            if (pid == 0 || pid == self || pid % shards != shard) continue;
                            List<Native.HandleEntry> entries;
                            if (!groups.TryGetValue(pid, out entries)) groups.Add(pid, entries = new List<Native.HandleEntry>());
                            entries.Add(entry);
                        }
                        // 按 PID 分组复用进程句柄；DuplicateHandle 从不关闭被检查程序的原句柄。
                        foreach (var group in groups)
                        {
                            IntPtr process = Native.OpenProcess(0x1040, false, group.Key);
                            if (process == IntPtr.Zero) { denied++; continue; }
                            try
                            {
                                long created = Native.CreationTime(process);
                                string image = Native.ImagePath(process);
                                foreach (var entry in group.Value)
                                {
                                    IntPtr copy;
                                    if (!Native.DuplicateHandle(process, new IntPtr(unchecked((long)entry.Handle.ToUInt64())),
                                        Native.GetCurrentProcess(), out copy, 0, false, 2)) { unresolved++; continue; }
                                    try
                                    {
                                        // 排除管道、控制台和套接字；磁盘查询仍由父进程提供超时保护。
                                        if (Native.GetFileType(copy) != 1) continue;
                                        var path = new StringBuilder(32768);
                                        uint length = Native.GetFinalPathNameByHandle(copy, path, (uint)path.Capacity, 0xA);
                                        if (length == 0 || length >= path.Capacity) { unresolved++; continue; }
                                        string match = MapPath(path.ToString(), device, drive);
                                        if (MatchesTarget(match, targetFile, targetIsDirectory)) EmitRow(group.Key, created, image, match, "文件 / 文件夹");
                                    }
                                    finally { Native.CloseHandle(copy); }
                                }
                            }
                            finally { Native.CloseHandle(process); }
                        }
                    }
                    finally { if (buffer != IntPtr.Zero) Marshal.FreeHGlobal(buffer); }
                }
                Emit(new ScanMessage { Type = "done", Denied = denied, Unresolved = unresolved, ModuleDenied = moduleDenied });
                return 0;
            }
            catch (Exception ex)
            {
                Emit(new ScanMessage { Type = "error", Text = ex.Message });
                return 1;
            }
        }

        internal static bool MatchesTarget(string path, string targetFile, bool isDirectory)
        {
            if (path == null) return false;
            if (targetFile == null) return true;
            if (!isDirectory) return string.Equals(path, targetFile, StringComparison.OrdinalIgnoreCase);
            string directory = targetFile.TrimEnd('\\');
            // 以目录分隔符作为边界，防止把同级的 report-old 误认为 report 的子目录。
            return string.Equals(path.TrimEnd('\\'), directory, StringComparison.OrdinalIgnoreCase) ||
                path.StartsWith(directory + "\\", StringComparison.OrdinalIgnoreCase);
        }
    }

    public static class ScanService
    {
        public static Task<ScanResult> ScanAsync(string drive, CancellationToken token)
        {
            return ScanCoreAsync(Program.NormalizeDrive(drive), null, token);
        }

        /// <summary>检查文件或文件夹；文件夹包含自身及后代路径。保留原接口名称与取消后返回部分结果的行为。</summary>
        public static Task<ScanResult> ScanFileAsync(string filePath, CancellationToken token)
        {
            string file = NormalizeFilePath(filePath);
            return ScanCoreAsync(Path.GetPathRoot(file), file, token);
        }

        internal static string NormalizeFilePath(string filePath)
        {
            string file = (filePath ?? "").Trim();
            if (file.Length >= 2 && file[0] == '"' && file[file.Length - 1] == '"')
                file = file.Substring(1, file.Length - 2);
            if (file.StartsWith(@"\\?\", StringComparison.Ordinal)) file = file.Substring(4);
            if (file.Length < 3 || file[1] != ':' || (file[2] != '\\' && file[2] != '/') ||
                !char.IsLetter(file[0]) || file[0] > 127)
                throw new ArgumentException("请选择本机或 U 盘上的文件 / 文件夹，或输入带盘符的完整路径；暂不支持网络共享路径。");
            file = Path.GetFullPath(file);
            if (file.IndexOfAny(new[] { '*', '?', '"' }) >= 0 || file.IndexOf(':', 2) >= 0)
                throw new ArgumentException("请输入文件或文件夹的完整路径，不能使用通配符。");
            return file.Length > 3 ? file.TrimEnd('\\') : file;
        }

        private static async Task<ScanResult> ScanCoreAsync(string drive, string targetFile, CancellationToken token)
        {
            drive = Program.NormalizeDrive(drive);
            var watch = Stopwatch.StartNew();
            var result = new ScanResult();
            object gate = new object();
            int finished = 0, denied = 0, unresolved = 0, moduleDenied = 0;
            var tasks = new List<Task>();
            // 分为三个互不重叠的进程组，单个驱动查询卡住时仍保留其他组的结果。
            for (int shard = 0; shard < 3; shard++)
            {
                int partition = shard;
                tasks.Add(Task.Run(() =>
                {
                    using (var worker = new Process())
                    {
                        // 文件路径用 UTF-8 Base64 传给子进程，保留空格、中文，且不经过 shell 解释。
                        string arguments = targetFile == null ? "--scan-worker " + drive.Substring(0, 2) :
                            "--scan-file-worker " + Convert.ToBase64String(Encoding.UTF8.GetBytes(targetFile));
                        worker.StartInfo = new ProcessStartInfo(Program.Executable,
                            arguments + " " + partition + " 3") {
                            UseShellExecute = false, CreateNoWindow = true,
                            WorkingDirectory = Path.GetDirectoryName(Program.Executable),
                            RedirectStandardOutput = true, RedirectStandardError = true,
                            StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8
                        };
                        bool done = false;
                        var serializer = new JavaScriptSerializer();
                        worker.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e)
                        {
                            if (string.IsNullOrEmpty(e.Data)) return;
                            lock (gate)
                            {
                                try
                                {
                                    var message = serializer.Deserialize<ScanMessage>(e.Data);
                                    if (message.Type == "row") result.Rows.Add(message.Row);
                                    else if (message.Type == "done")
                                    {
                                        done = true;
                                        denied += message.Denied;
                                        unresolved += message.Unresolved;
                                        moduleDenied += message.ModuleDenied;
                                    }
                                    else if (message.Type == "error") result.Warnings.Add(message.Text);
                                }
                                catch (Exception ex) { result.Warnings.Add("扫描结果解析失败：" + ex.Message); }
                            }
                        };
                        worker.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e)
                        {
                            if (!string.IsNullOrWhiteSpace(e.Data))
                                lock (gate) result.Warnings.Add("扫描进程：" + e.Data);
                        };
                        try
                        {
                            if (token.IsCancellationRequested) return;
                            worker.Start();
                            worker.BeginOutputReadLine();
                            worker.BeginErrorReadLine();
                            var elapsed = Stopwatch.StartNew();
                            while (!worker.WaitForExit(100))
                            {
                                if (token.IsCancellationRequested || elapsed.ElapsedMilliseconds > 25000)
                                {
                                    worker.Kill();
                                    lock (gate)
                                        if (!token.IsCancellationRequested)
                                            result.Warnings.Add("第 " + (partition + 1) + " 组查询超过 25 秒，已停止；结果不完整。");
                                    break;
                                }
                            }
                            if (!worker.WaitForExit(3000))
                            {
                                lock (gate) result.Warnings.Add("扫描进程仍在等待系统 I/O 清理；当前结果不完整，请稍后重试。");
                                return;
                            }
                            worker.WaitForExit();
                            lock (gate)
                            {
                                if (done && worker.ExitCode == 0) finished++;
                                else if (!token.IsCancellationRequested)
                                    result.Warnings.Add("第 " + (partition + 1) + " 组未完成，请重新扫描或使用管理员模式。");
                            }
                        }
                        catch (Exception ex)
                        {
                            lock (gate) result.Warnings.Add("无法完成扫描：" + ex.Message);
                        }
                        finally
                        {
                            try { if (!worker.HasExited) { worker.Kill(); worker.WaitForExit(3000); } }
                            catch (InvalidOperationException) { /* 未启动或已退出，无需释放子进程。 */ }
                            catch (Win32Exception ex) { lock (gate) result.Warnings.Add("扫描进程清理失败：" + ex.Message); }
                        }
                    }
                }));
            }
            await Task.WhenAll(tasks).ConfigureAwait(false);
            result.Cancelled = token.IsCancellationRequested;
            result.Completed = finished == 3 && !result.Cancelled;
            result.Seconds = watch.Elapsed.TotalSeconds;
            result.Rows = result.Rows.GroupBy(r => r.Pid + "|" + r.Created + "|" + r.Kind + "|" + r.Path,
                StringComparer.OrdinalIgnoreCase).Select(g => { var row = g.First(); row.Count = g.Sum(r => r.Count); return row; })
                .OrderBy(r => r.ProcessName).ThenBy(r => r.Pid).ThenBy(r => r.Path).ToList();
            if (denied + moduleDenied + unresolved > 0)
                result.Warnings.Add(string.Format("全机扫描限制：{0} 个进程的文件句柄不可读取，{1} 个进程的模块不可读取，{2} 个句柄已变化或无法解析。这些项目不一定占用所选目标。",
                    denied, moduleDenied, unresolved));
            if (result.Cancelled) result.Warnings.Add("扫描已取消，当前仅显示取消前得到的结果。");
            return result;
        }
    }
}
