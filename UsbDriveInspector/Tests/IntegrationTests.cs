// 使用独立测试进程验证磁盘、文件与文件夹占用、进程身份保护、关闭、扫描取消及界面布局。
using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.Win32.SafeHandles;

namespace UsbDriveInspector
{
    internal static class IntegrationTests
    {
        private static int assertions;

        [STAThread]
        private static int Main(string[] args)
        {
            if (args.Length == 4 && (args[0] == "--scan-worker" || args[0] == "--scan-file-worker"))
            {
                if (Environment.GetEnvironmentVariable("USB_INSPECTOR_TEST_HANG") == "1") Thread.Sleep(Timeout.Infinite);
                Console.SetOut(new StreamWriter(Console.OpenStandardOutput(), new UTF8Encoding(false)) { AutoFlush = true });
                string file = args[0] == "--scan-file-worker" ? Encoding.UTF8.GetString(Convert.FromBase64String(args[1])) : null;
                return HandleScanner.RunWorker(file == null ? args[1] : Path.GetPathRoot(file),
                    int.Parse(args[2]), int.Parse(args[3]), file);
            }
            if (args.Length == 3 && args[0] == "--hold") return Hold(args[1], args[2]);
            if (args.Length == 3 && args[0] == "--hold-exclusive") return Hold(args[1], args[2], true);
            if ((args.Length == 2 || args.Length == 3) && args[0] == "--render") return Render(args[1], args.Length == 3 ? args[2] : null);
            try { Run().GetAwaiter().GetResult(); Console.WriteLine("PASS: " + assertions + " assertions"); return 0; }
            catch (Exception ex) { Console.Error.WriteLine(ex); return 1; }
        }

        private static int Render(string folder, string file = null)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            var form = new MainForm(Path.GetPathRoot(Program.Executable), file);
            Console.WriteLine("Display: " + Screen.PrimaryScreen.WorkingArea + "; Scale: " + form.CurrentAutoScaleDimensions);
            var timer = new System.Windows.Forms.Timer { Interval = 500 };
            int ticks = 0;
            timer.Tick += delegate
            {
                bool running = (bool)typeof(MainForm).GetField("busy", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(form);
                if (running && ++ticks < 70) return;
                timer.Stop();
                Directory.CreateDirectory(folder);
                var table = (DataGridView)typeof(MainForm).GetField("grid", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(form);
                Check(table.Columns["Pid"].Width >= TextRenderer.MeasureText("999999", table.Font).Width + 12,
                    "PID remains readable at current display scale");
                if (file != null)
                {
                    var diskMode = (RadioButton)typeof(MainForm).GetField("driveMode", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(form);
                    var targetMode = (RadioButton)typeof(MainForm).GetField("fileMode", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(form);
                    var eject = (Button)typeof(MainForm).GetField("eject", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(form);
                    Check(targetMode.Checked && !diskMode.Checked && !eject.Enabled, "file mode excludes disk actions");
                }
                using (var bitmap = new Bitmap(form.Width, form.Height))
                {
                    form.DrawToBitmap(bitmap, new Rectangle(Point.Empty, form.Size));
                    bitmap.Save(Path.Combine(folder, "desktop.png"));
                }
                form.Size = form.MinimumSize;
                form.PerformLayout();
                using (var bitmap = new Bitmap(form.Width, form.Height))
                {
                    form.DrawToBitmap(bitmap, new Rectangle(Point.Empty, form.Size));
                    bitmap.Save(Path.Combine(folder, "compact.png"));
                }
                Console.WriteLine("Rendered default and minimum window sizes.");
                if (file != null)
                {
                    form.SelectFile(Path.Combine(Path.GetDirectoryName(file), "different-target.txt"));
                    Check(typeof(MainForm).GetField("latest", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(form) == null,
                        "changing the file clears stale scan results");
                    var terminate = (Button)typeof(MainForm).GetField("terminate", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(form);
                    Check(!terminate.Enabled, "changing the file disables stale termination action");
                }
                form.Close();
            };
            timer.Start();
            Application.Run(form);
            timer.Dispose();
            form.Dispose();
            return 0;
        }

        private static void Check(bool value, string message)
        {
            if (!value) throw new Exception("FAIL: " + message);
            assertions++;
            Console.WriteLine("PASS: " + message);
        }

        private static int Hold(string file, string directory, bool exclusive = false)
        {
            using (var first = File.Open(file, FileMode.Open, FileAccess.ReadWrite, exclusive ? FileShare.None : FileShare.ReadWrite | FileShare.Delete))
            using (var second = exclusive ? null : File.Open(file, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete))
            using (SafeFileHandle folder = Native.CreateFile(directory, 0, 7, IntPtr.Zero, 3, 0x02000000, IntPtr.Zero))
            {
                if (folder.IsInvalid) throw Native.Error("测试文件夹打开失败");
                var form = new Form { Text = "UsbDriveInspector integration fixture", Width = 420, Height = 100,
                    WindowState = FormWindowState.Minimized, ShowInTaskbar = true };
                form.Shown += delegate { Console.WriteLine("READY"); Console.Out.Flush(); };
                Application.Run(form);
            }
            return 0;
        }

        private static Process StartHolder(string executable, string file, string directory, bool exclusive = false)
        {
            var process = Process.Start(new ProcessStartInfo(executable,
                (exclusive ? "--hold-exclusive \"" : "--hold \"") + file + "\" \"" + directory + "\"") {
                UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true,
                WorkingDirectory = Path.GetDirectoryName(Program.Executable)
            });
            var ready = process.StandardOutput.ReadLineAsync();
            if (!ready.Wait(8000) || ready.Result != "READY")
            {
                if (!process.HasExited) process.Kill();
                process.Dispose();
                throw new Exception("测试进程未就绪。");
            }
            return process;
        }

        private static async Task Run()
        {
            await CheckFiles();
            string folder = Path.Combine("F:\\", "UsbDriveInspectorTest_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(folder);
            string file = Path.Combine(folder, "held file.txt");
            string copiedExe = Path.Combine(folder, "UsbDriveInspector.Tests.exe");
            File.WriteAllText(file, "Temporary integration fixture.", Encoding.UTF8);
            File.Copy(Program.Executable, copiedExe);
            Process holder = null;
            try
            {
                Check(HandleScanner.MapPath(@"\Device\HarddiskVolume10\a", @"\Device\HarddiskVolume1", @"F:\") == null,
                    "device prefix does not match a different volume");
                Check(HandleScanner.MapPath(@"\Device\HarddiskVolume1\folder\a.txt", @"\Device\HarddiskVolume1", @"F:\") == @"F:\folder\a.txt",
                    "NT path resolves to requested drive");
                Check(MainForm.CsvCell("=1+2") == "\"'=1+2\"", "CSV formula escaped");
                Check(MainForm.CsvCell("a,\"b") == "\"a,\"\"b\"", "CSV quotes escaped");
                holder = StartHolder(copiedExe, file, folder);
                string production = Path.Combine(Path.GetDirectoryName(Path.GetDirectoryName(Program.Executable)), "UsbDriveInspector.exe");
                using (var worker = Process.Start(new ProcessStartInfo(production, "--scan-worker F: 0 1") {
                    UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true,
                    StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8,
                    WorkingDirectory = Path.GetDirectoryName(production) }))
                {
                    var output = worker.StandardOutput.ReadToEndAsync();
                    var error = worker.StandardError.ReadToEndAsync();
                    if (!worker.WaitForExit(30000)) { worker.Kill(); throw new Exception("生产 EXE 扫描超时。"); }
                    string text = await output;
                    Check(worker.ExitCode == 0 && text.Contains("\"Type\":\"done\"") && text.Contains("held file.txt"),
                        "shipping GUI executable produces valid UTF-8 worker output: " + await error);
                }
                var initial = await ScanService.ScanAsync("F:", CancellationToken.None);
                Check(initial.Completed, "all scan workers complete: " + string.Join("; ", initial.Warnings));
                var row = initial.Rows.FirstOrDefault(r => r.Pid == holder.Id && r.Path == file);
                Check(row != null && row.Created != 0, "open file reports exact PID, path and creation identity");
                Check(row.Count == 2, "duplicate file handles grouped and counted");
                Check(initial.Rows.Any(r => r.Pid == holder.Id && r.Path.TrimEnd('\\') == folder), "open directory detected");
                Check(initial.Rows.Any(r => r.Pid == holder.Id && r.Path == copiedExe && r.Kind == "运行中的程序"), "running executable on USB detected");
                bool protectedPid = false;
                try { using (new ProcessAction(new Occupancy { Pid = 4, Created = 1 }, true)) { } }
                catch (InvalidOperationException) { protectedPid = true; }
                Check(protectedPid, "system PID cannot be terminated");
                bool stale = false;
                try { using (new ProcessAction(new Occupancy { Pid = row.Pid, Created = row.Created + 1 }, true)) { } }
                catch (InvalidOperationException) { stale = true; }
                Check(stale && !holder.HasExited, "stale PID identity rejected without killing live process");
                using (var action = new ProcessAction(row, true)) Console.WriteLine(action.Terminate());
                Check(holder.WaitForExit(5000), "force termination ends only test process");
                var after = await ScanService.ScanAsync("F:", CancellationToken.None);
                Check(after.Completed && !after.Rows.Any(r => r.Pid == row.Pid && r.Created == row.Created), "occupancy disappears after termination");
                holder.Dispose();
                holder = StartHolder(Program.Executable, file, folder);
                var normal = await ScanService.ScanAsync("F:", CancellationToken.None);
                row = normal.Rows.First(r => r.Pid == holder.Id && r.Path == file);
                using (var action = new ProcessAction(row, false)) Console.WriteLine(action.CloseNormally());
                Check(holder.WaitForExit(5000), "normal close lets fixture exit cleanly");
                Environment.SetEnvironmentVariable("USB_INSPECTOR_TEST_HANG", "1");
                using (var token = new CancellationTokenSource(700))
                {
                    var watch = Stopwatch.StartNew();
                    var cancelled = await ScanService.ScanAsync("F:", token.Token);
                    Check(cancelled.Cancelled && !cancelled.Completed && watch.Elapsed.TotalSeconds < 8,
                        "cancel kills blocked scan workers promptly");
                }
                var timeout = await ScanService.ScanAsync("F:", CancellationToken.None);
                Check(!timeout.Completed && timeout.Seconds < 32 && timeout.Warnings.Any(w => w.Contains("25 秒")),
                    "blocked workers time out and explicitly report partial results");
            }
            finally
            {
                Environment.SetEnvironmentVariable("USB_INSPECTOR_TEST_HANG", null);
                if (holder != null)
                {
                    if (!holder.HasExited) { holder.Kill(); holder.WaitForExit(5000); }
                    holder.Dispose();
                }
                // 只删除本测试创建的随机目录中的明确文件，不递归清理磁盘。
                File.Delete(file);
                File.Delete(copiedExe);
                Directory.Delete(folder, false);
            }
        }

        private static async Task CheckFiles()
        {
            string folder = Path.Combine(Path.GetTempPath(), "UsbDriveInspectorFileTest_" + Guid.NewGuid().ToString("N"));
            string otherFolder = Path.Combine(folder, "other");
            string siblingFolder = folder + "-other";
            Directory.CreateDirectory(otherFolder);
            Directory.CreateDirectory(siblingFolder);
            string fileName = "held " + "\u6587\u4ef6" + ".txt";
            string file = Path.Combine(folder, fileName);
            string otherFile = Path.Combine(otherFolder, fileName);
            string siblingFile = Path.Combine(siblingFolder, fileName);
            File.WriteAllText(file, "Test file.");
            File.WriteAllText(otherFile, "Same name, different path.");
            File.WriteAllText(siblingFile, "Sibling folder sharing a name prefix.");
            Process holder = null, other = null, sibling = null;
            try
            {
                Check(ScanService.NormalizeFilePath("\"" + file + "\"") == file, "quoted file path accepted");
                Check(ScanService.NormalizeFilePath(folder + "\\") == folder, "folder trailing separator normalized");
                string root = Path.GetPathRoot(folder);
                Check(ScanService.NormalizeFilePath(root) == root, "drive root accepted as a folder");
                Check(HandleScanner.MatchesTarget(file, root, true), "root folder includes descendant files");
                string launch = Program.RunArguments(null, file);
                Check(Encoding.UTF8.GetString(Convert.FromBase64String(launch.Substring("--run-file ".Length))) == file,
                    "file target survives local relaunch and administrator arguments");
                bool rejected = false;
                try { ScanService.NormalizeFilePath(@"\\server\share\file.txt"); }
                catch (ArgumentException) { rejected = true; }
                Check(rejected, "network path rejected with explicit guidance");
                holder = StartHolder(Program.Executable, file, folder, true);
                other = StartHolder(Program.Executable, otherFile, otherFolder);
                sibling = StartHolder(Program.Executable, siblingFile, siblingFolder);
                var result = await ScanService.ScanFileAsync(file.ToUpperInvariant(), CancellationToken.None);
                Check(result.Completed, "local file scan completes");
                var row = result.Rows.SingleOrDefault(r => r.Pid == holder.Id);
                Check(row != null && row.Count == 1 && row.Created != 0 && row.Path.Equals(file, StringComparison.OrdinalIgnoreCase),
                    "exclusive file with spaces and Chinese name reports correct PID and path");
                Check(result.Rows.All(r => r.Path.Equals(file, StringComparison.OrdinalIgnoreCase)) && !result.Rows.Any(r => r.Pid == other.Id),
                    "same file name in another directory is excluded");
                string production = Path.Combine(Path.GetDirectoryName(Path.GetDirectoryName(Program.Executable)), "UsbDriveInspector.exe");
                using (var worker = Process.Start(new ProcessStartInfo(production,
                    "--scan-file-worker " + Convert.ToBase64String(Encoding.UTF8.GetBytes(file)) + " 0 1") {
                    UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true,
                    StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8,
                    WorkingDirectory = Path.GetDirectoryName(production) }))
                {
                    var output = worker.StandardOutput.ReadToEndAsync();
                    var error = worker.StandardError.ReadToEndAsync();
                    if (!worker.WaitForExit(30000)) { worker.Kill(); throw new Exception("单文件扫描超时。"); }
                    string text = await output;
                    Check(worker.ExitCode == 0 && text.Contains("\"Type\":\"done\"") && text.Contains("\"Pid\":" + holder.Id),
                        "shipping executable supports single-file worker" + (worker.ExitCode == 0 && text.Contains("\"Pid\":" + holder.Id)
                            ? "" : " (PID=" + holder.Id + ", exit=" + worker.ExitCode + ", output=" + text + ", error=" + await error + ")"));
                }
                var folderResult = await ScanService.ScanFileAsync(folder.ToUpperInvariant() + "\\", CancellationToken.None);
                Check(folderResult.Completed && folderResult.Rows.Any(r => r.Pid == holder.Id && r.Path == file),
                    "folder scan finds a directly contained file");
                Check(folderResult.Rows.Any(r => r.Pid == holder.Id && r.Path.TrimEnd('\\') == folder),
                    "folder scan finds a handle to the selected folder itself");
                Check(folderResult.Rows.Any(r => r.Pid == other.Id && r.Path == otherFile) &&
                    folderResult.Rows.Any(r => r.Pid == other.Id && r.Path.TrimEnd('\\') == otherFolder),
                    "folder scan includes nested files and directory handles");
                Check(!folderResult.Rows.Any(r => r.Pid == sibling.Id || r.Path.StartsWith(siblingFolder, StringComparison.OrdinalIgnoreCase)),
                    "folder scan excludes siblings with the same name prefix");
                using (var worker = Process.Start(new ProcessStartInfo(production,
                    "--scan-file-worker " + Convert.ToBase64String(Encoding.UTF8.GetBytes(folder)) + " 0 1") {
                    UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true,
                    StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8,
                    WorkingDirectory = Path.GetDirectoryName(production) }))
                {
                    var output = worker.StandardOutput.ReadToEndAsync();
                    var error = worker.StandardError.ReadToEndAsync();
                    if (!worker.WaitForExit(30000)) { worker.Kill(); throw new Exception("文件夹扫描超时。"); }
                    string text = await output;
                    Check(worker.ExitCode == 0 && text.Contains("\"Type\":\"done\"") &&
                        text.Contains("\"Pid\":" + holder.Id + ",") && text.Contains("\"Pid\":" + other.Id + ",") &&
                        !text.Contains("\"Pid\":" + sibling.Id + ","),
                        "shipping executable scans a folder with correct descendant boundaries: " + await error);
                }
                using (var action = new ProcessAction(row, true)) action.Terminate();
                Check(holder.WaitForExit(5000) && !other.HasExited, "single-file termination leaves unrelated holder alive");
                var after = await ScanService.ScanFileAsync(file, CancellationToken.None);
                Check(after.Completed && after.Rows.Count == 0, "file occupancy disappears after holder exits");
                var folderAfter = await ScanService.ScanFileAsync(folder, CancellationToken.None);
                Check(folderAfter.Completed && !folderAfter.Rows.Any(r => r.Pid == holder.Id) &&
                    folderAfter.Rows.Any(r => r.Pid == other.Id), "folder rescan removes exited process and retains active descendant");
                File.Delete(file);
                var missing = await ScanService.ScanFileAsync(file, CancellationToken.None);
                Check(!missing.Completed && missing.Rows.Count == 0 && missing.Warnings.Count > 0,
                    "missing file reports scan failure instead of no occupancy");
            }
            finally
            {
                foreach (var process in new[] { holder, other, sibling })
                    if (process != null)
                    {
                        if (!process.HasExited) { process.Kill(); process.WaitForExit(5000); }
                        process.Dispose();
                    }
                File.Delete(file);
                File.Delete(otherFile);
                File.Delete(siblingFile);
                Directory.Delete(otherFolder, false);
                Directory.Delete(folder, false);
                Directory.Delete(siblingFolder, false);
            }
        }
    }
}
