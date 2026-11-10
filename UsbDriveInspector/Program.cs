// 提供便携入口、本机运行副本和扫描子进程入口，避免工具自身占用 U 盘。
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Security.Cryptography;
using System.Security.Principal;
using System.Text;
using System.Windows.Forms;

[assembly: AssemblyTitle("U 盘占用检查器")]
[assembly: AssemblyDescription("检查磁盘、文件和文件夹的占用进程与 PID，支持正常关闭与强制结束进程")]
[assembly: AssemblyProduct("UsbDriveInspector")]
[assembly: AssemblyVersion("1.2.1.0")]
[assembly: AssemblyFileVersion("1.2.1.0")]

namespace UsbDriveInspector
{
    internal static class Program
    {
        internal static string Executable { get { return Assembly.GetExecutingAssembly().Location; } }
        internal static bool IsAdministrator
        {
            get
            {
                using (var identity = WindowsIdentity.GetCurrent())
                    return new WindowsPrincipal(identity).IsInRole(WindowsBuiltInRole.Administrator);
            }
        }

        [STAThread]
        private static int Main(string[] args)
        {
            if (args.Length == 4 && (args[0] == "--scan-worker" || args[0] == "--scan-file-worker"))
            {
                // GUI 子进程没有控制台，设置 OutputEncoding 会触发无效句柄；直接使用重定向输出流。
                Console.SetOut(new StreamWriter(Console.OpenStandardOutput(), new UTF8Encoding(false)) { AutoFlush = true });
                string file = args[0] == "--scan-file-worker" ? Encoding.UTF8.GetString(Convert.FromBase64String(args[1])) : null;
                return HandleScanner.RunWorker(file == null ? args[1] : Path.GetPathRoot(file),
                    int.Parse(args[2]), int.Parse(args[3]), file);
            }
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            try
            {
                string file = args.Length == 2 && args[0] == "--run-file"
                    ? ScanService.NormalizeFilePath(Encoding.UTF8.GetString(Convert.FromBase64String(args[1]))) : null;
                string drive = file != null ? Path.GetPathRoot(file) :
                    args.Length >= 2 && args[0] == "--run" ? args[1] : Path.GetPathRoot(Executable);
                string local = PrepareLocalCopy();
                if (!string.Equals(local, Executable, StringComparison.OrdinalIgnoreCase))
                {
                    Process.Start(new ProcessStartInfo(local, RunArguments(drive, file)) {
                        UseShellExecute = true, WorkingDirectory = Path.GetDirectoryName(local)
                    });
                    return 0;
                }
                // 当前目录也必须离开 U 盘，否则仅复制 EXE 仍可能阻止移除。
                Environment.CurrentDirectory = Path.GetDirectoryName(local);
                Application.Run(new MainForm(drive, file));
                return 0;
            }
            catch (Exception ex)
            {
                MessageBox.Show("无法启动 U 盘占用检查器：\r\n" + ex.Message,
                    "启动失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return 1;
            }
        }

        internal static string RunArguments(string drive, string file)
        {
            return file == null ? "--run " + NormalizeDrive(drive).Substring(0, 2) :
                "--run-file " + Convert.ToBase64String(Encoding.UTF8.GetBytes(ScanService.NormalizeFilePath(file)));
        }

        internal static string NormalizeDrive(string drive)
        {
            if (string.IsNullOrWhiteSpace(drive) || drive.Length < 2 || drive[1] != ':' ||
                !char.IsLetter(drive[0]) || drive[0] > 127)
                throw new ArgumentException("请选择有效的盘符，例如 F:。");
            return char.ToUpperInvariant(drive[0]) + ":\\";
        }

        internal static string PrepareLocalCopy()
        {
            string hash;
            using (var sha = SHA256.Create())
            using (var stream = File.OpenRead(Executable))
                hash = BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "").Substring(0, 20);
            string folder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "UsbDriveInspector", hash);
            string target = Path.Combine(folder, "UsbDriveInspector.exe");
            if (string.Equals(Executable, target, StringComparison.OrdinalIgnoreCase)) return target;
            Directory.CreateDirectory(folder);
            // 不覆盖正在运行的版本；内容散列使不同构建使用不同目录。
            if (!File.Exists(target)) File.Copy(Executable, target, false);
            using (var sha = SHA256.Create())
            using (var stream = File.OpenRead(target))
                if (BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "").Substring(0, 20) != hash)
                    throw new IOException("本机运行副本校验失败，请删除本机 UsbDriveInspector 缓存后重试。");
            return target;
        }
    }
}
