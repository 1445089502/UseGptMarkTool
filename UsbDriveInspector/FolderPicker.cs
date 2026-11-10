// 使用 Windows 通用打开对话框选择文件夹，提供地址栏、路径输入和标准导航。
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace UsbDriveInspector
{
    internal static class FolderPicker
    {
        /// <summary>在所属窗口上显示文件夹选择器；取消返回 null，系统接口失败时抛出异常。</summary>
        internal static string Show(IWin32Window owner, string initialPath)
        {
            IFileDialog dialog = null;
            IShellItem initialFolder = null;
            IShellItem selectedFolder = null;
            IntPtr selectedPath = IntPtr.Zero;
            try
            {
                dialog = (IFileDialog)new FileOpenDialog();
                uint options;
                dialog.GetOptions(out options);
                // 保留系统默认选项；只选现存文件系统目录，避免改变进程当前目录而额外占用 U 盘。
                const uint PickFolders = 0x20, ForceFileSystem = 0x40, PathMustExist = 0x800, NoChangeDirectory = 0x8;
                dialog.SetOptions(options | PickFolders | ForceFileSystem | PathMustExist | NoChangeDirectory);
                dialog.SetTitle("选择需要检查占用的文件夹");
                dialog.SetOkButtonLabel("选择文件夹");
                string folder = GetInitialFolder(initialPath);
                if (folder != null)
                {
                    Guid shellItemId = typeof(IShellItem).GUID;
                    SHCreateItemFromParsingName(folder, IntPtr.Zero, ref shellItemId, out initialFolder);
                    dialog.SetFolder(initialFolder);
                }
                int result = dialog.Show(owner == null ? IntPtr.Zero : owner.Handle);
                // HRESULT_FROM_WIN32(ERROR_CANCELLED) 是正常取消；其他失败不能伪装成用户取消。
                if (result == unchecked((int)0x800704C7)) return null;
                Marshal.ThrowExceptionForHR(result);
                dialog.GetResult(out selectedFolder);
                selectedFolder.GetDisplayName(0x80058000, out selectedPath); // SIGDN_FILESYSPATH
                return Marshal.PtrToStringUni(selectedPath);
            }
            finally
            {
                if (selectedPath != IntPtr.Zero) Marshal.FreeCoTaskMem(selectedPath);
                if (selectedFolder != null) Marshal.FinalReleaseComObject(selectedFolder);
                // 选择初始目录时两个结果可能共享同一个 RCW，不能重复释放。
                if (initialFolder != null && !ReferenceEquals(initialFolder, selectedFolder))
                    Marshal.FinalReleaseComObject(initialFolder);
                if (dialog != null) Marshal.FinalReleaseComObject(dialog);
            }
        }

        private static string GetInitialFolder(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) return null;
            // 已有文件目标从所在目录打开；无效输入交给系统默认位置，不能阻止用户重新选择。
            try
            {
                string normalized = ScanService.NormalizeFilePath(path);
                if (Directory.Exists(normalized)) return normalized;
                string parent = Path.GetDirectoryName(normalized);
                return Directory.Exists(parent) ? parent : null;
            }
            catch (ArgumentException) { return null; }
            catch (NotSupportedException) { return null; }
            catch (PathTooLongException) { return null; }
        }

        [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
        private static extern void SHCreateItemFromParsingName(string path, IntPtr binding, ref Guid id,
            [MarshalAs(UnmanagedType.Interface)] out IShellItem item);

        [ComImport, Guid("DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7")]
        private class FileOpenDialog { }

        // COM 方法必须保持原生 vtable 顺序，未使用的方法也不可移除或重排。
        [ComImport, Guid("42F85136-DB7E-439C-85F1-E4075D135FC8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        private interface IFileDialog
        {
            [PreserveSig] int Show(IntPtr owner);
            void SetFileTypes(uint count, IntPtr filters);
            void SetFileTypeIndex(uint index);
            void GetFileTypeIndex(out uint index);
            void Advise(IntPtr events, out uint cookie);
            void Unadvise(uint cookie);
            void SetOptions(uint options);
            void GetOptions(out uint options);
            void SetDefaultFolder(IShellItem folder);
            void SetFolder(IShellItem folder);
            void GetFolder(out IShellItem folder);
            void GetCurrentSelection(out IShellItem item);
            void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string name);
            void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string name);
            void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string title);
            void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string text);
            void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string label);
            void GetResult(out IShellItem item);
        }

        [ComImport, Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        private interface IShellItem
        {
            void BindToHandler(IntPtr binding, ref Guid handler, ref Guid id, out IntPtr result);
            void GetParent(out IShellItem parent);
            void GetDisplayName(uint displayName, out IntPtr name);
            void GetAttributes(uint mask, out uint attributes);
            void Compare(IShellItem other, uint hint, out int order);
        }
    }
}
