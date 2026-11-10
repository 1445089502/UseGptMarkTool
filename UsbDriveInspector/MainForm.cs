// 提供磁盘、文件与文件夹选择、占用列表、进程操作与报告导出；扫描状态仅由此窗口维护。
using System;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace UsbDriveInspector
{
    internal sealed class MainForm : Form
    {
        private readonly ComboBox drives = new ComboBox();
        private readonly RadioButton driveMode = new RadioButton();
        private readonly RadioButton fileMode = new RadioButton();
        private readonly TextBox filePath = new TextBox();
        private readonly Button browse = new Button();
        private readonly Button browseFolder = new Button();
        private readonly Icon applicationIcon;
        private readonly TextBox search = new TextBox();
        private readonly DataGridView grid = new DataGridView();
        private readonly Label summary = new Label();
        private readonly Label driveInfo = new Label();
        private readonly TextBox details = new TextBox();
        private readonly TextBox diagnostics = new TextBox();
        private readonly ProgressBar progress = new ProgressBar();
        private readonly Button scan = new Button();
        private readonly Button cancel = new Button();
        private readonly Button refresh = new Button();
        private readonly Button close = new Button();
        private readonly Button terminate = new Button();
        private readonly Button copy = new Button();
        private readonly Button export = new Button();
        private readonly Button elevate = new Button();
        private readonly Button eject = new Button();
        private CancellationTokenSource cancellation;
        private ScanResult latest;
        private bool busy;

        private sealed class DriveChoice
        {
            internal string Root;
            internal string Display;
            public override string ToString() { return Display; }
        }

        internal MainForm(string preferredDrive, string initialFile = null)
        {
            SuspendLayout();
            Text = "U 盘与文件占用检查器";
            Font = new Font("Microsoft YaHei UI", 9F);
            ClientSize = new Size(1120, 786);
            MinimumSize = new Size(960, 726);
            StartPosition = FormStartPosition.CenterScreen;
            BackColor = Color.FromArgb(246, 248, 250);
            ForeColor = Color.FromArgb(30, 41, 54);
            applicationIcon = Icon.ExtractAssociatedIcon(Program.Executable);
            Icon = applicationIcon ?? SystemIcons.Application;
            Padding = new Padding(24, 20, 24, 18);

            var layout = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 11 };
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            float[] heights = { 42, 30, 46, 28, 46, 42, 0, 58, 44, 76, 28 };
            foreach (float height in heights) layout.RowStyles.Add(new RowStyle(height == 0 ? SizeType.Percent : SizeType.Absolute,
                height == 0 ? 100 : height));
            Controls.Add(layout);
            var title = new Label { Text = "找出谁在占用磁盘、文件或文件夹", Dock = DockStyle.Fill,
                Font = new Font(Font.FontFamily, 20, FontStyle.Bold), AutoSize = false };
            layout.Controls.Add(title, 0, 0);
            layout.Controls.Add(new Label { Text = "选择文件夹时包含其子目录和文件。关闭应用前请先保存工作。",
                Dock = DockStyle.Fill, ForeColor = Color.FromArgb(78, 92, 108) }, 0, 1);

            var toolbar = new FlowLayoutPanel { Dock = DockStyle.Fill, WrapContents = false };
            driveMode.Text = "磁盘占用";
            driveMode.Checked = true;
            driveMode.Width = 140;
            driveMode.Margin = new Padding(0, 8, 5, 0);
            toolbar.Controls.Add(driveMode);
            drives.DropDownStyle = ComboBoxStyle.DropDownList;
            drives.Width = 260;
            drives.Margin = new Padding(0, 5, 12, 0);
            drives.AccessibleName = "目标磁盘";
            toolbar.Controls.Add(drives);
            ConfigureButton(scan, "扫描占用", 112, true);
            ConfigureButton(cancel, "取消", 70, false);
            ConfigureButton(refresh, "刷新磁盘", 96, false);
            ConfigureButton(elevate, Program.IsAdministrator ? "已是管理员" : "管理员扫描", 118, false);
            toolbar.Controls.AddRange(new Control[] { scan, cancel, refresh, elevate });
            layout.Controls.Add(toolbar, 0, 2);
            driveInfo.Dock = DockStyle.Fill;
            driveInfo.ForeColor = Color.FromArgb(78, 92, 108);
            layout.Controls.Add(driveInfo, 0, 3);

            var fileSelector = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 4 };
            fileSelector.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 145));
            fileSelector.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            fileSelector.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 128));
            fileSelector.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 128));
            fileMode.Text = "文件 / 文件夹";
            fileMode.Dock = DockStyle.Fill;
            fileMode.Margin = new Padding(0);
            fileSelector.Controls.Add(fileMode, 0, 0);
            filePath.Dock = DockStyle.Fill;
            filePath.Margin = new Padding(0, 9, 12, 0);
            filePath.AccessibleName = "需要检查的完整文件或文件夹路径";
            fileSelector.Controls.Add(filePath, 1, 0);
            ConfigureButton(browse, "选择文件…", 120, false);
            fileSelector.Controls.Add(browse, 2, 0);
            ConfigureButton(browseFolder, "选择文件夹…", 120, false);
            fileSelector.Controls.Add(browseFolder, 3, 0);
            layout.Controls.Add(fileSelector, 0, 4);

            var filters = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 3 };
            filters.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            filters.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 150));
            filters.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 245));
            summary.Text = "准备扫描";
            summary.Dock = DockStyle.Fill;
            summary.TextAlign = ContentAlignment.MiddleLeft;
            summary.Font = new Font(Font, FontStyle.Bold);
            filters.Controls.Add(summary, 0, 0);
            filters.Controls.Add(new Label { Text = "搜索文件 / 进程 / PID", Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleRight }, 1, 0);
            search.Dock = DockStyle.Fill;
            search.Margin = new Padding(10, 8, 0, 0);
            search.AccessibleName = "搜索文件、进程或 PID";
            filters.Controls.Add(search, 2, 0);
            layout.Controls.Add(filters, 0, 5);

            grid.Dock = DockStyle.Fill;
            grid.BackgroundColor = Color.White;
            grid.BorderStyle = BorderStyle.FixedSingle;
            grid.ReadOnly = true;
            grid.AllowUserToAddRows = false;
            grid.AllowUserToDeleteRows = false;
            grid.AllowUserToResizeRows = false;
            grid.RowHeadersVisible = false;
            grid.SelectionMode = DataGridViewSelectionMode.FullRowSelect;
            grid.MultiSelect = false;
            grid.AutoGenerateColumns = false;
            grid.AutoSizeRowsMode = DataGridViewAutoSizeRowsMode.None;
            grid.RowTemplate.Height = 32;
            grid.ColumnHeadersHeight = 36;
            grid.ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.DisableResizing;
            grid.EnableHeadersVisualStyles = false;
            grid.ColumnHeadersDefaultCellStyle.BackColor = Color.FromArgb(232, 237, 242);
            grid.ColumnHeadersDefaultCellStyle.ForeColor = ForeColor;
            grid.ColumnHeadersDefaultCellStyle.Font = new Font(Font, FontStyle.Bold);
            grid.DefaultCellStyle.SelectionBackColor = Color.FromArgb(218, 235, 250);
            grid.DefaultCellStyle.SelectionForeColor = ForeColor;
            grid.DefaultCellStyle.Padding = new Padding(5, 0, 5, 0);
            grid.AlternatingRowsDefaultCellStyle.BackColor = Color.FromArgb(248, 250, 252);
            grid.CellBorderStyle = DataGridViewCellBorderStyle.SingleHorizontal;
            grid.GridColor = Color.FromArgb(231, 235, 240);
            AddColumn("ProcessName", "占用进程", 170);
            AddColumn("Pid", "PID", 90);
            AddColumn("Kind", "占用类型", 132);
            AddColumn("Count", "数量", 58);
            AddColumn("Path", "完整文件 / 文件夹路径", 380);
            grid.Columns[4].AutoSizeMode = DataGridViewAutoSizeColumnMode.Fill;
            grid.Columns[4].MinimumWidth = 240;
            layout.Controls.Add(grid, 0, 6);

            details.Dock = DockStyle.Fill;
            details.Multiline = true;
            details.ReadOnly = true;
            details.BorderStyle = BorderStyle.None;
            details.BackColor = BackColor;
            details.Margin = new Padding(0, 9, 0, 0);
            details.AccessibleName = "所选占用详情";
            layout.Controls.Add(details, 0, 7);

            var actions = new FlowLayoutPanel { Dock = DockStyle.Fill, WrapContents = false };
            ConfigureButton(close, "正常关闭应用", 128, false);
            ConfigureButton(terminate, "强制结束进程", 128, false);
            terminate.ForeColor = Color.FromArgb(170, 38, 38);
            ConfigureButton(copy, "复制所选信息", 128, false);
            ConfigureButton(export, "导出全部结果", 128, false);
            ConfigureButton(eject, "打开安全移除", 128, false);
            var help = new Button();
            ConfigureButton(help, "使用说明", 96, false);
            actions.Controls.AddRange(new Control[] { close, terminate, copy, export, eject, help });
            layout.Controls.Add(actions, 0, 8);

            diagnostics.Dock = DockStyle.Fill;
            diagnostics.Multiline = true;
            diagnostics.ReadOnly = true;
            diagnostics.ScrollBars = ScrollBars.Vertical;
            diagnostics.BackColor = Color.White;
            diagnostics.BorderStyle = BorderStyle.FixedSingle;
            diagnostics.AccessibleName = "扫描说明与限制";
            layout.Controls.Add(diagnostics, 0, 9);
            var footer = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2 };
            footer.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            footer.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 180));
            footer.Controls.Add(new Label { Text = "本机副本运行 · 无需安装 · F5 重新扫描 · Ctrl+C 复制所选",
                Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft,
                ForeColor = Color.FromArgb(78, 92, 108) }, 0, 0);
            progress.Dock = DockStyle.Fill;
            progress.Margin = new Padding(0, 7, 0, 7);
            progress.Style = ProgressBarStyle.Marquee;
            progress.Visible = false;
            footer.Controls.Add(progress, 1, 0);
            layout.Controls.Add(footer, 0, 10);

            scan.Click += async delegate { await ScanAsync(); };
            cancel.Click += delegate { if (cancellation != null) { cancellation.Cancel(); cancel.Enabled = false; } };
            refresh.Click += delegate { LoadDrives(SelectedDrive); };
            drives.SelectedIndexChanged += delegate { if (!busy) { ResetTarget(); UpdateDriveInfo(); } };
            driveMode.CheckedChanged += delegate
            {
                if (driveMode.Checked) fileMode.Checked = false;
                ResetTarget();
            };
            fileMode.CheckedChanged += delegate
            {
                if (fileMode.Checked) driveMode.Checked = false;
                ResetTarget();
            };
            filePath.TextChanged += delegate { if (!busy) ResetTarget(); };
            browse.Click += delegate { ChooseFile(); };
            browseFolder.Click += delegate { ChooseFolder(); };
            search.TextChanged += delegate { ApplyFilter(); };
            grid.SelectionChanged += delegate { UpdateSelection(); };
            close.Click += async delegate { await ActAsync(false); };
            terminate.Click += async delegate { await ActAsync(true); };
            copy.Click += delegate { CopySelection(); };
            export.Click += delegate { ExportResults(); };
            elevate.Click += delegate { Elevate(); };
            eject.Click += delegate
            {
                TryAction(() => Process.Start(new ProcessStartInfo(Path.Combine(Environment.SystemDirectory, "rundll32.exe"),
                    "shell32.dll,Control_RunDLL hotplug.dll") { UseShellExecute = true, WorkingDirectory = Environment.SystemDirectory }));
            };
            help.Click += delegate
            {
                MessageBox.Show(this,
                    "1. 选“磁盘占用”扫描整盘；或点“选择文件” / “选择文件夹”，选择目标后点“扫描占用”。也可在“文件 / 文件夹”下粘贴完整路径。\r\n" +
                    "2. 优先保存工作，并使用“正常关闭应用”。后台进程可能没有主窗口。\r\n" +
                    "3. “强制结束”会结束整个进程，其所有未保存内容都可能丢失；不是只关闭列表中的文件。\r\n" +
                    "4. 清理后重新扫描，再通过“打开安全移除”使用 Windows 的安全删除功能。\r\n\r\n" +
                    "没有结果不等于保证可拔出：系统驱动、受保护进程、无文件句柄的数据映射等可能无法识别。" +
                    "扫描是某一时刻的快照；文件占用会随时变化。若仍无法移除，请尝试管理员扫描，关闭资源管理器窗口，等待复制或同步结束。\r\n\r\n" +
                    "选择文件时仅检查该文件；选择文件夹时检查文件夹自身及其子目录和文件。不会混入同级的近似名称目录。暂不支持网络共享路径；硬链接别名、目录内链接指向的外部位置和受保护进程可能无法完整识别。\r\n\r\n" +
                    "程序会复制到本机 LocalAppData\\UsbDriveInspector 后运行，避免自身占用 U 盘。" +
                    "无需联网，不会自动结束任何程序；系统、服务与桌面进程不允许在这里结束。\r\n\r\n" +
                    "当前运行位置：\r\n" + Program.Executable,
                    "使用说明", MessageBoxButtons.OK, MessageBoxIcon.Information);
            };
            FormClosing += delegate { if (cancellation != null) cancellation.Cancel(); };
            FormClosed += delegate { if (applicationIcon != null) applicationIcon.Dispose(); };
            KeyPreview = true;
            KeyDown += async delegate(object sender, KeyEventArgs e)
            {
                if (e.KeyCode == Keys.F5 && !busy) { e.Handled = true; await ScanAsync(); }
                if (e.Control && e.KeyCode == Keys.C && grid.ContainsFocus)
                { e.SuppressKeyPress = true; CopySelection(); }
            };
            LoadDrives(preferredDrive);
            if (initialFile != null) SelectFile(initialFile);
            Shown += async delegate { await ScanAsync(); };
            // 全部控件创建后统一缩放，避免高 DPI 下字号已放大而布局仍保留 96 DPI 尺寸。
            AutoScaleDimensions = new SizeF(96F, 96F);
            AutoScaleMode = AutoScaleMode.Dpi;
            ResumeLayout(true);
            // DataGridView 的列宽和行高不随父布局自动缩放，需要同步其内部尺寸。
            float scale = CurrentAutoScaleDimensions.Width / 96F;
            foreach (DataGridViewColumn column in grid.Columns)
            {
                column.MinimumWidth = (int)Math.Ceiling(column.MinimumWidth * scale);
                if (column.AutoSizeMode != DataGridViewAutoSizeColumnMode.Fill)
                    column.Width = (int)Math.Ceiling(column.Width * scale);
            }
            grid.ColumnHeadersHeight = (int)Math.Ceiling(36 * scale);
            grid.RowTemplate.Height = (int)Math.Ceiling(32 * scale);
        }

        private string SelectedDrive { get { var item = drives.SelectedItem as DriveChoice; return item == null ? null : item.Root; } }

        private void ResetTarget()
        {
            if (busy) return;
            latest = null;
            search.Clear();
            ApplyFilter();
            diagnostics.Text = fileMode.Checked
                ? "选择文件 / 文件夹或粘贴完整路径，然后点击“扫描占用”。文件夹会包含所有子目录和文件。"
                : "选择磁盘后点击“扫描占用”。";
            SetBusy(false);
        }

        internal void SelectFile(string path)
        {
            if (busy) return;
            filePath.Text = ScanService.NormalizeFilePath(path);
            fileMode.Checked = true;
        }

        private void ChooseFile()
        {
            using (var dialog = new OpenFileDialog { Title = "选择需要检查占用的文件", Filter = "所有文件 (*.*)|*.*",
                CheckFileExists = true, Multiselect = false, RestoreDirectory = true })
            {
                if (dialog.ShowDialog(this) == DialogResult.OK) TryAction(() => SelectFile(dialog.FileName));
            }
        }

        private void ChooseFolder()
        {
            TryAction(() =>
            {
                string selected = FolderPicker.Show(this, fileMode.Checked ? filePath.Text : SelectedDrive);
                if (selected != null) SelectFile(selected);
            });
        }

        private Task<ScanResult> ScanTargetAsync(CancellationToken token)
        {
            return fileMode.Checked ? ScanService.ScanFileAsync(filePath.Text, token) : ScanService.ScanAsync(SelectedDrive, token);
        }

        private void ConfigureButton(Button button, string text, int width, bool primary)
        {
            button.Text = text;
            button.Width = width;
            button.Height = 34;
            button.Margin = new Padding(0, 3, 8, 3);
            button.FlatStyle = FlatStyle.Flat;
            button.FlatAppearance.BorderColor = primary ? Color.FromArgb(24, 86, 143) : Color.FromArgb(178, 188, 200);
            button.BackColor = primary ? Color.FromArgb(24, 86, 143) : Color.White;
            button.ForeColor = primary ? Color.White : ForeColor;
            button.UseVisualStyleBackColor = false;
        }

        private void AddColumn(string property, string title, int width)
        {
            grid.Columns.Add(new DataGridViewTextBoxColumn { Name = property, DataPropertyName = property,
                HeaderText = title, Width = width, SortMode = DataGridViewColumnSortMode.Automatic });
        }

        private void LoadDrives(string preferred)
        {
            drives.Items.Clear();
            foreach (var drive in DriveInfo.GetDrives().OrderBy(d => d.DriveType == DriveType.Removable ? 0 : 1))
            {
                if (drive.DriveType != DriveType.Removable && drive.DriveType != DriveType.Fixed) continue;
                string label = "";
                try { if (drive.IsReady) label = drive.VolumeLabel; } catch (IOException) { label = "未就绪"; }
                drives.Items.Add(new DriveChoice { Root = drive.Name, Display = drive.Name + "  " + label +
                    (drive.DriveType == DriveType.Removable ? "  [可移动]" : "  [本地磁盘]") });
            }
            int selected = -1;
            for (int i = 0; i < drives.Items.Count; i++)
                if (((DriveChoice)drives.Items[i]).Root.Equals(preferred, StringComparison.OrdinalIgnoreCase)) selected = i;
            if (drives.Items.Count > 0) drives.SelectedIndex = selected >= 0 ? selected : 0;
            UpdateDriveInfo();
            SetBusy(false);
        }

        private void UpdateDriveInfo()
        {
            try
            {
                var drive = new DriveInfo(SelectedDrive);
                driveInfo.Text = drive.IsReady ? string.Format("{0}   ·   可用 {1:F1} GB / 共 {2:F1} GB   ·   {3}",
                    drive.DriveFormat, drive.AvailableFreeSpace / 1073741824.0, drive.TotalSize / 1073741824.0,
                    Program.IsAdministrator ? "管理员权限" : "普通权限；发现不全时可切换管理员扫描") : "磁盘未就绪，请重新连接后刷新磁盘。";
            }
            catch (Exception ex) { driveInfo.Text = "无法读取磁盘信息：" + ex.Message; }
        }

        private async Task ScanAsync()
        {
            if (busy || (!fileMode.Checked && SelectedDrive == null)) return;
            SetBusy(true);
            latest = null;
            ApplyFilter();
            summary.Text = fileMode.Checked ? "正在检查所选路径，最多等待约 25 秒…" : "正在扫描 " + SelectedDrive + "，最多等待约 25 秒…";
            diagnostics.Text = "正在检查文件、文件夹以及运行中的程序和模块。扫描期间可以取消。";
            cancellation = new CancellationTokenSource();
            cancel.Enabled = true;
            try
            {
                var result = await ScanTargetAsync(cancellation.Token);
                if (IsDisposed || Disposing) return;
                latest = result;
                ApplyFilter();
                diagnostics.Text = (result.Warnings.Count == 0 ? "" : string.Join("\r\n", result.Warnings) + "\r\n") +
                    (fileMode.Checked ? "文件夹包含自身及其子目录和文件；未发现占用时仍可能存在无法读取的进程或文件映射。" :
                    "结果为即时快照；未发现占用不代表保证可拔出，请以 Windows 安全移除结果为准。");
            }
            catch (Exception ex)
            {
                if (!IsDisposed) { summary.Text = "扫描失败"; diagnostics.Text = ex.Message; }
            }
            finally
            {
                cancellation.Dispose();
                cancellation = null;
                if (!IsDisposed && !Disposing) { SetBusy(false); UpdateDriveInfo(); }
            }
        }

        private void ApplyFilter()
        {
            var table = new DataTable();
            table.Columns.Add("ProcessName", typeof(string));
            table.Columns.Add("Pid", typeof(int));
            table.Columns.Add("Kind", typeof(string));
            table.Columns.Add("Count", typeof(int));
            table.Columns.Add("Path", typeof(string));
            table.Columns.Add("Occupancy", typeof(Occupancy));
            string query = search.Text.Trim();
            if (latest != null)
            {
                foreach (var row in latest.Rows)
                    if (query.Length == 0 || (row.ProcessName + " " + row.Pid + " " + row.Path + " " + row.Image)
                        .IndexOf(query, StringComparison.OrdinalIgnoreCase) >= 0)
                        table.Rows.Add(row.ProcessName, row.Pid, row.Kind, row.Count, row.Path, row);
                summary.Text = string.Format("{0}  ·  {1} 个进程 / {2} 项占用  ·  {3:F1} 秒{4}",
                    latest.Cancelled ? "已取消" : latest.Completed ? "扫描结束" : "扫描未完成",
                    latest.Rows.Select(r => r.Pid).Distinct().Count(), latest.Rows.Count, latest.Seconds,
                    query.Length == 0 ? "" : "  ·  显示 " + table.Rows.Count + " 项");
                if (latest.Rows.Count == 0 && latest.Completed) summary.Text = "未发现可识别占用  ·  " + latest.Seconds.ToString("F1") + " 秒";
            }
            else if (!busy) summary.Text = "请点击“扫描占用”";
            var previous = grid.DataSource as DataTable;
            grid.DataSource = table;
            if (previous != null) previous.Dispose();
            UpdateSelection();
        }

        private Occupancy SelectedRow
        {
            get
            {
                var row = grid.CurrentRow == null ? null : grid.CurrentRow.DataBoundItem as DataRowView;
                return row == null ? null : row["Occupancy"] as Occupancy;
            }
        }

        private void UpdateSelection()
        {
            var row = SelectedRow;
            details.Text = row == null ? "选中一项占用后，可查看完整路径并处理对应进程。" :
                "占用路径：" + row.Path + "\r\n程序位置：" + (string.IsNullOrEmpty(row.Image) ? "无法读取" : row.Image);
            close.Enabled = terminate.Enabled = copy.Enabled = !busy && row != null;
            export.Enabled = !busy && latest != null;
        }

        private void SetBusy(bool value)
        {
            busy = value;
            scan.Enabled = !value && (fileMode.Checked ? !string.IsNullOrWhiteSpace(filePath.Text) : SelectedDrive != null);
            cancel.Enabled = value && cancellation != null;
            drives.Enabled = refresh.Enabled = eject.Enabled = !value && !fileMode.Checked;
            driveMode.Enabled = fileMode.Enabled = browse.Enabled = browseFolder.Enabled = !value;
            filePath.Enabled = !value && fileMode.Checked;
            elevate.Enabled = !value && !Program.IsAdministrator;
            progress.Visible = value;
            UpdateSelection();
        }

        private async Task ActAsync(bool force)
        {
            var row = SelectedRow;
            if (busy || row == null) return;
            SetBusy(true);
            string outcome = null;
            cancellation = new CancellationTokenSource();
            cancel.Enabled = true;
            try
            {
                summary.Text = fileMode.Checked ? "正在确认所选进程仍占用此路径…" : "正在确认所选进程仍占用此磁盘…";
                var fresh = await ScanTargetAsync(cancellation.Token);
                if (IsDisposed || Disposing) return;
                latest = fresh;
                ApplyFilter();
                diagnostics.Text = string.Join("\r\n", fresh.Warnings);
                if (fresh.Cancelled) return;
                if (!fresh.Rows.Any(r => r.Pid == row.Pid && r.Created == row.Created))
                    throw new InvalidOperationException("当前未再次确认该进程的占用，已停止操作。请查看最新结果后重新选择。");
                using (var action = new ProcessAction(row, force))
                {
                    string prompt = row.ProcessName + "  (PID " + row.Pid + ")\r\n" + row.Path + "\r\n\r\n" +
                        (force ? "将强制结束整个进程。该进程所有未保存的内容可能丢失，正在写入的文件可能损坏。\r\n确定继续？"
                        : "将请求应用正常关闭；如出现保存提示，请在该应用中处理。\r\n确定继续？");
                    if (MessageBox.Show(this, prompt, force ? "确认强制结束" : "确认正常关闭", MessageBoxButtons.YesNo,
                        force ? MessageBoxIcon.Warning : MessageBoxIcon.Question, MessageBoxDefaultButton.Button2) != DialogResult.Yes) return;
                    outcome = force ? await Task.Run(() => action.Terminate()) : action.CloseNormally();
                }
            }
            catch (Exception ex) { if (!IsDisposed) MessageBox.Show(this, ex.Message, "未执行操作", MessageBoxButtons.OK, MessageBoxIcon.Information); }
            finally
            {
                cancellation.Dispose();
                cancellation = null;
                if (!IsDisposed && !Disposing) SetBusy(false);
            }
            if (outcome != null && !IsDisposed)
            {
                await ScanAsync();
                if (!IsDisposed) diagnostics.Text = outcome + "\r\n" + diagnostics.Text;
            }
        }

        private void CopySelection()
        {
            var row = SelectedRow;
            if (row == null) return;
            TryAction(() => Clipboard.SetText("进程：" + row.ProcessName + "\r\nPID：" + row.Pid +
                "\r\n占用路径：" + row.Path + "\r\n类型：" + row.Kind + "\r\n程序位置：" + row.Image));
        }

        internal static string CsvCell(string value)
        {
            value = value ?? "";
            // 防止导出的文件名被表格软件解释成公式。
            if (value.Length > 0 && "=+-@\t\r\n".IndexOf(value[0]) >= 0) value = "'" + value;
            return "\"" + value.Replace("\"", "\"\"") + "\"";
        }

        private void ExportResults()
        {
            if (latest == null) return;
            using (var dialog = new SaveFileDialog { Filter = "CSV 文件 (*.csv)|*.csv", RestoreDirectory = true,
                InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory),
                FileName = "UsbOccupancy_" + DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".csv" })
            {
                if (dialog.ShowDialog(this) != DialogResult.OK) return;
                TryAction(() =>
                {
                    using (var writer = new StreamWriter(dialog.FileName, false, new UTF8Encoding(true)))
                    {
                        writer.WriteLine("进程名,PID,占用类型,数量,完整路径,程序位置");
                        foreach (var row in latest.Rows)
                            writer.WriteLine(string.Join(",", new[] { CsvCell(row.ProcessName), row.Pid.ToString(),
                                CsvCell(row.Kind), row.Count.ToString(), CsvCell(row.Path), CsvCell(row.Image) }));
                        writer.WriteLine("\r\n扫描说明," + CsvCell(latest.Completed ? "已完成；仍可能存在无法识别的占用" : "未完成，结果不完整"));
                        writer.WriteLine("扫描目标," + CsvCell(fileMode.Checked ? ScanService.NormalizeFilePath(filePath.Text) : SelectedDrive));
                        foreach (string warning in latest.Warnings) writer.WriteLine("限制," + CsvCell(warning));
                    }
                    diagnostics.Text = "已导出全部结果：" + dialog.FileName + "\r\n" + diagnostics.Text;
                });
            }
        }

        private void Elevate()
        {
            TryAction(() =>
            {
                Process.Start(new ProcessStartInfo(Program.Executable,
                    Program.RunArguments(SelectedDrive, fileMode.Checked ? filePath.Text : null)) {
                    UseShellExecute = true, Verb = "runas", WorkingDirectory = Path.GetDirectoryName(Program.Executable) });
                Close();
            });
        }

        private void TryAction(Action action)
        {
            try { action(); }
            catch (Exception ex) { MessageBox.Show(this, ex.Message, "操作未完成", MessageBoxButtons.OK, MessageBoxIcon.Information); }
        }
    }
}
