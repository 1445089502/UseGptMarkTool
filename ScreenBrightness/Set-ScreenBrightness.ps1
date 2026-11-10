# 用于通过 DDC/CI 调节 Windows 台式机外接显示器亮度。
[CmdletBinding(DefaultParameterSetName = 'Gui')]
param(
    [Parameter(ParameterSetName = 'Gui')]
    [switch] $Gui,

    [Parameter(ParameterSetName = 'List')]
    [switch] $List,

    [Parameter(ParameterSetName = 'Set', Mandatory = $true)]
    [ValidateRange(0, 100)]
    [int] $Brightness,

    [Parameter(ParameterSetName = 'Increase', Mandatory = $true)]
    [ValidateRange(1, 100)]
    [int] $Increase,

    [Parameter(ParameterSetName = 'Decrease', Mandatory = $true)]
    [ValidateRange(1, 100)]
    [int] $Decrease,

    [Parameter(ParameterSetName = 'List')]
    [Parameter(ParameterSetName = 'Set')]
    [Parameter(ParameterSetName = 'Increase')]
    [Parameter(ParameterSetName = 'Decrease')]
    [Parameter(ParameterSetName = 'Gui')]
    [ValidateRange(-1, 32)]
    [int] $Monitor = -1
)

$nativeTypeName = 'ScreenBrightness.NativeMethods'
if (-not ($nativeTypeName -as [type])) {
    $nativeCode = @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace ScreenBrightness
{
    public sealed class MonitorState
    {
        public int Index { get; set; }
        public string DeviceName { get; set; }
        public string Description { get; set; }
        public uint Minimum { get; set; }
        public uint Current { get; set; }
        public uint Maximum { get; set; }
    }

    public sealed class BrightnessResult
    {
        public bool Found { get; set; }
        public bool Success { get; set; }
        public int ErrorCode { get; set; }
    }

    public static class NativeMethods
    {
        [StructLayout(LayoutKind.Sequential)]
        private struct Rect
        {
            public int Left;
            public int Top;
            public int Right;
            public int Bottom;
        }

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        private struct MonitorInfoEx
        {
            public int Size;
            public Rect Monitor;
            public Rect Work;
            public uint Flags;

            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)]
            public string DeviceName;
        }

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        private struct PhysicalMonitor
        {
            public IntPtr Handle;

            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
            public string Description;
        }

        private delegate bool MonitorEnumProc(
            IntPtr monitor,
            IntPtr deviceContext,
            ref Rect monitorRectangle,
            IntPtr data);

        [DllImport("user32.dll")]
        private static extern bool EnumDisplayMonitors(
            IntPtr deviceContext,
            IntPtr clipRectangle,
            MonitorEnumProc callback,
            IntPtr data);

        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        private static extern bool GetMonitorInfo(
            IntPtr monitor,
            ref MonitorInfoEx monitorInfo);

        [DllImport("Dxva2.dll", SetLastError = true)]
        private static extern bool GetNumberOfPhysicalMonitorsFromHMONITOR(
            IntPtr monitor,
            out uint count);

        [DllImport("Dxva2.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern bool GetPhysicalMonitorsFromHMONITOR(
            IntPtr monitor,
            uint count,
            [Out] PhysicalMonitor[] physicalMonitors);

        [DllImport("Dxva2.dll", SetLastError = true)]
        private static extern bool GetMonitorBrightness(
            IntPtr physicalMonitor,
            out uint minimum,
            out uint current,
            out uint maximum);

        [DllImport("Dxva2.dll", SetLastError = true)]
        private static extern bool SetMonitorBrightness(
            IntPtr physicalMonitor,
            uint brightness);

        [DllImport("Dxva2.dll", SetLastError = true)]
        private static extern bool DestroyPhysicalMonitors(
            uint count,
            [In] PhysicalMonitor[] physicalMonitors);

        public static MonitorState[] GetMonitorStates()
        {
            var states = new List<MonitorState>();
            EnumeratePhysicalMonitors(delegate(
                int index,
                string deviceName,
                PhysicalMonitor physicalMonitor,
                uint minimum,
                uint current,
                uint maximum)
            {
                states.Add(new MonitorState
                {
                    Index = index,
                    DeviceName = deviceName,
                    Description = physicalMonitor.Description,
                    Minimum = minimum,
                    Current = current,
                    Maximum = maximum
                });
            });
            return states.ToArray();
        }

        public static BrightnessResult SetBrightness(int targetIndex, uint brightness)
        {
            var result = new BrightnessResult();
            EnumeratePhysicalMonitors(delegate(
                int index,
                string deviceName,
                PhysicalMonitor physicalMonitor,
                uint minimum,
                uint current,
                uint maximum)
            {
                if (index != targetIndex)
                {
                    return;
                }

                result.Found = true;
                result.Success = SetMonitorBrightness(physicalMonitor.Handle, brightness);
                if (!result.Success)
                {
                    result.ErrorCode = Marshal.GetLastWin32Error();
                }
            });
            return result;
        }

        private delegate void PhysicalMonitorAction(
            int index,
            string deviceName,
            PhysicalMonitor physicalMonitor,
            uint minimum,
            uint current,
            uint maximum);

        private static void EnumeratePhysicalMonitors(PhysicalMonitorAction action)
        {
            int index = 0;
            EnumDisplayMonitors(
                IntPtr.Zero,
                IntPtr.Zero,
                delegate(
                    IntPtr monitor,
                    IntPtr deviceContext,
                    ref Rect monitorRectangle,
                    IntPtr data)
                {
                    uint count;
                    if (!GetNumberOfPhysicalMonitorsFromHMONITOR(monitor, out count) || count == 0)
                    {
                        return true;
                    }

                    var physicalMonitors = new PhysicalMonitor[count];
                    if (!GetPhysicalMonitorsFromHMONITOR(monitor, count, physicalMonitors))
                    {
                        return true;
                    }

                    string deviceName = GetDeviceName(monitor);
                    for (uint i = 0; i < count; i++)
                    {
                        uint minimum;
                        uint current;
                        uint maximum;
                        if (GetMonitorBrightness(
                            physicalMonitors[i].Handle,
                            out minimum,
                            out current,
                            out maximum))
                        {
                            action(
                                index,
                                deviceName,
                                physicalMonitors[i],
                                minimum,
                                current,
                                maximum);
                            index++;
                        }
                    }

                    DestroyPhysicalMonitors(count, physicalMonitors);
                    return true;
                },
                IntPtr.Zero);
        }

        private static string GetDeviceName(IntPtr monitor)
        {
            var monitorInfo = new MonitorInfoEx();
            monitorInfo.Size = Marshal.SizeOf(typeof(MonitorInfoEx));
            return GetMonitorInfo(monitor, ref monitorInfo)
                ? monitorInfo.DeviceName
                : string.Empty;
        }
    }
}
'@

    Add-Type -TypeDefinition $nativeCode -Language CSharp
}

function Get-MonitorStates {
    $states = @([ScreenBrightness.NativeMethods]::GetMonitorStates())
    if ($states.Count -eq 0) {
        throw '没有检测到支持 DDC/CI 的显示器。请在显示器菜单中开启 DDC/CI，并确认显卡驱动允许控制显示器。'
    }

    return $states
}

function ConvertTo-BrightnessPercent {
    param(
        [Parameter(Mandatory = $true)]
        [ScreenBrightness.MonitorState] $State
    )

    if ($State.Maximum -le $State.Minimum) {
        return 0
    }

    return [int][math]::Round(
        (($State.Current - $State.Minimum) * 100.0) /
        ($State.Maximum - $State.Minimum))
}

function ConvertTo-RawBrightness {
    param(
        [Parameter(Mandatory = $true)]
        [ScreenBrightness.MonitorState] $State,

        [Parameter(Mandatory = $true)]
        [ValidateRange(0, 100)]
        [int] $Percent
    )

    $raw = $State.Minimum + (($State.Maximum - $State.Minimum) * $Percent / 100.0)
    return [uint32][math]::Round($raw)
}

function Format-MonitorState {
    param(
        [Parameter(Mandatory = $true)]
        [ScreenBrightness.MonitorState] $State
    )

    [pscustomobject] @{
        Monitor     = $State.Index
        Device      = $State.DeviceName
        Description = $State.Description
        Brightness  = "$(ConvertTo-BrightnessPercent -State $State)%"
        RawValue    = "$($State.Current)/$($State.Maximum)"
    }
}

$states = @(Get-MonitorStates)

function Show-BrightnessWindow {
    param(
        [Parameter(Mandatory = $true)]
        [object[]] $InitialStates,

        [Parameter(Mandatory = $true)]
        [int] $InitialMonitor
    )

    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing

    [System.Windows.Forms.Application]::EnableVisualStyles()
    [System.Windows.Forms.Application]::SetCompatibleTextRenderingDefault($false)

    $form = New-Object System.Windows.Forms.Form
    $form.Text = '屏幕亮度'
    $form.ClientSize = New-Object System.Drawing.Size(560, 400)
    $form.MinimumSize = New-Object System.Drawing.Size(500, 380)
    $form.StartPosition = 'CenterScreen'
    $form.FormBorderStyle = 'Sizable'
    $form.MaximizeBox = $true
    $form.MinimizeBox = $true
    $form.AutoScaleMode = [System.Windows.Forms.AutoScaleMode]::Dpi
    $form.ShowInTaskbar = $true
    $form.BackColor = [System.Drawing.Color]::FromArgb(247, 248, 250)
    $form.Font = New-Object System.Drawing.Font('Microsoft YaHei UI', 9)

    $titleLabel = New-Object System.Windows.Forms.Label
    $titleLabel.Location = New-Object System.Drawing.Point(24, 18)
    $titleLabel.AutoSize = $true
    $titleLabel.Text = '屏幕亮度'
    $titleLabel.Font = New-Object System.Drawing.Font('Microsoft YaHei UI', 17, [System.Drawing.FontStyle]::Bold)
    $titleLabel.ForeColor = [System.Drawing.Color]::FromArgb(31, 41, 55)
    $form.Controls.Add($titleLabel)

    $subtitleLabel = New-Object System.Windows.Forms.Label
    $subtitleLabel.Location = New-Object System.Drawing.Point(26, 52)
    $subtitleLabel.AutoSize = $true
    $subtitleLabel.Text = '通过 DDC/CI 调节显示器硬件亮度'
    $subtitleLabel.ForeColor = [System.Drawing.Color]::FromArgb(107, 114, 128)
    $form.Controls.Add($subtitleLabel)

    $monitorLabel = New-Object System.Windows.Forms.Label
    $monitorLabel.Location = New-Object System.Drawing.Point(26, 86)
    $monitorLabel.AutoSize = $true
    $monitorLabel.Text = '显示器'
    $monitorLabel.ForeColor = [System.Drawing.Color]::FromArgb(55, 65, 81)
    $form.Controls.Add($monitorLabel)

    $monitorCombo = New-Object System.Windows.Forms.ComboBox
    $monitorCombo.Location = New-Object System.Drawing.Point(26, 107)
    $monitorCombo.Size = New-Object System.Drawing.Size(388, 28)
    $monitorCombo.DropDownStyle = 'DropDownList'
    $monitorCombo.FlatStyle = 'Standard'
    foreach ($state in $InitialStates) {
        [void]$monitorCombo.Items.Add(
            "[$($state.Index)] $($state.Description)  $($state.DeviceName)")
    }
    $form.Controls.Add($monitorCombo)

    $brightnessCaption = New-Object System.Windows.Forms.Label
    $brightnessCaption.Location = New-Object System.Drawing.Point(26, 153)
    $brightnessCaption.AutoSize = $true
    $brightnessCaption.Text = '目标亮度'
    $brightnessCaption.ForeColor = [System.Drawing.Color]::FromArgb(107, 114, 128)
    $form.Controls.Add($brightnessCaption)

    $brightnessValue = New-Object System.Windows.Forms.Label
    $brightnessValue.Location = New-Object System.Drawing.Point(305, 143)
    $brightnessValue.Size = New-Object System.Drawing.Size(109, 36)
    $brightnessValue.TextAlign = 'MiddleRight'
    $brightnessValue.Font = New-Object System.Drawing.Font('Microsoft YaHei UI', 20, [System.Drawing.FontStyle]::Bold)
    $brightnessValue.ForeColor = [System.Drawing.Color]::FromArgb(37, 99, 235)
    $form.Controls.Add($brightnessValue)

    $brightnessTrackBar = New-Object System.Windows.Forms.TrackBar
    $brightnessTrackBar.Location = New-Object System.Drawing.Point(21, 177)
    $brightnessTrackBar.Size = New-Object System.Drawing.Size(398, 45)
    $brightnessTrackBar.Minimum = 0
    $brightnessTrackBar.Maximum = 100
    $brightnessTrackBar.TickFrequency = 10
    $brightnessTrackBar.SmallChange = 1
    $brightnessTrackBar.LargeChange = 10
    $brightnessTrackBar.TickStyle = 'BottomRight'
    $form.Controls.Add($brightnessTrackBar)

    $minLabel = New-Object System.Windows.Forms.Label
    $minLabel.Location = New-Object System.Drawing.Point(27, 216)
    $minLabel.AutoSize = $true
    $minLabel.Text = '0'
    $minLabel.ForeColor = [System.Drawing.Color]::FromArgb(107, 114, 128)
    $form.Controls.Add($minLabel)

    $maxLabel = New-Object System.Windows.Forms.Label
    $maxLabel.Location = New-Object System.Drawing.Point(398, 216)
    $maxLabel.AutoSize = $true
    $maxLabel.Text = '100'
    $maxLabel.ForeColor = [System.Drawing.Color]::FromArgb(107, 114, 128)
    $form.Controls.Add($maxLabel)

    $decreaseButton = New-Object System.Windows.Forms.Button
    $decreaseButton.Location = New-Object System.Drawing.Point(26, 247)
    $decreaseButton.Size = New-Object System.Drawing.Size(72, 34)
    $decreaseButton.Text = '-10'
    $decreaseButton.UseVisualStyleBackColor = $true
    $form.Controls.Add($decreaseButton)

    $increaseButton = New-Object System.Windows.Forms.Button
    $increaseButton.Location = New-Object System.Drawing.Point(105, 247)
    $increaseButton.Size = New-Object System.Drawing.Size(72, 34)
    $increaseButton.Text = '+10'
    $increaseButton.UseVisualStyleBackColor = $true
    $form.Controls.Add($increaseButton)

    $statusLabel = New-Object System.Windows.Forms.Label
    $statusLabel.Location = New-Object System.Drawing.Point(190, 255)
    $statusLabel.Size = New-Object System.Drawing.Size(140, 22)
    $statusLabel.TextAlign = 'MiddleRight'
    $statusLabel.AutoEllipsis = $true
    $statusLabel.Text = '拖动滑块后点击“应用”'
    $statusLabel.ForeColor = [System.Drawing.Color]::FromArgb(107, 114, 128)
    $form.Controls.Add($statusLabel)

    $applyButton = New-Object System.Windows.Forms.Button
    $applyButton.Location = New-Object System.Drawing.Point(342, 247)
    $applyButton.Size = New-Object System.Drawing.Size(72, 34)
    $applyButton.Text = '应用'
    $applyButton.DialogResult = [System.Windows.Forms.DialogResult]::None
    $applyButton.UseVisualStyleBackColor = $true
    $form.Controls.Add($applyButton)
    $form.AcceptButton = $applyButton

    $layoutWindow = {
        $clientWidth = $form.ClientSize.Width
        $clientHeight = $form.ClientSize.Height
        $left = 26
        $right = 26
        $contentWidth = [math]::Max(360, $clientWidth - $left - $right)

        $titleLabel.Location = [System.Drawing.Point]::new([int]$left, 18)
        $subtitleLabel.Location = [System.Drawing.Point]::new(
            [int]$left,
            [int]($titleLabel.Bottom + 2))

        $monitorLabel.Location = [System.Drawing.Point]::new(
            [int]$left,
            [int]($subtitleLabel.Bottom + 12))
        $monitorCombo.Location = [System.Drawing.Point]::new(
            [int]$left,
            [int]($monitorLabel.Bottom + 4))
        $monitorCombo.Size = [System.Drawing.Size]::new([int]$contentWidth, 28)

        $brightnessCaption.Location = [System.Drawing.Point]::new(
            [int]$left,
            [int]($monitorCombo.Bottom + 18))
        $brightnessValue.Location = [System.Drawing.Point]::new(
            [int]($clientWidth - $right - 120),
            [int]($brightnessCaption.Top - 10))
        $brightnessValue.Size = [System.Drawing.Size]::new(120, 36)

        $trackY = $brightnessCaption.Bottom + 2
        $brightnessTrackBar.Location = [System.Drawing.Point]::new(
            [int]($left - 5),
            [int]$trackY)
        $brightnessTrackBar.Size = [System.Drawing.Size]::new(
            [int]($contentWidth + 10),
            48)

        $minLabel.Location = [System.Drawing.Point]::new(
            [int]$left,
            [int]($brightnessTrackBar.Bottom - 7))
        $maxLabel.Location = [System.Drawing.Point]::new(
            [int]($clientWidth - $right - $maxLabel.PreferredWidth),
            [int]($brightnessTrackBar.Bottom - 7))

        $buttonY = $clientHeight - 57
        $decreaseButton.Location = [System.Drawing.Point]::new([int]$left, [int]$buttonY)
        $increaseButton.Location = [System.Drawing.Point]::new([int]($left + 79), [int]$buttonY)
        $applyX = $clientWidth - $right - 72
        $applyButton.Location = [System.Drawing.Point]::new([int]$applyX, [int]$buttonY)
        $statusLabel.Location = [System.Drawing.Point]::new(190, [int]($buttonY + 8))
        $statusLabel.Size = [System.Drawing.Size]::new(
            [int][math]::Max(120, $applyX - 202),
            22)
    }.GetNewClosure()

    $form.Add_Resize({ & $layoutWindow }.GetNewClosure())
    & $layoutWindow

    $monitorStates = @($InitialStates)
    $getMonitorStates = ${function:Get-MonitorStates}.GetNewClosure()
    $convertToBrightnessPercent = ${function:ConvertTo-BrightnessPercent}.GetNewClosure()
    $convertToRawBrightness = ${function:ConvertTo-RawBrightness}.GetNewClosure()

    $updateTargetValue = {
        if ($monitorCombo.SelectedIndex -ge 0) {
            $brightnessValue.Text = "$($brightnessTrackBar.Value)%"
        }
    }.GetNewClosure()

    $loadSelectedMonitor = {
        if ($monitorCombo.SelectedIndex -lt 0) {
            return
        }

        $selectedState = $monitorStates[$monitorCombo.SelectedIndex]
        $currentPercent = & $convertToBrightnessPercent -State $selectedState
        $brightnessTrackBar.Value = [math]::Max(0, [math]::Min(100, $currentPercent))
        & $updateTargetValue
        $statusLabel.Text = '拖动滑块后点击“应用”'
        $statusLabel.ForeColor = [System.Drawing.Color]::FromArgb(107, 114, 128)
    }.GetNewClosure()

    $applySelectedBrightness = {
        if ($monitorCombo.SelectedIndex -lt 0) {
            return
        }

        $selectedState = $monitorStates[$monitorCombo.SelectedIndex]
        $targetPercent = $brightnessTrackBar.Value
        $rawBrightness = & $convertToRawBrightness -State $selectedState -Percent $targetPercent
        $result = [ScreenBrightness.NativeMethods]::SetBrightness(
            $selectedState.Index,
            $rawBrightness)

        if (-not $result.Found) {
            throw "显示器 $($selectedState.Index) 在设置过程中消失。"
        }

        if (-not $result.Success) {
            throw "显示器设置失败，Windows 错误码：$($result.ErrorCode)。请确认显示器支持并开启 DDC/CI。"
        }

        $monitorStates = @(& $getMonitorStates)
        $refreshedState = $monitorStates |
            Where-Object { $_.Index -eq $selectedState.Index } |
            Select-Object -First 1
        if ($null -ne $refreshedState) {
            $brightnessTrackBar.Value = [math]::Max(
                0,
                [math]::Min(100, (& $convertToBrightnessPercent -State $refreshedState)))
        }

        & $updateTargetValue
        $statusLabel.Text = "已应用到显示器 $($selectedState.Index)"
        $statusLabel.ForeColor = [System.Drawing.Color]::FromArgb(22, 128, 82)
    }.GetNewClosure()

    $monitorCombo.Add_SelectedIndexChanged($loadSelectedMonitor)
    $brightnessTrackBar.Add_Scroll($updateTargetValue)
    $decreaseButton.Add_Click({
        $brightnessTrackBar.Value = [math]::Max(0, $brightnessTrackBar.Value - 10)
        & $updateTargetValue
    }.GetNewClosure())
    $increaseButton.Add_Click({
        $brightnessTrackBar.Value = [math]::Min(100, $brightnessTrackBar.Value + 10)
        & $updateTargetValue
    }.GetNewClosure())
    $applyButton.Add_Click({
        try {
            & $applySelectedBrightness
        }
        catch {
            $statusLabel.Text = '应用失败，请检查 DDC/CI'
            $statusLabel.ForeColor = [System.Drawing.Color]::FromArgb(185, 28, 28)
            [void][System.Windows.Forms.MessageBox]::Show(
                $form,
                $_.Exception.Message,
                '设置亮度失败',
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Error)
        }
    }.GetNewClosure())

    $initialIndex = 0
    if ($InitialMonitor -ge 0) {
        $candidateIndex = 0
        foreach ($state in $monitorStates) {
            if ($state.Index -eq $InitialMonitor) {
                $initialIndex = $candidateIndex
                break
            }
            $candidateIndex++
        }
    }

    $monitorCombo.SelectedIndex = $initialIndex
    $form.Add_Shown({ $monitorCombo.Focus() }.GetNewClosure())

    try {
        [void]$form.ShowDialog()
    }
    finally {
        $form.Dispose()
    }
}

if ($PSCmdlet.ParameterSetName -eq 'Gui') {
    Show-BrightnessWindow -InitialStates $states -InitialMonitor $Monitor
    return
}

if ($PSCmdlet.ParameterSetName -eq 'List') {
    $states | ForEach-Object { Format-MonitorState -State $_ }
    return
}

$selectedStates = if ($Monitor -ge 0) {
    @($states | Where-Object { $_.Index -eq $Monitor })
}
else {
    $states
}

if ($selectedStates.Count -eq 0) {
    throw "找不到编号为 $Monitor 的显示器。请先运行 -List 查看编号。"
}

foreach ($state in $selectedStates) {
    $currentPercent = ConvertTo-BrightnessPercent -State $state
    switch ($PSCmdlet.ParameterSetName) {
        'Set'      { $targetPercent = $Brightness }
        'Increase' { $targetPercent = [math]::Min(100, $currentPercent + $Increase) }
        'Decrease' { $targetPercent = [math]::Max(0, $currentPercent - $Decrease) }
        default    { throw '未指定有效的亮度操作。' }
    }

    $rawBrightness = ConvertTo-RawBrightness -State $state -Percent $targetPercent
    $result = [ScreenBrightness.NativeMethods]::SetBrightness(
        $state.Index,
        $rawBrightness)

    if (-not $result.Found) {
        throw "显示器 $($state.Index) 在设置过程中消失。"
    }

    if (-not $result.Success) {
        throw "显示器 $($state.Index) 设置失败，Windows 错误码：$($result.ErrorCode)。请确认显示器支持并开启 DDC/CI。"
    }
}

$updatedStates = @(Get-MonitorStates)
$updatedStates |
    Where-Object {
        $Monitor -lt 0 -or $_.Index -eq $Monitor
    } |
    ForEach-Object { Format-MonitorState -State $_ }
