# 项目自带 Bellhop

`bellhop.exe` 原样复制自本机已安装的 Acoustics Toolbox：`atWin10_2020_11_4/windows-bin-20201102/bellhop.exe`。这是 Windows x64 程序；原 GPL v3 许可证保留在 `LICENSE`，未修改二进制。

主线通过项目相对路径直接调用此程序，不依赖 MATLAB 全局路径中的 `bellhop.m` 或其他项目的 Bellhop 安装。只检查了文件与导入信息，未运行程序；其导入表包含 Windows 系统库 KERNEL32、msvcrt、USER32。

当前整理面向 Windows 间迁移。其他操作系统需提供对应平台的 Bellhop 可执行程序，并在根目录 `RUN_DATA_GEN_V6.m` 修改 `bellhop_executable`。MATLAB 与 Signal Processing Toolbox 仍需在目标电脑安装。
