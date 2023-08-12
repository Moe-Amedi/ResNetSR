import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[('style.qss', '.'), ('datasets.py', '.'), ('main.ui', '.'), ('main_ui.py', '.'), ('mainwindow.py', '.'), ('ResNetArch.py', '.'), ('run.py', '.'), ('utils.py', '.')],
    hiddenimports=['torch.nn.functional', 'torchvision', 'PIL', 'os', 'PyQt5', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets', 'torch', 'torch.nn', 'torchvision.transforms', 'typing', 'torch.optim', 'torch.utils.data', 'numpy', 'matplotlib.pyplot', 'torchvision.utils', 'pytorch_msssim'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data,
          cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name='ResNetSR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ResNetSR'
)