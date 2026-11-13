"""A separate real Windows-console process; no rendering from a saved capture.

Displays the wall clock and a continuous monotonic timer. The request marker
only adds a second timer; it never resets or stops the continuous timer.
"""
import ctypes
from ctypes import wintypes as w
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(sys.argv[1])
X, Y = map(int, sys.argv[2:4])
ctypes.windll.shcore.SetProcessDpiAwareness(1)
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
user = ctypes.WinDLL('user32', use_last_error=True)
kernel.GetConsoleWindow.restype = w.HWND
kernel.GetStdHandle.argtypes = [w.DWORD]
kernel.GetStdHandle.restype = w.HANDLE
user.SetWindowPos.argtypes = [w.HWND, w.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, w.UINT]
kernel.SetConsoleTitleW.argtypes = [w.LPCWSTR]
user.ShowWindow.argtypes = [w.HWND, ctypes.c_int]
user.GetWindowRect.argtypes = [w.HWND, ctypes.POINTER(w.RECT)]

class Coord(ctypes.Structure):
    _fields_ = [('X', ctypes.c_short), ('Y', ctypes.c_short)]

class SmallRect(ctypes.Structure):
    _fields_ = [('Left', ctypes.c_short), ('Top', ctypes.c_short), ('Right', ctypes.c_short), ('Bottom', ctypes.c_short)]

class Font(ctypes.Structure):
    _fields_ = [('cbSize', w.ULONG), ('nFont', w.DWORD), ('dwFontSize', Coord),
                ('FontFamily', w.UINT), ('FontWeight', w.UINT), ('FaceName', w.WCHAR * 32)]

class Cursor(ctypes.Structure):
    _fields_ = [('dwSize', w.DWORD), ('bVisible', w.BOOL)]

kernel.SetCurrentConsoleFontEx.argtypes = [w.HANDLE, w.BOOL, ctypes.POINTER(Font)]
kernel.SetConsoleWindowInfo.argtypes = [w.HANDLE, w.BOOL, ctypes.POINTER(SmallRect)]
kernel.SetConsoleScreenBufferSize.argtypes = [w.HANDLE, Coord]
kernel.SetConsoleCursorPosition.argtypes = [w.HANDLE, Coord]
kernel.SetConsoleCursorInfo.argtypes = [w.HANDLE, ctypes.POINTER(Cursor)]
kernel.SetConsoleTextAttribute.argtypes = [w.HANDLE, w.WORD]

try:
    kernel.SetConsoleTitleW('Command Prompt - LIVE CLOCK')
    hwnd = kernel.GetConsoleWindow()
    assert hwnd, 'A native Windows console is required'
    stdout = kernel.GetStdHandle(w.DWORD(-11))
    font = Font(ctypes.sizeof(Font), 0, Coord(0, 16), 54, 400, 'Consolas')
    assert kernel.SetCurrentConsoleFontEx(stdout, False, ctypes.byref(font))
    kernel.SetConsoleScreenBufferSize(stdout, Coord(80, 300))
    kernel.SetConsoleWindowInfo(stdout, True, ctypes.byref(SmallRect(0, 0, 38, 10)))
    kernel.SetConsoleCursorInfo(stdout, ctypes.byref(Cursor(25, False)))
    kernel.SetConsoleTextAttribute(stdout, 15)
    # Position only this process's own console. Keep it visible over the app.
    user.ShowWindow(hwnd, 4)
    assert user.SetWindowPos(hwnd, w.HWND(-1), X, Y, 505, 300, 0x0040)
    bounds = w.RECT()
    user.GetWindowRect(hwnd, ctypes.byref(bounds))
    start = time.perf_counter()
    (HERE/'clock-ready.json').write_text(json.dumps(dict(pid=os.getpid(), started=dt.datetime.now().astimezone().isoformat(),
        perf_counter=start, x=X, y=Y, width=505, height=300, clock='datetime.now + time.perf_counter',
        independent_process=True, actual_bounds=[bounds.left,bounds.top,bounds.right,bounds.bottom])), encoding='utf-8')
    with (HERE/'clock-ticks.jsonl').open('x', encoding='utf-8', buffering=1) as log:
        while not (HERE/'clock.stop').exists():
            now = dt.datetime.now().astimezone()
            counter = time.perf_counter()
            signal = {}
            try:
                signal = json.loads((HERE/'clock-signal.json').read_text(encoding='utf-8'))
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            elapsed = counter-start
            request_elapsed = None
            phase = 'WAIT'
            if 'request_perf_counter' in signal:
                request_elapsed = signal.get('request_finished_perf_counter', counter)-signal['request_perf_counter']
                phase = 'DONE' if 'request_finished_perf_counter' in signal else 'LIVE'
            lines = [
                'WINDOWS / INDEPENDENT CLOCK',
                '------------------------------------',
                'DATE     '+now.strftime('%Y-%m-%d'),
                'JST      '+now.strftime('%H:%M:%S.%f')[:-3],
                '',
                f'ELAPSED  {elapsed:010.2f} s',
                'REQUEST  '+(f'{request_elapsed:010.2f} s' if request_elapsed is not None else 'WAITING'),
                'STATUS   '+phase,
                'Clock continues after completion.'
            ]
            kernel.SetConsoleCursorPosition(stdout, Coord(0, 0))
            sys.stdout.write('\n'.join(line.ljust(37) for line in lines))
            sys.stdout.flush()
            log.write(json.dumps(dict(wall=now.isoformat(timespec='milliseconds'), perf_counter=counter,
                                     elapsed_s=elapsed, request_elapsed_s=request_elapsed, phase=phase))+'\n')
            time.sleep(.1)
except Exception as exc:
    (HERE/'clock-failed.json').write_text(json.dumps({'error':repr(exc)}), encoding='utf-8')
    raise
