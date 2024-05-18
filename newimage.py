import win32gui
from PIL import ImageGrab
toplist,winlist=[],[]
def e(hwnd,result):
    winlist.append((hwnd,win32gui.GetWindowText(hwnd)))
win32gui.EnumWindows(e,toplist)
lg=[(hwnd,title)for hwnd,title in winlist if 'LM-G900'in title.lower()]
lg=lg[0]
hwnd=lg[0]
win32gui.SetForegroundWindow(hwnd)
bbox=win32gui.GetWindowRect(hwnd)
img=ImageGrab.grab(bbox)
img.show()