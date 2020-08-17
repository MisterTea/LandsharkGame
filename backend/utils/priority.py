import os
import sys

try:
    sys.getwindowsversion()
except AttributeError:
    isWindows = False
else:
    isWindows = True


def lowpriority():
    """ Set the priority of the process to below-normal."""

    if isWindows:
        # Based on:
        #   "Recipe 496767: Set Process Priority In Windows" on ActiveState
        #   http://code.activestate.com/recipes/496767/
        import win32api, win32process, win32con

        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.IDLE_PRIORITY_CLASS)
    else:
        os.nice(20)
