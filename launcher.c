#define UNICODE
#define _UNICODE
#include <windows.h>
#include <wchar.h>
#include <stdio.h>

/* A small Windows launcher; all application behavior lives in readable Python. */
int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR args, int show) {
    wchar_t root[32768], python[32768], script[32768], command[32768];
    DWORD length = GetModuleFileNameW(NULL, root, 32768);
    if (!length || length >= 32768) return 1;
    wchar_t *slash = wcsrchr(root, L'\\');
    if (!slash) return 1;
    *slash = 0;
    if (wcslen(root) > 12000 || wcslen(args) > 1000) return 1;
    swprintf(python, 32768, L"%ls\\runtime\\pythonw.exe", root);
    swprintf(script, 32768, L"%ls\\app\\bootstrap.py", root);
    if (GetFileAttributesW(python) == INVALID_FILE_ATTRIBUTES ||
        GetFileAttributesW(script) == INVALID_FILE_ATTRIBUTES) {
        MessageBoxW(NULL,
            L"Extract the ENTIRE AuraCam folder from the ZIP first.\n"
            L"Keep AuraCam.exe, app and runtime together.", L"AuraCam", MB_OK | MB_ICONERROR);
        return 2;
    }
    swprintf(command, 32768, L"\"%ls\" -X utf8 \"%ls\" %ls", python, script, args);
    STARTUPINFOW si = {0};
    PROCESS_INFORMATION pi = {0};
    si.cb = sizeof(si);
    if (!CreateProcessW(python, command, NULL, NULL, FALSE, CREATE_NO_WINDOW,
                        NULL, root, &si, &pi)) {
        wchar_t message[256];
        swprintf(message, 256, L"Cannot start AuraCam. Windows error: %lu.\n"
                 L"Extract the entire ZIP and try again.", GetLastError());
        MessageBoxW(NULL, message, L"AuraCam", MB_OK | MB_ICONERROR);
        return 3;
    }
    CloseHandle(pi.hThread);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    if (code >= 0xC0000000) {
        wchar_t message[256];
        swprintf(message, 256, L"AuraCam stopped with Windows code 0x%08lX.\n"
            L"Please report this code. See README_RU.txt for troubleshooting.", code);
        MessageBoxW(NULL, message, L"AuraCam", MB_OK | MB_ICONERROR);
    }
    return (int)code;
}
