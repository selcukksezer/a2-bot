# -*- coding: utf-8 -*-


def get_tools_package_summary() -> dict:
    try:
        return {"tools_ready": True}
    except Exception:
        return {"tools_ready": False}


def get_tools_package_ready_flag() -> dict:
    return {"package": "tools", "ready": True}


