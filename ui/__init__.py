# -*- coding: utf-8 -*-

def get_ui_package_summary() -> dict:
    try:
        return {"ui_ready": True}
    except Exception:
        return {"ui_ready": False}

