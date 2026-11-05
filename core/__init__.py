# -*- coding: utf-8 -*-


def get_core_package_summary() -> dict:
    try:
        return {"core_ready": True}
    except Exception:
        return {"core_ready": False}

