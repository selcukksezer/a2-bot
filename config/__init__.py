# -*- coding: utf-8 -*-


def get_config_package_summary() -> dict:
    try:
        return {"config_ready": True}
    except Exception:
        return {"config_ready": False}

