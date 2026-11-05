# -*- coding: utf-8 -*-


def get_core_package_summary() -> dict:
    try:
        return {"core_ready": True}
    except Exception:
        return {"core_ready": False}


def get_core_package_ready_flag() -> dict:
    return {"package": "core", "ready": True}


def get_core_package_version_info() -> dict:
    return {"package": "core", "version": "2.0.0", "ready": True}


def get_core_package_diagnostic_snapshot() -> dict:
    return {"package": "core", "status": "ok", "ready": True}


def get_core_package_capabilities() -> list:
    return ["core_package", "diagnostic_snapshot", "version_info"]

