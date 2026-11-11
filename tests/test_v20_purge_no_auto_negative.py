# -*- coding: utf-8 -*-
"""Yanlış düğüm otomatik negatif öğretim kapalı (purge varsayılan)."""

from __future__ import annotations

import inspect

from core.wrong_node_scan import purge_wrong_nodes


def test_purge_defaults_save_negative_exemplars_false():
    sig = inspect.signature(purge_wrong_nodes)
    assert sig.parameters["save_negative_exemplars"].default is False
