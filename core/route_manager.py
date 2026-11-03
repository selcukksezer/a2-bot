# -*- coding: utf-8 -*-
"""
Route & Node Database Manager
Handles saving nodes, loading routes, cooldown management, and map patches.
"""

import os
import time
import json
import cv2
import numpy as np

ROUTES_FILE = "routes/routes.json"
PATCH_DIR = "routes/patches"

class RouteManager:
    def __init__(self, routes_file=ROUTES_FILE, patch_dir=PATCH_DIR):
        self.routes_file = routes_file
        self.patch_dir = patch_dir
        os.makedirs(os.path.dirname(self.routes_file), exist_ok=True)
        os.makedirs(self.patch_dir, exist_ok=True)
        self.data = self.load_data()

    def load_data(self):
        if os.path.exists(self.routes_file):
            try:
                with open(self.routes_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[RouteManager] Hata veri okunurken: {e}")
        return {
            "regions": {
                "DefaultRegion": {
                    "routes": {
                        "All": []
                    },
                    "nodes": []
                }
            },
            "active_region": "DefaultRegion",
            "active_route": "All"
        }

    def save_data(self):
        with open(self.routes_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def add_node(self, map_x, map_y, patch_img=None, region_name=None, route_name=None):
        """
        Saves a newly discovered node.
        """
        region_name = region_name or self.data.get("active_region", "DefaultRegion")
        route_name = route_name or self.data.get("active_route", "All")

        if region_name not in self.data["regions"]:
            self.data["regions"][region_name] = {"routes": {"All": []}, "nodes": []}

        node_id = len(self.data["regions"][region_name]["nodes"]) + 1
        patch_filename = f"{region_name}_node_{node_id}.png"
        patch_path = os.path.join(self.patch_dir, patch_filename)

        if patch_img is not None:
            cv2.imwrite(patch_path, patch_img)

        node_info = {
            "id": node_id,
            "name": f"Node {node_id}",
            "map_x": int(map_x),
            "map_y": int(map_y),
            "patch_file": patch_filename if patch_img is not None else None,
            "last_visited": 0,
            "visit_count": 0,
            "failures": 0,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        self.data["regions"][region_name]["nodes"].append(node_info)

        # Add to active route and All
        if "All" not in self.data["regions"][region_name]["routes"]:
            self.data["regions"][region_name]["routes"]["All"] = []
        self.data["regions"][region_name]["routes"]["All"].append(node_id)

        if route_name != "All":
            if route_name not in self.data["regions"][region_name]["routes"]:
                self.data["regions"][region_name]["routes"][route_name] = []
            if node_id not in self.data["regions"][region_name]["routes"][route_name]:
                self.data["regions"][region_name]["routes"][route_name].append(node_id)

        self.save_data()
        return node_info

    def get_route_nodes(self, region_name=None, route_name=None):
        region_name = region_name or self.data.get("active_region", "DefaultRegion")
        route_name = route_name or self.data.get("active_route", "All")

        region = self.data["regions"].get(region_name, {})
        all_nodes = {n["id"]: n for n in region.get("nodes", [])}
        route_node_ids = region.get("routes", {}).get(route_name, [])

        return [all_nodes[nid] for nid in route_node_ids if nid in all_nodes]

    def get_next_ready_node(self, rest_duration_sec=300):
        """
        Returns the next node in the route that has rested for at least rest_duration_sec (default 5 min).
        """
        nodes = self.get_route_nodes()
        now = time.time()
        for node in nodes:
            if now - node.get("last_visited", 0) >= rest_duration_sec:
                return node
        return None

    def mark_node_visited(self, node_id, region_name=None):
        region_name = region_name or self.data.get("active_region", "DefaultRegion")
        for node in self.data["regions"].get(region_name, {}).get("nodes", []):
            if node["id"] == node_id:
                node["last_visited"] = time.time()
                node["visit_count"] = node.get("visit_count", 0) + 1
                node["failures"] = 0
                self.save_data()
                return

    def mark_node_failed(self, node_id, region_name=None):
        region_name = region_name or self.data.get("active_region", "DefaultRegion")
        for node in self.data["regions"].get(region_name, {}).get("nodes", []):
            if node["id"] == node_id:
                node["failures"] = node.get("failures", 0) + 1
                self.save_data()
                return
