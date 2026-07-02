"""
Unit tests for src/kss_retrieval.py (KSS retrieval module ported from MSGNav).

Mocks call_openai_api to avoid real VLM calls.
"""

import os

# Clear proxy env vars before importing src.pred_eqa (which constructs OpenAI
# client at module load). socks5h scheme is unsupported by httpx.
for _k in list(os.environ):
    if _k.lower().endswith("_proxy"):
        del os.environ[_k]

import numpy as np
import pytest

from src import kss_retrieval


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------
class FakeBBox:
    """Mimics Pred-EQA bbox with .center attribute."""
    def __init__(self, center):
        self.center = center


class FakeEdge:
    """Mimics MSGNav edge object with .rel_img attribute."""
    def __init__(self, rel_img):
        self.rel_img = rel_img


class FakeObj(dict):
    """Mimics MapObjectDict: supports both obj['key'] and obj.key access."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value


def make_obj(class_name, room_label, center):
    """Build a mock MapObjectDict-like object (dict + attribute access)."""
    obj = FakeObj()
    obj.class_name = class_name
    obj.room_label = room_label
    obj.bbox = FakeBBox(center)
    obj.pcd = "mock_pcd"  # placeholder; not used by KSS but required by task spec
    return obj


@pytest.fixture
def mock_objs():
    return {
        1: make_obj("tv", "living_room", (1.0, 1.0, 1.0)),
        2: make_obj("speaker", "living_room", (2.0, 1.0, -1.0)),
        3: make_obj("sofa", "living_room", (0.5, 1.9, 2.6)),
    }


@pytest.fixture
def mock_edges():
    # edge (1,2) visible in img "img_a"; edge (1,3) visible in img "img_b"
    return {
        (1, 2): FakeEdge(rel_img=["img_a", "img_b"]),
        (1, 3): FakeEdge(rel_img=["img_b"]),
    }


@pytest.fixture
def mock_images():
    return {
        "img_a": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
        "img_b": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
    }


@pytest.fixture
def mock_image_to_edges():
    return {
        "img_a": [(1, 2)],
        "img_b": [(1, 2), (1, 3)],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestRelatedObjectKSS:
    def test_returns_list_of_int(self, mock_objs, mock_edges, monkeypatch):
        """related_object_KSS should return list[int] of selected object ids."""
        monkeypatch.setattr(
            kss_retrieval, "call_openai_api",
            lambda sys_prompt, contents: "1\n3\n",
        )
        result = kss_retrieval.related_object_KSS(
            question="Where can I sit?",
            objs=mock_objs,
            edges=mock_edges,
            top_k=10,
        )
        assert isinstance(result, list)
        assert all(isinstance(i, int) for i in result)
        assert result == [1, 3]

    def test_filters_invalid_ids(self, mock_objs, mock_edges, monkeypatch):
        """Ids not in objs should be filtered out."""
        monkeypatch.setattr(
            kss_retrieval, "call_openai_api",
            lambda sys_prompt, contents: "1\n99\n2\n",
        )
        result = kss_retrieval.related_object_KSS(
            question="test",
            objs=mock_objs,
            edges=mock_edges,
            top_k=10,
        )
        assert result == [1, 2]

    def test_respects_top_k(self, mock_objs, mock_edges, monkeypatch):
        """Result should be truncated to top_k."""
        monkeypatch.setattr(
            kss_retrieval, "call_openai_api",
            lambda sys_prompt, contents: "1\n2\n3\n",
        )
        result = kss_retrieval.related_object_KSS(
            question="test",
            objs=mock_objs,
            edges=mock_edges,
            top_k=2,
        )
        assert len(result) <= 2
        assert result == [1, 2]

    def test_none_response_returns_empty(self, mock_objs, mock_edges, monkeypatch):
        """If VLM returns None, result is empty list."""
        monkeypatch.setattr(
            kss_retrieval, "call_openai_api",
            lambda sys_prompt, contents: None,
        )
        result = kss_retrieval.related_object_KSS(
            question="test",
            objs=mock_objs,
            edges=mock_edges,
        )
        assert result == []


class TestEdgePruningKSS:
    def test_returns_three_dicts(self, mock_objs, mock_edges, mock_images, mock_image_to_edges):
        """edge_pruning_KSS should return (dict, dict, dict)."""
        connected, edges, imgs = kss_retrieval.edge_pruning_KSS(
            edges=mock_edges,
            objs=mock_objs,
            images=mock_images,
            selected_obj_id=[1],
            image_to_edges=mock_image_to_edges,
            prompt_h=32,
            prompt_w=32,
        )
        assert isinstance(connected, dict)
        assert isinstance(edges, dict)
        assert isinstance(imgs, dict)

    def test_empty_selection_returns_empties(self, mock_objs, mock_edges, mock_images, mock_image_to_edges):
        """Empty selected_obj_id should return three empty dicts."""
        connected, edges, imgs = kss_retrieval.edge_pruning_KSS(
            edges=mock_edges,
            objs=mock_objs,
            images=mock_images,
            selected_obj_id=[],
            image_to_edges=mock_image_to_edges,
            prompt_h=32,
            prompt_w=32,
        )
        assert connected == {}
        assert edges == {}
        assert imgs == {}

    def test_connected_objs_includes_neighbors(self, mock_objs, mock_edges, mock_images, mock_image_to_edges):
        """Selecting obj 1 should drag in neighbors 2 and 3 into connected_objs."""
        connected, edges, imgs = kss_retrieval.edge_pruning_KSS(
            edges=mock_edges,
            objs=mock_objs,
            images=mock_images,
            selected_obj_id=[1],
            image_to_edges=mock_image_to_edges,
            prompt_h=32,
            prompt_w=32,
        )
        # selected obj 1 plus neighbors 2,3
        assert 1 in connected
        assert 2 in connected
        assert 3 in connected

    def test_selected_edges_subset(self, mock_objs, mock_edges, mock_images, mock_image_to_edges):
        """selected_edges should contain edges between selected and connected objs."""
        connected, edges, imgs = kss_retrieval.edge_pruning_KSS(
            edges=mock_edges,
            objs=mock_objs,
            images=mock_images,
            selected_obj_id=[1],
            image_to_edges=mock_image_to_edges,
            prompt_h=32,
            prompt_w=32,
        )
        # edges (1,2) and (1,3) both expected
        assert (1, 2) in edges
        assert (1, 3) in edges

    def test_processed_images_are_base64_strings(self, mock_objs, mock_edges, mock_images, mock_image_to_edges):
        """processed_images values should be base64-encoded strings."""
        connected, edges, imgs = kss_retrieval.edge_pruning_KSS(
            edges=mock_edges,
            objs=mock_objs,
            images=mock_images,
            selected_obj_id=[1],
            image_to_edges=mock_image_to_edges,
            prompt_h=32,
            prompt_w=32,
        )
        for img_id, b64 in imgs.items():
            assert isinstance(b64, str)
            # base64 strings decode without error
            import base64 as b64mod
            b64mod.b64decode(b64)
