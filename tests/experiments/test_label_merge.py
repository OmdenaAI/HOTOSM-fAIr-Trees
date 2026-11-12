import subprocess
import sys

import numpy as np

from experiments.label_study.merge import apply_clusters, cluster_boxes, merge_crowns
from fair_trees.decode import Detections


def dets(boxes, scores=None):
    b = np.asarray(boxes, np.float32).reshape(-1, 4)
    s = np.asarray(scores if scores is not None else [0.5] * len(b), np.float32)
    return Detections(b, s)


def test_touching_and_overlapping_boxes_join_one_cluster():
    labels = cluster_boxes(np.array([[0, 0, 2, 2], [2, 0, 4, 2], [3, 1, 5, 3], [10, 10, 12, 12]], float), 0.0)
    assert labels.tolist() == [0, 0, 0, 1]


def test_gap_links_boxes_up_to_the_gap_in_both_axes():
    boxes = np.array([[0, 0, 2, 2], [2.5, 0, 4, 2], [0, 2.6, 2, 4]], float)
    assert cluster_boxes(boxes, 0.5).tolist() == [0, 0, 1]
    assert cluster_boxes(boxes, 0.6).tolist() == [0, 0, 0]


def test_diagonal_neighbours_need_both_axes_within_gap():
    assert cluster_boxes(np.array([[0, 0, 2, 2], [3, 3, 5, 5]], float), 0.5).tolist() == [0, 1]


def test_chain_is_transitive():
    boxes = np.array([[i * 2, 0, i * 2 + 2, 2] for i in range(6)], float)
    assert set(cluster_boxes(boxes, 0.0).tolist()) == {0}


def test_clusters_of_at_least_n_become_one_group_box():
    d = dets([[0, 0, 2, 2], [2, 0, 4, 2], [2, 2, 4, 5], [10, 10, 12, 12]], [0.3, 0.9, 0.4, 0.6])
    m = merge_crowns(d, 0.0, 3)
    assert m.groups.boxes.tolist() == [[0, 0, 4, 5]]
    assert m.groups.scores.tolist() == [np.float32(0.9)]
    assert m.group_sizes.tolist() == [3]
    assert m.trees.boxes.tolist() == [[10, 10, 12, 12]] and m.tree_index.tolist() == [3]


def test_small_clusters_stay_trees():
    m = merge_crowns(dets([[0, 0, 2, 2], [2, 0, 4, 2]]), 0.0, 3)
    assert len(m.groups.boxes) == 0 and m.tree_index.tolist() == [0, 1]


def test_empty_input():
    m = merge_crowns(Detections.empty(), 1.0, 2)
    assert m.trees.boxes.shape == (0, 4) and m.groups.boxes.shape == (0, 4) and m.group_sizes.shape == (0,)


def test_apply_clusters_reuses_labels_for_any_n():
    d = dets([[0, 0, 2, 2], [2, 0, 4, 2], [10, 10, 12, 12]])
    labels = cluster_boxes(d.boxes, 0.0)
    assert len(apply_clusters(d, labels, 2).groups.boxes) == 1
    assert len(apply_clusters(d, labels, 3).groups.boxes) == 0


def test_merge_imports_only_numpy():
    code = (
        "import sys, experiments.label_study.merge; "
        "print(sorted(m for m in ('torch','scipy','pandas','shapely','geopandas') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
