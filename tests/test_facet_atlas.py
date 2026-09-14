"""Unit tests for Decoupled Two-Eye Compound-Eye Facet Atlas."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from fly_doom.sensory.encoders.facet_atlas import CompoundEyeFacetAtlas, CompoundEyeFacet


def test_facet_atlas_initialization():
    atlas = CompoundEyeFacetAtlas()

    assert atlas.num_receptors == 3335
    assert len(atlas.facets) == 825
    assert atlas.provenance.tier == "biological_reconstruction"

    # Verify anatomical left/right separation
    left_facets = [f for f in atlas.facets if f.eye == "L"]
    right_facets = [f for f in atlas.facets if f.eye == "R"]

    assert len(left_facets) == 300
    assert len(right_facets) == 525
    assert len(left_facets) + len(right_facets) == 825

    # Verify all 3,335 receptors are partitioned into facets without loss or duplicate
    all_indices = []
    for f in atlas.facets:
        assert len(f.indices) >= 1
        assert len(f.body_ids) == len(f.indices)
        all_indices.extend(f.indices)

    assert len(all_indices) == 3335
    assert sorted(all_indices) == list(range(3335))


def test_square_to_oval_curvature_warp():
    atlas = CompoundEyeFacetAtlas()

    for f in atlas.facets:
        # Display coordinates x, y must be bounded within [-1.0, 1.0] oval radius
        r_sq = f.x * f.x + f.y * f.y
        assert r_sq <= 1.15, f"Facet {f.facet_id} outside oval boundary: ({f.x}, {f.y})"

        # Decoupling check: Display coordinates differ from original UV coordinates
        # (u_orig in [0, 1], v_orig in [0, 1], while display x, y in [-1, 1])
        assert f.x != pytest.approx(f.u_orig, abs=0.01) or f.y != pytest.approx(f.v_orig, abs=0.01)


def test_neural_superposition_aggregation():
    atlas = CompoundEyeFacetAtlas()

    # Test uniform receptor inputs
    uniform_vals = np.full(3335, 0.75, dtype=np.float32)
    facet_vals = atlas.compute_facet_values(uniform_vals)

    assert facet_vals.shape == (825,)
    assert np.allclose(facet_vals, 0.75)

    # Test distinct per-receptor signals
    np.random.seed(42)
    rand_vals = np.random.uniform(0.0, 1.0, size=3335).astype(np.float32)
    rand_facet_vals = atlas.compute_facet_values(rand_vals)

    assert rand_facet_vals.shape == (825,)
    # Verify facet value is exact arithmetic mean of its converging photoreceptors
    for f in atlas.facets[:10]:
        expected = float(np.mean(rand_vals[f.indices]))
        assert rand_facet_vals[f.facet_id] == pytest.approx(expected, abs=1e-5)


def test_facet_atlas_render_pil():
    atlas = CompoundEyeFacetAtlas()

    img = Image.new("RGB", (200, 100), (5, 10, 14))
    draw = ImageDraw.Draw(img)

    # Render with uniform values in copper mode
    test_signals = np.linspace(0.0, 1.0, 3335, dtype=np.float32)
    atlas.render_pil(draw, x=5, y=5, w=190, h=90, values=test_signals, mode="copper")

    arr = np.array(img)
    # Background is (5, 10, 14), facets should be rendered with copper tones
    assert np.any(arr != [5, 10, 14])

    # Also test amber_cyan mode
    img2 = Image.new("RGB", (200, 100), (5, 10, 14))
    draw2 = ImageDraw.Draw(img2)
    atlas.render_pil(draw2, x=5, y=5, w=190, h=90, values=test_signals, mode="amber_cyan")

    arr2 = np.array(img2)
    assert np.any(arr2 != [5, 10, 14])


def test_facet_atlas_serialization():
    atlas = CompoundEyeFacetAtlas()
    data = atlas.to_dict()

    assert data["num_receptors"] == 3335
    assert data["num_facets"] == 825
    assert len(data["facets"]) == 825

    facet0 = data["facets"][0]
    assert "id" in facet0
    assert "eye" in facet0
    assert "x" in facet0
    assert "y" in facet0
    assert "indices" in facet0
