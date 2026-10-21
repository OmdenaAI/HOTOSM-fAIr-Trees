"""Biome lookup from an ecoregion polygon file (e.g. RESOLVE Ecoregions 2017)."""

from collections.abc import Callable
from pathlib import Path


def resolve_biome_fn(path: Path, field: str = "BIOME_NAME") -> Callable[[float, float], str]:
    """A ``(lon, lat) -> biome`` lookup over the polygons in ``path``; ``"unknown"`` outside every polygon."""
    import geopandas
    import shapely

    frame = geopandas.read_file(path)[[field, "geometry"]].to_crs(4326)
    index = frame.sindex

    def biome(lon: float, lat: float) -> str:
        hits = index.query(shapely.Point(lon, lat), predicate="intersects")
        return str(frame[field].iloc[hits.min()]) if len(hits) else "unknown"

    return biome
