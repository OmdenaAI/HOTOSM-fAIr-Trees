"""Merge agent CSVs of one batch into relabel boxes (GeoJSON EPSG:4326 + CSV).
Each tree: status, raw box and offset box (per-edge correction fitted on the gold pilot against the gold boxes),
in crop px, global z-px and lon/lat. usage: python -m experiments.relabel.merge_batch BATCH_DIR"""
import csv, glob, json, math, sys, collections
B = sys.argv[1]
# median (drawn - gold) per edge on the gold pilot, crop px: x0, y0, x1, y1
import os
OFFSET = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'offset.json')))['edge_median_px']
items = {i['id']: i for i in json.load(open(B + '/items.json'))['items']}
rows = {}
for f in sorted(glob.glob(B + '/out/part_*.csv')):
    for r in csv.reader(open(f)):
        r = [c.strip() for c in r]
        if r and r[0] in items: rows[r[0]] = r   # last wins
missing = [k for k in items if k not in rows]
def lonlat(gx, gy, z):
    n = 256 * 2 ** z
    return gx / n * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * gy / n))))
feats, out = [], []
for k, it in items.items():
    if k not in rows: continue
    r = rows[k]; st = r[1]
    rec = {'id': k, 'osm_id': it['osm_id'], 'status': st, 'm_per_px': it['m_per_px']}
    if st == 'box':
        x0, y0, x1, y1 = map(float, r[2:6]); x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ox0, oy0, ox1, oy1 = x0 - OFFSET[0], y0 - OFFSET[1], x1 - OFFSET[2], y1 - OFFSET[3]
        if ox1 - ox0 < 4: ox0, ox1 = cx - 2, cx + 2   # keep at least 4 px (0.6 m)
        if oy1 - oy0 < 4: oy0, oy1 = cy - 2, cy + 2
        for tag, b in (('raw', (x0, y0, x1, y1)), ('offset', (ox0, oy0, ox1, oy1))):
            g = (b[0] + it['crop_x0'], b[1] + it['crop_y0'], b[2] + it['crop_x0'], b[3] + it['crop_y0'])
            (lo0, la0), (lo1, la1) = lonlat(g[0], g[3], it['z']), lonlat(g[2], g[1], it['z'])
            rec[tag] = [round(v, 2) for v in b]; rec[tag + '_lonlat'] = [lo0, la0, lo1, la1]
            rec[tag + '_diam_m'] = round(((b[2] - b[0]) + (b[3] - b[1])) / 2 * it['m_per_px'], 2)
        lo0, la0, lo1, la1 = rec['offset_lonlat']
        feats.append({'type': 'Feature', 'properties': {'osm_id': it['osm_id'], 'id': k, 'status': st,
                      'diam_m': rec['offset_diam_m'], 'raw_diam_m': rec['raw_diam_m'], 'raw_lonlat': rec['raw_lonlat']},
                      'geometry': {'type': 'Polygon', 'coordinates': [[[lo0, la0], [lo1, la0], [lo1, la1], [lo0, la1], [lo0, la0]]]}})
    else:
        feats.append({'type': 'Feature', 'properties': {'osm_id': it['osm_id'], 'id': k, 'status': st},
                      'geometry': {'type': 'Point', 'coordinates': list(lonlat(it['point'][0] + it['crop_x0'], it['point'][1] + it['crop_y0'], it['z']))}})
    out.append(rec)
json.dump({'type': 'FeatureCollection', 'features': feats}, open(B + '/relabel_boxes.geojson', 'w'))
json.dump(out, open(B + '/relabel_boxes.json', 'w'))
c = collections.Counter(r['status'] for r in out)
d = sorted(r['offset_diam_m'] for r in out if r['status'] == 'box'); dr = sorted(r['raw_diam_m'] for r in out if r['status'] == 'box')
print(f'{len(out)} of {len(items)} trees, missing {len(missing)} {missing[:10]}; {dict(c)}')
print(f'diameter median raw {dr[len(dr)//2]:.2f} m, offset {d[len(d)//2]:.2f} m')
