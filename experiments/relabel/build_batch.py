"""Pick whole HOT scoring units of one split (seeded order) up to N trees outside groups, write a 20 m crop per
tree + items.json (point, neighbour points) + 4-up gridded sheets for hand annotation.
usage: python -m experiments.relabel.build_batch HOT_RAW OUT_DIR SPLIT N_TREES SEED ID_PREFIX [DONE_ITEMS_JSON ...]
Units listed in any DONE_ITEMS_JSON are skipped."""
import io, json, math, sys
import numpy as np, pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageEnhance
from experiments.label_study.hot import read_hot, split_units
from experiments.gold_crowns import crop_window, window_tiles, assemble, TILE_PX
from fair_trees.preprocess import parse_oam_name
RAW, OUT, SPLIT, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); SEED = int(sys.argv[5]); PREFIX = sys.argv[6]
DONE = {u for f in sys.argv[7:] for u in json.load(open(f))['units']}
d = read_hot(f'{RAW}/stac/project-50661/labels.parquet', f'{RAW}/stac/project-50661/tiles.parquet')
hs = {'val': 'validation'}.get(SPLIT, SPLIT)
trees = d.trees.copy(); ing = np.zeros(len(trees), bool); ing[d.groups.sindex.query(trees.geometry, predicate='within')[0]] = True
trees['in_group'] = ing
tiles = d.tiles[d.tiles['split'] == hs]
units = split_units(tiles, hs)
rng = np.random.default_rng(SEED); order = rng.permutation(len(units))
tile_xy = {t: (x, y) for t, x, y in zip(d.tiles['tile_id'], d.tiles['x'], d.tiles['y'])}
z = int(d.tiles['z'].iloc[0]); world = TILE_PX * 2**z
gx = (trees.geometry.x + 180) / 360 * world
gy = (1 - np.arcsinh(np.tan(np.radians(trees.geometry.y))) / math.pi) / 2 * world
trees['tx'], trees['ty'] = (gx // TILE_PX).astype(int), (gy // TILE_PX).astype(int)
trees['gx'], trees['gy'] = gx, gy
own = trees[(trees['split'] == hs) & ~trees['in_group']]
by_tile = own.groupby(['tx', 'ty']).groups
chosen, picked_units = [], []
for u in order:
    uid, tids = units[u]
    if uid in DONE: continue
    idx = [i for t in tids for i in by_tile.get(tile_xy[t], [])]
    if not idx: continue
    chosen += idx; picked_units.append(uid)
    if len(chosen) >= N: break
sel = own.loc[chosen].sort_values('osm_id')
print(len(picked_units), 'units', len(sel), 'trees')
wins = [crop_window(p.x, p.y, z) for p in sel.geometry]
need = {t for w in wins for t in window_tiles(w)}
img = {}
import glob
for f in sorted(glob.glob(f'{RAW}/data/*.parquet')):
    for b in pq.ParquetFile(f).iter_batches(columns=['tile_id', 'image'], batch_size=256):
        for row in b.to_pylist():
            xyz = parse_oam_name(row['tile_id'] + ('' if row['tile_id'].endswith('.png') else '.png'))
            if xyz and (xyz[0], xyz[1]) in need: img[(xyz[0], xyz[1])] = row['image']['bytes']
cache = {}
def rgb(x, y):
    if (x, y) not in img: return None
    if (x, y) not in cache: cache[(x, y)] = np.asarray(Image.open(io.BytesIO(img[(x, y)])).convert('RGB'))
    return cache[(x, y)]
import os; os.makedirs(f'{OUT}/crops', exist_ok=True); os.makedirs(f'{OUT}/sheets', exist_ok=True)
allx, ally = trees['gx'].to_numpy(), trees['gy'].to_numpy()
items = []
for k, (row, w) in enumerate(zip(sel.itertuples(), wins)):
    iid = f'{PREFIX}_{k:05d}'
    a, grey = assemble(w, rgb); Image.fromarray(a).save(f'{OUT}/crops/{iid}.png')
    m = (allx >= w.x0) & (allx < w.x0 + w.size) & (ally >= w.y0) & (ally < w.y0 + w.size)
    nb = [[round(float(x - w.x0), 2), round(float(y - w.y0), 2)] for x, y in zip(allx[m], ally[m])
          if abs(x - w.x0 - w.point[0]) + abs(y - w.y0 - w.point[1]) > 0.5]
    items.append({'id': iid, 'osm_id': int(row.osm_id), 'file': f'crops/{iid}.png', 'size_px': w.size, 'm_per_px': w.m_per_px,
                  'point': [round(v, 2) for v in w.point], 'neighbours': nb, 'crop_x0': w.x0, 'crop_y0': w.y0, 'z': w.z,
                  'grey_share': round(grey, 4)})
json.dump({'set': f'relabel_{PREFIX}', 'split': SPLIT, 'crop_m': 20.0, 'units': picked_units, 'items': items}, open(f'{OUT}/items.json', 'w'))
U = 4
def tile(it):
    im = Image.open(f"{OUT}/{it['file']}").convert('RGB'); n = it['size_px']
    im = ImageEnhance.Brightness(im).enhance(1.2).resize((n * U, n * U), Image.NEAREST); d_ = ImageDraw.Draw(im)
    for v in range(0, n + 1, 10):
        c = (255, 255, 0) if v % 50 == 0 else (255, 255, 255)
        for a_, b_ in (((v*U, 0), (v*U, 6)), ((0, v*U), (6, v*U)), ((v*U, n*U-6), (v*U, n*U)), ((n*U-6, v*U), (n*U, v*U))): d_.line([a_, b_], fill=c)
        if v % 20 == 0: d_.text((v*U+1, 8), str(v), fill=c); d_.text((8, v*U+1), str(v), fill=c)
    for v in range(10, n, 10):
        for t in range(0, n*U, 8): d_.point((v*U, t), fill=(200, 200, 200)); d_.point((t, v*U), fill=(200, 200, 200))
    for x, y in it['neighbours']: d_.line([(x*U-4, y*U-4), (x*U+4, y*U+4)], fill=(60, 140, 255), width=2); d_.line([(x*U-4, y*U+4), (x*U+4, y*U-4)], fill=(60, 140, 255), width=2)
    px, py = it['point']; d_.ellipse([px*U-4, py*U-4, px*U+4, py*U+4], outline=(255, 0, 80), width=2)
    d_.text((n*U-80, n*U-20), it['id'], fill=(0, 255, 255)); return im
for s in range(0, len(items), 4):
    grp = items[s:s+4]; n = 136 * U; sh = Image.new('RGB', (n*2+10, n*2+10), (40, 40, 40))
    for j, it in enumerate(grp): sh.paste(tile(it), ((j % 2) * (n + 10), (j // 2) * (n + 10)))
    sh.save(f'{OUT}/sheets/s{s//4:04d}.png')
with open(f'{OUT}/ids_by_sheet.txt', 'w') as f:
    for s in range(0, len(items), 4): f.write(f's{s//4:04d}: ' + ' '.join(i['id'] for i in items[s:s+4]) + '\n')
print('items', len(items), 'sheets', math.ceil(len(items)/4), 'missing tiles', len(need - set(img)))
