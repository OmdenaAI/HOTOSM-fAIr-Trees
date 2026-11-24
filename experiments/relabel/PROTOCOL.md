# Crown box protocol v2 (tight boxes)

Each sheet = up to 4 tiles (2x2: top-left, top-right, bottom-left, bottom-right), item id in cyan at each
tile's bottom-right; ids per sheet are listed in ids_by_sheet.txt. A tile is a 136x136 px aerial crop
(~0.148 m/px, 20 m wide), shown 4x. Tick labels on top/left edges are CROP pixels (ticks every 10, labels every
20, faint dotted grid every 10). Red circle = the tree to box. Blue X marks = OTHER mapped trees: they belong to
neighbouring crowns, so the box normally must not swallow a blue X unless it clearly sits on the same crown.

Box rules (the previous round drew boxes ~2-4 px too large on every side - fix that):
- The box edge goes ON the outermost green foliage pixels of this crown. Not shadow, not ground, no margin.
- Read each of the 4 edges separately against the dotted grid: find the grid line nearest the foliage edge,
  then count pixels (1 grid cell = 10 px = 40 display px). Write integers.
- When unsure where the edge is, choose the TIGHTER value. A box that is slightly small is better than one
  with a margin.
- Crowns touching neighbours: cut at the visible boundary (texture/shade change, or midway toward the blue X).
- A small crown is small: 1.5-3 m crowns are only 10-20 px wide.

Statuses: box | not_a_tree (dot on roof/ground/road/grass, no crown) | cant_tell (too ambiguous) |
merged (continuous canopy, no crown edge can be placed even roughly). Prefer box when an edge can be placed.

Output CSV, no header, one line per item: id,status,x0,y0,x1,y1 (coords empty for non-box). Every item in
your sheet range exactly once. Append to the file every few sheets.
