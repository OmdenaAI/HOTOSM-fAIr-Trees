"""Score pilot boxes drawn with the protocol against the gold boxes: diameter bias, IoU and per-edge medians
(the medians are what offset.json holds). usage: python -m experiments.relabel.score_pilot GOLD_ITEMS_JSON
GOLD_BOXES_JSON FIRST_ROUND_JSON PILOT_CSV"""
import csv, json, sys, statistics as st
items={i['id']:i for i in json.load(open(sys.argv[1]))['items']}
O=json.load(open(sys.argv[2])); C1=json.load(open(sys.argv[3]))
new={r[0]:r for r in csv.reader(open(sys.argv[4])) if r}
def iou(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0])); iy=max(0,min(a[3],b[3])-max(a[1],b[1])); i=ix*iy
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)
for name,get in (('v1 (first round)',lambda k:[C1[k][e] for e in ('x0','y0','x1','y1')] if C1[k]['status']=='box' else None),
                 ('v2 (tight protocol)',lambda k:[float(v) for v in new[k][2:6]] if k in new and new[k][1]=='box' else None)):
    dd=[];ious=[];edges=[[],[],[],[]]
    for k in new:
        b=get(k); o=[O[k][e] for e in ('x0','y0','x1','y1')]
        if not b: continue
        m=items[k]['m_per_px']; dd.append(((b[2]-b[0]+b[3]-b[1])/2-(o[2]-o[0]+o[3]-o[1])/2)*m); ious.append(iou(b,o))
        for j in range(4): edges[j].append(b[j]-o[j])
    print(f"{name}: n={len(dd)} diam bias {st.mean(dd):+.2f} m  MAE {st.mean(map(abs,dd)):.2f}  IoU mean {st.mean(ious):.3f}  >=0.5 {sum(i>=.5 for i in ious)/len(ious):.0%}  edge medians x0,y0,x1,y1 {[round(st.median(e),1) for e in edges]}")
print('v2 statuses', {s:sum(1 for r in new.values() if r[1]==s) for s in ('box','not_a_tree','cant_tell','merged')})
