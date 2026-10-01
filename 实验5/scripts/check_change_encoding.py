import numpy as np, rasterio
from pathlib import Path

exp5 = Path(r'E:\人参种在哪\实验5')
inp = exp5 / '00_input_from_experiment4'

def load(p):
    with rasterio.open(p) as src:
        d = src.read(1).astype(np.float32); d[d==src.nodata]=np.nan
    return d

cur = load(inp/'current_baseline/current_binary_suitability.tif')
scen = 'ACCESS-CM2/ssp126/2041-2060'
fut = load(inp/f'binary_predictions/{scen}/binary_suitability.tif')
cha = load(inp/f'change_class_maps/{scen}/change_class.tif')

v = ~np.isnan(cur) & ~np.isnan(fut)
cur_b = (cur[v]==1); fut_b = (fut[v]==1); ch = cha[v]

print('Cross-tabulation: rows=(current,future), cols=change class value')
header = '(cur,fut)      class0        class1        class2      class3       nan'
print(header)
for cb, fb, lab in [(False,False,'(0,0) unsuit'),(False,True,'(0,1) gain  '),(True,False,'(1,0) loss  '),(True,True,'(1,1) stable')]:
    m = (cur_b==cb)&(fut_b==fb)
    vals = ch[m]
    u, c = np.unique(vals[~np.isnan(vals)], return_counts=True)
    d = dict(zip(u.astype(int).tolist(), c.tolist()))
    print('%-14s %11d %12d %12d %10d %10d' % (lab, d.get(0,0), d.get(1,0), d.get(2,0), d.get(3,0), int(np.isnan(vals).sum())))
print()
print('Total valid (cur&fut):', int(v.sum()), ' change nan on those:', int(np.isnan(cha[v]).sum()))
print('Change map total non-nan:', int((~np.isnan(cha)).sum()), '(full grid =', cha.size, ')')
