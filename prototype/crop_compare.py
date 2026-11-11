"""Side-by-side crops of raw 1080x1920 BGRA renders.

    crop_compare.py dir f1,f2,... x0 y0 w h out.png name1 name2 ...
reads dir/<name>.bgra; one row per frame, one column per render.
"""
import sys, numpy as np, cv2
d=sys.argv[1]; frames=[int(f) for f in sys.argv[2].split(',')]
x0,y0,w,h=map(int,sys.argv[3:7]); out=sys.argv[7]; names=sys.argv[8:]
W,H=1080,1920; fb=W*H*4
rows=[]
for f in frames:
    tiles=[]
    for n in names:
        m=np.memmap(f"{d}/{n}.bgra",np.uint8,'r',offset=f*fb,shape=(H,W,4))
        t=np.ascontiguousarray(m[y0:y0+h,x0:x0+w,:3])
        cv2.putText(t,f"{n} {f}",(8,28),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)
        tiles.append(t); tiles.append(np.zeros((h,4,3),np.uint8))
    rows.append(np.hstack(tiles[:-1]))
cv2.imwrite(out,np.vstack(rows))
