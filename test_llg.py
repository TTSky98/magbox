from magbox import llg,spin
sp=spin([0.1,0.1],[0,0]," ")
sf=llg(sp)

tmp=sf.llg_kernal(sp)
print(tmp)