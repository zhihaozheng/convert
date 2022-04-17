
def ls(t1,t2=4): return list(range(t1,t1+t2))

def get_ls(initials,step=4): return [item for j in [ls(i,step) for i in initials] for item in j]

def get_tile_size(n):
    for i in range(5):
        j = np.ceil(n/1000 + 0.5)+ i
        if j % 4 == 0:
            return int(j*1000)
