import collections
import sys
dx=[-1,0,1,0,0,0]
dy=[0,-1,0,1,0,0]
dz=[0,0,0,0,1,-1]
q=collections.deque()
def sol(n,m,T,num):# T에는 익은 토마토들 
    for i in range(h):
        for j in range(n):
            for k in range(m):
                if arr[i][j][k]==1:
                    q.append((i,j,k,0))
    while q:
        z,x,y,day=q.popleft()
        for i in range(6):
            nx= x+dx[i]
            ny= y+dy[i]
            nz= z+dz[i]
            if nx>=n or ny>=m or nz>=h or nx<0 or nz<0 or ny<0:
                continue
            if arr[nz][nx][ny]!=0:
                continue
            arr[nz][nx][ny]=1
            q.append((nz,nx,ny,day+1))
            num-=1
            if num==0:
                return day+1 
    return -1


m,n,h=map(int,input().split())
arr=[]
for _ in range(h):
    arr.append([list(map(int,sys.stdin.readline().split()))for _ in range(n)])
zero=0
for i in range(h):
    for j in range(n):
        for k in range(m):
            if arr[i][j][k]==0:
                zero+=1

if zero==0:
    print(0)
else :
    print(sol(n,m,arr,zero))