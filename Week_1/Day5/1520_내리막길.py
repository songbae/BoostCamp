import sys
sys.setrecursionlimit(5000)
n,m=map(int,input().split())

def rec(x,y,arr,dp):
    if x==0 and y==0 :
        return 1
    if dp[x][y]!=-1:
        return dp[x][y]
    dp[x][y]=0
    for i in range(4):
        nx= x-(int('1012'[i])-1)
        ny= y-(int('2101'[i])-1)
        if ny < 0 or nx < 0 or ny >= m or nx >= n:
            continue
        if arr[nx][ny]<=arr[x][y]:
           continue
        dp[x][y]+=rec(nx,ny,arr,dp)
    return dp[x][y]

arr=[list(map(int,input().split()))for _ in range(n)]
dp =[[-1 for _ in range(m)]for _ in range(n)]
print(rec(n-1,m-1,arr,dp))
