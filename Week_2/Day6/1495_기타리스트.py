
n,s,m=map(int,input().split())
arr=list(map(int,input().split()))
arr.insert(0,0)
dp=[[-1 for _ in range(102)]for _ in range(1002)]
def recur(now,idx):
    if now>m or now<0:
        return -2
    if idx==n:
        return now
    if dp[now][idx]!=-1:
        return dp[now][idx]
    dp[now][idx]=max(recur(now+arr[idx+1],idx+1), recur(now-arr[idx+1],idx+1))
    return dp[now][idx]


ans=recur(s,0)
if ans==-1:
    print(-1)
else :
    print(ans)