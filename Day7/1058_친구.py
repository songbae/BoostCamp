# from collections import defaultdict
# n= int(input())
# board=defaultdict(list)
# for i in range(n):
#     row=list(input())

#     for j in range(n):
#         if row[j]=='Y':
#             board[i].append(j)

# def add(root, node, friend):
#     for f in board[node]:
#         if f==root: continue
#         if f not in friend: friend.add(f)

# ans=0
# for i in range(n):
#     friend=set()
#     for f in board[i]:
#         friend.add(f)
#         add(i,f,friend)
#     ans=max(ans,len(friend))
# print(ans)

n = int(input())
t=[input()for i in range(n)]
f=[[0 for _ in range(n)]for i in range(n)]
for i in range(n):
    for j in range(n):
        if t[i][j]=='Y':
            f[i][j]+=j
print(f)
m=0
for i in f:
    s=[x for x in i]
    for j in i:
        s+=f[j]
    m=max(m,len(set(s))-1)
print(m)