# # 노드 생성과 삽입

# class Node:
#     def __init__(self,data):
#         self.left=None
#         self.right=None
#         self.data=data
# class BST(Node):
#     def insert(self,data):
#         self.root=self._insert_value(self.rott,data)
#         return self.root is not None

#     def _insert_value(self,node,data):
#         if node is None:
#             node=Node(data)
#         else:
#             if data<=node.data:
#                 node.left=self._insert_value(node.left,data)
#             else:
#                 node.right=self._insert_value(node.right,data)
#         return node
# # 노드 탐색

# def find(self,key):
#     return self._find_value(self.root,key)
# def _find_value(self,root,key):
#     if roor is None or root.data ==key:
#         return root is not None
#     elif key<root.data:
#         return self._find_value(root.left,key)
#     else:
#         return self._find_value(root.right,key)
    
# def delete(self,key):
#     self.root,deleted =self._delete_value(self.root,key)
#     return deleted

import sys
sys.getrecursionlimit(5000)
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

