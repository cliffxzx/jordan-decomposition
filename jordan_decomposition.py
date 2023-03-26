import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import device_put, pmap
from jax import lax
from jax import random
from jax import tree_util

# 使用jit編譯，使程式碼可以運行在GPU上
@jit
def jordan_block(eigenvalue, size):
    if size == 1:
        return jnp.array([eigenvalue])
    block = jnp.eye(size) * eigenvalue
    block = block + jnp.eye(size, k=1)
    return block

@jit
def jordan_block_diag(eigenvalues, sizes):
    blocks = []
    for i, (eigenvalue, size) in enumerate(zip(eigenvalues, sizes)):
        block = jordan_block(eigenvalue, size)
        blocks.append(block)
    return jnp.block_diag(*blocks)

# 使用pmap在GPU上進行平行化計算
@pmap
def jordan_decomposition(A):
    # 計算矩陣A的特徵值和特徵向量
    eigenvalues, eigenvectors = jnp.linalg.eig(A)

    # 將特徵值從大到小排序
    idx = jnp.argsort(-jnp.abs(eigenvalues))
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    n = A.shape[0]
    D = jnp.zeros((n, n))
    P = jnp.zeros((n, n))

    i = 0
    while i < n:
        if i == n-1 or jnp.abs(eigenvalues[i]-eigenvalues[i+1]) > 1e-10:
            # 如果第i個特徵值不等於第(i+1)個特徵值，則它是一個簡單的特徵值
            D = D.at[i, i].set(eigenvalues[i])
            P = P.at[:,i].set(eigenvectors[:,i])
            i += 1
        else:
            # 如果第i個特徵值等於第(i+1)個特徵值，則它們是一對特徵值
            j = i + 1
            while j < n and jnp.abs(eigenvalues[i]-eigenvalues[j]) < 1e-10:
                j += 1
            m = j - i # 該特徵值對應的Jordan block的個數
            v = eigenvectors[:,i:j] # 該特徵值對應的特徵向量
            sizes = [k+1 for k in range(m)]
            sizes[-1] = n - i - sum(sizes[:-1])
            # 計算該特徵值對應的Jordan block
            blocks = vmap(jordan_block_diag, in_axes=(0, 0))(eigenvalues[i:i+m], sizes)
            # 將Jordan block放入D和P矩陣中
            D = D.at[i:i+m, i:i+m].set(jnp.diag(jnp.ones(m)*eigenvalues[i]))
            P = P.at[:,i:i+m].set(v)
            for k in range(m):
                D = D.at[i+k+1, i+k].set(1.)
                if k < m-1:
                    P = P.at[:,i+k+1].set(v[:,k+1])

                i += m

                return P, D

if __name__ == '__main__':
    print(jordan_decomposition([[1, 2, 1], [0, 2, 0], [0, 1, 1]]))

    # 定義一個隨機矩陣
    key = random.PRNGKey(0)
    A = random.normal(key, (1000, 1000))

    # 將矩陣A放入GPU中計算Jordan decomposition
    A_gpu = device_put(A)
    P, D = jordan_decomposition(A_gpu)

    # 將計算結果從GPU中移回CPU並顯示
    P_cpu = P.get()
    D_cpu = D.get()
    print("P:", P_cpu)
    print("D:", D_cpu)