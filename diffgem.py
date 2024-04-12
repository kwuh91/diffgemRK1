import numpy as np
import sympy as sp

from typing import TypeAlias

from extra import get_super, get_sub, getMatrixInverse, write_matrix3D

Matrix: TypeAlias = list[list[int]]
Vector: TypeAlias = list[int]

"""
ТИ. Задание 11. 
Нахождение компонент скалярного произведения тензоров, 
заданных своими компонентами в ортонормированном базисе, 
в некотором другом базисе. 
"""

def componentOfTheScalarWork(A: Matrix, 
                             B: Matrix,
                             Q: Matrix) -> None:
    print(f'\n{"-"*100}')
    print(f'Нахождение компонент скалярного произведения тензоров,')
    print(f'заданных своими компонентами в ортонормированном базисе,')
    print(f'в некотором другом базисе.')
    print(f'{"-"*100}',end='\n\n')

    temp: int = 0
    dim:  int = len(Q)

    g:      Matrix = [[0 for _ in range(dim)] for _ in range(dim)]
    vecC:   Matrix = [[0 for _ in range(dim)] for _ in range(dim)]
    Ccontr: Matrix = [[0 for _ in range(dim)] for _ in range(dim)]
    Ccovar: Matrix = [[0 for _ in range(dim)] for _ in range(dim)]
    vecP:   Matrix

    print(f'g_(ij) = e_i ⋅ e_j = Q^s_i * Q^p_j * δ_(sp)', end='\n\n')

    i: int = 1
    j: int = 1

    for i in range(1, dim + 1):
        for j in range(1, dim + 1):

            print(f'g_({i}{j}) = ', end='')

            s: int = 1
            p: int = 1

            temp = 0
            while (s < dim + 1 and p < dim + 1):
                temp += Q[s - 1][i - 1] * \
                        Q[p - 1][j - 1]

                print(f'{Q[s - 1][i - 1]} * {Q[p - 1][j - 1]}', end='')

                if (s < dim - 1 + 1 or p < dim - 1 + 1):
                    print(f' + ', end='')
                else:
                    print(f' = ', end='')

                s += 1
                p += 1
            
            print(f'{temp}')
            g[i - 1][j - 1] = temp

    print('\ng_(ij):\n',np.matrix(g))
    print(f'\n(vec)P^i_j = (vec)(Q^i_j)^-1')

    vecP = np.linalg.inv(Q)

    print('\n(vec)P^i_j:\n',vecP, end='\n\n')
    print(f'(vec)C^(ik) = (vec)T^(ij) * (vec)B^k_j', end='\n\n')

    i: int 
    for i in range(1, dim + 1):
        k: int
        for k in range(1, dim + 1):
            print(f'(vec)C^({i}{k}) = ', end='')

            temp = 0
            j: int
            for j in range(1, dim + 1):
                temp += A[i - 1][j - 1] * \
                        B[k - 1][j - 1]

                print(f'{A[i - 1][j - 1]} * {B[k - 1][j - 1]}',end='') 

                if (j < dim - 1 + 1):
                    print(f' + ', end='')
                else:
                    print(f' = ', end='')

            print(f'{temp}')
            vecC[i - 1][k - 1] = temp

    print('\n(vec)C^(ik):\n',np.matrix(vecC), end='\n\n')
    print(f'C^(ml) = (vec)C^(ik) * (vec)P^m_i * (vec)P^l_k', end='\n\n')

    m: int 
    for m in range(1, dim + 1):
        l: int
        for l in range(1, dim + 1):
            print(f'C^({m}{l}) = ', end='')
            temp = 0

            i: int
            for i in range(1, dim + 1):
                k: int
                for k in range(1, dim + 1):
                    temp += vecC[i - 1][k - 1] * \
                            vecP[m - 1][i - 1] * \
                            vecP[l - 1][k - 1]

                    print(f'{vecC[i - 1][k - 1]} * {vecP[m - 1][i - 1]} * {vecP[l - 1][k - 1]}',end='') 

                    if (i < dim - 1 + 1 or k < dim - 1 + 1):
                        print(f' + ', end='')
                    else:
                        print(f' = ', end='')

            print(f'{temp}')
            Ccontr[m - 1][l - 1] = temp

    print('\nC^(ml):\n',np.matrix(Ccontr), end='\n\n')
    print(f'C_(st) = C^(ml) * g_(ms) * g_(lt)', end='\n\n')

    s: int 
    for s in range(1, dim + 1):
        t: int
        for t in range(1, dim + 1):
            print(f'C_({s}{t}) = ', end='')
            temp = 0

            m: int
            for m in range(1, dim + 1):
                l: int
                for l in range(1, dim + 1):
                    temp += Ccontr[m - 1][l - 1] * \
                                 g[m - 1][s - 1] * \
                                 g[l - 1][t - 1]

                    print(f'{Ccontr[m - 1][l - 1]} * {g[m - 1][s - 1]} * {g[l - 1][t - 1]}',end='') 

                    if (m < dim -1 + 1 or l < dim - 1 + 1):
                        print(f' + ', end='')
                    else:
                        print(f' = ', end='')

            print(f'{temp}')
            Ccovar[s - 1][t - 1] = temp

    print('\nC_(st):\n',np.matrix(Ccovar), end='\n')

"""
ТИ. Задание 12. 
Нахождение компонент скалярного произведения тензора на вектор, 
заданных своими компонентами в ортонормированном базисе, 
в некотором другом базисе.
"""

def findingComponents(T: Matrix, a: Vector, Q: Matrix) -> None:
    print(f'\n{"-"*100}')
    print(f'Нахождение компонент скалярного произведения тензора на вектор,')
    print(f'заданных своими компонентами в ортонормированном базисе,')
    print(f'в некотором другом базисе.')
    print(f'{"-"*100}',end='\n\n')

    dim: int = len(Q)
    vecb:   Vector = [0 for _ in range (dim)]
    bcontr: Vector = [0 for _ in range (dim)]
    bcovar: Vector = [0 for _ in range (dim)]
    vecc:   Vector = [0 for _ in range (dim)]
    ccontr: Vector = [0 for _ in range (dim)]
    ccovar: Vector = [0 for _ in range (dim)]
    vecP:   Matrix = [[0 for _ in range(dim)] for _ in range(dim)]
    g:      Matrix = [[0 for _ in range(dim)] for _ in range(dim)]

    # # # # # # b # # # # # #

    print(f'(vec)b^i = (vec)T^(ij) * (vec)a_j')

    i: int
    for i in range(1, dim + 1):
        print(f'(vec)b^{i} = ', end='')

        temp = 0
        j: int
        for j in range(1, dim + 1):
            temp += T[i - 1][j - 1] * \
                    a[j - 1]

            print(f'{T[i - 1][j - 1]} * {a[j - 1]}',end='') 

            if (j < dim -1 + 1):
                print(f' + ', end='')
            else:
                print(f' = ', end='')

        vecb[i - 1] = temp
        print(f'{temp}')
    
    print('\n(vec)b^i:\n',np.matrix(vecb), end='\n\n')

    print(f'\n(vec)P^i_j = (vec)(Q^i_j)^-1')

    vecP = np.linalg.inv(Q)

    print('\n(vec)P^i_j:\n',vecP, end='\n\n')

    print(f'b^j = (vec)b^i * (vec)P^j_i')

    j: int
    for j in range(1, dim + 1):
        print(f'b^{j} = ', end='')

        temp = 0
        i: int
        for i in range(1, dim + 1):
            temp += vecb[i - 1] * \
                    vecP[j - 1][i - 1]

            print(f'{vecb[i - 1]} * {vecP[j - 1][i - 1]}',end='') 

            if (i < dim - 1 + 1):
                print(f' + ', end='')
            else:
                print(f' = ', end='')

        bcontr[j - 1] = temp
        print(f'{temp}')
    
    print('\nb^j:\n',np.matrix(bcontr), end='\n\n')

    print(f'g_(ij) = e_i ⋅ e_j = Q^s_i * Q^p_j * δ_(sp)', end='\n\n')

    i: int = 1
    j: int = 1

    for i in range(1, dim + 1):
        for j in range(1, dim + 1):

            print(f'g_({i}{j}) = ', end='')

            s: int = 1
            p: int = 1

            temp = 0
            while (s < dim + 1 and p < dim + 1):
                temp += Q[s - 1][i - 1] * \
                        Q[p - 1][j - 1]

                print(f'{Q[s - 1][i - 1]} * {Q[p - 1][j - 1]}', end='')

                if (s < dim - 1 + 1 or p < dim - 1 + 1):
                    print(f' + ', end='')
                else:
                    print(f' = ', end='')

                s += 1
                p += 1
            
            print(f'{temp}')
            g[i - 1][j - 1] = temp

    print('\ng_(ij):\n',np.matrix(g), end='\n\n')

    print(f'b_i = g_(ij) * b^j')

    i: int
    for i in range(1, dim + 1):
        print(f'b_{i} = ', end='')

        temp = 0
        j: int
        for j in range(1, dim + 1):
            temp += g[i - 1][j - 1] * \
                    bcontr[j - 1]

            print(f'{g[i - 1][j - 1]} * {bcontr[j - 1]}',end='') 

            if (j < dim - 1 + 1):
                print(f' + ', end='')
            else:
                print(f' = ', end='')

        bcovar[i - 1] = temp
        print(f'{temp}')
    
    print('\nb_i:\n',np.matrix(bcovar), end='\n\n')

    # # # # # # c # # # # # #

    print(f'(vec)c^j = (vec)a_i * (vec)T^(ij)')

    j: int
    for j in range(1, dim + 1):
        print(f'(vec)c^{j} = ', end='')

        temp = 0
        i: int
        for i in range(1, dim + 1):
            temp += a[i - 1] * \
                    T[i - 1][j - 1]

            print(f'{a[i - 1]} * {T[i - 1][j - 1]}',end='') 

            if (i < dim -1 + 1):
                print(f' + ', end='')
            else:
                print(f' = ', end='')

        vecc[j - 1] = temp
        print(f'{temp}')
    
    print('\n(vec)c^j:\n',np.matrix(vecc), end='\n\n')

    print(f'c^j = (vec)c^i * (vec)P^j_i')

    j: int
    for j in range(1, dim + 1):
        print(f'c^{j} = ', end='')

        temp = 0
        i: int
        for i in range(1, dim + 1):
            temp += vecc[i - 1] * \
                    vecP[j - 1][i - 1]

            print(f'{vecc[i - 1]} * {vecP[j - 1][i - 1]}',end='') 

            if (i < dim - 1 + 1):
                print(f' + ', end='')
            else:
                print(f' = ', end='')

        ccontr[j - 1] = temp
        print(f'{temp}')
    
    print('\nc^j:\n',np.matrix(ccontr), end='\n\n')

    print(f'c_i = g_(ij) * c^j')

    i: int
    for i in range(1, dim + 1):
        print(f'c_{i} = ', end='')

        temp = 0
        j: int
        for j in range(1, dim + 1):
            temp += g[i - 1][j - 1] * \
                    ccontr[j - 1]

            print(f'{g[i - 1][j - 1]} * {ccontr[j - 1]}',end='') 

            if (j < dim - 1 + 1):
                print(f' + ', end='')
            else:
                print(f' = ', end='')

        ccovar[i - 1] = temp
        print(f'{temp}')
    
    print('\nc_j:\n',np.matrix(ccovar), end='\n')

"""
ТИ. Задание 13. 
Нахождение полной свертки тензоров, 
заданных в ортонормированном базисе. 
"""

def fullConvolution(A: Matrix, 
                    B: Matrix) -> None:
    print(f'\n{"-"*100}')
    print(f'Нахождение полной свертки тензоров,')
    print(f'заданных в ортонормированном базисе.')
    print(f'{"-"*100}',end='\n\n')

    res: int = 0
    dim: int = len(A)

    print(f'T ⋅ ⋅ B = (vec)T^(ij) * (vec)B_(ji) = ', end='')

    i: int 
    for i in range(1, dim + 1):
        j: int
        for j in range(1, dim + 1):
            res += A[i - 1][j - 1] * \
                   B[j - 1][i - 1]

            print(f'{A[i - 1][j - 1]} * {B[i - 1][j - 1]}',end='') 

            if (i < dim - 1 + 1 or j < dim - 1 + 1):
                print(f' + ', end='')
            else:
                print(f' = ', end='')
    
    print(f'{res}')

"""
ТИ. Задание 17. 
Нахождение векторного произведения тензора на вектор, 
заданных в декартовом ортонормированном некотором базисе.
"""

def vectorProductOfTheTensorOnTheVector(a: Vector, T: Matrix) -> None:
    print(f'\n{"-"*100}')
    print(f'Нахождение векторного произведения тензора на вектор,')
    print(f'заданных в декартовом ортонормированном некотором базисе.')
    print(f'{"-"*100}',end='\n\n')

    dim:  int = len(T)
    temp: int = 0
    A:    Matrix = [[0 for _ in range(dim)] for _ in range(dim)]

    def LeviChivita(nums: str) -> int:
        i: int
        j: int
        k: int

        i, j, k = map(int, [*nums])
        matr: Matrix = [[0 for _ in range(dim)] for _ in range(dim)]

        matr[0][i - 1] = 1
        matr[1][j - 1] = 1
        matr[2][k - 1] = 1

        determinant: int = np.linalg.det(matr)

        return determinant if determinant in [1, -1] else 0 
    
    print(f'A = (vec)a^i * (vec)T^(kl) * levi-chevita(ikm) * (vec)e^m ⊗ (vec)e_l = (vec)a_m * ^l(vec)e^m ⊗ (vec)e_l', end='\n\n')
    print(f'(vec)A^l_m = (vec)a^i * (vec)T^(kl) * levi-chevita(ikm)', end='\n\n')

    l: int 
    for l in range(1, dim + 1):
        m: int
        for m in range(1, dim + 1): 
            print(f'(vec)A^{l}_{m} = (vec)a^i * (vec)T^(k{l}) * levi-chevita(ik{m}) = ',end='')

            temp = 0
            i: int 
            for i in range(1, dim + 1):
                k: int
                for k in range(1, dim + 1):
                    temp += a[i - 1]        * \
                            T[k - 1][l - 1] * \
                            LeviChivita(f"{i}{k}{m}")

                    print(f'{a[i - 1]} * {T[k - 1][l - 1]} * {LeviChivita(f"{i}{k}{m}")}',end='') 

                    if (i < dim - 1 + 1 or k < dim - 1 + 1):
                        print(f' + ', end='')
                    else:
                        print(f' = ', end='')

            A[l - 1][m - 1] = temp
            print(f'{temp}')

    print('\n(vec)a_m^l = Transposed((vec)A^l_m):\n',np.matrix(A).T, end='\n')

"""
ТИ. Задание 23. 
Дифференциальные характеристики цилиндрических координат.
"""

def differentialCharacteristics(Aⁱⱼ: Matrix):
    print(f'\n{"-"*100}')
    print(f'Дифференциальные характеристики цилиндрических координат.')
    print(f'{"-"*100}',end='\n\n')
    
    dim: int = len(Aⁱⱼ)

    X, Y, Z = sp.symbols('X¹ X² X³')

    # связь с цилиндрическими координатами
    cyl_coord = {
        '(vec)x¹' : X * sp.cos(Y),
        '(vec)x²' : X * sp.sin(Y),
        '(vec)x³' : Z,
    }

    # якобиева матрица цилиндрической системы координат
    Qⁱₖ: Matrix = [[0 for _ in range(dim)] for _ in range(dim)]

    # якобиева матрица для криволинейных координат
    Q: Matrix = [[0 for _ in range(dim)] for _ in range(dim)]

    # метрическая матрица для цилиндрической системы координат
    g: Matrix = [[0 for _ in range(dim)] for _ in range(dim)]

    # обратная метрическая матрица для цилиндрической системы координат
    g_inverse: Matrix

    # матрица
    Qⁱᵖ: Matrix = [[0 for _ in range(dim)] for _ in range(dim)]

    # матрица
    Qⁱᵐ: Matrix = [[0 for _ in range(dim)] for _ in range(dim)]

    # контейнер для символов Кристоффеля 2-го рода
    Г: list[Matrix] = [[[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)] # mij

    # контейнер для символов Кристоффеля 1-го рода
    Гᵢⱼₖ: list[Matrix] = [[[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)] # ijk

    # массив коэффициентов Ламе
    H: list[int] = [0 for _ in range(dim)]

    # Найдем якобиеву матрицу для криволинейных координат Xi по формуле
    print(f"Найдем якобиеву матрицу для криволинейных координат X{get_super('i')} по формуле:\n", end='')

    print(f"Q{get_super('i')}{get_sub('k')} = ", end='')
    print(f"∂(vec)x\{get_super('i')} / ∂X{get_super('k')} = ", end='')
    print(f"A{get_super('i')}{get_sub('j')} * ∂(vec)x\{get_super('i')} / ∂X{get_super('k')} = ", end='')
    print(f"A{get_super('i')}{get_sub('j')} * (vec)Q{get_super('j')}{get_sub('k')}", end='')

    print(f"\n\nгде ((vec)Q{get_super('j')}{get_sub('k')}) - якобиева матрица цилиндрической системы координат.", end='')

    # алгоритм для вычисления компонент якобиевой матрицы цилиндрической системы координат
    print(f"\n\nВычислим её компоненты:\n", end='')
    print(f"(vec)Q{get_super('i')}{get_sub('k')} = ∂(vec)x\{get_super('i')} / ∂X{get_super('k')}\n\n", end='')

    # Вычислим её компоненты
    for i in range(1, dim + 1):
        for k in range(1, dim + 1):

            a = cyl_coord[f'(vec)x{get_super(i)}']
            
            res = sp.diff(cyl_coord[f'(vec)x{get_super(i)}'], f'X{get_super(k)}')
            print(f"(vec)Q{get_super(i)}{get_sub(k)} = ∂(vec)x{get_super(i)} / ∂X{get_super(k)} = {res}\n", end='')
            Qⁱₖ[i - 1][k - 1] = res

    # Тогда якобиева матрица цилиндрической системы координат имеет вид
    write_matrix3D(Qⁱₖ, '(vec)Qⁱₖ', "\nТогда якобиева матрица цилиндрической системы координат имеет вид")

    # Теперь мы можем найти компоненты якобиевой матрицы для криволинейных координат: 
    print(f"\nТеперь мы можем найти компоненты якобиевой матрицы для криволинейных координат:\n", end='')
    print(f"Q{get_super('i')}{get_sub('k')} = ", end='')
    print(f"A{get_super('i')}{get_super('j')} * (vec)Q{get_super('j')}{get_sub('k')}\n\n", end='')

    # Вычислим её компоненты
    for i in range(1, dim + 1):
        for k in range(1, dim + 1):

            res = 0
            print(f"Q{get_super(i)}{get_sub(k)} = A{get_super(i)}{get_sub('j')} * (vec)Q{get_super('j')}{get_sub(k)} = ", end='')

            for j in range(1, dim + 1):
                res += Aⁱⱼ[i - 1][j - 1] * Qⁱₖ[j - 1][k - 1]
                print(f"A{get_super(i)}{get_sub(j)} * (vec)Q{get_super(j)}{get_sub(k)} ", end='')
                print(f"+ ", end='') if j < dim else print(f"= ", end='')

            res = sp.simplify(res)
            Q[i - 1][k - 1] = res
            print(f"{res}\n", end='')

    # Тогда якобиева матрица цилиндрической системы координат имеет вид
    write_matrix3D(Q, 'Qⁱₖ', "\nПолученная якобиева матрица имеет вид")
    
    # Найдем локальные векторы базиса для криволинейных координат Xi:
    print(f"\nНайдем локальные векторы базиса для криволинейных координат X{get_super('i')}:\n", end='')
    print(f"r{get_sub('k')} = ", end='')
    print(f"∂(vec)x{get_super('i')} / ∂X{get_super('k')} * (vec)e{get_sub('k')} = ", end='')
    print(f"Q{get_super('i')}{get_sub('k')} * (vec)e{get_sub('k')}\n\n", end='')

    # Матрица Qik была найдена на предыдущем шаге.
    print(f"Матрица Q{get_super('i')}{get_sub('k')} была найдена на предыдущем шаге.\n\n", end='')

    # Найдем метрическую матрицу для криволинейных координат Xi. Она, по выведенным в алгоритме формулам, совпадает с метрической матрицей для цилиндрических координат
    print(f"Найдем метрическую матрицу для криволинейных координат X{get_super('i')}. Она, по выведенным в алгоритме формулам, совпадает с метрической матрицей для цилиндрических координат:\n", end='')
    print(f"g{get_sub('i')}{get_sub('j')} = ", end='')
    print(f"r{get_sub('i')} * r{get_sub('j')} = ", end='')
    print(f"(vec)Q{get_super('s')}{get_sub('i')} * (vec)Q{get_super('p')}{get_sub('j')} * δ{get_sub('s')}{get_sub('p')} = ", end='')
    print(f"(vec)g{get_sub('i')}{get_sub('j')}\n\n", end='')

    # Найдем компоненты метрической матрицы для цилиндрической системы координат: 
    print(f"Найдем компоненты метрической матрицы для цилиндрической системы координат:\n", end='')

    # Вычислим её компоненты
    for i in range(1, dim + 1):
        for j in range(1, dim + 1):
            
            res = 0
            print(f"(vec)g{get_sub(i)}{get_sub(j)} = ", end='')
            print(f"(vec)Q{get_super('s')}{get_sub(i)} * (vec)Q{get_super('p')}{get_sub(j)} * δ{get_sub('s')}{get_sub('p')} = ", end='')
            
            mid_res_arr = []
            s: int = 1
            p: int = 1
            while (s < dim + 1 and p < dim + 1):

                mid_res = Qⁱₖ[s - 1][i - 1] * \
                          Qⁱₖ[p - 1][j - 1]
                mid_res_arr.append(mid_res)
                res += mid_res

                print(f"(vec)Q{get_super(s)}{get_sub(i)} * (vec)Q{get_super(p)}{get_sub(j)} * δ{get_sub(s)}{get_sub(p)} ", end='')
                print(f"+ ", end='') if s < dim or p < dim else print(f"= ", end='')

                s += 1
                p += 1

            for item in range(len(mid_res_arr)):
                print(f"{mid_res_arr[item]} + ", end='') if item < len(mid_res_arr) - 1 else print(f"{mid_res_arr[item]} = ", end='')
            
            res = sp.simplify(res)
            g[i - 1][j - 1] = res
            print(f"{res}\n", end='')

    # Запишем полученную метрическую матрицу для цилиндрической (и для нашей искомой) системы координат
    write_matrix3D(g, f"g{get_sub('i')}{get_sub('j')} = (vec)g{get_sub('i')}{get_sub('j')}", "\nЗапишем полученную метрическую матрицу для цилиндрической (и для нашей искомой) системы координат")
    
    # Найдем обратную метрическую матрицу:
    g_inverse = getMatrixInverse(g)
    write_matrix3D(g_inverse, f"g{get_super('i')}{get_super('j')} = (vec)g{get_super('i')}{get_super('j')}", "\nНайдем обратную метрическую матрицу")

    # Найдем векторы взаимного локального базиса для криволинейных координат Xi по формуле:
    print(f"\nНайдем векторы взаимного локального базиса для криволинейных координат X{get_super('i')} по формуле:\n", end='')

    print(f"r{get_super('i')} = ", end='')
    print(f"g{get_super('i')}{get_super('j')} * r{get_sub('j')} = ", end='')
    print(f"(vec)g{get_super('i')}{get_super('j')} * Q{get_super('m')}{get_sub('j')} * (vec)e{get_sub('m')} = ", end='')
    print(f"(vec)g{get_super('i')}{get_super('j')} * A{get_super('m')}{get_sub('p')} * (vec)Q{get_super('p')}{get_sub('j')} * e{get_sub('m')} = ", end='')
    print(f"(vec)Q{get_super('i')}{get_super('p')} * A{get_super('m')}{get_sub('p')} * (vec)e{get_sub('m')} = ", end='')
    print(f"Q{get_super('i')}{get_super('m')} * (vec)e{get_sub('m')}\n\n", end='')

    # Таким образом, для того, чтобы найти векторы взаимного базиса, найдем компоненты матрицы (Qim):
    print(f"Таким образом, для того, чтобы найти векторы взаимного базиса, найдем компоненты матрицы (Q{get_super('i')}{get_super('m')}):\n", end='')
    print(f"Q{get_super('i')}{get_super('m')} = (vec)Q{get_super('i')}{get_super('p')} * A{get_super('m')}{get_sub('p')}\n\n", end='')

    # В свою очередь, для этого нам понадобятся компоненты матрицы
    print(f"В свою очередь, для этого нам понадобятся компоненты матрицы\n", end='')
    print(f"(vec)Q{get_super('i')}{get_super('p')} = (vec)g{get_super('i')}{get_super('j')} * (vec)Q{get_super('p')}{get_sub('j')}\n\n", end='')

    # Вычислим их: 
    print(f"Вычислим их:\n", end='')
    for i in range(1, dim + 1):
        for p in range(1, dim + 1):

            res = 0
            print(f"(vec)Q{get_super(i)}{get_super(p)} = (vec)g{get_super(i)}{get_super('j')} * Q{get_super(p)}{get_sub('j')} = ", end='')

            for j in range(1, dim + 1):
                res += g_inverse[i - 1][j - 1] * Qⁱₖ[p - 1][j - 1]
                print(f"(vec)g{get_super(i)}{get_super(j)} * (vec)Q{get_super(p)}{get_sub(j)} ", end='')
                print(f"+ ", end='') if j < dim else print(f"= ", end='')

            res = sp.simplify(res)
            Qⁱᵖ[i - 1][p - 1] = res
            print(f"{res}\n", end='')

    # Запишем полученную матрицу:
    write_matrix3D(Qⁱᵖ, '(vec)Qⁱᵖ', "\nЗапишем полученную матрицу")

    # Найдем теперь компоненты матрицы
    print(f"\nНайдем теперь компоненты матрицы\n", end='')
    print(f"Q{get_super('i')}{get_super('m')} = (vec)Q{get_super('i')}{get_super('p')} * A{get_super('m')}{get_sub('p')}:\n\n", end='')

    # Вычислим их: 
    for i in range(1, dim + 1):
        for m in range(1, dim + 1):

            res = 0
            print(f"Q{get_super(i)}{get_super(m)} = (vec)Q{get_super(i)}{get_super('p')} * A{get_super(m)}{get_sub('p')} = ", end='')

            for p in range(1, dim + 1):
                res += Qⁱᵖ[i - 1][p - 1] * Aⁱⱼ[m - 1][p - 1]
                print(f"(vec)Q{get_super(i)}{get_super(p)} * A{get_super(m)}{get_sub(p)} ", end='')
                print(f"+ ", end='') if p < dim else print(f"= ", end='')

            res = sp.simplify(res)
            Qⁱᵐ[i - 1][m - 1] = res
            print(f"{res}\n", end='')

    # Запишем полученную матрицу:
    write_matrix3D(Qⁱᵐ, 'Qⁱᵐ', "\nЗапишем полученную матрицу")

    # Найдем символы Кристоффеля по формуле
    print(f"\nНайдем символы Кристоффеля по формуле:\n", end='')
    print(f"Г{get_super('m')}{get_sub('i')}{get_sub('j')} = ", end='')
    print(f"1/2 * g{get_super('k')}{get_super('m')} * (∂g{get_sub('k')}{get_sub('j')} / ∂X{get_super('i')} + ∂g{get_sub('i')}{get_sub('k')} / ∂X{get_super('j')} - ∂g{get_sub('i')}{get_sub('j')} / ∂X{get_super('k')})\n\n", end='')

    for i in range(1, dim + 1):
        for j in range(1, dim + 1):
            for m in range(1, dim + 1):
                res = 0

                print(f"Г{get_super(m)}{get_sub(i)}{get_sub(j)} = ", end='')
                print(f"1/2 * g{get_super('k')}{get_super(m)} * (∂g{get_sub('k')}{get_sub(j)} / ∂X{get_super(i)} + ∂g{get_sub(i)}{get_sub('k')} / ∂X{get_super(j)} - ∂g{get_sub(i)}{get_sub(j)} / ∂X{get_super('k')}) = ", end='')

                for k in range(1, dim + 1):
                    res += 1/2 * g_inverse[k - 1][m - 1] * (sp.diff(g[k - 1][j - 1], f'X{get_super(i)}') + sp.diff(g[i - 1][k - 1], f'X{get_super(j)}') - sp.diff(g[i - 1][j - 1], f'X{get_super(k)}'))

                    print(f"1/2 * g{get_super(k)}{get_super(m)} * (∂g{get_sub(k)}{get_sub(j)} / ∂X{get_super(i)} + ∂g{get_sub(i)}{get_sub(k)} / ∂X{get_super(j)} - ∂g{get_sub(i)}{get_sub(j)} / ∂X{get_super(k)}) ", end='')
                    print(f"+ ", end='') if k < dim else print(f"= ", end='')

                res = sp.simplify(res)
                Г[m - 1][i - 1][j - 1] = res
                print(f"{res}\n", end='')

    # Запишем результат:
    print(f"\nЗапишем результат:\n", end='')
    write_matrix3D(Г[1 - 1], f"Г{get_super(1)}{get_sub('i')}{get_sub('j')}", f"\nГ|{get_sub('m')}{get_sub('=')}{get_sub('1')}")
    write_matrix3D(Г[2 - 1], f"Г{get_super(2)}{get_sub('i')}{get_sub('j')}", f"\nГ|{get_sub('m')}{get_sub('=')}{get_sub('2')}")
    write_matrix3D(Г[3 - 1], f"Г{get_super(3)}{get_sub('i')}{get_sub('j')}", f"\nГ|{get_sub('m')}{get_sub('=')}{get_sub('3')}")

    # Вычислим символы Кристоффеля 1-го рода по формуле. "Опустим" индекс у ненулевых символов Кристоффеля 2-го рода: 
    print(f"\nВычислим символы Кристоффеля 1-го рода по формуле. \"Опустим\" индекс у ненулевых символов Кристоффеля 2-го рода:\n")
    print(f"Г{get_sub('i')}{get_sub('j')}{get_sub('k')} = ", end='')
    print(f"Г{get_super('m')}{get_sub('i')}{get_sub('j')} * g{get_sub('m')}{get_sub('k')}\n\n", end='')

    for i in range(1, dim + 1):
        for j in range(1, dim + 1):
            for k in range(1, dim + 1):
                res = 0

                print(f"Г{get_sub(i)}{get_sub(j)}{get_sub(k)} = ", end='')
                print(f"Г{get_super('m')}{get_sub(i)}{get_sub(j)} * g{get_sub('m')}{get_sub(k)} = ", end='')

                for m in range(1, dim + 1):
                    res += Г[m - 1][i - 1][j - 1] * g[m - 1][k - 1]

                    print(f"Г{get_super(m)}{get_sub(i)}{get_sub(j)} * g{get_sub(m)}{get_sub(k)} = ", end='')
                    print(f"+ ", end='') if m < dim else print(f"= ", end='')

                res = sp.simplify(res)
                Гᵢⱼₖ[i - 1][j - 1][k - 1] = res
                print(f"{res}\n", end='')

    # Запишем результат:
    print(f"\nЗапишем результат:\n", end='')
    write_matrix3D(Гᵢⱼₖ[1 - 1], f"Г{get_sub(1)}{get_sub('j')}{get_sub('k')}", f"\nГ|{get_sub('i')}{get_sub('=')}{get_sub('1')}")
    write_matrix3D(Гᵢⱼₖ[2 - 1], f"Г{get_sub(2)}{get_sub('j')}{get_sub('k')}", f"\nГ|{get_sub('i')}{get_sub('=')}{get_sub('2')}")
    write_matrix3D(Гᵢⱼₖ[3 - 1], f"Г{get_sub(3)}{get_sub('j')}{get_sub('k')}", f"\nГ|{get_sub('i')}{get_sub('=')}{get_sub('3')}")

    # Вычислим коэффициенты Ламе по формуле
    print(f"\nВычислим коэффициенты Ламе по формуле:\n", end='')
    print(f"H{get_sub('a')} = √(g{get_sub('a')}{get_sub('a')})\n\n", end='')

    for a in range (1, dim + 1):
        res = sp.sqrt(g[a - 1][a - 1])
        
        res = sp.simplify(res)
        H[a - 1] = res
        print(f"H{get_sub(a)} = √(g{get_sub(a)}{get_sub(a)}) = {res}\n", end='')

    # Запишем результат:
    print(f"\nЗапишем результат:\n", end='')
    for a in range(len(H)):
        print(f"H{get_sub(a+1)} = {H[a]}\n", end='')
