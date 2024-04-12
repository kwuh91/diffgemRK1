import diffgem
from fractions import Fraction
import sympy as sp

from diffgem import Matrix, Vector

if __name__ == '__main__':

    """
    ТИ. Задание 11. 
    Нахождение компонент скалярного произведения тензоров, 
    заданных своими компонентами в ортонормированном базисе, 
    в некотором другом базисе. 
    """

    A: Matrix = [[1, 0],
                 [6, 2]]
    B: Matrix = [[8, 1],
                 [3, 7]]
    Q: Matrix = [[1, 0],
                 [1, 2]]
    diffgem.componentOfTheScalarWork(A, B, Q)

    """
    ТИ. Задание 12. 
    Нахождение компонент скалярного произведения тензора на вектор, 
    заданных своими компонентами в ортонормированном базисе, 
    в некотором другом базисе.
    """

    T: Matrix = [[1, 1],
                 [0, 0]]
    a: Vector = [0,
                 1]
    Q: Matrix = [[1, 1],
                 [0, 1]]
    
    diffgem.findingComponents(T, a, Q)

    """
    ТИ. Задание 13. 
    Нахождение полной свертки тензоров, 
    заданных в ортонормированном базисе. 
    """

    A: Matrix = [[1, 2],
                 [0, 7]]
    B: Matrix = [[3, -1],
                 [2, 1]]
    
    diffgem.fullConvolution(A, B)
    
    """
    ТИ. Задание 17. 
    Нахождение векторного произведения тензора на вектор, 
    заданных в декартовом ортонормированном некотором базисе.
    """

    a: Vector = [1, 
                 0, 
                 1]
    T: Matrix = [[2, 3, -1],
                 [1, 2, 0 ],
                 [1, 0, -1]]

    diffgem.vectorProductOfTheTensorOnTheVector(a, T)

    """
    ТИ. Задание 23. 
    Дифференциальные характеристики цилиндрических координат.
    """

    A: Matrix = [[1, 0,                0               ],
                 [0, sp.Rational(1/2), -sp.sqrt(3)/2   ],
                 [0, sp.sqrt(3)/2,     sp.Rational(1/2)]]

    diffgem.differentialCharacteristics(A)
