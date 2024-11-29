using System;
using System.Numerics;
using MathNet.Numerics.LinearAlgebra;

class Program
{
    static void Main(string[] args)
    {
        // Початкове наближення
        double[] x0 = { 0.0, -2.0 }; // Початкові здогадки для x і y
        int maxIter = 5; // Максимальна кількість ітерацій
        double tol = 1e-6; // Допуск

        // Розв'язок системи
        double[] solution = NewtonMethod(F, J, x0, tol, maxIter);

        // Результат
        if (solution != null)
        {
            Console.WriteLine("Solution: [{0}, {1}]", solution[0], solution[1]);
        }
        else
        {
            Console.WriteLine("No solution found.");
        }
    }

    // Система рівнянь
    static Vector<double> F(Vector<double> x)
    {
        return Vector<double>.Build.DenseOfArray(new double[]
        {
            Math.Sin(x[0] - 0.6) - x[1] - 1.6,
            3 * x[0] - Math.Cos(x[1]) - 0.9
        });
    }

    // Якобіан
    static Matrix<double> J(Vector<double> x)
    {
        return Matrix<double>.Build.DenseOfArray(new double[,]
        {
            { Math.Cos(x[0] - 0.6), -1 }, // Часткові похідні F1
            { 3, Math.Sin(x[1]) }        // Часткові похідні F2
        });
    }

    // Метод Ньютона
    static double[] NewtonMethod(
        Func<Vector<double>, Vector<double>> F,
        Func<Vector<double>, Matrix<double>> J,
        double[] x0,
        double tol,
        int maxIter)
    {
        var x = Vector<double>.Build.DenseOfArray(x0);

        for (int i = 0; i < maxIter; i++)
        {
            var Fx = F(x);
            var Jx = J(x);

            try
            {
                // Розв'язуємо J(x) * deltaX = -F(x)
                var deltaX = Jx.Solve(-Fx);

                // Оновлюємо x
                x += deltaX;

                Console.WriteLine($"Iteration {i + 1}: x = [{x[0]}, {x[1]}]");

                // Перевіряємо на збіжність
                if (Fx.L2Norm() < tol)
                {
                    Console.WriteLine($"Converged after {i + 1} iterations.");
                    return x.ToArray();
                }
            }
            catch
            {
                Console.WriteLine("Jacobian is singular.");
                return null;
            }
        }

        Console.WriteLine("Failed to converge after maximum iterations.");
        return x.ToArray();
    }
}
