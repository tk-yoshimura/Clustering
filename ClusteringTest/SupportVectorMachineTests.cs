using Clustering;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.ObjectModel;

namespace ClusteringTest {
    [TestClass()]
    public class SupportVectorMachineTests {

        static readonly Random random = new(1234);

        static SupportVectorMachineTests() {
            Directory.CreateDirectory("../../testplot/");
        }

        readonly ReadOnlyCollection<Vector> separable_positive_vectors = new(
            Array.AsReadOnly(new Vector[]{
                (-1.0, -1.0),
                ( 0.0, -1.0),
                (+1.0, -1.0),
                (+0.4, -0.8),
                (-1.0, -0.6),
                (-0.6, -0.6),
                (-0.2, -0.6),
                (+0.8, -0.6),
                (-0.8, -0.4),
                (+0.2, -0.4),
                (-0.4, -0.2),
                (-0.8,  0.0),
        }));

        readonly ReadOnlyCollection<Vector> separable_negative_vectors = new(
            Array.AsReadOnly(new Vector[]{
                (+0.8, -0.4),
                (+0.4, -0.2),
                ( 0.0,  0.0),
                (+0.8,  0.0),
                (-0.8, +0.2),
                (-0.4, +0.4),
                ( 0.0, +0.4),
                (+0.6, +0.4),
                (+1.0, +0.4),
                (-1.0, +0.6),
                (-0.6, +0.8),
                (+0.6, +0.8),
                (+0.2, +1.0),
        }));

        readonly ReadOnlyCollection<Vector> notseparable_positive_vectors = new(
            Array.AsReadOnly(new Vector[]{
                (-1, +1),
                (+1, -1),
        }));

        readonly ReadOnlyCollection<Vector> notseparable_negative_vectors = new(
            Array.AsReadOnly(new Vector[]{
                (-1, -1),
                (+1, +1),
        }));

        readonly ReadOnlyCollection<Vector> random_positive_vectors = new(
            Array.AsReadOnly((new Vector[25]).Select(_ =>
                new Vector(1.8 * random.NextDouble() - 1.4, 1.8 * random.NextDouble() - 1.4)
            ).ToArray()
        ));

        readonly ReadOnlyCollection<Vector> random_negative_vectors = new(
            Array.AsReadOnly((new Vector[20]).Select(_ =>
                new Vector(1.8 * random.NextDouble() - 0.4, 1.8 * random.NextDouble() - 0.4)
            ).ToArray()
        ));

        [TestMethod()]
        public void LinearSVMTest() {
            for (int cost = 1; cost <= 10000; cost *= 10) {
                SupportVectorMachine linear_svm = new LinearSupportVectorMachine(cost);

                linear_svm.Learn(separable_positive_vectors, separable_negative_vectors);
                Vis.Plot(linear_svm, separable_positive_vectors, separable_negative_vectors, $"../../testplot/linearsvm_separable_cost{cost}_svm.png");

                linear_svm.Learn(notseparable_positive_vectors, notseparable_negative_vectors);
                Vis.Plot(linear_svm, notseparable_positive_vectors, notseparable_negative_vectors, $"../../testplot/linearsvm_notseparable_cost{cost}_svm.png");

                linear_svm.Learn(random_positive_vectors, random_negative_vectors);
                Vis.Plot(linear_svm, random_positive_vectors, random_negative_vectors, $"../../testplot/linearsvm_random_cost{cost}_svm.png");
            }
        }

        [TestMethod()]
        public void GaussianSVMTest() {
            for (int cost = 1; cost <= 10000; cost *= 10) {
                SupportVectorMachine gaussian_svm = new GaussianSupportVectorMachine(cost, sigma: 1);

                gaussian_svm.Learn(separable_positive_vectors, separable_negative_vectors);
                Vis.Plot(gaussian_svm, separable_positive_vectors, separable_negative_vectors, $"../../testplot/gaussiansvm_separable_cost{cost}_svm.png");

                gaussian_svm.Learn(notseparable_positive_vectors, notseparable_negative_vectors);
                Vis.Plot(gaussian_svm, notseparable_positive_vectors, notseparable_negative_vectors, $"../../testplot/gaussiansvm_notseparable_cost{cost}_svm.png");

                gaussian_svm.Learn(random_positive_vectors, random_negative_vectors);
                Vis.Plot(gaussian_svm, random_positive_vectors, random_negative_vectors, $"../../testplot/gaussiansvm_random_cost{cost}_svm.png");
            }
        }

        [TestMethod()]
        public void InvalidCreateTest() {
            SupportVectorMachine linear_svm = new LinearSupportVectorMachine(1);

            Assert.ThrowsException<ArgumentException>(() => {
                linear_svm.Learn();
            });

            Assert.ThrowsException<ArgumentException>(() => {
                linear_svm.Learn(separable_positive_vectors);
            });

            Assert.ThrowsException<ArgumentException>(() => {
                linear_svm.Learn(
                    Array.AsReadOnly(new Vector[] { (-1, +1), (+1, -1) }),
                    Array.AsReadOnly(new Vector[] { (-1, +1), (+1, -1, +1) })
                );
            });

            Assert.ThrowsException<ArgumentException>(() => {
                linear_svm.Learn(
                    Array.AsReadOnly(new Vector[] { (-1, +1), (+1, -1) }),
                    Array.AsReadOnly(new Vector[] { (-1, +1), (+1, +1) }),
                    Array.AsReadOnly(new Vector[] { (-1, +1), (-1, -1) })
                );
            });

            Assert.ThrowsException<ArgumentException>(() => {
                linear_svm.Learn(
                    Array.AsReadOnly(new Vector[] { }),
                    Array.AsReadOnly(new Vector[] { (-1, +1), (+1, -1) })
                );
            });
        }
    }
}
