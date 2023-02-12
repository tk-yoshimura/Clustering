using Clustering;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ClusteringTest {
    [TestClass()]
    public class KMeansClusteringTests {

        static KMeansClusteringTests() {
            Directory.CreateDirectory("../../testplot/");
        }

        [TestMethod()]
        public void PlotTest() {
            Random random = new(9);

            List<Vector> vectors = new();

            Vector center;

            center = (2 * random.NextDouble() - 1, 2 * random.NextDouble() - 1);
            for (int i = 0; i < 25; i++) {
                vectors.Add(center + (0.8 * random.NextDouble() - 0.4, 0.8 * random.NextDouble() - 0.4));
            }
            center = (2 * random.NextDouble() - 1, 2 * random.NextDouble() - 1);
            for (int i = 0; i < 20; i++) {
                vectors.Add(center + (0.8 * random.NextDouble() - 0.4, 0.8 * random.NextDouble() - 0.4));
            }
            center = (2 * random.NextDouble() - 1, 2 * random.NextDouble() - 1);
            for (int i = 0; i < 20; i++) {
                vectors.Add(center + (0.8 * random.NextDouble() - 0.4, 0.8 * random.NextDouble() - 0.4));
            }
            center = (2 * random.NextDouble() - 1, 2 * random.NextDouble() - 1);
            for (int i = 0; i < 25; i++) {
                vectors.Add(center + (0.8 * random.NextDouble() - 0.4, 0.8 * random.NextDouble() - 0.4));
            }

            KMeansClustering kmean = new(group_count: 4);
            kmean.Learn(Array.AsReadOnly(vectors.ToArray()));

            Vis.Plot(kmean, Array.AsReadOnly(vectors.ToArray()), $"../../testplot/kmeans.png");
        }

        [TestMethod()]
        public void InvalidCreateTest() {
            KMeansClustering kmean = new(group_count: 4);

            Assert.ThrowsException<ArgumentException>(() => {
                kmean.Learn();
            });

            Assert.ThrowsException<ArgumentException>(() => {
                kmean.Learn(
                    Array.AsReadOnly(new Vector[] { (-1, +1), (+1, -1), (-1, -1), (+1, +1), (+1, 1, 1) })
                );
            });

            Assert.ThrowsException<ArgumentException>(() => {
                kmean.Learn(
                    Array.AsReadOnly(new Vector[] { (-1, +1), (+1, -1), (-1, -1) })
                );
            });

            Assert.ThrowsException<ArgumentException>(() => {
                kmean.Learn(
                    Array.AsReadOnly(new Vector[] { (-1, +1), (+1, -1), (-2, +2), (+1, -2) }),
                    Array.AsReadOnly(new Vector[] { (-1, -1), (+1, +1), (-2, -2), (+2, +2) })
                );
            });
        }
    }
}
