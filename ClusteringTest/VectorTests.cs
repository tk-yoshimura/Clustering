using Clustering;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ClusteringTest {
    [TestClass()]
    public class VectorTests {
        [TestMethod()]
        public void VectorTest() {
            Vector vector = new(1, 2, 3, 4);

            Assert.AreEqual(1, vector[0]);
            Assert.AreEqual(2, vector[1]);
            Assert.AreEqual(3, vector[2]);
            Assert.AreEqual(4, vector[3]);

            Assert.AreEqual(1, vector.X);
            Assert.AreEqual(2, vector.Y);
            Assert.AreEqual(3, vector.Z);
            Assert.AreEqual(4, vector.W);

            Assert.AreEqual(4, vector.Dim);

            vector.X = 4;
            vector.Y = 3;
            vector.Z = 2;
            vector.W = 1;

            Assert.AreEqual(4, vector.X);
            Assert.AreEqual(3, vector.Y);
            Assert.AreEqual(2, vector.Z);
            Assert.AreEqual(1, vector.W);

            vector[0] = 1;
            vector[1] = 2;
            vector[2] = 3;
            vector[3] = 4;

            Assert.AreEqual(1, vector[0]);
            Assert.AreEqual(2, vector[1]);
            Assert.AreEqual(3, vector[2]);
            Assert.AreEqual(4, vector[3]);

            vector[^1] = 1;
            vector[^2] = 2;
            vector[^3] = 3;
            vector[^4] = 4;

            Assert.AreEqual(1, vector[^1]);
            Assert.AreEqual(2, vector[^2]);
            Assert.AreEqual(3, vector[^3]);
            Assert.AreEqual(4, vector[^4]);

            string str = string.Empty;
            foreach ((int index, double val) in vector) {
                str += $"({index},{val}),";
            }

            Assert.AreEqual("(0,4),(1,3),(2,2),(3,1),", str);
        }

        [TestMethod()]
        public void RangeIndexerGetterTest() {
            Vector vector = new(1, 2, 3, 4, 5);

            Assert.AreEqual(new Vector(1, 2, 3, 4, 5), vector[..]);

            Assert.AreEqual(new Vector(2, 3, 4, 5), vector[1..]);
            Assert.AreEqual(new Vector(3, 4, 5), vector[2..]);
            Assert.AreEqual(new Vector(1, 2, 3, 4), vector[..^1]);
            Assert.AreEqual(new Vector(1, 2, 3, 4), vector[..4]);
            Assert.AreEqual(new Vector(1, 2, 3), vector[..^2]);
            Assert.AreEqual(new Vector(1, 2, 3), vector[..3]);

            Assert.AreEqual(new Vector(2, 3, 4), vector[1..4]);
            Assert.AreEqual(new Vector(2, 3, 4), vector[1..^1]);
        }

        [TestMethod()]
        public void RangeIndexerSetterTest() {
            Vector vector_src = new(1, 2, 3, 4, 5);
            Vector vector_dst;

            vector_dst = Vector.Zero(vector_src.Dim);
            vector_dst[..] = vector_src;
            Assert.AreEqual(new Vector(1, 2, 3, 4, 5), vector_dst);

            vector_dst = Vector.Zero(vector_src.Dim);
            vector_dst[1..] = vector_src[1..];
            Assert.AreEqual(new Vector(0, 2, 3, 4, 5), vector_dst);

            vector_dst = Vector.Zero(vector_src.Dim);
            vector_dst[2..] = vector_src[2..];
            Assert.AreEqual(new Vector(0, 0, 3, 4, 5), vector_dst);

            vector_dst = Vector.Zero(vector_src.Dim);
            vector_dst[..^1] = vector_src[..^1];
            Assert.AreEqual(new Vector(1, 2, 3, 4, 0), vector_dst);

            vector_dst = Vector.Zero(vector_src.Dim);
            vector_dst[..4] = vector_src[..4];
            Assert.AreEqual(new Vector(1, 2, 3, 4, 0), vector_dst);

            vector_dst = Vector.Zero(vector_src.Dim);
            vector_dst[..^2] = vector_src[..^2];
            Assert.AreEqual(new Vector(1, 2, 3, 0, 0), vector_dst);

            vector_dst = Vector.Zero(vector_src.Dim);
            vector_dst[..3] = vector_src[..3];
            Assert.AreEqual(new Vector(1, 2, 3, 0, 0), vector_dst);

            vector_dst = Vector.Zero(vector_src.Dim);
            vector_dst[1..4] = vector_src[1..4];
            Assert.AreEqual(new Vector(0, 2, 3, 4, 0), vector_dst);

            vector_dst = Vector.Zero(vector_src.Dim);
            vector_dst[1..^1] = vector_src[1..^1];
            Assert.AreEqual(new Vector(0, 2, 3, 4, 0), vector_dst);

            vector_dst = Vector.Zero(vector_src.Dim);
            vector_dst[0..^2] = vector_src[1..^1];
            Assert.AreEqual(new Vector(2, 3, 4, 0, 0), vector_dst);

            Assert.ThrowsException<ArgumentOutOfRangeException>(() => {
                vector_dst = Vector.Zero(vector_src.Dim);
                vector_dst[0..^2] = vector_src[1..^2];
            });

            Assert.ThrowsException<ArgumentOutOfRangeException>(() => {
                vector_dst = Vector.Zero(vector_src.Dim);
                vector_dst[0..^2] = vector_src[1..];
            });
        }

        [TestMethod()]
        public void NormTest() {
            Vector vector = new(-3, 4);

            Assert.AreEqual(5, vector.Norm);
            Assert.AreEqual(25, vector.SquareNorm);
        }

        [TestMethod()]
        public void SumTest() {
            Vector vector = new(1, 2, 3, 4);

            Assert.AreEqual(10, vector.Sum);
        }

        [TestMethod()]
        public void NormalTest() {
            Vector vector = new(1, 2, -3);

            Assert.AreEqual(1 / Math.Sqrt(1 + 4 + 9), vector.Normal.X);
            Assert.AreEqual(2 / Math.Sqrt(1 + 4 + 9), vector.Normal.Y);
            Assert.AreEqual(-3 / Math.Sqrt(1 + 4 + 9), vector.Normal.Z);
        }

        [TestMethod()]
        public void OperatorTest() {
            Vector vector1 = new(1, 2);
            Vector vector2 = new(3, 4);

            Assert.AreEqual(new Vector(1, 2), +vector1);
            Assert.AreEqual(new Vector(-1, -2), -vector1);
            Assert.AreEqual(new Vector(4, 6), vector1 + vector2);
            Assert.AreEqual(new Vector(4, 6), vector2 + vector1);
            Assert.AreEqual(new Vector(4, 5), vector1 + 3);
            Assert.AreEqual(new Vector(4, 5), 3 + vector1);
            Assert.AreEqual(new Vector(-2, -2), vector1 - vector2);
            Assert.AreEqual(new Vector(2, 2), vector2 - vector1);
            Assert.AreEqual(new Vector(-2, -1), vector1 - 3);
            Assert.AreEqual(new Vector(2, 1), 3 - vector1);
            Assert.AreEqual(new Vector(2, 4), vector1 * 2);
            Assert.AreEqual(new Vector(2, 4), 2 * vector1);
            Assert.AreEqual(new Vector(0.5, 1), vector1 / 2);
            Assert.AreEqual(new Vector(2, 1), 2 / vector1);
            Assert.AreEqual(new Vector(3, 8), vector1 * vector2);
            Assert.AreEqual(new Vector((double)1 / 3, 0.5), vector1 / vector2);
            Assert.AreEqual(new Vector(3, 2), vector2 / vector1);
        }

        [TestMethod()]
        public void DotTest() {
            Vector vector1 = new(1, 2);
            Vector vector2 = new(3, 4);
            Vector vector3 = new(2, -1);

            Assert.AreEqual(11, Vector.Dot(vector1, vector2));
            Assert.AreEqual(11, Vector.Dot(vector2, vector1));
            Assert.AreEqual(0, Vector.Dot(vector1, vector3));
            Assert.AreEqual(2, Vector.Dot(vector2, vector3));
        }

        [TestMethod()]
        public void CrossTest() {
            Vector vector1 = new(1, 2, 3);
            Vector vector2 = new(4, -5, -6);
            Vector vector3 = new(3, 1, 0);
            Vector vector4 = new(2, 5, 1);

            Assert.AreEqual(Vector.Zero(3), Vector.Cross(vector1, vector2) + Vector.Cross(vector2, vector1));
            Assert.AreEqual(new Vector(1, -3, 13), Vector.Cross(vector3, vector4));
        }

        [TestMethod()]
        public void PolynomialTest() {
            Vector x = new(1, 2, 3, 4);
            Vector coef = new(5, 7, 11, 13, 17);

            Assert.AreEqual(new Vector(53, 439, 1853, 5393), Vector.Polynomial(x, coef));
            Assert.AreEqual(13, Vector.Polynomial(-1, coef));

            Assert.AreEqual(new Vector(5, 5, 5, 5), Vector.Polynomial(x, new Vector(5)));
            Assert.AreEqual(5, Vector.Polynomial(-1, new Vector(5)));

            Assert.AreEqual(new Vector(0, 0, 0, 0), Vector.Polynomial(x, new Vector(new double[0])));
            Assert.AreEqual(0, Vector.Polynomial(-1, new Vector(new double[0])));
        }

        [TestMethod()]
        public void DistanceTest() {
            Vector vector1 = new(1, 2);
            Vector vector2 = new(2, 1);

            Assert.AreEqual(Math.Sqrt(2), Vector.Distance(vector1, vector2));
            Assert.AreEqual(2, Vector.SquareDistance(vector1, vector2));
        }

        [TestMethod()]
        public void ZeroTest() {
            Vector vector = Vector.Zero(3);

            Assert.AreEqual(0, vector.X);
            Assert.AreEqual(0, vector.Y);
            Assert.AreEqual(0, vector.Z);
        }

        [TestMethod()]
        public void FillTest() {
            Vector vector = Vector.Fill(3, value: 7);

            Assert.AreEqual(7, vector.X);
            Assert.AreEqual(7, vector.Y);
            Assert.AreEqual(7, vector.Z);
        }

        [TestMethod()]
        public void ArangeTest() {
            Vector vector = Vector.Arange(8);

            for (int i = 0; i < vector.Dim; i++) {
                Assert.AreEqual(i, vector[i]);
            }
        }

        [TestMethod()]
        public void InvalidTest() {
            Vector vector = Vector.Invalid(3);

            Assert.IsTrue(double.IsNaN(vector.X));
            Assert.IsTrue(double.IsNaN(vector.Y));
            Assert.IsTrue(double.IsNaN(vector.Z));
        }

        [TestMethod()]
        public void IsZeroTest() {
            Vector vector = Vector.Zero(3);

            Assert.IsTrue(Vector.IsZero(vector));

            vector.X = 1;

            Assert.IsFalse(Vector.IsZero(vector));
        }

        [TestMethod()]
        public void IsValidTest() {
            Vector vector = Vector.Zero(3);

            Assert.IsTrue(Vector.IsValid(vector));

            vector.X = double.NaN;

            Assert.IsFalse(Vector.IsValid(vector));
        }

        [TestMethod()]
        public void OperatorEqualTest() {
            Vector vector = new(1, 2);

            Assert.IsTrue(vector == new Vector(1, 2));
            Assert.IsTrue(vector != new Vector(2, 1));
            Assert.IsTrue(vector != new Vector(1));
            Assert.IsTrue(vector != new Vector(1, 2, 3));
            Assert.IsFalse(vector == null);
            Assert.IsTrue(vector != null);
            Assert.IsTrue(vector != new Vector(1, double.NaN));
        }

        [TestMethod()]
        public void EqualsTest() {
            Vector vector = new(1, 2);

            Assert.IsTrue(vector.Equals(new Vector(1, 2)));
            Assert.IsFalse(vector.Equals(new Vector(2, 1)));
            Assert.IsFalse(vector.Equals(new Vector(1)));
            Assert.IsFalse(vector.Equals(new Vector(1, 2, 3)));
            Assert.IsFalse(vector.Equals(null));
            Assert.IsFalse(vector.Equals(new Vector(1, double.NaN)));
        }

        [TestMethod()]
        public void FuncTest() {
            Vector vector1 = new(1, 2, 4, 8);
            Vector vector2 = new(2, 3, 5, 9);
            Vector vector3 = new(3, 4, 6, 10);
            Vector vector4 = new(4, 5, 7, 11);
            Vector vector5 = new(5, 6, 8, 12, 20);

            Assert.AreEqual(new Vector(2, 4, 8, 16), Vector.Func(vector1, v => 2 * v));
            Assert.AreEqual(new Vector(5, 8, 14, 26), Vector.Func(vector1, vector2, (v1, v2) => v1 + 2 * v2));
            Assert.AreEqual(new Vector(17, 24, 38, 66), Vector.Func(vector1, vector2, vector3, (v1, v2, v3) => v1 + 2 * v2 + 4 * v3));
            Assert.AreEqual(new Vector(49, 64, 94, 154), Vector.Func(vector1, vector2, vector3, vector4, (v1, v2, v3, v4) => v1 + 2 * v2 + 4 * v3 + 8 * v4));
            Assert.AreEqual(new Vector(49, 64, 94, 154), Vector.Func(vector1, vector2, vector3, vector4, (v1, v2, v3, v4) => v1 + 2 * v2 + 4 * v3 + 8 * v4));

            Assert.ThrowsException<ArgumentException>(() => {
                Vector.Func(vector1, vector5, (v1, v2) => v1 + 2 * v2);
            });

            Assert.ThrowsException<ArgumentException>(() => {
                Vector.Func(vector5, vector1, (v1, v2) => v1 + 2 * v2);
            });

            Assert.ThrowsException<ArgumentException>(() => {
                Vector.Func(vector1, vector2, vector5, (v1, v2, v3) => v1 + 2 * v2 + 4 * v3);
            });

            Assert.ThrowsException<ArgumentException>(() => {
                Vector.Func(vector1, vector5, vector2, (v1, v2, v3) => v1 + 2 * v2 + 4 * v3);
            });

            Assert.ThrowsException<ArgumentException>(() => {
                Vector.Func(vector5, vector1, vector2, (v1, v2, v3) => v1 + 2 * v2 + 4 * v3);
            });

            Assert.ThrowsException<ArgumentException>(() => {
                Vector.Func(vector1, vector2, vector3, vector5, (v1, v2, v3, v4) => v1 + 2 * v2 + 4 * v3 + 8 * v4);
            });

            Assert.ThrowsException<ArgumentException>(() => {
                Vector.Func(vector1, vector2, vector5, vector3, (v1, v2, v3, v4) => v1 + 2 * v2 + 4 * v3 + 8 * v4);
            });

            Assert.ThrowsException<ArgumentException>(() => {
                Vector.Func(vector1, vector5, vector2, vector3, (v1, v2, v3, v4) => v1 + 2 * v2 + 4 * v3 + 8 * v4);
            });

            Assert.ThrowsException<ArgumentException>(() => {
                Vector.Func(vector5, vector1, vector2, vector3, (v1, v2, v3, v4) => v1 + 2 * v2 + 4 * v3 + 8 * v4);
            });
        }

        [TestMethod()]
        public void ConcatTest() {
            Vector vector1 = Vector.Fill(1, value: -1);
            Vector vector2 = Vector.Fill(2, value: -2);
            Vector vector4 = Vector.Fill(4, value: -3);

            Assert.AreEqual(new Vector(-1, -2, -2, -3, -3, -3, -3), Vector.Concat(vector1, vector2, vector4));
            Assert.AreEqual(new Vector(-2, -2, -3, -3, -3, -3, -1), Vector.Concat(vector2, vector4, vector1));
            Assert.AreEqual(new Vector(-1, -2, -2, 1, -3, -3, -3, -3), Vector.Concat(vector1, vector2, 1, vector4));
            Assert.AreEqual(new Vector(-1, -2, -2, 2, -3, -3, -3, -3), Vector.Concat(vector1, vector2, 2L, vector4));
            Assert.AreEqual(new Vector(-1, -2, -2, 3, -3, -3, -3, -3), Vector.Concat(vector1, vector2, (double)3, vector4));
            Assert.AreEqual(new Vector(-1, -2, -2, 4, -3, -3, -3, -3), Vector.Concat(vector1, vector2, 4d, vector4));
            Assert.AreEqual(new Vector(-1, -2, -2, 5, -3, -3, -3, -3), Vector.Concat(vector1, vector2, 5f, vector4));

            Assert.ThrowsException<ArgumentException>(() => {
                Vector.Concat(vector1, vector2, 'b', vector4);
            });
        }

        [TestMethod()]
        public void TupleTest() {
            double x, y, z, w, e0, e1, e2, e3, e4, e5, e6, e7;

            Vector vector2 = (2, 4);
            (x, y) = vector2;
            Assert.AreEqual((2, 4), (x, y));

            Vector vector3 = (2, 4, 6);
            (x, y, z) = vector3;
            Assert.AreEqual((2, 4, 6), (x, y, z));

            Vector vector4 = (2, 4, 6, 8);
            (x, y, z, w) = vector4;
            Assert.AreEqual((2, 4, 6, 8), (x, y, z, w));

            Vector vector5 = (2, 4, 6, 8, 1);
            (e0, e1, e2, e3, e4) = vector5;
            Assert.AreEqual((2, 4, 6, 8, 1), (e0, e1, e2, e3, e4));

            Vector vector6 = (2, 4, 6, 8, 1, 3);
            (e0, e1, e2, e3, e4, e5) = vector6;
            Assert.AreEqual((2, 4, 6, 8, 1, 3), (e0, e1, e2, e3, e4, e5));

            Vector vector7 = (2, 4, 6, 8, 1, 3, 5);
            (e0, e1, e2, e3, e4, e5, e6) = vector7;
            Assert.AreEqual((2, 4, 6, 8, 1, 3, 5), (e0, e1, e2, e3, e4, e5, e6));

            Vector vector8 = (2, 4, 6, 8, 1, 3, 5, 7);
            (e0, e1, e2, e3, e4, e5, e6, e7) = vector8;
            Assert.AreEqual((2, 4, 6, 8, 1, 3, 5, 7), (e0, e1, e2, e3, e4, e5, e6, e7));

            Assert.ThrowsException<InvalidOperationException>(() => {
                (x, y, z) = vector2;
            });
            Assert.ThrowsException<InvalidOperationException>(() => {
                (x, y, z, w) = vector3;
            });
            Assert.ThrowsException<InvalidOperationException>(() => {
                (e0, e1, e2, e3, e4) = vector4;
            });
            Assert.ThrowsException<InvalidOperationException>(() => {
                (e0, e1, e2, e3, e4, e5) = vector5;
            });
            Assert.ThrowsException<InvalidOperationException>(() => {
                (e0, e1, e2, e3, e4, e5, e6) = vector6;
            });
            Assert.ThrowsException<InvalidOperationException>(() => {
                (e0, e1, e2, e3, e4, e5, e6, e7) = vector7;
            });
            Assert.ThrowsException<InvalidOperationException>(() => {
                (e0, e1, e2, e3, e4, e5, e6) = vector8;
            });
        }

        [TestMethod()]
        public void CopyTest() {
            Vector vector1 = new(1, 2);
            Vector vector2 = vector1.Copy();

            vector2.X = 2;

            Assert.AreEqual(1, vector1.X);
            Assert.AreEqual(2, vector1.Y);
            Assert.AreEqual(2, vector2.X);
            Assert.AreEqual(2, vector2.Y);
        }

        [TestMethod()]
        public void ToStringTest() {
            Vector vector1 = new(1, 2, 3);
            Vector vector2 = new(new double[0]);
            Vector vector3 = new(1d);

            Assert.AreEqual("1,2,3", vector1.ToString());
            Assert.AreEqual(string.Empty, vector2.ToString());
            Assert.AreEqual("1", vector3.ToString());
        }
    }
}