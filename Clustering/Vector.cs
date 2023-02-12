using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace Clustering {
    [DebuggerDisplay("{ToString(),nq}")]
    public partial class Vector : ICloneable, IEnumerable<(int index, double val)> {
        internal readonly double[] v;
        internal Vector(double[] v, bool cloning) {
            this.v = cloning ? (double[])v.Clone() : v;
        }

        /// <summary>コンストラクタ</summary>
        protected Vector(int size) {
            this.v = new double[size];
        }

        /// <summary>コンストラクタ</summary>
        public Vector(params double[] v) : this(v, cloning: true) { }

        /// <summary>コンストラクタ</summary>
        public Vector(IEnumerable<double> v) : this(v.ToArray()) { }

        /// <summary>コンストラクタ</summary>
        public Vector(IReadOnlyCollection<double> v) : this(v.ToArray()) { }

        /// <summary>X成分</summary>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public double X {
            get => v[0];
            set => v[0] = value;
        }

        /// <summary>Y成分</summary>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public double Y {
            get => v[1];
            set => v[1] = value;
        }

        /// <summary>Z成分</summary>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public double Z {
            get => v[2];
            set => v[2] = value;
        }

        /// <summary>W成分</summary>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public double W {
            get => v[3];
            set => v[3] = value;
        }

        /// <summary>次元数</summary>
        public int Dim => v.Length;

        /// <summary>キャスト</summary>
        public static implicit operator double[](Vector vector) {
            return (double[])vector.v.Clone();
        }

        /// <summary>キャスト</summary>
        public static implicit operator Vector(double[] arr) {
            return new Vector(arr);
        }

        /// <summary>正規化</summary>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public Vector Normal => this / Norm;

        /// <summary>ベクトル間距離</summary>
        public static double Distance(Vector vector1, Vector vector2) {
            return (vector1 - vector2).Norm;
        }

        /// <summary>ベクトル間距離2乗</summary>
        public static double SquareDistance(Vector vector1, Vector vector2) {
            return (vector1 - vector2).SquareNorm;
        }

        /// <summary>ノルム</summary>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public double Norm => Math.Sqrt(SquareNorm);

        /// <summary>ノルム2乗</summary>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public double SquareNorm {
            get {
                double sum_sq = 0d;

                foreach (var vi in v) {
                    sum_sq += vi * vi;
                }

                return sum_sq;
            }
        }

        /// <summary>合計</summary>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public double Sum => v.Sum();

        /// <summary>最大指数</summary>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public int MaxExponent {
            get {
                int max_exponent = int.MinValue;

                for (int i = 0; i < Dim; i++) {
                    if (double.IsFinite(v[i])) {
                        max_exponent = Math.Max(Math.ILogB((double)v[i]), max_exponent);
                    }
                }

                return max_exponent;
            }
        }

        /// <summary>2べき乗スケーリング</summary>
        public static Vector ScaleB(Vector vector, int n) {
            Vector ret = vector.Copy();

            for (int i = 0; i < ret.Dim; i++) {
                ret.v[i] = Math.ScaleB(ret.v[i], n);
            }

            return ret;
        }

        /// <summary>ゼロベクトル</summary>
        public static Vector Zero(int size) {
            return new Vector(size);
        }

        /// <summary>定数ベクトル</summary>
        public static Vector Fill(int size, double value) {
            double[] v = new double[size];

            for (int i = 0; i < v.Length; i++) {
                v[i] = value;
            }

            return new Vector(v, cloning: false);
        }

        /// <summary>連番ベクトル</summary>
        public static Vector Arange(int size) {
            double[] v = new double[size];

            for (int i = 0; i < v.Length; i++) {
                v[i] = i;
            }

            return new Vector(v, cloning: false);
        }

        /// <summary>射影</summary>
        public static Vector Func(Vector vector, Func<double, double> f) {
            double[] x = vector.v, v = new double[vector.Dim];

            for (int i = 0; i < v.Length; i++) {
                v[i] = f(x[i]);
            }

            return new Vector(v, cloning: false);
        }

        /// <summary>射影</summary>
        public static Vector Func(Vector vector1, Vector vector2, Func<double, double, double> f) {
            if (vector1.Dim != vector2.Dim) {
                throw new ArgumentException("mismatch size", $"{nameof(vector1)},{nameof(vector2)}");
            }

            double[] x = vector1.v, y = vector2.v, v = new double[vector1.Dim];

            for (int i = 0; i < v.Length; i++) {
                v[i] = f(x[i], y[i]);
            }

            return new Vector(v, cloning: false);
        }

        /// <summary>射影</summary>
        public static Vector Func(Vector vector1, Vector vector2, Vector vector3, Func<double, double, double, double> f) {
            if (vector1.Dim != vector2.Dim || vector1.Dim != vector3.Dim) {
                throw new ArgumentException("mismatch size", $"{nameof(vector1)},{nameof(vector2)},{nameof(vector3)}");
            }

            double[] x = vector1.v, y = vector2.v, z = vector3.v, v = new double[vector1.Dim];

            for (int i = 0; i < v.Length; i++) {
                v[i] = f(x[i], y[i], z[i]);
            }

            return new Vector(v, cloning: false);
        }

        /// <summary>射影</summary>
        public static Vector Func(Vector vector1, Vector vector2, Vector vector3, Vector vector4, Func<double, double, double, double, double> f) {
            if (vector1.Dim != vector2.Dim || vector1.Dim != vector3.Dim || vector1.Dim != vector4.Dim) {
                throw new ArgumentException("mismatch size", $"{nameof(vector1)},{nameof(vector2)},{nameof(vector3)},{nameof(vector4)}");
            }

            double[] x = vector1.v, y = vector2.v, z = vector3.v, w = vector4.v, v = new double[vector1.Dim];

            for (int i = 0; i < v.Length; i++) {
                v[i] = f(x[i], y[i], z[i], w[i]);
            }

            return new Vector(v, cloning: false);
        }

        /// <summary>不正なベクトル</summary>
        public static Vector Invalid(int size) {
            return Fill(size, value: double.NaN);
        }

        /// <summary>ゼロベクトルか判定</summary>
        public static bool IsZero(Vector vector) {
            for (int i = 0; i < vector.Dim; i++) {
                if (vector.v[i] != 0d) {
                    return false;
                }
            }

            return true;
        }

        /// <summary>有効なベクトルか判定</summary>
        public static bool IsValid(Vector vector) {
            for (int i = 0; i < vector.Dim; i++) {
                if (!double.IsFinite(vector.v[i])) {
                    return false;
                }
            }

            return true;
        }

        /// <summary>等しいか判定</summary>
        public override bool Equals(object obj) {
            return (obj is not null) && obj is Vector vector && vector == this;
        }

        /// <summary>ハッシュ値</summary>
        public override int GetHashCode() {
            return Dim > 0 ? v[0].GetHashCode() : 0;
        }

        /// <summary>クローン</summary>
        public object Clone() {
            return new Vector(v);
        }

        /// <summary>コピー</summary>
        public Vector Copy() {
            return new Vector(v);
        }

        /// <summary>文字列化</summary>
        public override string ToString() {
            if (Dim <= 0) {
                return string.Empty;
            }

            StringBuilder str = new($"{v[0]}");

            for (int i = 1; i < Dim; i++) {
                str.Append($",{v[i]}");
            }

            return str.ToString();
        }

        public IEnumerator<(int index, double val)> GetEnumerator() {
            for (int i = 0; i < Dim; i++) {
                yield return (i, v[i]);
            }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
