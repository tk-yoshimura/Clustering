using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace Clustering {
    /// <summary>線形サポートベクタマシン</summary>
    public class LinearSupportVectorMachine : SupportVectorMachine {

        /// <summary>コンストラクタ</summary>
        /// <param name="cost">誤識別に対するペナルティの大きさ</param>
        public LinearSupportVectorMachine(double cost) : base(cost) { }

        /// <summary>カーネル関数</summary>
        protected override double Kernel(Vector vector1, Vector vector2) {
            double dot = Vector.Dot(vector1, vector2);

            return dot;
        }
    }

    /// <summary>ガウシアンサポートベクタマシン</summary>
    public class GaussianSupportVectorMachine : SupportVectorMachine {
        private double sigma, gamma;

        /// <param name="cost">誤識別に対するペナルティの大きさ</param>
        /// <param name="sigma">ガウシアン関数の尺度パラメータ</param>
        public GaussianSupportVectorMachine(double cost, double sigma) : base(cost) {
            this.Sigma = sigma;
        }

        /// <summary>ガウシアン関数の尺度パラメータ</summary>
        public double Sigma {
            get {
                return sigma;
            }
            protected set {
                sigma = value;
                gamma = 1 / (2 * sigma * sigma);
            }
        }

        /// <summary>カーネル関数</summary>
        protected override double Kernel(Vector vector1, Vector vector2) {
            double norm = (vector1 - vector2).SquareNorm;

            return Math.Exp(-gamma * norm);
        }
    }

    /// <summary>サポートベクタマシン</summary>
    public abstract class SupportVectorMachine : IClusteringMethod {

        private double bias;
        private readonly double cost;
        List<(Vector vector, double weight)> support_vectors;

        /// <summary>コンストラクタ</summary>
        /// <param name="cost">誤識別に対するペナルティの大きさ</param>
        public SupportVectorMachine(double cost) {
            if (!(cost > 0)) {
                throw new ArgumentOutOfRangeException(nameof(cost));
            }

            Initialize();
            this.cost = cost;
        }

        /// <summary>データクラス数</summary>
        /// <remarks>SVMは常に2</remarks>
        public int GroupCount => 2;

        /// <summary>ベクトルの次元数</summary>
        public int VectorDim {
            get; private set;
        }

        /// <summary>単一サンプルを分類</summary>
        /// <param name="vector">サンプルベクタ</param>
        public int Classify(Vector vector) {
            return ClassifyRaw(vector) > 0 ? +1 : ClassifyRaw(vector) < 0 ? -1 : 0;
        }

        /// <summary>単一サンプルを分類</summary>
        /// <param name="vector">サンプルベクタ</param>
        /// <param name="threshold">弁別しきい値</param>
        public int Classify(Vector vector, double threshold) {
            return ClassifyRaw(vector) > threshold ? +1 : ClassifyRaw(vector) < -threshold ? -1 : 0;
        }

        /// <summary>複数サンプルを分類</summary>
        /// <param name="vectors">サンプルベクタ集合</param>
        public IEnumerable<int> Classify(IEnumerable<Vector> vectors) {
            return vectors.Select((vector) => Classify(vector));
        }

        /// <summary>複数サンプルを分類</summary>
        /// <param name="vectors">サンプルベクタ集合</param>
        /// <param name="threshold">弁別しきい値</param>
        public IEnumerable<int> Classify(IEnumerable<Vector> vectors, double threshold) {
            return vectors.Select((vector) => Classify(vector, threshold));
        }

        /// <summary>単一サンプルの識別値</summary>
        public double ClassifyRaw(Vector vector) {
            if (vector == null || vector.Dim != VectorDim) {
                throw new ArgumentException("Mismatch vector dim.", nameof(vector));
            }

            double s = -bias;
            foreach ((Vector v, double w) in support_vectors) {
                s += w * Kernel(vector, v);
            }
            return s;
        }

        /// <summary>複数サンプルの識別値</summary>
        public IEnumerable<double> ClassifyRaw(IEnumerable<Vector> vectors) {
            return vectors.Select((vector) => ClassifyRaw(vector));
        }

        /// <summary>学習</summary>
        /// <param name="vectors_groups">データクラスごとのサンプルベクタ集合</param>
        /// <remarks>サンプルベクタ集合は正例と負例の2つ</remarks>
        public void Learn(params ReadOnlyCollection<Vector>[] vectors_groups) {
            Initialize();
            ClusteringMethodUtil.ValidateSample(GroupCount, vectors_groups);

            // サポートベクターとなる最小のベクトル重み
            double epsilon = 1.0e-3;

            // ベクトルの次元数
            VectorDim = vectors_groups[0][0].Dim;

            // ベクトル
            ReadOnlyCollection<Vector> positive_vectors = vectors_groups[0];
            ReadOnlyCollection<Vector> negative_vectors = vectors_groups[1];
            List<Vector> inputs = [.. positive_vectors, .. negative_vectors];

            // ラベル
            double[] outputs = (new double[inputs.Count]).Select((_, i) => i < positive_vectors.Count ? +1.0 : -1.0).ToArray();

            // 逐次最小問題最適化法実行
            var smo = new SequentialMinimalOptimization(inputs.ToArray(), outputs, cost, Kernel);
            smo.Optimize();

            bias = smo.Bias;
            ReadOnlyCollection<double> vector_weight = smo.VectorWeight;

            //サポートベクターの格納
            for (int i = 0; i < vector_weight.Count; i++) {
                if (vector_weight[i] > epsilon) {
                    support_vectors.Add(((Vector)inputs[i].Clone(), vector_weight[i] * outputs[i]));
                }
            }
        }

        /// <summary>カーネル関数</summary>
        protected abstract double Kernel(Vector vector1, Vector vector2);

        /// <summary>初期化</summary>
        public void Initialize() {
            bias = 0;
            support_vectors = new List<(Vector, double)>();
        }
    }
}
