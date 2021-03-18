using System;
using System.Collections.Generic;
using System.Linq;
using Algebra;

namespace Clustering {

    /// <summary>k-means++法</summary>
    public class KMeansClustering : IClusteringMethod {
        protected class LabelVector {
            public Vector Vector { get; set; }
            public int Label { get; set; }
        }

        Vector[] center_vectors;

        /// <summary>コンストラクタ</summary>
        /// <param name="group_count">データクラス数</param>
        public KMeansClustering(int group_count) {
            if(group_count <= 1) {
                throw new ArgumentException(nameof(group_count));
            }

            this.GroupCount = group_count;
        }

        /// <summary>データクラス数</summary>
        public int GroupCount {
            get; private set;
        }

        /// <summary>ベクトルの次元数</summary>
        public int VectorDim {
            get; private set;
        }

        /// <summary>クラスタ中心ベクトル</summary>
        public Vector[] CenterVectors => center_vectors;

        /// <summary>単一サンプルを分類</summary>
        /// <param name="vector">サンプルベクタ</param>
        public int Classify(Vector vector) {
            return NearestVector(vector);
        }

        /// <summary>複数サンプルを分類</summary>
        /// <param name="vectors">サンプルベクタ集合</param>
        public IEnumerable<int> Classify(IEnumerable<Vector> vectors) {
            return vectors.Select((vector) => Classify(vector));
        }

        /// <summary>学習</summary>
        /// <param name="vector_dim">サンプルベクタ次元数</param>
        /// <param name="vectors_groups">データクラスごとのサンプルベクタ集合</param>
        public void Learn(int vector_dim, params List<Vector>[] vectors_groups) {
            Initialize();
            ValidateSample(vector_dim, vectors_groups);

            center_vectors = new Vector[GroupCount];
            VectorDim = vector_dim;

            Random random = new Random(0);
            var vectors = vectors_groups[0].ToArray();
            int vector_count = vectors.Length;

            // K-means++初期値決定シークエンス
            center_vectors[0] = vectors[random.Next(vector_count)];
            for(int group_index = 1; group_index < GroupCount; group_index++) {
                double dist_sum = 0;
                double[] dist_list = new double[vector_count];

                for(int vector_index = 0; vector_index < vector_count; vector_index++) {
                    double dist_min = double.PositiveInfinity;

                    for(int cluster_index = 0; cluster_index < group_index; cluster_index++) {
                        double dist = Vector.SquareDistance(vectors[vector_index], center_vectors[cluster_index]);
                        if(dist < dist_min) {
                            dist_min = dist;
                        }
                    }

                    dist_sum += dist_list[vector_index] = dist_min;
                }

                double r = random.NextDouble() * dist_sum;

                for(int vector_index = 0; vector_index < vector_count; vector_index++) {
                    r -= dist_list[vector_index];
                    if(r < 0) {
                        center_vectors[group_index] = vectors[vector_index];
                        break;
                    }
                    center_vectors[group_index] = vectors[vector_count - 1];
                }
            }

            // クラスタ割当て
            var labeled_vectors = vectors.Select((vector) => new LabelVector { Vector = vector, Label = NearestVector(vector) }).ToArray();
            bool ischanged_label = true;

            // k-mean収束ループ
            while(ischanged_label) {
                ischanged_label = false;

                for(int cluster_index = 0; cluster_index < center_vectors.Length; cluster_index++) {
                    center_vectors[cluster_index] = Vector.Zero(VectorDim);
                }

                int[] label_count = new int[center_vectors.Length];

                foreach(var vector in labeled_vectors) {
                    center_vectors[vector.Label] += vector.Vector;
                    label_count[vector.Label]++;
                }

                for(int cluster_index = 0; cluster_index < center_vectors.Length; cluster_index++) {
                    center_vectors[cluster_index] /= label_count[cluster_index];
                }

                for(int vector_index = 0; vector_index < vector_count; vector_index++) {
                    var labeled_vector = labeled_vectors[vector_index];

                    int label_old = labeled_vector.Label;
                    int label_new = NearestVector(labeled_vector.Vector);

                    if(label_old != label_new) {
                        ischanged_label = true;
                    }

                    labeled_vector.Label = label_new;
                }
            }
        }

        /// <summary>初期化</summary>
        public void Initialize() {
            center_vectors = null;
        }

        /// <summary>最近傍のベクトルを探索</summary>
        protected int NearestVector(Vector vector) {
            double dist_min = double.PositiveInfinity;
            int nearest_cluster_index = 0;

            for(int cluster_index = 0; cluster_index < center_vectors.Length; cluster_index++) {
                double dist = Vector.SquareDistance(vector, center_vectors[cluster_index]);
                if(dist < dist_min) {
                    dist_min = dist;
                    nearest_cluster_index = cluster_index;
                }
            }

            return nearest_cluster_index;
        }

        /// <summary>サンプルの正当性を検証</summary>
        private void ValidateSample(int vector_dim, List<Vector>[] vectors_groups) {
            if(vector_dim < 1) {
                throw new ArgumentException(nameof(vector_dim));
            }
            if(vectors_groups == null) {
                throw new ArgumentNullException(nameof(vectors_groups));
            }
            if(vectors_groups.Length != 1) {
                throw new ArgumentException(nameof(vectors_groups));
            }
            foreach(var vectors in vectors_groups) {
                if(vectors.Count < GroupCount) {
                    throw new ArgumentException(nameof(vectors_groups));
                }
                foreach(var vector in vectors) {
                    if(vector.Dim != vector_dim) {
                        throw new ArgumentException(nameof(vectors_groups));
                    }
                }
            }
        }
    }
}
