using System;
using System.Collections.ObjectModel;
using System.Linq;

namespace Clustering {
    /// <summary>クラスタリング手法ユーティリティ</summary>
    public static class ClusteringMethodUtil {
        /// <summary>サンプルの正当性を検証</summary>
        public static void ValidateSample(int group_counts, ReadOnlyCollection<Vector>[] vectors_groups) {
            if (group_counts < 1) {
                throw new ArgumentOutOfRangeException(nameof(group_counts));
            }
            if (vectors_groups == null) {
                throw new ArgumentNullException(nameof(vectors_groups));
            }
            if (vectors_groups.Length != group_counts) {
                throw new ArgumentException("Mismatch group count.", nameof(vectors_groups));
            }

            if (vectors_groups[0].Count < 1) {
                throw new ArgumentException("Empty group.", nameof(vectors_groups));
            }

            int vector_dim = vectors_groups[0][0].Dim;

            foreach (var vectors in vectors_groups) {
                if (vectors.Count < 1) {
                    throw new ArgumentException("Empty group.", nameof(vectors_groups));
                }
                if (vectors.Any(vector => vector.Dim != vector_dim)) {
                    throw new ArgumentException("Mismatch vector dim.", nameof(vectors_groups));
                }
            }
        }
    }
}
