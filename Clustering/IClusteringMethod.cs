using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace Clustering {
    /// <summary>クラスタリング手法基本クラス</summary>
    public interface IClusteringMethod {

        /// <summary>単一サンプルを分類</summary>
        /// <param name="vector">サンプルベクタ</param>
        int Classify(Vector vector);

        /// <summary>複数サンプルを分類</summary>
        /// <param name="vectors">サンプルベクタ集合</param>
        IEnumerable<int> Classify(IEnumerable<Vector> vectors);

        /// <summary>学習</summary>
        /// <param name="vector_dim">サンプルベクタ次元数</param>
        /// <param name="vectors_groups">データクラスごとのサンプルベクタ集合</param>
        void Learn(params ReadOnlyCollection<Vector>[] vectors_groups);

        /// <summary>初期化</summary>
        void Initialize();

        /// <summary>サンプルベクタ次元数</summary>
        int VectorDim { get; }

        /// <summary>分類数</summary>
        int GroupCount { get; }
    }
}
