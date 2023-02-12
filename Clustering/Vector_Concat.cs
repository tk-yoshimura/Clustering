using System;
using System.Collections.Generic;

namespace Clustering {
    ///<summary>ベクトルクラス</summary>
    public partial class Vector {
        /// <summary>結合</summary>
        public static Vector Concat(params object[] blocks) {
            List<double> v = new();

            foreach (object obj in blocks) {
                if (obj is Vector vector) {
                    v.AddRange(vector.v);
                }
                else if (obj is double vd) {
                    v.Add(vd);
                }
                else if (obj is int vi) {
                    v.Add(vi);
                }
                else if (obj is long vl) {
                    v.Add(vl);
                }
                else if (obj is float vf) {
                    v.Add(vf);
                }
                else {
                    throw new ArgumentException($"unsupported type '{obj.GetType().Name}'", nameof(blocks));
                }
            }

            return new Vector(v);
        }
    }
}
