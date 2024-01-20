using System;

namespace Clustering {
    ///<summary>ベクトルクラス</summary>
    public partial class Vector {
        public static implicit operator Vector((double x, double y) v) {
            return new Vector([v.x, v.y], cloning: false);
        }

        public void Deconstruct(out double x, out double y) {
            if (Dim != 2) {
                throw new InvalidOperationException($"vector dim={Dim}");
            }

            (x, y) = (v[0], v[1]);
        }

        public static implicit operator Vector((double x, double y, double z) v) {
            return new Vector([v.x, v.y, v.z], cloning: false);
        }

        public void Deconstruct(out double x, out double y, out double z) {
            if (Dim != 3) {
                throw new InvalidOperationException($"vector dim={Dim}");
            }

            (x, y, z) = (v[0], v[1], v[2]);
        }

        public static implicit operator Vector((double x, double y, double z, double w) v) {
            return new Vector([v.x, v.y, v.z, v.w], cloning: false);
        }

        public void Deconstruct(out double x, out double y, out double z, out double w) {
            if (Dim != 4) {
                throw new InvalidOperationException($"vector dim={Dim}");
            }

            (x, y, z, w) = (v[0], v[1], v[2], v[3]);
        }

        public static implicit operator Vector((double e0, double e1, double e2, double e3, double e4) v) {
            return new Vector([v.e0, v.e1, v.e2, v.e3, v.e4], cloning: false);
        }

        public void Deconstruct(out double e0, out double e1, out double e2, out double e3, out double e4) {
            if (Dim != 5) {
                throw new InvalidOperationException($"vector dim={Dim}");
            }

            (e0, e1, e2, e3, e4) = (v[0], v[1], v[2], v[3], v[4]);
        }

        public static implicit operator Vector((double e0, double e1, double e2, double e3, double e4, double e5) v) {
            return new Vector([v.e0, v.e1, v.e2, v.e3, v.e4, v.e5], cloning: false);
        }

        public void Deconstruct(out double e0, out double e1, out double e2, out double e3, out double e4, out double e5) {
            if (Dim != 6) {
                throw new InvalidOperationException($"vector dim={Dim}");
            }

            (e0, e1, e2, e3, e4, e5) = (v[0], v[1], v[2], v[3], v[4], v[5]);
        }

        public static implicit operator Vector((double e0, double e1, double e2, double e3, double e4, double e5, double e6) v) {
            return new Vector([v.e0, v.e1, v.e2, v.e3, v.e4, v.e5, v.e6], cloning: false);
        }

        public void Deconstruct(out double e0, out double e1, out double e2, out double e3, out double e4, out double e5, out double e6) {
            if (Dim != 7) {
                throw new InvalidOperationException($"vector dim={Dim}");
            }

            (e0, e1, e2, e3, e4, e5, e6) = (v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
        }

        public static implicit operator Vector((double e0, double e1, double e2, double e3, double e4, double e5, double e6, double e7) v) {
            return new Vector([v.e0, v.e1, v.e2, v.e3, v.e4, v.e5, v.e6, v.e7], cloning: false);
        }

        public void Deconstruct(out double e0, out double e1, out double e2, out double e3, out double e4, out double e5, out double e6, out double e7) {
            if (Dim != 8) {
                throw new InvalidOperationException($"vector dim={Dim}");
            }

            (e0, e1, e2, e3, e4, e5, e6, e7) = (v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
        }
    }
}
