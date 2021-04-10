using Algebra;
using System;
using System.Collections.ObjectModel;

namespace Clustering {
    /// <summary>逐次最小問題最適化法</summary>
    public class SequentialMinimalOptimization {
        private readonly int vectors;
        private readonly Vector[] inputs;
        private readonly double[] outputs;
        private readonly double cost;
        private double tolerance;
        private double epsilon;
        private double[] a, errors;
        private Random random;
        private readonly Func<Vector, Vector, double> kernel;

        /// <summary>コンストラクタ</summary>
        public SequentialMinimalOptimization(Vector[] inputs, double[] outputs, double cost, Func<Vector, Vector, double> kernel) {
            this.vectors = inputs.Length;
            this.inputs = inputs;
            this.outputs = outputs;
            this.cost = cost;
            this.kernel = kernel;
        }

        /// <summary>ベクトル重み</summary>
        public ReadOnlyCollection<double> VectorWeight => Array.AsReadOnly(a);

        /// <summary>バイアス</summary>
        public double Bias { get; private set; }

        /// <summary>最適化シークエンスを実行</summary>
        public void Optimize() {
            errors = new double[vectors];
            random = new Random(0);
            a = new double[vectors];
            Bias = 0;
            tolerance = 1e-3;
            epsilon = 1e-3;

            int changed_count = 0;
            bool examine_all = true;
            while (changed_count > 0 || examine_all) {
                changed_count = 0;
                if (examine_all) {
                    for (int i = 0; i < vectors; i++)
                        changed_count += ExamineExample(i);
                }
                else {
                    for (int i = 0; i < vectors; i++)
                        if (a[i] != 0 && a[i] != cost)
                            changed_count += ExamineExample(i);
                }
                if (examine_all)
                    examine_all = false;
                else if (changed_count == 0)
                    examine_all = true;
            }
        }

        /// <summary>i2と同時に最適化するi1を探索し2パラメータの最適化を実行</summary>
        private int ExamineExample(int i2) {
            int i1;

            Vector v2 = inputs[i2];
            double y2 = outputs[i2];
            double a2 = a[i2];
            double e2 = (a2 > 0 && a2 < cost) ? errors[i2] : ComputeF(v2) - y2;
            double r2 = y2 * e2;

            if (!(r2 < -tolerance && a2 < cost) && !(r2 > tolerance && a2 > 0))
                return 0;

            // 誤差値の差分を最大化するi1の探索
            i1 = i2;
            double maxerr = 0;
            for (int i = 0; i < inputs.Length; i++) {
                if (i == i2)
                    continue;

                if (a[i] > 0 && a[i] < cost) {
                    double e1 = errors[i];
                    double err = Math.Abs(e2 - e1);
                    if (err > maxerr) {
                        maxerr = err;
                        i1 = i;
                    }
                }
            }
            if (i1 != i2 && TakeStep(i1, i2))
                return 1;

            // i1をランダムに決定
            int start = random.Next(inputs.Length);
            for (i1 = start; i1 < inputs.Length; i1++) {
                if (a[i1] > 0 && a[i1] < cost)
                    if (TakeStep(i1, i2))
                        return 1;
            }
            for (i1 = 0; i1 < start; i1++) {
                if (a[i1] > 0 && a[i1] < cost)
                    if (TakeStep(i1, i2))
                        return 1;
            }

            start = random.Next(inputs.Length);
            for (i1 = start; i1 < inputs.Length; i1++) {
                if (TakeStep(i1, i2))
                    return 1;
            }
            for (i1 = 0; i1 < start; i1++) {
                if (TakeStep(i1, i2))
                    return 1;
            }

            return 0;
        }

        /// <summary>2パラメータの最適化を実行</summary>
        private bool TakeStep(int i1, int i2) {
            if (i1 == i2)
                return false;

            Vector v1 = inputs[i1], v2 = inputs[i2];
            double a1_old = a[i1], a2_old = a[i2];
            double y1 = outputs[i1], y2 = outputs[i2];

            double e1 = (a1_old > 0 && a1_old < cost) ? errors[i1] : ComputeF(v1) - y1;
            double e2 = (a2_old > 0 && a2_old < cost) ? errors[i2] : ComputeF(v2) - y2;

            double s = y1 * y2;

            // a1,a2を最適化
            double clip_l, clip_h;
            if (y1 != y2) {
                clip_l = Math.Max(0, a2_old - a1_old);
                clip_h = Math.Min(cost, cost + a2_old - a1_old);
            }
            else {
                clip_l = Math.Max(0, a2_old + a1_old - cost);
                clip_h = Math.Min(cost, a2_old + a1_old);
            }
            if (clip_l >= clip_h)
                return false;

            double k11, k22, k12, eta;
            k11 = kernel(v1, v1);
            k12 = kernel(v1, v2);
            k22 = kernel(v2, v2);
            eta = k11 + k22 - 2 * k12;

            double a1_new, a2_new;
            if (eta > 0) {
                a2_new = a2_old - y2 * (e2 - e1) / eta;
                if (a2_new < clip_l)
                    a2_new = clip_l;
                else if (a2_new > clip_h)
                    a2_new = clip_h;
            }
            else {
                double l1 = a1_old + s * (a2_old - clip_l);
                double h1 = a1_old + s * (a2_old - clip_h);
                double f1 = y1 * (e1 + Bias) - a1_old * k11 - s * a2_old * k12;
                double f2 = y2 * (e2 + Bias) - a2_old * k22 - s * a1_old * k12;
                double obj_l = -0.5 * l1 * l1 * k11 - 0.5 * clip_l * clip_l * k22 - s * clip_l * l1 * k12 - l1 * f1 - clip_l * f2;
                double obj_h = -0.5 * h1 * h1 * k11 - 0.5 * clip_h * clip_h * k22 - s * clip_h * h1 * k12 - h1 * f1 - clip_h * f2;
                if (obj_l > obj_h + epsilon)
                    a2_new = clip_l;
                else if (obj_l < obj_h - epsilon)
                    a2_new = clip_h;
                else
                    a2_new = a2_old;
            }

            if (Math.Abs(a2_new - a2_old) < epsilon * (a2_new + a2_old + epsilon))
                return false;

            a1_new = a1_old + s * (a2_old - a2_new);
            if (a1_new < 0) {
                a2_new += s * a1_new;
                a1_new = 0;
            }
            else if (a1_new > cost) {
                double d = a1_new - cost;
                a2_new += s * d;
                a1_new = cost;
            }

            // バイアスを更新
            double b1 = 0, b2 = 0;
            double delta_bias;
            double new_bias;
            if (a1_new > 0 && a1_new < cost) {
                new_bias = e1 + y1 * (a1_new - a1_old) * k11 + y2 * (a2_new - a2_old) * k12 + Bias;
            }
            else {
                if (a2_new > 0 && a2_new < cost) {
                    new_bias = e2 + y1 * (a1_new - a1_old) * k12 + y2 * (a2_new - a2_old) * k22 + Bias;
                }
                else {
                    b1 = e1 + y1 * (a1_new - a1_old) * k11 + y2 * (a2_new - a2_old) * k12 + Bias;
                    b2 = e2 + y1 * (a1_new - a1_old) * k12 + y2 * (a2_new - a2_old) * k22 + Bias;
                    new_bias = (b1 + b2) / 2;
                }
            }
            delta_bias = new_bias - Bias;
            Bias = new_bias;

            // 誤差値を更新
            double t1 = y1 * (a1_new - a1_old);
            double t2 = y2 * (a2_new - a2_old);
            for (int i = 0; i < inputs.Length; i++) {
                if (0 < a[i] && a[i] < cost) {
                    Vector vi = inputs[i];
                    errors[i] += t1 * kernel(v1, vi) + t2 * kernel(v2, vi) - delta_bias;
                }
            }
            errors[i1] = errors[i2] = 0;

            // a1,a2を格納
            a[i1] = a1_new;
            a[i2] = a2_new;

            return true;
        }

        /// <summary>F値を計算</summary>
        private double ComputeF(Vector v) {
            double sum = -Bias;
            for (int i = 0; i < inputs.Length; i++) {
                if (a[i] > 0)
                    sum += a[i] * outputs[i] * kernel(inputs[i], v);
            }
            return sum;
        }
    }
}
