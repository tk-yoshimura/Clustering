using Clustering;
using System.Collections.ObjectModel;
using System.Drawing;
using System.Drawing.Imaging;

namespace ClusteringTest {
    public static class Vis {
        public static void Plot(KMeansClustering kmean, ReadOnlyCollection<Vector> vectors, string filepath) {
            Bitmap image = new(500, 500);

            Color[] colors = new Color[] { Color.FromArgb(255, 128, 128), Color.FromArgb(128, 128, 255), Color.FromArgb(128, 255, 128), Color.FromArgb(255, 255, 128) };

            for (int x, y = 0; y < image.Height; y++) {
                for (x = 0; x < image.Width; x++) {
                    double vx = (x - image.Width / 2) / (image.Width * 0.4);
                    double vy = (y - image.Height / 2) / (image.Height * 0.4);

                    int cluster_index = kmean.Classify((vx, vy));

                    image.SetPixel(x, y, colors[cluster_index]);
                }
            }

            using (Graphics g = Graphics.FromImage(image)) {
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;

                foreach (var vector in vectors) {
                    int x = (int)((4 * vector.X + 5) / 10 * image.Width);
                    int y = (int)((4 * vector.Y + 5) / 10 * image.Height);

                    int cluster_index = kmean.Classify(vector);
                    var color = colors[cluster_index];

                    g.DrawEllipse(new Pen(Color.Black, 2), new Rectangle(new Point(x - 3, y - 3), new Size(6, 6)));
                    g.FillEllipse(new SolidBrush(color), new Rectangle(new Point(x - 3, y - 3), new Size(6, 6)));
                }

                foreach (var vector in kmean.CenterVectors) {
                    int x = (int)((4 * vector.X + 5) / 10 * image.Width);
                    int y = (int)((4 * vector.Y + 5) / 10 * image.Height);

                    int cluster_index = kmean.Classify(vector);
                    var color = colors[cluster_index];

                    g.DrawRectangle(new Pen(Color.Black, 2), new Rectangle(new Point(x - 5, y - 5), new Size(10, 10)));
                    g.FillRectangle(new SolidBrush(color), new Rectangle(new Point(x - 5, y - 5), new Size(10, 10)));
                }
            }

            image.Save(filepath, ImageFormat.Png);
        }

        public static void Plot(SupportVectorMachine svm, ReadOnlyCollection<Vector> positive_vectors, ReadOnlyCollection<Vector> negative_vectors, string filepath) {
            Bitmap image = new(500, 500);

            Func<double, Color> color_func = (s) => {
                int d = (int)((Math.Abs(s) >= 1) ? 100 : (256 - 128 * Math.Abs(s)));

                if (Math.Abs(s) < 1.0e-4) {
                    return Color.FromArgb(255, 255, 255);
                }

                return (s < 0) ? Color.FromArgb(255, d, d) : Color.FromArgb(d, d, 255);
            };

            for (int x, y = 0; y < image.Height; y++) {
                for (x = 0; x < image.Width; x++) {
                    double vx = (x - image.Width / 2) / (image.Width * 0.4);
                    double vy = (y - image.Height / 2) / (image.Height * 0.4);

                    double s = svm.ClassifyRaw((vx, vy));

                    image.SetPixel(x, y, color_func(s));
                }
            }

            using (Graphics g = Graphics.FromImage(image)) {
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;

                foreach (var positive_vector in positive_vectors) {
                    int x = (int)((4 * positive_vector.X + 5) / 10 * image.Width);
                    int y = (int)((4 * positive_vector.Y + 5) / 10 * image.Height);

                    g.DrawEllipse(new Pen(Color.Black, 2), new Rectangle(new Point(x - 3, y - 3), new Size(6, 6)));
                    g.FillEllipse(new SolidBrush(Color.Blue), new Rectangle(new Point(x - 3, y - 3), new Size(6, 6)));
                }

                foreach (var negative_vector in negative_vectors) {
                    int x = (int)((4 * negative_vector.X + 5) / 10 * image.Width);
                    int y = (int)((4 * negative_vector.Y + 5) / 10 * image.Height);

                    g.DrawEllipse(new Pen(Color.Black, 2), new Rectangle(new Point(x - 3, y - 3), new Size(6, 6)));
                    g.FillEllipse(new SolidBrush(Color.Red), new Rectangle(new Point(x - 3, y - 3), new Size(6, 6)));
                }
            }

            image.Save(filepath, ImageFormat.Png);
        }
    }
}
