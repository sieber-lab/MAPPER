import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgb
from matplotlib.markers import MarkerStyle

from mapper.figures.growth_observations import CLASSES, DOSE_STYLES, individual_growth
from mapper.figures.overview_helpers import short_moa
from mapper.figures.style import read


class IndividualGrowthTest(unittest.TestCase):
    def test_dots_and_means_match_source(self):
        data = read("overview", "growth_profiles.csv")
        control = data.loc[data.bucket == 0]
        fig = individual_growth()
        try:
            self.assertEqual(len(fig.axes), 10)
            self.assertEqual(len({c for c, _ in DOSE_STYLES.values()}), len(DOSE_STYLES))
            self.assertEqual(len({m for _, m in DOSE_STYLES.values()}), len(DOSE_STYLES))
            self.assertEqual([t.get_text() for t in fig.legends[0].get_texts()],
                             [f"{dose}×" for dose in DOSE_STYLES] + ["Solvent control"])
            point_count = 0
            curve_count = 0
            for i, ax in enumerate(fig.axes):
                if i < len(CLASSES):
                    rows = data.loc[(data.mode_of_action.map(short_moa) == CLASSES[i][0]) & (data.bucket > 0)]
                else:
                    rows = control
                points = {c.get_gid(): c for c in ax.collections}
                self.assertTrue(all(isinstance(c, PathCollection) for c in ax.collections))
                self.assertEqual(set(points), {f"observations-dose-{d}" for d in rows.bucket.unique()})
                lines = {l.get_gid(): l for l in ax.lines}
                means = control.groupby("hour").od.mean()
                np.testing.assert_allclose(lines["mean-control"].get_xydata(), np.c_[means.index, means])
                previous_max = -np.inf
                for d, group in rows.groupby("bucket", sort=True):
                    collection = points[f"observations-dose-{d}"]
                    coords = np.asarray(collection.get_offsets())
                    np.testing.assert_array_equal(coords[:, 1], group.od.to_numpy())
                    shifts = coords[:, 0] - group.hour.to_numpy()
                    self.assertLess(np.max(np.abs(shifts)), .45)
                    self.assertGreater(shifts.min(), previous_max)
                    previous_max = shifts.max()
                    for compound in group.compound_id.unique():
                        self.assertLess(np.ptp(shifts[group.compound_id == compound]), 1e-12)
                    point_count += len(coords)
                    self.assertTrue(np.all(coords[:, 1] <= ax.get_ylim()[1]))
                    if d != 0:
                        curve_count += 1
                        avg = group.groupby("hour").od.mean()
                        line = lines[f"mean-dose-{d}"]
                        np.testing.assert_allclose(line.get_xydata(), np.c_[avg.index, avg])
                        colour, symbol = DOSE_STYLES[d]
                        np.testing.assert_allclose(collection.get_facecolors()[0, :3], to_rgb(colour))
                        self.assertEqual(line.get_color(), colour)
                        marker = MarkerStyle(symbol)
                        expected = marker.get_path().transformed(marker.get_transform())
                        np.testing.assert_allclose(collection.get_paths()[0].vertices, expected.vertices)
                self.assertEqual(ax.get_ylim(), fig.axes[0].get_ylim())
            self.assertEqual(point_count, len(data))
            self.assertEqual(curve_count, 38)
        finally:
            plt.close(fig)
