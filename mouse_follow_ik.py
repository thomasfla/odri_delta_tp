#!/usr/bin/env python3
"""Interactive IK viewer: move the mouse, the delta robot follows if reachable."""

from math import cos, sin

import matplotlib.pyplot as plt
import numpy as np

import delta_utils as du


def sample_workspace_from_joints(samples=140):
    q_values = np.linspace(-np.pi, np.pi, samples)
    points = []
    for q0 in q_values:
        for q1 in q_values:
            for positive in (True, False):
                p = du.fk_delta(np.array([q0, q1], dtype=float), positive=positive)
                if np.isfinite(p).all():
                    points.append(p)
    return np.array(points, dtype=float)


class DeltaMouseFollower:
    def __init__(self, ax):
        self.ax = ax

        self.left_line, = ax.plot([], [], "-", lw=2.0, color="tab:red")
        self.right_line, = ax.plot([], [], "-", lw=2.0, color="tab:blue")
        self.q0_arc, = ax.plot([], [], "--", lw=1.6, color="tab:red", alpha=0.9)
        self.q1_arc, = ax.plot([], [], "--", lw=1.6, color="tab:blue", alpha=0.9)
        self.target_pt, = ax.plot([], [], "x", ms=7, mew=2, color="black")
        self.ee_pt, = ax.plot([], [], "o", ms=5, color="tab:green")
        self.q0_label = ax.text(0.0, 0.0, "", color="tab:red", fontsize=9)
        self.q1_label = ax.text(0.0, 0.0, "", color="tab:blue", fontsize=9)
        self.text = ax.text(
            0.02,
            0.98,
            "Move mouse in plot area",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )
        self.jac_text = ax.text(
            0.02,
            0.02,
            r"$J(q)=\frac{d\mathbf{x}}{d\mathbf{q}}$"
            "\n"
            "[[., .], [., .]]",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )

        self.x_left_base = -du.d
        self.x_right_base = du.d
        self.arc_radius = 0.018

    def _arc_xy(self, x_center, theta):
        n = max(10, int(abs(theta) / 0.04))
        ts = np.linspace(0.0, theta, n)
        return x_center + self.arc_radius * np.cos(ts), self.arc_radius * np.sin(ts)

    def _draw_robot(self, q, target):
        x, y = target
        xl = self.x_left_base + du.l1 * cos(q[0])
        yl = du.l1 * sin(q[0])
        xr = self.x_right_base + du.l1 * cos(q[1])
        yr = du.l1 * sin(q[1])
        ee = du.fk_delta(q)
        jac = du.J(q)

        self.left_line.set_data([self.x_left_base, xl, ee[0]], [0.0, yl, ee[1]])
        self.right_line.set_data([self.x_right_base, xr, ee[0]], [0.0, yr, ee[1]])

        x_arc0, y_arc0 = self._arc_xy(self.x_left_base, q[0])
        x_arc1, y_arc1 = self._arc_xy(self.x_right_base, q[1])
        self.q0_arc.set_data(x_arc0, y_arc0)
        self.q1_arc.set_data(x_arc1, y_arc1)

        q0_mid = 0.5 * q[0]
        q1_mid = 0.5 * q[1]
        rlab = self.arc_radius + 0.008
        self.q0_label.set_position(
            (self.x_left_base + rlab * np.cos(q0_mid), rlab * np.sin(q0_mid))
        )
        self.q1_label.set_position(
            (self.x_right_base + rlab * np.cos(q1_mid), rlab * np.sin(q1_mid))
        )
        self.q0_label.set_text(f"q0={q[0]:+.2f}")
        self.q1_label.set_text(f"q1={q[1]:+.2f}")

        self.target_pt.set_data([x], [y])
        self.ee_pt.set_data([ee[0]], [ee[1]])
        self.target_pt.set_color("black")
        self.text.set_text(
            f"reachable\nx={x:.3f} y={y:.3f}\nq0={q[0]:.3f} q1={q[1]:.3f}"
        )
        self.jac_text.set_text(
            r"$J(q)=\frac{d\mathbf{x}}{d\mathbf{q}}$"
            "\n"
            f"[[{jac[0,0]:+.4f}, {jac[0,1]:+.4f}], [{jac[1,0]:+.4f}, {jac[1,1]:+.4f}]]"
        )

    def _draw_unreachable(self, target):
        x, y = target
        self.left_line.set_data([], [])
        self.right_line.set_data([], [])
        self.q0_arc.set_data([], [])
        self.q1_arc.set_data([], [])
        self.q0_label.set_text("")
        self.q1_label.set_text("")
        self.ee_pt.set_data([], [])
        self.target_pt.set_data([x], [y])
        self.target_pt.set_color("crimson")
        self.text.set_text(f"unreachable\nx={x:.3f} y={y:.3f}")
        self.jac_text.set_text(
            r"$J(q)=\frac{d\mathbf{x}}{d\mathbf{q}}$"
            "\n"
            "[[NaN, NaN], [NaN, NaN]]"
        )

    def on_move(self, event):
        if event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        target = np.array([event.xdata, event.ydata], dtype=float)
        q = du.ik_delta(target)

        if np.isnan(q).any():
            self._draw_unreachable(target)
        else:
            self._draw_robot(q, target)

        self.ax.figure.canvas.draw_idle()


def main():
    workspace = sample_workspace_from_joints(samples=140)

    fig, ax = plt.subplots(figsize=(7, 7))
    if workspace.size:
        ax.scatter(workspace[:, 0], workspace[:, 1], s=1, alpha=0.22, color="tab:green")
        xmin, xmax = workspace[:, 0].min(), workspace[:, 0].max()
        ymin, ymax = workspace[:, 1].min(), workspace[:, 1].max()
        mx = 0.05 * max(1e-9, xmax - xmin)
        my = 0.05 * max(1e-9, ymax - ymin)
        ax.set_xlim(xmin - mx, xmax + mx)
        ax.set_ylim(ymin - my, ymax + my)
    else:
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)

    du.plot_box()
    ax.set_title("Delta IK Mouse Follower")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    follower = DeltaMouseFollower(ax)
    fig.canvas.mpl_connect("motion_notify_event", follower.on_move)
    plt.show()


if __name__ == "__main__":
    main()
