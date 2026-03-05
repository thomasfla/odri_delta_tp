#!/usr/bin/env python3
"""Simple tests + workspace plot for delta_utils."""

import matplotlib.pyplot as plt
import numpy as np

import delta_utils as du


ROUND_TRIP_TOL = 1e-6


def run_tests():
    reachable_points = [
        np.array([0.00, 0.02]),
        np.array([0.01, 0.04]),
        np.array([-0.01, 0.05]),
        np.array([0.00, 0.06]),
        np.array([-0.05, 0.10]),
    ]
    unreachable_points = [
        np.array([0.30, 0.00]),
        np.array([0.00, -0.30]),
        np.array([0.20, 0.20]),
    ]

    failures = 0
    print("Round-trip tests:")
    for p in reachable_points:
        q = du.ik_delta(p)
        if np.isnan(q).any():
            print(f"  [FAIL] IK returned NaN for {p}")
            failures += 1
            continue
        err = np.linalg.norm(du.fk_delta(q) - p)
        if err > ROUND_TRIP_TOL:
            print(f"  [FAIL] FK(IK(p)) error={err:.3e} at {p}")
            failures += 1
        else:
            print(f"  [OK]   {p} (error={err:.3e})")

    print("Unreachable-point tests:")
    for p in unreachable_points:
        q = du.ik_delta(p)
        if np.isnan(q).all():
            print(f"  [OK]   {p} -> NaN")
        else:
            print(f"  [FAIL] expected NaN for {p}, got {q}")
            failures += 1

    print("Jacobian tests:")
    for p in reachable_points:
        q = du.ik_delta(p)
        jac = du.J(q)
        if jac.shape == (2, 2) and np.isfinite(jac).all():
            print(f"  [OK]   J finite at {p}")
        else:
            print(f"  [FAIL] invalid J at {p}: {jac}")
            failures += 1

    return failures, reachable_points


def sample_workspace_from_joints(samples=180):
    q_values = np.linspace(-np.pi, np.pi, samples)
    points = []
    for q0 in q_values:
        for q1 in q_values:
            for positive in (True, False):
                p = du.fk_delta(np.array([q0, q1], dtype=float), positive=positive)
                if np.isfinite(p).all():
                    points.append(p)
    return np.array(points, dtype=float)


def plot_workspace(points, test_points):
    fig, ax = plt.subplots(figsize=(7, 7))

    if points.size:
        ax.scatter(points[:, 0], points[:, 1], s=1, alpha=0.25, color="tab:green")
        xmin, xmax = points[:, 0].min(), points[:, 0].max()
        ymin, ymax = points[:, 1].min(), points[:, 1].max()
        mx = 0.05 * max(1e-9, xmax - xmin)
        my = 0.05 * max(1e-9, ymax - ymin)
        ax.set_xlim(xmin - mx, xmax + mx)
        ax.set_ylim(ymin - my, ymax + my)

    plt.sca(ax)
    for p in test_points:
        q = du.ik_delta(p)
        if np.isnan(q).any():
            continue
        du.plot_delta(q, line="-")
        ax.plot(p[0], p[1], "ko", ms=4)

    du.plot_box()
    ax.set_title("Delta Workspace + Example Poses")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    plt.show()


def main():
    failures, test_points = run_tests()
    workspace_points = sample_workspace_from_joints(samples=180)
    plot_workspace(workspace_points, test_points)

    if failures:
        print(f"[RESULT] {failures} test(s) failed.")
        raise SystemExit(1)
    print("[RESULT] all tests passed.")


if __name__ == "__main__":
    main()
