import argparse
import json
import math
import os
import time

import numpy as np

from fem_visualization import render_manufactured_validation_visuals


FIELD_NAMES = ["ux", "uy", "sxx", "syy", "sxy", "lambda", "mu"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path, payload):
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {key: to_serializable(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(item) for item in obj]
        return obj

    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, ensure_ascii=False)


def save_text(path, text):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def relative_l2(pred, ref):
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    denom = np.linalg.norm(ref.reshape(-1))
    if denom <= 1e-12:
        return float(np.linalg.norm(pred.reshape(-1)))
    return float(np.linalg.norm((pred - ref).reshape(-1)) / denom)


def make_grid(nx, ny):
    xs = np.linspace(0.0, 1.0, int(nx))
    ys = np.linspace(0.0, 1.0, int(ny))
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    return points, xx, yy


def manufactured_displacement_xy(x, y):
    ux = x**2 * (1.0 - x) ** 2 * y * (1.0 - y)
    uy = -x * (1.0 - x) * y**2 * (1.0 - y) ** 2
    return ux, uy


def manufactured_strain_xy(x, y):
    exx = -4.0 * x**3 * y**2 + 4.0 * x**3 * y + 6.0 * x**2 * y**2 - 6.0 * x**2 * y - 2.0 * x * y**2 + 2.0 * x * y
    eyy = 4.0 * x**2 * y**3 - 6.0 * x**2 * y**2 + 2.0 * x**2 * y - 4.0 * x * y**3 + 6.0 * x * y**2 - 2.0 * x * y
    exy = (
        -x**4 * y
        + 0.5 * x**4
        + 2.0 * x**3 * y
        - x**3
        - x**2 * y
        + 0.5 * x**2
        + x * y**4
        - 2.0 * x * y**3
        + x * y**2
        - 0.5 * y**4
        + y**3
        - 0.5 * y**2
    )
    return exx, eyy, exy


def manufactured_state(points, lambda_value, mu_value):
    points = np.asarray(points, dtype=float)
    x = points[:, 0:1]
    y = points[:, 1:2]
    ux, uy = manufactured_displacement_xy(x, y)
    exx, eyy, exy = manufactured_strain_xy(x, y)
    lmbd = np.full_like(x, float(lambda_value), dtype=float)
    mu = np.full_like(x, float(mu_value), dtype=float)
    trace = exx + eyy
    sxx = lmbd * trace + 2.0 * mu * exx
    syy = lmbd * trace + 2.0 * mu * eyy
    sxy = 2.0 * mu * exy
    return np.hstack((ux, uy, sxx, syy, sxy, lmbd, mu))


def manufactured_body_force_xy(x, y, lambda_value, mu_value):
    lam = float(lambda_value)
    mu = float(mu_value)
    fx = (
        12.0 * lam * x**2 * y**2
        - 12.0 * lam * x**2 * y
        - 8.0 * lam * x * y**3
        + 8.0 * lam * x * y
        + 4.0 * lam * y**3
        - 4.0 * lam * y**2
        + 2.0 * mu * x**4
        - 4.0 * mu * x**3
        + 24.0 * mu * x**2 * y**2
        - 24.0 * mu * x**2 * y
        + 2.0 * mu * x**2
        - 8.0 * mu * x * y**3
        - 12.0 * mu * x * y**2
        + 20.0 * mu * x * y
        + 4.0 * mu * y**3
        - 2.0 * mu * y**2
        - 2.0 * mu * y
    )
    fy = (
        8.0 * lam * x**3 * y
        - 4.0 * lam * x**3
        - 12.0 * lam * x**2 * y**2
        + 4.0 * lam * x**2
        + 12.0 * lam * x * y**2
        - 8.0 * lam * x * y
        + 8.0 * mu * x**3 * y
        - 4.0 * mu * x**3
        - 24.0 * mu * x**2 * y**2
        + 12.0 * mu * x**2 * y
        + 2.0 * mu * x**2
        + 24.0 * mu * x * y**2
        - 20.0 * mu * x * y
        + 2.0 * mu * x
        - 2.0 * mu * y**4
        + 4.0 * mu * y**3
        - 2.0 * mu * y**2
    )
    return fx, fy


def build_mesh(nx, ny):
    from skfem import MeshTri

    xs = np.linspace(0.0, 1.0, int(nx))
    ys = np.linspace(0.0, 1.0, int(ny))
    return MeshTri.init_tensor(xs, ys).with_boundaries(
        {
            "left": lambda x: np.isclose(x[0], 0.0),
            "right": lambda x: np.isclose(x[0], 1.0),
            "bottom": lambda x: np.isclose(x[1], 0.0),
            "top": lambda x: np.isclose(x[1], 1.0),
        }
    )


def build_scalar_element(order):
    from skfem.element import ElementTriP1, ElementTriP2

    if int(order) == 1:
        return ElementTriP1()
    if int(order) == 2:
        return ElementTriP2()
    raise ValueError(f"Unsupported element_order: {order}")


def solve_manufactured_case(mesh_nx, mesh_ny, element_order, lambda_value, mu_value):
    from skfem import Basis, BilinearForm, LinearForm, asm, condense, solve
    from skfem.element import ElementVector
    from skfem.helpers import ddot, sym_grad, trace

    mesh = build_mesh(mesh_nx, mesh_ny)
    scalar_element = build_scalar_element(element_order)
    vector_element = ElementVector(scalar_element)
    vector_basis = Basis(mesh, vector_element)
    scalar_basis = Basis(mesh, scalar_element, quadrature=vector_basis.quadrature)

    @BilinearForm
    def elasticity(u, v, w):
        return float(lambda_value) * trace(sym_grad(u)) * trace(sym_grad(v)) + 2.0 * float(mu_value) * ddot(sym_grad(u), sym_grad(v))

    @LinearForm
    def body_force(v, w):
        fx, fy = manufactured_body_force_xy(w.x[0], w.x[1], lambda_value, mu_value)
        return fx * v[0] + fy * v[1]

    stiffness = asm(elasticity, vector_basis)
    rhs = asm(body_force, vector_basis)
    prescribed = vector_basis.zeros()

    boundary_dofs = vector_basis.get_dofs()
    boundary_flat = boundary_dofs.flatten()
    dof_locations = vector_basis.doflocs

    ux_dofs = np.asarray(boundary_dofs.nodal["u^1"], dtype=int)
    uy_dofs = np.asarray(boundary_dofs.nodal["u^2"], dtype=int)
    ux_points = np.column_stack((dof_locations[0, ux_dofs], dof_locations[1, ux_dofs]))
    uy_points = np.column_stack((dof_locations[0, uy_dofs], dof_locations[1, uy_dofs]))
    prescribed_ux, _ = manufactured_displacement_xy(ux_points[:, 0:1], ux_points[:, 1:2])
    _, prescribed_uy = manufactured_displacement_xy(uy_points[:, 0:1], uy_points[:, 1:2])
    prescribed[ux_dofs] = prescribed_ux[:, 0]
    prescribed[uy_dofs] = prescribed_uy[:, 0]

    solution = solve(*condense(stiffness, rhs, x=prescribed, D=boundary_flat))

    discrete_field = vector_basis.interpolate(solution)
    exx = discrete_field.grad[0, 0]
    eyy = discrete_field.grad[1, 1]
    exy = 0.5 * (discrete_field.grad[0, 1] + discrete_field.grad[1, 0])
    trace_strain = exx + eyy
    sxx = float(lambda_value) * trace_strain + 2.0 * float(mu_value) * exx
    syy = float(lambda_value) * trace_strain + 2.0 * float(mu_value) * eyy
    sxy = 2.0 * float(mu_value) * exy
    sxx_projection = scalar_basis.project(sxx)
    syy_projection = scalar_basis.project(syy)
    sxy_projection = scalar_basis.project(sxy)

    return {
        "mesh": mesh,
        "vector_basis": vector_basis,
        "scalar_basis": scalar_basis,
        "solution": solution,
        "sxx_projection": sxx_projection,
        "syy_projection": syy_projection,
        "sxy_projection": sxy_projection,
    }


def evaluate_state(solution_bundle, points, lambda_value, mu_value):
    points = np.asarray(points, dtype=float)
    coords = points.T
    displacement = solution_bundle["vector_basis"].interpolator(solution_bundle["solution"])(coords)
    sxx = solution_bundle["scalar_basis"].interpolator(solution_bundle["sxx_projection"])(coords)
    syy = solution_bundle["scalar_basis"].interpolator(solution_bundle["syy_projection"])(coords)
    sxy = solution_bundle["scalar_basis"].interpolator(solution_bundle["sxy_projection"])(coords)
    lmbd = np.full((len(points), 1), float(lambda_value), dtype=float)
    mu = np.full((len(points), 1), float(mu_value), dtype=float)
    return np.column_stack(
        (
            displacement[0],
            displacement[1],
            np.asarray(sxx, dtype=float),
            np.asarray(syy, dtype=float),
            np.asarray(sxy, dtype=float),
            lmbd[:, 0],
            mu[:, 0],
        )
    )


def compute_metrics(prediction, truth):
    prediction = np.asarray(prediction, dtype=float)
    truth = np.asarray(truth, dtype=float)
    per_field = {}
    for field_index, field_name in enumerate(FIELD_NAMES):
        per_field[field_name] = relative_l2(prediction[:, field_index], truth[:, field_index])
    return {
        "global_relative_l2_by_field": per_field,
        "global_displacement_relative_l2": relative_l2(prediction[:, :2], truth[:, :2]),
        "global_stress_relative_l2": relative_l2(prediction[:, 2:5], truth[:, 2:5]),
        "global_material_relative_l2_lambda_mu": relative_l2(prediction[:, 5:7], truth[:, 5:7]),
    }


def estimate_order(coarse_error, fine_error, coarse_h, fine_h):
    coarse_error = float(coarse_error)
    fine_error = float(fine_error)
    if coarse_error <= 0.0 or fine_error <= 0.0:
        return None
    if coarse_h <= 0.0 or fine_h <= 0.0 or math.isclose(coarse_h, fine_h):
        return None
    return float(math.log(coarse_error / fine_error) / math.log(coarse_h / fine_h))


def run_mesh(save_dir, mesh_nx, mesh_ny, element_order, eval_nx, eval_ny, lambda_value, mu_value):
    start = time.time()
    bundle = solve_manufactured_case(mesh_nx, mesh_ny, element_order, lambda_value, mu_value)
    eval_points, xx, yy = make_grid(eval_nx, eval_ny)
    prediction = evaluate_state(bundle, eval_points, lambda_value, mu_value)
    truth = manufactured_state(eval_points, lambda_value, mu_value)
    metrics = compute_metrics(prediction, truth)
    elapsed = float(time.time() - start)

    mesh_dir = ensure_dir(os.path.join(save_dir, f"mesh_{int(mesh_nx)}x{int(mesh_ny)}"))
    np.savez(
        os.path.join(mesh_dir, "grid_comparison.npz"),
        points=eval_points,
        prediction=prediction,
        truth=truth,
        xx=xx,
        yy=yy,
        field_names=np.asarray(FIELD_NAMES),
    )
    save_json(
        os.path.join(mesh_dir, "metrics.json"),
        {
            **metrics,
            "mesh_nx": int(mesh_nx),
            "mesh_ny": int(mesh_ny),
            "element_order": int(element_order),
            "elapsed_seconds": elapsed,
        },
    )
    save_text(
        os.path.join(mesh_dir, "summary.txt"),
        "\n".join(
            [
                f"mesh={int(mesh_nx)}x{int(mesh_ny)}",
                f"element_order={int(element_order)}",
                f"elapsed_seconds={elapsed:.3f}",
                f"global_displacement_relative_l2={metrics['global_displacement_relative_l2']:.8e}",
                f"global_stress_relative_l2={metrics['global_stress_relative_l2']:.8e}",
            ]
        )
        + "\n",
    )
    return {
        "mesh_nx": int(mesh_nx),
        "mesh_ny": int(mesh_ny),
        "element_order": int(element_order),
        "elapsed_seconds": elapsed,
        **metrics,
    }


def parse_mesh_list(raw):
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if len(values) < 2:
        raise ValueError("mesh_list must contain at least two mesh sizes.")
    return values


def parse_args():
    parser = argparse.ArgumentParser(description="Validate FEM solver against a manufactured solution.")
    parser.add_argument("--mesh_list", type=str, default="41,81,161")
    parser.add_argument("--element_order", type=int, choices=[1, 2], default=2)
    parser.add_argument("--eval_nx", type=int, default=161)
    parser.add_argument("--eval_ny", type=int, default=161)
    parser.add_argument("--lambda_value", type=float, default=1.0)
    parser.add_argument("--mu_value", type=float, default=0.7)
    parser.add_argument("--save_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = os.path.abspath(args.save_dir)
    ensure_dir(save_dir)
    mesh_sizes = parse_mesh_list(args.mesh_list)

    per_mesh = []
    for mesh_size in mesh_sizes:
        print(f"[run] mesh={mesh_size}x{mesh_size}")
        per_mesh.append(
            run_mesh(
                save_dir=save_dir,
                mesh_nx=mesh_size,
                mesh_ny=mesh_size,
                element_order=args.element_order,
                eval_nx=args.eval_nx,
                eval_ny=args.eval_ny,
                lambda_value=args.lambda_value,
                mu_value=args.mu_value,
            )
        )

    convergence = []
    for coarse, fine in zip(per_mesh[:-1], per_mesh[1:]):
        coarse_h = 1.0 / float(coarse["mesh_nx"] - 1)
        fine_h = 1.0 / float(fine["mesh_nx"] - 1)
        convergence.append(
            {
                "from_mesh": f"{coarse['mesh_nx']}x{coarse['mesh_ny']}",
                "to_mesh": f"{fine['mesh_nx']}x{fine['mesh_ny']}",
                "displacement_order_estimate": estimate_order(
                    coarse["global_displacement_relative_l2"],
                    fine["global_displacement_relative_l2"],
                    coarse_h,
                    fine_h,
                ),
                "stress_order_estimate": estimate_order(
                    coarse["global_stress_relative_l2"],
                    fine["global_stress_relative_l2"],
                    coarse_h,
                    fine_h,
                ),
            }
        )

    summary = {
        "problem": "manufactured_solution_single_material",
        "model": "small-strain isotropic linear elasticity, plane strain",
        "forcing": "nonzero manufactured body force",
        "lambda_value": float(args.lambda_value),
        "mu_value": float(args.mu_value),
        "element_order": int(args.element_order),
        "per_mesh": per_mesh,
        "convergence": convergence,
    }
    save_json(os.path.join(save_dir, "manufactured_summary.json"), summary)
    save_text(
        os.path.join(save_dir, "manufactured_summary.txt"),
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    try:
        render_manufactured_validation_visuals(save_dir)
    except Exception as exc:
        print(f"[warn] failed to render manufactured FEM visuals: {exc}")

    print("=" * 80)
    print("Manufactured FEM validation finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
