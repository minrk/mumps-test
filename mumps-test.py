import argparse
import os
import json
import sys
import time
from multiprocessing import cpu_count
from pathlib import Path
from subprocess import run

from scipy.sparse.linalg import LaplacianNd
import numpy as np

import typing

if typing.TYPE_CHECKING:
    from dolfinx.fem.petsc import LinearProblem

here = Path(__file__).parent


def parse_mumps_times(output: str) -> dict[str, float]:
    elapsed_time_in = "elapsed time"
    times = {}
    for line in output.splitlines():
        line = line.strip()
        if not line.lower().startswith(elapsed_time_in):
            if "time" in line:
                print("no match", line, file=sys.stderr)
            continue
        rest = line[len(elapsed_time_in) :]
        _first_word, rest = rest.split(None, 1)

        label, _, rhs = rest.partition("=")
        label = label.strip()
        try:
            value = float(rhs.strip())
        except ValueError:
            print(f"Not a float: {rhs!r} in {line!r}")
        else:
            times[label] = value
    return times


def create_poisson_problem(size: int) -> "LinearProblem":
    from mpi4py import MPI
    import ufl
    from dolfinx import default_scalar_type
    from dolfinx import fem, mesh
    from dolfinx.fem.petsc import LinearProblem

    # create a poisson problem
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD, size, size, mesh.CellType.quadrilateral
    )

    V = fem.functionspace(domain, ("Lagrange", 1))

    uD = fem.Function(V)
    uD.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)

    # Create facet to cell connectivity required to determine boundary facets
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)

    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f = fem.Constant(domain, default_scalar_type(-6))

    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "cholesky",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_4": "3",
            # "mat_mumps_use_omp_threads": os.environ.get("OMP_NUM_THREADS", "1"),
        },
    )
    return problem


def time_solve_poisson(size: int) -> float:
    """Time solving a poisson problem of a given size"""
    import wurlitzer

    problem = create_poisson_problem(size)
    with wurlitzer.pipes(stderr=None) as (stdout, _):
        tic = time.perf_counter()
        _uh = problem.solve()
        toc = time.perf_counter()
    times = parse_mumps_times(stdout.getvalue())
    times["analyze"] = times.pop("analysis driver", 0)
    times["factorize"] = times.pop("factorization driver", 0)
    times["solve"] = times.pop("solve driver", 0)
    times["overall"] = toc - tic
    # print(stdout.getvalue())
    return times


def time_mumps_python(size: int) -> float:
    """Time a simple 2D Laplacian solve with python-mumps"""
    import mumps

    lap = LaplacianNd(grid_shape=(size, size))
    A = lap.tosparse()
    # sin(x) * sin(y)
    x = np.sin(np.linspace(0, np.pi, size))
    b = np.outer(x, x).flatten()
    times = {}
    with mumps.Context() as ctx:
        ctx.set_matrix(A, overwrite_a=True, symmetric=True)
        start = time.perf_counter()
        ctx.analyze(ordering="metis")
        ctx.factor(reuse_analysis=True)
        times["analyze"] = ctx.analysis_stats.time
        times["factorize"] = ctx.factor_stats.time
        tic = time.perf_counter()
        ctx.solve(b)
        toc = time.perf_counter()
        times["solve"] = toc - tic
        times["overall"] = toc - start

    return times


def inner_main(size: int, samples: int, fname: str, kind: str):
    from mpi4py import MPI

    name = os.environ.get("PIXI_ENVIRONMENT_NAME")
    if not name:
        name = os.environ.get("CONDA_DEFAULT_ENV")
    if not name:
        name = Path(sys.prefix).name
    common_fields = {
        "env": name,
        "threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
        "size": size,
        "kind": kind,
    }
    if kind == "fenics":
        common_fields["procs"] = MPI.COMM_WORLD.size
        rank = MPI.COMM_WORLD.rank
    else:
        common_fields["procs"] = 1
        rank = 0
    print(f"Collecting {common_fields} to {fname}")
    for i in range(samples):
        if kind == "fenics":
            times = time_solve_poisson(size)
        else:
            times = time_mumps_python(size)
        times.update(common_fields)
        if rank == 0:
            print(times)
            if fname:
                print(f"Writing to {fname}")
                with open(fname, "a") as f:
                    f.write(json.dumps(times))
                    f.write("\n")


def run_one(env_name, samples, size, omp_threads, mpi_size, kind):
    fname = f"{env_name}-omp{omp_threads}-mpi{mpi_size}.jsonl"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = str(omp_threads)

    # if parallelizing with OMP or mpi,
    # set other thread counts to 1 to avoid oversubscribing
    if omp_threads > 1 or mpi_size > 1:
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        # for Accelerate, probably has no effect on arm mac
        env["VECLIB_MAXIMUM_THREADS"] = "1"
    print(f"Collecting {fname}")
    # don't use mamba/conda run, which suppresses output
    if kind == "fenics":
        prefix = ["mpiexec", "-n", f"{mpi_size}"]
    else:
        prefix = []
    _p = run(
        ["pixi", "run", "--frozen", "--environment", env_name]
        + prefix
        + [
            "python",
            __file__,
            f"--kind={kind}",
            f"--size={size}",
            f"--samples={samples}",
            "inner",
            "--out",
            fname,
        ],
        check=True,
        env=env,
    )

def core_count():
    if os.environ.get("CPU_LIMIT"):
        return int(float(os.environ["CPU_LIMIT"]))
    else:
        return cpu_count()


def collect(samples, size, threads, mpi_sizes, envs, kinds):
    for mpi_size in mpi_sizes:
        for kind in kinds:
            if kind == "mumps" and mpi_size > 1:
                continue
            for omp_threads in threads:
                if omp_threads * mpi_size > core_count():
                    continue
                for env_name in envs:
                    if omp_threads > 1 and "before" in env_name:
                        continue
                    run_one(env_name, samples, size, omp_threads, mpi_size, kind)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="command")
    collect_parser = subparsers.add_parser("collect")
    collect_parser.set_defaults(action="collect")
    inner_parser = subparsers.add_parser("inner")
    inner_parser.set_defaults(action="inner")
    envs_parser = subparsers.add_parser("envs")
    envs_parser.set_defaults(action="envs")

    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--kind", type=str, default="mumps", choices=["mumps", "fenics", "*"]
    )

    inner_parser.add_argument("--out", type=str)

    default_blas = "mkl" if sys.platform == "linux" else "openblas"

    collect_parser.add_argument(
        "--envs",
        type=str,
        nargs="+",
    )
    collect_parser.add_argument(
        "--blas",
        type=str,
        choices=["openblas", "mkl", "accelerate"],
    )
    collect_parser.add_argument("--np", type=int, nargs="+", default=[1, 2, 4, 8])
    collect_parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])

    args = parser.parse_args()
    if args.action == "inner":
        inner_main(size=args.size, samples=args.samples, fname=args.out, kind=args.kind)
    elif args.action == "collect":
        if args.kind =='*':
            kinds = ["mumps", "fenics"]
        else:
            kinds = [args.kind]
        if not args.envs:
            blas = args.blas or default_blas
            args.envs = [f"{build}-{blas}" for build in ("before", "omp", "gemmt")]
        elif args.envs == ["*"]:
            blases = ["openblas"]
            if sys.platform == "linux":
                blases.append("mkl")
            if sys.platform == "Darwin":
                blases.append("accelerate")
            args.envs = [f"{build}-{blas}" for build in ("before", "omp", "gemmt") for blas in blases]
        
        try:
            # there is no gemmt build for accelerate
            args.envs.remove("accelerate-gemmt")
        except IndexError:
            pass
        collect(
            size=args.size,
            samples=args.samples,
            threads=args.threads,
            mpi_sizes=args.np,
            envs=args.envs,
            kinds=kinds,
        )
    else:
        raise ValueError("Specify an action: 'inner' or 'collect'")


if __name__ == "__main__":
    main()
