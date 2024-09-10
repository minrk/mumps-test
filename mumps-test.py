import argparse
import os
import json
import sys
import time
from pathlib import Path
from subprocess import run


try:
    import wurlitzer
    from mpi4py import MPI
    import ufl
    from dolfinx import default_scalar_type
    from dolfinx import fem, mesh
    from dolfinx.fem.petsc import LinearProblem
except ImportError:
    # not needed in outer env
    pass


def create_envs(blas):
    spec = [
        "python=3.12",
        "pip",
        "mpich",
        "fenics-dolfinx",
        f"blas=*=*{blas}",
        "mumps-mpi=5.7.3",
    ]
    for name, extra_spec in [
        ("before", ["mumps-mpi=5.7.3=*_0"]),
        ("omp", ["mumps-mpi=5.7.3=*_4"]),
        ("gemmt", ["mumps-mpi=5.7.3=*_104"]),
    ]:
        env_name = f"{name}-{blas}"
        run(
            ["mamba", "create", "-y", "-n", env_name, "-c", "minrk/label/mumps-gemmt"]
            + spec
            + extra_spec,
            check=True,
        )
        run(
            ["mamba", "run", "-n", env_name, "python3", "-mpip", "install", "wurlitzer"]
        )


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
        first_word, rest = rest.split(None, 1)

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
    problem = create_poisson_problem(size)
    with wurlitzer.pipes(stderr=None) as (stdout, _):
        tic = time.perf_counter()
        uh = problem.solve()
        toc = time.perf_counter()
    times = parse_mumps_times(stdout.getvalue())
    times["overall"] = toc - tic
    print(stdout.getvalue())
    return times


def inner_main(size: int, samples: int, fname: str):
    common_fields = {
        "env": Path(sys.prefix).name,
        "threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
        "procs": MPI.COMM_WORLD.size,
        "size": size,
    }
    print(f"Collecting {common_fields} to {fname}")
    for i in range(samples):
        times = time_solve_poisson(size)
        times.update(common_fields)
        if MPI.COMM_WORLD.rank == 0:
            print(times)
            if fname:
                print(f"Writing to {fname}")
                with open(fname, "a") as f:
                    f.write(json.dumps(times))
                    f.write("\n")


def run_one(env_name, samples, size, omp_threads, mpi_size):
    fname = f"{env_name}-omp{omp_threads}-mpi{mpi_size}.jsonl"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = str(omp_threads)
    env_bin = Path(env["CONDA_ROOT"]) / "envs" / env_name / "bin"
    assert env_bin.exists()
    env["PATH"] = f"{env_bin}:{env['PATH']}"
    print(f"Collecting {fname}")
    # don't use mamba/conda run, which suppresses output
    p = run(
        [
            "mpiexec",
            "-n",
            f"{mpi_size}",
            "python",
            __file__,
            f"--size={size}",
            f"--samples={samples}",
            "inner",
            "--out",
            fname,
        ],
        check=True,
        env=env,
    )


def collect(samples, size, threads, mpi_sizes, envs):
    for mpi_size in mpi_sizes:
        for omp_threads in threads:
            for env_name in envs:
                if omp_threads > 1 and env_name == "mumps-before":
                    continue
                run_one(env_name, samples, size, omp_threads, mpi_size)


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

    inner_parser.add_argument("--out", type=str)

    collect_parser.add_argument(
        "--envs",
        type=str,
        nargs="+",
        default=["mumps-before", "mumps-openmp", "mumps-gemmt"],
    )
    collect_parser.add_argument("--np", type=int, nargs="+", default=[1, 2, 4])
    collect_parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])

    envs_parser.add_argument("blas", nargs="?", default="openblas")

    args = parser.parse_args()
    if args.action == "inner":
        inner_main(size=args.size, samples=args.samples, fname=args.out)
    elif args.action == "collect":
        collect(
            size=args.size,
            samples=args.samples,
            threads=args.threads,
            mpi_sizes=args.np,
            envs=args.envs,
        )
    elif args.action == "envs":
        create_envs(blas=args.blas)
    else:
        raise ValueError("Specify an action: 'inner' or 'collect'")


if __name__ == "__main__":
    main()
