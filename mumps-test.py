import argparse
import os
import json
import shlex
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

try:
    import mumps
    import scipy.sparse as sp
    import numpy as np
except ImportError:
    pass

here = Path(__file__).parent


def define_envs():
    builds = {
        "omp": 3,
        "before": 2,
        "gemmt": 104,
    }

    for blas in ("openblas", "mkl"):
        if blas == "mkl":
            platforms = ["linux-64"]
        else:
            platforms = ["linux-64", "osx-arm64"]
        base_spec = [
            "python=3.12",
            "pip",
            f"blas=*=*{blas}",
        ]
        for kind in ("fenics", "mumps"):
            if kind == "fenics":
                mumps_pkg = "mumps-mpi"
                test_spec = ["fenics-dolfinx", "mpich", "wurlitzer"]
            else:
                mumps_pkg = "mumps-seq"
                test_spec = ["scipy", "python-mumps"]

            for name, build_number in builds.items():
                channels = ["conda-forge"]
                pin_spec = [f"{mumps_pkg}=5.7.3=*_{build_number}"]
                if name == "gemmt":
                    channels.insert(0, "minrk/label/mumps-gemmt")
                env_name = f"{kind}-{name}-{blas}"
                env_path = here / "envs" / env_name
                env_path.mkdir(exist_ok=True, parents=True)
                
                pixi_toml = env_path / "pixi.toml"
                if not pixi_toml.exists():
                    with pixi_toml.open("w") as f:
                        f.write("[project]\n")
                        f.write(f'name = "{env_name}"\n')
                        f.write(f"channels = {channels}\n")
                        f.write(f"platforms = {platforms}\n")

                def pixi(cmd):
                    cmd_s = shlex.join(cmd)
                    print(f"[{env_path}]> pixi {cmd_s}")
                    return run(["pixi"] + cmd, cwd=env_path, check=True)

                # if not (env_path / "pixi.toml").exists():
                #     platform_args = [f"--platform={platform}" for platform in platforms]
                #     channel_args = [f"--channel={channel}" for channel in channels]
                #     pixi(["init"] + platform_args + channel_args)

                pixi(["add", "--no-install"] + base_spec + test_spec + pin_spec)


def create_envs(blas, kind="mumps"):
    spec = [
        "python=3.12",
        "pip",
        f"blas=*=*{blas}",
    ]
    if kind == "fenics":
        mumps_pkg = "mumps-mpi"
        spec.extend([
            "mpich",
            "fenics-dolfinx",
            "wurlitzer",
        ])
    else:
        mumps_pkg = "mumps-seq"
        spec.extend([
            "scipy",
            "python-mumps",
        ])

    for name, extra_spec in [
        ("before", [f"{mumps_pkg}=5.7.3=*_2"]),
        ("omp", [f"{mumps_pkg}=5.7.3=*_4"]),
        ("gemmt", [f"{mumps_pkg}=5.7.3=*_104"]),
    ]:
        env_name = f"{kind}-{name}-{blas}"
        run(
            ["mamba", "create", "-y", "-n", env_name, "-c", "minrk/label/mumps-gemmt"]
            + spec
            + extra_spec,
            check=True,
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
    # print(stdout.getvalue())
    return times


def time_mumps_python(size: int) -> float:
    sq_size = size * size
    A = sp.diags_array(
        [
            -30 / 12 * np.ones(sq_size),
            16 / 12 * np.ones(sq_size - 1),
            16 / 12 * np.ones(sq_size - 1),
            # -1 / 12 * np.ones(size - 2),
            # -1 / 12 * np.ones(size - 2),
        ],
        offsets=[0, -1, 1],  # , -2, 2],
    )
    b = np.linspace(0, 1, sq_size)
    times = {}
    with mumps.Context() as ctx:
        ctx.set_matrix(A, overwrite_a=True, symmetric=True)
        ctx.analyze(ordering="metis")
        ctx.factor(reuse_analysis=True)
        times["analyze"] = ctx.analysis_stats.time
        times["factorize"] = ctx.factor_stats.time
        tic = time.perf_counter()
        ctx.solve(b)
        toc = time.perf_counter()
        times["solve"] = toc - tic

    return times


def inner_main(size: int, samples: int, fname: str, kind: str):
    name = os.environ.get("PIXI_PROJECT_NAME")
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
    env_path = here / "envs" / env_name
    print(f"Running in {env_path}")
    assert env_path.exists()
    print(f"Collecting {fname}")
    # don't use mamba/conda run, which suppresses output
    if kind == "fenics":
        prefix = ["mpiexec", "-n", f"{mpi_size}"]
    else:
        prefix = []
    p = run(
        ["pixi", "run", "--frozen", "--manifest-path", str(here / "envs" / env_name / "pixi.toml")] +
        prefix
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


def collect(samples, size, threads, mpi_sizes, envs, kind):
    for mpi_size in mpi_sizes:
        for omp_threads in threads:
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
        "--kind", type=str, default="mumps", choices=["mumps", "fenics"]
    )

    inner_parser.add_argument("--out", type=str)

    default_blas = "mkl" if sys.platform == "linux" else "openblas"

    collect_parser.add_argument(
        "--envs",
        type=str,
        nargs="+",
    )
    collect_parser.add_argument("--np", type=int, nargs="+", default=[1, 2, 4])
    collect_parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])

    envs_parser.add_argument("blas", nargs="?", default=default_blas)

    args = parser.parse_args()
    if args.action == "inner":
        inner_main(size=args.size, samples=args.samples, fname=args.out, kind=args.kind)
    elif args.action == "collect":
        if args.kind == "mumps":
            args.np = [1]
        if not args.envs:
            args.envs = [
                f"{args.kind}-{build}-{default_blas}"
                for build in ("before", "omp", "gemmt")
            ]
        collect(
            size=args.size,
            samples=args.samples,
            threads=args.threads,
            mpi_sizes=args.np,
            envs=args.envs,
            kind=args.kind,
        )
    elif args.action == "envs":
        create_envs(blas=args.blas, kind=args.kind)
    else:
        raise ValueError("Specify an action: 'inner' or 'collect'")


if __name__ == "__main__":
    # define_envs()
    main()
