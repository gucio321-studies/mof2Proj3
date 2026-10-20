from . import ex

def main() -> None:
    ex.ex3()
    ex.ex4()
    ex.ex5()
    print("Experiment: re-run for time=1e5*dt")
    ex.ex3(out="exX_mesh", target_iterations=1e5)
    ex.ex4(out="exX_evolution", target_iterations=1e5)
