import matplotlib.pyplot as plt

def plot_x_error_from_nek_log(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    x_err, y_err, p_err = [], [], []

    for _, l in enumerate(lines):
        if 'X err' in l:
            x_err.append(float(l.split()[2]))
        if 'Y err' in l:
            y_err.append(float(l.split()[2]))
        if 'P err' in l:
            p_err.append(float(l.split()[2]))

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(x_err)), x_err, label='X Error')
    plt.plot(range(len(y_err)), y_err, label='Y Error')
    plt.plot(range(len(p_err)), p_err, label='P Error')

    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error (log scale)')
    plt.title('X, Y, P Errors over Iterations')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    log_file_name = input('input log file name')
    plot_x_error_from_nek_log(log_file_name)